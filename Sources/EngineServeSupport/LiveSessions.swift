import Foundation

/// Every request the server is handling, and the last few it finished, for `GET /v1/engine/sessions`.
/// Host-side bookkeeping only: connection threads write it at phase changes and once per emitted token,
/// the stats reader takes the same lock, and nothing here touches the model thread or an MLXArray. It
/// holds counts and timings, never prompt or completion text (the studio can be reached through a tunnel).
public final class LiveSessions: @unchecked Sendable {
    public struct Entry: Sendable {
        public let id: Int
        public let client: String
        public let startedAt: Date
        public let promptTokens: Int
        public let maxTokens: Int
        public let stream: Bool
        public let reasoningEffort: String
        public var phase = "prefill"            // prefill -> decode -> done
        public var cachedTokens = 0
        public var cacheSource = "none"
        public var prefilledTo = 0              // prompt position the caches have reached (includes the cache hit)
        public var mtp = false
        public var firstTokenAt: Date? = nil
        public var generated = 0
        public var mtpDrafted = 0
        public var mtpAccepted = 0
        public var finishReason = ""
        public var endedAt: Date? = nil
    }

    private let lock = NSLock()
    private var nextId = 1
    private var active: [Int: Entry] = [:]
    private var recent: [Entry] = []
    private let recentCap: Int
    public let startedAt = Date()
    private var totalRequests = 0
    private var totalPromptTokens = 0
    private var totalCachedTokens = 0
    private var totalGenerated = 0
    private var totalAborted = 0

    public init(recentCap: Int = 40) { self.recentCap = recentCap }

    public func begin(client: String, promptTokens: Int, maxTokens: Int, stream: Bool, reasoningEffort: String) -> Int {
        lock.lock(); defer { lock.unlock() }
        let id = nextId; nextId += 1
        active[id] = Entry(id: id, client: client, startedAt: Date(), promptTokens: promptTokens,
                           maxTokens: maxTokens, stream: stream, reasoningEffort: reasoningEffort)
        totalRequests += 1
        totalPromptTokens += promptTokens
        return id
    }

    public func update(_ id: Int, _ body: (inout Entry) -> Void) {
        lock.lock(); defer { lock.unlock() }
        guard var e = active[id] else { return }
        body(&e)
        active[id] = e
    }

    public func tokenEmitted(_ id: Int) {
        lock.lock(); defer { lock.unlock() }
        active[id]?.generated += 1
        totalGenerated += 1
    }

    /// An empty `finishReason` means the request ended without reaching its normal exit (an error or a
    /// dropped connection) and is counted as aborted.
    public func end(_ id: Int) {
        lock.lock(); defer { lock.unlock() }
        guard var e = active.removeValue(forKey: id) else { return }
        e.phase = "done"
        e.endedAt = Date()
        if e.finishReason.isEmpty { e.finishReason = "aborted"; totalAborted += 1 }
        totalCachedTokens += e.cachedTokens
        recent.insert(e, at: 0)
        if recent.count > recentCap { recent.removeLast(recent.count - recentCap) }
    }

    public var activeCount: Int { lock.lock(); defer { lock.unlock() }; return active.count }

    public func snapshot(extra: [String: Any] = [:]) -> [String: Any] {
        lock.lock()
        let act = active.values.sorted { $0.id < $1.id }
        let rec = recent
        let totals: [String: Any] = [
            "requests": totalRequests, "prompt_tokens": totalPromptTokens, "cached_tokens": totalCachedTokens,
            "generated_tokens": totalGenerated, "aborted": totalAborted,
        ]
        lock.unlock()
        let now = Date()
        let rows = act.map { Self.row($0, now: now) }
        let decoding = act.filter { $0.phase == "decode" }
        let aggregate: [String: Any] = [
            "active": act.count,
            "prefilling": act.filter { $0.phase == "prefill" }.count,
            "decoding": decoding.count,
            // the sum of each decoding session's own rate since its first token: what the fleet is getting now
            "decode_tokens_per_second": rows.filter { ($0["phase"] as? String) == "decode" }
                .compactMap { $0["decode_tokens_per_second"] as? Double }.reduce(0, +),
            "prompt_tokens_in_flight": act.reduce(0) { $0 + $1.promptTokens },
        ]
        var out: [String: Any] = [
            "object": "engine.sessions",
            "now": now.timeIntervalSince1970,
            "uptime_s": now.timeIntervalSince(startedAt),
            "aggregate": aggregate,
            "active": rows,
            "recent": rec.map { Self.row($0, now: now) },
            "totals": totals,
        ]
        for (k, v) in extra { out[k] = v }
        return out
    }

    static func row(_ e: Entry, now: Date) -> [String: Any] {
        let end = e.endedAt ?? now
        var r: [String: Any] = [
            "id": e.id, "client": e.client, "phase": e.phase, "started_at": e.startedAt.timeIntervalSince1970,
            "elapsed_s": end.timeIntervalSince(e.startedAt), "prompt_tokens": e.promptTokens,
            "cached_tokens": e.cachedTokens, "cache_source": e.cacheSource, "prefilled_to": e.prefilledTo,
            "max_tokens": e.maxTokens, "generated": e.generated, "stream": e.stream, "mtp": e.mtp,
            "reasoning_effort": e.reasoningEffort, "finish_reason": e.finishReason,
        ]
        if let f = e.firstTokenAt {
            r["ttft_ms"] = f.timeIntervalSince(e.startedAt) * 1000
            let prefilled = max(0, e.promptTokens - e.cachedTokens)
            let pf = f.timeIntervalSince(e.startedAt)
            if pf > 0 { r["prefill_tokens_per_second"] = Double(prefilled) / pf }
            let dt = end.timeIntervalSince(f)
            if dt > 0.05, e.generated > 0 { r["decode_tokens_per_second"] = Double(e.generated) / dt }
        } else if e.phase == "prefill" {
            let t = now.timeIntervalSince(e.startedAt)
            let done = max(0, e.prefilledTo - e.cachedTokens)
            if t > 0 { r["prefill_tokens_per_second"] = Double(done) / t }
        }
        if e.mtpDrafted > 0 {
            r["mtp_drafted"] = e.mtpDrafted; r["mtp_accepted"] = e.mtpAccepted
            r["mtp_acceptance"] = Double(e.mtpAccepted) / Double(e.mtpDrafted)
        }
        return r
    }
}

public let liveSessions = LiveSessions()
