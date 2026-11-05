import Foundation

/// T-0217: a content-free census of how much of each finished text request a copy-mode lookup (an n-gram match in the
/// prompt + output so far, its continuation verified as drafts) would have drafted. It decides whether batched rounds
/// should get copy drafts (C-0684: +13..23% on output that re-types its prompt, ~0% on extract-style output). Only
/// counts leave this file: the token ids are read once, after the response, off the request and model threads.
public struct CopyCensusConfig: Sendable {
    public let minMatch: Int       // a lookup round starts on a match of at least this many tokens
    public let startLen: Int       // drafts in the first lookup round of a run
    public let maxLen: Int         // the block doubles after every fully accepted lookup round, up to this
    public init(minMatch: Int, startLen: Int, maxLen: Int) { self.minMatch = minMatch; self.startLen = startLen; self.maxLen = maxLen }
    public var name: String { "min\(minMatch)_l\(startLen)_max\(maxLen)" }
}

public struct CopyCensusCounts: Sendable, Equatable {
    public var lookupRounds = 0    // rounds whose drafts came from a lookup
    public var lookupTokens = 0    // tokens those rounds committed (accepted + 1)
    public var accepted = 0        // lookup drafts equal to the output
    public var drafted = 0         // lookup drafts proposed
    public init(lookupRounds: Int = 0, lookupTokens: Int = 0, accepted: Int = 0, drafted: Int = 0) {
        self.lookupRounds = lookupRounds; self.lookupTokens = lookupTokens; self.accepted = accepted; self.drafted = drafted
    }
}

/// The latest EARLIER occurrence of every n-gram (n = 2...6) of the context, as the serial path's CopyIndex keeps it:
/// the current suffix is never indexed, and a 6-token match is extended backwards (up to 64) to its full length.
struct CopyCensusIndex {
    private(set) var ctx: [Int] = []
    private var maps: [[UInt64: Int]] = Array(repeating: [:], count: 7)
    init(_ tokens: [Int]) { ctx.reserveCapacity(tokens.count + 1024); append(tokens) }
    private func key(_ start: Int, _ n: Int) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        for i in start ..< start + n { h = (h ^ UInt64(bitPattern: Int64(ctx[i]))) &* 1099511628211 }
        return h
    }
    mutating func append(_ toks: some Sequence<Int>) {
        for t in toks {
            let e = ctx.count - 1
            if e >= 0 { for n in 2 ... 6 where e - n + 1 >= 0 { maps[n][key(e - n + 1, n)] = e - n + 1 } }
            ctx.append(t)
        }
    }
    func lookup(_ L: Int) -> (match: Int, cont: ArraySlice<Int>)? {
        let c = ctx.count
        for n in stride(from: 6, through: 2, by: -1) where c > n {
            guard let st = maps[n][key(c - n, n)], ctx[st ..< (st + n)] == ctx[(c - n) ..< c] else { continue }   // hash collision
            let from = st + n, to = min(from + L, c)
            guard to > from else { continue }
            var m = n
            if n == 6 { while m < 64, st - (m - n) - 1 >= 0, ctx[st - (m - n) - 1] == ctx[c - m - 1] { m += 1 } }
            return (m, ctx[from ..< to])
        }
        return nil
    }
}

/// Walk one finished request: output[0] came from the prefill; at every later pending position j a lookup round is
/// tried (a match >= minMatch, or >= 2 right after a fully accepted lookup round), its drafts are compared with the
/// output, and it commits accepted + 1 tokens; otherwise one token is committed by the ordinary drafter. Lookups are
/// tried at every position, so this is an upper bound of what a round-boundary rule captures (C-0684 checks only at
/// the MTP rounds' boundaries).
public func copyCensusWalk(prompt: [Int], output: [Int], minMatch: Int, startLen: Int, maxLen: Int,
                           window: Int = 65536) -> CopyCensusCounts {
    var r = CopyCensusCounts()
    guard !output.isEmpty else { return r }
    var ix = CopyCensusIndex(Array((prompt + output.prefix(1)).suffix(window)))
    let end = output.count - 1
    var j = 0, full = false, L = 0
    while j < end {
        let Lnext = full ? min(maxLen, max(1, L) * 2) : startLen
        if let hit = ix.lookup(Lnext), hit.match >= (full ? 2 : minMatch) {
            let cont = Array(hit.cont)
            L = Lnext
            var acc = 0
            for t in cont { guard j + 1 + acc <= end, output[j + 1 + acc] == t else { break }; acc += 1 }
            let commit = min(acc + 1, end - j)
            r.lookupRounds += 1; r.drafted += cont.count; r.accepted += acc; r.lookupTokens += commit
            ix.append(output[(j + 1) ... (j + commit)]); j += commit
            full = acc == cont.count
        } else {
            full = false; L = 0
            ix.append(output[(j + 1) ... (j + 1)]); j += 1
        }
    }
    return r
}

/// T-0244: one request's MTP rounds, drafts proposed and drafts accepted (its specStats), counted per label
public struct CopyCensusMTP: Sendable, Equatable {
    public var rounds = 0, drafted = 0, accepted = 0
    /// T-0271: the request's batched rows -- MTP row-rounds and copy (lookup) row-rounds with their drafts and acceptances
    public var batchRounds = 0, batchDrafted = 0, batchAccepted = 0
    public var copyRows = 0, copyDrafted = 0, copyAccepted = 0
    public init(rounds: Int = 0, drafted: Int = 0, accepted: Int = 0, batchRounds: Int = 0, batchDrafted: Int = 0,
                batchAccepted: Int = 0, copyRows: Int = 0, copyDrafted: Int = 0, copyAccepted: Int = 0) {
        self.rounds = rounds; self.drafted = drafted; self.accepted = accepted
        self.batchRounds = batchRounds; self.batchDrafted = batchDrafted; self.batchAccepted = batchAccepted
        self.copyRows = copyRows; self.copyDrafted = copyDrafted; self.copyAccepted = copyAccepted
    }
}

/// One bucket of census totals: requests, tokens and the per-config lookup counts.
struct CopyCensusBucket {
    var requests = 0, outputTokens = 0, promptTokens = 0
    var mtpRounds = 0, mtpDrafted = 0, mtpAccepted = 0   // T-0244: the requests' own SERIAL MTP rounds and drafts (T-0271: not batched)
    var batchRounds = 0, batchDrafted = 0, batchAccepted = 0   // T-0271: batched MTP row-rounds
    var copyRows = 0, copyDrafted = 0, copyAccepted = 0        // T-0271: batched copy row-rounds
    var lastSeen: Double = 0                    // unix seconds of the last request counted here
    var counts: [CopyCensusCounts]
    init(configCount: Int) { counts = Array(repeating: CopyCensusCounts(), count: configCount) }

    mutating func add(promptCount: Int, outputCount: Int, _ rs: [CopyCensusCounts], mtp: CopyCensusMTP, at t: Double) {
        requests += 1; outputTokens += outputCount; promptTokens += promptCount; lastSeen = max(lastSeen, t)
        mtpRounds += mtp.rounds; mtpDrafted += mtp.drafted; mtpAccepted += mtp.accepted
        batchRounds += mtp.batchRounds; batchDrafted += mtp.batchDrafted; batchAccepted += mtp.batchAccepted
        copyRows += mtp.copyRows; copyDrafted += mtp.copyDrafted; copyAccepted += mtp.copyAccepted
        for i in rs.indices {
            counts[i].lookupRounds += rs[i].lookupRounds; counts[i].lookupTokens += rs[i].lookupTokens
            counts[i].accepted += rs[i].accepted; counts[i].drafted += rs[i].drafted
        }
    }

    func json(_ configs: [CopyCensusConfig], share: Bool) -> [String: Any] {
        var d: [String: Any] = ["requests": requests, "output_tokens": outputTokens, "prompt_tokens": promptTokens,
                                "mtp_rounds": mtpRounds, "mtp_drafted": mtpDrafted, "mtp_accepted": mtpAccepted,
                                "batch_rounds": batchRounds, "batch_drafted": batchDrafted, "batch_accepted": batchAccepted,
                                "copy_rows": copyRows, "copy_drafted": copyDrafted, "copy_accepted": copyAccepted]
        if lastSeen > 0 { d["last_seen"] = lastSeen }
        d["configs"] = configs.indices.map { i -> [String: Any] in
            var c: [String: Any] = ["name": configs[i].name, "lookup_rounds": counts[i].lookupRounds,
                                    "lookup_tokens": counts[i].lookupTokens, "accepted": counts[i].accepted,
                                    "drafted": counts[i].drafted]
            if share { c["lookup_token_share"] = outputTokens > 0 ? Double(counts[i].lookupTokens) / Double(outputTokens) : 0 }
            return c
        }
        return d
    }

    /// restores a persisted bucket; configs are matched by name, so a file written under other configs keeps its
    /// request and token totals and loses only the counts of configs no longer measured
    init(json d: [String: Any], configs: [CopyCensusConfig]) {
        self.init(configCount: configs.count)
        requests = d["requests"] as? Int ?? 0; outputTokens = d["output_tokens"] as? Int ?? 0
        promptTokens = d["prompt_tokens"] as? Int ?? 0; lastSeen = d["last_seen"] as? Double ?? 0
        mtpRounds = d["mtp_rounds"] as? Int ?? 0; mtpDrafted = d["mtp_drafted"] as? Int ?? 0     // files before T-0244: 0
        mtpAccepted = d["mtp_accepted"] as? Int ?? 0
        batchRounds = d["batch_rounds"] as? Int ?? 0; batchDrafted = d["batch_drafted"] as? Int ?? 0   // files before T-0271: 0
        batchAccepted = d["batch_accepted"] as? Int ?? 0
        copyRows = d["copy_rows"] as? Int ?? 0; copyDrafted = d["copy_drafted"] as? Int ?? 0; copyAccepted = d["copy_accepted"] as? Int ?? 0
        for c in d["configs"] as? [[String: Any]] ?? [] {
            guard let name = c["name"] as? String, let i = configs.firstIndex(where: { $0.name == name }) else { continue }
            counts[i] = CopyCensusCounts(lookupRounds: c["lookup_rounds"] as? Int ?? 0, lookupTokens: c["lookup_tokens"] as? Int ?? 0,
                                         accepted: c["accepted"] as? Int ?? 0, drafted: c["drafted"] as? Int ?? 0)
        }
    }
}

/// Server totals per configuration, overall and per client label, for /v1/engine/sessions ("copy_census"). Thread-safe;
/// record() is meant to run on a background queue after the response has been sent.
///
/// T-0221: totals are also kept per client label (x-engine-client ?? user-agent ?? "unknown"), so gate and lab load can be
/// netted out of the client's. With a `file`, the totals are restored from it at start and rewritten (atomically) after
/// every record, so a restart keeps them. The file holds counts and client labels only -- never token ids or text.
public final class CopyCensusTotals: @unchecked Sendable {
    public static let maxLabelLength = 64
    public static let maxLabels = 64            // labels past this many are counted as "other"
    public static let otherLabel = "other"

    public let configs: [CopyCensusConfig]
    public let file: URL?
    private let lock = NSLock()
    private var total: CopyCensusBucket
    private var byClient: [String: CopyCensusBucket] = [:]
    private var since: Double                   // unix seconds the totals began (kept across restarts by the file)
    private var processStarts = 1               // starts of the server that have added to these totals
    private var restored = false
    private var fileErrors = 0
    private var persistDisabled = false     // T-0260: an unreadable file that could not be moved aside is left alone
    private var lastFileError: String?
    private let now: () -> Double

    public init(configs: [CopyCensusConfig], file: URL? = nil, now: @escaping () -> Double = { Date().timeIntervalSince1970 }) {
        self.configs = configs; self.file = file; self.now = now
        total = CopyCensusBucket(configCount: configs.count)
        since = now()
        guard let file else { return }
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        do {
            let data = try Data(contentsOf: file)
            guard let d = try JSONSerialization.jsonObject(with: data) as? [String: Any], d["version"] as? Int == 1 else {
                throw NSError(domain: "CopyCensus", code: 1, userInfo: [NSLocalizedDescriptionKey: "not a version-1 census file"])
            }
            total = CopyCensusBucket(json: d["total"] as? [String: Any] ?? [:], configs: configs)
            for (label, b) in d["by_client"] as? [String: Any] ?? [:] {
                if let b = b as? [String: Any] { byClient[label] = CopyCensusBucket(json: b, configs: configs) }
            }
            since = d["since"] as? Double ?? since
            processStarts = (d["process_starts"] as? Int ?? 0) + 1
            restored = true
        } catch {
            // an unreadable file is kept aside, never overwritten: the counts start afresh in memory and in `file`
            fileErrors += 1; lastFileError = "restore: \(error.localizedDescription)"
            let aside = file.appendingPathExtension("unreadable-\(Int(since))")
            do { try FileManager.default.moveItem(at: file, to: aside) } catch {
                // T-0260 review: it could not be moved aside, so it must not be overwritten either -- stop persisting
                persistDisabled = true; lastFileError = "restore: \(lastFileError ?? ""); move aside failed: \(error.localizedDescription)"
            }
        }
    }

    /// the label a request is counted under: trimmed, a trailing "-<number>" dropped (lab tools label each request of a
    /// burst burst-<tag>-<i>, rq-<tag>-<i>, smoke-big-<i>: one burst would otherwise fill every label slot), at most
    /// maxLabelLength characters, "unknown" when empty
    public static func label(_ raw: String?) -> String {
        var t = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        // T-0260 review: lab tools label each run burst-<tag>-<i> / rq-<tag>-<i>; with persisted totals the per-run tags
        // would use up the label slots for good, so they count as one label each
        for lab in ["burst-", "rq-"] where t.hasPrefix(lab) { return String(lab.dropLast()) }
        if let dash = t.lastIndex(of: "-"), dash > t.startIndex {
            let tail = t[t.index(after: dash)...]
            if !tail.isEmpty, tail.allSatisfy(\.isASCII), tail.allSatisfy(\.isNumber) { t = String(t[..<dash]) }
        }
        return t.isEmpty ? "unknown" : String(t.prefix(maxLabelLength))
    }

    public func record(prompt: [Int], output: [Int], client: String? = nil, mtp: CopyCensusMTP = CopyCensusMTP()) {
        let rs = configs.map { copyCensusWalk(prompt: prompt, output: output, minMatch: $0.minMatch, startLen: $0.startLen, maxLen: $0.maxLen) }
        let t = now()
        var label = Self.label(client)
        lock.lock(); defer { lock.unlock() }
        if byClient[label] == nil, byClient.count >= Self.maxLabels { label = Self.otherLabel }
        total.add(promptCount: prompt.count, outputCount: output.count, rs, mtp: mtp, at: t)
        byClient[label, default: CopyCensusBucket(configCount: configs.count)].add(promptCount: prompt.count, outputCount: output.count, rs, mtp: mtp, at: t)
        persistLocked()
    }

    private func persistLocked() {
        guard let file, !persistDisabled else { return }
        let d: [String: Any] = ["version": 1, "since": since, "written": now(), "process_starts": processStarts,
                                "total": total.json(configs, share: false),
                                "by_client": byClient.mapValues { $0.json(configs, share: false) }]
        do {
            let data = try JSONSerialization.data(withJSONObject: d, options: [.sortedKeys])
            try data.write(to: file, options: .atomic)
        } catch {
            fileErrors += 1; lastFileError = "write: \(error.localizedDescription)"
        }
    }

    public func snapshot() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        var s = total.json(configs, share: true)
        s.removeValue(forKey: "last_seen")
        s["by_client"] = byClient.mapValues { $0.json(configs, share: true) }
        s["since"] = since
        s["process_starts"] = processStarts
        if let file {
            s["file"] = file.path; s["restored_from_file"] = restored; s["file_errors"] = fileErrors
            if let e = lastFileError { s["last_file_error"] = e }
        }
        return s
    }
}
