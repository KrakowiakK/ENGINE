import Foundation

/// T-0259 (C-0729): graceful stop. On SIGTERM the server stops admitting requests (new ones get 503 + Retry-After) and
/// exits once every admitted request has finished, or at the deadline, so a window's stop_engine no longer cuts a client
/// request in flight. This is the state machine only; Serve.swift wires a signal source and the request entry to it.
public final class DrainState: @unchecked Sendable {
    public let graceSeconds: Double
    private let lock = NSLock()
    private var startedAt: Double?
    private var inflight = 0                 // admitted requests not yet finished (counted from the very entry)

    public init(graceSeconds: Double = 120) { self.graceSeconds = max(0, graceSeconds) }

    /// the first SIGTERM starts the drain; a second one changes nothing (the deadline stays)
    public func begin(now: Double) { lock.lock(); defer { lock.unlock() }; if startedAt == nil { startedAt = now } }

    public var draining: Bool { lock.lock(); defer { lock.unlock() }; return startedAt != nil }

    /// a request arriving now: admit it, or refuse it with a Retry-After (seconds) because the server is draining
    public func admit() -> (admitted: Bool, retryAfter: Int) {
        lock.lock(); defer { lock.unlock() }
        return startedAt == nil ? (true, 0) : (false, 30)
    }

    /// whether the process should exit now: draining, and no request in flight or the grace period is over
    public func shouldExit(active: Int, now: Double) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let t0 = startedAt else { return false }
        return active <= 0 || now - t0 >= graceSeconds
    }

    /// T-0259 wiring: a request at the server's entry. true = admitted (count it until leave()); false = refuse (draining)
    public func enter() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard startedAt == nil else { return false }
        inflight += 1; return true
    }
    public func leave() { lock.lock(); defer { lock.unlock() }; inflight = max(0, inflight - 1) }
    public var inFlight: Int { lock.lock(); defer { lock.unlock() }; return inflight }
    /// shouldExit on the requests admitted through enter()
    public func shouldExit(now: Double) -> Bool { shouldExit(active: inFlight, now: now) }
}
