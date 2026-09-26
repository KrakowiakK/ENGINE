import Foundation

/// P116: the request gate. At most `max` chat requests run at once. With `maxWaiting > 0` a request that finds every
/// slot taken WAITS in a first-in-first-out queue and is admitted the moment a slot frees, instead of being refused
/// with a 503 at once. A client that got the 503 retried with its own exponential backoff (aider/litellm: up to 4096 s
/// between attempts, 7.9 h of summed sleep in one benchmark run) while slots sat free. The queue stays bounded -- when
/// `maxWaiting` requests already wait, the next one is still refused -- and a wait ends after `timeout` seconds or when
/// the client is gone. `maxWaiting == 0` is the old behaviour exactly: refuse as soon as `max` requests run.
public final class RequestGate: @unchecked Sendable {
    public enum Outcome: String, Sendable { case admitted, queueFull = "queue_full", timedOut = "timed_out", clientGone = "client_gone" }

    private let cond = NSCondition()
    private var active = 0
    private var queue: [Int] = []          // tickets in arrival order
    private var nextTicket = 0
    private var admittedNow = 0, admittedAfterWait = 0, queueFull = 0, timedOut = 0, clientGone = 0
    private var totalWait = 0.0, maxWait = 0.0

    public init() {}

    /// The pre-P116 gate: admit if a slot is free, else refuse. Never waits.
    public func enter(max: Int) -> Bool {
        cond.lock(); defer { cond.unlock() }
        guard active < max else { queueFull += 1; return false }
        active += 1; admittedNow += 1
        return true
    }

    /// Admit now, or wait in FIFO order. `gone` is polled between waits (it must not block).
    public func acquire(max: Int, maxWaiting: Int, timeout: Double, pollInterval: Double = 0.5,
                        gone: () -> Bool = { false }) -> (outcome: Outcome, waited: Double) {
        let start = Date()
        cond.lock()
        if active < max && queue.isEmpty {
            active += 1; admittedNow += 1
            cond.unlock()
            return (.admitted, 0)
        }
        guard queue.count < maxWaiting else { queueFull += 1; cond.unlock(); return (.queueFull, 0) }
        let ticket = nextTicket
        nextTicket += 1
        queue.append(ticket)
        while true {
            if queue.first == ticket && active < max {
                queue.removeFirst()
                active += 1; admittedAfterWait += 1
                let w = Date().timeIntervalSince(start)
                totalWait += w; maxWait = Swift.max(maxWait, w)
                cond.broadcast()               // the next waiter may also fit
                cond.unlock()
                return (.admitted, w)
            }
            let waited = Date().timeIntervalSince(start)
            if waited >= timeout {
                return leaveQueue(ticket, .timedOut, waited)
            }
            cond.wait(until: Date().addingTimeInterval(Swift.min(pollInterval, timeout - waited)))
            cond.unlock()
            let isGone = gone()
            cond.lock()
            if isGone { return leaveQueue(ticket, .clientGone, Date().timeIntervalSince(start)) }
        }
    }

    /// Called with the lock held; releases it.
    private func leaveQueue(_ ticket: Int, _ outcome: Outcome, _ waited: Double) -> (outcome: Outcome, waited: Double) {
        queue.removeAll { $0 == ticket }
        if outcome == .timedOut { timedOut += 1 } else { clientGone += 1 }
        cond.broadcast()                       // the head may have changed
        cond.unlock()
        return (outcome, waited)
    }

    public func leave() {
        cond.lock(); active -= 1; cond.broadcast(); cond.unlock()
    }

    public var current: Int { cond.lock(); defer { cond.unlock() }; return active }
    public var waiting: Int { cond.lock(); defer { cond.unlock() }; return queue.count }

    public func snapshot() -> [String: Any] {
        cond.lock(); defer { cond.unlock() }
        return ["running": active, "waiting": queue.count, "admitted_immediately": admittedNow,
                "admitted_after_wait": admittedAfterWait, "refused_queue_full": queueFull, "timed_out": timedOut,
                "client_gone": clientGone, "total_wait_seconds": totalWait, "max_wait_seconds": maxWait]
    }
}
