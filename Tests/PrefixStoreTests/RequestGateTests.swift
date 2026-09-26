import Foundation
import XCTest
@testable import EngineServeSupport

/// A lock-protected box for values the test threads share (Swift 6 forbids mutating captured vars concurrently).
private final class Shared<T>: @unchecked Sendable {
    private let lock = NSLock(); private var v: T
    init(_ v: T) { self.v = v }
    func update<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&v) }
    var value: T { lock.lock(); defer { lock.unlock() }; return v }
}

final class RequestGateTests: XCTestCase {
    /// Negative control: without a queue the gate is the old one -- the (max+1)-th request is refused at once.
    func testNoQueueRefusesImmediatelyLikeTheOldGate() {
        let g = RequestGate()
        for _ in 0..<2 { XCTAssertTrue(g.enter(max: 2)) }
        XCTAssertFalse(g.enter(max: 2))
        let t0 = Date()
        XCTAssertEqual(g.acquire(max: 2, maxWaiting: 0, timeout: 5).outcome, .queueFull)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 0.1)
        g.leave(); XCTAssertTrue(g.enter(max: 2))
    }

    func testAWaiterIsAdmittedWhenASlotFrees() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 4, timeout: 5).outcome, .admitted)
        let done = expectation(description: "admitted after wait")
        DispatchQueue.global().async {
            let r = g.acquire(max: 1, maxWaiting: 4, timeout: 5, pollInterval: 0.05)
            XCTAssertEqual(r.outcome, .admitted); XCTAssertGreaterThan(r.waited, 0.15)
            done.fulfill()
        }
        Thread.sleep(forTimeInterval: 0.2)
        XCTAssertEqual(g.waiting, 1)
        g.leave()
        wait(for: [done], timeout: 3)
        XCTAssertEqual(g.current, 1); XCTAssertEqual(g.waiting, 0)
        XCTAssertEqual(g.snapshot()["admitted_after_wait"] as? Int, 1)
    }

    func testWaitersAreAdmittedInArrivalOrder() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 8, timeout: 5).outcome, .admitted)
        let order = Shared<[Int]>([])
        let group = DispatchGroup()
        for i in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                XCTAssertEqual(g.acquire(max: 1, maxWaiting: 8, timeout: 10, pollInterval: 0.02).outcome, .admitted)
                order.update { $0.append(i) }
                Thread.sleep(forTimeInterval: 0.05)
                g.leave()
                group.leave()
            }
            Thread.sleep(forTimeInterval: 0.1)             // arrival order 0, 1, 2, 3
        }
        XCTAssertEqual(g.waiting, 4)
        g.leave()
        XCTAssertEqual(group.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(order.value, [0, 1, 2, 3])
    }

    func testAFullQueueRefuses() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 1, timeout: 5).outcome, .admitted)
        DispatchQueue.global().async { _ = g.acquire(max: 1, maxWaiting: 1, timeout: 1, pollInterval: 0.05) }
        Thread.sleep(forTimeInterval: 0.2)
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 1, timeout: 5).outcome, .queueFull)
    }

    func testATimedOutWaiterLeavesAndTheNextOneStillGetsIn() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 4, timeout: 5).outcome, .admitted)
        let r = g.acquire(max: 1, maxWaiting: 4, timeout: 0.2, pollInterval: 0.05)
        XCTAssertEqual(r.outcome, .timedOut)
        XCTAssertEqual(g.waiting, 0)
        g.leave()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 4, timeout: 1).outcome, .admitted)
        XCTAssertEqual(g.snapshot()["timed_out"] as? Int, 1)
    }

    func testAGoneClientLeavesTheQueue() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 4, timeout: 5).outcome, .admitted)
        var polls = 0
        let r = g.acquire(max: 1, maxWaiting: 4, timeout: 5, pollInterval: 0.02) { polls += 1; return polls >= 3 }
        XCTAssertEqual(r.outcome, .clientGone)
        XCTAssertEqual(g.waiting, 0); XCTAssertEqual(g.current, 1)
    }

    /// A newcomer must not jump a waiter when a slot frees: while the waiter is still queued (held inside its `gone`
    /// poll, deterministically), a freed slot is refused to the newcomer and then taken by the waiter.
    func testANewcomerDoesNotOvertakeAWaiter() {
        let g = RequestGate()
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 4, timeout: 5).outcome, .admitted)
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let calls = Shared(0)
        let admitted = expectation(description: "waiter admitted")
        DispatchQueue.global().async {
            let r = g.acquire(max: 1, maxWaiting: 4, timeout: 5, pollInterval: 0.05) {
                let first = calls.update { c -> Bool in c += 1; return c == 1 }
                if first { entered.signal(); release.wait() }
                return false
            }
            XCTAssertEqual(r.outcome, .admitted)
            admitted.fulfill()
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 3), .success)
        g.leave()                                            // a slot is free, the waiter is still in the queue
        XCTAssertEqual(g.acquire(max: 1, maxWaiting: 0, timeout: 1).outcome, .queueFull)
        release.signal()
        wait(for: [admitted], timeout: 3)
        XCTAssertEqual(g.current, 1)
    }
}
