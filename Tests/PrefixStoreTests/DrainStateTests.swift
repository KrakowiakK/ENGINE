import XCTest
@testable import EngineServeSupport

/// T-0259 (C-0729): the graceful-stop state machine
final class DrainStateTests: XCTestCase {
    func testAdmitsUntilTheDrainBeginsThenRefusesWithRetryAfter() {
        let d = DrainState(graceSeconds: 120)
        XCTAssertEqual(d.admit().admitted, true)
        XCTAssertFalse(d.shouldExit(active: 0, now: 0))          // not draining: never exit, even when idle
        d.begin(now: 100)
        XCTAssertTrue(d.draining)
        XCTAssertEqual(d.admit().admitted, false)
        XCTAssertEqual(d.admit().retryAfter, 30)
    }

    func testExitsWhenTheLastRequestFinishesOrAtTheDeadline() {
        let d = DrainState(graceSeconds: 120)
        d.begin(now: 100)
        XCTAssertFalse(d.shouldExit(active: 3, now: 150))         // requests still in flight, inside the grace period
        XCTAssertTrue(d.shouldExit(active: 0, now: 150))          // all done
        XCTAssertTrue(d.shouldExit(active: 3, now: 220))          // grace over: exit anyway
        XCTAssertFalse(d.shouldExit(active: 3, now: 219.9))
    }

    func testASecondSignalDoesNotExtendTheDeadline() {
        let d = DrainState(graceSeconds: 10)
        d.begin(now: 0); d.begin(now: 9)
        XCTAssertTrue(d.shouldExit(active: 5, now: 10))
    }

    /// requests are counted from the entry: the drain waits for one admitted just before the signal
    func testEnterLeaveCountsAdmittedRequestsAndRefusesWhileDraining() {
        let d = DrainState(graceSeconds: 60)
        XCTAssertTrue(d.enter()); XCTAssertTrue(d.enter())
        d.begin(now: 0)
        XCTAssertFalse(d.enter())                                  // refused: not counted
        XCTAssertEqual(d.inFlight, 2)
        XCTAssertFalse(d.shouldExit(now: 1))
        d.leave(); XCTAssertFalse(d.shouldExit(now: 2))
        d.leave(); XCTAssertTrue(d.shouldExit(now: 3))
        d.leave(); XCTAssertEqual(d.inFlight, 0)                   // never below zero
    }
}
