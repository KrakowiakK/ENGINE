// P087/P089 -- admission control. The gate that used to live beside it is gone: serialisation is now
// an ownership property of `ModelThread`, tested in ModelThreadTests.
import Foundation
import XCTest

@testable import EngineServeSupport

final class ActiveCountTests: XCTestCase {
    func testAdmitsUpToTheCapAndRefusesBeyondIt() {
        let a = ActiveCount()
        for _ in 0 ..< 4 { XCTAssertTrue(a.enter(max: 4)) }
        XCTAssertFalse(a.enter(max: 4), "the cap is a memory bound, not a hint")
        XCTAssertEqual(a.current, 4)
    }

    func testAFreedSlotIsReused() {
        let a = ActiveCount()
        for _ in 0 ..< 4 { _ = a.enter(max: 4) }
        a.leave()
        XCTAssertTrue(a.enter(max: 4), "a finished request must give its slot back")
        XCTAssertEqual(a.current, 4)
    }

    func testTheCapHoldsUnderContention() {
        let a = ActiveCount()
        let cap = 8
        let over = NSLock()
        var maxSeen = 0
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 24
        for _ in 0 ..< 24 {
            Thread {
                for _ in 0 ..< 200 {
                    if a.enter(max: cap) {
                        over.lock(); maxSeen = max(maxSeen, a.current); over.unlock()
                        a.leave()
                    }
                }
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 60)
        XCTAssertLessThanOrEqual(maxSeen, cap, "the cap was exceeded under contention")
    }
}
