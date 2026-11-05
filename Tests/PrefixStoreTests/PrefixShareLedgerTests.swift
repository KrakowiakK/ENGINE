import Foundation
import XCTest
@testable import EngineServeSupport

/// T-0095: the hash-only ledger of how much of each prompt an earlier prompt began with, and the hot-store floor.
final class PrefixShareLedgerTests: XCTestCase {
    func testLongestSharedPrefixInSteps() {
        let l = PrefixShareLedger(step: 64, capacity: 16, maxLength: 4096, rung: 1024, rungMargin: 256)
        let head = Array(0 ..< 1100)
        XCTAssertEqual(l.record(head + Array(5000 ..< 6000)), 0, "first prompt: nothing seen")
        XCTAssertEqual(l.record(head + Array(7000 ..< 8000)), 1088, "1100 shared tokens, last full step 1088")
        XCTAssertEqual(l.record(Array(9000 ..< 9500)), 0)
        let s = l.snapshot()
        XCTAssertEqual(s["prompts"] as? Int, 3)
        XCTAssertEqual(s["shared_prefix_tokens"] as? Int, 1088)
        XCTAssertEqual(s["rung_eligible_prompts"] as? Int, 1)
        XCTAssertEqual(s["rung_saveable_tokens"] as? Int, 1024)
        XCTAssertEqual((s["shared_prefix_histogram"] as? [String: Int])?["1024-2047"], 1)
    }

    func testExactRepeatAndShortPrompts() {
        let l = PrefixShareLedger(step: 64, capacity: 16, maxLength: 4096)
        let p = Array(100 ..< 150)                      // shorter than a step: only the whole-prompt key
        XCTAssertEqual(l.record(p), 0)
        XCTAssertEqual(l.record(p), 50, "an exact repeat shares the whole prompt")
        XCTAssertEqual(l.record(Array(100 ..< 149)), 0, "a strict prefix shorter than a step is not counted")
        let s = l.snapshot()
        XCTAssertEqual(s["exact_repeats"] as? Int, 1)
        XCTAssertEqual(s["shared_prefix_tokens"] as? Int, 49, "capped at prompt - 1: the last token is always prefilled")
    }

    func testWindowForgetsOldPrompts() {
        let l = PrefixShareLedger(step: 8, capacity: 2, maxLength: 64)
        let a = Array(0 ..< 32)
        l.record(a); l.record(Array(100 ..< 132)); l.record(Array(200 ..< 232))
        XCTAssertEqual(l.record(a), 0, "the first prompt left the 2-prompt window")
        XCTAssertEqual(l.record(a), 32, "but the copy just recorded is in it")
    }

    func testPrefixesBeyondMaxLengthAreNotHashed() {
        let l = PrefixShareLedger(step: 8, capacity: 4, maxLength: 64)
        let a = Array(0 ..< 200)
        l.record(a)
        XCTAssertEqual(l.record(a), 64, "a long exact repeat is seen up to maxLength only")
    }

    func testHotFloorKeepsAdmissionBelowCapacityAndTheStoreTargetAboveTheFloor() {
        func budget(floor: Double) -> AdmissionBudget {
            AdmissionBudget(maxContext: 100_000, capacityBytes: 10_000, bytesPerToken: 1, fixedBytesPerSequence: 0,
                            historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 10_000, historyFactor: 1,
                            perRow: true, historyCapacityBytesForRow: { length, _ in Double(length) }, hotFloorBytes: floor)
        }
        var targets: [Double] = []
        let noFloor = budget(floor: 0), floored = budget(floor: 2_000)
        var a = 0, b = 0
        while noFloor.reserve(length: 1_000, current: 500, prepareCache: { _, _ in true }) != nil { a += 1 }
        while floored.reserve(length: 1_000, current: 500, prepareCache: { t, _ in targets.append(t); return true }) != nil { b += 1 }
        XCTAssertEqual(a, 6, "6 x 1_512 = 9_072 fits 10_000")
        XCTAssertEqual(b, 5, "5 x 1_512 = 7_560 fits 8_000, 6 x does not")
        XCTAssertGreaterThanOrEqual(targets.min() ?? 0, 2_000)
        XCTAssertGreaterThanOrEqual(floored.snapshot()["hot_target_bytes"] ?? 0, 2_000)
        XCTAssertEqual(floored.snapshot()["hot_floor_bytes"], 2_000)
        XCTAssertFalse(floored.estimateFits(length: 1_000, prompt: 500))
    }
}
