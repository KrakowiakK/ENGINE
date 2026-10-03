import Foundation
import XCTest
@testable import EngineServeSupport

/// P120: the hot target follows each row's committed length (prompt + growth window, grown ahead of the row), while
/// admission keeps pricing the admitted length (prompt + max_tokens + lookahead).
final class AdmissionGrowthTests: XCTestCase {
    private func budget(window: Int, capacity: Double = 20_000, ceiling: Double = 20_000) -> AdmissionBudget {
        AdmissionBudget(maxContext: 1000, capacityBytes: capacity, bytesPerToken: 1, fixedBytesPerSequence: 0,
                        historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: ceiling, growthWindow: window)
    }
    private func bound(_ count: Int, _ longest: Int) -> Double { Double(count) * 3 * Double(longest + 512) }

    /// The lock-only estimate a waiting request repeats agrees with reserve's capacity test: it pads to the longest
    /// active row, never reserves, and turns true when that row is released (2026-10-03, T-0048).
    func testEstimateFitsMatchesReserveCapacityAndReservesNothing() {
        let b = budget(window: 0, capacity: 6_000)            // one row of 900 bounds at 3 x (900 + 512) = 4_236
        let long = b.reserve(length: 900) { _, _ in true }!
        XCTAssertFalse(b.estimateFits(length: 10), "a short request is priced at the longest active row: 2 x 4_236 > 6_000")
        XCTAssertNil(b.reserve(length: 10) { _, _ in true })
        XCTAssertEqual(b.active, 1, "the estimate reserved nothing")
        b.release(long)
        XCTAssertTrue(b.estimateFits(length: 10))
        XCTAssertNotNil(b.reserve(length: 10) { _, _ in true })
        XCTAssertFalse(b.estimateFits(length: 1001), "past max_context it never fits")
    }

    /// T-0051: the history factor scales the history part of every row's price and nothing else; 3 is the old bound.
    func testHistoryFactorScalesOnlyTheHistoryPrice() {
        func make(_ f: Double) -> AdmissionBudget {
            AdmissionBudget(maxContext: 1000, capacityBytes: 10_000, bytesPerToken: 1, fixedBytesPerSequence: 100,
                            historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 10_000, historyFactor: f)
        }
        func admitted(_ f: Double) -> Int {
            let b = make(f); var n = 0
            while b.reserve(length: 900, prepareCache: { _, _ in true }) != nil { n += 1 }
            return n
        }
        XCTAssertEqual(admitted(3), 2, "3 x (900 + 512) + 100 = 4_336 per row: two fit 10_000, three do not")
        XCTAssertEqual(admitted(2), 3, "2 x (900 + 512) + 100 = 2_924 per row: three fit, four do not")
        XCTAssertEqual(make(2).snapshot()["admission_history_factor"], 2)
        XCTAssertEqual(AdmissionBudget(maxContext: 10, capacityBytes: 1, bytesPerToken: 1).historyFactor, 3, "the default is the old bound")
    }

    func testWindowZeroIsTheB53FormulaAndNeverGrows() {
        let b = budget(window: 0)
        var seen: [(Double, Double)] = []
        let id = b.reserve(length: 900, current: 100) { t, g in seen.append((t, g)); return true }!
        XCTAssertEqual(seen.first?.0, 20_000 - bound(1, 900))
        XCTAssertEqual(seen.first?.1, bound(1, 900))
        XCTAssertEqual(b.committedLength(id), 900)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 20_000 - bound(1, 900))
        XCTAssertEqual(b.snapshot()["active_committed_bytes"], b.snapshot()["active_reservation_bytes"])
        XCTAssertNil(b.grow(id, current: 899) { _, _ in XCTFail("no prepare with the window off"); return true })
    }

    func testAdmissionDecisionsAreIdenticalWithAndWithoutTheWindow() {
        let lengths = [1000, 300, 1000, 50, 700, 1000, 1000, 20]
        func run(_ w: Int) -> [Bool] {
            let b = budget(window: w)
            return lengths.map { b.reserve(length: $0, current: 10) { _, _ in true } != nil }
        }
        XCTAssertEqual(run(0), run(256))
        XCTAssertEqual(run(0), run(4096))
        XCTAssertTrue(run(256).contains(false), "the sequence must reach the admission bound (instrument sensitivity)")
    }

    func testHotTargetUsesCommittedLengthAndGrowShrinksItAhead() {
        let b = budget(window: 200)
        let id = b.reserve(length: 1000, current: 100) { _, _ in true }!
        XCTAssertEqual(b.committedLength(id), 300)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 20_000 - bound(1, 300))
        XCTAssertEqual(b.snapshot()["active_reservation_bytes"], bound(1, 1000))   // admission still prices 1000
        XCTAssertNil(b.grow(id, current: 200) { _, _ in XCTFail("not within half a window"); return true })
        var seen: [(Double, Double)] = []
        XCTAssertEqual(b.grow(id, current: 201) { t, g in seen.append((t, g)); return true }, true)
        XCTAssertEqual(b.committedLength(id), 401)
        XCTAssertEqual(seen.first?.0, 20_000 - bound(1, 401))
        XCTAssertEqual(seen.first?.1, bound(1, 401) - bound(1, 300))
        XCTAssertEqual(b.snapshot()["admission_growths"], 1)
        // capped at the admitted length, then inert
        XCTAssertEqual(b.grow(id, current: 950) { _, _ in true }, true)
        XCTAssertEqual(b.committedLength(id), 1000)
        XCTAssertNil(b.grow(id, current: 999) { _, _ in XCTFail("at the admitted length"); return true })
        b.release(id)
        XCTAssertEqual(b.committedLength(id), 0)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 20_000)
    }

    func testRefusedReclaimStillCommitsTheGrowthAndIsCounted() {
        let b = budget(window: 200)
        let id = b.reserve(length: 1000, current: 100) { _, _ in true }!
        XCTAssertEqual(b.grow(id, current: 290) { _, _ in false }, false)
        XCTAssertEqual(b.committedLength(id), 490)
        XCTAssertEqual(b.snapshot()["admission_growths_refused"], 1)
    }

    func testLongestCommittedRowPricesEveryRowAndTargetNeverExceedsTheB53TargetInverse() {
        // Two rows: the hot target is priced at the longest committed length (padded pool), and with every row grown to
        // its admitted length the target equals the window-off target exactly.
        let b = budget(window: 100), off = budget(window: 0)
        let a = b.reserve(length: 900, current: 50) { _, _ in true }!, c = b.reserve(length: 600, current: 400) { _, _ in true }!
        _ = off.reserve(length: 900, current: 50) { _, _ in true }; _ = off.reserve(length: 600, current: 400) { _, _ in true }
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 20_000 - bound(2, 500))
        XCTAssertGreaterThan(b.snapshot()["hot_target_bytes"]!, off.snapshot()["hot_target_bytes"]!)
        for cur in stride(from: 60, through: 900, by: 10) { _ = b.grow(a, current: cur) { _, _ in true } }
        for cur in stride(from: 410, through: 600, by: 10) { _ = b.grow(c, current: cur) { _, _ in true } }
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], off.snapshot()["hot_target_bytes"])
    }
}
