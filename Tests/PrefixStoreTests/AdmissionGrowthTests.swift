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
