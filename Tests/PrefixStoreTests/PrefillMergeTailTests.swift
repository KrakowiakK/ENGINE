import XCTest
@testable import EngineServeSupport

/// T-0111: ENGINE_PREFILL_MERGE_TAIL -- a prompt finishes in one chunk across the QSA budget cut when the remainder fits.
final class PrefillMergeTailTests: XCTestCase {
    private func end(_ from: Int, _ n: Int, merge: Int, reserve: Int = 0) -> Int {
        PrefillChunkPlan.end(from: from, width: 4096, rungAlign: 1, promptCount: n, qsaBudgetCut: 2048,
                             canonicalTarget: nil, hotReserve: reserve, mergeTail: merge)
    }
    func testOffKeepsTheBudgetCut() {
        XCTAssertEqual(end(0, 2500, merge: 0), 2048)
        XCTAssertEqual(end(1024, 2500, merge: 0), 2048)
    }
    func testMergesARemainderThatFits() {
        XCTAssertEqual(end(0, 2500, merge: 4096), 2500)
        XCTAssertEqual(end(1024, 2500, merge: 4096), 2500)
        XCTAssertEqual(end(1024, 2500, merge: 4096, reserve: 1), 2499)
    }
    func testKeepsTheCutWhenTheRemainderIsTooLong() {
        XCTAssertEqual(end(0, 5000, merge: 4096), 2048)
        XCTAssertEqual(end(1024, 5200, merge: 4096), 2048)
    }
    func testPromptsBelowTheBudgetAreUnchanged() {
        XCTAssertEqual(end(0, 1800, merge: 4096), end(0, 1800, merge: 0))
        XCTAssertEqual(end(2048, 3000, merge: 4096), end(2048, 3000, merge: 0))
    }
}
