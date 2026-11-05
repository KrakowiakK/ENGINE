// T-0142: the row bookkeeping of the mixed prefill+verify forward (Qwen4ExpMixedRows). A wrong split hands one sequence's
// rows to another's state silently, so the layout is pinned here, on the CPU, without a model.
import XCTest
import Foundation
import MLX
@testable import Qwen4Exp

final class MixedRowsTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    /// a [1, n, F] block whose row r holds r*10 + f in feature f: every row is recognisable after a reshape
    private func rows(_ n: Int, features F: Int, base: Int = 0) -> MLXArray {
        MLXArray((0 ..< n).flatMap { r in (0 ..< F).map { Float((base + r) * 10 + $0) } }).reshaped([1, n, F])
    }

    func testSplitPutsChunkRowsFirstThenVerifyRowsRowMajor() {
        let L = Qwen4ExpMixedRows(chunkRows: 5, batch: 3, positions: 4)
        XCTAssertEqual(L.total, 17)
        let (c, v) = L.split(rows(17, features: 2))
        XCTAssertEqual(c.shape, [1, 5, 2]); XCTAssertEqual(v.shape, [3, 4, 2])
        XCTAssertEqual(c[0, 4, 0].item(Float.self), 40)            // the chunk's last row
        XCTAssertEqual(v[0, 0, 0].item(Float.self), 50)            // verify row 0, position 0 = block row 5
        XCTAssertEqual(v[1, 0, 0].item(Float.self), 90)            // verify row 1 starts S = 4 rows later
        XCTAssertEqual(v[2, 3, 1].item(Float.self), 161)           // the last block row, feature 1
    }

    func testJoinInvertsSplitExactly() {
        let L = Qwen4ExpMixedRows(chunkRows: 7, batch: 2, positions: 4)
        let a = rows(L.total, features: 3)
        let (c, v) = L.split(a)
        XCTAssertTrue(arrayEqual(L.join(c, v), a).item(Bool.self))
    }

    func testJoinOfSeparateGroupsMatchesTheirConcatenation() {
        // what the mixed forward feeds the shared layers: the chunk's rows, then each verify row's S positions in order
        let L = Qwen4ExpMixedRows(chunkRows: 3, batch: 2, positions: 2)
        let chunk = rows(3, features: 1)
        let verify = MLXArray([Float(100), 101, 200, 201]).reshaped([2, 2, 1])
        XCTAssertEqual(L.join(chunk, verify).reshaped([-1]).asArray(Float.self), [0, 10, 20, 100, 101, 200, 201])
    }

    func testSingleVerifyRowAndSinglePosition() {
        let L = Qwen4ExpMixedRows(chunkRows: 1, batch: 1, positions: 1)
        let (c, v) = L.split(rows(2, features: 4))
        XCTAssertEqual(c.shape, [1, 1, 4]); XCTAssertEqual(v.shape, [1, 1, 4])
        XCTAssertEqual(v[0, 0, 3].item(Float.self), 13)
    }
}
