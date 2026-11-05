// T-0152: the packed per-row-depth layout (Qwen4ExpPackedRows). A wrong gather hands one row's positions to another
// row's logits silently, so the layout is pinned here, on the CPU, without a model.
import XCTest
import Foundation
import MLX
@testable import Qwen4Exp

final class PackedRowsTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    func testGatherKeepsEachRowsLeadingPositionsRowMajor() {
        let L = Qwen4ExpPackedRows(depths: [2, 4, 1], positions: 4)
        XCTAssertEqual(L.total, 7)
        XCTAssertEqual(L.gather, [0, 1, 4, 5, 6, 7, 8])
    }

    func testSpreadRepeatsTheRowsLastValidPosition() {
        let L = Qwen4ExpPackedRows(depths: [2, 4, 1], positions: 4)
        // row 0: packed 0,1 then pads repeat 1; row 1: packed 2..5; row 2: packed 6 four times
        XCTAssertEqual(L.spread, [0, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6])
    }

    func testSpreadThenGatherIsIdentityOnPackedRows() {
        let L = Qwen4ExpPackedRows(depths: [3, 1, 4, 2], positions: 4)
        let idx = Qwen4ExpPackedIndex(L)
        let F = 3
        let a = MLXArray((0 ..< L.total * F).map { Float($0) }).reshaped([1, L.total, F])
        let full = idx.spread(a)
        XCTAssertEqual(full.shape, [4, 4, F])
        let back = idx.gather(full)
        XCTAssertEqual(back.shape, [1, L.total, F])
        XCTAssertEqual(back.asArray(Float.self), a.asArray(Float.self))
    }

    func testGatherOfPaddedBlockPicksValidPositions() {
        let L = Qwen4ExpPackedRows(depths: [1, 3], positions: 3)
        let idx = Qwen4ExpPackedIndex(L)
        // padded block value = b*10 + s
        let full = MLXArray((0 ..< 2).flatMap { b in (0 ..< 3).map { Float(b * 10 + $0) } }).reshaped([2, 3, 1])
        XCTAssertEqual(idx.gather(full).asArray(Float.self), [0, 10, 11, 12])
    }

    func testUniformDepthIsTheIdentityLayout() {
        let L = Qwen4ExpPackedRows(depths: [4, 4, 4], positions: 4)
        XCTAssertEqual(L.gather, (0 ..< 12).map { Int32($0) })
        XCTAssertEqual(L.spread, (0 ..< 12).map { Int32($0) })
    }
}
