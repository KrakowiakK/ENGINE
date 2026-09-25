import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import Qwen4Exp

/// CPU-only cache geometry/value tests; no model or checkpoint needed.
final class CacheCapacityTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }
    private func rows(_ start: Int, _ count: Int, batch: Int = 1) -> MLXArray {
        MLXArray((0..<(batch * count)).map { Float(start + $0) }).reshaped([batch, 1, count, 1])
    }
    func testScalarAndLeanGrowthKeepExactHistoryAcrossSoftLimit() {
        for lean in [false, true] {
            let bounded = KVCacheSimple(), control = KVCacheSimple()
            bounded.step = 4; control.step = 4; bounded.capacityGrowthLimit = 13
            var offset = 0
            for width in [5, 3, 4, 1, 2] {
                let k = rows(offset, width), v = k + 100
                if lean { bounded.updateNoReturn(keys: k, values: v); control.updateNoReturn(keys: k, values: v) }
                else { _ = bounded.update(keys: k, values: v); _ = control.update(keys: k, values: v) }
                offset += width
                eval(bounded.state + control.state)
                XCTAssertEqual(bounded.state[0].asArray(Float.self), control.state[0].asArray(Float.self))
                XCTAssertEqual(bounded.state[1].asArray(Float.self), control.state[1].asArray(Float.self))
                XCTAssertEqual(bounded.offset, offset)
                if offset <= 13 { XCTAssertLessThanOrEqual(bounded.rawKeys!.dim(2), 13) }
                else { XCTAssertEqual(bounded.rawKeys!.dim(2), offset) }
            }
            XCTAssertEqual((bounded.copy() as! KVCacheSimple).capacityGrowthLimit, 13)
            _ = bounded.trim(5); _ = control.trim(5)
            let k = rows(80, 4)
            _ = bounded.update(keys: k, values: k); _ = control.update(keys: k, values: k)
            eval(bounded.state + control.state)
            XCTAssertEqual(bounded.state[0].asArray(Float.self), control.state[0].asArray(Float.self))
        }
    }
    func testRaggedGrowthKeepsEveryWrittenRowAndSupportsBeyondLimit() {
        let bounded = KVCacheSimple(), control = KVCacheSimple()
        bounded.step = 4; control.step = 4; bounded.capacityGrowthLimit = 13
        var positions = [0, 0]
        for width in [5, 3, 4, 1, 2] {
            let k = rows(positions[0], width, batch: 2), v = k + 100
            bounded.updateRagged(keys: k, values: v, at: positions)
            control.updateRagged(keys: k, values: v, at: positions)
            positions = positions.map { $0 + width }
            eval(bounded.state + control.state)
            XCTAssertEqual(bounded.state[0].asArray(Float.self), control.state[0].asArray(Float.self))
            XCTAssertEqual(bounded.state[1].asArray(Float.self), control.state[1].asArray(Float.self))
            XCTAssertEqual(bounded.rawKeys!.dim(2), min(control.rawKeys!.dim(2), max(13, positions[0])))
        }
        // Unequal positions overwrite each row at its own location, preserving other values.
        let k = rows(80, 1, batch: 2)
        bounded.updateRagged(keys: k, values: k, at: [13, 10])
        control.updateRagged(keys: k, values: k, at: [13, 10])
        eval(bounded.state + control.state)
        XCTAssertEqual(bounded.state[0].asArray(Float.self), control.state[0].asArray(Float.self))
    }
    func testRawIndexerAndVisualPositionCapacityPreserveExactWrittenValues() throws {
        let a = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data("{}".utf8))
        let indexer = Qwen4ExpQSAIndexer(a), controlIndexer = Qwen4ExpQSAIndexer(a)
        let rope = Qwen4ExpRotary(dim: 64, base: 10_000_000)
        let capped = ArraysCache(size: 3), control = ArraysCache(size: 3)
        defer { Qwen4ExpCacheCapacity.configureGrowthLimit(nil) }
        var offset = 0
        for width in [5, 8, 1] {
            let qk = MLXArray((0..<(width * indexer.projDim)).map { Float(offset + $0) })
                .reshaped([1, width, indexer.projDim])
            let pos = MLXArray((0..<(3 * width)).map { Int32(offset + $0) }).reshaped([3, 1, width])
            Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
            XCTAssertNil(controlIndexer(qk: qk, rope: rope, keyCache: control, offset: offset, pos3: pos))
            Qwen4ExpCacheCapacity.configureGrowthLimit(13)
            XCTAssertNil(indexer(qk: qk, rope: rope, keyCache: capped, offset: offset, pos3: pos))
            offset += width
            eval(capped.state + control.state)
            XCTAssertEqual(capped[0]!.dim(1), max(13, offset))
            XCTAssertEqual(capped[2]!.dim(2), max(13, offset))
            XCTAssertEqual(capped[0]![0..., 0..<offset, 0...].asArray(Float.self),
                           control[0]![0..., 0..<offset, 0...].asArray(Float.self))
            XCTAssertEqual(capped[2]![0..., 0..., 0..<offset].asArray(Int32.self),
                           control[2]![0..., 0..., 0..<offset].asArray(Int32.self))
        }
    }

    func testQwenFactoryAndIndexerUnitsUseOnlyOptionalAllocationPolicy() {
        Qwen4ExpCacheCapacity.configureGrowthLimit(13)
        defer { Qwen4ExpCacheCapacity.configureGrowthLimit(nil) }
        XCTAssertEqual(Qwen4ExpCacheCapacity.makeKVCache().capacityGrowthLimit, 13)
        XCTAssertEqual(Qwen4ExpCacheCapacity.bounded(1024, need: 12), 13)
        XCTAssertEqual(Qwen4ExpCacheCapacity.bounded(1024, need: 14), 14)
        XCTAssertEqual(Qwen4ExpCacheCapacity.bounded(256, need: 3, ratio: 4), 3)
        XCTAssertEqual(Qwen4ExpCacheCapacity.bounded(256, need: 4, ratio: 4), 4)
        Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
        XCTAssertNil(Qwen4ExpCacheCapacity.makeKVCache().capacityGrowthLimit)
        XCTAssertEqual(Qwen4ExpCacheCapacity.bounded(1024, need: 12), 1024)
    }

    func testHotRungBoundIncludesPrefillRollbackBuffersAndAllocationSlack() throws {
        var a = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data("{}".utf8))
        a.layerTypes = Array(repeating: "linear_attention", count: 36)
            + Array(repeating: "full_attention", count: 12)
        // Independent E9 exported-array geometry: 36 SSM,36 conv,PLE state,
        // ngram state and (for S>1) the two rollback buffers.
        let gdn = 36 * (48 * 128 * 128 * 4 + 3 * 10240 * 2)
        let stable = Double(gdn + 9 * 10240 * 2 + 2 * 4)
        XCTAssertEqual(stable, 115_642_376)
        XCTAssertEqual(Qwen4ExpCacheCapacity.hotRungBytes(a, maxForwardRows: 1, allocationSlack: 0), stable)
        let scratch1024 = Double((1024 + 9) * 10240 * 2 + (1024 + 2) * 4)
        XCTAssertEqual(Qwen4ExpCacheCapacity.hotRungBytes(a, maxForwardRows: 1024, allocationSlack: 0), stable + scratch1024)
        XCTAssertEqual(Qwen4ExpCacheCapacity.hotRungBytes(a, maxForwardRows: 1024, allocationSlack: 49_152),
                       stable + scratch1024 + 76 * 49_152)
        a.pleLayerIds = []
        XCTAssertEqual(Qwen4ExpCacheCapacity.hotRungBytes(a, maxForwardRows: 4096, allocationSlack: 0),
                       Double(gdn))
    }
}
