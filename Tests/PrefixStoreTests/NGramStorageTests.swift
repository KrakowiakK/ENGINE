import Foundation
import XCTest
import MLX
import MLXNN
@testable import Qwen4Exp

final class NGramStorageTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Module constructors use MLXRandom's process-global initial state as well as task-local
        // streams. Match the existing storage tests so no initializer needs a GPU metallib.
        Device.setDefault(device: Device.cpu)
    }

    private func makeTable() -> Qwen4ExpShardedEmbedding {
        let table = Qwen4ExpShardedEmbedding(nShards: 3, rows: 16, dim: 64)
        for (i, shard) in table.shards.enumerated() {
            let values = (0..<(16 * 64)).map { Float(($0 * 13 + i * 31) % 257 - 128) / 32 }
            shard.update(parameters: .unflattened(["weight": MLXArray(values).reshaped(16, 64)]))
        }
        quantize(model: table, groupSize: 32, bits: 8)
        eval(table.parameters())
        return table
    }

    private func assertView(_ view: MLXArray, of merged: MLXArray, startingAt row: Int,
                            file: StaticString = #filePath, line: UInt = #line) {
        let source = merged.asData(access: .noCopy)
        let slice = view.asData(access: .noCopy)
        XCTAssertEqual(slice.strides, source.strides, file: file, line: line)
        source.data.withUnsafeBytes { base in
            slice.data.withUnsafeBytes { part in
                XCTAssertEqual(Int(bitPattern: part.baseAddress!),
                               Int(bitPattern: base.baseAddress!) + row * source.strides[0] * merged.itemSize,
                               "the shard must alias the merged allocation at its own row offset",
                               file: file, line: line)
            }
        }
    }

    func testMergedStorageSharesAllParametersAndPreservesExactHostAndDeviceLookup() throws {
        try Device.withDefaultDevice(.cpu) {
            let table = makeTable()
            let ids: [Int64] = [0, 15, 16, 17, 31, 32, 47, 3, 17]
            let before = table(ids).asArray(Float.self)
            let parametersBefore = table.parameters().flattened()
            let layout = Dictionary(uniqueKeysWithValues: parametersBefore.map { ($0.0, $0.1.shape) })
            let moduleIDs = table.shards.map(ObjectIdentifier.init)
            let bytes = parametersBefore.reduce(0) { $0 + $1.1.nbytes }

            let merged = table.mergeQuantizedStorage()

            XCTAssertEqual(table(ids).asArray(Float.self), before)
            XCTAssertEqual(table.shards.map(ObjectIdentifier.init), moduleIDs)
            XCTAssertEqual(Dictionary(uniqueKeysWithValues: table.parameters().flattened().map { ($0.0, $0.1.shape) }), layout)
            XCTAssertEqual(merged.witness["logical_bytes"], bytes)
            XCTAssertEqual(merged.witness["rebound_shard_bytes"], bytes)
            for (i, shard) in table.shards.enumerated() {
                let q = try XCTUnwrap(shard as? QuantizedEmbedding)
                assertView(q.weight, of: merged.weight, startingAt: i * table.rows)
                assertView(q.scales, of: merged.scales, startingAt: i * table.rows)
                assertView(try XCTUnwrap(q.biases), of: try XCTUnwrap(merged.biases), startingAt: i * table.rows)
            }
            let selected = MLXArray(ids.map(Int32.init))
            let deviceLookup = dequantized(merged.weight[selected], scales: merged.scales[selected],
                                           biases: merged.biases?[selected],
                                           groupSize: merged.groupSize, bits: merged.bits)
            XCTAssertEqual(deviceLookup.asType(.float32).asArray(Float.self), before)
        }
    }

    func testShardParameterUpdateCanBeRemergedWithoutChangingLayoutOrOtherRows() throws {
        try Device.withDefaultDevice(.cpu) {
            let table = makeTable()
            let initial = table.mergeQuantizedStorage()
            let oldOutput = table([0, 16, 32]).asArray(Float.self)
            let replacement = QuantizedEmbedding(weight: MLXArray.ones([16, 64]) * Float(7),
                                                 groupSize: 32, bits: 8)
            eval(replacement.parameters())
            let patch = Dictionary(uniqueKeysWithValues: replacement.parameters().flattened().map {
                ("shards.1." + $0.0, $0.1)
            })
            try table.update(parameters: .unflattened(patch), verify: [.noUnusedKeys, .shapeMismatch])
            let expected = table([0, 16, 32]).asArray(Float.self)
            XCTAssertNotEqual(expected, oldOutput)
            XCTAssertEqual(Array(expected[..<64]), Array(oldOutput[..<64]))
            XCTAssertEqual(Array(expected[128...]), Array(oldOutput[128...]))
            let refreshed = table.mergeQuantizedStorage()
            XCTAssertEqual(table([0, 16, 32]).asArray(Float.self), expected)
            let middle = try XCTUnwrap(table.shards[1] as? QuantizedEmbedding)
            assertView(middle.weight, of: refreshed.weight, startingAt: 16)
            // Keeping a previous merged allocation alive cannot make reloading mutate it.
            let original = dequantized(initial.weight[MLXArray([Int32(0), 16, 32])],
                                       scales: initial.scales[MLXArray([Int32(0), 16, 32])],
                                       biases: initial.biases?[MLXArray([Int32(0), 16, 32])],
                                       groupSize: initial.groupSize, bits: initial.bits)
            XCTAssertEqual(original.asArray(Float.self), oldOutput)
        }
    }
}
