import XCTest
import MLX
import MLXRandom
@testable import EngineServeSupport

/// P095 U3-K -- the block sampler is row-wise: row b of a B-row block, drawn with key k_b, is the token
/// the same function produces on row b alone with the same key. Deterministic (keys fixed).
final class BlockSamplerTests: XCTestCase {
    func testBlockEqualsPerRowWithTheSameKeys() {
        MLXRandom.seed(7)
        let B = 5, V = 4096
        let logits = (MLXRandom.normal([B, V]) * 3).asType(.bfloat16)
        let keys = (0 ..< B).map { MLXRandom.key(UInt64(100 + $0)) }
        for (temp, topK, topP) in [(Float(1.0), 20, Float(0.95)), (Float(0.7), 40, Float(1.0)), (Float(1.0), 20, Float(0.5))] {
            let block = sampleTopKBlock(logits, temp: temp, topK: topK, topP: topP) { b in keys[b] }!.asArray(Int32.self)
            for b in 0 ..< B {
                let single = sampleTopKBlock(logits[b ..< (b + 1)], temp: temp, topK: topK, topP: topP) { _ in keys[b] }!.asArray(Int32.self)
                XCTAssertEqual(block[b], single[0], "row \(b) temp \(temp) k \(topK) p \(topP)")
            }
        }
        // the paths this does not cover return nil so the caller can fall back
        XCTAssertNil(sampleTopKBlock(logits, temp: 0, topK: 20, topP: 1) { _ in nil })
        XCTAssertNil(sampleTopKBlock(logits, temp: 1, topK: 0, topP: 1) { _ in nil })
    }
}
