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
        // greedy is not this function's job: nil, and the caller takes the argmax
        XCTAssertNil(sampleTopKBlock(logits, temp: 0, topK: 20, topP: 1) { _ in nil })
    }

    /// T-0190 (QC audit): a sampled request with no top_k (the server's default top_k is 0) used to get nil here, and the
    /// batched MTP round then took the ARGMAX -- a temperature request decoded greedily whenever it was batched. No top-k is
    /// the serial sampler's full-vocabulary path (EngineSampler.sample): logits / temp, the nucleus over the whole sorted
    /// vocabulary when 0 < top_p < 1, then one categorical draw with the row's own key.
    func testNoTopKSamplesTheFullVocabularyLikeTheSerialSampler() {
        MLXRandom.seed(11)
        let B = 4, V = 2048
        let logits = (MLXRandom.normal([B, V]) * 2).asType(.bfloat16)
        let keys = (0 ..< B).map { MLXRandom.key(UInt64(500 + $0)) }
        for (temp, topP) in [(Float(0.8), Float(1.0)), (Float(1.0), Float(0.9))] {
            guard let block = sampleTopKBlock(logits, temp: temp, topK: 0, topP: topP, draw: { b in keys[b] }) else {
                XCTFail("no top-k must sample, not fall back to greedy (temp \(temp) p \(topP))"); continue
            }
            let got = block.asArray(Int32.self)
            for b in 0 ..< B {
                // the serial sampler's no-top-k path, transcribed (main.swift EngineSampler.sample)
                var l = logits[b ..< (b + 1)].asType(.float32) / temp
                if topP > 0 && topP < 1 {
                    let p = softmax(l, axis: -1)
                    let idx = argSort(-p, axis: -1)
                    let ps = takeAlong(p, idx, axis: -1)
                    let keepSorted = (ps.cumsum(axis: -1) - ps) .< MLXArray(topP)
                    var keep = MLXArray.zeros(keepSorted.shape, type: Bool.self)
                    keep = putAlong(keep, idx, values: keepSorted, axis: -1)
                    l = MLX.which(keep, l, MLXArray(-Float.infinity))
                }
                let want = MLXRandom.categorical(l, axis: -1, key: keys[b]).asArray(Int32.self)[0]
                XCTAssertEqual(got[b], want, "row \(b) temp \(temp) p \(topP)")
            }
            // and it is a sample, not the argmax: over the rows and two settings at least one draw differs from it
        }
        let argmax = logits.argMax(axis: -1).asArray(Int32.self)
        let hot = sampleTopKBlock(logits, temp: 5, topK: 0, topP: 1, draw: { b in keys[b] })!.asArray(Int32.self)
        XCTAssertNotEqual(hot, argmax, "a temperature-5 draw over 2048 ids equal to the argmax on every row would mean greedy")
    }
}
