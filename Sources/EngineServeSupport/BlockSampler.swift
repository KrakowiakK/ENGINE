import Foundation
import MLX
import MLXRandom

/// P095 U3-K -- top-k / nucleus sampling of B rows that share (temp, topK, topP), each row drawing
/// with its OWN key. The O(V) work (temperature, argPartition over the vocabulary, the k-wide sort,
/// the nucleus) runs once over the [B, V] block; the k-wide categorical draw runs per row with that
/// row's key, so a seeded request stays reproducible whatever it is batched with. Every op is
/// row-wise, so row b's token is what the single-row sampler produces on row b's slice with the
/// same key -- see BlockSamplerTests. Returns nil only for greedy (temp <= 0): the caller takes the argmax.
/// T-0190: no top-k (0, or >= V) is the serial sampler's full-vocabulary path; it used to return nil too, and the batched MTP
/// round's fallback for nil is the ARGMAX, so a temperature request with no top_k decoded greedily whenever it was batched.
public func sampleTopKBlock(_ logits: MLXArray, temp: Float, topK: Int, topP: Float, draw: (Int) -> MLXArray?) -> MLXArray? {
    let V = logits.dim(-1)
    guard temp > 0 else { return nil }
    let B = logits.dim(0)
    guard topK > 0, topK < V else { return sampleFullBlock(logits, temp: temp, topP: topP, draw: draw) }
    let l = logits.asType(.float32) / temp
    let part = argPartition(-l, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]   // top-k ids, unordered
    var vals = takeAlong(l, part, axis: -1)                                        // (B, k)
    let ord = argSort(-vals, axis: -1)                                             // descending, k-wide
    vals = takeAlong(vals, ord, axis: -1)
    let ids = takeAlong(part, ord, axis: -1)
    if topP > 0 && topP < 1 {
        let p = softmax(vals, axis: -1)
        vals = MLX.which((p.cumsum(axis: -1) - p) .< MLXArray(topP), vals, MLXArray(-Float.infinity))
    }
    var picks: [MLXArray] = []
    for b in 0 ..< B { picks.append(MLXRandom.categorical(vals[b ..< (b + 1)], axis: -1, key: draw(b))) }
    let pick = concatenated(picks, axis: 0)                                        // (B)
    return takeAlong(ids, pick[.ellipsis, .newAxis], axis: -1).squeezed(axis: -1)  // (B)
}

/// T-0190: the no-top-k rows of a block, row-wise exactly as EngineSampler.sample's no-top-k path (main.swift): logits / temp
/// in float32, the nucleus over the whole sorted vocabulary when 0 < top_p < 1, one categorical draw per row with its key.
func sampleFullBlock(_ logits: MLXArray, temp: Float, topP: Float, draw: (Int) -> MLXArray?) -> MLXArray {
    let B = logits.dim(0)
    var l = logits.asType(.float32) / temp
    if topP > 0 && topP < 1 {
        let p = softmax(l, axis: -1)
        let idx = argSort(-p, axis: -1)
        let ps = takeAlong(p, idx, axis: -1)
        let keepSorted = (ps.cumsum(axis: -1) - ps) .< MLXArray(topP)
        var keep = MLXArray.zeros(keepSorted.shape, type: Bool.self)
        keep = putAlong(keep, idx, values: keepSorted, axis: -1)
        l = MLX.which(keep, l, MLXArray(-Float.infinity))
    }
    var picks: [MLXArray] = []
    for b in 0 ..< B { picks.append(MLXRandom.categorical(l[b ..< (b + 1)], axis: -1, key: draw(b))) }
    return concatenated(picks, axis: 0)
}
