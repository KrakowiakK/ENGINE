import Foundation
import MLX
import MLXRandom

/// P095 U3-K -- top-k / nucleus sampling of B rows that share (temp, topK, topP), each row drawing
/// with its OWN key. The O(V) work (temperature, argPartition over the vocabulary, the k-wide sort,
/// the nucleus) runs once over the [B, V] block; the k-wide categorical draw runs per row with that
/// row's key, so a seeded request stays reproducible whatever it is batched with. Every op is
/// row-wise, so row b's token is what the single-row sampler produces on row b's slice with the
/// same key -- see BlockSamplerTests. Returns nil when the parameters take a path this does not
/// cover (greedy, no top-k, top-k >= V); the caller falls back to its per-row loop.
public func sampleTopKBlock(_ logits: MLXArray, temp: Float, topK: Int, topP: Float, draw: (Int) -> MLXArray?) -> MLXArray? {
    let V = logits.dim(-1)
    guard temp > 0, topK > 0, topK < V else { return nil }
    let B = logits.dim(0)
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
