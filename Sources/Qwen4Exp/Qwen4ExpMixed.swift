// T-0142 (D-0043): one forward over a prefill chunk and a batched verify block -- the "mixed step" vLLM runs for
// Qwen3-Next-style hybrids (C-0401), in the split-merge form: the row-wise work (hyper-connections, the projections
// inside them, router, routed and shared experts, combines, final mixer) runs ONCE over the concatenated rows, so the
// verify rows share the expert weight reads of the chunk; the parts with per-sequence state run per group on their
// existing paths and caches:
//   - PLE (short conv over the sequence and the n-gram lookback; not row-wise, unlike D-0042 assumed),
//   - the mixer: GatedDeltaNet (conv + recurrence on the chunk's own cache / on the stacked pool with the verify tape)
//     and QSA attention (prefill branch with the chunk's mask / the ragged branch under the batch row globals),
//   - the attention mask and the n-gram history.
// Row layout everywhere: the chunk's Sc rows first, then the verify block's B x S rows (row-major), as [1, Sc + B*S, .].
// Lab path only (batchprobe --mixed-chunk); nothing in the server calls it yet.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// The row bookkeeping of a mixed block, kept apart so it is testable without a model.
public struct Qwen4ExpMixedRows: Sendable, Equatable {
    public let chunkRows: Int
    public let batch: Int
    public let positions: Int
    public init(chunkRows: Int, batch: Int, positions: Int) {
        precondition(chunkRows > 0 && batch > 0 && positions > 0)
        self.chunkRows = chunkRows; self.batch = batch; self.positions = positions
    }
    public var total: Int { chunkRows + batch * positions }

    /// [1, Sc + B*S, F] -> ([1, Sc, F], [B, S, F])
    public func split(_ a: MLXArray) -> (chunk: MLXArray, verify: MLXArray) {
        precondition(a.dim(0) == 1 && a.dim(1) == total, "mixed block has \(a.shape), expected [1, \(total), F]")
        return (a[0..., 0 ..< chunkRows], a[0..., chunkRows...].reshaped([batch, positions, a.dim(-1)]))
    }

    /// ([1, Sc, F], [B, S, F]) -> [1, Sc + B*S, F]
    public func join(_ chunk: MLXArray, _ verify: MLXArray) -> MLXArray {
        precondition(chunk.dim(0) == 1 && chunk.dim(1) == chunkRows && verify.dim(0) == batch && verify.dim(1) == positions)
        return concatenated([chunk, verify.reshaped([1, batch * positions, verify.dim(-1)])], axis: 1)
    }
}

/// What one group needs besides its rows: caches, PLE n-gram inputs and the attention mask.
struct Qwen4ExpMixedGroup {
    let cache: [KVCache]
    let ids: MLXArray
    let prev: MLXArray
    let mask: MLXFast.ScaledDotProductAttentionMaskMode
    let pos3: MLXArray?
    /// the verify group of a batch pool: its per-row offsets and rope shifts (the ragged path); nil = the group's caches
    /// are one sequence's own and it runs the ordinary (serial) path -- T-0148, a lone decoder's verify block
    var rows: [Int]? = nil
    var deltas: [Int]? = nil
    /// run `body` the way this group's per-sequence modules must run: under the batch row globals when ragged
    func run<R>(_ body: () -> R) -> R {
        guard let rows else { return body() }
        return Qwen4ExpBatch.with(rows, deltas: deltas, body)
    }
}

extension Qwen4ExpDecoderLayer {
    /// The layer over a mixed block (see the file header). `rows`/`deltas` are the verify rows' cache offsets and rope
    /// shifts, installed only around the verify group's per-sequence calls.
    func callChainedMixed(_ h: MLXArray, layout: Qwen4ExpMixedRows, layerIndex i: Int, rope: Qwen4ExpRotary,
                          chunk: Qwen4ExpMixedGroup, verify: Qwen4ExpMixedGroup,
                          normedIn: MLXArray?, next: Qwen4ExpGatedResidual?, mropeSection: [Int]) -> (MLXArray, MLXArray?) {
        precondition(inputNorm == nil, "the mixed forward is for the hyper-connection model")
        var h = h
        var normedIn = normedIn
        let cC = chunk.cache[i], cV = verify.cache[i]
        if let ple, !Qwen4ExpEnv.noPLE {
            let (hC, hV) = layout.split(h)
            let pC = ple(hC, ids: chunk.ids, prev: chunk.prev, cache: cC as? ArraysCache)
            let pV = verify.run { ple(hV, ids: verify.ids, prev: verify.prev, cache: cV as? ArraysCache) }
            h = h + layout.join(pC, pV)
            normedIn = nil
        }
        let (x0, inj) = attnHC(h, normed: normedIn)
        var x = x0
        let (xC, xV) = layout.split(x)
        if let linearAttn {
            let yC = linearAttn(xC, cache: cC as? ArraysCache)
            let yV = verify.run { linearAttn(xV, cache: cV as? ArraysCache) }
            x = layout.join(yC, yV)
        } else {
            let lC = cC as? CacheList, lV = cV as? CacheList
            let yC = selfAttn!(xC, rope: rope, mask: chunk.mask, cache: lC?[0], indexerCache: lC?[1] as? ArraysCache,
                               pos3: chunk.pos3, mropeSection: mropeSection)
            let yV = verify.run {
                selfAttn!(xV, rope: rope, mask: verify.mask, cache: lV?[0], indexerCache: lV?[1] as? ArraysCache,
                          pos3: verify.pos3, mropeSection: mropeSection)
            }
            x = layout.join(yC, yV)
        }
        let (h1, n1) = combineChained(h, x, inj!, next: mlpHC)
        h = h1
        let (y, inj2) = mlpHC(h, normed: n1)
        let m = mlp(y)
        return combineChained(h, m, inj2!, next: next)
    }
}

extension Qwen4ExpModelInner {
    /// The n-gram inputs of one group, with the same cache update as the ordinary forward's preamble.
    private func mixedNgram(_ inputs: MLXArray, _ caches: [KVCache]) -> (ids: MLXArray, prev: MLXArray) {
        let idsDev = inputs.asType(.int32)
        var prev = MLXArray.full([inputs.dim(0), ngramContext], values: MLXArray(Int32(eos)), dtype: .int32)
        if let pleIdx, let pc = caches[pleIdx] as? ArraysCache {
            if let stored = pc[3] { prev = stored }
            let hist = concatenated([prev, idsDev], axis: 1)
            pc[3] = hist[0..., (hist.dim(1) - ngramContext)...]
            pc[5] = inputs.dim(1) > 1 ? hist : nil
        }
        return (idsDev, prev)
    }

    /// Returns the final mixer output and the pre-mixer stream of both groups:
    /// chunk [1, Sc, d] / [1, Sc, hc*d], verify [B, S, d] / [B, S, hc*d].
    func callMixed(chunk: MLXArray, chunkCache: [KVCache], chunkEmbeds: MLXArray?, chunkPos3: MLXArray?,
                   verify: MLXArray, verifyCache: [KVCache], rows: [Int]?, deltas: [Int]?, verifyPos3: MLXArray? = nil)
        -> (chunk: MLXArray, verify: MLXArray, chunkHidden: MLXArray, verifyHidden: MLXArray) {
        precondition(norm == nil, "the mixed forward is for the hyper-connection model")
        precondition(chunk.dim(0) == 1 && (rows.map { $0.count == verify.dim(0) } ?? (verify.dim(0) == 1)),
                     "a ragged verify group has one offset per row; a serial one is a single sequence")
        precondition(rows == nil || verifyPos3 == nil, "verify pos3 is for the serial (own-cache) group only")
        let layout = Qwen4ExpMixedRows(chunkRows: chunk.dim(1), batch: verify.dim(0), positions: verify.dim(1))
        let eC = chunkEmbeds ?? embedTokens(chunk)
        let eV = embedTokens(verify)
        let maskC = createAttentionMask(h: eC, cache: ((chunkCache[faIdx] as? CacheList)?[0]).map { [$0] })
        let nC = mixedNgram(chunk, chunkCache)
        let nV = mixedNgram(verify, verifyCache)
        let gC = Qwen4ExpMixedGroup(cache: chunkCache, ids: nC.ids, prev: nC.prev, mask: maskC, pos3: chunkPos3)
        var gV = Qwen4ExpMixedGroup(cache: verifyCache, ids: nV.ids, prev: nV.prev, mask: .none, pos3: verifyPos3,
                                    rows: rows, deltas: deltas)
        let maskV = gV.run { createAttentionMask(h: eV, cache: ((verifyCache[faIdx] as? CacheList)?[0]).map { [$0] }) }
        gV = Qwen4ExpMixedGroup(cache: verifyCache, ids: nV.ids, prev: nV.prev, mask: maskV, pos3: verifyPos3,
                                rows: rows, deltas: deltas)
        var h = tiled(layout.join(eC, eV), repetitions: [1, 1, hc])
        var normedNext: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            if i >= Qwen4ExpDecoderLayer.ablLayers { break }
            let nextHC: Qwen4ExpGatedResidual? = i + 1 < layers.count ? layers[i + 1].attnHC : mixer
            let (hh, nn) = layer.callChainedMixed(h, layout: layout, layerIndex: i, rope: rope, chunk: gC, verify: gV,
                                                  normedIn: normedNext, next: nextHC, mropeSection: mropeSection)
            h = hh; normedNext = nn
        }
        let (hidC, hidV) = layout.split(h)
        let (outC, outV) = layout.split(mixer(h, normed: normedNext).0)
        return (outC, outV, hidC, hidV)
    }
}

extension Qwen4ExpModel {
    /// T-0142: the mixed step. Returns what the two separate calls return: the chunk's LAST-row logits [1, 1, V] and its
    /// pre-mixer stream [1, Sc, hc*d] (as `forwardHidden` on the chunk), and the verify block's logits [B, S, V] and
    /// stream [B, S, hc*d] (as `forwardHidden` on the block under `Qwen4ExpBatch.with(rows, deltas:)`). The heads run per
    /// group through the same code the separate calls use, so the head's kernel choice does not change.
    /// `rows: nil` (T-0148): the verify block is ONE sequence on its own caches -- a lone decoder's serial MTP round -- and
    /// its per-sequence modules run the ordinary path (`forwardHidden(block, cache:)`, with `verifyPos3` for an image
    /// row), exactly the serial program; with rows it is a batch pool's ragged block.
    public func forwardMixed(chunk: MLXArray, chunkCache: [KVCache], chunkEmbeds: MLXArray? = nil, chunkPos3: MLXArray? = nil,
                             verify: MLXArray, verifyCache: [KVCache], rows: [Int]?, deltas: [Int]? = nil,
                             verifyPos3: MLXArray? = nil)
        -> (chunkLogits: MLXArray, chunkHidden: MLXArray, verifyLogits: MLXArray, verifyHidden: MLXArray) {
        let r = model.callMixed(chunk: chunk, chunkCache: chunkCache, chunkEmbeds: chunkEmbeds, chunkPos3: chunkPos3,
                                verify: verify, verifyCache: verifyCache, rows: rows, deltas: deltas, verifyPos3: verifyPos3)
        let sc = r.chunk.dim(1)
        let lastC = Self.lastLogitsOnly && sc > 16 ? r.chunk[0..., (sc - 1)..., 0...] : r.chunk
        let chunkVision = chunkEmbeds != nil || chunkPos3 != nil
        let cl = chunkVision ? (lmHead.map { $0(lastC) } ?? model.embedTokens.asLinear(lastC)) : headLogits(lastC)
        // the head the separate call would use: the multimodal entry point's lmHead for an image row's serial block
        let vl = verifyPos3 != nil ? (lmHead.map { $0(r.verify) } ?? model.embedTokens.asLinear(r.verify)) : headLogits(r.verify)
        return (cl, r.chunkHidden, vl, r.verifyHidden)
    }
}

// T-0152 (C-0439): per-row draft depth, PACKED. A batched verify block [B, S] in which row b has only depths[b] valid
// positions (its first depths[b] tokens; S = the deepest row). The row-wise work (hyper-connections and their
// projections, router, experts, combines, final mixer, head) runs on the P = sum(depths) valid rows only; the parts
// with per-sequence state (PLE, GDN, QSA) keep the padded [B, S] block on their existing ragged paths, so no pool
// cache is split. A pad position repeats its row's last valid row: the mixers are causal, so a valid position never
// sees a pad, and the pads' cache writes are what the round's rollback discards anyway (forced rejection, a <= k_b).
// Lab path only (batchprobe --depths); nothing in the server calls it.
public struct Qwen4ExpPackedRows: Sendable, Equatable {
    public let depths: [Int]
    public let positions: Int
    /// packed row p -> flat padded index b*S + s
    public let gather: [Int32]
    /// flat padded index b*S + s -> packed row of (b, min(s, depths[b] - 1))
    public let spread: [Int32]
    public init(depths: [Int], positions S: Int) {
        precondition(!depths.isEmpty && S > 0 && depths.allSatisfy { $0 >= 1 && $0 <= S },
                     "every row keeps 1...S positions (S = \(S)): \(depths)")
        self.depths = depths; positions = S
        var g: [Int32] = [], s: [Int32] = []
        for (b, d) in depths.enumerated() {
            let first = g.count
            for t in 0 ..< d { g.append(Int32(b * S + t)) }
            for t in 0 ..< S { s.append(Int32(first + min(t, d - 1))) }
        }
        gather = g; spread = s
    }
    public var batch: Int { depths.count }
    public var total: Int { gather.count }
}

/// The packed layout's two gathers on device.
struct Qwen4ExpPackedIndex {
    let rows: Qwen4ExpPackedRows
    let g: MLXArray, s: MLXArray
    init(_ rows: Qwen4ExpPackedRows) { self.rows = rows; g = MLXArray(rows.gather); s = MLXArray(rows.spread) }
    /// [1, P, F] -> [B, S, F]
    func spread(_ a: MLXArray) -> MLXArray {
        let F = a.dim(-1)
        return a.reshaped([rows.total, F]).take(s, axis: 0).reshaped([rows.batch, rows.positions, F])
    }
    /// [B, S, F] -> [1, P, F]
    func gather(_ a: MLXArray) -> MLXArray {
        let F = a.dim(-1)
        return a.reshaped([rows.batch * rows.positions, F]).take(g, axis: 0).reshaped([1, rows.total, F])
    }
}

extension Qwen4ExpDecoderLayer {
    /// The layer over a packed verify block (see Qwen4ExpPackedRows): `h` is [1, P, hc*d].
    func callChainedPacked(_ h: MLXArray, packed: Qwen4ExpPackedIndex, layerIndex i: Int, rope: Qwen4ExpRotary,
                           group: Qwen4ExpMixedGroup, normedIn: MLXArray?, next: Qwen4ExpGatedResidual?,
                           mropeSection: [Int]) -> (MLXArray, MLXArray?) {
        precondition(inputNorm == nil, "the packed forward is for the hyper-connection model")
        var h = h
        var normedIn = normedIn
        let c = group.cache[i]
        if let ple, !Qwen4ExpEnv.noPLE {
            let p = group.run { ple(packed.spread(h), ids: group.ids, prev: group.prev, cache: c as? ArraysCache) }
            h = h + packed.gather(p)
            normedIn = nil
        }
        let (x0, inj) = attnHC(h, normed: normedIn)
        let xs = packed.spread(x0)
        let y: MLXArray
        if let linearAttn {
            y = group.run { linearAttn(xs, cache: c as? ArraysCache) }
        } else {
            let l = c as? CacheList
            y = group.run {
                selfAttn!(xs, rope: rope, mask: group.mask, cache: l?[0], indexerCache: l?[1] as? ArraysCache,
                          pos3: nil, mropeSection: mropeSection)
            }
        }
        let (h1, n1) = combineChained(h, packed.gather(y), inj!, next: mlpHC)
        h = h1
        let (yy, inj2) = mlpHC(h, normed: n1)
        let m = mlp(yy)
        return combineChained(h, m, inj2!, next: next)
    }
}

extension Qwen4ExpModelInner {
    /// The final mixer output [1, P, d] and the pre-mixer stream [1, P, hc*d] of a packed verify block.
    func callPacked(_ verify: MLXArray, cache: [KVCache], rows: [Int], deltas: [Int]?, packed: Qwen4ExpPackedRows)
        -> (out: MLXArray, hidden: MLXArray) {
        precondition(norm == nil, "the packed forward is for the hyper-connection model")
        precondition(verify.dim(0) == packed.batch && verify.dim(1) == packed.positions && rows.count == packed.batch,
                     "packed layout \(packed.batch) x \(packed.positions) for a block \(verify.shape) with \(rows.count) offsets")
        let idx = Qwen4ExpPackedIndex(packed)
        let e = embedTokens(verify)
        let n = mixedNgram(verify, cache)
        var g = Qwen4ExpMixedGroup(cache: cache, ids: n.ids, prev: n.prev, mask: .none, pos3: nil, rows: rows, deltas: deltas)
        let mask = g.run { createAttentionMask(h: e, cache: ((cache[faIdx] as? CacheList)?[0]).map { [$0] }) }
        g = Qwen4ExpMixedGroup(cache: cache, ids: n.ids, prev: n.prev, mask: mask, pos3: nil, rows: rows, deltas: deltas)
        var h = idx.gather(tiled(e, repetitions: [1, 1, hc]))
        var normedNext: MLXArray? = nil
        let every = Self.dispatchEvery   // the ordinary verify block's sliced dispatch (2..16 positions)
        for (i, layer) in layers.enumerated() {
            if i >= Qwen4ExpDecoderLayer.ablLayers { break }
            if every > 0 && i > 0 && i % every == 0 { asyncEval(h) }
            let nextHC: Qwen4ExpGatedResidual? = i + 1 < layers.count ? layers[i + 1].attnHC : mixer
            let (hh, nn) = layer.callChainedPacked(h, packed: idx, layerIndex: i, rope: rope, group: g,
                                                   normedIn: normedNext, next: nextHC, mropeSection: mropeSection)
            h = hh; normedNext = nn
        }
        return (mixer(h, normed: normedNext).0, h)
    }
}

extension Qwen4ExpModel {
    /// T-0152: a packed per-row-depth verify block. Returns the logits [1, P, V] and the pre-mixer stream [1, P, hc*d]
    /// of the valid positions, in Qwen4ExpPackedRows.gather order; the caller installs nothing (the rows go in here).
    public func forwardPacked(_ verify: MLXArray, cache: [KVCache], rows: [Int], deltas: [Int]? = nil,
                              packed: Qwen4ExpPackedRows) -> (logits: MLXArray, hidden: MLXArray) {
        let r = model.callPacked(verify, cache: cache, rows: rows, deltas: deltas, packed: packed)
        return (headLogits(r.out), r.hidden)
    }
}

extension Qwen4ExpModel {
    /// T-0151 slot probe witness (ENGINE_GDN_SLOTS=1): verify forwards that took the slot kernel
    public static var gdnSlotForwards: Int { Qwen4ExpGatedDeltaNet.slotForwards }
}

// MARK: - T-0180: varlen multi-prompt prefill (lab prototype)
//
// N independent prompts' prefill chunks in ONE forward, vLLM-style flattening in the split-merge form of the mixed step
// above: the row-wise work (hyper-connections and their projections, router, routed and shared experts, combines, final
// mixer) runs ONCE over the concatenated rows, so the segments share every weight read and the experts see N times the
// rows; PLE, the mixer (GDN / QSA attention), the attention mask and the n-gram history run per segment on its own caches
// through the ordinary single-sequence path, exactly as the mixed step's chunk does. Text segments only. Lab path only
// (`engine packprobe`); nothing in the server calls it.

/// The row bookkeeping of a packed block: segment i owns rows [offset_i, offset_i + len_i) of [1, sum(len), F].
public struct Qwen4ExpPackedSegments: Sendable, Equatable {
    public let lengths: [Int]
    public init(lengths: [Int]) { precondition(!lengths.isEmpty && lengths.allSatisfy { $0 > 0 }); self.lengths = lengths }
    public var total: Int { lengths.reduce(0, +) }
    public func split(_ a: MLXArray) -> [MLXArray] {
        precondition(a.dim(0) == 1 && a.dim(1) == total, "packed block has \(a.shape), expected [1, \(total), F]")
        var o = 0
        return lengths.map { n in defer { o += n }; return a[0..., o ..< (o + n)] }
    }
    public func join(_ parts: [MLXArray]) -> MLXArray {
        precondition(parts.count == lengths.count && zip(parts, lengths).allSatisfy { $0.dim(0) == 1 && $0.dim(1) == $1 })
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }
}

extension Qwen4ExpDecoderLayer {
    func callChainedPacked(_ h: MLXArray, segments: Qwen4ExpPackedSegments, layerIndex i: Int, rope: Qwen4ExpRotary,
                           groups: [Qwen4ExpMixedGroup], normedIn: MLXArray?, next: Qwen4ExpGatedResidual?,
                           mropeSection: [Int]) -> (MLXArray, MLXArray?) {
        precondition(inputNorm == nil, "the packed forward is for the hyper-connection model")
        var h = h
        var normedIn = normedIn
        if let ple, !Qwen4ExpEnv.noPLE {
            let parts = segments.split(h)
            h = h + segments.join(zip(parts, groups).map { x, g in ple(x, ids: g.ids, prev: g.prev, cache: g.cache[i] as? ArraysCache) })
            normedIn = nil
        }
        let (x0, inj) = attnHC(h, normed: normedIn)
        let xs = segments.split(x0)
        let ys: [MLXArray] = zip(xs, groups).map { x, g in
            if let linearAttn { return linearAttn(x, cache: g.cache[i] as? ArraysCache) }
            let l = g.cache[i] as? CacheList
            return selfAttn!(x, rope: rope, mask: g.mask, cache: l?[0], indexerCache: l?[1] as? ArraysCache, pos3: nil,
                             mropeSection: mropeSection)
        }
        let (h1, n1) = combineChained(h, segments.join(ys), inj!, next: mlpHC)
        h = h1
        let (y, inj2) = mlpHC(h, normed: n1)
        let m = mlp(y)
        return combineChained(h, m, inj2!, next: next)
    }
}

extension Qwen4ExpModelInner {
    /// Final mixer output and pre-mixer stream per segment ([1, S_i, d] / [1, S_i, hc*d]).
    func callPacked(inputs: [MLXArray], caches: [[KVCache]]) -> [(out: MLXArray, hidden: MLXArray)] {
        precondition(norm == nil, "the packed forward is for the hyper-connection model")
        precondition(inputs.count == caches.count && inputs.allSatisfy { $0.dim(0) == 1 })
        let segments = Qwen4ExpPackedSegments(lengths: inputs.map { $0.dim(1) })
        let embeds = inputs.map { embedTokens($0) }
        let groups: [Qwen4ExpMixedGroup] = zip(inputs, zip(embeds, caches)).map { x, ec in
            let (e, c) = ec
            let n = mixedNgram(x, c)
            return Qwen4ExpMixedGroup(cache: c, ids: n.ids, prev: n.prev,
                                      mask: createAttentionMask(h: e, cache: ((c[faIdx] as? CacheList)?[0]).map { [$0] }), pos3: nil)
        }
        var h = tiled(segments.join(embeds), repetitions: [1, 1, hc])
        var normedNext: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            if i >= Qwen4ExpDecoderLayer.ablLayers { break }
            let nextHC: Qwen4ExpGatedResidual? = i + 1 < layers.count ? layers[i + 1].attnHC : mixer
            let (hh, nn) = layer.callChainedPacked(h, segments: segments, layerIndex: i, rope: rope, groups: groups,
                                                   normedIn: normedNext, next: nextHC, mropeSection: mropeSection)
            h = hh; normedNext = nn
        }
        let hidden = segments.split(h)
        let out = segments.split(mixer(h, normed: normedNext).0)
        return zip(out, hidden).map { ($0, $1) }
    }
}

extension Qwen4ExpModel {
    /// T-0180: N text prompts' chunks in one forward. Per segment it returns what `forwardHidden(inputs[i], cache:)`
    /// returns for a prefill chunk: the LAST-row logits [1, 1, V] (the head on the last row, as a long chunk's
    /// `lastLogitsOnly` does) and the pre-mixer stream [1, S_i, hc*d].
    public func forwardPacked(_ inputs: [MLXArray], caches: [[KVCache]]) -> [(logits: MLXArray, hidden: MLXArray)] {
        // T-0186: through the general packed forward (segment-aware canonical blocking); the last row's logits as before
        forwardPackedGroups(chunks: zip(inputs, caches).map { Qwen4ExpPackedChunk(tokens: $0, cache: $1) }).chunks.map { r in
            let s = r.logits.dim(1)
            return (s > 1 ? r.logits[0..., (s - 1)..., 0...] : r.logits, r.hidden)
        }
    }
}
