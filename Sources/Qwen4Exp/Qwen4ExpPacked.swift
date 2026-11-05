// T-0186: several prompts' prefill chunks -- and optionally a batched verify block -- in ONE forward. The general form of
// forwardMixed (one chunk + verify, Qwen4ExpMixed.swift) and of T-0180's text-only forwardPacked: the row-wise work
// (hyper-connections, router, routed and shared experts, combines, final mixer) runs once over the concatenated rows, so
// every segment shares one read of the expert weights; PLE, the mixer (GDN / QSA attention), the mask and the n-gram
// history run per group on their own caches through the ordinary path, exactly as forwardMixed's chunk and verify groups.
// Row layout: chunk 0's rows, chunk 1's rows, ..., then the verify block's B x S rows (row-major), as [1, total, .].
//
// Canonical width (C-0582): with ENGINE_PREFILL_WIDTH_CANONICAL = R the skinny projections of a B1 forward run in R-row
// blocks from the chunk's first row. Over a packed block those blocks would straddle segment boundaries and change the
// rounding of every segment after the first; while this forward runs, `Qwen4ExpPrefillWidth.segments` makes them block
// each segment from its own first row, as its solo forward does.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// One prefill chunk of a packed forward: its tokens [1, S], its sequence's caches, and an image row's embeddings / mRoPE
/// positions for this range (nil for text).
public struct Qwen4ExpPackedChunk {
    public let tokens: MLXArray
    public let cache: [KVCache]
    public let embeds: MLXArray?
    public let pos3: MLXArray?
    public init(tokens: MLXArray, cache: [KVCache], embeds: MLXArray? = nil, pos3: MLXArray? = nil) {
        precondition(tokens.dim(0) == 1 && tokens.dim(1) > 0, "a packed chunk is one sequence's [1, S] tokens")
        self.tokens = tokens; self.cache = cache; self.embeds = embeds; self.pos3 = pos3
    }
}

/// The batched verify block riding in a packed forward (a batch pool's ragged block, as forwardMixed with rows).
public struct Qwen4ExpPackedVerify {
    public let tokens: MLXArray
    public let cache: [KVCache]
    public let rows: [Int]
    public let deltas: [Int]?
    public init(tokens: MLXArray, cache: [KVCache], rows: [Int], deltas: [Int]?) {
        precondition(rows.count == tokens.dim(0), "a ragged verify group has one offset per row")
        self.tokens = tokens; self.cache = cache; self.rows = rows; self.deltas = deltas
    }
}

/// The row bookkeeping, testable without a model: chunk i owns rows [offset_i, offset_i + chunks[i]); the verify block
/// (batch x positions, may be 0 x 0) owns the tail.
public struct Qwen4ExpPackedLayout: Sendable, Equatable {
    public let chunks: [Int]
    public let batch: Int
    public let positions: Int
    public init(chunks: [Int], batch: Int = 0, positions: Int = 0) {
        precondition(!chunks.isEmpty && chunks.allSatisfy { $0 > 0 } && batch >= 0 && positions >= 0 && (batch == 0) == (positions == 0))
        self.chunks = chunks; self.batch = batch; self.positions = positions
    }
    public var verifyRows: Int { batch * positions }
    public var total: Int { chunks.reduce(0, +) + verifyRows }
    /// the row partition the canonical blocking follows (the verify block is one segment, as its separate call is one matmul)
    public var segments: [Int] { verifyRows > 0 ? chunks + [verifyRows] : chunks }
    public func split(_ a: MLXArray) -> (chunks: [MLXArray], verify: MLXArray?) {
        precondition(a.dim(0) == 1 && a.dim(1) == total, "packed block has \(a.shape), expected [1, \(total), F]")
        var o = 0
        let cs = chunks.map { n -> MLXArray in defer { o += n }; return a[0..., o ..< (o + n)] }
        return (cs, verifyRows > 0 ? a[0..., o...].reshaped([batch, positions, a.dim(-1)]) : nil)
    }
    public func join(_ cs: [MLXArray], _ verify: MLXArray?) -> MLXArray {
        precondition(cs.count == chunks.count && zip(cs, chunks).allSatisfy { $0.dim(0) == 1 && $0.dim(1) == $1 })
        precondition((verify == nil) == (verifyRows == 0))
        var parts = cs
        if let verify {
            precondition(verify.dim(0) == batch && verify.dim(1) == positions)
            parts.append(verify.reshaped([1, verifyRows, verify.dim(-1)]))
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }
}

extension Qwen4ExpDecoderLayer {
    func callChainedPackedGroups(_ h: MLXArray, layout: Qwen4ExpPackedLayout, layerIndex i: Int, rope: Qwen4ExpRotary,
                                 chunks: [Qwen4ExpMixedGroup], verify: Qwen4ExpMixedGroup?,
                                 normedIn: MLXArray?, next: Qwen4ExpGatedResidual?, mropeSection: [Int]) -> (MLXArray, MLXArray?) {
        precondition(inputNorm == nil, "the packed forward is for the hyper-connection model")
        var h = h
        var normedIn = normedIn
        if let ple, !Qwen4ExpEnv.noPLE {
            let (hs, hV) = layout.split(h)
            let pC = zip(hs, chunks).map { x, g in ple(x, ids: g.ids, prev: g.prev, cache: g.cache[i] as? ArraysCache) }
            let pV = hV.map { x in verify!.run { ple(x, ids: verify!.ids, prev: verify!.prev, cache: verify!.cache[i] as? ArraysCache) } }
            h = h + layout.join(pC, pV)
            normedIn = nil
        }
        let (x0, inj) = attnHC(h, normed: normedIn)
        let (xs, xV) = layout.split(x0)
        func mix(_ x: MLXArray, _ g: Qwen4ExpMixedGroup) -> MLXArray {
            if let linearAttn { return linearAttn(x, cache: g.cache[i] as? ArraysCache) }
            let l = g.cache[i] as? CacheList
            return selfAttn!(x, rope: rope, mask: g.mask, cache: l?[0], indexerCache: l?[1] as? ArraysCache, pos3: g.pos3,
                             mropeSection: mropeSection)
        }
        let ys = zip(xs, chunks).map { mix($0, $1) }
        let yV = xV.map { x in verify!.run { mix(x, verify!) } }
        let (h1, n1) = combineChained(h, layout.join(ys, yV), inj!, next: mlpHC)
        h = h1
        let (y, inj2) = mlpHC(h, normed: n1)
        let m = mlp(y)
        return combineChained(h, m, inj2!, next: next)
    }
}

extension Qwen4ExpModelInner {
    /// Final mixer output and pre-mixer stream per chunk ([1, S_i, d] / [1, S_i, hc*d]) and of the verify block
    /// ([B, S, d] / [B, S, hc*d]).
    func callPackedGroups(chunks: [Qwen4ExpPackedChunk], verify: Qwen4ExpPackedVerify?)
        -> (chunks: [(out: MLXArray, hidden: MLXArray)], verify: (out: MLXArray, hidden: MLXArray)?) {
        precondition(norm == nil, "the packed forward is for the hyper-connection model")
        let layout = Qwen4ExpPackedLayout(chunks: chunks.map { $0.tokens.dim(1) }, batch: verify?.tokens.dim(0) ?? 0,
                                          positions: verify?.tokens.dim(1) ?? 0)
        var embeds: [MLXArray] = []
        var groups: [Qwen4ExpMixedGroup] = []
        for c in chunks {
            let e = c.embeds ?? embedTokens(c.tokens)
            let n = packedNgram(c.tokens, c.cache)
            embeds.append(e)
            groups.append(Qwen4ExpMixedGroup(cache: c.cache, ids: n.ids, prev: n.prev,
                                             mask: createAttentionMask(h: e, cache: ((c.cache[faIdx] as? CacheList)?[0]).map { [$0] }),
                                             pos3: c.pos3))
        }
        var eV: MLXArray? = nil
        var gV: Qwen4ExpMixedGroup? = nil
        if let v = verify {
            let e = embedTokens(v.tokens)
            let n = packedNgram(v.tokens, v.cache)
            let g0 = Qwen4ExpMixedGroup(cache: v.cache, ids: n.ids, prev: n.prev, mask: .none, pos3: nil, rows: v.rows, deltas: v.deltas)
            let maskV = g0.run { createAttentionMask(h: e, cache: ((v.cache[faIdx] as? CacheList)?[0]).map { [$0] }) }
            eV = e
            gV = Qwen4ExpMixedGroup(cache: v.cache, ids: n.ids, prev: n.prev, mask: maskV, pos3: nil, rows: v.rows, deltas: v.deltas)
        }
        let saved = Qwen4ExpPrefillWidth.segments
        Qwen4ExpPrefillWidth.segments = layout.segments
        defer { Qwen4ExpPrefillWidth.segments = saved }
        var h = tiled(layout.join(embeds, eV), repetitions: [1, 1, hc])
        var normedNext: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            if i >= Qwen4ExpDecoderLayer.ablLayers { break }
            let nextHC: Qwen4ExpGatedResidual? = i + 1 < layers.count ? layers[i + 1].attnHC : mixer
            let (hh, nn) = layer.callChainedPackedGroups(h, layout: layout, layerIndex: i, rope: rope, chunks: groups, verify: gV,
                                                         normedIn: normedNext, next: nextHC, mropeSection: mropeSection)
            h = hh; normedNext = nn
        }
        let (hid, hidV) = layout.split(h)
        let (out, outV) = layout.split(mixer(h, normed: normedNext).0)
        return (zip(out, hid).map { ($0, $1) }, outV.map { ($0, hidV!) })
    }

    /// The n-gram inputs of one group, with the same cache update as the ordinary forward's preamble (= mixedNgram).
    private func packedNgram(_ inputs: MLXArray, _ caches: [KVCache]) -> (ids: MLXArray, prev: MLXArray) {
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
}

extension Qwen4ExpModel {
    /// T-0186: per chunk what `forwardHidden(chunk, cache:, embedsOverride:, pos3:)` returns for a prefill chunk (the
    /// head as forwardMixed's chunk: last row only when lastLogitsOnly and S > 16; an image chunk through lmHead), and for
    /// the verify block what forwardMixed returns for it.
    public func forwardPackedGroups(chunks: [Qwen4ExpPackedChunk], verify: Qwen4ExpPackedVerify? = nil)
        -> (chunks: [(logits: MLXArray, hidden: MLXArray)], verify: (logits: MLXArray, hidden: MLXArray)?) {
        let r = model.callPackedGroups(chunks: chunks, verify: verify)
        let cs = zip(r.chunks, chunks).map { o, c -> (logits: MLXArray, hidden: MLXArray) in
            let sc = o.out.dim(1)
            let last = Self.lastLogitsOnly && sc > 16 ? o.out[0..., (sc - 1)..., 0...] : o.out
            let vision = c.embeds != nil || c.pos3 != nil
            return (vision ? (lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)) : headLogits(last), o.hidden)
        }
        return (cs, r.verify.map { (headLogits($0.out), $0.hidden) })
    }
}
