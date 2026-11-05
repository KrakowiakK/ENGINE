import Foundation
import MLX
import MLXLMCommon
import EngineServeSupport
import Qwen4Exp

// P099 -- batched MTP: the draft / verify / accept / roll-back cycle for B rows at once, on the batch pool's stacked
// caches. The draft chain runs the MTP head over the B rows with its own stacked cache (ragged, one length per row);
// the verify block [B, K+1] goes through the trunk on the ragged S > 1 path (the serial verify block's arithmetic per
// row, U1); acceptance per row is `serveSpecRound`'s rule (teacher-forced equality, greedy or the row's own draw);
// the roll-back is per row (`rollbackRows`), the head is re-primed on every row's accepted rows in one padded call.

/// A row's draft state while it is a batch member: the head's cache at `headLen` positions, the first draft for the
/// next round, and the head's stream at the last committed position.
struct BatchRowSpec {
    var mtpCache: KVCache
    var headLen: Int
    var d1: Int
    var Slast: MLXArray            // (1, 1, hc*d)
}

/// The group's draft state: the rows' head caches stacked, one length per row.
final class BatchSpecPool {
    var headCache: KVCache
    var headLens: [Int]
    /// the rows' mRoPE shifts (0 for text rows), in pool order; set by the batch owner when it stacks the pool
    var ropeDeltas: [Int]? = nil
    /// P106 H56: the next round's first drafts stay on the device (`d1Dev`) when a round ends, so the owner does not wait for
    /// the re-prime: the GPU runs it while the host commits the round, dispatches the next step and builds its draft chain
    /// (Metal trace, H53: 6.8 % GPU idle in batched decode, ~2.5 ms/round of it this wait). Host readers (`d1`: unstack, the
    /// keep-alive, diagnostics) materialise it on first use. ENGINE_ROUND_D1_DEVICE=0 restores the per-round host read.
    private var d1Host: [Int]
    var d1Dev: MLXArray? = nil
    var d1: [Int] {
        get { if let d = d1Dev { d1Host = d.asArray(Int32.self).map { Int($0) }; d1Dev = nil }; return d1Host }
        set { d1Host = newValue; d1Dev = nil }
    }
    /// The first drafts as a device array (int32, (B)), without a host round trip when they are still on the device.
    func d1Array() -> MLXArray { d1Dev ?? MLXArray(d1Host.map { Int32($0) }) }
    var Slast: MLXArray            // (B, 1, hc*d)
    var p1: MLXArray? = nil        // H53 diagnostic: the head probability of each row's d1 (ENGINE_ROUND_TRACE_PROBS only)
    /// P106 H60: the next round's draft chain, queued on the GPU right behind this round's re-prime so the GPU drafts while
    /// the host commits the round and schedules the next step. `pooledAfter` is the head's pooled watermark the chain left;
    /// the host state is restored to the round's end, so every other reader (keep-alive, dissolve, captures) sees exactly
    /// the state it saw before H60, and the round that takes the chain reinstates the watermark it would have produced.
    /// Only positions past the rows' committed lengths are written early (don't-care rows, as a rejected draft's).
    /// ENGINE_ROUND_PREQUEUE_DRAFTS=0 restores drafting at round entry.
    var prequeued: (K: Int, drafts: [MLXArray], pooledAfter: [Int])? = nil
    init(headCache: KVCache, headLens: [Int], d1: [Int], Slast: MLXArray) {
        self.headCache = headCache; self.headLens = headLens; self.d1Host = d1; self.Slast = Slast
    }
}

/// P106 H53 (diagnostic): ENGINE_ROUND_TRACE=<path> appends one JSON line per `runBatchedStep` call (host stamps, ns
/// on the uptime clock) and, inside a batched MTP round, the points where the host queues or waits on GPU work, so the
/// GPU-idle host intervals between and inside rounds can be summed. ENGINE_ROUND_TRACE_PROBS=1 adds each draft's head
/// probability and each row's accepted count (read after the round's last stamp). Unset: no file, no graph change.
final class RoundTrace {
    let fh: FileHandle?
    let probs: Bool
    var cur: [String: Any] = [:]
    init() {
        let env = ProcessInfo.processInfo.environment
        if let path = env["ENGINE_ROUND_TRACE"], !path.isEmpty {
            FileManager.default.createFile(atPath: path, contents: nil)
            fh = FileHandle(forWritingAtPath: path)
        } else { fh = nil }
        probs = fh != nil && env["ENGINE_ROUND_TRACE_PROBS"] == "1"
    }
    var enabled: Bool { fh != nil }
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    func stamp(_ k: String) { if fh != nil { cur[k] = RoundTrace.now() } }
    func set(_ k: String, _ v: Any) { if fh != nil { cur[k] = v } }
    func flush() {
        guard let fh, !cur.isEmpty, let d = try? JSONSerialization.data(withJSONObject: cur) else { return }
        fh.write(d); fh.write("\n".data(using: .utf8)!); cur = [:]
    }
}
nonisolated(unsafe) let roundTraceShared = RoundTrace()
let roundD1Device: Bool = ProcessInfo.processInfo.environment["ENGINE_ROUND_D1_DEVICE"] != "0"
let roundPrequeueDrafts: Bool = ProcessInfo.processInfo.environment["ENGINE_ROUND_PREQUEUE_DRAFTS"] != "0"
/// T-0151 (lab instrument, batchprobe): ENGINE_ROLLBACK_TIMING=1 synchronises the GPU before the trunk roll-back and
/// again after it, and records the host build time and the synchronised wall time of the roll-back alone, per round.
/// =null keeps both synchronisations but builds no roll-back (the sync overhead; the rounds' state is then wrong).
let rollbackTimingMode: String? = ProcessInfo.processInfo.environment["ENGINE_ROLLBACK_TIMING"].flatMap { $0.isEmpty ? nil : $0 }   // "" = off (W10 first try)
nonisolated(unsafe) var rollbackTimingBuildMs: [Double] = []
nonisolated(unsafe) var rollbackTimingWallMs: [Double] = []
nonisolated(unsafe) var rollbackTimingPartialRows: [Int] = []
/// H60 activation witness: queued chains taken by the next round vs dropped (a plain step or a different K). The discard gate
/// requires both > 0 on its pre-queue arm, so it cannot PASS without exercising the paths it vouches for. Model thread only.
nonisolated(unsafe) var roundPrequeueTaken = 0, roundPrequeueDiscarded = 0

/// The K - 1 chained head steps of a round, ragged over the rows' head lengths: [d1, d2, ..., dK], each (B) int32.
func draftChain(_ qm: Qwen4ExpModel, spec: BatchSpecPool, K: Int) -> [MLXArray] {
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    var draftArrs: [MLXArray] = [spec.d1Array()]
    var Slast = spec.Slast
    var headRows = spec.headLens
    for _ in 1 ..< K {
        let (m, s2) = Qwen4ExpBatch.with(headRows, deltas: spec.ropeDeltas) { mtp(hidden: Slast, tokens: draftArrs.last![0..., .newAxis], embed: embed, cache: spec.headCache) }
        draftArrs.append(qm.draftToken(m[0..., -1, 0...]))
        Slast = s2
        headRows = headRows.map { $0 + 1 }
    }
    return draftArrs
}

func stackSpecPool(_ qm: Qwen4ExpModel, _ rows: [BatchRowSpec]) -> BatchSpecPool {
    let pool = BatchSpecPool(headCache: qm.stackHeadCaches(rows.map { $0.mtpCache }), headLens: rows.map { $0.headLen },
                             d1: rows.map { $0.d1 }, Slast: concatenated(rows.map { $0.Slast }, axis: 0))
    qm.resetHeadRaggedState()
    return pool
}

/// P106 H48: the draft state pooled row-resident -- the heads' caches stay the rows' own (a registered
/// marker stands in the pool), only Slast/d1/lengths are gathered. `BatchPool.dissolve` is the inverse.
func stackSpecPoolRowResident(_ qm: Qwen4ExpModel, _ rows: [BatchRowSpec]) -> BatchSpecPool {
    let pool = BatchSpecPool(headCache: qm.stackHeadCachesRowResident(rows.map { $0.mtpCache }), headLens: rows.map { $0.headLen },
                             d1: rows.map { $0.d1 }, Slast: concatenated(rows.map { $0.Slast }, axis: 0))
    qm.resetHeadRaggedState()
    return pool
}

func unstackSpecPool(_ qm: Qwen4ExpModel, _ pool: BatchSpecPool, slot: Int, ratio: Int) -> BatchRowSpec {
    let n = pool.headLens[slot]
    return BatchRowSpec(mtpCache: qm.unstackHeadCache(pool.headCache, slot: slot, length: n, pooledBlocks: n / ratio),
                        headLen: n, d1: pool.d1[slot], Slast: pool.Slast[slot ..< (slot + 1)])
}

/// Prime a row's draft state after its prompt: the trunk's hidden stream over the prompt through the head with the
/// tokens shifted by one, the last one being the greedy next token (the batchprobe harness; the server primes chunk
/// by chunk in its own loop). Returns the pending token n0 and the spec.
func primeRowSpec(_ qm: Qwen4ExpModel, ids: [Int], cache: [KVCache], chunk: Int) -> (n0: Int, spec: BatchRowSpec) {
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let mtpCache = mtp.newCache()
    let x = MLXArray(ids.map { Int32($0) })[.newAxis]
    var c0 = 0; var n0 = 0; var mixed = MLXArray(0), S = MLXArray(0)
    while c0 < ids.count {
        let c1 = min(ids.count, c0 + max(1, chunk))
        let (lg, h) = qm.forwardHidden(x[0..., c0 ..< c1], cache: cache)
        var toks = Array(ids[(c0 + 1) ..< min(c1 + 1, ids.count)])
        if c1 == ids.count { n0 = lg[0..., -1, 0...].argMax(axis: -1).item(Int.self); toks.append(n0) }
        (mixed, S) = mtp(hidden: h, tokens: MLXArray(toks.map { Int32($0) })[.newAxis], embed: embed, cache: mtpCache)
        if c1 < ids.count { asyncEval(S) }
        c0 = c1
    }
    let d1 = qm.draftToken(mixed[0..., -1, 0...])
    let Slast = S[0..., (S.dim(1) - 1)..., 0...]
    eval(d1, Slast, cache.flatMap { $0.state }, mtpCache.state)
    return (n0, BatchRowSpec(mtpCache: mtpCache, headLen: ids.count, d1: d1.item(Int.self), Slast: Slast))
}

/// Per-row inputs of a round beyond the token stream.
struct BatchRoundRow {
    var ropeDelta: Int = 0             // the row's mRoPE shift (image requests); 0 for text
    var greedy: Bool = true
    var bias: Float = 0                // the row's soft-stop bias at this position (applied to every verify row)
    var biasGated = false              // T-0045: the bias acts only on verify rows whose input token ends a line / paragraph
    var thinkOpen: Bool = false
    var stopIds: Set<Int> = []
    var drawKey: (() -> MLXArray?)? = nil
}

/// One batched round. `pending[b]` is row b's token not yet in the cache, `lengths[b]` its committed length.
/// Returns, per row, the committed tokens (accepted drafts + the bonus) and the accepted count; the caches hold
/// `lengths[b] + accepted[b] + 1` tokens per row afterwards and `spec` is re-primed.
/// T-0144: set by Serve's mixed step for exactly one batched MTP round (model thread only): the round's trunk verify forward
/// (block [B, K+1], the pool caches, row lengths, rope shifts) -> (logits, pre-mixer stream). Consumed (reset) by the round;
/// a step that runs no MTP round leaves it set and the caller clears it.
nonisolated(unsafe) var roundForwardOverride: ((MLXArray, [KVCache], [Int], [Int]) -> (MLXArray, MLXArray))? = nil

/// T-0247 (D-0088, design C-0723): copy-mode lookup drafts inside a batched round. `K` is the round's verify width (the
/// block holds K drafts per row), `realK[b]` row b's real drafts, `lookups[b]` row b's lookup drafts (nil: its MTP drafts,
/// the last one repeated past the MTP depth). Built by the caller from planCopyBlock; nil = the plain MTP round.
struct BatchCopyRound {
    let K: Int
    let realK: [Int]
    let lookups: [[Int]?]
    /// T-0247 (C-0720): verify through the packed per-row-depth forward (forwardPacked): row b runs realK[b] + 1 real
    /// positions, pads cost ~0.4 ms instead of a full position (C-0443). GDN, PLE and QSA still see the padded block,
    /// so the roll-back is unchanged (C-0723).
    var packed: Bool = false
}

/// T-0265 (C-0680): ENGINE_BATCH_MTP_DEPTH_CUM=<p> -- a third MTP draft only for rows whose head is confident: the round
/// drafts 3, reads each row's p1*p2*p3 on the host (no prequeue: the read needs the chain) and verifies through the packed
/// per-row-depth forward with depth 3 where the product >= p, else 2. Greedy rows without a bias, outside copy and mixed rounds.
let batchDepthCum: Double? = ProcessInfo.processInfo.environment["ENGINE_BATCH_MTP_DEPTH_CUM"].flatMap(Double.init)
nonisolated(unsafe) var batchDepthRounds = 0, batchDepthRows = 0, batchDepthRows3 = 0

func runBatchedMTPRound(_ qm: Qwen4ExpModel, caches: [KVCache], spec: BatchSpecPool, pending: [Int], lengths: [Int],
                        K: Int, rows: [BatchRoundRow], samplerParams: (temp: Float, topK: Int, topP: Float)? = nil,
                        copy copyIn: BatchCopyRound? = nil)
    -> [(tokens: [Int], accepted: Int)] {
    var copy = copyIn
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let B = pending.count
    let tr = roundTraceShared
    var probArrs: [MLXArray] = []
    if tr.probs || batchDepthCum != nil { qm.draftProbSink = { probArrs.append($0) } }
    defer { qm.draftProbSink = nil }
    tr.stamp("r_start")
    // drafts: K-1 chained head steps, ragged over the rows' head lengths (H60: already queued by the previous round)
    let draftArrs: [MLXArray]
    if let pq = spec.prequeued, pq.K == K, pq.drafts.count == K, pq.pooledAfter.count == B {
        draftArrs = pq.drafts
        qm.headRaggedPooled = pq.pooledAfter
        roundPrequeueTaken += 1
    } else {
        if spec.prequeued != nil { roundPrequeueDiscarded += 1 }
        draftArrs = draftChain(qm, spec: spec, K: K)
        asyncEval(draftArrs.last!)
    }
    spec.prequeued = nil
    tr.stamp("r_drafts_queued")
    // T-0265: per-row depth from the head's confidence (d1's from the last re-prime in spec.p1, d2/d3 from this chain)
    if copy == nil, K == 3, let thr = batchDepthCum, roundForwardOverride == nil,
       rows.allSatisfy({ $0.greedy && $0.bias <= 0 }), probArrs.count >= 2 {
        var cum = probArrs[probArrs.count - 2].reshaped(-1) * probArrs[probArrs.count - 1].reshaped(-1)
        if let p1 = spec.p1, p1.dim(0) == B { cum = cum * p1.reshaped(-1) }
        let c = cum.asType(.float32).asArray(Float.self)
        let realK = c.map { Double($0) >= thr ? 3 : 2 }
        copy = BatchCopyRound(K: realK.max()!, realK: realK, lookups: Array(repeating: nil, count: B), packed: true)
        batchDepthRounds += 1; batchDepthRows += B; batchDepthRows3 += realK.filter { $0 == 3 }.count
    }
    // the verify block [B, K+1]: pending + drafts per row
    // T-0247: with copy drafts the block is KB+1 wide; MTP rows repeat their last draft past K (never accepted: realK cap)
    let KB = copy?.K ?? K
    var blockCols: [MLXArray] = draftArrs.map { $0[0..., .newAxis] }
    if let copy {
        let mask = MLXArray(copy.lookups.map { $0 != nil }).reshaped(B, 1)
        blockCols = (0 ..< KB).map { t in
            let m = draftArrs[min(t, K - 1)][0..., .newAxis].asType(.int32)
            let lk = MLXArray(copy.lookups.map { l -> Int32 in
                guard let l, !l.isEmpty else { return 0 }; return Int32(l[min(t, l.count - 1)]) }).reshaped(B, 1)
            return which(mask, lk, m)
        }
    }
    let block = concatenated([MLXArray(pending.map { Int32($0) })[0..., .newAxis]] + blockCols, axis: 1)
    // T-0156: ENGINE_NAN_WITNESS=1 (diagnostic): which row state already holds NaN BEFORE this round's forward
    let nanBefore: [String]? = nanWitnessOn ? nanWitnessState(qm, caches: caches, lengths: lengths) : nil
    // T-0144: a mixed step replaces this forward once with one over [prefill chunk ; this block] (Serve's mixedStep)
    let biases = rows.map { $0.bias }
    // T-0247 (C-0720): the packed copy round -- greedy rows without a think-close bias, outside a mixed step; its
    // predictions are taken on the P real positions and spread back to [B, KB+1] (a pad repeats its row's last real
    // position, which the realK cap never reads past)
    let packedLayout: Qwen4ExpPackedRows? = {
        guard let copy, copy.packed, roundForwardOverride == nil, rows.allSatisfy({ $0.greedy }),
              !biases.contains(where: { $0 > 0 }) else { return nil }
        return Qwen4ExpPackedRows(depths: copy.realK.map { $0 + 1 }, positions: KB + 1)
    }()
    let (vl, vh): (MLXArray, MLXArray)
    var packedPreds: MLXArray? = nil
    roundForwardOverrideWasSet = roundForwardOverride != nil
    if let mixed = roundForwardOverride {
        roundForwardOverride = nil
        (vl, vh) = mixed(block, caches, lengths, rows.map { $0.ropeDelta })
    } else if let layout = packedLayout {
        let (pl, ph) = qm.forwardPacked(block, cache: caches, rows: lengths, deltas: rows.map { $0.ropeDelta }, packed: layout)
        let spread = MLXArray(layout.spread)
        packedPreds = pl.reshaped(layout.total, -1).argMax(axis: -1).asType(.int32).take(spread, axis: 0).reshaped(B, KB + 1)
        vh = ph.reshaped(layout.total, -1).take(spread, axis: 0).reshaped(B, KB + 1, -1)
        vl = pl
        batchCopyPackedRounds += 1
    } else {
        (vl, vh) = Qwen4ExpBatch.with(lengths, deltas: rows.map { $0.ropeDelta }) { qm.forwardHidden(block, cache: caches) }   // (B,K+1,V), (B,K+1,hc*d)
    }
    var l = vl
    if let nanBefore, packedPreds == nil { nanWitnessRound(vl, before: nanBefore, lengths: lengths, pending: pending, mixed: roundForwardOverrideWasSet) }
    if packedPreds == nil, biases.contains(where: { $0 > 0 }) {
        var add = MLXArray(biases).reshaped([B, 1])
        if rows.contains(where: { $0.biasGated }), let mask = engineBoundaryMask {
            let g = MLXArray(rows.map { $0.biasGated ? Float(1) : 0 }).reshaped([B, 1])
            add = add * (g * mask.take(block) + (1 - g))                  // (B, K+1)
        }
        let col = l[0..., 0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
        l[0..., 0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + add.reshaped([B, -1, 1]).asType(l.dtype)
    }
    let V = l.dim(-1)
    let preds: MLXArray
    if let packedPreds {
        preds = packedPreds
    } else if rows.allSatisfy({ $0.greedy }) {
        preds = l.argMax(axis: -1).asType(.int32)                                           // (B, K+1)
    } else if let p = samplerParams, let t = sampleTopKBlock(l.reshaped(B * (KB + 1), V), temp: p.temp, topK: p.topK, topP: p.topP,
                                                             draw: { i in rows[i / (KB + 1)].drawKey?() }) {
        preds = t.reshaped(B, KB + 1).asType(.int32)
        EngineWitness.hit(p.topK > 0 && p.topK < V ? "sampled_rows_topk" : "sampled_rows_fullvocab", B * (KB + 1))   // T-0191
    } else {
        preds = l.argMax(axis: -1).asType(.int32)
    }
    let draftsCat = concatenated(blockCols, axis: 1)                                         // (B, KB)
    let bothArr = concatenated([preds.reshaped(-1), draftsCat.reshaped(-1)], axis: 0)
    tr.stamp("r_verify_built")
    let both = bothArr.asArray(Int32.self).map { Int($0) }   // ONE host read
    tr.stamp("r_verify_read")
    var out: [(tokens: [Int], accepted: Int)] = []
    var keep: [Int] = []
    for b in 0 ..< B {
        let pr = Array(both[(b * (KB + 1)) ..< ((b + 1) * (KB + 1))])
        let dr = Array(both[(B * (KB + 1) + b * KB) ..< (B * (KB + 1) + (b + 1) * KB)])
        let realK = copy?.realK[b] ?? K
        var a = 0
        while a < realK && pr[a] == dr[a] { a += 1 }
        if rows[b].thinkOpen, rows[b].bias > 0, let ci = (0 ..< a).first(where: { dr[$0] == engineThinkCloseId }) { a = ci }
        if let si = (0 ..< a).first(where: { rows[b].stopIds.contains(dr[$0]) }) { a = si }
        out.append((Array(dr.prefix(a)) + [pr[a]], a))
        keep.append(a + 1)
    }
    // roll back per row, then re-prime the head on every row's accepted rows in one padded call. H56: the head re-prime
    // does not read the trunk caches, so with ENGINE_ROUND_D1_DEVICE (default) it is queued FIRST and the trunk roll-back
    // graph is built while the GPU runs it (the same operations, only their construction order changes).
    if !roundD1Device { qm.rollbackRows(caches, blockRows: KB + 1, keep: keep, lengths: lengths) }   // clears the tapes too
    let Sp = keep.max()!
    var tokPad: [Int32] = []
    for b in 0 ..< B { let t = out[b].tokens; tokPad += (t + Array(repeating: t.last!, count: Sp - t.count)).map { Int32($0) } }
    let (m2, s3) = Qwen4ExpBatch.with(spec.headLens, deltas: spec.ropeDeltas) {
        mtp(hidden: vh[0..., 0 ..< Sp, 0...], tokens: MLXArray(tokPad).reshaped(B, Sp), embed: embed, cache: spec.headCache)
    }
    let pick = MLXArray((0 ..< B).map { Int32(keep[$0] - 1) })                                   // each row's last real re-prime row
    let mLast = takeAlong(m2, expandedDimensions(pick.reshaped(B, 1), axis: -1), axis: 1)[0..., 0, 0...]     // (B, d)
    let sLast = takeAlong(s3, expandedDimensions(pick.reshaped(B, 1), axis: -1), axis: 1)         // (B, 1, hc*d)
    let draftProbCount = probArrs.count
    let d1 = qm.draftToken(mLast)
    asyncEval(d1, sLast)
    tr.stamp("r_reprime_queued")
    if let mode = rollbackTimingMode, roundD1Device {
        let gdn = caches.compactMap { $0 as? ArraysCache }
        eval([d1, sLast] + gdn.flatMap { $0.state })                      // GPU idle: everything queued so far is done
        let t0 = DispatchTime.now().uptimeNanoseconds
        if mode != "null" { qm.rollbackRows(caches, blockRows: KB + 1, keep: keep, lengths: lengths) }
        let t1 = DispatchTime.now().uptimeNanoseconds
        eval(gdn.flatMap { $0.state })
        let t2 = DispatchTime.now().uptimeNanoseconds
        rollbackTimingBuildMs.append(Double(t1 - t0) / 1e6); rollbackTimingWallMs.append(Double(t2 - t0) / 1e6)
        rollbackTimingPartialRows.append(keep.filter { $0 < KB + 1 }.count)
    } else if roundD1Device { qm.rollbackRows(caches, blockRows: KB + 1, keep: keep, lengths: lengths) }
    if roundD1Device { spec.d1Dev = d1.dtype == .int32 ? d1 : d1.asType(.int32) } else { spec.d1 = d1.asArray(Int32.self).map { Int($0) } }
    tr.stamp("r_reprime_read")
    spec.Slast = sLast
    spec.headLens = (0 ..< B).map { spec.headLens[$0] + keep[$0] }
    qm.rewindHeadRows(lengths: spec.headLens)
    if batchDepthCum != nil, !tr.probs { spec.p1 = probArrs.count > draftProbCount ? probArrs[draftProbCount].reshaped(-1) : nil }
    if roundPrequeueDrafts, roundD1Device, K > 1, !tr.probs, batchDepthCum == nil {
        let pooled = qm.headRaggedPooled
        let next = draftChain(qm, spec: spec, K: K)
        asyncEval(next.last!)
        spec.prequeued = (K, next, qm.headRaggedPooled)
        qm.headRaggedPooled = pooled
    }
    tr.stamp("r_end")
    if nanWitnessOn {   // T-0156: the state AFTER this round's roll-back (dirty after = the round made it; clean after but dirty
                        // before the next round = something between rounds)
        let after = nanWitnessState(qm, caches: caches, lengths: (0 ..< B).map { lengths[$0] + keep[$0] })
        for b in 0 ..< B where !after[b].isEmpty && after[b].split(separator: ",").count < 20 && nanWitnessLines < 400 {
            nanWitnessLines += 1
            FileHandle.standardError.write(("engine nan witness: AFTER round \(nanWitnessRounds) B=\(B) row \(b) length \(lengths[b] + keep[b]) "
                + "keep \(keep[b]) state_bad_after=[\(after[b].prefix(600))] lengths=\(lengths)\n").data(using: .utf8)!)
        }
    }
    if tr.enabled {
        tr.set("B", B); tr.set("K", K); tr.set("acc", out.map { $0.accepted }); tr.set("greedy", rows.allSatisfy { $0.greedy })
        tr.set("lengths", lengths)
        if tr.probs {
            // position 1 came from the previous re-prime / keep-alive (spec.p1); positions 2..K from this round's chain
            var p: [[Float]] = Array(repeating: [], count: B)
            if let p1 = spec.p1, p1.dim(0) == B { let v = p1.asArray(Float.self); for b in 0 ..< B { p[b].append(v[b]) } }
            else { for b in 0 ..< B { p[b].append(-1) } }
            for a in probArrs.prefix(draftProbCount) { let v = a.reshaped(-1).asArray(Float.self); for b in 0 ..< B { p[b].append(v[b]) } }
            tr.set("p", p.map { $0.map { Double($0) } })
            spec.p1 = probArrs.count > draftProbCount ? probArrs[draftProbCount].reshaped(-1) : nil
        }
    }
    return out
}

/// A plain batched step that keeps the rows' draft state alive: the trunk forward, the caller's sampling, then one
/// head step on (hidden, next token) per row so the policy can draft again next round.
/// P099: a lone step's keep-alive for ONE row's draft state (its own head cache, serial arithmetic): without it a row that
/// steps alone once (a fragmented group) has a head one position behind its trunk and can never join a round again.
func rowHeadKeepAlive(_ qm: Qwen4ExpModel, spec: inout BatchRowSpec, hidden: MLXArray, next: Int) {
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let (m, s) = mtp(hidden: hidden, tokens: MLXArray([Int32(next)]).reshaped(1, 1), embed: embed, cache: spec.mtpCache)
    let d1 = qm.draftToken(m[0..., -1, 0...])
    eval(d1, s)
    spec.d1 = d1.item(Int.self); spec.Slast = s; spec.headLen += 1
}

func batchedHeadKeepAlive(_ qm: Qwen4ExpModel, spec: BatchSpecPool, hidden: MLXArray, next: [Int]) {
    var probArrs: [MLXArray] = []
    if roundTraceShared.probs || batchDepthCum != nil { qm.draftProbSink = { probArrs.append($0) } }
    defer { qm.draftProbSink = nil; spec.p1 = probArrs.first?.reshaped(-1) }
    if spec.prequeued != nil { roundPrequeueDiscarded += 1 }
    spec.prequeued = nil            // H60: a plain step instead of the round the chain was queued for (host state is the round's end)
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let B = next.count
    let (m, s) = Qwen4ExpBatch.with(spec.headLens, deltas: spec.ropeDeltas) { mtp(hidden: hidden, tokens: MLXArray(next.map { Int32($0) }).reshaped(B, 1), embed: embed, cache: spec.headCache) }
    let d1 = qm.draftToken(m[0..., -1, 0...])
    asyncEval(d1, s)
    // H60: the keep-alive's first drafts stay on the device too (H56's rule); host readers materialise them lazily
    if roundD1Device { spec.d1Dev = d1.dtype == .int32 ? d1 : d1.asType(.int32) } else { spec.d1 = d1.asArray(Int32.self).map { Int($0) } }
    spec.Slast = s
    spec.headLens = spec.headLens.map { $0 + 1 }
}
let q4GDNSlotsEnv: Bool = ProcessInfo.processInfo.environment["ENGINE_GDN_SLOTS"] == "1"


// MARK: - T-0156 NaN witness (diagnostic, ENGINE_NAN_WITNESS=1; off by default, never in a production launcher)
/// A short-rung E9 server answered "li" + 1398 '!' for one resumed request under concurrency (C-0486): NaN logits from the
/// second token. The witness reports, per batched round, every row whose logits hold NaN, and for each such row the
/// (layer, slot) cache arrays that ALREADY held NaN before the round's forward -- corrupted state in -- or none: the
/// round's own forward made it. Only an incident's first rounds are reported (a row bad in >= 20 arrays has propagated), at most 400 lines.
let nanWitnessOn = ProcessInfo.processInfo.environment["ENGINE_NAN_WITNESS"] == "1"
nonisolated(unsafe) var roundForwardOverrideWasSet = false
nonisolated(unsafe) private var nanWitnessLines = 0
nonisolated(unsafe) private var nanWitnessRounds = 0

/// Per row: the names of the cache arrays (layer.slot) holding NaN or |x| > 1e5 in that row's valid range (empty: clean).
func nanWitnessState(_ qm: Qwen4ExpModel, caches: [KVCache], lengths: [Int]) -> [String] {
    let B = lengths.count
    var names: [String] = []
    var flags: [MLXArray] = []
    func add(_ name: String, _ x: MLXArray?) {
        guard let x, [DType.float32, .float16, .bfloat16].contains(x.dtype) else { return }
        let f = x.asType(.float32)   // NaN, or a magnitude no healthy state reaches (garbage memory reads as huge finite values too)
        names.append(name); flags.append((isNaN(f) .|| (abs(f) .> Float(1e5))).any().reshaped([1]))
    }
    for b in 0 ..< B {
        let view: [KVCache?] = caches.map { c in
            if let rows = Qwen4ExpModel.rowResidentRows(c) { return rows[b] }
            if B == 1, c is CacheList { return c }   // a single row's own caches (prefill end, pool join)
            if let a = c as? ArraysCache {
                let o = ArraysCache(size: 6)
                for sl in 0 ..< 4 { if let x = a[sl], x.dim(0) > b { o[sl] = x[b ..< (b + 1)] } }
                return o
            }
            return nil
        }
        for (i, c) in view.enumerated() {
            if let l = c as? CacheList, let kv = l[0] as? KVCacheSimple {
                let n = min(lengths[b], kv.rawKeys?.dim(2) ?? 0)
                add("b\(b).L\(i).k", kv.rawKeys.map { $0[0..., 0..., 0 ..< n, 0...] })
                add("b\(b).L\(i).v", kv.rawValues.map { $0[0..., 0..., 0 ..< n, 0...] })
                let idx = l[1] as! ArraysCache
                if let i0 = idx[0] { add("b\(b).L\(i).i0", i0[0..., 0 ..< min(n, i0.dim(1)), 0...]) }
                add("b\(b).L\(i).i1", idx[1])
            } else if let a = c as? ArraysCache {
                for sl in 0 ..< 4 { add("b\(b).L\(i).a\(sl)", a[sl]) }
            }
        }
    }
    guard !flags.isEmpty else { return Array(repeating: "", count: B) }
    let f = concatenated(flags, axis: 0).asArray(Bool.self)
    var per = Array(repeating: "", count: B)
    for (k, name) in names.enumerated() where f[k] {
        let b = Int(name.dropFirst().prefix { $0 != "." })!
        per[b] += (per[b].isEmpty ? "" : ",") + String(name.drop { $0 != "." }.dropFirst())
    }
    return per
}

func nanWitnessRound(_ logits: MLXArray, before: [String], lengths: [Int], pending: [Int], mixed: Bool) {
    nanWitnessRounds += 1
    let B = lengths.count
    let rowNaN = isNaN(logits.asType(.float32)).any(axes: [1, 2]).asArray(Bool.self)
    // only the FIRST rounds of an incident: a row whose state is already bad in >= 20 arrays has fully propagated
    for b in 0 ..< B where (rowNaN[b] || !before[b].isEmpty) && before[b].split(separator: ",").count < 20 && nanWitnessLines < 400 {
        nanWitnessLines += 1
        FileHandle.standardError.write(("engine nan witness: round \(nanWitnessRounds) B=\(B) mixed=\(mixed) row \(b) length \(lengths[b]) "
            + "pending \(pending[b]) logits_nan=\(rowNaN[b]) state_nan_before=[\(before[b].prefix(600))] "
            + "lengths=\(lengths)\n").data(using: .utf8)!)
    }
}

/// T-0156: one row's own caches at a named point (prefill end, pool join); logs only when something is bad.
func nanWitnessPoint(_ qm: Qwen4ExpModel, _ point: String, caches: [KVCache], length: Int, note: String) {
    let bad = nanWitnessState(qm, caches: caches, lengths: [length])[0]
    if !bad.isEmpty, bad.split(separator: ",").count < 20, nanWitnessLines < 400 {   // a propagated row is not news
        nanWitnessLines += 1
        FileHandle.standardError.write("engine nan witness: \(point) length \(length) \(note) state_bad=[\(bad.prefix(600))]\n".data(using: .utf8)!)
    }
}
