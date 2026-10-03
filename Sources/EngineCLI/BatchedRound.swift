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
        let (m, s2) = Qwen4ExpBatch.with(headRows) { mtp(hidden: Slast, tokens: draftArrs.last![0..., .newAxis], embed: embed, cache: spec.headCache) }
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
func runBatchedMTPRound(_ qm: Qwen4ExpModel, caches: [KVCache], spec: BatchSpecPool, pending: [Int], lengths: [Int],
                        K: Int, rows: [BatchRoundRow], samplerParams: (temp: Float, topK: Int, topP: Float)? = nil)
    -> [(tokens: [Int], accepted: Int)] {
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let B = pending.count
    let tr = roundTraceShared
    var probArrs: [MLXArray] = []
    if tr.probs { qm.draftProbSink = { probArrs.append($0) } }
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
    // the verify block [B, K+1]: pending + drafts per row
    let block = concatenated([MLXArray(pending.map { Int32($0) })[0..., .newAxis]] + draftArrs.map { $0[0..., .newAxis] }, axis: 1)
    let (vl, vh) = Qwen4ExpBatch.with(lengths) { qm.forwardHidden(block, cache: caches) }          // (B,K+1,V), (B,K+1,hc*d)
    var l = vl
    let biases = rows.map { $0.bias }
    if biases.contains(where: { $0 > 0 }) {
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
    if rows.allSatisfy({ $0.greedy }) {
        preds = l.argMax(axis: -1).asType(.int32)                                           // (B, K+1)
    } else if let p = samplerParams, let t = sampleTopKBlock(l.reshaped(B * (K + 1), V), temp: p.temp, topK: p.topK, topP: p.topP,
                                                             draw: { i in rows[i / (K + 1)].drawKey?() }) {
        preds = t.reshaped(B, K + 1).asType(.int32)
    } else {
        preds = l.argMax(axis: -1).asType(.int32)
    }
    let draftsCat = concatenated(draftArrs.map { $0[0..., .newAxis] }, axis: 1)               // (B, K)
    let bothArr = concatenated([preds.reshaped(-1), draftsCat.reshaped(-1)], axis: 0)
    tr.stamp("r_verify_built")
    let both = bothArr.asArray(Int32.self).map { Int($0) }   // ONE host read
    tr.stamp("r_verify_read")
    var out: [(tokens: [Int], accepted: Int)] = []
    var keep: [Int] = []
    for b in 0 ..< B {
        let pr = Array(both[(b * (K + 1)) ..< ((b + 1) * (K + 1))])
        let dr = Array(both[(B * (K + 1) + b * K) ..< (B * (K + 1) + (b + 1) * K)])
        var a = 0
        while a < K && pr[a] == dr[a] { a += 1 }
        if rows[b].thinkOpen, rows[b].bias > 0, let ci = (0 ..< a).first(where: { dr[$0] == engineThinkCloseId }) { a = ci }
        if let si = (0 ..< a).first(where: { rows[b].stopIds.contains(dr[$0]) }) { a = si }
        out.append((Array(dr.prefix(a)) + [pr[a]], a))
        keep.append(a + 1)
    }
    // roll back per row, then re-prime the head on every row's accepted rows in one padded call. H56: the head re-prime
    // does not read the trunk caches, so with ENGINE_ROUND_D1_DEVICE (default) it is queued FIRST and the trunk roll-back
    // graph is built while the GPU runs it (the same operations, only their construction order changes).
    if !roundD1Device { qm.rollbackRows(caches, blockRows: K + 1, keep: keep, lengths: lengths) }   // clears the tapes too
    let Sp = keep.max()!
    var tokPad: [Int32] = []
    for b in 0 ..< B { let t = out[b].tokens; tokPad += (t + Array(repeating: t.last!, count: Sp - t.count)).map { Int32($0) } }
    let (m2, s3) = Qwen4ExpBatch.with(spec.headLens) {
        mtp(hidden: vh[0..., 0 ..< Sp, 0...], tokens: MLXArray(tokPad).reshaped(B, Sp), embed: embed, cache: spec.headCache)
    }
    let pick = MLXArray((0 ..< B).map { Int32(keep[$0] - 1) })                                   // each row's last real re-prime row
    let mLast = takeAlong(m2, expandedDimensions(pick.reshaped(B, 1), axis: -1), axis: 1)[0..., 0, 0...]     // (B, d)
    let sLast = takeAlong(s3, expandedDimensions(pick.reshaped(B, 1), axis: -1), axis: 1)         // (B, 1, hc*d)
    let draftProbCount = probArrs.count
    let d1 = qm.draftToken(mLast)
    asyncEval(d1, sLast)
    tr.stamp("r_reprime_queued")
    if roundD1Device { qm.rollbackRows(caches, blockRows: K + 1, keep: keep, lengths: lengths) }
    if roundD1Device { spec.d1Dev = d1.dtype == .int32 ? d1 : d1.asType(.int32) } else { spec.d1 = d1.asArray(Int32.self).map { Int($0) } }
    tr.stamp("r_reprime_read")
    spec.Slast = sLast
    spec.headLens = (0 ..< B).map { spec.headLens[$0] + keep[$0] }
    qm.rewindHeadRows(lengths: spec.headLens)
    if roundPrequeueDrafts, roundD1Device, K > 1, !tr.probs {
        let pooled = qm.headRaggedPooled
        let next = draftChain(qm, spec: spec, K: K)
        asyncEval(next.last!)
        spec.prequeued = (K, next, qm.headRaggedPooled)
        qm.headRaggedPooled = pooled
    }
    tr.stamp("r_end")
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
    if roundTraceShared.probs { qm.draftProbSink = { probArrs.append($0) } }
    defer { qm.draftProbSink = nil; spec.p1 = probArrs.first?.reshaped(-1) }
    if spec.prequeued != nil { roundPrequeueDiscarded += 1 }
    spec.prequeued = nil            // H60: a plain step instead of the round the chain was queued for (host state is the round's end)
    let mtp = qm.mtp!, embed = qm.model.embedTokens
    let B = next.count
    let (m, s) = Qwen4ExpBatch.with(spec.headLens) { mtp(hidden: hidden, tokens: MLXArray(next.map { Int32($0) }).reshaped(B, 1), embed: embed, cache: spec.headCache) }
    let d1 = qm.draftToken(m[0..., -1, 0...])
    asyncEval(d1, s)
    // H60: the keep-alive's first drafts stay on the device too (H56's rule); host readers materialise them lazily
    if roundD1Device { spec.d1Dev = d1.dtype == .int32 ? d1 : d1.asType(.int32) } else { spec.d1 = d1.asArray(Int32.self).map { Int($0) } }
    spec.Slast = s
    spec.headLens = spec.headLens.map { $0 + 1 }
}
