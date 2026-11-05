// T-0142 (D-0043): the mixed step priced and checked against the two separate calls it replaces.
//
//   engine batchprobe --model DIR --prompt FILE --prompt-tokens L --batch B --ragged 1 --verify-rows 4
//       --mixed-chunk Sc [--ragged-stride 64] [--ragged-steps 4] [--mixed-reps 6]
//
// B verify rows at lengths L + b*stride (prefilled twice: one pool per arm, row-resident with ENGINE_ROW_RESIDENT_KV=1
// as the server builds it) take teacher-forced blocks of VS tokens; each step also prefills one chunk of Sc new tokens
// on a fresh cache (offset 0, the first chunk of a prompt; a different corpus window per step).
//   separate arm: the verify step under Qwen4ExpBatch.with + forwardHidden(chunk) -- what the server runs today
//   mixed arm:    forwardMixed(chunk, verify) -- one forward
// Checks, every check step (G1-G3 of research_os/plans/P141, T-0142):
//   G1 chunk: last-row logits, pre-mixer stream and every cache array of the chunk, max|d| and bit-identity
//   G2 verify: logits max|d|, argmax agreement over B*VS positions and the largest reference margin among the
//      disagreements (a disagreement on a knife edge is rounding; one with a clear margin is a bug), stream max|d|
//   G3 state: after the mixed steps one more SEPARATE step on both pools -- equal inputs must give the same logits
// Timing: ABBA over --mixed-reps reps, each arm synchronised per step (eval), plus the verify step and the chunk alone.
import Foundation
import MLX
import MLXLMCommon
import Qwen4Exp

func t0142MixedProbe(_ o: Options, model qm: Qwen4ExpModel, corpus all: [Int], batch B: Int, promptTokens: Int,
                     prefillStep: Int, verifyRows VS: Int) throws {
    let Sc = try o.int("--mixed-chunk", 2048)
    let steps = try o.int("--ragged-steps", 4)
    let stride = try o.int("--ragged-stride", 64)
    let reps = try o.int("--mixed-reps", 6)
    let lens = (0 ..< B).map { promptTokens + $0 * stride }
    let verifyTokens = VS * (steps + 1 + 2 * reps + 2)
    let chunkBase = lens.max()! + verifyTokens + 64
    let chunkWindows = steps + 2 * reps + 2
    guard Sc > 16, all.count > chunkBase + Sc * chunkWindows else {
        throw EngineError.invalid("mixed probe: corpus too short (\(all.count) tokens) for \(chunkWindows) chunks of \(Sc)")
    }
    let rowResident = ProcessInfo.processInfo.environment["ENGINE_ROW_RESIDENT_KV"] == "1"
    func prefill(_ n: Int) -> [KVCache] {
        let cache = qm.newCache(parameters: nil)
        let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
        var last = MLXArray(0); var c0 = 0
        while c0 < n {
            let c1 = min(n, c0 + max(1, prefillStep))
            last = qm(ids[0..., c0 ..< c1], cache: cache)
            if c1 < n { asyncEval(last) }
            c0 = c1
        }
        eval(last, cache.flatMap { $0.state })
        return cache
    }
    func pool() -> [KVCache] {
        let rows = lens.map { prefill($0) }
        let p = rowResident ? qm.stackCachesRowResident(rows) : qm.stackCaches(rows)
        qm.resetRaggedState()
        return p
    }
    let poolS = pool(), poolM = pool()
    // each pool keeps its own indexer watermarks (Qwen4ExpModel.trunkRaggedPooled): swapped in around every use
    var pooledS = qm.trunkRaggedPooled, pooledM = qm.trunkRaggedPooled
    func onS<R>(_ body: () -> R) -> R { qm.trunkRaggedPooled = pooledS; let r = body(); pooledS = qm.trunkRaggedPooled; return r }
    func onM<R>(_ body: () -> R) -> R { qm.trunkRaggedPooled = pooledM; let r = body(); pooledM = qm.trunkRaggedPooled; return r }
    var at = lens
    var window = 0
    func nextChunk() -> MLXArray {
        let c0 = chunkBase + window * Sc; window += 1
        return MLXArray(all[c0 ..< (c0 + Sc)].map { Int32($0) }).reshaped([1, Sc])
    }
    func block(_ rows: [Int]) -> MLXArray {
        MLXArray((0 ..< B).flatMap { b in all[rows[b] ..< (rows[b] + VS)].map { Int32($0) } }).reshaped([B, VS])
    }
    func maxd(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return -1 }
        return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    func cacheMaxd(_ a: [KVCache], _ b: [KVCache]) -> Float {
        var m: Float = 0
        for (x, y) in zip(a, b) {
            let sx = x.state, sy = y.state
            guard sx.count == sy.count else { return -1 }
            for (p, q) in zip(sx, sy) { let d = maxd(p, q); if d < 0 { return -1 }; m = max(m, d) }
        }
        return m
    }
    func separate(_ blk: MLXArray, _ rows: [Int], _ chunk: MLXArray, _ cc: [KVCache]) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let (vl, vh) = onS { Qwen4ExpBatch.with(rows) { qm.forwardHidden(blk, cache: poolS) } }
        let (cl, ch) = qm.forwardHidden(chunk, cache: cc)
        return (cl, ch, vl, vh)
    }

    // ---- checks
    var chunkLogitsD: Float = 0, chunkHiddenD: Float = 0, chunkCacheD: Float = 0
    var verifyLogitsD: Float = 0, verifyHiddenD: Float = 0
    var positions = 0, agree = 0
    // against the corpus itself: position s of row b predicts token at[b] + s + 1 (the arm-independent yardstick)
    var refHits = 0, mixHits = 0
    var refNLL = 0.0, mixNLL = 0.0
    func targetStats(_ logits: MLXArray, _ rows: [Int]) -> (hits: Int, nll: Double) {
        let flat = logits.asType(.float32).reshaped([B * VS, -1])
        let tgt = MLXArray((0 ..< B).flatMap { b in (0 ..< VS).map { Int32(all[rows[b] + $0 + 1]) } }).reshaped([-1, 1])
        let hits = (argMax(flat, axis: -1).reshaped([-1, 1]) .== tgt).asType(.int32).sum().item(Int.self)
        let lp = takeAlong(flat, tgt, axis: -1).reshaped([-1]) - logSumExp(flat, axis: -1)
        return (hits, Double(-lp.sum().item(Float.self)))
    }
    var worstDisagreeMargin: Float = 0
    for _ in 0 ..< steps {
        let blk = block(at), chunk = nextChunk()
        let ccS = qm.newCache(parameters: nil), ccM = qm.newCache(parameters: nil)
        let (clS, chS, vlS, vhS) = separate(blk, at, chunk, ccS)
        eval(clS, chS, vlS, vhS, ccS.flatMap { $0.state }, poolS.flatMap { $0.state })
        let m = onM { qm.forwardMixed(chunk: chunk, chunkCache: ccM, verify: blk, verifyCache: poolM, rows: at) }
        eval(m.chunkLogits, m.chunkHidden, m.verifyLogits, m.verifyHidden, ccM.flatMap { $0.state }, poolM.flatMap { $0.state })
        chunkLogitsD = max(chunkLogitsD, maxd(clS, m.chunkLogits))
        chunkHiddenD = max(chunkHiddenD, maxd(chS, m.chunkHidden))
        chunkCacheD = max(chunkCacheD, cacheMaxd(ccS, ccM))
        verifyLogitsD = max(verifyLogitsD, maxd(vlS, m.verifyLogits))
        verifyHiddenD = max(verifyHiddenD, maxd(vhS, m.verifyHidden))
        let ref = vlS.asType(.float32).reshaped([B * VS, -1]), got = m.verifyLogits.asType(.float32).reshaped([B * VS, -1])
        let rTop = argMax(ref, axis: -1), gTop = argMax(got, axis: -1)
        let same = (rTop .== gTop).asType(.int32)
        positions += B * VS; agree += same.sum().item(Int.self)
        let rs = targetStats(vlS, at), ms = targetStats(m.verifyLogits, at)
        refHits += rs.hits; mixHits += ms.hits; refNLL += rs.nll; mixNLL += ms.nll
        // the reference margin between its own choice and the mixed arm's choice, at the disagreements
        let margin = takeAlong(ref, rTop.reshaped([-1, 1]), axis: -1) - takeAlong(ref, gTop.reshaped([-1, 1]), axis: -1)
        worstDisagreeMargin = max(worstDisagreeMargin, (margin.reshaped([-1]) * (1 - same).asType(.float32)).max().item(Float.self))
        at = at.map { $0 + VS }
    }
    // G3: the pools' states after the mixed history, read through one more separate step on each
    let blk3 = block(at)
    let l3S = onS { Qwen4ExpBatch.with(at) { qm(blk3, cache: poolS) } }
    let l3M = onM { Qwen4ExpBatch.with(at) { qm(blk3, cache: poolM) } }
    eval(l3S, l3M)
    let nextStepD = maxd(l3S, l3M)
    at = at.map { $0 + VS }

    // ---- timing (ABBA), each arm on its own pool, every step synchronised
    func timeSeparate() -> (Double, Double, Double) {
        let blk = block(at), chunk = nextChunk(), cc = qm.newCache(parameters: nil)
        let t0 = Date()
        let (vl, vh) = onS { Qwen4ExpBatch.with(at) { qm.forwardHidden(blk, cache: poolS) } }
        eval(vl, vh)
        let t1 = Date()
        let (cl, ch) = qm.forwardHidden(chunk, cache: cc)
        eval(cl, ch)
        let t2 = Date()
        return (t1.timeIntervalSince(t0) * 1e3, t2.timeIntervalSince(t1) * 1e3, t2.timeIntervalSince(t0) * 1e3)
    }
    func timeMixed() -> Double {
        let blk = block(at), chunk = nextChunk(), cc = qm.newCache(parameters: nil)
        let t0 = Date()
        let m = onM { qm.forwardMixed(chunk: chunk, chunkCache: cc, verify: blk, verifyCache: poolM, rows: at) }
        eval(m.chunkLogits, m.chunkHidden, m.verifyLogits, m.verifyHidden)
        return Date().timeIntervalSince(t0) * 1e3
    }
    var sepV: [Double] = [], sepC: [Double] = [], sep: [Double] = [], mix: [Double] = []
    for r in 0 ..< reps {
        let order = (r % 2 == 0) ? [0, 1] : [1, 0]
        for arm in order {
            if arm == 0 { let (v, c, s) = timeSeparate(); sepV.append(v); sepC.append(c); sep.append(s) }
            else { mix.append(timeMixed()) }
        }
        at = at.map { $0 + VS }
    }
    func med(_ a: [Double]) -> Double { let s = a.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
    let mv = med(sepV), mc = med(sepC), ms = med(sep), mm = med(mix)
    print(String(format: "{\"mixed_chunk\": %d, \"batch\": %d, \"verify_rows\": %d, \"row_resident\": %@, \"check_steps\": %d, "
                 + "\"chunk_bit_identical\": %@, \"chunk_logits_maxd\": %.3e, \"chunk_hidden_maxd\": %.3e, \"chunk_cache_maxd\": %.3e, "
                 + "\"verify_logits_maxd\": %.3e, \"verify_hidden_maxd\": %.3e, \"verify_argmax_agree\": %.4f, \"verify_positions\": %d, "
                 + "\"worst_disagree_margin\": %.3e, \"next_step_maxd\": %.3e, "
                 + "\"verify_ref_acc\": %.4f, \"verify_mixed_acc\": %.4f, \"verify_ref_nll\": %.4f, \"verify_mixed_nll\": %.4f, "
                 + "\"ms_verify\": %.2f, \"ms_chunk\": %.2f, \"ms_separate\": %.2f, \"ms_mixed\": %.2f, \"saving_ms\": %.2f, "
                 + "\"saving_frac_of_verify\": %.3f, \"reps\": %d, \"ms_mixed_all\": \"%@\", \"ms_separate_all\": \"%@\"}",
                 Sc, B, VS, rowResident ? "true" : "false", steps,
                 (chunkLogitsD == 0 && chunkHiddenD == 0 && chunkCacheD == 0) ? "true" : "false",
                 chunkLogitsD, chunkHiddenD, chunkCacheD, verifyLogitsD, verifyHiddenD,
                 Double(agree) / Double(max(1, positions)), positions, worstDisagreeMargin, nextStepD,
                 Double(refHits) / Double(max(1, positions)), Double(mixHits) / Double(max(1, positions)),
                 refNLL / Double(max(1, positions)), mixNLL / Double(max(1, positions)),
                 mv, mc, ms, mm, ms - mm, mv > 0 ? (ms - mm) / mv : 0, reps,
                 mix.map { String(format: "%.1f", $0) }.joined(separator: ","),
                 sep.map { String(format: "%.1f", $0) }.joined(separator: ",")))
}

// T-0148: the LONE row's serial verify riding in a chunk (forwardMixed with rows: nil) against the serial round's own
// verify forward. One row on its own caches (two copies, one per arm), teacher-forced blocks of VS tokens; after each
// check step BOTH caches are rolled back to a partial acceptance (keep = 2 of VS rows, the serial `rollback`) so the next
// step also checks the state a mixed round leaves for the serial rollback.
//   engine batchprobe ... --ragged 1 --verify-rows 4 --mixed-chunk Sc --serial-verify 1 [--ragged-steps 6] [--mixed-reps 6]
func t0148SerialMixedProbe(_ o: Options, model qm: Qwen4ExpModel, corpus all: [Int], promptTokens: Int,
                           prefillStep: Int, verifyRows VS: Int) throws {
    let Sc = try o.int("--mixed-chunk", 2048)
    let steps = try o.int("--ragged-steps", 6)
    let reps = try o.int("--mixed-reps", 6)
    let keep = min(2, VS)
    let chunkBase = promptTokens + VS * (steps + 2 * reps + 4) + 64
    guard Sc > 16, all.count > chunkBase + Sc * (steps + 2 * reps + 2) else {
        throw EngineError.invalid("serial mixed probe: corpus too short (\(all.count) tokens)")
    }
    func prefill(_ n: Int) -> [KVCache] {
        let cache = qm.newCache(parameters: nil)
        let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
        var last = MLXArray(0); var c0 = 0
        while c0 < n {
            let c1 = min(n, c0 + max(1, prefillStep))
            last = qm(ids[0..., c0 ..< c1], cache: cache)
            if c1 < n { asyncEval(last) }
            c0 = c1
        }
        eval(last, cache.flatMap { $0.state })
        return cache
    }
    let rowS = prefill(promptTokens), rowM = prefill(promptTokens)
    var at = promptTokens
    var window = 0
    func nextChunk() -> MLXArray {
        let c0 = chunkBase + window * Sc; window += 1
        return MLXArray(all[c0 ..< (c0 + Sc)].map { Int32($0) }).reshaped([1, Sc])
    }
    func block() -> MLXArray { MLXArray(all[at ..< (at + VS)].map { Int32($0) }).reshaped([1, VS]) }
    func maxd(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return -1 }
        return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    func cacheMaxd(_ a: [KVCache], _ b: [KVCache]) -> Float {
        var m: Float = 0
        for (x, y) in zip(a, b) {
            let sx = x.state, sy = y.state
            guard sx.count == sy.count else { return -1 }
            for (p, q) in zip(sx, sy) { let d = maxd(p, q); if d < 0 { return -1 }; m = max(m, d) }
        }
        return m
    }
    func stats(_ logits: MLXArray) -> (hits: Int, nll: Double) {
        let flat = logits.asType(.float32).reshaped([VS, -1])
        let tgt = MLXArray((0 ..< VS).map { Int32(all[at + $0 + 1]) }).reshaped([-1, 1])
        let hits = (argMax(flat, axis: -1).reshaped([-1, 1]) .== tgt).asType(.int32).sum().item(Int.self)
        let lp = takeAlong(flat, tgt, axis: -1).reshaped([-1]) - logSumExp(flat, axis: -1)
        return (hits, Double(-lp.sum().item(Float.self)))
    }
    var cl: Float = 0, chd: Float = 0, cc: Float = 0, vld: Float = 0, vhd: Float = 0, rowD: Float = 0
    // per (layer, state array): the largest max|d| after rollback with the reference's max|ref| there -- which array a
    // cache difference lives in and how large it is RELATIVE to that array's values
    var worst: [String: (d: Float, ref: Float)] = [:], worstPre: [String: (d: Float, ref: Float)] = [:]
    func perArray(_ a: [KVCache], _ b: [KVCache], into table: inout [String: (d: Float, ref: Float)]) {
        for (li, (x, y)) in zip(a, b).enumerated() {
            for (ai, (p, q)) in zip(x.state, y.state).enumerated() where p.shape == q.shape {
                let d = maxd(p, q), r = abs(p.asType(.float32)).max().item(Float.self)
                let k = "L\(li)a\(ai)"
                if d > (table[k]?.d ?? -1) { table[k] = (d, max(r, table[k]?.ref ?? 0)) }
            }
        }
    }
    var positions = 0, agree = 0, refHits = 0, mixHits = 0
    var refNLL = 0.0, mixNLL = 0.0
    for _ in 0 ..< steps {
        let blk = block(), chunk = nextChunk()
        let ccS = qm.newCache(parameters: nil), ccM = qm.newCache(parameters: nil)
        let (vlS, vhS) = qm.forwardHidden(blk, cache: rowS)
        let (clS, chS) = qm.forwardHidden(chunk, cache: ccS)
        eval(vlS, vhS, clS, chS, rowS.flatMap { $0.state }, ccS.flatMap { $0.state })
        let m = qm.forwardMixed(chunk: chunk, chunkCache: ccM, verify: blk, verifyCache: rowM, rows: nil)
        eval(m.chunkLogits, m.chunkHidden, m.verifyLogits, m.verifyHidden, rowM.flatMap { $0.state }, ccM.flatMap { $0.state })
        cl = max(cl, maxd(clS, m.chunkLogits)); chd = max(chd, maxd(chS, m.chunkHidden)); cc = max(cc, cacheMaxd(ccS, ccM))
        vld = max(vld, maxd(vlS, m.verifyLogits)); vhd = max(vhd, maxd(vhS, m.verifyHidden))
        let rTop = argMax(vlS.asType(.float32).reshaped([VS, -1]), axis: -1), gTop = argMax(m.verifyLogits.asType(.float32).reshaped([VS, -1]), axis: -1)
        positions += VS; agree += (rTop .== gTop).asType(.int32).sum().item(Int.self)
        let rs = stats(vlS), ms = stats(m.verifyLogits)
        refHits += rs.hits; mixHits += ms.hits; refNLL += rs.nll; mixNLL += ms.nll
        // the same arrays right after the verify forward, BEFORE the rollback: a rollback that only selects rows of the
        // forward's (rounding-different) data leaves differences of the same size in the same arrays
        perArray(rowS, rowM, into: &worstPre)
        // a partial acceptance: both rows keep `keep` of the block (the serial rollback)
        qm.rollback(rowS, blockRows: VS, keep: keep); qm.rollback(rowM, blockRows: VS, keep: keep)
        eval(rowS.flatMap { $0.state }, rowM.flatMap { $0.state })
        rowD = max(rowD, cacheMaxd(rowS, rowM))
        perArray(rowS, rowM, into: &worst)
        at += keep
    }
    // G3: one more separate serial step on both rows after the mixed + rollback history
    let blk3 = block()
    let l3S = qm(blk3, cache: rowS), l3M = qm(blk3, cache: rowM)
    eval(l3S, l3M)
    let nextD = maxd(l3S, l3M)
    at += VS
    var sepV: [Double] = [], sepC: [Double] = [], sep: [Double] = [], mix: [Double] = []
    for r in 0 ..< reps {
        for arm in (r % 2 == 0 ? [0, 1] : [1, 0]) {
            let blk = block(), chunk = nextChunk(), cc0 = qm.newCache(parameters: nil)
            if arm == 0 {
                let t0 = Date()
                let (vl, vh) = qm.forwardHidden(blk, cache: rowS); eval(vl, vh)
                let t1 = Date()
                let (c1, c2) = qm.forwardHidden(chunk, cache: cc0); eval(c1, c2)
                let t2 = Date()
                sepV.append(t1.timeIntervalSince(t0) * 1e3); sepC.append(t2.timeIntervalSince(t1) * 1e3); sep.append(t2.timeIntervalSince(t0) * 1e3)
                qm.rollback(rowS, blockRows: VS, keep: VS)
            } else {
                let t0 = Date()
                let mm = qm.forwardMixed(chunk: chunk, chunkCache: cc0, verify: blk, verifyCache: rowM, rows: nil)
                eval(mm.chunkLogits, mm.chunkHidden, mm.verifyLogits, mm.verifyHidden)
                mix.append(Date().timeIntervalSince(t0) * 1e3)
                qm.rollback(rowM, blockRows: VS, keep: VS)
            }
        }
        at += VS
    }
    let top = worst.sorted { $0.value.d > $1.value.d }.prefix(6).map { e -> String in
        let pre = worstPre[e.key]
        return String(format: "%@ d=%.3e ref=%.3e rel=%.2e pre_d=%.3e", e.key, e.value.d, e.value.ref,
                      e.value.ref > 0 ? e.value.d / e.value.ref : 0, pre?.d ?? -1)
    }
    let preRelWorst = worstPre.values.map { $0.ref > 0 ? $0.d / $0.ref : 0 }.max() ?? 0
    let relWorst = worst.values.map { $0.ref > 0 ? $0.d / $0.ref : 0 }.max() ?? 0
    FileHandle.standardError.write(("serial-probe worst cache arrays after rollback: " + top.joined(separator: " | ") + "\n").data(using: .utf8)!)
    print(String(format: "{\"row_cache_worst_rel\": %.3e, \"pre_rollback_worst_rel\": %.3e, \"row_cache_worst\": \"%@\"}",
                 relWorst, preRelWorst, top.first ?? "-"))
    func med(_ a: [Double]) -> Double { let s = a.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
    let mv = med(sepV), mc = med(sepC), ms = med(sep), mm = med(mix)
    print(String(format: "{\"serial_mixed_chunk\": %d, \"verify_rows\": %d, \"check_steps\": %d, \"chunk_bit_identical\": %@, "
                 + "\"chunk_logits_maxd\": %.3e, \"chunk_hidden_maxd\": %.3e, \"chunk_cache_maxd\": %.3e, \"verify_logits_maxd\": %.3e, "
                 + "\"verify_hidden_maxd\": %.3e, \"row_cache_maxd_after_rollback\": %.3e, \"next_step_maxd\": %.3e, "
                 + "\"verify_argmax_agree\": %.4f, \"verify_positions\": %d, \"verify_ref_acc\": %.4f, \"verify_mixed_acc\": %.4f, "
                 + "\"verify_ref_nll\": %.4f, \"verify_mixed_nll\": %.4f, \"ms_verify\": %.2f, \"ms_chunk\": %.2f, \"ms_separate\": %.2f, "
                 + "\"ms_mixed\": %.2f, \"saving_ms\": %.2f, \"saving_frac_of_verify\": %.3f}",
                 Sc, VS, steps, (cl == 0 && chd == 0 && cc == 0) ? "true" : "false", cl, chd, cc, vld, vhd, rowD, nextD,
                 Double(agree) / Double(max(1, positions)), positions, Double(refHits) / Double(max(1, positions)),
                 Double(mixHits) / Double(max(1, positions)), refNLL / Double(max(1, positions)), mixNLL / Double(max(1, positions)),
                 mv, mc, ms, mm, ms - mm, mv > 0 ? (ms - mm) / mv : 0))
}

// T-0152 (C-0439): a PACKED per-row-depth verify block priced and checked against the ordinary batched verify block.
//   engine batchprobe ... --ragged 1 --verify-rows S --batch B --depths d0,d1,..,d(B-1) [--uniform-s S2]
//       [--ragged-stride 64] [--mixed-reps 8]
// B rows at lengths L + b*stride, prefilled once per arm (row-resident with ENGINE_ROW_RESIDENT_KV=1). Teacher-forced
// blocks of S corpus tokens per row; row b keeps depths[b] of them.
//   check (first step, both pools fresh): the ordinary block's logits at the valid positions against the packed
//     block's -- max|d|, argmax agreement, the largest reference margin among disagreements, accuracy and NLL
//     against the corpus. Only the first step: the packed arm's pads leave garbage in its pool (the server's
//     rollback would discard it), so later steps are timing only.
//   timing (ABBA over --mixed-reps): ordinary uniform S, packed, and (with --uniform-s) ordinary uniform S2, each on
//     its own pool, each step synchronised.
func t0152PackedDepthProbe(_ o: Options, model qm: Qwen4ExpModel, corpus all: [Int], batch B: Int, promptTokens: Int,
                           prefillStep: Int, verifyRows S: Int) throws {
    let depths = (o.values["--depths"] ?? "").split(separator: ",").compactMap { Int($0) }
    guard depths.count == B else { throw EngineError.invalid("--depths needs \(B) values, got \(depths.count)") }
    let layout = Qwen4ExpPackedRows(depths: depths, positions: S)
    let S2 = Int(o.values["--uniform-s"] ?? "0") ?? 0   // 0 = no third arm (o.int rejects 0)
    let stride = try o.int("--ragged-stride", 64)
    let reps = try o.int("--mixed-reps", 8)
    let lens = (0 ..< B).map { promptTokens + $0 * stride }
    guard all.count > lens.max()! + S * (2 * reps + 4) + 8 else {
        throw EngineError.invalid("packed depth probe: corpus too short (\(all.count) tokens)")
    }
    let rowResident = ProcessInfo.processInfo.environment["ENGINE_ROW_RESIDENT_KV"] == "1"
    func prefill(_ n: Int) -> [KVCache] {
        let cache = qm.newCache(parameters: nil)
        let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
        var last = MLXArray(0); var c0 = 0
        while c0 < n {
            let c1 = min(n, c0 + max(1, prefillStep))
            last = qm(ids[0..., c0 ..< c1], cache: cache)
            if c1 < n { asyncEval(last) }
            c0 = c1
        }
        eval(last, cache.flatMap { $0.state })
        return cache
    }
    // one pool per arm, each with its own row offsets and indexer watermarks
    final class Arm {
        let pool: [KVCache]; var pooled: [[Int]]; var at: [Int]; var ms: [Double] = []
        init(_ p: [KVCache], _ w: [[Int]], _ at: [Int]) { pool = p; pooled = w; self.at = at }
    }
    func makeArm() -> Arm {
        let rows = lens.map { prefill($0) }
        let p = rowResident ? qm.stackCachesRowResident(rows) : qm.stackCaches(rows)
        qm.resetRaggedState()
        return Arm(p, qm.trunkRaggedPooled, lens)
    }
    func on<R>(_ a: Arm, _ body: () -> R) -> R { qm.trunkRaggedPooled = a.pooled; let r = body(); a.pooled = qm.trunkRaggedPooled; return r }
    func block(_ at: [Int], _ width: Int) -> MLXArray {
        MLXArray((0 ..< B).flatMap { b in all[at[b] ..< (at[b] + width)].map { Int32($0) } }).reshaped([B, width])
    }
    let armU = makeArm(), armP = makeArm()
    let armU2: Arm? = S2 > 0 ? makeArm() : nil

    // ---- check, first step on fresh pools
    let blk = block(armU.at, S)
    let (refL, _) = on(armU) { Qwen4ExpBatch.with(armU.at) { qm.forwardHidden(blk, cache: armU.pool) } }
    let pk = on(armP) { qm.forwardPacked(blk, cache: armP.pool, rows: armP.at, packed: layout) }
    eval(refL, pk.logits, pk.hidden)
    let P = layout.total
    let g = MLXArray(layout.gather)
    let ref = refL.asType(.float32).reshaped([B * S, -1]).take(g, axis: 0)
    let got = pk.logits.asType(.float32).reshaped([P, -1])
    let logitsD = abs(ref - got).max().item(Float.self)
    let rTop = argMax(ref, axis: -1), gTop = argMax(got, axis: -1)
    let same = (rTop .== gTop).asType(.int32)
    let agree = same.sum().item(Int.self)
    let margin = takeAlong(ref, rTop.reshaped([-1, 1]), axis: -1) - takeAlong(ref, gTop.reshaped([-1, 1]), axis: -1)
    let worstMargin = (margin.reshaped([-1]) * (1 - same).asType(.float32)).max().item(Float.self)
    // corpus targets: packed row (b, t) predicts the token at at[b] + t + 1
    let tgt = MLXArray((0 ..< B).flatMap { b in (0 ..< depths[b]).map { Int32(all[armU.at[b] + $0 + 1]) } }).reshaped([-1, 1])
    func stats(_ l: MLXArray) -> (Int, Double) {
        let hits = (argMax(l, axis: -1).reshaped([-1, 1]) .== tgt).asType(.int32).sum().item(Int.self)
        let lp = takeAlong(l, tgt, axis: -1).reshaped([-1]) - logSumExp(l, axis: -1)
        return (hits, Double(-lp.sum().item(Float.self)) / Double(P))
    }
    let (refHits, refNLL) = stats(ref), (pkHits, pkNLL) = stats(got)
    for a in [armU, armP] { a.at = a.at.map { $0 + S } }
    if let a = armU2 {   // keep its pool in step with the others' first block
        let b2 = block(a.at, S2)
        let (l2, h2) = on(a) { Qwen4ExpBatch.with(a.at) { qm.forwardHidden(b2, cache: a.pool) } }
        eval(l2, h2); a.at = a.at.map { $0 + S2 }
    }

    // ---- timing (ABBA)
    func timeUniform(_ a: Arm, _ width: Int) {
        let b = block(a.at, width)
        let t0 = Date()
        let (l, h) = on(a) { Qwen4ExpBatch.with(a.at) { qm.forwardHidden(b, cache: a.pool) } }
        eval(l, h)
        a.ms.append(Date().timeIntervalSince(t0) * 1e3); a.at = a.at.map { $0 + width }
    }
    func timePacked() {
        let b = block(armP.at, S)
        let t0 = Date()
        let r = on(armP) { qm.forwardPacked(b, cache: armP.pool, rows: armP.at, packed: layout) }
        eval(r.logits, r.hidden)
        armP.ms.append(Date().timeIntervalSince(t0) * 1e3); armP.at = armP.at.map { $0 + S }
    }
    for r in 0 ..< reps {
        var order: [Int] = armU2 == nil ? [0, 1] : [0, 1, 2]
        if r % 2 == 1 { order.reverse() }
        for arm in order {
            switch arm {
            case 0: timeUniform(armU, S)
            case 1: timePacked()
            default: timeUniform(armU2!, S2)
            }
        }
    }
    func med(_ a: [Double]) -> Double { let s = a.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
    func list(_ a: [Double]) -> String { a.map { String(format: "%.1f", $0) }.joined(separator: ",") }
    // C-0438 fixed-B fit, for the reader's comparison
    func fit(_ pos: Int) -> Double { 47.39 + 0.278 * Double(B) + 1.472 * Double(pos) }
    print(String(format: "{\"packed_depths\": \"%@\", \"batch\": %d, \"S\": %d, \"packed_positions\": %d, \"row_resident\": %@, "
                 + "\"logits_maxd\": %.3e, \"argmax_agree\": %.4f, \"worst_disagree_margin\": %.3e, "
                 + "\"ref_acc\": %.4f, \"packed_acc\": %.4f, \"ref_nll\": %.4f, \"packed_nll\": %.4f, "
                 + "\"ms_uniform_S\": %.2f, \"ms_packed\": %.2f, \"uniform_s2\": %d, \"ms_uniform_S2\": %.2f, "
                 + "\"fit_uniform_S\": %.2f, \"fit_packed\": %.2f, \"fit_uniform_S2\": %.2f, \"reps\": %d, "
                 + "\"ms_uniform_all\": \"%@\", \"ms_packed_all\": \"%@\", \"ms_uniform2_all\": \"%@\"}",
                 depths.map(String.init).joined(separator: ","), B, S, P, rowResident ? "true" : "false",
                 logitsD, Double(agree) / Double(P), worstMargin,
                 Double(refHits) / Double(P), Double(pkHits) / Double(P), refNLL, pkNLL,
                 med(armU.ms), med(armP.ms), S2, med(armU2?.ms ?? []),
                 fit(B * S), fit(P), fit(B * S2), reps, list(armU.ms), list(armP.ms), list(armU2?.ms ?? [])))
}
