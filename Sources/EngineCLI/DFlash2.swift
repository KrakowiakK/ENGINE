// T-0163: a native DFlash2 block drafter (public incoai/Qwen3.8-Flash-Next-DFlash2, Apache-2.0) for E9's SERIAL
// speculative path, with a confidence-gated verify length.
//
//   engine generate --model weights/e9 --chat 1 ... --dflash DIR [--dflash-block 8] [--dflash-tau 0.35] --tokens N
//
// Written from reading the public z-lab/dflash reference (model_mlx.py), never from running it. Measured design inputs
// (C-0513, C-0514): with the drafter's 7 drafts per round, verifying the longest draft prefix whose cumulative selector
// confidence stays >= tau (0.3-0.4) commits ~3.1 tokens per verify at a mean of ~4.2 verified positions, where E9's MTP
// K=3 commits 2.83 at 4; the drafter's linear weights are quantized to 8-bit g64 at load (no acceptance change, C-0513)
// and its lm_head is E9's draft-vocabulary-trimmed head (98304 rows, no acceptance change, W30).
//
// The drafter: target features = E9's completed layer 5/15/25/35/45 outputs averaged over the 4 hyper-connection streams
// (Qwen4ExpModelInner.captureLayers), projected by fc + hidden_norm into a per-layer K/V context; a block
// [last committed token, mask x (L-1)] runs 5 non-causal layers (grouped dynamic causal convs around attention and MLP);
// the candidate selector walks the top-16 candidates per position greedily with a low-rank pairwise score.
// Greedy decoding stays exact: every committed token is the target's argmax.
import Foundation
import MLX
import MLXLMCommon
import Qwen4Exp

// T-0163 drafter kernels (ENGINE_DFLASH_KERNELS=0 falls back to the MLX-op graphs above them)
// grouped dynamic causal conv: out[r, c] = sum_off (base[w, off, c] + dyn[r, w, off, c / GS]) * h[r - off, c]
private let dfConvKernel = MLXFast.metalKernel(
    name: "df_gconv", inputNames: ["h", "dyn", "base"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;
        uint r = t / H, c = t % H, g = c / GS;
        float acc = 0.0f;
        for (uint off = 0; off < KS; ++off) {
            if (r >= off) {
                float k = float(base[WH * KS * H + off * H + c]) + float(dyn[r * (2 * KS * G) + WH * KS * G + off * G + g]);
                acc += k * float(h[(r - off) * H + c]);
            }
        }
        y[t] = T(acc);
        """)
// candidate-path selector: one threadgroup of R threads walks J positions; score[c] = unary[j, c] +
// sum_r pc[pred, r] * hp[j, r] * sc[cand[j, c], r]; greedy pick, its softmax probability as the confidence
private let dfSelectKernel = MLXFast.metalKernel(
    name: "df_select", inputNames: ["cand", "unary", "hp", "pc", "sc", "last"], outputNames: ["path", "conf"],
    source: """
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        threadgroup float red[NC * (R / 32)];
        threadgroup int predS;
        if (tid == 0) predS = last[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < J; ++j) {
            int pred = predS;
            float a = float(pc[(ulong)pred * R + tid]) * hp[j * R + tid];
            for (uint c = 0; c < NC; ++c) {
                int cid = cand[j * NC + c];
                float v = simd_sum(a * float(sc[(ulong)cid * R + tid]));
                if (lane == 0) red[c * (R / 32) + sg] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float sc_[NC]; float mx = -INFINITY; uint best = 0;
                for (uint c = 0; c < NC; ++c) {
                    float s = unary[j * NC + c];
                    for (uint k = 0; k < R / 32; ++k) s += red[c * (R / 32) + k];
                    sc_[c] = s;
                    if (s > mx) { mx = s; best = c; }
                }
                float z = 0.0f;
                for (uint c = 0; c < NC; ++c) z += exp(sc_[c] - mx);
                path[j] = cand[j * NC + best];
                conf[j] = 1.0f / z;
                predS = cand[j * NC + best];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """)
// ENGINE_DFLASH_KERNELS: 1 = both (default), 0 = none, conv / select = only that one
nonisolated(unsafe) var dflashConvKernel = ["1", "conv"].contains(ProcessInfo.processInfo.environment["ENGINE_DFLASH_KERNELS"] ?? "1")
nonisolated(unsafe) var dflashSelectKernel = ["1", "select"].contains(ProcessInfo.processInfo.environment["ENGINE_DFLASH_KERNELS"] ?? "1")
var dflashKernels: Bool { dflashConvKernel || dflashSelectKernel }
/// T-0165 TIMING-ONLY ablations for the draft bench (wrong drafts): ENGINE_DFLASH_ABL_LAYERS=N runs only the first N drafter
/// layers, ENGINE_DFLASH_ABL_NOHEAD=1 stops before the draft head and the selector
let dflashAblLayers = Int(ProcessInfo.processInfo.environment["ENGINE_DFLASH_ABL_LAYERS"] ?? "") ?? 99
let dflashAblNoHead = ProcessInfo.processInfo.environment["ENGINE_DFLASH_ABL_NOHEAD"] == "1"
/// T-0165: ENGINE_DFLASH_SPLITK=1 runs the drafter's 8-bit projections (rows <= 8) through the project's split-K kernel
let dflashSplitK = ProcessInfo.processInfo.environment["ENGINE_DFLASH_SPLITK"] == "1"

final class DFlash2Drafter {
    let H = 2560, NH = 32, NKV = 8, HD = 128, NL = 5, KS = 2, GS = 16, TOPK = 16
    let maskId: Int32 = 248077
    let window = 2048
    let eps: Float = 1e-6, theta: Float = 10000
    static let targetLayers = [5, 15, 25, 35, 45]
    private var W: [String: MLXArray] = [:]
    private var Q: [String: (MLXArray, MLXArray, MLXArray)] = [:]
    /// T-0165: ENGINE_DFLASH_BITS (8): the drafter's linear weights quantized to this many bits (g64); 4 halves its ~470 MB read
    let bits: Int = Int(ProcessInfo.processInfo.environment["ENGINE_DFLASH_BITS"] ?? "8") ?? 8

    init(dir: String) throws {
        let all = try loadArrays(url: URL(fileURLWithPath: dir).appendingPathComponent("model.safetensors"))
        for (k, v) in all {
            if k.hasSuffix("_proj.weight") || k == "fc.weight" || k.hasSuffix("kernel_projection.weight") || k.hasSuffix("hidden_projection.weight") {
                let r = quantized(v, groupSize: 64, bits: bits)
                Q[k] = (r.wq, r.scales, r.biases!)
            } else {
                W[k] = v
            }
        }
        // merged projections (fewer dispatches): q|k|v and gate|up per layer, quantized from the concatenated rows
        for l in 0 ..< NL {
            let a = "layers.\(l).self_attn.", m = "layers.\(l).mlp."
            let qkv = concatenated([all[a + "q_proj.weight"]!, all[a + "k_proj.weight"]!, all[a + "v_proj.weight"]!], axis: 0)
            let r1 = quantized(qkv, groupSize: 64, bits: bits); Q[a + "qkv"] = (r1.wq, r1.scales, r1.biases!)
            let gu = concatenated([all[m + "gate_proj.weight"]!, all[m + "up_proj.weight"]!], axis: 0)
            let r2 = quantized(gu, groupSize: 64, bits: bits); Q[m + "gate_up"] = (r2.wq, r2.scales, r2.biases!)
            let kv = concatenated([all[a + "k_proj.weight"]!, all[a + "v_proj.weight"]!], axis: 0)
            let r3 = quantized(kv, groupSize: 64, bits: bits); Q[a + "kv"] = (r3.wq, r3.scales, r3.biases!)
        }
        eval(Array(W.values) + Q.values.flatMap { [$0.0, $0.1, $0.2] })
    }

    private func ql(_ x: MLXArray, _ n: String) -> MLXArray {
        let (w, s, b) = Q[n]!
        if dflashSplitK, bits == 8, let y = q8SplitKMatmul(x, w: w, scales: s, biases: b, groupSize: 64) { return y }
        return quantizedMatmul(x, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
    }
    private func rms(_ x: MLXArray, _ n: String) -> MLXArray { MLXFast.rmsNorm(x, weight: W[n]!, eps: eps) }

    /// grouped dynamic causal conv over the block rows: h (L, H), dyn (L, KS, G), base (KS, H)
    private func gconv(_ h: MLXArray, _ dyn: MLXArray, _ base: MLXArray) -> MLXArray {
        let L = h.dim(0), G = H / GS
        let blocks = h.reshaped(L, G, GS)
        var out = MLXArray.zeros(like: blocks)
        for off in 0 ..< KS {
            let vals = off == 0 ? blocks
                : concatenated([MLXArray.zeros([off, G, GS], dtype: blocks.dtype), blocks[0 ..< (L - off)]], axis: 0)
            out = out + base[off].reshaped(1, G, GS).asType(h.dtype) * vals + dyn[0..., off].reshaped(L, G, 1) * vals
        }
        return out.reshaped(L, H)
    }
    private func kconv(_ h: MLXArray, _ dynRaw: MLXArray, _ base: MLXArray, which: Int) -> MLXArray {
        dfConvKernel([h, dynRaw, base], template: [("T", h.dtype), ("H", H), ("GS", GS), ("G", H / GS), ("KS", KS), ("WH", which)],
                     grid: (h.size, 1, 1), threadGroup: (256, 1, 1), outputShapes: [h.shape], outputDTypes: [h.dtype])[0]
    }
    private func convPrepare(_ x: MLXArray, _ p: String) -> (MLXArray, MLXArray) {
        let raw = ql(x, p + ".kernel_projection.weight")                        // (L, 2 * KS * G)
        if dflashConvKernel { return (kconv(x, raw, W[p + ".base_kernel"]!, which: 0), raw) }
        let dyn = raw.reshaped(x.dim(0), 2, KS, H / GS)
        return (gconv(x, dyn[0..., 0], W[p + ".base_kernel"]![0]), dyn[0..., 1])
    }
    private func convFinish(_ x: MLXArray, _ dyn: MLXArray, _ p: String) -> MLXArray {
        if dflashConvKernel { return kconv(x, dyn, W[p + ".base_kernel"]!, which: 1) }
        return gconv(x, dyn, W[p + ".base_kernel"]![1])
    }

    /// append the context entries of `r` committed positions starting at absolute position `startPos`;
    /// feats (r, 5 * H) in the target-layer order. Keeps at most window - 8 entries (the block's own rows fill the rest).
    func appendContext(_ ctx: DFlashContext, _ feats: MLXArray, startPos: Int) {
        if ctx.ctxLen == 0 { ctx.ctxStart = startPos }   // a context may start mid-sequence (a resumed prefix has no features)
        precondition(startPos == ctx.ctxStart + ctx.ctxLen, "dflash context: append at \(startPos), held up to \(ctx.ctxStart + ctx.ctxLen)")
        let r = feats.dim(0)
        let hn = rms(ql(feats.asType(.bfloat16), "fc.weight"), "hidden_norm.weight")
        for l in 0 ..< NL {
            let a = "layers.\(l).self_attn."
            let kvp = ql(hn, a + "kv")
            var k = MLXFast.rmsNorm(kvp[0..., 0 ..< (NKV * HD)].reshaped(r, NKV, HD), weight: W[a + "k_norm.weight"]!, eps: eps)
            k = MLXFast.RoPE(k.transposed(1, 0, 2)[.newAxis], dimensions: HD, traditional: false, base: theta, scale: 1, offset: startPos)
            let v = kvp[0..., (NKV * HD)...].reshaped(r, NKV, HD).transposed(1, 0, 2)[.newAxis]
            ctx.kc[l] = ctx.kc[l].map { concatenated([$0, k], axis: 2) } ?? k
            ctx.vc[l] = ctx.vc[l].map { concatenated([$0, v], axis: 2) } ?? v
        }
        ctx.ctxLen += r
        let keep = window - 8
        if ctx.ctxLen > keep {
            let drop = ctx.ctxLen - keep
            for l in 0 ..< NL { ctx.kc[l] = ctx.kc[l]![0..., 0..., drop..., 0...]; ctx.vc[l] = ctx.vc[l]![0..., 0..., drop..., 0...] }
            ctx.ctxLen = keep; ctx.ctxStart += drop
        }
    }

    /// one block proposal after `last` (the last committed token, at absolute position ctxStart + ctxLen):
    /// L - 1 lazy draft ids and the selector's confidence of each pick
    func propose(_ ctx: DFlashContext, last: Int, L: Int, model: Qwen4ExpModel) -> (MLXArray, MLXArray) {
        let ids = MLXArray([Int32(last)] + Array(repeating: maskId, count: L - 1))
        var x = model.embedTokenIds(ids)                                       // (L, H)
        let pos0 = ctx.ctxStart + ctx.ctxLen
        for l in 0 ..< min(NL, dflashAblLayers) {
            let p = "layers.\(l).", a = p + "self_attn."
            var res = x
            let (xc, k1) = convPrepare(rms(x, p + "input_layernorm.weight"), p + "attention_conv")
            let qkv = ql(xc, a + "qkv")
            var q = MLXFast.rmsNorm(qkv[0..., 0 ..< (NH * HD)].reshaped(L, NH, HD), weight: W[a + "q_norm.weight"]!, eps: eps)
            var kb = MLXFast.rmsNorm(qkv[0..., (NH * HD) ..< ((NH + NKV) * HD)].reshaped(L, NKV, HD), weight: W[a + "k_norm.weight"]!, eps: eps)
            let vb = qkv[0..., ((NH + NKV) * HD)...].reshaped(L, NKV, HD).transposed(1, 0, 2)[.newAxis]
            q = MLXFast.RoPE(q.transposed(1, 0, 2)[.newAxis], dimensions: HD, traditional: false, base: theta, scale: 1, offset: pos0)
            kb = MLXFast.RoPE(kb.transposed(1, 0, 2)[.newAxis], dimensions: HD, traditional: false, base: theta, scale: 1, offset: pos0)
            let keys = ctx.kc[l].map { concatenated([$0, kb], axis: 2) } ?? kb
            let vals = ctx.vc[l].map { concatenated([$0, vb], axis: 2) } ?? vb
            // non-causal block; the context is held at <= window - 8 entries, so no sliding-window mask is needed
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: vals, scale: 1 / Float(HD).squareRoot(), mask: nil)
            x = res + convFinish(ql(o[0].transposed(1, 0, 2).reshaped(L, NH * HD), a + "o_proj.weight"), k1, p + "attention_conv")
            res = x
            let (xm, k2) = convPrepare(rms(x, p + "post_attention_layernorm.weight"), p + "mlp_conv")
            let gu = ql(xm, p + "mlp.gate_up")
            let I = gu.dim(-1) / 2
            let g = gu[0..., 0 ..< I]
            let y = ql(g * sigmoid(g) * gu[0..., I...], p + "mlp.down_proj.weight")
            x = res + convFinish(y, k2, p + "mlp_conv")
        }
        let hid = rms(x, "norm.weight")[1...]                                   // logits_start = 1
        if dflashAblNoHead { return (hid[0..., 0].asType(.int32), hid[0..., 1].asType(.float32)) }   // TIMING ONLY
        let (lt, rowIds) = model.draftLogits(hid)                               // E9's (trimmed) draft head
        let ci = argPartition(-lt, kth: TOPK - 1, axis: -1)[0..., 0 ..< TOPK]
        let unary = takeAlong(lt, ci, axis: -1).asType(.float32)
        let cand = rowIds.map { $0[ci] } ?? ci
        let hp = ql(hid, "candidate_selector.hidden_projection.weight").asType(.float32)
        let pc = W["candidate_selector.predecessor_codebook"]!, sc = W["candidate_selector.successor_codebook"]!
        if dflashSelectKernel {
            let r = dfSelectKernel([cand.asType(.int32), unary, hp, pc, sc, MLXArray([Int32(last)])],
                                   template: [("J", L - 1), ("NC", TOPK), ("R", hp.dim(-1))],
                                   grid: (hp.dim(-1), 1, 1), threadGroup: (hp.dim(-1), 1, 1),
                                   outputShapes: [[L - 1], [L - 1]], outputDTypes: [.int32, .float32])
            return (r[0], r[1])
        }
        var pred = MLXArray(Int32(last))
        var path: [MLXArray] = [], conf: [MLXArray] = []
        for j in 0 ..< (L - 1) {
            let cj = cand[j]
            let e = (pc[pred].asType(.float32)[.newAxis] * hp[j][.newAxis] * sc[cj].asType(.float32)).sum(axis: -1)
            let s = unary[j] + e
            let k = s.argMax()
            conf.append(softmax(s)[k])
            pred = cj[k].asType(.int32)
            path.append(pred)
        }
        return (stacked(path), stacked(conf))
    }
}

/// one sequence's drafter context: per-layer K/V of its committed positions' target features
final class DFlashContext {
    var kc: [MLXArray?] = Array(repeating: nil, count: 5), vc: [MLXArray?] = Array(repeating: nil, count: 5)
    var ctxLen = 0           // entries held
    var ctxStart = 0         // absolute position of the first held entry
    /// the next absolute position to append (the committed length minus one: the last token's features come next round)
    var end: Int { ctxStart + ctxLen }
}

/// the target features of the last forward's rows (S, 5 * H), bf16, in DFlash2Drafter.targetLayers order
func dflashFeatures() -> MLXArray {
    let parts = DFlash2Drafter.targetLayers.map { Qwen4ExpModelInner.captured[$0]![0] }
    return concatenated(parts, axis: -1).asType(.bfloat16)
}

/// greedy speculative decoding on the serial path with the DFlash2 drafter (see the file header)
func dflashGenerate(model: Qwen4ExpModel, drafter: DFlash2Drafter, prompt: [Int], maxTokens: Int,
                    block L: Int, tau: Float, onPrefill: (() -> Void)? = nil) -> ([Int], SpecStats) {
    Qwen4ExpModelInner.captureLayers = Set(DFlash2Drafter.targetLayers)
    defer { Qwen4ExpModelInner.captureLayers = nil; Qwen4ExpModelInner.captured = [:] }
    let ctx = DFlashContext()
    let cache = model.newCache(parameters: nil)
    var stats = SpecStats()
    let timing = ProcessInfo.processInfo.environment["ENGINE_STEP_TIMING"] != nil
    var draftNs = 0.0, draftBuildNs = 0.0, verifyNs = 0.0, tailNs = 0.0
    // prefill in chunks; every prompt row's features enter the drafter context
    let chunk = Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "4096") ?? 4096
    let x = MLXArray(prompt.map { Int32($0) })[.newAxis]
    var logits = MLXArray(0)
    var s0 = 0
    while s0 < prompt.count {
        let s1 = min(prompt.count, s0 + chunk)
        let (lg, _) = model.forwardHidden(x[0..., s0 ..< s1], cache: cache)
        drafter.appendContext(ctx, dflashFeatures(), startPos: s0)
        logits = lg
        asyncEval(logits)
        s0 = s1
    }
    engineGenCount = 0
    let first = engineApplyThinkBiasBlock(logits[0, (logits.dim(1) - 1)...], generated0: 0).argMax(axis: -1)
    var n0 = first.item(Int.self)
    onPrefill?()
    var out: [Int] = [n0]
    if ProcessInfo.processInfo.environment["ENGINE_DFLASH_KERNEL_CHECK"] != nil {
        // instrument: the same proposals with each kernel on and off, on the prompt's context with 64 different last tokens
        let saved = (dflashConvKernel, dflashSelectKernel)
        func run(_ conv: Bool, _ sel: Bool, _ last: Int) -> ([Int32], [Float]) {
            dflashConvKernel = conv; dflashSelectKernel = sel
            let (d, c) = drafter.propose(ctx, last: last, L: L, model: model)
            return (d.asArray(Int32.self), c.asArray(Float.self))
        }
        var diff = [0, 0, 0], firstPosDiff = [0, 0, 0], maxConf: [Float] = [0, 0, 0]
        for t in prompt.suffix(64) {
            let ref = run(false, false, t)
            for (i, (cv, se)) in [(true, false), (false, true), (true, true)].enumerated() {
                let r = run(cv, se, t)
                if r.0 != ref.0 { diff[i] += 1; if r.0[0] != ref.0[0] { firstPosDiff[i] += 1 } }
                if r.0 == ref.0 { maxConf[i] = max(maxConf[i], zip(r.1, ref.1).map { abs($0 - $1) }.max() ?? 0) }
            }
        }
        (dflashConvKernel, dflashSelectKernel) = saved
        FileHandle.standardError.write("dflash kernel check over 64 proposals vs op graphs: conv-only drafts differ \(diff[0]) (pos0 \(firstPosDiff[0])), select-only \(diff[1]) (pos0 \(firstPosDiff[1])), both \(diff[2]) (pos0 \(firstPosDiff[2])); max |d conf| when equal \(maxConf)\n".data(using: .utf8)!)
        return (out, stats)
    }
    if let nb = ProcessInfo.processInfo.environment["ENGINE_DFLASH_DRAFT_BENCH"], let N = Int(nb) {
        // instrument: N proposals on the prompt's context; host build alone, then build + eval per proposal
        for rep in 0 ..< 3 {
            var t0 = Date()
            var keep: [MLXArray] = []
            for _ in 0 ..< N { let (d, c) = drafter.propose(ctx, last: n0, L: L, model: model); keep += [d, c] }
            let build = Date().timeIntervalSince(t0) * 1e3 / Double(N)
            eval(keep)
            t0 = Date()
            for _ in 0 ..< N { let (d, c) = drafter.propose(ctx, last: n0, L: L, model: model); eval(d, c) }
            let full = Date().timeIntervalSince(t0) * 1e3 / Double(N)
            FileHandle.standardError.write(String(format: "dflash draft bench L=%d ctx %d: host build %.3f ms, build+eval %.3f ms per proposal (kernels %d) [rep %d]\n",
                                                  L, ctx.ctxLen, build, full, dflashKernels ? 1 : 0, rep).data(using: .utf8)!)
        }
        return (out, stats)
    }
    if engineStopIdSet.contains(n0) { return (out, stats) }
    var pre: (MLXArray, MLXArray)? = nil
    let prequeue = ProcessInfo.processInfo.environment["ENGINE_DFLASH_PREQUEUE"] != "0"
    while out.count < maxTokens {
        let t0 = Anchors.now()
        // --- draft: one block pass, then ONE host read of the drafts and their confidences (decides the verify length)
        let (dArr, cArr) = pre ?? drafter.propose(ctx, last: n0, L: L, model: model)
        pre = nil
        let both = concatenated([dArr.asType(.float32), cArr.asType(.float32)], axis: 0)
        let tb = Anchors.now()
        let bh = both.asArray(Float.self)
        let drafts = bh[0 ..< (L - 1)].map { Int($0) }
        let conf = Array(bh[(L - 1)...])
        var m = 0
        var cp: Float = 1
        for c in conf { cp *= c; if cp < tau { break }; m += 1 }
        m = max(1, m)
        let t1 = Anchors.now()
        // --- verify [n0, d1..dm] on the trunk; ONE host sync for the predictions
        let blockIds = MLXArray(([n0] + drafts.prefix(m)).map { Int32($0) })[.newAxis]
        let (vl, _) = model.forwardHidden(blockIds, cache: cache)
        let feats = dflashFeatures()                                            // (m + 1, 5H), lazy
        engineGenCount = out.count
        let forced = engineForcedClose()
        let vlb = engineApplyThinkBiasBlock(vl[0], generated0: out.count)
        let predsArr = forced.map { MLXArray.full([vlb.dim(-2)], values: $0) } ?? vlb.argMax(axis: -1).asType(.int32)
        let preds = predsArr.asArray(Int32.self).map { Int($0) }
        let t2 = Anchors.now()
        var a = 0
        if forced == nil { while a < m && preds[a] == drafts[a] { a += 1 } }
        if engineThinkOpen, engineThinkBias.active, let ci = (0 ..< a).first(where: { drafts[$0] == engineThinkCloseId }),
           ((ci + 1) ... m).contains(where: { engineThinkBias.bias(out.count + $0) > 0 }) { a = ci }
        let bonus = preds[a]
        if a < m { model.rollback(cache, blockRows: m + 1, keep: a + 1) }
        // the committed positions c-1 .. c-1+a enter the drafter context (the bonus token's features come next round)
        drafter.appendContext(ctx, feats[0 ..< (a + 1)], startPos: prompt.count + out.count - 1)
        let newTokens = Array(drafts.prefix(a)) + [bonus]
        stats.rounds += 1; stats.drafted += m; stats.accepted += a; stats.committed += newTokens.count
        stats.kHistogram[m, default: 0] += 1
        if newTokens.contains(engineThinkCloseId) { engineThinkOpen = false }
        if timing { draftNs += Double(t1 - t0); draftBuildNs += Double(tb - t0); verifyNs += Double(t2 - t1); tailNs += Double(Anchors.now() - t2) }
        if let cut = newTokens.firstIndex(where: { engineStopIdSet.contains($0) }) { out += newTokens[...cut]; break }
        out += newTokens
        n0 = bonus
        if prequeue { pre = drafter.propose(ctx, last: n0, L: L, model: model); asyncEval(pre!.0, pre!.1) }
    }
    if timing && stats.rounds > 0 {
        let n = Double(stats.rounds)
        FileHandle.standardError.write(String(format: "dflash round timing: draft (build+GPU+read) %.2f ms (host build %.2f)  verify (build+GPU+read) %.2f ms  tail %.2f ms  per round over %d rounds\n",
                                              draftNs / n / 1e6, draftBuildNs / n / 1e6, verifyNs / n / 1e6, tailNs / n / 1e6, stats.rounds).data(using: .utf8)!)
    }
    return (Array(out.prefix(maxTokens)), stats)
}
