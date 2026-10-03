// Fused Metal kernels for the Qwen4Exp decode step.
//
// Each kernel replaces a chain of 3-7 tiny MLX primitives with one dispatch. The
// point is dispatch count: the decode step issues ~5 100 primitives, the encode of
// which (~4.5 ms) is serialised behind the previous GPU step (OBS-ENG-016), and
// every tiny kernel costs ~3 us of GPU time on its own. Math is float32 inside,
// bf16 at the edges, same as the unfused MLX chain; `engine kernel-check`
// compares every kernel against the unfused ops on random data.
//
// Disable with ENGINE_NO_FUSED=1.
import Foundation
import MLX
import MLXFast
import MLXNN

enum Q4Fused {
    static let enabled: Bool = ProcessInfo.processInfo.environment["ENGINE_NO_FUSED"] == nil
    /// per-kernel mask (ENGINE_FUSE_MASK, default all): 1 grouped_norm, 2 hc_mix, 4 hc_inject, 8 silu_div,
    /// 16 moe_combine, 32 head_norm, 64 gated_norm, 128 rope
    /// 6038 = 2|4|16|128|256|512|1024|4096: hc_mix, hc_inject, moe_combine, rope, fused experts, fused HC chain, router top-k, lane down kernel.
    /// Measured 2026-08-27 (M3 Ultra, MIX-1v, 512/128): serial 46.5 -> 54.6 tok/s, MTP K=3 81 -> 105 tok/s (384 tokens), greedy tokens identical.
    /// Not in the default: 1/8/32/64/2048 (no measurable effect: sub-microsecond kernels are not on the critical path).
    /// 38806 = 6038 | 32768: + split-K (deterministic partials) for the hyper-connection mixDown, the latency-bound skinny qmv on the
    /// critical path. Measured 2026-08-28: serial 55.7 -> 61.7 tok/s, MTP K=3 102.6 -> 105.7, greedy tokens identical (lab/hc_chain_dram.py).
    /// 28940182 = 3774358 | 8388608 | 16777216 -- P025 adds the 8-bit g64 fused expert path (8388608)
    /// and the MULTI-ROW form of its gate/up half (16777216), which reads each DISTINCT expert once
    /// for all the trunk rows that selected it. At 262144 on E8h, three interleaved passes:
    /// MTP K=2 56.41 -> 58.56 tok/s, K=3 51.10 -> 54.74, serial 44.79 -> 48.13 (OBS-ENG-081).
    static let mask: Int = Int(ProcessInfo.processInfo.environment["ENGINE_FUSE_MASK"] ?? "28940182") ?? 28940182   // unit 36 adds 2097152 (dense hc chain). 4194304 (the coalesced simdgroup body) is OFF: REFUTED, it computes garbage (REFUT-ENG-012)
    static var groupedNorm: Bool { enabled && mask & 1 != 0 }
    static var hcMix: Bool { enabled && mask & 2 != 0 }
    static var hcInject: Bool { enabled && mask & 4 != 0 }
    static var siluDiv: Bool { enabled && mask & 8 != 0 }
    static var moeCombine: Bool { enabled && mask & 16 != 0 }
    static var headNorm: Bool { enabled && mask & 32 != 0 }
    static var gatedNorm: Bool { enabled && mask & 64 != 0 }
    static var rope: Bool { enabled && mask & 128 != 0 }
    static var moeExperts: Bool { enabled && mask & 256 != 0 }   // fused gate/up + down expert path (M <= 8)
    static var hcFused: Bool { enabled && mask & 512 != 0 }
    static var moeTopK: Bool { enabled && mask & 1024 != 0 }     // router top-k + softmax (+ shared gate slice) in one kernel
    static var gdnGate: Bool { enabled && mask & 2048 != 0 }     // GDN g/beta gate math in one kernel
    static var moeDownLane: Bool { enabled && mask & 4096 != 0 } // lane-per-row expert down kernel + combine (needs 256)
    static var hcChainNorm: Bool { enabled && mask & 8192 != 0 } // combine kernel also emits the next hyper-connection's grouped norm (needs 512)
    static var splitKM: Bool { enabled && mask & 16384 != 0 }    // 8-bit g64 projections at 2..8 rows through q8_qmv_splitk (MLX's qmm is 2-3x off bandwidth there)      // hyper-connection: norm kernel -> qmv -> up+silu+sigmoid+mix+inject kernel -> combine(partials)
    static var hcSplitK: Bool { enabled && mask & 32768 != 0 }
    /// unit 36: the DENSE (bf16) hyper-connection fused chain. Separate bit so it can be ablated
    /// against its own MLX fallback on the SAME checkpoint -- the only test that can show it is
    /// numerically indistinguishable, which greedy identity across models cannot.
    static var hcFusedDense: Bool { enabled && mask & 2097152 != 0 }
    static var hcDenseSG: Bool { enabled && mask & 4194304 != 0 }   // dense hc up-mix: simdgroup-per-column body
    static var gdnFront: Bool { enabled && mask & 65536 != 0 }
    static var attnFront: Bool { enabled && mask & 131072 != 0 }  // attention decode front end (q/k norm + rope + v + sigmoid(gate)) in one kernel, S = 1: 153.9 -> 133.9 us per layer in the lab (lab/attn_chain_dram.py)
    static var moeFold: Bool { enabled && mask & 262144 != 0 }    // MoE combine folded into the hyper-connection inject kernel (one dispatch less per MoE block)   // GDN decode front end (conv+silu+head-norm, no concat) in one kernel + simdgroup-per-head gated norm: 167.8 -> 136.7 us per layer in the dependent-chain lab (lab/gdn_chain_dram.py), bit-exact   // hyper-connection mixDown (N=320, K=10240) through split-K at 1..8 rows: MLX's qmv is LATENCY-bound on the critical path (28 us dependent vs 6 us throughput slope; lab/hc_chain_dram.py)
    static var attnChunk: Bool { enabled && mask & 524288 != 0 }  // decode blocks of 3..16 rows attend in 32/gqa-row chunks so MLX's fused sdpa_vector runs instead of the unfused fallback (P016)
    static var qsaGather: Bool { enabled && mask & 1048576 != 0 }
    /// P025: accept 8-bit g64 expert banks into the fused expert path. Separate bit from 256 so the
    /// FORMAT and the PATH can be ablated independently on one binary -- on a 4-bit artifact this bit
    /// is inert, on an 8-bit one bit 256 alone is (OBS-ENG-077).
    static var moeExperts8: Bool { enabled && mask & 8388608 != 0 }
    /// P025 unit 6: read each DISTINCT expert once for all the trunk rows that selected it, instead of
    /// once per (row, slot) pair. Needs 8388608. Separate bit so the format, the fused path and the
    /// row sharing can each be ablated on one binary.
    static var moeExperts8MR: Bool { enabled && mask & 16777216 != 0 }  // decode blocks attend ONLY to the QSA-selected blocks (gather kernel) instead of masking the whole kv (P017)
}

/// float constants for kernels (template args may only be Dtype/Int/Bool); cached per value
nonisolated(unsafe) private var q4ConstCache: [Float: MLXArray] = [:]
private let q4ConstLock = NSLock()
func q4Const(_ v: Float) -> MLXArray {
    q4ConstLock.lock(); defer { q4ConstLock.unlock() }
    if let a = q4ConstCache[v] { return a }
    let a = MLXArray([v]); eval(a); q4ConstCache[v] = a; return a
}

// MARK: grouped zero-centered RMSNorm  y = rms_group(x) * (1 + w)
// x [R, G*D]; one threadgroup (TG threads) per (row, group); simd + threadgroup reduction
private let q4GroupedNormKernel = MLXFast.metalKernel(
    name: "q4_grouped_norm", inputNames: ["x", "w1", "eps"], outputNames: ["y"],
    source: """
        uint tg = threadgroup_position_in_grid.x;   // row * G + g
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = tg / G, g = tg % G;
        ulong base = (ulong)row * (ulong)(G * D) + (ulong)g * D;
        threadgroup float part[32];
        float acc = 0.0f;
        for (uint i = tid; i < D; i += TG) { float v = float(x[base + i]); acc += v * v; }
        acc = simd_sum(acc);
        if (lane == 0) part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint k = 0; k < TG / 32; ++k) tot += part[k];
        float inv = rsqrt(tot / float(D) + eps[0]);
        // ROUND2=1 reproduces MLX's two-step rounding (rms_norm writes T(x*inv), the caller then multiplies by the
        // weight in T) so a caller can be bit-identical to the unfused chain; ROUND2=0 is float32 throughout, which is
        // what HF does (P020 unit 1) and what the decode kernels already do.
        for (uint i = tid; i < D; i += TG) {
            float v = float(x[base + i]) * inv;
            y[base + i] = ROUND2 ? T(float(T(v)) * float(w1[g * D + i])) : T(v * float(w1[g * D + i]));
        }
        """)

/// P019 unit 6: the same grouped RMS norm with (a) ONE read of x -- the strided version read it twice, once for the
/// sum of squares and once to scale -- and (b) VEC contiguous elements per thread instead of a 2-byte stride, so a
/// simdgroup issues one 512-byte transaction instead of 32 two-byte ones. Requires D = TG * VEC exactly; the caller
/// picks TG = D/8 when that is a multiple of 32 and at most 1024, and falls back to the strided kernel otherwise.
private let q4GroupedNormVecKernel = MLXFast.metalKernel(
    name: "q4_grouped_norm_vec", inputNames: ["x", "w1", "eps"], outputNames: ["y"],
    source: """
        constexpr int VEC = D / TG;
        uint tg = threadgroup_position_in_grid.x;   // row * G + g
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = tg / G, g = tg % G;
        ulong base = (ulong)row * (ulong)(G * D) + (ulong)g * D + (ulong)tid * VEC;
        threadgroup float part[32];
        float v[VEC];
        float acc = 0.0f;
        for (int i = 0; i < VEC; ++i) { v[i] = float(x[base + i]); acc += v[i] * v[i]; }
        acc = simd_sum(acc);
        if (lane == 0) part[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint k = 0; k < TG / 32; ++k) tot += part[k];
        float inv = rsqrt(tot / float(D) + eps[0]);
        uint wbase = g * D + tid * VEC;
        for (int i = 0; i < VEC; ++i) y[base + i] = T(v[i] * inv * float(w1[wbase + i]));
        """)
/// ENGINE_NORM_VEC=0 restores the strided kernel
let q4NormVec: Bool = (ProcessInfo.processInfo.environment["ENGINE_NORM_VEC"] ?? "0") != "0"   // MEASURED EQUAL to the strided kernel (1.600 vs 1.610 ms per call at a 4096-row chunk): the double read of x and the 2-byte stride were NOT what limits it. Kept as a knob, default OFF so the champion carries no unmeasured change

func q4GroupedNorm(_ x: MLXArray, onePlusW: MLXArray, groups G: Int, groupSize D: Int, eps: Float) -> MLXArray {
    let rows = x.size / (G * D)
    let tgv = D / 8
    if q4NormVec && D % 8 == 0 && tgv % 32 == 0 && tgv <= 1024 {
        return q4GroupedNormVecKernel(
            [x, onePlusW, q4Const(eps)],
            template: [("T", x.dtype), ("G", G), ("D", D), ("TG", tgv)],
            grid: (rows * G * tgv, 1, 1), threadGroup: (tgv, 1, 1),
            outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }
    let TG = 256
    return q4GroupedNormKernel(
        [x, onePlusW, q4Const(eps)],
        template: [("T", x.dtype), ("G", G), ("D", D), ("TG", TG), ("ROUND2", q4GroupedNormRound2 ? 1 : 0)],
        grid: (rows * G * TG, 1, 1), threadGroup: (TG, 1, 1),
        outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

// MARK: hyper-connection mix  mixed[r,j] = mean_h sigmoid(u[r,h,j]) * normed[r,h,j]
private let q4HCMixKernel = MLXFast.metalKernel(
    name: "q4_hc_mix", inputNames: ["normed", "u"], outputNames: ["mixed"],
    source: """
        uint t = thread_position_in_grid.x;         // row * D + j
        uint row = t / D, j = t % D;
        ulong base = (ulong)row * (ulong)(H * D) + j;
        float acc = 0.0f;
        for (uint h = 0; h < H; ++h) {
            float uu = float(u[base + h * D]);
            float sg = 1.0f / (1.0f + exp(-uu));
            acc += sg * float(normed[base + h * D]);
        }
        mixed[(ulong)row * D + j] = T(acc / float(H));
        """)

func q4HCMix(normed: MLXArray, u: MLXArray, hc H: Int, d D: Int) -> MLXArray {
    let rows = normed.size / (H * D)
    let lead = Array(normed.shape.dropLast())
    return q4HCMixKernel(
        [normed, u],
        template: [("T", normed.dtype), ("H", H), ("D", D)],
        grid: (rows * D, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [lead + [D]], outputDTypes: [normed.dtype])[0]
}

// MARK: hyper-connection mix + inject in one pass (body in Kernels/lab/hc_mix_inject.metal)
private let q4HCMixInjectKernel = MLXFast.metalKernel(
    name: "q4_hc_mix_inject", inputNames: ["normed", "u", "winj"], outputNames: ["mixed", "inj"], source: Q4LabKernelSource.hc_mix_inject)
/// ENGINE_HC_MIX_INJECT=0 restores the separate mix kernel and the inject qmv
let q4HCMixInjectOn: Bool = (ProcessInfo.processInfo.environment["ENGINE_HC_MIX_INJECT"] ?? "1") != "0"
let q4HCMixInjectRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_HC_MIX_INJECT_ROWS"] ?? "1") ?? 1
/// -> (mixed (lead..., D), raw inject logits (lead..., H)); the combine applies 2*sigmoid(./H)
func q4HCMixInject(normed: MLXArray, u: MLXArray, winj: MLXArray, hc H: Int, d D: Int, rows R: Int) -> (MLXArray, MLXArray) {
    let rows = normed.size / (H * D)
    let lead = Array(normed.shape.dropLast())
    let TG = 256
    precondition(D % TG == 0 && winj.dim(0) == H)
    let r = q4HCMixInjectKernel(
        [normed, u, winj],
        template: [("T", normed.dtype), ("H", H), ("D", D), ("TG", TG), ("ROWS", R), ("NROWS", rows)],
        grid: (((rows + R - 1) / R) * TG, 1, 1), threadGroup: (TG, 1, 1),
        outputShapes: [lead + [D], lead + [H]], outputDTypes: [normed.dtype, normed.dtype])
    return (r[0], r[1])
}

// MARK: inject + combine  out[r,h,j] = hyper[r,h,j] + 2*sigmoid(inj[r,h]/H) * x[r,j]
private let q4HCInjectKernel = MLXFast.metalKernel(
    name: "q4_hc_inject", inputNames: ["hyper", "x", "inj"], outputNames: ["out"],
    source: """
        uint t = thread_position_in_grid.x;         // row*H*D + h*D + j
        uint row = t / (H * D), rem = t % (H * D), h = rem / D, j = rem % D;
        float g = float(inj[(ulong)row * H + h]) / float(H);
        float w = 2.0f / (1.0f + exp(-g));
        out[t] = T(float(hyper[t]) + w * float(x[(ulong)row * D + j]));
        """)

// MARK: combine + next grouped norm for PREFILL widths (body in Kernels/lab/hc_inject_norm2.metal)
private let q4HCInjectNorm2Kernel = MLXFast.metalKernel(
    name: "q4_hc_inject_norm2", inputNames: ["hyper", "x", "inj", "w1", "epsv"], outputNames: ["out", "normed"], source: Q4LabKernelSource.hc_inject_norm2)
/// ENGINE_GROUPED_NORM_PREFILL=0 restores MLX's rms_norm-then-multiply for grouped norms at prefill widths
/// ENGINE_GROUPED_NORM_ROUND2=0 makes the grouped norm float32 throughout (HF's form); default 1 reproduces MLX's
/// two-step rounding so the prefill program stays bit-identical to the champion's
let q4GroupedNormRound2: Bool = (ProcessInfo.processInfo.environment["ENGINE_GROUPED_NORM_ROUND2"] ?? "1") != "0"
let q4GroupedNormPrefill: Bool = (ProcessInfo.processInfo.environment["ENGINE_GROUPED_NORM_PREFILL"] ?? "1") != "0"
/// ENGINE_HC_CHAIN_PREFILL=0 restores the separate combine and norm
let q4HCChainPrefill: Bool = (ProcessInfo.processInfo.environment["ENGINE_HC_CHAIN_PREFILL"] ?? "1") != "0"
/// -> (residual out, the NEXT hyper-connection's grouped norm of it). Bit-identical to q4HCInject + q4GroupedNorm.
func q4HCInjectNorm2(hyper: MLXArray, x: MLXArray, inj: MLXArray, nextOnePlusW: MLXArray, eps: Float, hc H: Int, d D: Int) -> (MLXArray, MLXArray) {
    let rows = hyper.size / (H * D)
    let TGS = 256
    precondition(D % TGS == 0)
    let r = q4HCInjectNorm2Kernel([hyper, x, inj, nextOnePlusW, q4Const(eps)],
                                  template: [("T", hyper.dtype), ("H", H), ("D", D), ("TGS", TGS), ("ROUND2", q4GroupedNormRound2 ? 1 : 0)],
                                  grid: (rows * H * TGS, 1, 1), threadGroup: (TGS, 1, 1),
                                  outputShapes: [hyper.shape, hyper.shape], outputDTypes: [hyper.dtype, hyper.dtype])
    return (r[0], r[1])
}

func q4HCInject(hyper: MLXArray, x: MLXArray, inj: MLXArray, hc H: Int, d D: Int) -> MLXArray {
    return q4HCInjectKernel(
        [hyper, x, inj],
        template: [("T", hyper.dtype), ("H", H), ("D", D)],
        grid: (hyper.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [hyper.shape], outputDTypes: [hyper.dtype])[0]
}

// MARK: silu(x / H)
private let q4SiluDivKernel = MLXFast.metalKernel(
    name: "q4_silu_div", inputNames: ["x"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;
        float v = float(x[t]) / float(H);
        y[t] = T(v / (1.0f + exp(-v)));
        """)

// MARK: silu(g) * u  (one dispatch; replaces mlx-swift-lm's compiled SwiGLU closure)
private let q4SiluMulKernel = MLXFast.metalKernel(
    name: "q4_silu_mul", inputNames: ["g", "u"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;
        float v = float(g[t]);
        y[t] = T((v / (1.0f + exp(-v))) * float(u[t]));
        """)

func q4SiluMul(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
    q4SiluMulKernel([g, u], template: [("T", g.dtype)],
                    grid: (g.size, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [g.shape], outputDTypes: [g.dtype])[0]
}

// MARK: silu(h[r, :I]) * h[r, I:]  for a merged gate|up projection output (rows, 2I) -> (rows, I)
private let q4SiluMulSplitKernel = MLXFast.metalKernel(
    name: "q4_silu_mul_split", inputNames: ["h"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;         // row * I + j
        uint row = t / I, j = t % I;
        float v = float(h[row * 2 * I + j]);
        y[t] = T((v / (1.0f + exp(-v))) * float(h[row * 2 * I + I + j]));
        """)

func q4SiluMulSplit(_ h: MLXArray, hidden I: Int) -> MLXArray {
    var shape = h.shape; shape[shape.count - 1] = I
    return q4SiluMulSplitKernel([h], template: [("T", h.dtype), ("I", I)],
                                grid: (h.size / 2, 1, 1), threadGroup: (256, 1, 1),
                                outputShapes: [shape], outputDTypes: [h.dtype])[0]
}

func q4SiluDiv(_ x: MLXArray, by H: Int) -> MLXArray {
    q4SiluDivKernel([x], template: [("T", x.dtype), ("H", H)],
                    grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

// MARK: MoE combine  out[r,j] = sum_k w[r,k]*y[r,k,j] + sigmoid(g[r]) * s[r,j]
private let q4MoECombineKernel = MLXFast.metalKernel(
    name: "q4_moe_combine", inputNames: ["y", "w", "s", "g"], outputNames: ["out"],
    source: """
        uint t = thread_position_in_grid.x;         // row * D + j
        uint row = t / D, j = t % D;
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            acc += float(w[(ulong)row * K + k]) * float(y[((ulong)row * K + k) * D + j]);
        }
        float gg = float(g[row]);
        float sg = 1.0f / (1.0f + exp(-gg));
        out[t] = T(acc + sg * float(s[t]));
        """)

func q4MoECombine(y: MLXArray, w: MLXArray, shared s: MLXArray, gate g: MLXArray, topK K: Int, d D: Int) -> MLXArray {
    q4MoECombineKernel([y, w, s, g], template: [("T", s.dtype), ("K", K), ("D", D)],
                       grid: (s.size, 1, 1), threadGroup: (256, 1, 1),
                       outputShapes: [s.shape], outputDTypes: [s.dtype])[0]
}

// MARK: per-head RMS norm (no weight) times a scalar   y = x / sqrt(mean(x^2)+eps) * scale
// x [R, HEADS, DH] -> one thread per (row, head)
private let q4HeadNormScaleKernel = MLXFast.metalKernel(
    name: "q4_head_norm_scale", inputNames: ["x", "eps", "scale"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;         // row*HEADS + h
        ulong base = (ulong)t * DH;
        float acc = 0.0f;
        for (uint i = 0; i < DH; ++i) { float v = float(x[base + i]); acc += v * v; }
        float inv = rsqrt(acc / float(DH) + eps[0]) * scale[0];
        for (uint i = 0; i < DH; ++i) y[base + i] = T(float(x[base + i]) * inv);
        """)

func q4HeadNormScale(_ x: MLXArray, headDim DH: Int, scale: Float, eps: Float) -> MLXArray {
    let n = x.size / DH
    return q4HeadNormScaleKernel([x, q4Const(eps), q4Const(scale)], template: [("T", x.dtype), ("DH", DH)],
                                 grid: (n, 1, 1), threadGroup: (min(n, 256), 1, 1),
                                 outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

// MARK: gated RMS norm (GDN output)  y = rms(x)*w * sigmoid(z)   (float32 math, bf16 out)
private let q4GatedNormKernel = MLXFast.metalKernel(
    name: "q4_gated_norm", inputNames: ["x", "w", "z", "eps"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;         // row*HEADS + h
        ulong base = (ulong)t * DH;
        float acc = 0.0f;
        for (uint i = 0; i < DH; ++i) { float v = float(x[base + i]); acc += v * v; }
        float inv = rsqrt(acc / float(DH) + eps[0]);
        for (uint i = 0; i < DH; ++i) {
            float zz = float(z[base + i]);
            float sg = SIG ? 1.0f / (1.0f + exp(-zz)) : zz / (1.0f + exp(-zz));
            float nv = float(TW(float(x[base + i]) * inv * float(w[i])));   // rms_norm output rounded to the weight dtype first
            y[base + i] = T(sg * nv);
        }
        """)

func q4GatedNorm(_ x: MLXArray, weight w: MLXArray, gate z: MLXArray, headDim DH: Int, eps: Float, sigmoidGate: Bool, outDType: DType) -> MLXArray {
    let n = x.size / DH
    return q4GatedNormKernel([x, w, z, q4Const(eps)], template: [("T", outDType), ("TW", w.dtype), ("DH", DH), ("SIG", sigmoidGate)],
                             grid: (n, 1, 1), threadGroup: (min(n, 256), 1, 1),
                             outputShapes: [x.shape], outputDTypes: [outDType])[0]
}

// MARK: partial RoPE (half rotation on the first ROT dims), x [B, H, S, HD], cos/sin [S, ROT]
private let q4RopeKernel = MLXFast.metalKernel(
    name: "q4_rope_partial", inputNames: ["x", "c", "s"], outputNames: ["y"],
    source: """
        uint t = thread_position_in_grid.x;         // (b*H + h)*S*HD + pos*HD + i
        uint i = t % HD;
        uint pos = (t / HD) % S;
        float v = float(x[t]);
        if (i >= ROT) { y[t] = T(v); return; }
        uint hf = ROT / 2;                          // `half` is a Metal type name
        ulong tab = (ulong)pos * ROT + i;
        float cc = c[tab], ss = s[tab];
        if (i < hf) {
            float x2 = float(x[t + hf]);
            y[t] = T(v * cc - x2 * ss);
        } else {
            float x1 = float(x[t - hf]);
            y[t] = T(v * cc + x1 * ss);
        }
        """)

func q4RopePartial(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, rotDim ROT: Int) -> MLXArray {
    let S = x.dim(2), HD = x.dim(3)
    return q4RopeKernel([x, c, s], template: [("T", x.dtype), ("S", S), ("HD", HD), ("ROT", ROT)],
                        grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                        outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

// MARK: fused MoE expert path (affine 4-bit g32), M = T*K rows small (decode, drafts, verify blocks)
// bodies shared with lab/moe_kernel_lab.py (Kernels/lab/moe_gateup.metal, moe_down.metal)
private let q4MoEGateUpKernel = MLXFast.metalKernel(
    name: "q4_moe_gateup", inputNames: ["x", "idx", "gw", "gs", "gb", "uw", "us", "ub"], outputNames: ["h"],
    source: """
        constexpr uint NSG = 2, VPT = 16, BLOCK = 512;   // ROWS from template
        uint tg = threadgroup_position_in_grid.x;        // t*K + k
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint t = tg / K, k = tg % K;
        int e = idx[tg];
        uint row0 = tile * (NSG * ROWS) + sg * ROWS;
        const uint DW = D / 8, DG = D / 32;
        float accg[ROWS], accu[ROWS];
        for (uint r = 0; r < ROWS; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }
        for (uint kb = 0; kb < D; kb += BLOCK) {
            uint off = kb + lane * VPT;
            float xv[VPT]; float xs = 0.0f;
            for (uint i = 0; i < VPT; ++i) { xv[i] = float(x[t * D + off + i]); xs += xv[i]; }
            uint g = off / 32;
            for (uint r = 0; r < ROWS; ++r) {
                ulong rowbase = ((ulong)e * I + row0 + r);
                const device uint32_t* gp = gw + rowbase * DW + off / 8;
                const device uint32_t* up = uw + rowbase * DW + off / 8;
                float dg = 0.0f, du = 0.0f;
                for (uint p = 0; p < 2; ++p) {
                    uint32_t wg = gp[p], wu = up[p];
                    for (uint j = 0; j < 8; ++j) {
                        float xj = xv[p * 8 + j];
                        dg += float((wg >> (4 * j)) & 0xF) * xj;
                        du += float((wu >> (4 * j)) & 0xF) * xj;
                    }
                }
                accg[r] += float(gs[rowbase * DG + g]) * dg + float(gb[rowbase * DG + g]) * xs;
                accu[r] += float(us[rowbase * DG + g]) * du + float(ub[rowbase * DG + g]) * xs;
            }
        }
        for (uint r = 0; r < ROWS; ++r) {
            float gsum = simd_sum(accg[r]);
            float usum = simd_sum(accu[r]);
            if (lane == 0) {
                float sil = gsum / (1.0f + exp(-gsum));
                h[((ulong)t * K + k) * I + row0 + r] = T(sil * usum);
            }
        }
        """)

private let q4MoEDownKernel = MLXFast.metalKernel(
    name: "q4_moe_down", inputNames: ["h", "idx", "wts", "dw", "ds", "db"], outputNames: ["y"],
    source: """
        constexpr uint NSG = 2, ROWS = 4, VPT = 16, BLOCK = 512;
        uint t = threadgroup_position_in_grid.x;
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint row0 = tile * (NSG * ROWS) + sg * ROWS;
        const uint IW = I / 8, IG = I / 32;
        float acc[ROWS];
        for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int e = idx[t * K + k];
            float wk = float(wts[t * K + k]);
            float part[ROWS];
            for (uint r = 0; r < ROWS; ++r) part[r] = 0.0f;
            for (uint kb = 0; kb < I; kb += BLOCK) {
                uint off = kb + lane * VPT;
                if (off < I) {
                    float xv[VPT]; float xs = 0.0f;
                    for (uint i = 0; i < VPT; ++i) { xv[i] = float(h[((ulong)t * K + k) * I + off + i]); xs += xv[i]; }
                    uint g = off / 32;
                    for (uint r = 0; r < ROWS; ++r) {
                        ulong rowbase = ((ulong)e * D + row0 + r);
                        const device uint32_t* wp = dw + rowbase * IW + off / 8;
                        float d = 0.0f;
                        for (uint p = 0; p < 2; ++p) {
                            uint32_t w = wp[p];
                            for (uint j = 0; j < 8; ++j) d += float((w >> (4 * j)) & 0xF) * xv[p * 8 + j];
                        }
                        part[r] += float(ds[rowbase * IG + g]) * d + float(db[rowbase * IG + g]) * xs;
                    }
                }
            }
            for (uint r = 0; r < ROWS; ++r) acc[r] += wk * part[r];
        }
        for (uint r = 0; r < ROWS; ++r) {
            float s = simd_sum(acc[r]);
            if (lane == 0) y[(ulong)t * D + row0 + r] = T(s);
        }
        """)

// P025: the 8-bit g64 twin of q4_moe_gateup. SAME k-loop order, SAME simd_sum reduction, SAME fp32
// accumulation -- only the unpack changes (4 bytes per uint32, one scale group per 64 elements), so
// the arithmetic differs from the 4-bit body by nothing except the weights it is reading.
private let q8MoEGateUpKernel = MLXFast.metalKernel(
    name: "q8_moe_gateup", inputNames: ["x", "idx", "gw", "gs", "gb", "uw", "us", "ub"], outputNames: ["h"],
    source: """
        constexpr uint NSG = 2, VPT = 16, BLOCK = 512;   // ROWS from template
        uint tg = threadgroup_position_in_grid.x;        // t*K + k
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint t = tg / K, k = tg % K;
        int e = idx[tg];
        uint row0 = tile * (NSG * ROWS) + sg * ROWS;
        const uint DW = D / 4, DG = D / 64;
        float accg[ROWS], accu[ROWS];
        for (uint r = 0; r < ROWS; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }
        for (uint kb = 0; kb < D; kb += BLOCK) {
            uint off = kb + lane * VPT;
            float xv[VPT]; float xs = 0.0f;
            for (uint i = 0; i < VPT; ++i) { xv[i] = float(x[t * D + off + i]); xs += xv[i]; }
            uint g = off / 64;
            for (uint r = 0; r < ROWS; ++r) {
                ulong rowbase = ((ulong)e * I + row0 + r);
                const device uint32_t* gp = gw + rowbase * DW + off / 4;
                const device uint32_t* up = uw + rowbase * DW + off / 4;
                float dg = 0.0f, du = 0.0f;
                for (uint p = 0; p < 4; ++p) {
                    uint32_t wg = gp[p], wu = up[p];
                    for (uint j = 0; j < 4; ++j) {
                        float xj = xv[p * 4 + j];
                        dg += float((wg >> (8 * j)) & 0xFF) * xj;
                        du += float((wu >> (8 * j)) & 0xFF) * xj;
                    }
                }
                accg[r] += float(gs[rowbase * DG + g]) * dg + float(gb[rowbase * DG + g]) * xs;
                accu[r] += float(us[rowbase * DG + g]) * du + float(ub[rowbase * DG + g]) * xs;
            }
        }
        for (uint r = 0; r < ROWS; ++r) {
            float gsum = simd_sum(accg[r]);
            float usum = simd_sum(accu[r]);
            if (lane == 0) {
                float sil = gsum / (1.0f + exp(-gsum));
                h[((ulong)t * K + k) * I + row0 + r] = T(sil * usum);
            }
        }
        """)

private let q8MoEDownLaneKernel = MLXFast.metalKernel(
    name: "q8_moe_down_lane", inputNames: ["h", "idx", "dw", "ds", "db"], outputNames: ["yk"], source: Q4LabKernelSource.moe_down_lane_q8)

struct Q4ExpertBank { let w: MLXArray; let s: MLXArray; let b: MLXArray; var bits: Int = 4; var group: Int = 32 }

let q4MoEDownLaneThreads: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_DOWN_THREADS"] ?? "256") ?? 256
private let q4MoEDownLaneKernel = MLXFast.metalKernel(
    name: "q4_moe_down_lane", inputNames: ["h", "idx", "dw", "ds", "db"], outputNames: ["yk"], source: Q4LabKernelSource.moe_down_lane)

// MARK: P025 unit 6 -- the MULTI-ROW fused expert path (8-bit g64)
//
// The per-pair kernels above launch one threadgroup per (trunk row, expert slot) and re-read that
// expert's weights for every row that selected it. At the deployment point that is 32.1% waste:
// three trunk rows produce 30 (row, slot) pairs but only 20.35 DISTINCT experts, and at four rows
// 39.9 pairs give 24.44 distinct -- 38.7% (OBS-ENG-080). These bodies read each distinct expert
// ONCE and apply it to every row that selected it. The arithmetic per (row, expert) is byte-for-byte
// the same expression as the per-pair body -- same k-order, same per-group scale application, same
// fp32 accumulator, same simd_sum -- so the multi-row path is not a numerical change on top of the
// 8-bit one, it is the SAME numbers computed by fewer weight reads.
//
// The first-occurrence test replaces a sort: every threadgroup reads the same MR*K index buffer and
// the pair with the lowest index owns the expert; the rest return before touching a weight. That
// costs at most MR*K int loads out of cache and needs no host sync, which at decode is the whole
// point (OBS-ENG-250 already prices exposed host time at 0.394 ms/round).
private let q8MoEGateUpMRKernel = MLXFast.metalKernel(
    name: "q8_moe_gateup_mr", inputNames: ["x", "idx", "gw", "gs", "gb", "uw", "us", "ub"], outputNames: ["h"],
    source: """
        constexpr uint NSG = 2, VPT = 16, BLOCK = 512;   // ROWS, MR from template
        uint p = threadgroup_position_in_grid.x;         // the (row, slot) pair
        uint tile = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        int e = idx[p];
        // THE OWNERSHIP DECISION IS PER THREADGROUP, SO ONE THREAD MAKES IT. The first version had
        // every thread of every threadgroup scan the index buffer, and grid.y is the tile count:
        // 30 pairs x 160 tiles x 64 threads x ~60 loads = 18 M index loads per layer of pure
        // overhead, against 35 MB of useful weight reads. That is why the first multi-row build
        // measured SLOWER while reading 32% fewer weight bytes.
        uint tid = thread_index_in_threadgroup;
        threadgroup int tgFirst;
        threadgroup int tgSlot[MR];
        if (tid == 0) {
            int f = 1;
            for (uint q = 0; q < p; ++q) { if (idx[q] == e) { f = 0; break; } }
            tgFirst = f;
            for (uint r2 = 0; r2 < uint(MR); ++r2) {
                tgSlot[r2] = -1;
                for (uint k2 = 0; k2 < uint(K); ++k2) { if (idx[r2 * K + k2] == e) { tgSlot[r2] = int(k2); break; } }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tgFirst == 0) return;                                   // uniform across the threadgroup
        int slot[MR];
        for (uint r2 = 0; r2 < uint(MR); ++r2) slot[r2] = tgSlot[r2];
        uint row0 = tile * (NSG * ROWS) + sg * ROWS;
        const uint DW = D / 4, DG = D / 64;
        float accg[ROWS][MR], accu[ROWS][MR];
        for (uint r = 0; r < ROWS; ++r) for (uint r2 = 0; r2 < uint(MR); ++r2) { accg[r][r2] = 0.0f; accu[r][r2] = 0.0f; }
        // LOOP ORDER: the WEIGHT words are hoisted into registers and the ACTIVATION is re-read from
        // L1 per row, not the other way round. The first build hoisted x into xv[MR][VPT] -- 48
        // registers at MR=3 against the per-pair body's 16 -- and lost more to occupancy than it
        // saved in device traffic. x is 2560 floats per row and stays in cache; the expert weights
        // are the thing that must not be read twice.
        for (uint kb = 0; kb < D; kb += BLOCK) {
            uint off = kb + lane * VPT;
            uint g = off / 64;
            for (uint r = 0; r < ROWS; ++r) {
                ulong rowbase = ((ulong)e * I + row0 + r);
                const device uint32_t* gp = gw + rowbase * DW + off / 4;
                const device uint32_t* upp = uw + rowbase * DW + off / 4;
                uint32_t wgv[4], wuv[4];
                for (uint pw = 0; pw < 4; ++pw) { wgv[pw] = gp[pw]; wuv[pw] = upp[pw]; }
                float gsc = float(gs[rowbase * DG + g]), gbi = float(gb[rowbase * DG + g]);
                float usc = float(us[rowbase * DG + g]), ubi = float(ub[rowbase * DG + g]);
                for (uint r2 = 0; r2 < uint(MR); ++r2) {
                    if (slot[r2] < 0) continue;                 // uniform across the threadgroup
                    const device T* xr = x + r2 * D + off;
                    float dg = 0.0f, du = 0.0f, xs = 0.0f;
                    for (uint pw = 0; pw < 4; ++pw) {
                        uint32_t wg = wgv[pw], wu = wuv[pw];
                        for (uint j = 0; j < 4; ++j) {
                            float xj = float(xr[pw * 4 + j]);
                            xs += xj;
                            dg += float((wg >> (8 * j)) & 0xFF) * xj;
                            du += float((wu >> (8 * j)) & 0xFF) * xj;
                        }
                    }
                    accg[r][r2] += gsc * dg + gbi * xs;
                    accu[r][r2] += usc * du + ubi * xs;
                }
            }
        }
        for (uint r = 0; r < ROWS; ++r) {
            for (uint r2 = 0; r2 < uint(MR); ++r2) {
                if (slot[r2] < 0) continue;
                float gsum = simd_sum(accg[r][r2]);
                float usum = simd_sum(accu[r][r2]);
                if (lane == 0) {
                    float sil = gsum / (1.0f + exp(-gsum));
                    h[((ulong)(r2 * K + uint(slot[r2]))) * I + row0 + r] = T(sil * usum);
                }
            }
        }
        """)

private let q8MoEDownLaneMRKernel = MLXFast.metalKernel(
    name: "q8_moe_down_lane_mr", inputNames: ["h", "idx", "dw", "ds", "db"], outputNames: ["yk"], source: Q4LabKernelSource.moe_down_lane_q8mr)

/// P026 unit 1: the ROWS-hoisted multi-row down lane. `moe_down_lane_q8mr` cut 32.1% of the weight
/// bytes and LOST 2.31 ms/round at every TGS in {128,256,512,1024} -- and the gap is SMALLEST at the
/// shipped TGS=256 and WORST at 1024, where threadgroup pressure is lowest, so it is not occupancy.
/// It is the innermost trip count: 20 distinct x D x I x MR against 30 pairs x D x I, i.e. DOUBLE the
/// iterations for a third fewer bytes. This body hoists the staged x across ROWS output rows, which is
/// the same ratio that makes the trade pay in `q8_moe_gate_up_mr`.
private let q8MoEDownLaneMR2Kernel = MLXFast.metalKernel(
    name: "q8_moe_down_lane_mr2", inputNames: ["h", "idx", "dw", "ds", "db"], outputNames: ["yk"], source: Q4LabKernelSource.moe_down_lane_q8mr2)

/// P094: rows per simdgroup for the PER-PAIR gate/up body (T = 1, and the 4-bit path). Bit-identical
/// at every value -- each output row has its own accumulator, the same k-order and the same simd_sum
/// -- and it only changes how many threadgroups the 35 MB read is spread over: at ROWS=4 a decode
/// token launches 10 x 80 = 800 threadgroups of 64 threads on 80 cores and reads at 350 GB/s (59% of
/// the size roof, kernelbench moe-decode T=1); ROWS=1 launches 3200.
let q8MoEGateUpRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_GATEUP_ROWS"] ?? "1") ?? 1   // P094: 1 is the default -- 512 serial +2.2%, 8192 +1.7%, tokens identical (OBS-ENG-179)
/// P025 unit 6: rows-per-simdgroup for the multi-row gate/up body. Lower than the per-pair kernel's 4
/// because the accumulator is now ROWS x MR: sweepable, because register pressure is exactly the kind
/// of thing that must be measured rather than reasoned about.
/// P106 H51b: the widest row count (tokens of one MoE call) that takes the fused per-pair expert kernels instead
/// of SwitchGLU's sorted gather_qmm. 8 = the shipped cut (the multi-row kernels' bound); above 8 the per-pair
/// kernels run (they have no row limit). A batched MTP round is B x (K+1) rows: 20 at B5, 32 at B8.
let q4MoEFusedMaxRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_FUSED_MAX_ROWS"] ?? "8") ?? 8
let q8MoEMRRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_MR_ROWS"] ?? "2") ?? 2
nonisolated(unsafe) var q8MoEMRAnnounced = false
/// P025 unit 6: which of the two kernels takes the multi-row form. 1 = gate/up, 2 = down lane,
/// 3 = both. They are separable because the down lane pays for it in THREADGROUP MEMORY -- hs[MR][I]
/// is 7.7 kB at MR=3 against 2.6 -- while the gate/up pays only in registers.
let q8MoEMRParts: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_MR_PARTS"] ?? "1") ?? 1
/// P026 unit 1: output rows per thread in the multi-row down lane. 0 selects the original
/// `moe_down_lane_q8mr` body (float staging, one row per thread); >= 1 selects `moe_down_lane_q8mr2`
/// with that many rows, where ROWS=1 differs from the original ONLY in staging `hs` as bf16 -- which
/// lets the staging dtype and the hoist be read apart.
let q8MoEDownRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MOE_DOWN_ROWS"] ?? "0") ?? 0

/// x [T, D], idx [T, K] -> yk [T, K, D] = down_{idx}(silu(gate x) * (up x))  (unweighted; combine with q4MoECombine)
func q4MoEExpertsK(x: MLXArray, idx: MLXArray, gate: Q4ExpertBank, up: Q4ExpertBank, down: Q4ExpertBank,
                   K: Int, D: Int, I: Int) -> MLXArray {
    let h = q4MoEGateUpShipped(x: x, idx: idx, gate: gate, up: up, K: K, D: D, I: I)
    return q4MoEDownShipped(h: h, idx: idx, down: down, K: K, D: D, I: I, dtype: x.dtype)
}

/// P094: the gate/up half of q4MoEExpertsK, split out so `engine kernelbench moe-decode` can time the
/// two halves apart. The dispatch decisions are the ones q4MoEExpertsK made in one body.
func q4MoEGateUpShipped(x: MLXArray, idx: MLXArray, gate: Q4ExpertBank, up: Q4ExpertBank, K: Int, D: Int, I: Int) -> MLXArray {
    let T = x.dim(0)
    let NSG = 2
    let eight = gate.bits == 8
    // P025 unit 6: the multi-row path. Only 8-bit (the operating artifact), only when there is
    // something to share (T > 1), and only up to MR = 8 so the compile-time accumulator stays bounded.
    if eight, Q4Fused.moeExperts8MR, T > 1, T <= 8 {
        let ROWS = q8MoEMRRows
        let parts = q8MoEMRParts
        // BOUND-ENG-007: the branch says so itself, once, before any timing is read. The load-time
        // fusionWitness() cannot cover this one -- it evaluates at rows = 1, where MR does not apply.
        if !q8MoEMRAnnounced {
            q8MoEMRAnnounced = true
            FileHandle.standardError.write("qwen4_exp: multi-row expert path LIVE -- MR=\(T) K=\(K) ROWS=\(ROWS) parts=\(parts)\n".data(using: .utf8)!)
        }
        let h = (parts & 1 != 0 ? q8MoEGateUpMRKernel : q8MoEGateUpKernel)(
            [x, idx, gate.w, gate.s, gate.b, up.w, up.s, up.b],
            template: (parts & 1 != 0
                       ? [("T", x.dtype), ("K", K), ("D", D), ("I", I), ("ROWS", ROWS), ("MR", T)]
                       : [("T", x.dtype), ("K", K), ("D", D), ("I", I), ("ROWS", 4)]),
            grid: (T * K * NSG * 32, I / (NSG * (parts & 1 != 0 ? ROWS : 4)), 1), threadGroup: (NSG * 32, 1, 1),
            outputShapes: [[T, K, I]], outputDTypes: [x.dtype])[0]
        return h
    }
    let ROWS = q8MoEGateUpRows
    return (eight ? q8MoEGateUpKernel : q4MoEGateUpKernel)(
        [x, idx, gate.w, gate.s, gate.b, up.w, up.s, up.b],
        template: [("T", x.dtype), ("K", K), ("D", D), ("I", I), ("ROWS", ROWS)],
        grid: (T * K * NSG * 32, I / (NSG * ROWS), 1), threadGroup: (NSG * 32, 1, 1),
        outputShapes: [[T, K, I]], outputDTypes: [x.dtype])[0]
}

/// P094: the down half of q4MoEExpertsK (see q4MoEGateUpShipped).
func q4MoEDownShipped(h: MLXArray, idx: MLXArray, down: Q4ExpertBank, K: Int, D: Int, I: Int, dtype: DType) -> MLXArray {
    let T = h.dim(0)
    let eight = down.bits == 8
    if eight, Q4Fused.moeExperts8MR, T > 1, T <= 8 {
        let parts = q8MoEMRParts
        let TGS = q4MoEDownLaneThreads
        if parts & 2 != 0, q8MoEDownRows >= 1 {
            let DR = q8MoEDownRows
            let per = TGS * DR                       // rows covered by one threadgroup
            return q8MoEDownLaneMR2Kernel(
                [h, idx, down.w, down.s, down.b],
                template: [("T", dtype), ("K", K), ("D", D), ("I", I), ("TGS", TGS), ("MR", T), ("ROWS", DR)],
                grid: (T * K * TGS, (D + per - 1) / per, 1), threadGroup: (TGS, 1, 1),
                outputShapes: [[T, K, D]], outputDTypes: [dtype])[0]
        }
        return (parts & 2 != 0 ? q8MoEDownLaneMRKernel : q8MoEDownLaneKernel)(
            [h, idx, down.w, down.s, down.b],
            template: (parts & 2 != 0
                       ? [("T", dtype), ("K", K), ("D", D), ("I", I), ("TGS", TGS), ("MR", T)]
                       : [("T", dtype), ("K", K), ("D", D), ("I", I), ("TGS", TGS)]),
            grid: (T * K * TGS, (D + TGS - 1) / TGS, 1), threadGroup: (TGS, 1, 1),
            outputShapes: [[T, K, D]], outputDTypes: [dtype])[0]
    }
    let TGS = q4MoEDownLaneThreads
    return (eight ? q8MoEDownLaneKernel : q4MoEDownLaneKernel)(
        [h, idx, down.w, down.s, down.b],
        template: [("T", dtype), ("K", K), ("D", D), ("I", I), ("TGS", TGS)],
        grid: (T * K * TGS, (D + TGS - 1) / TGS, 1), threadGroup: (TGS, 1, 1),
        outputShapes: [[T, K, D]], outputDTypes: [dtype])[0]
}



// MARK: P095 -- the QSA gather kernels exposed for `engine kernelbench --case gather`
public enum Q4GatherBench {
    /// P097: the partitioned, GQA-shared ragged gather with explicit (parts, hpg); parts 0 = the single kernel
    public static func raggedPart(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, rowOffsets: [Int], rowNBlocks: [Int], ratio: Int, scale: Float, parts: Int, hpg: Int) -> MLXArray {
        q4QSAGatherRagged(q: q, keys: keys, values: values, top: top, rowOffsets: rowOffsets, rowNBlocks: rowNBlocks, ratio: ratio, scale: scale, parts: parts, hpg: hpg)
    }
    /// P098: the verify block's gather (1 < S <= 16) with an explicit heads-per-simdgroup (0 = the shipped single-head form)
    public static func serialVerify(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, nBlocks: Int, kvLen: Int, offset: Int, ratio: Int, scale: Float, hpg: Int) -> MLXArray {
        q4QSAGather(q: q, keys: keys, values: values, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: ratio, scale: scale, hpgVerify: hpg)
    }
    /// P099: the single S = 1 kernel with the pipelined chain (pf 0 = the shipped kernel); serial and ragged forms
    public static func serialPipe(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, nBlocks: Int, kvLen: Int, offset: Int, ratio: Int, scale: Float, pf: Int) -> MLXArray {
        q4QSAGather(q: q, keys: keys, values: values, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: ratio, scale: scale, pf: pf)
    }
    public static func raggedPipe(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, rowOffsets: [Int], rowNBlocks: [Int], ratio: Int, scale: Float, pf: Int) -> MLXArray {
        q4QSAGatherRagged(q: q, keys: keys, values: values, top: top, rowOffsets: rowOffsets, rowNBlocks: rowNBlocks, ratio: ratio, scale: scale, parts: 0, hpg: 1, pf: pf)
    }
    public static var raggedKnobs: String { "RAGGED_PARTS=\(q4QSARaggedParts) HPG=\(q4QSARaggedHpg) MIN_ROWS=\(q4QSARaggedMinRows)" }
    public static func ragged(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, rowOffsets: [Int], rowNBlocks: [Int], ratio: Int, scale: Float) -> MLXArray {
        q4QSAGatherRagged(q: q, keys: keys, values: values, top: top, rowOffsets: rowOffsets, rowNBlocks: rowNBlocks, ratio: ratio, scale: scale)
    }
    public static func serial(q: MLXArray, keys: MLXArray, values: MLXArray, top: MLXArray, nBlocks: Int, kvLen: Int, offset: Int, ratio: Int, scale: Float) -> MLXArray {
        q4QSAGather(q: q, keys: keys, values: values, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: ratio, scale: scale)
    }
}

// MARK: P094 -- the shipped fused expert path, exposed for `engine kernelbench moe-decode`
//
// Every dispatch decision above is reproduced here by CALLING the same function; nothing is
// re-implemented. The split lets gate/up and the down lane be timed apart, which q4MoEExpertsK's
// single return value cannot. `Q4ExpertBank` is internal, so the bank fields come in flat.
/// P096 (INST-ENG-042): the mixer's per-row bodies exposed for `engine kernelbench --case proj|gdn|topk|idxscore`.
/// Every entry calls the SHIPPED body; the bench compares arms against these, never against a re-implementation.
public enum Q4MixerBench {
    /// the selection as the engine dispatches it (two-phase when q4TopKParts allows, else one group per row)
    public static func topK(scores: MLXArray, k: Int) -> MLXArray { q4TopK(scores: scores, k: k) }
    public static func topKParts(rows: Int, n: Int, k: Int) -> Int? { q4TopKParts(rows: rows, n: n, k: k) }
    /// one 1024-thread group per row, 4 radix rounds + 2 compaction passes (the form every row takes below n = 2048 today)
    public static func topKSingle(scores: MLXArray, k K: Int) -> MLXArray {
        let B = scores.dim(0), S = scores.dim(1), n = scores.dim(2), TG = q4TopKTG
        return q4TopKKernel([scores.reshaped(B * S, n), MLXArray([Int32(n), Int32(K)])], template: [("K", K), ("TG", TG)],
                            grid: (B * S * TG, 1, 1), threadGroup: (TG, 1, 1), outputShapes: [[B * S, K]], outputDTypes: [.int32])[0].reshaped(B, S, K)
    }
    /// the two-phase split (P parts per row, then a merge), as three dispatches or as the one-dispatch form
    public static func topKTwoPhase(scores: MLXArray, k: Int, parts: Int, oneDispatch: Bool) -> MLXArray {
        q4TopKTwoPhase(scores: scores, k: k, parts: parts, oneDispatch: oneDispatch)
    }
    public static var topKPartsMinN: Int { q4TopKPartsMinN }
    public static var topKKnobs: String { "PARTS=\(q4TopKPartsEnv) TG=\(q4TopKTG) 1D=\(q4TopKOneDispatch) ROUNDS=\(q4TopKRounds)" }
    public static func idxReluSum(_ raw: MLXArray, scale: Float, visibility: (kvLen: Int, ratio: Int)?) -> MLXArray {
        q4IdxReluSum(raw, scale: scale, visibility: visibility)
    }
    public static func idxScoreFused(q: MLXArray, pooled: MLXArray, scale: Float, visibility: (kvLen: Int, ratio: Int)?, mi: Int, ni: Int, sg: Int) -> MLXArray {
        q4IdxScore(q: q, pooled: pooled, scale: scale, visibility: visibility, mi: mi, ni: ni, sg: sg)
    }
    public static var idxScoreNI: Int { q4IdxScoreNI }
    /// Applied scalar settings needed to replay the S1 score/selection graph faithfully.
    /// This is plain metadata; no model state or MLX array escapes through it.
    public static var idxScoreReplayConfiguration: [String: String] {
        ["reduction_fused": String(q4IdxFused),
         "topk_select": String(Qwen4ExpQSAIndexer.topKSelect),
         "topk_parts": String(q4TopKPartsEnv), "topk_min_n": String(q4TopKPartsMinN),
         "topk_tg": String(q4TopKTG), "topk_rounds": String(q4TopKRounds),
         "topk_1d": String(q4TopKOneDispatch), "topk_fold_offset": String(q4TopKFoldOffset),
         "topk_reg": String(q4TopKReg), "topk_regmax": String(q4TopKRegMax),
         "topk_hist": String(q4TopKHistWitness), "score_ni": String(q4IdxScoreNI)]
    }
    public static func gdnConvNorm(qkvz: MLXArray, convState: MLXArray, convW: MLXArray, convDim: Int, kernel: Int, headDim: Int, keyDim: Int,
                                   scaleQ: Float, scaleK: Float, eps: Float) -> (MLXArray, MLXArray) {
        q4GDNConvNorm(qkvz: qkvz, convState: convState, convW: convW, convDim: convDim, kernel: kernel, headDim: headDim, keyDim: keyDim,
                      scaleQ: scaleQ, scaleK: scaleK, eps: eps)
    }
    public static func gatedNormSG(_ x: MLXArray, weight: MLXArray, gate: MLXArray, headDim: Int, eps: Float, sigmoidGate: Bool, outDType: DType) -> MLXArray {
        q4GatedNormSG(x, weight: weight, gate: gate, headDim: headDim, eps: eps, sigmoidGate: sigmoidGate, outDType: outDType)
    }
    public static func gdnGate(ba: MLXArray, aLog: MLXArray, dtBias: MLXArray, nV: Int) -> (MLXArray, MLXArray) { q4GDNGate(ba: ba, aLog: aLog, dtBias: dtBias, nV: nV) }
    public static func headNormScale(_ x: MLXArray, headDim: Int, scale: Float, eps: Float) -> MLXArray { q4HeadNormScale(x, headDim: headDim, scale: scale, eps: eps) }
    public static func silu(_ x: MLXArray) -> MLXArray { q4Silu(x) }
}

public struct Q4MoEDecodeBench {
    /// (h [T,K,I], yk [T,K,D]) for the SHIPPED path at this row count; `bits` 8 selects the g64 twin.
    public static func run(x: MLXArray, idx: MLXArray,
                           gate: (MLXArray, MLXArray, MLXArray), up: (MLXArray, MLXArray, MLXArray),
                           down: (MLXArray, MLXArray, MLXArray), bits: Int, K: Int, D: Int, I: Int) -> (h: MLXArray, yk: MLXArray) {
        let g = Q4ExpertBank(w: gate.0, s: gate.1, b: gate.2, bits: bits, group: bits == 8 ? 64 : 32)
        let u = Q4ExpertBank(w: up.0, s: up.1, b: up.2, bits: bits, group: bits == 8 ? 64 : 32)
        let d = Q4ExpertBank(w: down.0, s: down.1, b: down.2, bits: bits, group: bits == 8 ? 64 : 32)
        let h = q4MoEGateUpShipped(x: x, idx: idx, gate: g, up: u, K: K, D: D, I: I)
        let yk = q4MoEDownShipped(h: h, idx: idx, down: d, K: K, D: D, I: I, dtype: x.dtype)
        return (h, yk)
    }
    public static var fuseMask: Int { Q4Fused.mask }
    /// which body the shipped path takes at T rows, for the bench's own witness line (BOUND-ENG-007)
    public static func witness(T: Int, bits: Int) -> String {
        if bits == 8, Q4Fused.moeExperts8MR, T > 1, T <= 8 {
            return "gate/up \(q8MoEMRParts & 1 != 0 ? "q8_moe_gateup_mr ROWS=\(q8MoEMRRows) MR=\(T)" : "q8_moe_gateup ROWS=4"); down \(q8MoEMRParts & 2 != 0 ? "multi-row" : "q8_moe_down_lane per-pair") TGS=\(q4MoEDownLaneThreads)"
        }
        return "gate/up \(bits == 8 ? "q8" : "q4")_moe_gateup ROWS=\(q8MoEGateUpRows) per-pair; down \(bits == 8 ? "q8" : "q4")_moe_down_lane per-pair TGS=\(q4MoEDownLaneThreads)"
    }
}

/// x [T, D], idx [T, K] int32, wts [T, K] -> y [T, D] = sum_k wts * down(silu(gate x) * (up x))
func q4MoEExperts(x: MLXArray, idx: MLXArray, wts: MLXArray, gate: Q4ExpertBank, up: Q4ExpertBank, down: Q4ExpertBank,
                  K: Int, D: Int, I: Int) -> MLXArray {
    // P025: this variant (no lane-down kernel) has no 8-bit twin. The caller must not reach it with
    // 8-bit banks; `forwardUncompiled` checks moeDownLane first, and expertBanks() refuses 8-bit
    // unless the fused path is licensed. A precondition beats a silently wrong unpack.
    precondition(gate.bits == 4, "q4MoEExperts: 4-bit banks only (8-bit goes through q4MoEExpertsK)")
    let T = x.dim(0)
    let NSG = 2, ROWS = 4
    let h = q4MoEGateUpKernel(
        [x, idx, gate.w, gate.s, gate.b, up.w, up.s, up.b],
        template: [("T", x.dtype), ("K", K), ("D", D), ("I", I), ("ROWS", ROWS)],
        grid: (T * K * NSG * 32, I / (NSG * ROWS), 1), threadGroup: (NSG * 32, 1, 1),
        outputShapes: [[T, K, I]], outputDTypes: [x.dtype])[0]
    return q4MoEDownKernel(
        [h, idx, wts, down.w, down.s, down.b],
        template: [("T", x.dtype), ("K", K), ("D", D), ("I", I)],
        grid: (T * NSG * 32, D / (NSG * ROWS), 1), threadGroup: (NSG * 32, 1, 1),
        outputShapes: [[T, D]], outputDTypes: [x.dtype])[0]
}


// MARK: - fused hyper-connection chain (bodies in Kernels/lab/*.metal, embedded by tools/gen_lab_kernels.py)
let q4HCNormThreads: Int = Int(ProcessInfo.processInfo.environment["ENGINE_HC_NORM_THREADS"] ?? "512") ?? 512   // threads per (row, group) norm threadgroup
private let q4HCNormKernel = MLXFast.metalKernel(
    name: "q4_hc_norm", inputNames: ["x", "w1", "epsv"], outputNames: ["y"], source: Q4LabKernelSource.hc_norm)
private let q4HCUpMixKernel = MLXFast.metalKernel(
    name: "q4_hc_up_mix", inputNames: ["down", "normed", "uw", "us", "ub", "winj", "hcf"], outputNames: ["mixed", "injp"], source: Q4LabKernelSource.hc_up_mix)
private let q4HCUpMixBF16Kernel = MLXFast.metalKernel(
    name: "q4_hc_up_mix_bf16", inputNames: ["down", "normed", "uw", "winj", "hcf"], outputNames: ["mixed", "injp"], source: Q4LabKernelSource.hc_up_mix_bf16)
private let q4HCUpMixBF16SGKernel = MLXFast.metalKernel(
    name: "q4_hc_up_mix_bf16_sg", inputNames: ["down", "normed", "uw", "winj", "hcf"], outputNames: ["mixed", "injp"], source: Q4LabKernelSource.hc_up_mix_bf16_sg)
private let q4HCInjectPKernel = MLXFast.metalKernel(
    name: "q4_hc_inject_p", inputNames: ["hyper", "x", "injp"], outputNames: ["out"],
    source: """
        uint t = thread_position_in_grid.x;         // row*H*D + h*D + j
        uint row = t / (H * D), rem = t % (H * D), h = rem / D, j = rem % D;
        float g = 0.0f;
        for (uint p = 0; p < P; ++p) g += injp[((ulong)row * P + p) * H + h];
        g /= float(H);
        float w = 2.0f / (1.0f + exp(-g));
        out[t] = T(float(hyper[t]) + w * float(x[(ulong)row * D + j]));
        """)

/// grouped zero-centred RMS norm over G groups of D: one threadgroup (256) per (row, group)
func q4HCNorm(_ x: MLXArray, onePlusW: MLXArray, groups G: Int, groupSize D: Int, eps: Float) -> MLXArray {
    let rows = x.size / (G * D)
    let TGS = q4HCNormThreads
    return q4HCNormKernel([x, onePlusW, q4Const(eps)], template: [("T", x.dtype), ("D", D), ("G", G), ("TGS", TGS)],
                          grid: (rows * G * TGS, 1, 1), threadGroup: (TGS, 1, 1),
                          outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
}

/// (mixed (lead..., D), inject partials (rows, H*D/256, H) float32); `down` is [rows, R] or, with downSplits > 1, [rows, downSplits, R] split-K partials
func q4HCUpMix(down: MLXArray, normed: MLXArray, up: QuantizedLinear, winj: MLXArray, hc H: Int, d D: Int, downSplits DS: Int = 1) -> (MLXArray, MLXArray) {
    let rows = normed.size / (H * D)
    let lead = Array(normed.shape.dropLast())
    let R = up.weight.dim(0) == H * D ? up.scales.dim(1) * up.groupSize : 0
    precondition(up.bits == 8 && up.groupSize == 64 && R > 0 && (H * D) % 256 == 0, "q4HCUpMix: needs 8-bit g64 mix-up weights")
    let P = H * D / 256
    let r = q4HCUpMixKernel([down, normed, up.weight, up.scales, up.biases!, winj, q4Const(Float(H))],
                            template: [("T", normed.dtype), ("H", H), ("D", D), ("R", R), ("GS", 64), ("TGS", 256), ("DS", DS)],
                            grid: (rows * H * D, 1, 1), threadGroup: (256, 1, 1),
                            outputShapes: [lead + [D], [rows, P, H]], outputDTypes: [normed.dtype, .float32])
    return (r[0], r[1])
}

/// The bf16 twin of `q4HCUpMix`: same function, dense mix-up weights. Lets a FULL-PRECISION
/// checkpoint keep the fused hyper-connection chain, which is worth +2.57 ms/token (OBS-ENG-066).
func bf16HCUpMix(down: MLXArray, normed: MLXArray, up: Linear, winj: MLXArray, hc H: Int, d D: Int, downSplits DS: Int = 1) -> (MLXArray, MLXArray) {
    let rows = normed.size / (H * D)
    let lead = Array(normed.shape.dropLast())
    let R = up.weight.dim(1)
    precondition(up.weight.dim(0) == H * D && R % 4 == 0 && (H * D) % 256 == 0, "bf16HCUpMix: dense [H*D, R] mix-up weights")
    // TWO BODIES, one env switch: thread-per-output (uncoalesced, 640 B apart per lane) and
    // simdgroup-per-column (256 contiguous bytes per step). Which wins is measured, not assumed.
    if Q4Fused.hcDenseSG && (H * R / 4) % 32 == 0 && H <= 8 {
        let P = D * 32 / 256
        let r = q4HCUpMixBF16SGKernel([down, normed, up.weight, winj, q4Const(Float(H))],
                                      template: [("T", normed.dtype), ("H", H), ("D", D), ("R", R), ("TGS", 256), ("DS", DS)],
                                      grid: (rows * D * 32, 1, 1), threadGroup: (256, 1, 1),
                                      outputShapes: [lead + [D], [rows, P, H]], outputDTypes: [normed.dtype, .float32])
        return (r[0], r[1])
    }
    let P = H * D / 256
    let r = q4HCUpMixBF16Kernel([down, normed, up.weight, winj, q4Const(Float(H))],
                                template: [("T", normed.dtype), ("H", H), ("D", D), ("R", R), ("TGS", 256), ("DS", DS)],
                                grid: (rows * H * D, 1, 1), threadGroup: (256, 1, 1),
                                outputShapes: [lead + [D], [rows, P, H]], outputDTypes: [normed.dtype, .float32])
    return (r[0], r[1])
}

func q4HCInjectP(hyper: MLXArray, x: MLXArray, injp: MLXArray, hc H: Int, d D: Int) -> MLXArray {
    let P = injp.dim(1)
    return q4HCInjectPKernel([hyper, x, injp], template: [("T", hyper.dtype), ("H", H), ("D", D), ("P", P)],
                             grid: (hyper.size, 1, 1), threadGroup: (256, 1, 1),
                             outputShapes: [hyper.shape], outputDTypes: [hyper.dtype])[0]
}


// MARK: router top-K + softmax (body in Kernels/lab/moe_topk.metal)
private let q4MoETopKKernel = MLXFast.metalKernel(
    name: "q4_moe_topk", inputNames: ["logits"], outputNames: ["idx", "w", "sg"], source: Q4LabKernelSource.moe_topk)

/// logits (lead..., LD) with LD >= E (+1 when the shared-expert gate logit is column E) -> (idx uint32 (lead...,K), w (lead...,K), sg (lead...,1))
func q4MoETopK(logits: MLXArray, experts E: Int, topK K: Int, sharedGate: Bool) -> (MLXArray, MLXArray, MLXArray) {
    let LD = logits.dim(-1)
    let rows = logits.size / LD
    let lead = Array(logits.shape.dropLast())
    let r = q4MoETopKKernel([logits], template: [("T", logits.dtype), ("E", E), ("K", K), ("LD", LD), ("SG", sharedGate ? 1 : 0)],
                            grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
                            outputShapes: [lead + [K], lead + [K], lead + [1]], outputDTypes: [.uint32, logits.dtype, logits.dtype])
    return (r[0], r[1], r[2])
}


// MARK: GDN gate (g, beta) from the merged b|a projection (body in Kernels/lab/gdn_gate.metal)
private let q4GDNGateKernel = MLXFast.metalKernel(
    name: "q4_gdn_gate", inputNames: ["ba", "aLog", "dtBias"], outputNames: ["g", "beta"], source: Q4LabKernelSource.gdn_gate)

/// ba (B,S,2*NV) -> (g (B,S,NV) float32, beta (B,S,NV) float32)
func q4GDNGate(ba: MLXArray, aLog: MLXArray, dtBias: MLXArray, nV NV: Int) -> (MLXArray, MLXArray) {
    let rows = ba.size / (2 * NV)
    let lead = Array(ba.shape.dropLast())
    let r = q4GDNGateKernel([ba, aLog.asType(ba.dtype), dtBias.asType(ba.dtype)], template: [("T", ba.dtype), ("NV", NV)],
                            grid: (rows * NV, 1, 1), threadGroup: (min(256, rows * NV), 1, 1),
                            outputShapes: [lead + [NV], lead + [NV]], outputDTypes: [.float32, .float32])
    return (r[0], r[1])
}


// MARK: combine + next grouped norm (body in Kernels/lab/hc_inject_norm.metal)
private let q4HCInjectNormKernel = MLXFast.metalKernel(
    name: "q4_hc_inject_norm", inputNames: ["hyper", "x", "injp", "w1", "epsv"], outputNames: ["out", "normed"], source: Q4LabKernelSource.hc_inject_norm)

/// -> (residual out, normed for the next HC whose (1+w) is `nextOnePlusW`)
func q4HCInjectNorm(hyper: MLXArray, x: MLXArray, injp: MLXArray, nextOnePlusW: MLXArray, eps: Float, hc H: Int, d D: Int) -> (MLXArray, MLXArray) {
    let rows = hyper.size / (H * D)
    let P = injp.dim(1)
    let TGS = q4HCNormThreads
    let r = q4HCInjectNormKernel([hyper, x, injp, nextOnePlusW, q4Const(eps)], template: [("T", hyper.dtype), ("H", H), ("D", D), ("P", P), ("TGS", TGS)],
                                 grid: (rows * H * TGS, 1, 1), threadGroup: (TGS, 1, 1),
                                 outputShapes: [hyper.shape, hyper.shape], outputDTypes: [hyper.dtype, hyper.dtype])
    return (r[0], r[1])
}


// MARK: split-K 8-bit projection for a few rows (verify blocks); body in Kernels/lab/qmv_splitk.metal
private let q8SplitKKernel = MLXFast.metalKernel(
    name: "q8_qmv_splitk", inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: Q4LabKernelSource.qmv_splitk, atomicOutputs: true)

/// `lin(x)` through the split-K kernel when it applies (8-bit g64, 2...8 rows, K % 512 == 0), else the module itself
private let q8SplitKPartKernel = MLXFast.metalKernel(
    name: "q8_qmv_splitk_part", inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: Q4LabKernelSource.qmv_splitk_part)

/// split-K 8-bit g64 projection WITHOUT atomics: returns float32 partials [rows, SPLIT, N] for a consumer that sums them
/// (deterministic; no zero-fill dispatch). nil when the projection does not qualify.
func q8ProjSplitKPartials(_ lin: Linear, _ x: MLXArray, split SPLIT: Int) -> MLXArray? {
    let K = x.dim(-1)
    let rows = x.size / K
    guard rows >= 1, rows <= 8, K % (SPLIT * 32 * 16) == 0, let q = lin as? QuantizedLinear, q.bits == 8, q.groupSize == 64, let b = q.biases,
          q.weight.dim(1) * 4 == K else { return nil }
    let N = q.weight.dim(0)
    let xf = x.reshaped(rows, K)
    return q8SplitKPartKernel([xf, q.weight, q.scales, b], template: [("T", x.dtype), ("M", rows), ("N", N), ("K", K), ("GS", 64), ("SPLIT", SPLIT)],
                              grid: (N * SPLIT * 32, 1, 1), threadGroup: (256, 1, 1),
                              outputShapes: [[rows, SPLIT, N]], outputDTypes: [.float32])[0]
}

/// split-K DENSE (bf16) projection: float32 partials [rows, SPLIT, N] for a consumer that sums them.
/// The bf16 twin of `q8ProjSplitKPartials`; nil when the projection does not qualify.
private let bf16SplitKPartKernel = MLXFast.metalKernel(
    name: "bf16_mv_splitk_part", inputNames: ["x", "w"], outputNames: ["y"], source: Q4LabKernelSource.bf16mv_splitk_part)

func bf16ProjSplitKPartials(_ lin: Linear, _ x: MLXArray, split SPLIT: Int) -> MLXArray? {
    let K = x.dim(-1)
    let rows = x.size / K
    guard rows >= 1, rows <= 8, lin is QuantizedLinear == false, lin.weight.dim(1) == K,
          lin.weight.dtype == x.dtype, K % (SPLIT * 32 * 4) == 0 else { return nil }
    let N = lin.weight.dim(0)
    let xf = x.reshaped(rows, K)
    return bf16SplitKPartKernel([xf, lin.weight],
                                template: [("T", x.dtype), ("M", rows), ("N", N), ("K", K), ("SPLIT", SPLIT)],
                                grid: (N * SPLIT * 32, 1, 1), threadGroup: (256, 1, 1),
                                outputShapes: [[rows, SPLIT, N]], outputDTypes: [.float32])[0]
}

/// P029 unit 8. The split-K qmv was written for 2..8 rows because MLX's qmm is "2-3x off bandwidth
/// there", and SERIAL decode is one row, so it was never asked. P029 unit 6 measured MLX's own qmv
/// IN SITU at one row on two projections of the same layer -- qkvi 37.93 MB at 286 GB/s and o_proj
/// 16.71 MB at 401 -- i.e. the BIGGER read is the SLOWER one, which is backwards for OBS-ENG-063's
/// size-dependent roof, and the lab reads the qkvi shape at 498 GB/s on a cold 16-deep chain.
/// ENGINE_QMV_SPLITK_1ROW lets the same kernel answer at M = 1.
let q8SplitK1Row: Bool = ProcessInfo.processInfo.environment["ENGINE_QMV_SPLITK_1ROW"] != nil

/// P095 U3-B: MLX routes an 8-bit quantized_matmul to the vector kernel (`qmv_fast`, one grid column per
/// row, the weight streamed once PER ROW) below `vector_limit` = 12 rows on this die, and to the tiled
/// `qmm_t_splitk` (weight read once) from 12. For a 675 MB head at B = 8 that is seven extra passes
/// over DRAM. Pad the rows to 12 so the tiled kernel takes them, and slice the answer back. A
/// different reduction order than qmv, so only ever at rows >= `minRows` >= 2 (B = 1 untouched).
let q8DenseQmmMin: Int = Int(ProcessInfo.processInfo.environment["ENGINE_DENSE_QMM_MIN"] ?? "0") ?? 0
func q8ProjPadded(_ lin: Linear, _ x: MLXArray, minRows: Int) -> MLXArray {
    let K = x.dim(-1)
    let rows = x.size / K
    guard minRows >= 2, rows >= minRows, rows < 12, let q = lin as? QuantizedLinear else { return lin(x) }
    let xf = concatenated([x.reshaped(rows, K), MLXArray.zeros([12 - rows, K], dtype: x.dtype)], axis: 0)
    let y = q(xf)[0 ..< rows]
    return y.reshaped(Array(x.shape.dropLast()) + [q.weight.dim(0)])
}

func q8Proj(_ lin: Linear, _ x: MLXArray, minRows: Int = 2, force: Bool = false) -> MLXArray {
    let K = x.dim(-1)
    let rows = x.size / K
    if q8DenseQmmMin >= 2, rows >= q8DenseQmmMin, rows < 12, !Q4Fused.splitKM { return q8ProjPadded(lin, x, minRows: q8DenseQmmMin) }
    guard force || Q4Fused.splitKM, rows >= (q8SplitK1Row ? 1 : minRows), rows <= 8, K % 512 == 0, let q = lin as? QuantizedLinear, q.bits == 8, q.groupSize == 64, let b = q.biases,
          q.weight.dim(1) * 4 == K else { return lin(x) }
    let N = q.weight.dim(0)
    let SPLIT = K / 512
    let xf = x.reshaped(rows, K)
    let y = q8SplitKKernel([xf, q.weight, q.scales, b], template: [("T", x.dtype), ("M", rows), ("N", N), ("K", K), ("GS", 64), ("SPLIT", SPLIT)],
                           grid: (N * SPLIT * 32, 1, 1), threadGroup: (256, 1, 1),
                           outputShapes: [[rows, N]], outputDTypes: [.float32], initValue: 0)[0]
    return y.asType(x.dtype).reshaped(Array(x.shape.dropLast()) + [N])
}


// MARK: GDN decode front end + simdgroup gated norm (bodies in Kernels/lab/gdn_conv_norm.metal, gated_norm_sg.metal)
private let q4GDNConvNormKernel = MLXFast.metalKernel(
    name: "q4_gdn_conv_norm", inputNames: ["qkvz", "cs", "w", "eps", "sq", "sk"], outputNames: ["y", "cso"], source: Q4LabKernelSource.gdn_conv_norm)
private let q4GatedNormSGKernel = MLXFast.metalKernel(
    name: "q4_gated_norm_sg", inputNames: ["x", "w", "z", "eps"], outputNames: ["y"], source: Q4LabKernelSource.gated_norm_sg)

/// One token per sequence (S = 1): depthwise causal conv over (cached CK-1 rows + this token's mixed q|k|v = first convDim columns
/// of `qkvz`), silu, per-head RMS norm x scale for q and k, v passed through; also returns the shifted conv state for the next step.
/// qkvz (rows, QKVZ); convState (rows, CK-1, convDim); convW (convDim, CK, 1) -> (qkv (rows, convDim), newConvState (rows, CK-1, convDim))
func q4GDNConvNorm(qkvz: MLXArray, convState: MLXArray, convW: MLXArray, convDim: Int, kernel CK: Int, headDim DH: Int, keyDim KEY: Int,
                   scaleQ: Float, scaleK: Float, eps: Float) -> (MLXArray, MLXArray) {
    let rows = qkvz.dim(0)
    precondition(KEY % DH == 0 && (convDim - 2 * KEY) % DH == 0 && DH % 32 == 0 && convW.dim(0) == convDim && convW.dim(1) == CK)
    let r = q4GDNConvNormKernel([qkvz, convState, convW.reshaped(convDim, CK), q4Const(eps), q4Const(scaleQ), q4Const(scaleK)],
                                template: [("T", qkvz.dtype), ("CONVD", convDim), ("QKVZ", qkvz.dim(1)), ("CK", CK), ("DH", DH), ("KEY", KEY)],
                                grid: (rows * convDim, 1, 1), threadGroup: (DH, 1, 1),
                                outputShapes: [[rows, convDim], [rows, CK - 1, convDim]], outputDTypes: [qkvz.dtype, qkvz.dtype])
    return (r[0], r[1])
}

/// gated RMS norm with one simdgroup per (row, head): 5 us vs 11-15 us for the thread-per-head kernel on the dependent path
func q4GatedNormSG(_ x: MLXArray, weight w: MLXArray, gate z: MLXArray, headDim DH: Int, eps: Float, sigmoidGate: Bool, outDType: DType) -> MLXArray {
    let n = x.size / DH
    return q4GatedNormSGKernel([x, w, z, q4Const(eps)], template: [("T", outDType), ("TW", w.dtype), ("DH", DH), ("SIG", sigmoidGate), ("TGS", 128)],
                               grid: (n * 32, 1, 1), threadGroup: (128, 1, 1),
                               outputShapes: [x.shape], outputDTypes: [outDType])[0]
}


// MARK: attention decode front end (body in Kernels/lab/attn_front.metal)
private let q4AttnFrontKernel = MLXFast.metalKernel(
    name: "q4_attn_front", inputNames: ["qkvi", "wq", "wk", "c", "s", "eps"], outputNames: ["q", "k", "v", "sg"], source: Q4LabKernelSource.attn_front)

/// S = 1: qkvi (rows, QKVI) -> (q (rows,NH,1,HD) normed+roped, k (rows,NKV,1,HD), v (rows,NKV,1,HD), sigmoid(gate) (rows,1,NH*HD))
func q4AttnFront(qkvi: MLXArray, wq: MLXArray, wk: MLXArray, cos c: MLXArray, sin s: MLXArray, nHeads NH: Int, nKVHeads NKV: Int, headDim HD: Int, rotDim ROT: Int, eps: Float)
    -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let rows = qkvi.dim(0)
    precondition(HD % 32 == 0 && ROT <= HD && ROT % 2 == 0 && c.size == ROT && s.size == ROT)
    let r = q4AttnFrontKernel([qkvi, wq, wk, c, s, q4Const(eps)],
                              template: [("T", qkvi.dtype), ("NH", NH), ("NKV", NKV), ("HD", HD), ("ROT", ROT), ("QKVI", qkvi.dim(1)), ("TGS", HD)],
                              grid: (rows * (NH + 2 * NKV + NH) * HD, 1, 1), threadGroup: (HD, 1, 1),
                              outputShapes: [[rows, NH, 1, HD], [rows, NKV, 1, HD], [rows, NKV, 1, HD], [rows, 1, NH * HD]],
                              outputDTypes: [qkvi.dtype, qkvi.dtype, qkvi.dtype, qkvi.dtype])
    return (r[0], r[1], r[2], r[3])
}

// MARK: hyper-connection combine with the MoE combine folded in (body in Kernels/lab/hc_inject_moe.metal)
private let q4HCInjectMoEKernel = MLXFast.metalKernel(
    name: "q4_hc_inject_moe", inputNames: ["hyper", "yk", "w", "s", "g", "injp"], outputNames: ["out"], source: Q4LabKernelSource.hc_inject_moe)

/// out = hyper + 2*sigmoid(inject/H) * (sum_k w*yk + sigmoid(g)*shared)
func q4HCInjectMoE(hyper: MLXArray, yk: MLXArray, w: MLXArray, shared s: MLXArray, gate g: MLXArray, injp: MLXArray, hc H: Int, d D: Int) -> MLXArray {
    let P = injp.dim(1), K = yk.dim(1)
    return q4HCInjectMoEKernel([hyper, yk, w, s, g, injp], template: [("T", hyper.dtype), ("H", H), ("D", D), ("P", P), ("K", K)],
                               grid: (hyper.size, 1, 1), threadGroup: (256, 1, 1),
                               outputShapes: [hyper.shape], outputDTypes: [hyper.dtype])[0]
}


// MARK: QSA gather attention (bodies in Kernels/lab/qsa_sdpa_gather*.metal, qsa_sdpa_merge.metal)
private let q4QSAGatherKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["out"], source: Q4LabKernelSource.qsa_sdpa_gather)
/// P099: the same kernel with its per-simdgroup chain software-pipelined (Kernels/lab/qsa_sdpa_gather_pipe.metal): the
/// block ids and K/V rows of the next PF positions are loaded before the current position's arithmetic; values and
/// order unchanged, so BIT-IDENTICAL. ENGINE_QSA_GATHER_PF: 0 = the shipped kernel, else the prefetch depth.
private let q4QSAGatherPipeKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_pipe", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["out"], source: Q4LabKernelSource.qsa_sdpa_gather_pipe)
let q4QSAGatherPF: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_GATHER_PF"] ?? "0") ?? 0
/// P080 M1 -- ragged decode: the row's length comes from the batch index (see the source's comment).
private let q4QSAGatherRaggedKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_ragged", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["out"], source: Q4LabKernelSource.qsa_sdpa_gather_ragged)
/// P097: the partitioned, GQA-shared ragged twin (Kernels/lab/qsa_sdpa_gather_ragged_part.metal) + q4_qsa_merge.
/// ENGINE_QSA_RAGGED_PARTS: 0 = the single kernel (one 1024-thread group per (row, query head)); P > 0 = B*KVH*P
/// threadgroups of G/HPG simdgroups, block-aligned parts, merged. ENGINE_QSA_RAGGED_HPG query heads per simdgroup
/// (must divide G). ENGINE_QSA_RAGGED_MIN_ROWS: rows below it keep the single kernel (B = 1 stays bit-identical to
/// the serial program; the partition changes the online-softmax order in the last ulp).
/// P097 U2 (kernelbench --case gather, licensed, two runs): the single kernel 178-180 us per layer at B = 8, the
/// partitioned form 113 (P32/H2), 115 (P16/H2), 127 (P8/H2), 184 (P4/H1); at B = 4 135-138 vs 85-86; the B = 1 row
/// (66 vs 82-85) is NOT taken -- B = 1 stays bit-identical to the serial program by contract. Default 32 / 2 / 2.
private let q4QSAGatherRaggedPartKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_ragged_part", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_ragged_part)
let q4QSARaggedParts: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RAGGED_PARTS"] ?? "0") ?? 0
/// P099: the serial verify block's gather (position-aligned PARTS = ENGINE_QSA_PARTS, one simdgroup per head, merged) for
/// S > 1 rows per row -- the ragged twin of q4_qsa_gather_gqa_part, so a batched verify block reproduces the serial
/// block's arithmetic. ENGINE_QSA_RAGGED_VERIFY_PARTS=0 sends S > 1 through the single ragged kernel instead.
private let q4QSAGatherRaggedGQAPartKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_ragged_gqa_part", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_ragged_gqa_part)
let q4QSARaggedVerifyParts: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RAGGED_VERIFY_PARTS"] ?? "\(q4QSAParts)") ?? q4QSAParts
let q4QSARaggedHpg: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RAGGED_HPG"] ?? "2") ?? 2
let q4QSARaggedMinRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RAGGED_MIN_ROWS"] ?? "2") ?? 2

/// One decode step over B sequences of DIFFERENT lengths. `rowOffsets[b]` is the committed length of
/// row b before this token; `rowNBlocks[b]` its completed-block count. S is 1 by construction.
func q4QSAGatherRagged(q: MLXArray, keys k: MLXArray, values v: MLXArray, top: MLXArray,
                       rowOffsets: [Int], rowNBlocks: [Int], ratio R: Int, scale: Float, parts: Int? = nil, hpg: Int? = nil, pf: Int? = nil) -> MLXArray {
    let B = q.dim(0), H = q.dim(1), S = q.dim(2), D = q.dim(3), KVH = k.dim(1), kcap = k.dim(2), K = top.dim(2)
    precondition(D % 32 == 0 && H % KVH == 0 && top.dtype == .int32 && top.dim(1) == S)   // P099: S rows per row (the single kernel is S-generic)
    precondition(rowOffsets.count == B && rowNBlocks.count == B)
    var p: [Int32] = [Int32(B), 0, Int32(kcap), 0]
    p += rowOffsets.map { Int32($0) }
    p += rowNBlocks.map { Int32($0) }
    let G = H / KVH
    if S > 1, q4QSARaggedVerifyParts > 0 {
        let PV = q4QSARaggedVerifyParts
        let pr = q4QSAGatherRaggedGQAPartKernel([q, k, v, top, MLXArray(p), q4Const(scale)],
                                                template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K), ("PARTS", PV)],
                                                grid: (B * KVH * S * PV * 32, G, 1), threadGroup: (32, G, 1),
                                                outputShapes: [[B, H, S, PV, D], [B, H, S, PV], [B, H, S, PV]], outputDTypes: [.float32, .float32, .float32])
        return q4QSAMergeKernel([pr[0], pr[1], pr[2]], template: [("T", q.dtype), ("D", D), ("PARTS", PV)],
                                grid: (B * H * S * D, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
    }
    let P = parts ?? (B >= q4QSARaggedMinRows ? q4QSARaggedParts : 0)
    let hp = hpg ?? q4QSARaggedHpg
    if P > 0, S == 1, K % P == 0, hp >= 1, G % hp == 0 {
        let pr = q4QSAGatherRaggedPartKernel([q, k, v, top, MLXArray(p), q4Const(scale)],
                                             template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("R", R), ("K", K), ("PARTS", P), ("HPG", hp)],
                                             grid: (B * KVH * P * 32, G / hp, 1), threadGroup: (32, G / hp, 1),
                                             outputShapes: [[B, H, S, P, D], [B, H, S, P], [B, H, S, P]], outputDTypes: [.float32, .float32, .float32])
        return q4QSAMergeKernel([pr[0], pr[1], pr[2]], template: [("T", q.dtype), ("D", D), ("PARTS", P)],
                                grid: (B * H * S * D, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
    }
    let PF = pf ?? q4QSAGatherPF
    if PF > 0 {
        return q4QSAGatherPipeKernel([q, k, v, top, MLXArray(p), q4Const(scale)],
                                     template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K), ("PF", PF), ("RG", 1)],
                                     grid: (B * H * S * 1024, 1, 1), threadGroup: (1024, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
    }
    return q4QSAGatherRaggedKernel([q, k, v, top, MLXArray(p), q4Const(scale)],
                                   template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K)],
                                   grid: (B * H * S * 1024, 1, 1), threadGroup: (1024, 1, 1),
                                   outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
}

/// P106 H48 (row-resident KV): the ragged gather over B rows whose K/V live in B SEPARATE buffers
/// ([1, KVH, kcap_b, D] each). Every ragged kernel variant computes one (row, head, query row) per
/// threadgroup from that row's offset, block count and `top` row alone -- B, kvLenMax and kcap enter
/// only the address arithmetic -- so a one-row call on the row's own buffer is that row's slice of
/// the B-row call. The one input that depends on B is the S = 1 default partition choice, so it is
/// decided here at the batch's B and passed down explicitly.
func q4QSAGatherRaggedRows(q: MLXArray, keys: [MLXArray], values: [MLXArray], top: MLXArray,
                           rowOffsets: [Int], rowNBlocks: [Int], ratio R: Int, scale: Float) -> MLXArray {
    let B = q.dim(0)
    precondition(keys.count == B && values.count == B && rowOffsets.count == B && rowNBlocks.count == B)
    let P = B >= q4QSARaggedMinRows ? q4QSARaggedParts : 0
    // H48b: the S = 1 default (single kernel: P = 0, no pipelined prefetch) in ONE dispatch over the rows'
    // own buffers -- the same per-(row, head) body, only the base pointer and the row's capacity stride
    // come from the row. Every other variant keeps the per-row calls.
    if q.dim(2) == 1, P == 0, q4QSAGatherPF == 0, q4QSAGatherRaggedRowsMulti, B >= 2, B <= 8 {   // 2B+5 buffers; wider groups keep per-row calls
        return q4QSAGatherRaggedMulti(q: q, keys: keys, values: values, top: top, rowOffsets: rowOffsets,
                                      rowNBlocks: rowNBlocks, ratio: R, scale: scale)
    }
    let outs = (0 ..< B).map { b in
        q4QSAGatherRagged(q: q[b ..< (b + 1)], keys: keys[b], values: values[b], top: top[b ..< (b + 1)],
                          rowOffsets: [rowOffsets[b]], rowNBlocks: [rowNBlocks[b]], ratio: R, scale: scale, parts: P)
    }
    return outs.count == 1 ? outs[0] : concatenated(outs, axis: 0)
}

/// ENGINE_RR_GATHER_MULTI (default 1): the row-resident S = 1 gather as one multi-buffer dispatch; 0 = one call per row.
let q4QSAGatherRaggedRowsMulti: Bool = (ProcessInfo.processInfo.environment["ENGINE_RR_GATHER_MULTI"] ?? "1") != "0"
/// ENGINE_RR_POOLED_F32 (default 1): the row-resident equal-count score operand in one kernel pass (<= 8 rows);
/// 0 = the B38 `concatenated(views).asType(.float32)`. Model-owner thread only, like the kernel caches below.
public let q4PooledRowsF32On: Bool = (ProcessInfo.processInfo.environment["ENGINE_RR_POOLED_F32"] ?? "1") != "0"
/// `qsa_sdpa_gather_ragged` with the K/V base pointer and capacity stride taken from row b's OWN buffer
/// (inputs k0..k{B-1}, v0..v{B-1}; params [B, 0, 0, 0] ++ offset[B] ++ nBlocks[B] ++ kcap[B]). Only the
/// three address lines differ from the stacked kernel; the online softmax, its order and the merge are
/// the same source text.
private func q4QSAGatherRaggedMultiSource(_ B: Int) -> String {
    let src = Q4LabKernelSource.qsa_sdpa_gather_ragged
    let oldParams = "int Bn = params[0], kcap = params[2];"
    let oldK = "const device T* kb = k + (ulong)(b * KVH + kvh) * (ulong)kcap * D + lane * QK;"
    let oldV = "const device T* vb = v + (ulong)(b * KVH + kvh) * (ulong)kcap * D + lane * QK;"
    precondition(src.components(separatedBy: oldParams).count == 2 && src.components(separatedBy: oldK).count == 2
                 && src.components(separatedBy: oldV).count == 2, "qsa_sdpa_gather_ragged changed shape")
    var select = "const device T* kbase = k0;\nconst device T* vbase = v0;\n"
    for b in 1 ..< B { select += "if (b == \(b)) { kbase = k\(b); vbase = v\(b); }\n" }
    return src
        .replacingOccurrences(of: oldParams, with: "int Bn = params[0], kcap = params[4 + 2 * Bn + int(b)];")
        .replacingOccurrences(of: oldK, with: select + "const device T* kb = kbase + (ulong)kvh * (ulong)kcap * D + lane * QK;")
        .replacingOccurrences(of: oldV, with: "const device T* vb = vbase + (ulong)kvh * (ulong)kcap * D + lane * QK;")
}
nonisolated(unsafe) private var q4QSAGatherRaggedMultiKernels: [Int: MLXFast.MLXFastKernel] = [:]
func q4QSAGatherRaggedMulti(q: MLXArray, keys: [MLXArray], values: [MLXArray], top: MLXArray,
                            rowOffsets: [Int], rowNBlocks: [Int], ratio R: Int, scale: Float) -> MLXArray {
    let B = q.dim(0), H = q.dim(1), S = q.dim(2), D = q.dim(3), KVH = keys[0].dim(1), K = top.dim(2)
    precondition(S == 1 && D % 32 == 0 && H % KVH == 0 && top.dtype == .int32 && top.dim(1) == S && B >= 2 && B <= 8)
    precondition(keys.count == B && values.count == B && keys.allSatisfy { $0.dim(0) == 1 && $0.dim(1) == KVH }
                 && zip(keys, values).allSatisfy { $0.0.dim(2) == $0.1.dim(2) })
    let kernel: MLXFast.MLXFastKernel
    if let kk = q4QSAGatherRaggedMultiKernels[B] { kernel = kk } else {
        kernel = MLXFast.metalKernel(name: "q4_qsa_gather_ragged_rows\(B)",
                                     inputNames: ["q"] + (0 ..< B).map { "k\($0)" } + (0 ..< B).map { "v\($0)" } + ["top", "params", "scale"],
                                     outputNames: ["out"], source: q4QSAGatherRaggedMultiSource(B))
        q4QSAGatherRaggedMultiKernels[B] = kernel
    }
    var p: [Int32] = [Int32(B), 0, 0, 0]
    p += rowOffsets.map { Int32($0) }
    p += rowNBlocks.map { Int32($0) }
    p += keys.map { Int32($0.dim(2)) }
    return kernel([q] + keys + values + [top, MLXArray(p), q4Const(scale)],
                  template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K)],
                  grid: (B * H * S * 1024, 1, 1), threadGroup: (1024, 1, 1),
                  outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
}

/// H48b: the row-resident equal-count score's operand in one pass -- rows 0 ..< n of each row's own bf16
/// pooled buffer ([1, cap_b, D]) converted to f32 into [B, n, D]. bf16 -> f32 is exact and the placement is
/// `concatenated(views).asType(.float32)`'s, at the stacked path's traffic (the stacked view's asType).
private func q4PooledRowsF32Source(_ B: Int) -> String {
    var select = "const device T* base = p0;\n"
    for b in 1 ..< B { select += "if (b == \(b)) base = p\(b);\n" }
    return """
    uint gid = thread_position_in_grid.x;
    const int N = params[0], D4 = params[1];
    uint d4 = gid % D4, n = (gid / D4) % N, b = gid / (D4 * N);
    \(select)
    const device T* src = base + (ulong)n * (D4 * 4) + d4 * 4;
    device float* dst = out + ((ulong)b * N + n) * (D4 * 4) + d4 * 4;
    for (int j = 0; j < 4; ++j) dst[j] = float(src[j]);
    """
}
nonisolated(unsafe) private var q4PooledRowsF32Kernels: [Int: MLXFast.MLXFastKernel] = [:]
func q4PooledRowsF32(_ rows: [MLXArray], n: Int) -> MLXArray {
    let B = rows.count, D = rows[0].dim(2)
    precondition(B >= 1 && B <= 8 && D % 4 == 0 && rows.allSatisfy { $0.dim(0) == 1 && $0.dim(1) >= n && $0.dim(2) == D && $0.dtype == rows[0].dtype })
    let kernel: MLXFast.MLXFastKernel
    if let kk = q4PooledRowsF32Kernels[B] { kernel = kk } else {
        kernel = MLXFast.metalKernel(name: "q4_pooled_rows_f32_b\(B)", inputNames: (0 ..< B).map { "p\($0)" } + ["params"],
                                     outputNames: ["out"], source: q4PooledRowsF32Source(B))
        q4PooledRowsF32Kernels[B] = kernel
    }
    return kernel(rows + [MLXArray([Int32(n), Int32(D / 4)])], template: [("T", rows[0].dtype)],
                  grid: (B * n * (D / 4), 1, 1), threadGroup: (256, 1, 1),
                  outputShapes: [[B, n, D]], outputDTypes: [.float32])[0]
}

private let q4QSAGatherPartKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_gqa_part", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_gqa_part)
/// P019 unit 4: the same kernel with the loop over SELECTED BLOCKS instead of positions -- the position loop re-read
/// top[i/R] on every position (R-fold, plus an integer divide and a branch each time). 1.13x at a 4096-row chunk over
/// 131k of kv; threadgroup-staging the K/V rows instead was 0.68x (barriers + the occupancy the staging costs beat the
/// L1 hits it saves). ENGINE_QSA_BLKLOOP=0 restores the position loop.
private let q4QSAGatherPartBlkKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_gqa_part_blk", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_gqa_part_blk)
let q4QSABlkLoop: Bool = (ProcessInfo.processInfo.environment["ENGINE_QSA_BLKLOOP"] ?? "1") != "0"
private let q4QSAGatherPartHpgKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_gqa_part_hpg", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_gqa_part_hpg)
/// ENGINE_QSA_HPG: query heads per simdgroup in the prefill gather kernel (1 = the block-loop kernel). A K element is
/// loaded ONCE and used for HPG heads, so the load instructions per useful FLOP fall by HPG. Lab at a 4096-row chunk
/// over 135k of kv: 32.25 / 25.35 / 59.9 / 74.0 ms at HPG 1 / 2 / 3 / 4 -- 2 is 1.28x and BIT-IDENTICAL, 3 spills.
let q4QSAHpg: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_HPG"] ?? "2") ?? 2
/// P098: HPG query heads per simdgroup on the VERIFY block's kernel too (1 < S <= 16: the position-loop part kernel,
/// `qsa_sdpa_gather_gqa_part_hpgpos`), bit-identical to the single-head form. ENGINE_QSA_HPG_VERIFY: 0 = off, else HPG.
private let q4QSAGatherPartHpgPosKernel = MLXFast.metalKernel(
    name: "q4_qsa_gather_gqa_part_hpgpos", inputNames: ["q", "k", "v", "top", "params", "scale"], outputNames: ["po", "pm", "pl"], source: Q4LabKernelSource.qsa_sdpa_gather_gqa_part_hpgpos)
let q4QSAHpgVerify: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_HPG_VERIFY"] ?? "0") ?? 0
private let q4QSAMergeKernel = MLXFast.metalKernel(
    name: "q4_qsa_merge", inputNames: ["po", "pm", "pl"], outputNames: ["out"], source: Q4LabKernelSource.qsa_sdpa_merge)
/// ENGINE_QSA_PARTS: 0 = one threadgroup per (head, row) over all selected positions; P > 0 = GQA-shared kernel, P partials per (kv head, row) + merge
let q4QSAParts: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_PARTS"] ?? "16") ?? 16

/// out[b,h,s,:] = softmax over the tokens of the K selected blocks (top[b,s,:], ids >= nBlocks invisible) plus the tail
/// positions <= offset+s, of q.k^T * scale, applied to v. q [B,H,S,D]; k/v are the FULL cache buffers [B,KVH,kcap,D].
func q4QSAGather(q: MLXArray, keys k: MLXArray, values v: MLXArray, top: MLXArray, nBlocks: Int, kvLen: Int, offset: Int, ratio R: Int, scale: Float, hpgVerify: Int? = nil, pf: Int? = nil) -> MLXArray {
    let B = q.dim(0), H = q.dim(1), S = q.dim(2), D = q.dim(3), KVH = k.dim(1), kcap = k.dim(2), K = top.dim(2)
    precondition(D % 32 == 0 && S <= 16384 && H % KVH == 0 && top.dtype == .int32)
    let params = MLXArray([Int32(nBlocks), Int32(kvLen), Int32(kcap), Int32(offset)])
    let sc = q4Const(scale)
    // S = 1 (serial decode, drafts): the per-head single-dispatch kernel is as fast as the GQA partials at every context (lab 0.259 /
    // 0.260 / 0.279 ms vs 0.270 / 0.281 / 0.323 at 8k / 32k / 256k) and costs one dispatch instead of two -- in situ the two-kernel
    // form read 8k serial 56.03 vs 57.03. S > 1 (verify): the GQA-shared partials (0.39 vs 0.45 ms at 256k).
    // prefill chunks (S > 16): 8 partials -- lab S=512 gqa8 7.20/7.70 ms vs gqa16 7.27/7.79 at 128k/256k, half the partial traffic
    let P = S == 1 ? 0 : (S > 16 ? min(q4QSAParts, 8) : q4QSAParts)
    if P <= 0 {
        let PF = pf ?? q4QSAGatherPF
        if PF > 0 {
            return q4QSAGatherPipeKernel([q, k, v, top, params, sc], template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K), ("PF", PF), ("RG", 0)],
                                         grid: (B * H * S * 1024, 1, 1), threadGroup: (1024, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
        }
        return q4QSAGatherKernel([q, k, v, top, params, sc], template: [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K)],
                                 grid: (B * H * S * 1024, 1, 1), threadGroup: (1024, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
    }
    let G = H / KVH
    // MEASURED: the block loop wins on prefill chunks (S > 16: 131k continuation 943.5 -> 953.0 tok/s, 8k 1156.9 ->
    // 1169.3) and LOSES on the MTP verify block (S = K+1 <= 16, PARTS = 16: 28.52 -> 28.66 ms per round), where the
    // per-part block count is small and the tail branch is not amortised. Gate it on S, so the decode program is
    // bit-identical to the champion's.
    let hpgV = hpgVerify ?? q4QSAHpgVerify
    let verifyHpg = S <= 16 && hpgV > 1 && G % hpgV == 0
    let hpg = verifyHpg ? hpgV : ((q4QSABlkLoop && S > 16 && q4QSAHpg > 1 && G % q4QSAHpg == 0) ? q4QSAHpg : 1)
    let partKernel = verifyHpg ? q4QSAGatherPartHpgPosKernel : (hpg > 1 ? q4QSAGatherPartHpgKernel : ((q4QSABlkLoop && S > 16) ? q4QSAGatherPartBlkKernel : q4QSAGatherPartKernel))
    var tmpl: [(String, any KernelTemplateArg)] = [("T", q.dtype), ("D", D), ("H", H), ("KVH", KVH), ("S", S), ("R", R), ("K", K), ("PARTS", P)]
    if hpg > 1 { tmpl.append(("HPG", hpg)) }
    let parts = partKernel([q, k, v, top, params, sc], template: tmpl,
                                      grid: (B * KVH * S * P * 32, G / hpg, 1), threadGroup: (32, G / hpg, 1),
                                      outputShapes: [[B, H, S, P, D], [B, H, S, P], [B, H, S, P]], outputDTypes: [.float32, .float32, .float32])
    return q4QSAMergeKernel([parts[0], parts[1], parts[2]], template: [("T", q.dtype), ("D", D), ("PARTS", P)],
                            grid: (B * H * S * D, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[B, H, S, D]], outputDTypes: [q.dtype])[0]
}

// MARK: GDN prefill front end (body in Kernels/lab/gdn_conv_prefill.metal)
private let q4GDNConvPrefillKernel = MLXFast.metalKernel(
    name: "q4_gdn_conv_prefill", inputNames: ["qkvz", "cs", "w"], outputNames: ["y"], source: Q4LabKernelSource.gdn_conv_prefill)
/// ENGINE_GDN_CONV_PREFILL=0 restores the concatenate + Conv1d + silu chain
let q4GDNConvPrefillOn: Bool = (ProcessInfo.processInfo.environment["ENGINE_GDN_CONV_PREFILL"] ?? "1") != "0"
/// qkvz (rows, QKVZ) with the mixed q|k|v in the first convDim columns; convState (ck-1, convDim); w (convDim, ck)
/// -> silu(depthwise causal conv) (rows, convDim)
func q4GDNConvPrefill(qkvz: MLXArray, convState: MLXArray, convW: MLXArray, convDim: Int, kernel ck: Int) -> MLXArray {
    let rows = qkvz.dim(0), RS = qkvz.dim(1)
    let VEC = convDim % 8 == 0 ? 8 : (convDim % 4 == 0 ? 4 : 1)
    return q4GDNConvPrefillKernel(
        [qkvz, convState.reshaped(ck - 1, convDim), convW.reshaped(convDim, ck)],
        template: [("T", qkvz.dtype), ("CONVD", convDim), ("QKVZ", RS), ("CK", ck), ("VEC", VEC)],
        grid: (rows * convDim / VEC, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[rows, convDim]], outputDTypes: [qkvz.dtype])[0]
}

// MARK: QSA indexer head reduction (body in Kernels/lab/idx_relu_sum.metal)
private let q4IdxReluSumKernel = MLXFast.metalKernel(
    name: "q4_idx_relu_sum", inputNames: ["raw", "params", "scale"], outputNames: ["out"], source: Q4LabKernelSource.idx_relu_sum)
/// raw (B,S,H,N) float32 -> (B,S,N) float32, out = sum_h max(raw,0) / scale. ENGINE_IDX_FUSED=0 restores the graph form.
let q4IdxFused: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_FUSED"] ?? "1") != "0"
/// ENGINE_IDX_BF16: run the indexer's score matmul in bf16 (inputs already are) instead of casting both sides to
/// float32 -- halves the (B,S,H,N) intermediate and lets the hardware matmul path run. The accumulation inside the
/// gemm stays float32; what changes is that each score is rounded to bf16 before the relu and the head sum.
let q4IdxBF16: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_BF16"] ?? "0") != "0"
/// ENGINE_IDX_NCHUNK: reduce the block axis in slices of this many blocks. MEASURED WORSE at every size tried (131k
/// continuation 998.0 tok/s at 0 vs 988.8 / 986.4 / 993.5 / 991.0 at 1024 / 2048 / 4096 / 8192): the 2.21 GB float32
/// intermediate does not become cache-resident when sliced, and the extra launches and slice writes cost more. Default 0.
/// Kept as a knob because it is the cheap half of "fuse the reduction INTO the matmul", which is still open.
/// small enough to be served from cache instead of DRAM (0 = one call over the whole block axis)
let q4IdxNChunk: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_NCHUNK"] ?? "0") ?? 0
func q4IdxReluSum(_ raw: MLXArray, scale: Float, visibility: (kvLen: Int, ratio: Int)? = nil, blockOffset: Int = 0) -> MLXArray {
    let B = raw.dim(0), S = raw.dim(1), H = raw.dim(2), N = raw.dim(3)
    let VEC = 4
    let nblk = (N + VEC - 1) / VEC
    let v = visibility ?? (kvLen: 0, ratio: 1)
    return q4IdxReluSumKernel(
        [raw, MLXArray([Int32(N), Int32(H), Int32(S), Int32(v.kvLen), Int32(v.ratio), visibility == nil ? 0 : 1, Int32(blockOffset)]), q4Const(scale)],
        template: [("T", raw.dtype), ("VEC", VEC)],
        grid: (B * S * nblk, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[B, S, N]], outputDTypes: [.float32])[0]
}

// MARK: fused indexer score (body in Kernels/lab/idx_score_fused.metal): matmul + relu + head sum in one, no (B,S,H,N) tensor
private let q4IdxScoreKernel = MLXFast.metalKernel(
    name: "q4_idx_score_fused2", inputNames: ["q", "pooled", "params", "scale"], outputNames: ["out"], source: Q4LabKernelSource.idx_score_fused2)
/// ENGINE_IDX_SCORE=0 restores the two-op chain (float32 matmul into a (B,S,H,N) tensor, then q4IdxReluSum over it)
let q4IdxScoreFusedOn: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE"] ?? "1") != "0"
/// tile = SG*8*MI rows x 8*NI blocks (ENGINE_IDX_SCORE_MI / _NI / _SG). The lab plateau is flat across shapes because
/// the kernel sits AT the machine's arithmetic roof: 141.7 GFLOP in 6.18-6.21 ms = 22.9 TFLOPS.
let q4IdxScoreMI: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_MI"] ?? "2") ?? 2
let q4IdxScoreNI: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_NI"] ?? "2") ?? 2
let q4IdxScoreSG: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_SG"] ?? "4") ?? 4
/// ENGINE_IDX_SCORE_DEC=0 restores the graph form at S=1 (float32 cast of the whole pooled table, matmul, q4IdxReluSum).
/// The decode call is the SAME kernel with the single query row padded to the 8-row tile -- see P023 unit 5.
let q4IdxScoreDecodeOn: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_DEC"] ?? "1") != "0"
/// P096: the fused decode-branch score at B rows too (the kernel carries a batch index; bit-identical to the graph form,
/// kernelbench --case idxscore max|d| 0 at B = 4/8). P101 (OBS-ENG-197): a NO-OP for the server's ragged batched path,
/// which scores each row with the fused decode kernel already (`raggedDecode`); the census gain (graph 107-273 us vs fused
/// 57-101 per layer at n = 8192/16384, B = 4/8) does not transfer -- long probe at 32k B = 8: on 155.28 vs off 155.41 tok/s.
/// Default stays 0 (P096's state); ENGINE_IDX_SCORE_BATCH=1 enables it on the dense (non-ragged) batch.
let q4IdxScoreBatchOn: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_BATCH"] ?? "0") != "0"
/// q (B,S,H,D) T, pooled (B,N,D) T -> scores (B,S,N) float32 = sum_h max(q_h . p, 0) / scale, visibility applied arithmetically
func q4IdxScore(q: MLXArray, pooled: MLXArray, scale: Float, visibility: (kvLen: Int, ratio: Int)?,
                mi MI: Int, ni NI: Int, sg SG: Int) -> MLXArray {
    let B = q.dim(0), S = q.dim(1), H = q.dim(2), D = q.dim(3), N = pooled.dim(1)
    let TM = SG * 8 * MI, TN = 8 * NI
    precondition(S >= TM && N >= TN && D % 8 == 0)
    let v = visibility ?? (kvLen: 0, ratio: 1)
    let ntg = B * ((S + TM - 1) / TM) * ((N + TN - 1) / TN)
    return q4IdxScoreKernel(
        [q, pooled, MLXArray([Int32(N), Int32(S), Int32(v.kvLen), Int32(v.ratio), visibility == nil ? 0 : 1]), q4Const(scale)],
        template: [("T", q.dtype), ("D", D), ("H", H), ("MI", MI), ("NI", NI), ("SG", SG)],
        grid: (ntg * 32 * SG, 1, 1), threadGroup: (32 * SG, 1, 1),
        outputShapes: [[B, S, N]], outputDTypes: [.float32])[0]
}

// MARK: top-K block selection (body in Kernels/lab/topk_select.metal): radix select, one threadgroup per row, same SET as argPartition
private let q4TopKIxKernel = MLXFast.metalKernel(
    name: "q4_topk_select_ix", inputNames: ["scores", "params", "ids"], outputNames: ["top"], source: Q4LabKernelSource.topk_select_ix)
nonisolated(unsafe) private let q4TopKDummy: MLXArray = { let a = MLXArray([Int32(0)]); eval(a); return a }()
/// P026 unit 5: the register-resident twin of `q4_topk_select_ix`. Same algorithm, same order, same
/// output; it reads the keys ONCE instead of six times. Selected only when the per-thread chunk fits
/// in registers (CH <= q4TopKRegMax).
/// P026 unit 7: the SINGLE-DISPATCH exact selection. Same algorithm and same output as the two-phase
/// form; the merge runs in the same launch, in whichever threadgroup finds itself last. Tests
/// OBS-ENG-085 (4)'s prediction that one dispatch of the two is worth 0.37 ms/token.
private let q4TopK1DKernel = MLXFast.metalKernel(
    name: "q4_topk_select_1d", inputNames: ["scores", "params", "ctr"], outputNames: ["top", "cand"],
    source: Q4LabKernelSource.topk_select_1d)
/// The persistent, self-clearing arrival counter. A metal_kernel OUTPUT is allocated UNINITIALISED,
/// so a counter that must start at zero cannot be one; this is created and zeroed ONCE and the
/// merging threadgroup stores 0 back before it returns. 64 slots covers every row count the
/// partitioned path accepts (`rows <= 16`) with room to spare.
nonisolated(unsafe) private let q4TopK1DCounter: MLXArray = {
    let a = MLXArray.zeros([64], type: UInt32.self); eval(a); return a
}()
private let q4TopKRegKernel = MLXFast.metalKernel(
    name: "q4_topk_select_reg", inputNames: ["scores", "params", "ids"], outputNames: ["top"], source: Q4LabKernelSource.topk_select_reg)
private let q4TopKKernel = MLXFast.metalKernel(
    name: "q4_topk_select", inputNames: ["scores", "params"], outputNames: ["top"], source: Q4LabKernelSource.topk_select)
/// scores [B,S,n] float32 -> the K largest per row as int32 ids [B,S,K] (unordered; ties at the boundary in index order)
func q4TopK(scores: MLXArray, k K: Int) -> MLXArray {
    let B = scores.dim(0), S = scores.dim(1), n = scores.dim(2)
    precondition(scores.dtype == .float32 && K <= n)
    let TG = q4TopKTG
    // P026 unit 5, ONE-SHOT WITNESS (ENGINE_QSA_TOPK_HIST=1, never on in a timed run -- it forces a
    // host sync). The selection is six full passes over n by ONE threadgroup; whether the passes can
    // be narrowed by compacting survivors depends entirely on how many elements survive each radix
    // digit, which is a property of the SCORE DISTRIBUTION and has never been measured here. This
    // prints the survivor count after each of the four 8-bit digits, on the real keys, in situ.
    if q4TopKHistWitness, !q4TopKHistDone, B * S <= 16 {
        q4TopKHistDone = true
        let flat = scores.reshaped(B * S, n)
        eval(flat)
        let row: [Float] = flat[0].asArray(Float.self)
        var keys = row.map { (v: Float) -> UInt32 in
            let b = v.bitPattern
            return (b & 0x8000_0000) != 0 ? ~b : (b | 0x8000_0000)
        }
        var prefix: UInt32 = 0, mask: UInt32 = 0, need = K
        var line = "qwen4_exp: TOPK KEY WITNESS n=\(n) K=\(K) survivors"
        for d in stride(from: 3, through: 0, by: -1) {
            var hist = [Int](repeating: 0, count: 256)
            for k in keys where (k & mask) == prefix { hist[Int((k >> (8 * d)) & 255)] += 1 }
            var acc = 0, bin = 0, needOut = need
            for b in stride(from: 255, through: 0, by: -1) {
                if acc < need && need <= acc + hist[b] { bin = b; needOut = need - acc; break }
                acc += hist[b]
            }
            prefix |= UInt32(bin) << (8 * d); mask |= 0xFF << (8 * d); need = needOut
            let surv = keys.filter { ($0 & mask) == prefix }.count
            line += " d\(d)=\(surv)"
        }
        keys.removeAll()
        let uniqTop = Set(row.map { $0.bitPattern >> 20 }).count
        FileHandle.standardError.write((line + " distinct_top12bits=\(uniqTop)\n").data(using: .utf8)!)
    }
    if let p = q4TopKParts(rows: B * S, n: n, k: K) { return q4TopKTwoPhase(scores: scores, k: K, parts: p) }
    // The block count n grows by one every `ratio` tokens, so `n % P == 0` held on one decode step in P and every other
    // step selected the whole row in ONE threadgroup (250k blocks at 1M). Pad the row with -inf to a multiple of P:
    // the padding sits at the highest indices, so under the selection's order (key, then lower index first) it ranks
    // after every real element, and n >= K real elements exist -- the result is the unpadded selection, exactly.
    if q4TopKPad, n % q4TopKPartsEnv != 0 {
        let padded = ((n + q4TopKPartsEnv - 1) / q4TopKPartsEnv) * q4TopKPartsEnv
        if let p = q4TopKParts(rows: B * S, n: padded, k: K) {
            let fill = MLXArray.full([B, S, padded - n], values: MLXArray(-Float.infinity))
            return q4TopKTwoPhase(scores: concatenated([scores, fill], axis: 2), k: K, parts: p)
        }
    }
    let params = MLXArray([Int32(n), Int32(K)])
    return q4TopKKernel([scores.reshaped(B * S, n), params], template: [("K", K), ("TG", TG)],
                        grid: (B * S * TG, 1, 1), threadGroup: (TG, 1, 1), outputShapes: [[B * S, K]], outputDTypes: [.int32])[0].reshaped(B, S, K)
}

/// ENGINE_QSA_TOPK_PARTS: P for the two-phase decode selection (0 = off). P023 unit 7.
let q4TopKPartsEnv: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_PARTS"] ?? "4") ?? 4
/// ENGINE_QSA_TOPK_TG: threads per threadgroup for the selection. 1024 is the shipped value; at decode the row is
/// selected by ONE threadgroup, so the four radix passes and the two compaction passes are six threadgroup-wide
/// barriers over 32 simdgroups each. Must be a multiple of 32 and >= 256 (the suffix scan uses 256 threads).
let q4TopKTG: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_TG"] ?? "1024") ?? 1024
/// P026 unit 3: ENGINE_QSA_TOPK_FOLDOFS=0 restores the MLX offset add between the two selection
/// phases, so the removal can be measured paired on ONE binary rather than across a rebuild.
/// P026 unit 5: ENGINE_QSA_TOPK_REG=0 restores the six-pass body, so the change is measurable paired
/// on ONE binary. ENGINE_QSA_TOPK_REGMAX caps the per-thread chunk (registers) the reg body will take.
/// DEFAULT OFF. REFUT-ENG-014 (2) measured the register body at -1.60% at the shipped P=4 and inside
/// the noise everywhere else, so it is a knob for future round-reducing work, NOT the shipped path.
/// A neutral or negative arm does not become the default just because it is newer.
/// P026 unit 6: ENGINE_QSA_TOPK_ROUNDS < 4 is a TIMING-ONLY ablation of the radix round count -- the
/// selection it produces is WRONG by construction. It prices the round count before an exact
/// round-reducing body is built. 4 is the shipped body.
/// P026 unit 7: ENGINE_QSA_TOPK_1D=1 selects the single-dispatch body. Default OFF until measured.
let q4TopKOneDispatch: Bool = (ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_1D"] ?? "0") != "0"
let q4TopKRounds: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_ROUNDS"] ?? "4") ?? 4
let q4TopKReg: Bool = (ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_REG"] ?? "0") != "0"
let q4TopKRegMax: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_REGMAX"] ?? "16") ?? 16
let q4TopKHistWitness: Bool = ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_HIST"] != nil
nonisolated(unsafe) var q4TopKHistDone = false
/// DEFAULT OFF. REFUT-ENG-014 (1) measured +0.00% at the shipped P=4. It removes a real dispatch and
/// is bit-identical, but a change that measures zero is not a licensed win and does not move the
/// champion; it stays available for the round-reducing body that may yet make it matter.
let q4TopKFoldOffset: Bool = (ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_FOLDOFS"] ?? "0") != "0"
/// The single-threadgroup radix select is 1.55 ms per token at 262144 (12 layers x 6 full passes over 65536 float32 by ONE
/// threadgroup of 1024 threads on an 80-core GPU). It is only worth splitting when the row count is small -- at prefill
/// width there are thousands of rows and the machine is already full.
/// P096 (OBS-ENG-184): below this universe size one 1024-thread group per row beats the two-phase split on real
/// scores (n = 1024..4096: 40-48 us vs 63-69 per call, the same id set in the same order); in situ at B = 8 rows of
/// 8192+512i the split cost +1.3-1.4% of the step, at 262144 serial (n = 65536) it is worth +1.3% (P023).
/// ENGINE_QSA_TOPK_PAD (default on; 0 = off): pad a row whose block count is not a multiple of the part count so the
/// two-phase selection runs on every step, not on one step in P.
let q4TopKPad: Bool = ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_PAD"] != "0"
let q4TopKPartsMinN: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_PARTS_MIN_N"] ?? "32768") ?? 32768
func q4TopKParts(rows: Int, n: Int, k K: Int) -> Int? {
    let P = q4TopKPartsEnv
    guard P > 1, rows <= 16, n % P == 0, n / P >= K, K > 0, n >= q4TopKPartsMinN else { return nil }
    return P
}
/// EXACT two-phase selection. Order: a before b iff key(a) > key(b), or the keys are equal and idx(a) < idx(b) -- which is
/// exactly what q4_topk_select implements (all keys above the threshold, then the ties at it IN INDEX ORDER). Under a strict
/// total order, an element of the global top-K has fewer than K predecessors globally, hence fewer than K inside its OWN
/// part, hence it is in that part's top-K: the union of the per-part top-Ks CONTAINS the global one, with no tie exception.
/// Phase 1 is the same kernel on the row reshaped to (rows*P, n/P), so P parts run as P threadgroups instead of one.
/// Candidate ids are concatenated part by part, so the candidate array is in increasing original index order and phase 2's
/// tie rule reproduces the global one.
func q4TopKTwoPhase(scores: MLXArray, k K: Int, parts P: Int, oneDispatch: Bool = q4TopKOneDispatch) -> MLXArray {
    let B = scores.dim(0), S = scores.dim(1), n = scores.dim(2), R = B * S, m = n / P
    let TG = q4TopKTG
    if oneDispatch, P > 1, (P & (P - 1)) == 0, Qwen4ExpBatch.rowOffsets == nil {   // D39: never inside a batched step
        // params: [m, K, P, SROW, PK]. The part offset and the candidate gather are both inside the
        // kernel, so there is no MLX glue op between the phases -- there are no phases to glue.
        let out = q4TopK1DKernel(
            [scores.reshaped(R, n), MLXArray([Int32(m), Int32(K), Int32(P), Int32(n), Int32(P * K)]), q4TopK1DCounter],
            template: [("K", K), ("TG", TG), ("P", P)],
            grid: (R * P * TG, 1, 1), threadGroup: (TG, 1, 1),
            outputShapes: [[R, K], [R, P * K]], outputDTypes: [.int32, .int32])
        return out[0].reshaped(B, S, K)
    }
    let dummy = q4TopKDummy
    // phase 1: R*P threadgroups, one per (row, part). USEIDX=0, and the part offset is applied INSIDE the
    // kernel (params[2]), so the (R,P,K) ids are already global and in increasing order part by part.
    // The kernel indexes rows of `scores` by threadgroup, so the part offset is folded into the row stride:
    // row r*P+p starts at (r*P+p)*m of the flat (R,n) buffer, which is exactly part p of row r.
    // P026 unit 3: PARTS is passed to the kernel and the part offset is applied INSIDE it, so the
    // broadcast offset add that used to sit between the phases -- one whole MLX dispatch per
    // selection, twelve per token -- is gone. Bit-identical: the same integer is added to the same
    // id, in the same place in the order.
    let ch1 = (m + TG - 1) / TG
    let useReg1 = q4TopKReg && q4TopKFoldOffset && ch1 <= q4TopKRegMax
    let c1 = (useReg1 ? q4TopKRegKernel : q4TopKIxKernel)(
                            [scores.reshaped(R * P, m), MLXArray([Int32(m), Int32(K), Int32(0), Int32(m)]), dummy],
                            template: (useReg1
                                       ? [("K", K), ("TG", TG), ("USEIDX", 0), ("PARTS", P), ("CH", ch1), ("FULL", ch1 * TG == m ? 1 : 0)]
                                       : [("K", K), ("TG", TG), ("USEIDX", 0), ("PARTS", q4TopKFoldOffset ? P : 1), ("ROUNDS", q4TopKRounds)]),
                            grid: (R * P * TG, 1, 1), threadGroup: (TG, 1, 1),
                            outputShapes: [[R * P, K]], outputDTypes: [.int32])[0]
    let cand: MLXArray
    if q4TopKFoldOffset {
        cand = c1.reshaped(R, P * K)
    } else {
        let offs = Qwen4ExpQSAIndexer.int32Range(0, P, scale: m).reshaped(1, P, 1)
        cand = (c1.reshaped(R, P, K) + offs).reshaped(R, P * K)
    }
    // phase 2: ONE threadgroup per row over the P*K candidates. USEIDX=1 reads scores[cand[i]] and emits
    // cand[i] itself, so no candidate-score gather and no final id gather exist as separate dispatches.
    let ch2 = (P * K + TG - 1) / TG
    let useReg2 = q4TopKReg && ch2 <= q4TopKRegMax
    let c2 = (useReg2 ? q4TopKRegKernel : q4TopKIxKernel)(
                            [scores.reshaped(R, n), MLXArray([Int32(P * K), Int32(K), Int32(0), Int32(n)]), cand],
                            template: (useReg2
                                       ? [("K", K), ("TG", TG), ("USEIDX", 1), ("PARTS", 1), ("CH", ch2), ("FULL", ch2 * TG == P * K ? 1 : 0)]
                                       : [("K", K), ("TG", TG), ("USEIDX", 1), ("PARTS", 1), ("ROUNDS", q4TopKRounds)]),
                            grid: (R * TG, 1, 1), threadGroup: (TG, 1, 1),
                            outputShapes: [[R, K]], outputDTypes: [.int32])[0]
    return c2.reshaped(B, S, K)
}

/// Diagnostic (ENGINE_MOE_PROBE): a pure read of a uint32 buffer in 16-byte vectors, grid-stride, one xor per thread
/// written out -- the machine's streaming-read ceiling for the MoE probe to compare against.
private let q4ReadBWKernel = MLXFast.metalKernel(
    name: "q4_read_bw", inputNames: ["w", "n4"], outputNames: ["o"],
    source: """
        uint gid = thread_position_in_grid.x, nth = threads_per_grid.x;
        const device uint4* w4 = (const device uint4*)w;
        uint4 acc = uint4(0);
        for (ulong i = gid; i < (ulong)n4[0]; i += nth) acc ^= w4[i];
        o[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
        """)
func q4ReadBW(_ w: MLXArray, threads: Int = 1 << 18, tg: Int = 256) -> MLXArray {
    let flat = w.reshaped(-1)
    precondition(flat.dtype == .uint32 && flat.size % 4 == 0)
    return q4ReadBWKernel([flat, MLXArray([Int32(flat.size / 4)])], template: [],
                          grid: (threads, 1, 1), threadGroup: (tg, 1, 1), outputShapes: [[threads]], outputDTypes: [.uint32])[0]
}
