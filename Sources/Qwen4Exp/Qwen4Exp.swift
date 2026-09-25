// Qwen3.8-Flash-Next (`model_type` qwen4_exp) -- ENGINE's own Swift port.
//
// Text tower only (the vision tower and the MTP head are dropped at load).
// Written against three references kept under lab/reference/ and treated as
// UNTRUSTED until the logits fixtures match: the HF transformers modeling file
// (authoritative), mlx-lm PR #1788 (the vectorised MLX form this port mirrors
// op for op), and mlx-serve's Zig n-gram hashing (host-side id math).
//
// Architecture, from config.json text_config:
//   48 layers: 36 GatedDeltaNet ("linear_attention") + 12 full attention with the
//   Qwen Sparse Attention indexer (every 4th layer); every layer is a 512-expert
//   MoE (top-10 + shared expert) behind a hyper-connection residual stream of
//   hc_count=4 copies of the hidden state (10240 wide); a hashed n-gram
//   embedding (PLE) is added to the stream before layer index 1; the final
//   hyper-connection mixer replaces the final norm; lm_head is untied.
//
// Parity traps recorded while porting:
//   * RMSNorm is ZERO-CENTERED: y = norm(x) * (1 + weight). Only the GDN's gated
//     norm scales by `weight` alone.
//   * Hyper-connection norms take one statistic per stream (group of hidden_size)
//     but apply the flat (hc*hidden) weight afterwards.
//   * Attention q_proj carries the output gate interleaved per head
//     (n_heads * head_dim * 2, split on the last axis of (B,S,H,2*hd)).
//   * QSA is a no-op until the key length exceeds indexer_budget (2048).
//   * n-gram ids: int64 wrapping multiply, xor, FLOOR mod (Zig @mod / torch %),
//     shift never crosses the most recent EOS strictly before the position.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import MLXLLM
import MLXVLM

/// Opt-in census of indexer GRAPH CONSTRUCTION, not Metal kernel execution.
/// A speculative, ablated or reused graph can be discarded without evaluation;
/// publishing these counts after consumer eval does not change that distinction.
/// All recording and snapshot calls must remain on the model owner thread. The
/// server may publish a copied plain snapshot for its other threads to read.
public enum Qwen4ExpIndexerRouteWitness {
    public static let enabled = ProcessInfo.processInfo.environment["ENGINE_INDEXER_ROUTE_WITNESS"] == "1"
    nonisolated(unsafe) private static var counts: [String: Int] = [:]

    fileprivate enum Route: String {
        case raggedS1GraphEqual = "ragged_s1_graph_equal"
        case raggedS1GraphMasked = "ragged_s1_graph_masked"
        case raggedS1GraphUnequal = "ragged_s1_graph_unequal"
        case raggedVerifyFused = "ragged_verify_fused"
        case raggedVerifyGraph = "ragged_verify_graph"
        case denseFused = "dense_fused"
        case denseGraph = "dense_graph"
    }

    /// B is the consumer's batch width; scoreB is the selected operation's batch
    /// width (one for each per-row ragged score). S is the unpadded query length.
    /// The number of keys is bounded by route/B/S/budget/scoreB, not context length.
    @inline(__always) fileprivate static func record(_ route: Route, batch B: Int, sequence S: Int,
                                                     budget: Int, scoreBatch: Int, nBlocks: Int,
                                                     visibleMinBlocks: Int? = nil) {
        guard enabled else { return }
        let visible = visibleMinBlocks ?? nBlocks
        let key = "graph_construction.\(route.rawValue).B\(B).S\(S).budget\(budget).scoreB\(scoreBatch)"
        counts["graph_construction.total", default: 0] += 1
        counts[key + ".calls", default: 0] += 1
        counts[key + ".nblocks_min"] = min(counts[key + ".nblocks_min"] ?? nBlocks, nBlocks)
        counts[key + ".nblocks_max"] = max(counts[key + ".nblocks_max"] ?? nBlocks, nBlocks)
        counts[key + ".visible_nblocks_min"] = min(counts[key + ".visible_nblocks_min"] ?? visible, visible)
        counts[key + ".visible_nblocks_max"] = max(counts[key + ".visible_nblocks_max"] ?? visible, visible)
        if nBlocks >= 64_000 { counts[key + ".score_span_ge64000", default: 0] += 1 }
        // A short row padded to another row's full span is not a full-context row.
        if visible >= 64_000 { counts[key + ".all_scored_rows_ge64000", default: 0] += 1 }
    }

    /// Owner-thread-only, plain values. Empty means the census is disabled or no
    /// score graph has been constructed. Never interpret this as a GPU timer.
    public static func graphConstructionSnapshot() -> [String: Int] {
        enabled ? counts : [:]
    }
}

// P106 diagnostic/correctness arm: batching the token axis changes the backend's
// split-K selection for skinny projections. Keep each sequence's original M for
// these projections while the large projections and MoE remain batched.
private let prefillRowProjectionMode = ProcessInfo.processInfo.environment["ENGINE_PREFILL_ROW_PROJECTIONS"] ?? "none"
private let prefillRowProjectionParts = Set(prefillRowProjectionMode.split(separator: ",").map(String.init))
nonisolated(unsafe) private var prefillRowProjectionCalls: [String: Int] = [:]

/// P106 H50: the same split-K effect along the token axis of ONE row. A B1 chunk of 4096 rows and four
/// chunks of 1024 rows give these skinny projections different M, hence different split-K partitions and
/// different rounding, so the state at a 1024-aligned boundary depended on how the prefill was chunked
/// (which is why canonical prefix reuse pinned every chunk to 1024). With `canonicalRows` = R > 0 a B1
/// chunk wider than R computes them in R-row blocks: every chunk width then does the R-wide arithmetic.
/// 0 = off (the historical single matmul). Owner-thread state; the probe toggles it between arms.
public enum Qwen4ExpPrefillWidth {
    nonisolated(unsafe) public static var canonicalRows: Int =
        Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_WIDTH_CANONICAL"] ?? "0") ?? 0
    nonisolated(unsafe) public static var blockedCalls = 0
}

private func prefillProjection(_ linear: Linear, _ x: MLXArray, _ part: String) -> MLXArray {
    let R = Qwen4ExpPrefillWidth.canonicalRows
    if R > 0, x.ndim == 3, x.dim(0) == 1, x.dim(1) > R {
        Qwen4ExpPrefillWidth.blockedCalls += 1
        let S = x.dim(1)
        return concatenated(stride(from: 0, to: S, by: R).map { linear(x[0..., $0 ..< min(S, $0 + R), 0...]) }, axis: 1)
    }
    guard x.ndim == 3, x.dim(0) > 1, x.dim(1) > 64,
          prefillRowProjectionParts.contains("all") || prefillRowProjectionParts.contains(part) else { return linear(x) }
    prefillRowProjectionCalls[part, default: 0] += 1
    return concatenated((0 ..< x.dim(0)).map { linear(x[$0 ..< ($0 + 1)]) }, axis: 0)
}

/// Primitive-op activations. MLXNN's `silu`/`relu` are `compile`d closures: each call takes the
/// process-global `evalLock` and runs `compile_replace` on the host (~50 us, and it blocks any
/// concurrent graph build while an eval is encoding). These are plain lazy ops.
let q4SiluMode = Int(ProcessInfo.processInfo.environment["ENGINE_SILU_MODE"] ?? "0") ?? 0   // 0 kernel, 1 primitives, 2 MLXNN compiled
func q4Silu(_ x: MLXArray) -> MLXArray { q4SiluMode == 0 ? q4SiluDiv(x, by: 1) : (q4SiluMode == 1 ? x * sigmoid(x) : silu(x)) }
@inline(__always) func q4Relu(_ x: MLXArray) -> MLXArray { maximum(x, MLXArray(0).asType(x.dtype)) }

/// Serving-only allocation policy. Configure once on the model owner, before any
/// cache is made or warmed. Generic benchmarks leave nil and retain existing growth.
/// This never validates or truncates tokens; serving admission owns the horizon.
public enum Qwen4ExpCacheCapacity {
    nonisolated(unsafe) public private(set) static var growthLimit: Int? = nil
    public static var kvStep: Int { Qwen4ExpModelInner.kvStep }
    public static var indexerStep: Int { Qwen4ExpQSAIndexer.idxStep }
    public static func configureGrowthLimit(_ limit: Int?) {
        precondition(limit == nil || limit! > 0)
        precondition(kvStep > 0 && indexerStep > 0)
        growthLimit = limit
    }
    @inline(__always) static func bounded(_ wanted: Int, need: Int, ratio: Int = 1) -> Int {
        precondition(ratio > 0)
        return growthLimit.map { max(need, min(wanted, $0 / ratio)) } ?? wanted
    }
    static func makeKVCache() -> KVCacheSimple {
        let kv = KVCacheSimple()
        kv.step = kvStep; kv.capacityGrowthLimit = growthLimit
        return kv
    }
    /// Logical stable state per text row: fp32 GDN SSM, bf16 conv/PLE/Slast,
    /// int32 ngram. Verify tape and prefill workspace remain separately reserved.
    public static func fixedStateBytes(_ a: Qwen4ExpTextConfiguration, mtp: Bool) -> Double {
        let linear = Double(a.layerTypes.filter { $0 == "linear_attention" }.count)
        let ssm = Double(a.linearNumValueHeads) * Double(a.linearValueHeadDim) * Double(a.linearKeyHeadDim) * 4
        let conv = Double(a.linearConvKernelDim - 1)
            * (2 * Double(a.linearNumKeyHeads) * Double(a.linearKeyHeadDim)
               + Double(a.linearNumValueHeads) * Double(a.linearValueHeadDim)) * 2
        let pleLayers = Double(Set(a.pleLayerIds.filter { $0 > 0 && $0 <= a.hiddenLayers }).count)
        let hc = Double(a.hiddenSize) * Double(a.hcCount)
        let ple = pleLayers * Double(a.pleConvKernelSize - 1) * Double(a.ngramSize) * hc * 2
        let ngram = pleLayers > 0 ? Double(a.ngramSize - 1) * 4 : 0
        return linear * (ssm + conv) + ple + ngram + (mtp ? hc * 2 : 0)
    }
    /// Exported hot-rung upper bound, not a measurement of unique allocations.
    /// Includes PLE/ngram rollback buffers retained by an S>1 capture and compact
    /// allocation slack. GDN replay tapes are not exported into the rung.
    public static func hotRungBytes(_ a: Qwen4ExpTextConfiguration,
                                    maxForwardRows: Int, allocationSlack: Int) -> Double {
        precondition(maxForwardRows > 0 && allocationSlack >= 0)
        let linear = Double(a.layerTypes.filter { $0 == "linear_attention" }.count)
        let ple = Double(Set(a.pleLayerIds.filter { $0 > 0 && $0 <= a.hiddenLayers }).count)
        let hc = Double(a.hiddenSize) * Double(a.hcCount)
        let stateLen = Double(a.pleConvKernelSize - 1) * Double(a.ngramSize)
        let rows = Double(maxForwardRows)
        let tapes = maxForwardRows > 1
            ? ple * (rows + stateLen) * hc * 2
                + (ple > 0 ? (rows + Double(a.ngramSize - 1)) * 4 : 0) : 0
        // Two GDN arrays per linear layer, PLE state+tape per PLE layer,
        // and one shared ngram state+tape. Overcharges absent S1 tapes safely.
        let arrays = 2 * linear + 2 * ple + (ple > 0 ? 2 : 0)
        return fixedStateBytes(a, mtp: false) + tapes + arrays * Double(allocationSlack)
    }
}

// MARK: - Configuration

public struct Qwen4ExpTextConfiguration: Decodable, Sendable {
    public var modelType: String = "qwen4_exp_text"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 24
    public var kvHeads: Int = 2
    public var headDim: Int = 256
    public var vocabularySize: Int = 248_320
    public var rmsNormEps: Float = 1e-6
    public var layerTypes: [String] = []
    public var fullAttentionInterval: Int = 4
    public var numExperts: Int = 512
    public var numExpertsPerTok: Int = 10
    public var moeIntermediateSize: Int = 640
    public var sharedExpertIntermediateSize: Int = 640
    public var linearNumKeyHeads: Int = 16
    public var linearNumValueHeads: Int = 48
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var outputGateType: String = "sigmoid"
    public var hcCount: Int = 4
    public var hcLowrank: Int = 320
    public var indexerNHeads: Int = 4
    public var indexerKVHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4
    public var ngramSize: Int = 3
    public var headsPerNgram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var ngramDivisibleBy: Int = 128
    public var splitNgramParts: Int = 128
    public var pleEmbedDim: Int = 2560
    public var pleLayerIds: [Int] = [2]
    public var pleConvKernelSize: Int = 4
    public var seed: Int = 1234   // HF configuration default; the checkpoint config omits it and mlx-lm PR #1788 wrongly assumes 0
    public var eosTokenId: Int = 248_044
    public var partialRotaryFactor: Float = 0.25
    /// P037: [T, H, W] half-dim counts for interleaved mRoPE. Only vision tokens make the axes differ.
    public var mropeSection: [Int] = [11, 11, 10]
    public var ropeTheta: Float = 10_000_000
    public var tieWordEmbeddings: Bool = false
    /// first id of the special-token block at the top of the vocabulary (Qwen3.8-Flash-Next: 248044); used by the draft-head trim
    public var vocabSpecialsFrom: Int = 248044

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case outputGateType = "output_gate_type"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerNHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case ngramDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case pleEmbedDim = "ple_embed_dim"
        case pleLayerIds = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case seed
        case eosTokenId = "eos_token_id"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func get<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        modelType = try get(.modelType, modelType)
        hiddenSize = try get(.hiddenSize, hiddenSize)
        hiddenLayers = try get(.hiddenLayers, hiddenLayers)
        attentionHeads = try get(.attentionHeads, attentionHeads)
        kvHeads = try get(.kvHeads, kvHeads)
        headDim = try get(.headDim, headDim)
        vocabularySize = try get(.vocabularySize, vocabularySize)
        rmsNormEps = try get(.rmsNormEps, rmsNormEps)
        layerTypes = try get(.layerTypes, layerTypes)
        fullAttentionInterval = try get(.fullAttentionInterval, fullAttentionInterval)
        numExperts = try get(.numExperts, numExperts)
        numExpertsPerTok = try get(.numExpertsPerTok, numExpertsPerTok)
        moeIntermediateSize = try get(.moeIntermediateSize, moeIntermediateSize)
        sharedExpertIntermediateSize = try get(.sharedExpertIntermediateSize, sharedExpertIntermediateSize)
        linearNumKeyHeads = try get(.linearNumKeyHeads, linearNumKeyHeads)
        linearNumValueHeads = try get(.linearNumValueHeads, linearNumValueHeads)
        linearKeyHeadDim = try get(.linearKeyHeadDim, linearKeyHeadDim)
        linearValueHeadDim = try get(.linearValueHeadDim, linearValueHeadDim)
        linearConvKernelDim = try get(.linearConvKernelDim, linearConvKernelDim)
        outputGateType = try get(.outputGateType, outputGateType)
        hcCount = try get(.hcCount, hcCount)
        hcLowrank = try get(.hcLowrank, hcLowrank)
        indexerNHeads = try get(.indexerNHeads, indexerNHeads)
        indexerKVHeads = try get(.indexerKVHeads, indexerKVHeads)
        indexerHeadDim = try get(.indexerHeadDim, indexerHeadDim)
        indexerBudget = try get(.indexerBudget, indexerBudget)
        // P030 unit 5: `indexer_budget` as an ARM. Unit 4 measured the decode step at 262144 staying
        // inside the rounding class down to the top 128 of 512 blocks, so the shipped 2048 is 4x what
        // the step distinguishably uses on that text. This knob makes that testable end to end --
        // trunk AND prefill, where attn.core is 9.2% of TTFT. `ENGINE_MTP_IDX_BUDGET` still overrides
        // it for the draft head, which is a separate policy (P020 unit 4).
        if let b = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_BUDGET"] ?? "") { indexerBudget = b }
        indexerCompressRatio = try get(.indexerCompressRatio, indexerCompressRatio)
        ngramSize = try get(.ngramSize, ngramSize)
        headsPerNgram = try get(.headsPerNgram, headsPerNgram)
        ngramVocabSizeBase = try get(.ngramVocabSizeBase, ngramVocabSizeBase)
        ngramDivisibleBy = try get(.ngramDivisibleBy, ngramDivisibleBy)
        splitNgramParts = try get(.splitNgramParts, splitNgramParts)
        pleEmbedDim = try get(.pleEmbedDim, pleEmbedDim)
        pleLayerIds = try get(.pleLayerIds, pleLayerIds)
        pleConvKernelSize = try get(.pleConvKernelSize, pleConvKernelSize)
        seed = try get(.seed, seed)
        tieWordEmbeddings = try get(.tieWordEmbeddings, tieWordEmbeddings)
        partialRotaryFactor = try get(.partialRotaryFactor, partialRotaryFactor)
        ropeTheta = try get(.ropeTheta, ropeTheta)
        // eos_token_id may be an int or a list
        if let e = try? c.decodeIfPresent(Int.self, forKey: .eosTokenId) {
            eosTokenId = e
        } else if let es = try? c.decodeIfPresent([Int].self, forKey: .eosTokenId), let f = es.first {
            eosTokenId = f
        }
        if let rp = try? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters) {
            if case .float(let t)? = rp["rope_theta"] { ropeTheta = t }
            if case .float(let f)? = rp["partial_rotary_factor"] { partialRotaryFactor = f }

        }
        // `StringOrNumber` has no array case, so mrope_section gets its own decode
        struct _RopeParams: Decodable { let mrope_section: [Int]? }
        if let rp2 = try? c.decodeIfPresent(_RopeParams.self, forKey: .ropeParameters),
           let ms = rp2.mrope_section, ms.count == 3 { mropeSection = ms }
        if layerTypes.isEmpty {
            layerTypes = (0..<hiddenLayers).map {
                ($0 + 1) % fullAttentionInterval == 0 ? "full_attention" : "linear_attention"
            }
        }
    }
}

public struct Qwen4ExpConfiguration: Decodable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var text: Qwen4ExpTextConfiguration = .init()
    /// P037: present on the operating artifact -- `architectures = Qwen4ExpForConditionalGeneration`,
    /// `image_token_id 248056`, a 27-block tower, `Qwen3VLProcessor`. nil for a text-only checkpoint.
    public var vision: Qwen3VLConfiguration.VisionConfiguration? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        text = try c.decodeIfPresent(Qwen4ExpTextConfiguration.self, forKey: .text) ?? .init()
        vision = try? c.decodeIfPresent(Qwen3VLConfiguration.VisionConfiguration.self, forKey: .vision)
    }
}

// MARK: - Norms

/// Zero-centered RMSNorm: y = rmsnorm(x) * (1 + weight). With `groupSize` one
/// statistic per group of the last axis, the flat weight applied after.
final class Qwen4ExpRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float = 1e-6) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        self.groupSize = groupSize
        if let g = groupSize { precondition(dimensions % g == 0) }
        super.init()
    }

    private var onePlusW: MLXArray? = nil
    private var onePlusWDType: DType? = nil
    func onePlusWeight(_ dtype: DType) -> MLXArray {
        if onePlusW == nil || onePlusWDType != dtype {
            onePlusW = (1 + weight).asType(dtype); onePlusWDType = dtype; eval(onePlusW!)
        }
        return onePlusW!
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = onePlusWeight(x.dtype)
        guard let g = groupSize else {
            return MLXFast.rmsNorm(x, weight: w, eps: eps)
        }
        // P020 unit 7: at PREFILL widths take the grouped norm through our own kernel, which is float32 THROUGHOUT
        // (one rounding, like HF) instead of MLX's rms_norm-then-multiply (two). It is +0.5-0.6% on its own, and it is
        // what makes the combine + next-norm fusion below BIT-IDENTICAL to the unfused pair. Decode (rows <= 16) is
        // left on the MLX chain so the decode program is untouched. ENGINE_GROUPED_NORM_PREFILL=0 restores it.
        if Q4Fused.groupedNorm || (q4GroupedNormPrefill && x.size / x.dim(-1) > 16) {
            return q4GroupedNorm(x, onePlusW: w, groups: x.dim(-1) / g, groupSize: g, eps: eps)
        }
        let shape = x.shape
        var y = x.reshaped(Array(shape.dropLast()) + [-1, g])
        y = MLXFast.rmsNorm(y, weight: MLXArray.mlxNone, eps: eps).reshaped(shape)
        return y * w
    }
}

/// Conventional gated RMSNorm (the GDN output norm): rmsnorm(x) * weight,
/// gated by sigmoid(z) (or silu) in float32.
final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let sigmoidGate: Bool

    init(dimensions: Int, eps: Float, activation: String) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        self.sigmoidGate = activation == "sigmoid"
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray, outDType: DType? = nil) -> MLXArray {
        let out = MLXFast.rmsNorm(x, weight: weight, eps: eps).asType(.float32)
        let g32 = gate.asType(.float32)
        let g = sigmoidGate ? sigmoid(g32) : q4Silu(g32)
        return (g * out).asType(outDType ?? x.dtype)
    }
}

// MARK: - RoPE (partial, half-rotation, float32 tables)

final class Qwen4ExpRotary {
    let dim: Int
    let invFreq: MLXArray
    private var cosTable: MLXArray? = nil, sinTable: MLXArray? = nil, tableLen = 0

    init(dim: Int, base: Float) {
        self.dim = dim
        let idx = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        self.invFreq = pow(MLXArray(base), -(idx / Float(dim)))
    }

    /// positions (B,T) -> cos/sin (B,T,dim) in float32
    func tables(_ positions: MLXArray) -> (MLXArray, MLXArray) {
        let f = expandedDimensions(positions.asType(.float32), axis: -1) * invFreq
        let emb = concatenated([f, f], axis: -1)
        return (cos(emb), sin(emb))
    }

    /// P106 H60: per-row positions of a ragged step (row b at rows[b] + s), memoised on the host positions. Every attention
    /// layer's indexer AND attention of one ragged forward rope at the same positions -- 24 identical table graphs per
    /// forward before (12 QSA layers x 2); the same arrays are now shared (the same ops on the same inputs, once).
    private var rowMemo: (pos: [Int32], B: Int, S: Int, c: MLXArray, s: MLXArray)? = nil
    func tables(rowPositions pos: [Int32], B: Int, S: Int) -> (MLXArray, MLXArray) {
        if let m = rowMemo, m.B == B, m.S == S, m.pos == pos { return (m.c, m.s) }
        let (c, s) = tables(MLXArray(pos).reshaped(B, S))
        rowMemo = (pos, B, S, c, s)
        return (c, s)
    }

    /// contiguous positions [offset, offset+count): sliced from a cached table (1 op each, no per-step trig)
    func tables(offset: Int, count: Int) -> (MLXArray, MLXArray) {
        let need = offset + count
        if cosTable == nil || need > tableLen {
            tableLen = max(need, tableLen * 2, 8192)
            let (c, s) = tables(expandedDimensions(MLXArray(Int32(0) ..< Int32(tableLen)), axis: 0))
            cosTable = c; sinTable = s; eval(c, s)
        }
        return (cosTable![0..., offset ..< need], sinTable![0..., offset ..< need])
    }

    static func positions(offset: Int, count: Int) -> MLXArray {
        expandedDimensions(MLXArray(Int32(offset) ..< Int32(offset + count)), axis: 0)
    }

    /// P037 unit 2 -- mRoPE. `pos3` is (3, B, T): the T/H/W position of every token. Vision tokens are
    /// the only place the three axes differ; for text they are equal and this returns EXACTLY what
    /// `tables(_:)` returns, which is the gate, not the argument (OBS-ENG-041 (1) says the same thing
    /// from the source side: `mrope_interleaved` is a no-op for text).
    ///
    /// The interleave is transcribed from the authoritative HF file, `apply_interleaved_mrope`:
    /// start from the T axis everywhere, then for H (offset 1) and W (offset 2) overwrite the
    /// positions `offset, offset+3, ...` below `mrope_section[axis] * 3`. With section [11,11,10] over
    /// the 32 half-dims of the 64 rotary dims that lays out T,H,W,T,H,W,... and leaves 11/11/10.
    func tablesMRope(_ pos3: MLXArray, section: [Int]) -> (MLXArray, MLXArray) {
        precondition(pos3.dim(0) == 3, "mrope needs (3, B, T) positions")
        let half = invFreq.dim(0)                                   // rotary_dim / 2
        // (3, B, T, half)
        let f = expandedDimensions(pos3.asType(.float32), axis: -1) * invFreq
        var take = [Int32](repeating: 0, count: half)               // which axis feeds each half-dim
        for (axis, offset) in [(1, 1), (2, 2)] {
            let end = min(section[axis] * 3, half)
            var i = offset
            while i < end { take[i] = Int32(axis); i += 3 }
        }
        let sel = MLXArray(take)                                     // (half,), broadcasts over (B,T,half)
        var merged = f[0]
        for axis in 1 ... 2 { merged = MLX.which(sel .== MLXArray(Int32(axis)), f[axis], merged) }
        let emb = concatenated([merged, merged], axis: -1)
        return (cos(emb), sin(emb))
    }

    /// rotate the first `dim` features of the last axis; cos/sin broadcast to x
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let d = cos.dim(-1)
        let c = cos.asType(x.dtype), s = sin.asType(x.dtype)
        let xr = x[.ellipsis, 0 ..< d]
        let half = d / 2
        let x1 = xr[.ellipsis, 0 ..< half]
        let x2 = xr[.ellipsis, half ..< d]
        let rot = concatenated([-x2, x1], axis: -1)
        let rotated = xr * c + rot * s
        if x.dim(-1) > d {
            return concatenated([rotated, x[.ellipsis, d...]], axis: -1)
        }
        return rotated
    }
}

// MARK: - P072 behavioural arms (all default to the champion's value; each is a deliberate deviation from the official model)
/// P080 M1 -- ONE batched decode step over sequences of DIFFERENT lengths.
/// `rowOffsets[b]` is row b's committed length before the token being generated. While it is nil the
/// program is the champion's, unchanged; while it is set, S must be 1 and the attention takes a
/// separate ragged branch rather than threading a per-row length through the single-sequence path.
/// P106 H60: diagnostic switches read ONCE. `ProcessInfo.processInfo.environment` builds the whole environment dictionary
/// on every access (~20 us here, 52 variables), and these were read on every forward call (5 per decode forward).
/// Nothing sets ENGINE_* at run time.
enum Qwen4ExpEnv {
    static let env = ProcessInfo.processInfo.environment
    static let dumpCalls = env["ENGINE_DUMP_CALLS"] == "1"
    static let dumpActs: String? = env["ENGINE_DUMP_ACTS"]
    static let dumpLayers: Set<Int>? = env["ENGINE_DUMP_LAYERS"].map { Set($0.split(separator: ",").compactMap { Int($0) }) }
    static let dumpHC0 = env["ENGINE_DUMP_HC0"] == "1"
    static let noPLE = env["ENGINE_NO_PLE"] != nil
    static let dumpTop: String? = env["ENGINE_DUMP_TOP"]
}

public enum Qwen4ExpBatch {
    nonisolated(unsafe) public static var rowOffsets: [Int]? = nil
    public static var active: Bool { rowOffsets != nil }
    /// Run `body` with the per-row lengths installed, and always take them down again.
    public static func with<R>(_ rows: [Int], _ body: () throws -> R) rethrows -> R {
        rowOffsets = rows
        defer { rowOffsets = nil }
        return try body()
    }

    /// P106 H48 -- ROW-RESIDENT KV. A batch pool built by `stackCachesRowResident` does not copy any
    /// row's growing history: every attention layer (trunk and MTP head) gets a MARKER CacheList (a
    /// KVCacheSimple with no buffers at the longest row's offset, an empty indexer ArraysCache), and
    /// the marker's KV object keys that layer's per-row CacheLists here, in pool-slot order. The
    /// ragged branch reads and writes each row's OWN buffers through this table; only the fixed-size
    /// state (GDN conv/ssm, PLE conv, n-gram context) is stacked. Model-thread only, like `rowOffsets`.
    /// The marker is retained strongly next to its rows so its ObjectIdentifier cannot be reused
    /// while registered; `releaseRowResident` removes a pool's entries when it dissolves.
    nonisolated(unsafe) static var rowResidentLists: [ObjectIdentifier: (marker: KVCacheSimple, rows: [CacheList])] = [:]
    /// The per-row caches behind a marker, or nil for an ordinary (stacked or single-sequence) cache.
    @inline(__always) public static func rowResident(_ kv: KVCacheSimple) -> [CacheList]? {
        rowResidentLists.isEmpty ? nil : rowResidentLists[ObjectIdentifier(kv)]?.rows
    }
    public static var rowResidentRegistered: Int { rowResidentLists.count }
}

enum Qwen4ExpVariants {
    static let attnScaleMult: Float = Float(ProcessInfo.processInfo.environment["ENGINE_ATTN_SCALE_MULT"] ?? "") ?? 1
    static let ropeThetaMult: Float = Float(ProcessInfo.processInfo.environment["ENGINE_ROPE_THETA_MULT"] ?? "") ?? 1
    static let gdnDecayMult: Float = Float(ProcessInfo.processInfo.environment["ENGINE_GDN_DECAY_MULT"] ?? "") ?? 1
    static let keepRecent: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_KEEP_RECENT"] ?? "") ?? 0
    static let keepFirst: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_KEEP_FIRST"] ?? "") ?? 0
    /// P073 anti-recency arm: drop the R most recent COMPLETED blocks from the selection, so the top-k has to
    /// come from older context. The query's own partial-block tail stays visible either way (`rowTail`), so the
    /// model never loses the sentence it is writing -- only the last R*ratio tokens of settled context.
    static let recentPenalty: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RECENT_PENALTY"] ?? "") ?? 0
}

// MARK: - QSA indexer

final class Qwen4ExpQSAIndexer: Module {
    let nHeads: Int
    let kvHeads: Int
    let headDim: Int
    let budget: Int
    let ratio: Int
    let blockTopK: Int
    @ModuleInfo(key: "q_layernorm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: Qwen4ExpRMSNorm

    init(_ a: Qwen4ExpTextConfiguration) {
        nHeads = a.indexerNHeads
        kvHeads = a.indexerKVHeads
        headDim = a.indexerHeadDim
        budget = a.indexerBudget
        ratio = a.indexerCompressRatio
        blockTopK = budget / ratio
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        super.init()
    }

    /// Returns a boolean keep mask (B,1,S,kv_len), or nil when everything fits the budget.
    var projDim: Int { (nHeads + kvHeads) * headDim }
    /// `qk` is this layer's slice of the merged qkvi projection: (B,S,(nHeads+kvHeads)*headDim)
    /// pooled-block cache (slot 1 of the indexer ArraysCache) and host index arrays cached per length: ENGINE_IDX_NOCACHE=1 restores
    /// the per-step recomputation (OBS-ENG-027: at 4k-8k context the per-layer host array builds and the O(kvLen) pooling per step
    /// made the step host-bound)
    static let pooledCache: Bool = ProcessInfo.processInfo.environment["ENGINE_IDX_NOCACHE"] == nil
    /// P039 GATE-IS-NOT-INERT ABLATION: force the PRE-P039 scalar rope for pooled block starts and for
    /// the indexer queries even on a visual sequence. If the answer above the budget survives this,
    /// the mRoPE rope was never being tested and the gate is worthless. Outputs are wrong on purpose.
    static let scalarBlockRope: Bool = ProcessInfo.processInfo.environment["ENGINE_IDX_SCALAR_ROPE"] == "1"
    /// P039 ACTIVATION_WITNESS for the pooled-block rope: says which branch actually ran.
    static let ropeWitness: Bool = ProcessInfo.processInfo.environment["ENGINE_VISION_WITNESS"] == "1"
    nonisolated(unsafe) static var ropeWitnessDone = false
    /// ENGINE_IDX_POOL_BUF=0 restores the per-call concatenation of the pooled history (slot 1)
    static let pooledBuffer: Bool = (ProcessInfo.processInfo.environment["ENGINE_IDX_POOL_BUF"] ?? "1") != "0"
    static let decodeMode: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_DECODE_MODE"] ?? "0") ?? 0
    static let gatherMinKV: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_GATHER_MIN_KV"] ?? "12288") ?? 12288
    /// ENGINE_IDX_STEP: growth quantum of the indexer key buffer (slot 0); growth also adds capacity/8 so the copies stay amortised
    static let idxStep: Int = Int(ProcessInfo.processInfo.environment["ENGINE_IDX_STEP"] ?? "1024") ?? 1024
    /// ENGINE_QSA_TOPK=0 restores MLX argPartition; default = the radix top-K kernel (P019: 27.2 -> 2.5 ms per layer at 2048 x 65536)
    static let topKSelect: Bool = ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK"] != "0"
    /// P023 unit 5 witness (see the decode branch of the score); off the hot path in every measured run
    static let decWitness: Bool = ProcessInfo.processInfo.environment["ENGINE_IDX_SCORE_DEC_WITNESS"] != nil
    nonisolated(unsafe) static var decWitnessed = false
    /// P023 unit 6, in-situ ablation (TIMING ONLY, the selection and therefore the output are wrong):
    /// ENGINE_ABL_SKIP=topk replaces the block selection with the first k ids. Every downstream shape and
    /// every other kernel is unchanged, so the difference is the selection kernel and nothing else.
    static let ablTopK: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("topk")
    /// ENGINE_ABL_SKIP=idxscore replaces the SCORE with a cached ramp of the same shape and dtype. The top-k still
    /// runs, over non-degenerate strictly increasing keys, so `full - idxscore` is the score and
    /// `idxscore - topk` is the selection. TIMING ONLY.
    static let ablIdxScore: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("idxscore")
    /// P026 unit 6, and a CORRECTION to `topk` as an attribution: ENGINE_ABL_SKIP=topkspread replaces
    /// the selection with k ids EVENLY SPREAD over nBlocks (stride nBlocks/k) instead of the first k.
    /// WHY IT HAD TO EXIST. `topk`'s ids are 0,1,...,k-1 -- perfectly CONTIGUOUS -- while the real
    /// selection returns k ids scattered over nBlocks. The shapes are identical, which is what the
    /// `topk` comment relied on, but the VALUES set the downstream gather attention's access pattern,
    /// and a contiguous gather is not a scattered one. So `idxscore - topk` is the selection kernel
    /// PLUS a locality term, and P026 measured every intervention on the kernel at zero because the
    /// kernel was never the size that difference suggested. `idxscore - topkspread` is the kernel
    /// ALONE; `topkspread - topk` is the locality. TIMING ONLY, like its siblings.
    static let ablTopKSpread: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("topkspread")
    /// P029 unit 1: ENGINE_ABL_SKIP=qsagather returns the QUERY in place of the sparse-attention output.
    /// `q4AttnFront` emits q as [B,H,1,D] and `q4QSAGather` emits [B,H,S,D] with S=1 -- the SAME shape and
    /// dtype -- so the KV cache update, the indexer, the selection, the gate and o_proj all still run and
    /// the difference is the gather kernel alone. It is the one term of the attention layer that reads the
    /// 6.4 GB KV cache, and its access pattern is set by the selected ids. TIMING ONLY.
    static let ablQSAGather: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("qsagather")
    /// P095: ENGINE_RAGGED_SELECT=masked -- when the rows of a ragged step have DIFFERENT block counts,
    /// select once over `nBlocksMax` with every block past a row's own count masked to -inf, instead
    /// of one call per row. Same SET of block ids per row; the ORDER the radix top-k returns them in
    /// can differ, and the gather's online softmax sums in that order, so this is a numerics change
    /// at B > 1 only (B = 1 and equal-count batches never reach it). Operator-approved contract
    /// (P095): B = 1 stays max|d| = 0, the B = 4 lockstep divergence rate must not worsen.
    static let raggedSelectMasked: Bool = (ProcessInfo.processInfo.environment["ENGINE_RAGGED_SELECT"] ?? "") == "masked"

    /// P029 unit 9, a WITNESS and nothing else (it forces a host sync, so it is never on in a timed
    /// run). The block SELECTION costs 152.8 us per attention layer, 1.834 ms/token, 8.56% of the
    /// decode step at 262144, and four structural transformations of the kernel read ZERO
    /// (OBS-ENG-086). Every one of them asked how to make the selection CHEAPER. None asked whether
    /// it has to run EVERY TOKEN. Between two consecutive decode tokens the query moves by one
    /// position and the pooled key table grows by at most one block, so the selected set may barely
    /// move -- or may not; the KB does not know, and the answer bounds a whole axis.
    /// ENGINE_QSA_TOP_STABILITY=1 reports, per layer, |new ∩ prev| / k and where the newcomers sit.
    static let stabWitness: Bool = ProcessInfo.processInfo.environment["ENGINE_QSA_TOP_STABILITY"] != nil

    /// P032 -- the REFERENCE DESIGN, which this engine did not implement. The technical report states
    /// it twice: "The MTP module reuses QSA indices across speculative decoding steps" (Fig. 1) and
    /// "multi-step MTP reuses top-k indices across prediction steps to further reduce draft-model
    /// inference costs", and prices the quality cost at nil (Table 4: mean accepted length 4.06 full
    /// attention vs 4.07 with reuse, four-step speculative decoding).
    /// Set ONLY on the MTP head's indexer. The TRUNK is untouched: REFUT-ENG-016 closed reuse there,
    /// and the head is a different consumer -- a stale selection costs ACCEPTANCE and never
    /// correctness, because the trunk verifies every draft.
    /// `reusePeriod` = recompute every N head calls; N = 1 is the current program and the control.
    /// THE POOLED-KEY CACHE IS STILL MAINTAINED ON EVERY CALL and only the score and the top-k are
    /// skipped -- REFUT-ENG-018 is what happens when a cache stops being updated, and a later
    /// recompute must be correct.
    nonisolated(unsafe) var reusePeriod: Int = 1
    nonisolated(unsafe) var reuseTop: MLXArray? = nil
    nonisolated(unsafe) var reuseSince: Int = 0

    /// P030: does the EXACTNESS of the top-K buy anything? The selection costs 1.834 ms/token, 8.56%
    /// of the decode step (OBS-ENG-089 (4)/(7)), four structural transformations of the kernel read
    /// zero (REFUT-ENG-014) and reuse across tokens is dead (REFUT-ENG-016) -- so the only remaining
    /// lever on it is to stop computing an EXACT answer. The sharpest possible test of whether the
    /// ranking means anything: take ranks [J, J+k) instead of [0, k). J = k gives a set DISJOINT from
    /// the exact one. J = 0 is a CONTROL that must reproduce the champion, ids and all: the ids are
    /// returned ASCENDING, which is the index order `q4TopK`'s deterministic compaction produces, and
    /// the gather sums in id order -- so a J = 0 run that is not bit-identical means the instrument,
    /// not the model, moved. DECODE ONLY (S == 1); prefill and the pooled/KV state stay the champion's.
    static let rankShift: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_RANK_SHIFT"] ?? "") ?? -1
    static func shiftedTop(_ scores: MLXArray, k: Int, shift J: Int) -> MLXArray {
        let order = argSort(-scores, axis: -1)                 // full rank order over nBlocks
        return sorted(order[.ellipsis, J ..< (J + k)], axis: -1)
    }
    /// P030 unit 3. The rank SHIFT conflates two different errors: it drops the J BEST blocks and it
    /// adds J blocks past the K-th. A real approximate selector does not make that error -- the top
    /// ranks sit far above the threshold and any selector finds them; what it gets wrong is the
    /// BOUNDARY. `ENGINE_QSA_BOUNDARY_SWAP=J` keeps ranks [0, k-J) EXACT and replaces the worst J with
    /// ranks [k, k+J). J = k reproduces the disjoint set of the shift knob, which is the cross-check.
    static let boundarySwap: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_BOUNDARY_SWAP"] ?? "") ?? -1
    static func boundarySwapTop(_ scores: MLXArray, k: Int, swap J: Int) -> MLXArray {
        let order = argSort(-scores, axis: -1)
        if J <= 0 { return sorted(order[.ellipsis, 0 ..< k], axis: -1) }
        let head = order[.ellipsis, 0 ..< (k - J)]
        let tail = order[.ellipsis, k ..< (k + J)]
        return sorted(concatenated([head, tail], axis: -1), axis: -1)
    }
    /// P030 unit 4, the ACTIONABLE form of the question. Unit 3 showed ranks 257-512 can be replaced
    /// wholesale and the model cannot tell. That licenses the next question: are they needed AT ALL?
    /// `ENGINE_QSA_TOPK_KEEP=M` keeps the top M ranks and fills the rest with the id `nBlocks`, which
    /// every gather kernel skips (`if (blk >= nBlocks) continue`), so the attention normalises over M
    /// blocks instead of K. This is a STRONGER perturbation than a swap -- it changes the softmax
    /// denominator, not just which near-threshold blocks are in it -- and it is the one that would
    /// license halving `indexer_budget`.
    static let topKKeep: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_TOPK_KEEP"] ?? "") ?? -1
    static func keepTop(_ scores: MLXArray, k: Int, keep M: Int, nBlocks: Int) -> MLXArray {
        let order = argSort(-scores, axis: -1)
        if M >= k { return sorted(order[.ellipsis, 0 ..< k], axis: -1) }
        let head = order[.ellipsis, 0 ..< M]
        let fill = MLXArray.full([head.dim(0), head.dim(1), k - M], values: MLXArray(Int32(nBlocks)))
        return sorted(concatenated([head, fill.asType(head.dtype)], axis: -1), axis: -1)
    }
    /// The J = 0 control must select the SAME SET as `q4TopK`. If it does not, a divergence measured
    /// under a shift is the instrument and not the model, so this is checked rather than assumed.
    /// ENGINE_QSA_RANK_SHIFT_CHECK=1; forces a host sync, witness only.
    static let rankShiftCheck: Bool = ProcessInfo.processInfo.environment["ENGINE_QSA_RANK_SHIFT_CHECK"] != nil
    nonisolated(unsafe) static var rsChecked = 0
    nonisolated(unsafe) static var rsMismatch = 0
    nonisolated(unsafe) static var rsRows = 0
    nonisolated(unsafe) static var rsOrder = 0
    static func rankShiftCompare(_ shifted: MLXArray, _ scores: MLXArray, k: Int) {
        let exact = topKSelect ? q4TopK(scores: scores, k: k) : argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k]
        eval(shifted, exact)
        let av = shifted.asArray(Int32.self), bv = exact.asArray(Int32.self)
        let a = Set(av), b = Set(bv)
        rsRows += 1; rsMismatch += b.subtracting(a).count; rsChecked += k
        // ORDER matters independently of the SET: the gather sums the selected blocks in the order it
        // is handed, so two equal sets in different orders round differently. Count both.
        rsOrder += zip(av, bv).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
        if rsRows % 128 == 0 {
            FileHandle.standardError.write(String(format:
                "qsa-shift-check: rows=%d  ids compared=%d  SET diff (in exact, not in shifted): %d (%.4f%%)  POSITIONAL diff: %d (%.2f%%)\n",
                rsRows, rsChecked, rsMismatch, 100.0 * Double(rsMismatch) / Double(rsChecked),
                rsOrder, 100.0 * Double(rsOrder) / Double(rsChecked)).data(using: .utf8)!)
        }
    }
    nonisolated(unsafe) var stabPrev: [Int32]? = nil
    nonisolated(unsafe) var stabCalls = 0
    nonisolated(unsafe) var stabOverlap: [Int] = []
    nonisolated(unsafe) var stabFresh = 0
    nonisolated(unsafe) var stabLayerTag = -1
    /// ENGINE_QSA_GATHER_MAX_S: largest row block routed through the gather kernel (P018: 512 covers the prefill chunk -- the masked
    /// sdpa fallback at head_dim 256 is dense over the whole kv, 214 ms per layer-chunk at 256k vs 7.7 ms gathered)
    static let gatherMaxS: Int = Int(ProcessInfo.processInfo.environment["ENGINE_QSA_GATHER_MAX_S"] ?? "4096") ?? 4096
    nonisolated(unsafe) static var dumpedTop = false
    /// gather path: the last call's (block ids [B,S,k] int32 with nBlocks as the invisible id, nBlocks, kvLen); consumed by the attention right after
    var lastTop: (MLXArray, Int, Int)? = nil
    /// P080 M1: (block ids, per-row completed blocks) for a ragged decode step
    var lastTopRagged: (MLXArray, [Int])? = nil
    /// P080 U2: how many blocks of each row are already pooled. The single-sequence path derives
    /// this from `offset / ratio`; a batch cannot, because one shared watermark would re-pool every
    /// row's whole history on every step.
    var raggedPooled: [Int] = []
    /// P081 witness sink: `ENGINE_RAGGED_WITNESS=dir` saves row 0's scores and selected ids per call.
    nonisolated(unsafe) static let witnessDir: String? = ProcessInfo.processInfo.environment["ENGINE_RAGGED_WITNESS"]
    nonisolated(unsafe) static var witnessCalls = 0
    nonisolated(unsafe) static var witnessAttn = 0
    /// Untimed diagnostic only: one S1 fixture per allowed budget. Default is trunk2048;
    /// an optional head fixture must be explicitly requested before a timed workload starts.
    /// An unset directory adds no tensor retention, evaluation or I/O to serving.
    static let replayDumpDir = ProcessInfo.processInfo.environment["ENGINE_IDX_REPLAY_DUMP"]
    static let replayDumpBudgets: Set<Int> = Set(
        (ProcessInfo.processInfo.environment["ENGINE_IDX_REPLAY_BUDGETS"] ?? "2048")
            .split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
    nonisolated(unsafe) private static var replayDumpedBudgets = Set<Int>()
    nonisolated(unsafe) private static var rangeCache: [String: MLXArray] = [:]
    private static let rangeLock = NSLock()
    /// Int32 [from, to) as an MLXArray, cached (built once per distinct range)
    static func int32Range(_ from: Int, _ to: Int, scale: Int = 1) -> MLXArray {
        let key = "\(from):\(to):\(scale)"
        rangeLock.lock(); defer { rangeLock.unlock() }
        if let a = rangeCache[key] { return a }
        let a = MLXArray((from ..< to).map { Int32($0 * scale) })
        eval(a)
        if rangeCache.count > 4096 { rangeCache.removeAll() }
        rangeCache[key] = a
        return a
    }

    /// P080 M1 -- the indexer for a RAGGED decode step: one token per row, rows at DIFFERENT lengths.
    /// The arithmetic is `callAsFunction`'s; every length that is a scalar there is a row here.
    /// Returns the per-row completed-block counts, or nil when a row is still inside the budget (the
    /// dense window), which M1 does not batch.
    /// P106 H48: `rowCaches` non-nil = ROW-RESIDENT KV -- row b's slot 0 / slot 1 are ITS OWN capacity
    /// buffers (keyCache, the marker's, is untouched). Every quantity is the stacked path's: the same
    /// positions written, the same blocks pooled from the same raw keys, the same per-row score/top-k
    /// calls; the equal-count batched score reads a concatenation of the rows' pooled views, the same
    /// values in the same [B, nBlocks, D] layout `asType` makes of the stacked view. Growth is per row
    /// and carries only the row's valid prefix.
    func raggedDecode(qk: MLXArray, rope: Qwen4ExpRotary, keyCache: ArraysCache, rows: [Int],
                      rowCaches: [ArraysCache]? = nil) -> [Int]? {
        lastTop = nil; lastTopRagged = nil
        let B = qk.dim(0), S = qk.dim(1)
        // P099: S > 1 query rows per row (a verify block per row); S == 1 is the P080 path, untouched.
        let split = nHeads * headDim
        let q = qk[.ellipsis, 0 ..< split].reshaped(B, S, nHeads, headDim)
        let rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)
        let kvLenRow = rows.map { $0 + S }
        let kvLenMax = kvLenRow.max()!
        if let rc = rowCaches { precondition(rc.count == B, "row-resident indexer: one cache per row") }
        // slot 0 stays a capacity buffer; the difference is that each row writes at its own position
        let allK: MLXArray?
        if let rc = rowCaches {
            for b in 0 ..< B {
                let kc = rc[b]
                let cap = kc[0]?.dim(1) ?? 0
                if kc[0] == nil || kvLenRow[b] > cap {
                    let want = kvLenRow[b] + max(Self.idxStep, cap / 8)
                    let newCap = Qwen4ExpCacheCapacity.bounded(((want + Self.idxStep - 1) / Self.idxStep) * Self.idxStep, need: kvLenRow[b])
                    let nb = MLXArray.zeros([1, newCap, headDim], dtype: rawK.dtype)
                    if let old = kc[0] { let keep = min(cap, rows[b]); if keep > 0 { nb[0..., 0 ..< keep, 0...] = old[0..., 0 ..< keep, 0...] } }
                    kc[0] = nb
                }
                kc[0]![0..., rows[b] ..< (rows[b] + S), 0...] = rawK[b ..< (b + 1)]
            }
            allK = nil
        } else {
            let cap = keyCache[0]?.dim(1) ?? 0
            if keyCache[0] == nil || kvLenMax > cap {
                let want = kvLenMax + max(Self.idxStep, cap / 8)
                let newCap = Qwen4ExpCacheCapacity.bounded(((want + Self.idxStep - 1) / Self.idxStep) * Self.idxStep, need: kvLenMax)
                let nb = MLXArray.zeros([B, newCap, headDim], dtype: rawK.dtype)
                if let old = keyCache[0] { let keep = min(cap, newCap); nb[0..., 0 ..< keep, 0...] = old[0..., 0 ..< keep, 0...] }
                keyCache[0] = nb
            }
            for b in 0 ..< B { keyCache[0]![b ..< (b + 1), rows[b] ..< (rows[b] + S), 0...] = rawK[b ..< (b + 1)] }
            allK = keyCache[0]![0..., 0 ..< kvLenMax, 0...]
        }
        let nBlocksRow = kvLenRow.map { $0 / ratio }
        let nBlocksMax = nBlocksRow.max()!
        guard kvLenRow.min()! > budget, nBlocksMax > 0 else { return nil }
        // P080 U2 -- ONE WATERMARK PER ROW, and it is a correctness fix, not only a cost one.
        // The first cut used a single `min` over the batch and re-pooled every block of every row on
        // every step, trusting that re-pooling is idempotent. The values are the same function of the
        // same keys, but the REDUCTION is not the same shape: the serial path pools a block inside a
        // small `(1, n_b, ratio, D)` mean while the batch pooled it inside a
        // `(B, nBlocksMax - done, ratio, D)` one, and that extent is not required to reduce to the
        // same bits. It also cost ~98k block-poolings per step at B=8 with a 4k spread. Each row now
        // pools exactly the blocks it actually gained -- normally none or one -- from exactly the
        // rows of its own buffer the single-sequence path would have used.
        if raggedPooled.count != B { raggedPooled = rows.map { $0 / ratio } }
        let pooledStacked: MLXArray?
        if let rc = rowCaches {
            for b in 0 ..< B {
                let kc = rc[b]
                let poolCap = kc[1]?.dim(1) ?? 0
                // a watermark past what the row's own buffer holds would read zeros as pooled blocks
                // (the stacked path's `allSatisfy { $0[1] != nil }` gap); re-pool from the raw keys instead
                if raggedPooled[b] > poolCap { raggedPooled[b] = poolCap }
                if kc[1] == nil || nBlocksRow[b] > poolCap {
                    let want = Qwen4ExpCacheCapacity.bounded(nBlocksRow[b] + max(Self.idxStep / ratio, poolCap / 8), need: nBlocksRow[b], ratio: ratio)
                    let nb = MLXArray.zeros([1, want, headDim], dtype: rawK.dtype)
                    if let old = kc[1] { let keep = min(poolCap, raggedPooled[b]); if keep > 0 { nb[0..., 0 ..< keep, 0...] = old[0..., 0 ..< keep, 0...] } }
                    kc[1] = nb
                }
                if raggedPooled[b] < nBlocksRow[b] {
                    let from = raggedPooled[b], to = nBlocksRow[b]
                    let newRaw = kc[0]![0..., (from * ratio) ..< (to * ratio), 0...]
                        .reshaped(1, to - from, ratio, headDim)
                    var p = kNorm(newRaw.asType(.float32).mean(axis: 2).asType(rawK.dtype))
                    let starts = Self.int32Range(from, to, scale: ratio)
                    let (cosK, sinK) = rope.tables(expandedDimensions(starts, axis: 0))
                    p = Qwen4ExpRotary.apply(p, cos: cosK, sin: sinK)
                    kc[1]![0..., from ..< to, 0...] = p
                    raggedPooled[b] = to
                }
            }
            pooledStacked = nil
        } else {
            let allK = allK!
            let poolCap = keyCache[1]?.dim(1) ?? 0
            if keyCache[1] == nil || nBlocksMax > poolCap {
                let want = Qwen4ExpCacheCapacity.bounded(nBlocksMax + max(Self.idxStep / ratio, poolCap / 8), need: nBlocksMax, ratio: ratio)
                let nb = MLXArray.zeros([B, want, headDim], dtype: allK.dtype)
                if let old = keyCache[1] { let keep = min(poolCap, want); nb[0..., 0 ..< keep, 0...] = old[0..., 0 ..< keep, 0...] }
                keyCache[1] = nb
            }
            for b in 0 ..< B where raggedPooled[b] < nBlocksRow[b] {
                let from = raggedPooled[b], to = nBlocksRow[b]
                let newRaw = allK[b ..< (b + 1), (from * ratio) ..< (to * ratio), 0...]
                    .reshaped(1, to - from, ratio, headDim)
                var p = kNorm(newRaw.asType(.float32).mean(axis: 2).asType(allK.dtype))
                let starts = Self.int32Range(from, to, scale: ratio)
                let (cosK, sinK) = rope.tables(expandedDimensions(starts, axis: 0))
                p = Qwen4ExpRotary.apply(p, cos: cosK, sin: sinK)
                keyCache[1]![b ..< (b + 1), from ..< to, 0...] = p
                raggedPooled[b] = to
            }
            pooledStacked = keyCache[1]![0..., 0 ..< nBlocksMax, 0...]
        }
        /// row b's first `nb` pooled blocks: a slice of the stacked view, or of the row's own buffer
        func pooledRow(_ b: Int, _ nb: Int) -> MLXArray {
            if let pooled = pooledStacked { return pooled[b ..< (b + 1), 0 ..< nb, 0...] }
            return rowCaches![b][1]![0..., 0 ..< nb, 0...]
        }
        // the query of row b is roped at row b's own position (rows[b] + s for the s-th row of its block)
        let qPos: [Int32] = rows.flatMap { r in (0 ..< S).map { Int32(r + $0) } }
        let (cosQ, sinQ) = rope.tables(rowPositions: qPos, B: B, S: S)   // (B,S,d)
        var qn = qNorm(q)
        qn = Qwen4ExpRotary.apply(qn, cos: expandedDimensions(cosQ, axis: 2), sin: expandedDimensions(sinQ, axis: 2))
        if S > 1 {
            // P099: each row's block scored and selected exactly as the serial verify block scores its own
            // (the decode-branch fused score at TM = 8 with the row's visibility, the same top-k, the same
            // end-of-block fix), one call per row so the call has the serial shape. No ablation keys here.
            let k = min(blockTopK, nBlocksMax)
            var perRowTop: [MLXArray] = []
            for b in 0 ..< B {
                let nb = nBlocksRow[b], kvLenB = kvLenRow[b]
                let pb = pooledRow(b, nb)
                let qb = qn[b ..< (b + 1)]
                var scores: MLXArray
                let scoreMI = q4IdxScoreMI, scoreNI = q4IdxScoreNI, scoreSG = q4IdxScoreSG
                if q4IdxScoreDecodeOn, q4IdxScoreFusedOn, S < scoreSG * 8 * scoreMI, qb.dtype == pb.dtype, nb >= 8 * scoreNI {
                    var mi = 1
                    while 8 * mi < S { mi *= 2 }
                    let TM = 8 * mi
                    let qpad = S == TM ? qb : concatenated([tiled(qb[0..., 0 ..< 1, 0..., 0...], repetitions: [1, TM - S, 1, 1]), qb], axis: 1)
                    let sT = q4IdxScore(q: qpad, pooled: pb, scale: sqrt(Float(headDim)),
                                        visibility: (kvLen: kvLenB, ratio: ratio), mi: mi, ni: scoreNI, sg: 1)
                    scores = sT[0..., (TM - S) ..< TM, 0...]
                    Qwen4ExpIndexerRouteWitness.record(.raggedVerifyFused, batch: B, sequence: S,
                                                       budget: budget, scoreBatch: 1, nBlocks: nb)
                } else {
                    let q32b = qb.asType(.float32).reshaped(1, S * nHeads, headDim)
                    let raw = matmul(q32b, pb.asType(.float32).transposed(0, 2, 1)).reshaped(1, S, nHeads, nb)
                    scores = q4IdxFused ? q4IdxReluSum(raw, scale: sqrt(Float(headDim)), visibility: (kvLen: kvLenB, ratio: ratio))
                                        : maximum(raw.asType(.float32), MLXArray(Float(0))).sum(axis: 2) / sqrt(Float(headDim))
                    if !q4IdxFused {
                        let kvPos = Self.int32Range(kvLenB - S, kvLenB)
                        let blockEnd = Self.int32Range(0, nb, scale: ratio) + Int32(ratio - 1)
                        let visible = expandedDimensions(blockEnd, axes: [0, 1]) .<= expandedDimensions(kvPos, axes: [0, 2])
                        scores = which(visible, scores, MLXArray(-Float.infinity))
                    }
                    Qwen4ExpIndexerRouteWitness.record(.raggedVerifyGraph, batch: B, sequence: S,
                                                       budget: budget, scoreBatch: 1, nBlocks: nb)
                }
                let kb = min(blockTopK, nb)
                let kvPos = Self.int32Range(kvLenB - S, kvLenB)
                var t = Self.topKSelect ? q4TopK(scores: scores, k: kb) : argPartition(-scores, kth: kb - 1, axis: -1)[.ellipsis, 0 ..< kb]
                let endSel = t * Int32(ratio) + Int32(ratio - 1)
                t = which(endSel .<= expandedDimensions(kvPos, axes: [0, 2]), t, MLXArray(Int32(nb)))
                if kb < k { t = concatenated([t, MLXArray.full([1, S, k - kb], values: MLXArray(Int32(nb)))], axis: -1) }
                perRowTop.append(t.asType(.int32))
            }
            let top = perRowTop.count == 1 ? perRowTop[0] : concatenated(perRowTop, axis: 0)   // (B,S,k)
            lastTopRagged = (top, nBlocksRow)
            return nBlocksRow
        }
        let q32 = qn.asType(.float32).reshaped(B, nHeads, headDim)
        // the SAME score the single-sequence path computes in its `else` branch -- the fused relu-sum
        // kernel, not a graph transcription of it. A graph form differs in the last bits and that is
        // enough to flip a top-k tie, which is what made the first cut of this path wander.
        // P081 U1 ROOT CAUSE. Every stage from the score down must be computed over the ROW'S OWN
        // block count, not over `nBlocksMax`, and the reason is a three-link chain that the lockstep
        // instrument made visible:
        //   1. `q4IdxReluSum` reduces over H on a grid sized by N, so N = nBlocksMax and N = nBlocks_b
        //      give scores that differ in the last bits;
        //   2. the RADIX top-k buckets by bit pattern, so a one-ulp score difference returns the same
        //      SET of block ids in a different ORDER (MLX's argPartition, being score-ordered, hides
        //      this -- with ENGINE_QSA_TOPK=0 the failing case is bit-exact, which is what located it);
        //   3. the gather kernel's online softmax accumulates in `top` order, so a reordering is a
        //      different sum.
        // When every row has the same block count the batched call is provably the same call, and
        // that is measured (equal-length batches are bit-exact to B=8), so keep the one-shot path
        // there and pay the per-row dispatches only when the rows actually disagree.
        // EVERY stage per row, and the reason is measured rather than assumed: a batched matmul and
        // a single-row one differ in the last bits, the RADIX top-k turns a one-ulp score difference
        // into a different ORDER of the selected ids (MLX's score-ordered argPartition hides it --
        // with ENGINE_QSA_TOPK=0 the failing case is bit-exact, which is what located this), and the
        // gather kernel's online softmax accumulates in `top` order, so a reordering is a different
        // sum. One row per call is the shape the single-sequence path uses, so it is the only shape
        // that can be bit-equal to it. The cost is B small dispatches per attention layer per step.
        let k = min(blockTopK, nBlocksMax)
        var perRowTop: [MLXArray] = []
        var witnessScores: MLXArray? = nil          // row 0's score vector, for the P081 witness
        // When every row has the same block count the batched call IS the single-sequence call at
        // B rows, and that is measured bit-exact to B=8 (equal lengths at 4096), so do not pay B
        // dispatches for it. When the counts differ, only a per-row call has serial's shape.
        let equalCounts = nBlocksRow.allSatisfy({ $0 == nBlocksMax })
        if equalCounts || Self.raggedSelectMasked || Self.ablTopK || Self.ablTopKSpread {
            // row-resident: the rows' own pooled views stacked; masked/ablation arms never run row-resident
            // (their unequal rows would need a padded view -- `rowResidentEligible` refuses them)
            // H48b: one pass from the rows' own buffers straight to the f32 operand (no bf16 concatenation)
            let pooled32 = pooledStacked.map { $0.asType(.float32) }
                ?? ((q4PooledRowsF32On && B <= 8) ? q4PooledRowsF32(rowCaches!.map { $0[1]! }, n: nBlocksMax)
                    : concatenated((0 ..< B).map { pooledRow($0, nBlocksMax) }, axis: 0).asType(.float32))
            let raw = matmul(q32, pooled32.transposed(0, 2, 1)).reshaped(B, 1, nHeads, nBlocksMax)
            var scores = q4IdxFused ? q4IdxReluSum(raw, scale: sqrt(Float(headDim)), visibility: nil)
                                    : maximum(raw.asType(.float32), MLXArray(Float(0))).sum(axis: 2) / sqrt(Float(headDim))
            if Qwen4ExpIndexerRouteWitness.enabled {
                Qwen4ExpIndexerRouteWitness.record(equalCounts ? .raggedS1GraphEqual : .raggedS1GraphMasked,
                                                   batch: B, sequence: S, budget: budget, scoreBatch: B,
                                                   nBlocks: nBlocksMax, visibleMinBlocks: nBlocksRow.min())
            }
            // P095 ablation keys, ported from the single-sequence indexer: TIMING ONLY, same semantics
            if Self.ablIdxScore { scores = tiled(Self.int32Range(0, nBlocksMax).asType(.float32).reshaped(1, 1, nBlocksMax), repetitions: [B, 1, 1]) }
            if !equalCounts {
                // P095 masked arm: a block past the row's own count is invisible to that row (-inf),
                // exactly what the per-row slice below achieves by shape; the gather kernel already
                // masks ids >= the row's nBlocks, so a masked id that leaks through the top-k is inert.
                let blockIdx = Self.int32Range(0, nBlocksMax).reshaped(1, 1, nBlocksMax)
                let nbRow = MLXArray(nBlocksRow.map { Int32($0) }).reshaped(B, 1, 1)
                scores = which(blockIdx .< nbRow, scores, MLXArray(-Float.infinity))
            }
            witnessScores = scores[0 ..< 1, 0..., 0...]
            let t: MLXArray = Self.ablTopK ? tiled(Self.int32Range(0, k).reshaped(1, 1, k), repetitions: [B, 1, 1])
                : Self.ablTopKSpread ? tiled(Self.int32Range(0, k, scale: max(1, nBlocksMax / k)).reshaped(1, 1, k), repetitions: [B, 1, 1])
                : (Self.topKSelect ? q4TopK(scores: scores, k: k) : argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k])
            perRowTop.append(t.asType(.int32))
        } else {
        for b in 0 ..< B {
            let nb = nBlocksRow[b]
            let pb = pooledRow(b, nb).asType(.float32).transposed(0, 2, 1)
            let rawB = matmul(q32[b ..< (b + 1)], pb).reshaped(1, 1, nHeads, nb)
            var sb = q4IdxFused ? q4IdxReluSum(rawB, scale: sqrt(Float(headDim)), visibility: nil)
                                : maximum(rawB.asType(.float32), MLXArray(Float(0))).sum(axis: 2) / sqrt(Float(headDim))
            Qwen4ExpIndexerRouteWitness.record(.raggedS1GraphUnequal, batch: B, sequence: S,
                                               budget: budget, scoreBatch: 1, nBlocks: nb)
            if Self.ablIdxScore { sb = Self.int32Range(0, nb).asType(.float32).reshaped(1, 1, nb) }
            let kb = min(blockTopK, nb)
            var t = Self.topKSelect ? q4TopK(scores: sb, k: kb)
                                    : argPartition(-sb, kth: kb - 1, axis: -1)[.ellipsis, 0 ..< kb]
            if kb < k {
                // padding sits at the end and carries an id this row's own `nBlocks` makes invisible
                t = concatenated([t, MLXArray.full([1, 1, k - kb], values: MLXArray(Int32(nb)))], axis: -1)
            }
            if b == 0 { witnessScores = sb }
            perRowTop.append(t.asType(.int32))
        }
        }
        let top = perRowTop.count == 1 ? perRowTop[0] : concatenated(perRowTop, axis: 0)
        if Self.replayDumpDir != nil, Self.replayDumpBudgets.contains(budget), B == 8, nBlocksRow.min()! >= 64_000,
           let pooled = pooledStacked {
            dumpS1Replay(q: qn, pooled: pooled, storage: keyCache[1]!, top: top,
                         rows: rows, nBlocks: nBlocksRow,
                         route: equalCounts ? "equal" : (Self.raggedSelectMasked ? "masked" : "unequal"))
        }
        // P081 WITNESS: row 0's score vector and its selected block ids, per indexer call. Comparing
        // these between a B=1 and a B=2 run of the SAME row says whether the batch changes the
        // SCORES (upstream arithmetic) or only their ORDER (the selector).
        if let dir = Self.witnessDir, Self.witnessCalls < 256 {
            let i = Self.witnessCalls; Self.witnessCalls += 1
            let r0 = top[0 ..< 1, 0..., 0...].asType(.int32)
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? MLX.save(arrays: ["top": r0, "scores": (witnessScores ?? r0.asType(.float32))],
                          url: URL(fileURLWithPath: dir + "/call\(i).safetensors"))
        }
        // P080 U1 -- THE TOP-K MUST RUN OVER THE ROW'S OWN UNIVERSE, one call per row.
        // MEASURED: with equal lengths and matched kernels the batched step is bit-identical to
        // serial (max|d| = 0 over 128 comparisons). The entire residual gap at UNEQUAL lengths came
        // from selecting over `nBlocksMax` with `-inf` masking: that returns the same SET as serial
        // but in a different ORDER, and the gather's online softmax accumulates in `top` order, so a
        // reordering is a different sum. Slicing each row's scores to `[0, nBlocksRow[b])` makes the
        // call identical to the one the single-sequence path makes, which restores the order.
        // `K` itself stays uniform (every row past the budget has nBlocks > blockTopK, so every row
        // asks for exactly `blockTopK`), and that matters too: the kernel strides its virtual
        // position list across simdgroups by `NV = K*R + 16`, so a per-row K would change the
        // reduction tree even if the ids matched.
        lastTopRagged = (top, nBlocksRow)
        return nBlocksRow
    }

    /// Snapshot AFTER the actual score/selection graph was constructed. Evaluating/saving here
    /// deliberately perturbs this diagnostic call; never enable this during a timed workload.
    /// Keep the whole capacity tensor: saving only `pooled` would erase its inter-row gaps and
    /// the fused kernel's real ensureRowContiguous copy on replay.
    private func dumpS1Replay(q: MLXArray, pooled: MLXArray, storage: MLXArray, top: MLXArray,
                              rows: [Int], nBlocks: [Int], route: String) {
        guard let dir = Self.replayDumpDir,
              Self.replayDumpedBudgets.insert(budget).inserted else { return }
        guard !Self.ablIdxScore, !Self.ablTopK, !Self.ablTopKSpread else {
            FileHandle.standardError.write(Data("idx replay dump refused: ablated S1 graph\n".utf8))
            return
        }
        do {
            eval(q, pooled, storage, top)
            func layout(_ a: MLXArray) -> String {
                a.asData(access: .noCopy).strides.map(String.init).joined(separator: ",")
            }
            // This is the reachable storage span, NOT the Metal allocator's allocation size.
            func spanBytes(_ a: MLXArray) -> Int {
                let strides = a.asData(access: .noCopy).strides
                return (1 + zip(a.shape, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }) * a.itemSize
            }
            var metadata = Q4MixerBench.idxScoreReplayConfiguration
            let geometry: [String: String] = [
                "schema": "engine.idx-s1-replay.v1", "route": route,
                "budget": String(budget), "ratio": String(ratio), "k": String(top.dim(2)),
                "batch": String(q.dim(0)), "sequence": String(q.dim(1)),
                "heads": String(q.dim(2)), "head_dim": String(q.dim(3)),
                "nblocks_min": String(nBlocks.min()!), "nblocks_max": String(nBlocks.max()!),
                "pool_capacity": String(storage.dim(1)),
                "dump_budget_filter": Self.replayDumpBudgets.sorted().map(String.init).joined(separator: ","),
            ]
            metadata.merge(geometry) { _, new in new }
            let layouts: [String: String] = [
                "ragged_batch_active": String(Qwen4ExpBatch.active),
                "ragged_select_masked": String(Self.raggedSelectMasked),
                "q_dtype": String(describing: q.dtype), "pool_dtype": String(describing: storage.dtype),
                "q_strides": layout(q), "pool_storage_strides": layout(storage),
                "pool_view_strides": layout(pooled),
                "pool_view_logical_bytes": String(pooled.nbytes),
                "pool_storage_tensor_bytes": String(storage.nbytes),
                "pool_storage_span_bytes": String(spanBytes(storage)),
                "pool_view_span_bytes": String(spanBytes(pooled)),
            ]
            metadata.merge(layouts) { _, new in new }
            let provenance: [String: String] = [
                "timestamp_unix_s": String(Date().timeIntervalSince1970),
                "pid": String(ProcessInfo.processInfo.processIdentifier),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "tag": ProcessInfo.processInfo.environment["ENGINE_IDX_REPLAY_TAG"] ?? "",
                "timing_eligible": "false", "complete": "true",
                "scope": "one representative S1 invocation; no all-layer exactness claim"
            ]
            metadata.merge(provenance) { _, new in new }
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let stem = URL(fileURLWithPath: dir).appendingPathComponent("idx-s1-budget\(budget)")
            try MLX.save(arrays: ["q": q, "pooled_storage": storage, "reference_top": top.asType(.int32),
                                  "nblocks": MLXArray(nBlocks.map { Int32($0) }),
                                  "row_offsets": MLXArray(rows.map { Int32($0) })],
                         metadata: metadata, url: stem.appendingPathExtension("safetensors"))
            let json = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
            try json.write(to: stem.appendingPathExtension("json"), options: .atomic)
            FileHandle.standardError.write(Data("idx replay dump: \(stem.path).safetensors route=\(route) budget=\(budget) (UNTIMED)\n".utf8))
        } catch {
            FileHandle.standardError.write(Data("idx replay dump failed: \(error)\n".utf8))
        }
    }

    func callAsFunction(qk: MLXArray, rope: Qwen4ExpRotary, keyCache: ArraysCache?, offset: Int,
                        pos3: MLXArray? = nil, mropeSection: [Int] = [11, 11, 10]) -> MLXArray? {
        lastTop = nil
        let B = qk.dim(0), S = qk.dim(1)
        let split = nHeads * headDim
        let q = qk[.ellipsis, 0 ..< split].reshaped(B, S, nHeads, headDim)
        var rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)
        // P018 (OBS-ENG-032): slot 0 is a CAPACITY buffer written in place (like KVCacheSimple), not a per-step concatenation of the
        // whole key history -- at 256k the concat copied 67 MB per layer per step (0.237 vs 0.017 ms amortised, x13 layers). The
        // logical length is offset + S (the KV cache's offset before this block); rows beyond it are stale and get overwritten,
        // so a rollback needs no slice of this slot (the pooled slot 1 is still trimmed: its blocks beyond the new length are stale).
        let kvLen = keyCache == nil ? rawK.dim(1) : offset + S
        if let keyCache {
            let cap = keyCache[0]?.dim(1) ?? 0
            if keyCache[0] == nil || kvLen > cap {
                let want = kvLen + max(Self.idxStep, cap / 8)
                let newCap = Qwen4ExpCacheCapacity.bounded(((want + Self.idxStep - 1) / Self.idxStep) * Self.idxStep, need: kvLen)
                let nb = MLXArray.zeros([B, newCap, headDim], dtype: rawK.dtype)
                if let old = keyCache[0], offset > 0 { nb[0..., 0 ..< offset, 0...] = old[0..., 0 ..< offset, 0...] }
                keyCache[0] = nb
            }
            keyCache[0]?[0..., offset ..< kvLen, 0...] = rawK
            rawK = keyCache[0]![0..., 0 ..< kvLen, 0...]
        }
        // P039 unit 3 -- slot 2 is the FULL 3-axis position history, and it is the reference design,
        // not a workaround. HF's own `Qwen4ExpTextModel.forward` binds the full position ids to the
        // cache for exactly this reason: "We need the full position_ids in the indexer, not just the
        // current ones, so bind them to the cache if any (as they are 3D, it's otherwise not easy to
        // compute them back from only current positions)". Written in place at [offset, kvLen) like
        // slot 0, and ONLY when the sequence carries visual tokens -- a text sequence never allocates
        // it and takes byte-identical code below.
        if let keyCache, let pos3 {
            let cap2 = keyCache[2]?.dim(2) ?? 0
            if keyCache[2] == nil || kvLen > cap2 {
                let want = kvLen + max(Self.idxStep, cap2 / 8)
                let newCap = Qwen4ExpCacheCapacity.bounded(((want + Self.idxStep - 1) / Self.idxStep) * Self.idxStep, need: kvLen)
                let nb = MLXArray.zeros([3, B, newCap], dtype: .int32)
                if let old = keyCache[2], offset > 0 { nb[0..., 0..., 0 ..< offset] = old[0..., 0..., 0 ..< offset] }
                keyCache[2] = nb
            }
            keyCache[2]?[0..., 0..., offset ..< kvLen] = pos3.asType(.int32)
        }
        if kvLen <= budget { return nil }
        // P037's REFUSAL STOOD HERE and its premise was wrong (REFUT-ENG-024 retracted by P039). It
        // read: "a block start is not a scalar position once the T/H/W axes disagree inside the
        // block". True, and irrelevant -- the reference never ropes a block start FROM a scalar. HF
        // builds cos/sin for the FULL positions once and INDEXES THE ROWS at the block starts
        // (`full_cos[batch_idx].index_select(0, group_starts)` in Qwen4ExpTextQSAIndexer.forward), so
        // a block start needs a ROW, which mRoPE has for every token, not a scalar. Below is that.
        // ENGINE_IDX_DECODE_MODE (timing bisection only, outputs wrong): 1 = dense attention for blocks <= 16 rows,
        // 2 = keep every block (no top-k), 3 = scores + top-k computed but the mask not returned
        let bisect = Self.decodeMode
        if bisect == 1 && S <= 16 { return nil }

        let nBlocks = kvLen / ratio
        // pooled block keys: mean over `ratio` raw keys, k_layernorm, rope at the block start. A completed block never changes,
        // so only the blocks completed since the last call are computed (slot 1 holds the pooled history).
        var pooled: MLXArray
        let cached: MLXArray? = Self.pooledCache ? keyCache?[1] : nil
        // P020 unit 8: slot 1 is a CAPACITY buffer written in place, like slot 0 (OBS-ENG-032). It used to be an
        // exact-size array rebuilt with `concatenated([history, new])` on every call, which copies the WHOLE pooled
        // history -- 8.6 MB per attention layer per chunk at 135k of kv, for 0.26 MB of new blocks. The logical
        // length is derived, not stored: the blocks completed as of the previous call are exactly `offset / ratio`
        // (the KV offset before this block), so a rollback that trims the KV also trims this. Rows beyond the
        // logical length are stale and get overwritten. ENGINE_IDX_POOL_BUF=0 restores the concatenation.
        let capBuf = Self.pooledBuffer && keyCache != nil
        // P031 DEFECT FIX -- THE ENGINE CRASHED ON ORDINARY INPUT AND NO GATE COULD SEE IT.
        // `offset / ratio` is the number of blocks completed as of the previous call, and the
        // capacity-buffer path took it as the number ALREADY POOLED. That holds only if this indexer
        // ran on every previous call -- and it does NOT: `kvLen <= budget` returns nil ABOVE, so a
        // prefill at or below the budget never pools anything, while slot 0 (the raw keys) is always
        // written because that happens before the early return. The first decode step that crosses
        // the budget then pooled from `offset / ratio` into a buffer that was never allocated:
        // **SIGTRAP, no message, exit 133**, for ANY session that starts at or below
        // `indexer_budget` tokens and generates past it. Measured on the champion: prompts of 2044
        // and 2048 die at every depth AND serially; 2052, whose prefill runs the indexer and builds
        // the pool, is fine. Every gate this project owns sits at 512, 8192 or 262144 and none of
        // them crosses the window. Capping by the buffer's actual capacity is exact: an unallocated
        // pool reads 0 and everything is pooled from the raw keys, which are all present, and the
        // rollback property the derivation was written for survives because `offset / ratio` still
        // caps it from the other side.
        let done = min(capBuf ? min(offset / ratio, cached?.dim(1) ?? 0) : (cached?.dim(1) ?? 0), nBlocks)
        if done < nBlocks {
            let newRaw = rawK[0..., (done * ratio) ..< (nBlocks * ratio), 0...].reshaped(B, nBlocks - done, ratio, headDim)
            var p = kNorm(newRaw.asType(.float32).mean(axis: 2).asType(rawK.dtype))
            let starts = Self.int32Range(done, nBlocks, scale: ratio)
            // The block-start rows. Text: the cached contiguous table, exactly as before. Visual: the
            // same rows of the mRoPE table, gathered from slot 2's full position history -- the
            // transcription of `full_cos.index_select(0, group_starts)`.
            let (cosK, sinK): (MLXArray, MLXArray)
            if Self.ropeWitness, !Self.ropeWitnessDone {
                Self.ropeWitnessDone = true
                let has = keyCache?[2] != nil
                FileHandle.standardError.write((
                    "idx rope: pos3=\(pos3 != nil) slot2=\(has) scalarArm=\(Self.scalarBlockRope) "
                    + "-> block starts roped by "
                    + ((pos3 != nil && !Self.scalarBlockRope && has) ? "mRoPE (P039)" : "SCALAR")
                    + ", kvLen=\(kvLen) blocks=\(nBlocks) keep=\(blockTopK)\n").data(using: .utf8)!)
            }
            if pos3 != nil, !Self.scalarBlockRope, let posBuf = keyCache?[2] {
                let bs = take(posBuf[0..., 0..., 0 ..< kvLen], starts, axis: 2)   // (3, B, n)
                (cosK, sinK) = rope.tablesMRope(bs, section: mropeSection)
            } else {
                (cosK, sinK) = rope.tables(expandedDimensions(starts, axis: 0))  // (1,n,d)
            }
            p = Qwen4ExpRotary.apply(p, cos: cosK, sin: sinK)
            if capBuf {
                let cap = cached?.dim(1) ?? 0
                if cached == nil || nBlocks > cap {
                    let want = Qwen4ExpCacheCapacity.bounded(nBlocks + max(Self.idxStep / ratio, cap / 8), need: nBlocks, ratio: ratio)
                    let nb = MLXArray.zeros([B, want, headDim], dtype: p.dtype)
                    if let old = cached, done > 0 { nb[0..., 0 ..< done, 0...] = old[0..., 0 ..< done, 0...] }
                    keyCache?[1] = nb
                }
                keyCache?[1]?[0..., done ..< nBlocks, 0...] = p
                pooled = keyCache![1]![0..., 0 ..< nBlocks, 0...]
            } else {
                pooled = (done == 0 || cached == nil) ? p : concatenated([cached![0..., 0 ..< done, 0...], p], axis: 1)
                if Self.pooledCache { keyCache?[1] = pooled }
            }
        } else {
            pooled = cached![0..., 0 ..< nBlocks, 0...]
        }

        // HF slices the CURRENT positions off the full table for the queries
        // ("full_cos[:, -seq_length:, :]"); with pos3 those are the current call's own rows.
        let (cosQ, sinQ) = (pos3 != nil && !Self.scalarBlockRope)
            ? rope.tablesMRope(pos3!, section: mropeSection)
            : rope.tables(offset: offset, count: S)  // (1,S,d)
        var qn = qNorm(q)
        qn = Qwen4ExpRotary.apply(
            qn, cos: expandedDimensions(cosQ, axis: 2), sin: expandedDimensions(sinQ, axis: 2))

        // scores[b,s,n] = sum_h relu(q[b,s,h,:] . pooled[b,n,:]) / sqrt(d)
        Q4Prof.mark("idx.pool", [pooled, qn])
        let q32 = qn.asType(.float32).reshaped(B, S * nHeads, headDim)
        // the visibility predicate is folded into the fused kernel for S > 1 (it is pure arithmetic in (s, n)); the graph
        // form keeps the separate `which` below
        let fuseVis = q4IdxFused && S > 1
        var scores: MLXArray
        // P020 unit 2: the whole score in ONE kernel -- the (B,S,H,N) float32 intermediate (2.21 GB at a 4096-row
        // chunk over 135k of kv) never exists, because the head reduction happens in registers between the MMA and
        // the store. Prefill widths only: the tile is 8*MI rows and the decode row count is 1.
        let scoreMI = q4IdxScoreMI, scoreNI = q4IdxScoreNI, scoreSG = q4IdxScoreSG
        // P023 unit 5: the SAME kernel on the NARROW rows -- decode (S=1) and the MTP verify block (S=K+1). The graph
        // form below casts `pooled` to float32 on every step: at 262144 that is a 33.5 MB write and a 33.5 MB read per
        // attention layer per token on top of the 16.8 MB it already reads, then a (B,S,H,N) intermediate for the
        // relu-and-head-sum. The fused kernel materialises none of it. Its tile is 8*MI rows, so a narrow S cannot fill
        // it; the S rows are placed at the END of the tile and the front is padded with copies of row 0. THAT ORDER IS
        // THE POINT: the kernel's visibility limit is `kvLen - S + s0 + r`, so with S = TM the padded row r carries
        // exactly the position the real row r - (TM - S_true) has, and the predicate stays correct without touching a
        // kernel OBS-ENG-042 measured bit-identical (max|d| 0.000e+00). Front-padding is not cosmetic -- back-padding
        // would price every real row against the wrong position.
        if q4IdxScoreDecodeOn, q4IdxScoreFusedOn, S < scoreSG * 8 * scoreMI, B == 1 || q4IdxScoreBatchOn,
           qn.dtype == pooled.dtype, nBlocks >= 8 * scoreNI {
            var mi = 1
            while 8 * mi < S { mi *= 2 }                                            // smallest 8*MI tile that holds S
            let TM = 8 * mi
            let qpad = S == TM ? qn : concatenated([tiled(qn[0..., 0 ..< 1, 0..., 0...], repetitions: [1, TM - S, 1, 1]), qn], axis: 1)
            let sT = q4IdxScore(q: qpad, pooled: pooled, scale: sqrt(Float(headDim)),
                                visibility: (kvLen: kvLen, ratio: ratio), mi: mi, ni: scoreNI, sg: 1)
            scores = sT[0..., (TM - S) ..< TM, 0...]
            Qwen4ExpIndexerRouteWitness.record(.denseFused, batch: B, sequence: S,
                                               budget: budget, scoreBatch: B, nBlocks: nBlocks)
            // ENGINE_IDX_SCORE_DEC_WITNESS=1: say ONCE that this branch really executed. BOUND-ENG-007 and the retraction
            // in OBS-ENG-067 (3): a gated path that silently does not bind is measured against itself. Off by default so
            // the measured binary carries only a static-let load.
            if Self.decWitness, !Self.decWitnessed {
                Self.decWitnessed = true
                FileHandle.standardError.write("qwen4_exp: idx score DECODE branch active -- S=\(S) TM=\(TM) MI=\(mi) NI=\(scoreNI) nBlocks=\(nBlocks) kvLen=\(kvLen)\n".data(using: .utf8)!)
            }
        } else if q4IdxScoreFusedOn, fuseVis, S >= scoreSG * 8 * scoreMI, nBlocks >= 8 * scoreNI, qn.dtype == pooled.dtype {
            scores = q4IdxScore(q: qn, pooled: pooled, scale: sqrt(Float(headDim)),
                                visibility: (kvLen: kvLen, ratio: ratio), mi: scoreMI, ni: scoreNI, sg: scoreSG)
            Qwen4ExpIndexerRouteWitness.record(.denseFused, batch: B, sequence: S,
                                               budget: budget, scoreBatch: B, nBlocks: nBlocks)
            Q4Prof.mark("idx.score", [scores])
        } else if q4IdxFused, q4IdxNChunk > 0, nBlocks > q4IdxNChunk {
            // P019 unit 7: reduce the block axis in slices so the (B,S,H,NC) float32 intermediate never has to reach DRAM.
            // Bit-identical to the single call -- every output element is the same sum of the same four products.
            let out = MLXArray.zeros([B, S, nBlocks], dtype: .float32)
            var n0 = 0
            while n0 < nBlocks {
                let n1 = min(nBlocks, n0 + q4IdxNChunk)
                let pc = pooled[0..., n0 ..< n1, 0...].asType(.float32).transposed(0, 2, 1)
                let rawc = matmul(q32, pc).reshaped(B, S, nHeads, n1 - n0)
                out[0..., 0..., n0 ..< n1] = q4IdxReluSum(
                    rawc, scale: sqrt(Float(headDim)),
                    visibility: fuseVis ? (kvLen: kvLen, ratio: ratio) : nil, blockOffset: n0)
                n0 = n1
            }
            Q4Prof.mark("idx.matmul", [out])
            scores = out
            Qwen4ExpIndexerRouteWitness.record(.denseGraph, batch: B, sequence: S,
                                               budget: budget, scoreBatch: B, nBlocks: nBlocks)
            Q4Prof.mark("idx.reduce", [scores])
        } else {
            let raw: MLXArray
            if q4IdxBF16 {
                raw = matmul(qn.reshaped(B, S * nHeads, headDim), pooled.transposed(0, 2, 1)).reshaped(B, S, nHeads, nBlocks)
            } else {
                let p32 = pooled.asType(.float32).transposed(0, 2, 1)  // (B,d,n)
                raw = matmul(q32, p32).reshaped(B, S, nHeads, nBlocks)
            }
            Q4Prof.mark("idx.matmul", [raw])
            scores = q4IdxFused
                ? q4IdxReluSum(raw, scale: sqrt(Float(headDim)), visibility: fuseVis ? (kvLen: kvLen, ratio: ratio) : nil)
                : maximum(raw.asType(.float32), MLXArray(Float(0))).sum(axis: 2) / sqrt(Float(headDim))   // scores are float32 by design
            Qwen4ExpIndexerRouteWitness.record(.denseGraph, batch: B, sequence: S,
                                               budget: budget, scoreBatch: B, nBlocks: nBlocks)
            Q4Prof.mark("idx.reduce", [scores])
        }

        // P032: reuse. Placed AFTER the pooled cache is up to date and BEFORE the score, so the only
        // work skipped is the score and the selection. The cached ids index COMPLETED blocks, and a
        // completed block never changes, so they stay valid as nBlocks grows; nBlocks and kvLen are
        // passed fresh, which keeps the gather's tail handling exact.
        if reusePeriod > 1, S == 1, Q4Fused.qsaGather, kvLen >= Self.gatherMinKV, let t = reuseTop, reuseSince < reusePeriod - 1 {
            reuseSince += 1
            lastTop = (t, nBlocks, kvLen)
            return nil
        }
        if Self.ablIdxScore { scores = tiled(Self.int32Range(0, nBlocks).asType(.float32).reshaped(1, 1, nBlocks), repetitions: [B, S, 1]) }
        // P072 arms: ENGINE_QSA_KEEP_RECENT=R forces the R most recent completed blocks of every row into the
        // selection, ENGINE_QSA_KEEP_FIRST=F forces the first F blocks; both as a score bonus, so an invisible
        // block (-inf) stays out and nothing is selected twice. Unset = champion, byte for byte.
        if Qwen4ExpVariants.recentPenalty > 0 {
            let blockIdx = Self.int32Range(0, nBlocks).reshaped(1, 1, nBlocks)
            let nbRow = (Self.int32Range(kvLen - S, kvLen).reshaped(1, S, 1) + Int32(1)) / Int32(ratio)
            let tooNew = blockIdx .>= (nbRow - Int32(Qwen4ExpVariants.recentPenalty))
            scores = scores - tooNew.asType(.float32) * MLXArray(Float(1e6))
        }
        if Qwen4ExpVariants.keepRecent > 0 || Qwen4ExpVariants.keepFirst > 0 {
            let blockIdx = Self.int32Range(0, nBlocks).reshaped(1, 1, nBlocks)                     // (1,1,n)
            let rowPos = Self.int32Range(kvLen - S, kvLen).reshaped(1, S, 1)                        // (1,S,1)
            var force = MLXArray.zeros([1, S, nBlocks], dtype: .bool)
            if Qwen4ExpVariants.keepRecent > 0 {
                let nbRow = (rowPos + Int32(1)) / Int32(ratio)
                force = force .|| (blockIdx .>= (nbRow - Int32(Qwen4ExpVariants.keepRecent)))
            }
            if Qwen4ExpVariants.keepFirst > 0 { force = force .|| (blockIdx .< Int32(Qwen4ExpVariants.keepFirst)) }
            scores = scores + force.asType(.float32) * MLXArray(Float(1e6))
        }
        let k = min(blockTopK, nBlocks)
        if bisect == 2 && S <= 16 { return MLXArray.ones([B, 1, S, kvLen], dtype: .bool) }
        var top: MLXArray
        if S > 1 {
            // a block is visible to row s only if it ends at or before that row's position (multi-row blocks: prefill, verify)
            let kvPos = Self.int32Range(kvLen - S, kvLen)  // (S)
            if !fuseVis {
                let blockEnd = Self.int32Range(0, nBlocks, scale: ratio) + Int32(ratio - 1)  // (n)
                let visible = expandedDimensions(blockEnd, axes: [0, 1]) .<= expandedDimensions(kvPos, axes: [0, 2])  // (1,S,n)
                scores = which(visible, scores, MLXArray(-Float.infinity))
            }
            Q4Prof.mark("idx.mask", [scores])
            top = Self.ablTopK ? tiled(Self.int32Range(0, k).reshaped(1, 1, k), repetitions: [B, S, 1])
                : Self.ablTopKSpread ? tiled(Self.int32Range(0, k, scale: max(1, nBlocks / k)).reshaped(1, 1, k), repetitions: [B, S, 1])
                : (Self.topKSelect ? q4TopK(scores: scores, k: k) : argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k])  // (B,S,k)
            Q4Prof.mark("idx.topk", [top])
            // a row may still select an invisible block when fewer than k are visible; the predicate is arithmetic in the
            // SELECTED ids, so this needs no (1,S,nBlocks) tensor and no gather from one
            let endSel = top * Int32(ratio) + Int32(ratio - 1)                       // (B,S,k)
            top = which(endSel .<= expandedDimensions(kvPos, axes: [0, 2]), top, MLXArray(Int32(nBlocks)))
            Q4Prof.mark("idx.fix", [top])
        } else {
            // decode: every completed block ends before the current position -- all visible
            top = Self.ablTopK ? tiled(Self.int32Range(0, k).reshaped(1, 1, k), repetitions: [B, S, 1])
                : Self.ablTopKSpread ? tiled(Self.int32Range(0, k, scale: max(1, nBlocks / k)).reshaped(1, 1, k), repetitions: [B, S, 1])
                : Self.topKKeep >= 0 ? Self.keepTop(scores, k: k, keep: Self.topKKeep, nBlocks: nBlocks)
                : Self.boundarySwap >= 0 && k + Self.boundarySwap <= nBlocks ? {
                    let t = Self.boundarySwapTop(scores, k: k, swap: Self.boundarySwap)
                    if Self.rankShiftCheck { Self.rankShiftCompare(t, scores, k: k) }
                    return t }()
                : Self.rankShift >= 0 && Self.rankShift + k <= nBlocks ? {
                    let t = Self.shiftedTop(scores, k: k, shift: Self.rankShift)
                    if Self.rankShiftCheck { Self.rankShiftCompare(t, scores, k: k) }
                    return t }()
                : (Self.topKSelect ? q4TopK(scores: scores, k: k) : argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k])
        }
        if bisect == 3 && S <= 16 { eval(top); return nil }
        if Q4Fused.qsaGather && S <= Self.gatherMaxS && (S > 1 || kvLen >= Self.gatherMinKV) {
            // gather path (P017): hand the block ids to the attention kernel; no token mask over the whole kv. S > 1 always (token
            // fidelity, OBS-ENG-030); S = 1 above ENGINE_QSA_GATHER_MIN_KV (the masked vector kernel is as fast below ~12k: lab
            // 0.256 vs 0.259 ms at 8k, in situ 8k serial 57.03 vs 56.66)
            lastTop = (top.asType(.int32), nBlocks, kvLen)
            if reusePeriod > 1, S == 1 { reuseTop = lastTop!.0; reuseSince = 0 }
            if Self.stabWitness, S == 1 {
                let t = lastTop!.0
                eval(t)
                let ids = t.asArray(Int32.self)
                if let prev = stabPrev {
                    let ps = Set(prev)
                    var ov = 0, freshNew = 0
                    let prevBlocks = Int32(nBlocks - 1)
                    for i in ids { if ps.contains(i) { ov += 1 } else if i >= prevBlocks { freshNew += 1 } }
                    stabOverlap.append(ov); stabFresh += freshNew
                }
                stabPrev = ids
                stabCalls += 1
                if stabCalls % 32 == 0, stabOverlap.count > 0 {
                    let k = ids.count
                    let so = stabOverlap.sorted()
                    let med = so[so.count / 2], lo = so[0], hi = so[so.count - 1]
                    FileHandle.standardError.write(String(format:
                        "qsa-stability: layer_kv=%d k=%d n=%d  |top_t ∩ top_{t-1}| median %d (%.3f)  min %d (%.3f)  max %d  newcomers-that-are-brand-new-blocks %d of %d churned\n",
                        kvLen, k, so.count, med, Double(med)/Double(k), lo, Double(lo)/Double(k), hi,
                        stabFresh, so.reduce(0) { $0 + (k - $1) }).data(using: .utf8)!)
                    stabOverlap.removeAll(); stabFresh = 0
                }
            }
            return nil
        }
        var keep = MLXArray.zeros([B, S, nBlocks + 1], dtype: .bool)
        keep = putAlong(keep, top, values: MLXArray(true), axis: -1)[.ellipsis, 0 ..< nBlocks]
        // block -> tokens (np.repeat along the last axis)
        var tok = broadcast(expandedDimensions(keep, axis: -1), to: [B, S, nBlocks, ratio]).reshaped(B, S, nBlocks * ratio)
        let tail = kvLen - nBlocks * ratio
        if tail > 0 {
            tok = concatenated([tok, MLXArray.ones([B, S, tail], dtype: .bool)], axis: -1)
        }
        if S > 1 {
            // P017 (OBS-ENG-030): a verify row at position p must see what the S = 1 step at p sees -- its OWN incomplete block
            // [((p+1)/ratio)*ratio, p], not only the block's tail at kvLen. Blocks completed inside the verify block were invisible
            // to the earlier rows (K=3 lost the current token itself when kvLen % ratio == 0): tokens diverged from serial.
            let kvPos = Self.int32Range(kvLen - S, kvLen)                                       // (S)
            let rowTail = ((kvPos + Int32(1)) / Int32(ratio)) * Int32(ratio)                   // (S) first position of the row's own block
            let pos = Self.int32Range(0, kvLen)                                                // (kv)
            tok = tok .|| (expandedDimensions(pos, axes: [0, 1]) .>= expandedDimensions(rowTail, axes: [0, 2]))   // causal AND is applied by sparseMask
        }
        return expandedDimensions(tok, axis: 1)  // (B,1,S,kv)
    }
}

// MARK: - Attention (gated, partial RoPE, GQA)

final class Qwen4ExpAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    /// q_proj | k_proj | v_proj | indexer.index_qk_proj concatenated on the output axis (sanitize)
    @ModuleInfo(key: "qkvi_proj") var qkviProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer

    init(_ a: Qwen4ExpTextConfiguration) {
        nHeads = a.attentionHeads
        nKVHeads = a.kvHeads
        headDim = a.headDim
        scale = pow(Float(a.headDim), -0.5) * Qwen4ExpVariants.attnScaleMult   // P072 arm: ENGINE_ATTN_SCALE_MULT (1 = champion)
        let d = a.hiddenSize
        let idxDim = (a.indexerNHeads + a.indexerKVHeads) * a.indexerHeadDim
        _qkviProj.wrappedValue = Linear(d, nHeads * headDim * 2 + 2 * nKVHeads * headDim + idxDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, d, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(a)
        super.init()
    }

    /// P029 unit 5, the attention-layer split. Unit 1 put 446 us in a decode attention layer of which
    /// the indexer score (53) and the selection (104) are already attributed and the QSA gather is
    /// ZERO -- leaving ~289 us for the projections, the front end and the cache updates, against a
    /// ~94 us size-dependent read roof for the two qmvs. These three keys attribute it. Each REPLAYS
    /// a REAL captured tensor rather than a constant: the `topkspread` lesson (OBS-ENG-085) is that
    /// the VALUES set the downstream access pattern, so a zeroed qkvi would move the selection and
    /// charge the change to the projection. TIMING ONLY, and the captured tensor is one layer's,
    /// replayed for every step, so the generated text is wrong by construction.
    nonisolated(unsafe) var ablQKVIHold: MLXArray? = nil
    nonisolated(unsafe) var ablOutHold: MLXArray? = nil
    static let ablQKVI: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("attnqkvi")
    static let ablOProj: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("attnoproj")
    static let ablKVUpdate: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("attnkv")
    /// P095 U3-H: ENGINE_RAGGED_SDPA=1 -- below `gatherMinKV` the ragged step attends the way the
    /// serial program does (SDPA over the sparse mask) instead of the ragged gather kernel.
    static let raggedSDPA: Bool = (ProcessInfo.processInfo.environment["ENGINE_RAGGED_SDPA"] ?? "0") != "0"
    /// P054: like `attnkv` but the write still runs -- only the gather's read is stale. Isolates
    /// OBS-ENG-122 (2)'s growing cost between the write's own dispatch and a read-side confound.
    /// S=1 decode call site only. Timing only; output is wrong by construction, same as `attnkv`.
    static let ablKVStale: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("attnkvstale")
    /// P050: route the QSA gather's cache update through `updateNoReturn` (skips the
    /// discarded (returnedKeys, returnedValues) view `update` always builds) instead of
    /// `_ = kvc.update(...)`. Default off, unchanged behaviour.
    static let kvUpdateLean: Bool = (ProcessInfo.processInfo.environment["ENGINE_KV_UPDATE_LEAN"] ?? "0") != "0"

    /// sparse keep mask (B,1,S,kv) AND causal; for S == 1 the causal part is all-true (every cached position precedes the query)
    static let attnChunkMaxRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_ATTN_CHUNK_MAX"] ?? "16") ?? 16
    static func sparseMask(_ sparse: MLXArray, S: Int, offset: Int) -> MLXArray {
        if S == 1 { return sparse }
        let kvLen = offset + S
        let rinds = Qwen4ExpQSAIndexer.int32Range(0, kvLen)
        let linds = expandedDimensions(Qwen4ExpQSAIndexer.int32Range(kvLen - S, kvLen), axis: 1)
        return sparse .&& (linds .>= rinds)
    }

    /// `pos3` (P037 unit 3): (3, B, S) mRoPE positions, non-nil ONLY when the sequence carries visual
    /// tokens. nil takes the shipped cached-table path unchanged, which is the text guarantee.
    func callAsFunction(_ x: MLXArray, rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?, indexerCache: ArraysCache?, pos3: MLXArray? = nil,
                        mropeSection: [Int] = [11, 11, 10]) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let offset = cache?.offset ?? 0
        let qDim = nHeads * headDim * 2, kvDim = nKVHeads * headDim
        // P080 M1 -- RAGGED DECODE. One token for each of B sequences that are at DIFFERENT lengths:
        // every quantity the single-sequence path reads off `offset` is read per row instead. Taken
        // only when the caller installs the lengths, so the champion program below is untouched.
        if let rows = Qwen4ExpBatch.rowOffsets, let kvc = cache as? KVCacheSimple, let idxc = indexerCache {
            precondition(rows.count == B, "ragged decode needs one length per row, got \(rows.count) for B=\(B)")
            // P099: S > 1 = a verify block per row (teacher-forced or drafted); the S == 1 program is unchanged.
            var qkviR = q8Proj(qkviProj, x)
            // P095: the S = 1 ablation keys, ported so the batched step's attention can be split the
            // way OBS-ENG-089 split the serial one. TIMING ONLY, output wrong by construction, and each
            // key means exactly what it means on the single-sequence path below.
            if Self.ablQKVI {
                if let h = ablQKVIHold, h.shape == qkviR.shape { qkviR = h }
                else { eval(qkviR); ablQKVIHold = qkviR }
            }
            Q4Prof.mark("attn.qkvi", [qkviR])
            let pr = split(qkviR, indices: [qDim, qDim + kvDim, qDim + 2 * kvDim], axis: -1)
            // P106 H48: a row-resident pool hands this layer a marker; the rows' own caches are behind it
            let rowLists = Qwen4ExpBatch.rowResident(kvc)
            if let rowLists { precondition(rowLists.count == B, "row-resident pool: one cache per row, got \(rowLists.count) for B=\(B)") }
            guard let nBlocksRow = indexer.raggedDecode(qk: pr[3], rope: rope, keyCache: idxc, rows: rows,
                                                        rowCaches: rowLists?.map { $0[1] as! ArraysCache }),
                  let topR = indexer.lastTopRagged?.0 else {
                preconditionFailure("P080 M1 batches only rows past the indexer budget; got lengths \(rows)")
            }
            Q4Prof.mark("attn.indexer", [topR])
            let qg = pr[0].reshaped(B, S, nHeads, 2 * headDim)
            var qR = qNorm(qg[.ellipsis, 0 ..< headDim]).transposed(0, 2, 1, 3)
            let gateR = qg[.ellipsis, headDim...].reshaped(B, S, nHeads * headDim)
            var kR = kNorm(pr[1].reshaped(B, S, nKVHeads, headDim)).transposed(0, 2, 1, 3)
            let vR = pr[2].reshaped(B, S, nKVHeads, headDim).transposed(0, 2, 1, 3)
            let rPos: [Int32] = rows.flatMap { r in (0 ..< S).map { Int32(r + $0) } }
            let (cR, sR) = rope.tables(rowPositions: rPos, B: B, S: S)                       // (B,S,d)
            let ce = expandedDimensions(cR, axis: 1), se = expandedDimensions(sR, axis: 1)  // (B,1,S,d)
            qR = Qwen4ExpRotary.apply(qR, cos: ce, sin: se)
            kR = Qwen4ExpRotary.apply(kR, cos: ce, sin: se)
            if !(Self.ablKVUpdate && S == 1) {
                if let rowLists {
                    for b in 0 ..< B {
                        (rowLists[b][0] as! KVCacheSimple).updateRow(keys: kR[b ..< (b + 1)], values: vR[b ..< (b + 1)], at: rows[b])
                    }
                } else { kvc.updateRagged(keys: kR, values: vR, at: rows) }
            }
            let kvLenMax = rows.max()! + S
            let outR: MLXArray
            if Qwen4ExpQSAIndexer.ablQSAGather, S == 1 { outR = qR }
            else if let rowLists {
                // the ragged kernel's arithmetic is per (row, head, query row): B one-row calls on the
                // rows' own buffers compute what the B-row call computes on the stacked pool
                outR = q4QSAGatherRaggedRows(q: qR, keys: rowLists.map { ($0[0] as! KVCacheSimple).rawKeys! },
                                             values: rowLists.map { ($0[0] as! KVCacheSimple).rawValues! }, top: topR,
                                             rowOffsets: rows, rowNBlocks: nBlocksRow, ratio: indexer.ratio, scale: scale)
            }
            else if Self.raggedSDPA, S == 1, kvLenMax < Qwen4ExpQSAIndexer.gatherMinKV {
                // P095 U3-H: the serial program's attention at this context -- SDPA over the sparse
                // mask (selected blocks + the row's own tail), one mask row per sequence over kvLenMax.
                let ratio = indexer.ratio, nBlocksMax = nBlocksRow.max()!
                let nbRow = MLXArray(nBlocksRow.map { Int32($0) }).reshaped(B, 1, 1)
                var keep = MLXArray.zeros([B, 1, nBlocksMax + 1], dtype: .bool)
                keep = putAlong(keep, topR, values: MLXArray(true), axis: -1)[.ellipsis, 0 ..< nBlocksMax]
                keep = keep .&& (Qwen4ExpQSAIndexer.int32Range(0, nBlocksMax).reshaped(1, 1, nBlocksMax) .< nbRow)
                var tok = broadcast(expandedDimensions(keep, axis: -1), to: [B, 1, nBlocksMax, ratio]).reshaped(B, 1, nBlocksMax * ratio)
                let tailLen = kvLenMax - nBlocksMax * ratio
                if tailLen > 0 { tok = concatenated([tok, MLXArray.zeros([B, 1, tailLen], dtype: .bool)], axis: -1) }
                let pos = Qwen4ExpQSAIndexer.int32Range(0, kvLenMax).reshaped(1, 1, kvLenMax)
                let limit = MLXArray(rows.map { Int32($0) }).reshaped(B, 1, 1)
                let tailStart = nbRow * Int32(ratio)
                tok = (tok .|| (pos .>= tailStart)) .&& (pos .<= limit)
                let maskR = expandedDimensions(tok, axis: 1)                                   // (B,1,1,kvLenMax)
                outR = MLXFast.scaledDotProductAttention(queries: qR, keys: kvc.rawKeys![0..., 0..., 0 ..< kvLenMax, 0...],
                                                         values: kvc.rawValues![0..., 0..., 0 ..< kvLenMax, 0...],
                                                         scale: scale, mask: .array(maskR))
            } else {
                outR = q4QSAGatherRagged(q: qR, keys: kvc.rawKeys!, values: kvc.rawValues!, top: topR,
                                         rowOffsets: rows, rowNBlocks: nBlocksRow, ratio: indexer.ratio, scale: scale)
            }
            Q4Prof.mark("attn.core", [outR])
            let gatedR = (S == 1 ? outR.reshaped(B, 1, nHeads * headDim) : outR.transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim)) * sigmoid(gateR)
            if Self.ablOProj { return gatedR[.ellipsis, 0 ..< oProj.weight.dim(0)] }   // a slice keeps the front end alive
            let projR = q8Proj(oProj, gatedR)
            Q4Prof.mark("attn.oproj", [projR])
            // P081 witness, second stage: row 0's q/k after rope, the gather output and the layer's
            // attention result. With the indexer already proven identical between B=1 and B=2, this
            // says whether the batch changes the ATTENTION or something after it.
            if let dir = Qwen4ExpQSAIndexer.witnessDir, Qwen4ExpQSAIndexer.witnessAttn < 256 {
                let i = Qwen4ExpQSAIndexer.witnessAttn; Qwen4ExpQSAIndexer.witnessAttn += 1
                try? MLX.save(arrays: ["q": qR[0 ..< 1], "k": kR[0 ..< 1], "gather": outR[0 ..< 1],
                                       "proj": projR[0 ..< 1]],
                              url: URL(fileURLWithPath: dir + "/attn\(i).safetensors"))
            }
            return projR
        }
        var qkvi = q8Proj(qkviProj, x)                             // one qmv: q|gate, k, v, indexer qk
        if Self.ablQKVI, S == 1 {
            if let h = ablQKVIHold, h.shape == qkvi.shape { qkvi = h }
            else { eval(qkvi); ablQKVIHold = qkvi }
        }
        Q4Prof.mark("attn.qkvi", [qkvi])
        let parts = split(qkvi, indices: [qDim, qDim + kvDim, qDim + 2 * kvDim], axis: -1)
        let sparse = indexer(qk: parts[3], rope: rope, keyCache: indexerCache, offset: offset, pos3: pos3,
                             mropeSection: mropeSection)
        if Q4Prof.active { var a: [MLXArray] = []; if let t = indexer.lastTop?.0 { a.append(t) }; if let m = sparse { a.append(m) }; Q4Prof.mark("attn.indexer", a) }
        if Q4Fused.attnFront && S == 1 {
            // decode: q/k norm + partial rope + v + sigmoid(gate) in one kernel (no transposes, no slices); sdpa output (B,H,1,D) is
            // already in (B,1,H*D) order, so the gated product feeds o_proj without a copy
            let (ct, st) = pos3 == nil ? rope.tables(offset: offset, count: 1)
                                       : rope.tablesMRope(pos3!, section: mropeSection)
            let (q, k, v, sg) = q4AttnFront(qkvi: qkvi.reshaped(B, qkvi.dim(-1)), wq: qNorm.onePlusWeight(x.dtype), wk: kNorm.onePlusWeight(x.dtype),
                                            cos: ct.reshaped(rope.dim), sin: st.reshaped(rope.dim), nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, rotDim: rope.dim, eps: qNorm.eps)
            var fm = mask
            if let sparse { fm = .array(Self.sparseMask(sparse, S: S, offset: offset)) }
            let out: MLXArray
            if let (top, nBlocks, kvLen) = indexer.lastTop, let kvc = cache as? KVCacheSimple {
                // P054: ENGINE_ABL_SKIP=attnkvstale pays the write's own cost in full but hands the
                // gather the PRE-write arrays anyway, to separate OBS-ENG-122 (2)'s growing cost from
                // a read-side confound `attnkv` (which skips the write too) cannot isolate on its own.
                let staleK = Self.ablKVStale ? kvc.rawKeys : nil
                let staleV = Self.ablKVStale ? kvc.rawValues : nil
                if !Self.ablKVUpdate {
                    if Self.kvUpdateLean { kvc.updateNoReturn(keys: k, values: v) } else { _ = kvc.update(keys: k, values: v) }
                }
                out = Qwen4ExpQSAIndexer.ablQSAGather ? q
                    : q4QSAGather(q: q, keys: staleK ?? kvc.rawKeys!, values: staleV ?? kvc.rawValues!, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: indexer.ratio, scale: scale)
            } else {
                out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: fm)
            }
            if Self.ablOProj {
                return (out.reshaped(B, 1, nHeads * headDim) * sg)[.ellipsis, 0 ..< oProj.weight.dim(0)]
            }
            return q8Proj(oProj, out.reshaped(B, 1, nHeads * headDim) * sg)
        }

        let qg = parts[0].reshaped(B, S, nHeads, 2 * headDim)
        var q = qg[.ellipsis, 0 ..< headDim]
        let gate = qg[.ellipsis, headDim...].reshaped(B, S, nHeads * headDim)
        q = qNorm(q).transposed(0, 2, 1, 3)
        var k = kNorm(parts[1].reshaped(B, S, nKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(B, S, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let (c, s) = pos3 == nil ? rope.tables(offset: offset, count: S)   // (1,S,d)
                                 : rope.tablesMRope(pos3!, section: mropeSection)
        if Q4Fused.rope {
            let ct = c[0], st = s[0]                            // (S,d) float32
            q = q4RopePartial(q, cos: ct, sin: st, rotDim: rope.dim)
            k = q4RopePartial(k, cos: ct, sin: st, rotDim: rope.dim)
        } else {
            let ce = expandedDimensions(c, axis: 1), se = expandedDimensions(s, axis: 1)  // (1,1,S,d)
            q = Qwen4ExpRotary.apply(q, cos: ce, sin: se)
            k = Qwen4ExpRotary.apply(k, cos: ce, sin: se)
        }

        var finalMask = mask
        if let sparse {
            finalMask = .array(Self.sparseMask(sparse, S: S, offset: offset))
        }
        let out: MLXArray
        let gqa = nHeads / nKVHeads
        let rowsPerCall = max(1, 32 / gqa)                       // MLX's sdpa_vector takes q_len * gqa <= 32 query rows per kv head
        if let (top, nBlocks, kvLen) = indexer.lastTop, S <= Qwen4ExpQSAIndexer.gatherMaxS, let kvc = cache as? KVCacheSimple {
            if !(Self.ablKVUpdate && S == 1) {
                if Self.kvUpdateLean { kvc.updateNoReturn(keys: k, values: v) } else { _ = kvc.update(keys: k, values: v) }
            }
            // P019: ENGINE_DUMP_TOP=path -- write the selected block ids of the FIRST prefill chunk that reaches here and exit.
            // The question is whether adjacent query rows of one chunk select overlapping block sets (then a row TILE can read
            // K/V once for the union instead of once per row); nothing else reads this file.
            if S > 16, let dt = Qwen4ExpEnv.dumpTop, !Qwen4ExpQSAIndexer.dumpedTop {
                Qwen4ExpQSAIndexer.dumpedTop = true
                eval(top)
                try? MLX.save(arrays: ["top": top, "meta": MLXArray([Int32(nBlocks), Int32(kvLen), Int32(offset), Int32(indexer.ratio)])], url: URL(fileURLWithPath: dt))
                FileHandle.standardError.write("engine: wrote top \(top.shape) nBlocks=\(nBlocks) kvLen=\(kvLen) offset=\(offset) ratio=\(indexer.ratio) to \(dt)\n".data(using: .utf8)!)
            }
            out = q4QSAGather(q: q, keys: kvc.rawKeys!, values: kvc.rawValues!, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: indexer.ratio, scale: scale)
                .transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim)
        } else if Q4Fused.attnChunk && S > rowsPerCall && S <= Self.attnChunkMaxRows, let kvc = cache as? KVCacheSimple {
            // short decode blocks (MTP verify, S = K+1): above `rowsPerCall` rows MLX leaves the fused vector kernel for the
            // unfused fallback (scores matmul over the whole kv, where(mask), softmax, matmul), whose cost grows with kv
            // (+10 ms per round at 8k, OBS-ENG-029). Attend in row chunks the vector kernel accepts; one cache update, an
            // explicit bool mask per chunk (causal rows differ per chunk, so the kernel's own causal offset cannot be used).
            let (K, V) = kvc.update(keys: k, values: v)
            let fullMask: MLXArray
            if case .array(let m) = finalMask { fullMask = m } else {
                let kvLen = offset + S
                let rinds = Qwen4ExpQSAIndexer.int32Range(0, kvLen)
                let linds = expandedDimensions(Qwen4ExpQSAIndexer.int32Range(kvLen - S, kvLen), axis: 1)
                fullMask = expandedDimensions(linds .>= rinds, axes: [0, 1])            // (1,1,S,kv)
            }
            var parts: [MLXArray] = []
            var r0 = 0
            while r0 < S {
                let r1 = min(S, r0 + rowsPerCall)
                parts.append(MLXFast.scaledDotProductAttention(queries: q[0..., 0..., r0 ..< r1, 0...], keys: K, values: V, scale: scale,
                                                               mask: .array(fullMask[0..., 0..., r0 ..< r1, 0...])))
                r0 = r1
            }
            out = concatenated(parts, axis: 2).transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim)
        } else {
            out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: finalMask)
                .transposed(0, 2, 1, 3).reshaped(B, S, nHeads * headDim)
        }
        Q4Prof.mark("attn.core", [out])
        // ABLATION TRAP, hit and corrected: returning a HELD tensor makes `out * sigmoid(gate)` dead
        // under MLX's laziness, so the gather and q's whole front end get charged to o_proj. A SLICE
        // keeps every upstream node alive and replaces only the 16.7 MB qmv with a strided copy.
        if Self.ablOProj, S == 1 { return (out * sigmoid(gate))[.ellipsis, 0 ..< oProj.weight.dim(0)] }
        let r = q8Proj(oProj, out * sigmoid(gate))
        Q4Prof.mark("attn.oproj", [r])
        return r
    }
}

// MARK: - Gated DeltaNet

final class Qwen4ExpGatedDeltaNet: Module {
    let nV: Int, nK: Int, dK: Int, dV: Int
    let keyDim: Int, valueDim: Int, convKernel: Int, convDim: Int
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    /// in_proj_qkv | in_proj_z (quantised, one qmv) and in_proj_b | in_proj_a (bf16, one gemv) -- merged in sanitize
    @ModuleInfo(key: "in_proj_qkvz") var inProjQKVZ: Linear
    @ModuleInfo(key: "in_proj_ba") var inProjBA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear
    private var qScale: MLXArray? = nil, kScale: MLXArray? = nil
    private var aLogT: MLXArray? = nil, dtBiasT: MLXArray? = nil
    /// P072 arm: ENGINE_GDN_DECAY_MULT m scales the forget rate exp(A_log) by m, i.e. A_log + ln(m) (1 = champion)
    private var aLogEff: MLXArray { Qwen4ExpVariants.gdnDecayMult == 1 ? aLog : aLog + MLXArray(log(Qwen4ExpVariants.gdnDecayMult)) }

    init(_ a: Qwen4ExpTextConfiguration) {
        nV = a.linearNumValueHeads; nK = a.linearNumKeyHeads
        dK = a.linearKeyHeadDim; dV = a.linearValueHeadDim
        keyDim = dK * nK; valueDim = dV * nV
        convKernel = a.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        let d = a.hiddenSize
        _conv1d.wrappedValue = Conv1d(inputChannels: convDim, outputChannels: convDim, kernelSize: convKernel,
                                      stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _inProjQKVZ.wrappedValue = Linear(d, convDim + valueDim, bias: false)
        _inProjBA.wrappedValue = Linear(d, 2 * nV, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([nV])
        _aLog.wrappedValue = MLXArray.zeros([nV])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(dimensions: dV, eps: a.rmsNormEps, activation: a.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, d, bias: false)
        super.init()
    }

    /// state after the first `n` rows of the last multi-token block (no projections re-run)
    /// P099: per-row replay after a batched verify block -- row b keeps its first keep[b] rows of the tape; rows that
    /// kept the whole block are left as the forward wrote them (the same computation).
    func replayPrefixRows(cache: ArraysCache, keep: [Int]) {
        guard let tape = cache.prefixReplayTape else { return }
        let S = tape.rowCount, B = keep.count
        var state = cache[1]!, conv = cache[0]!
        for b in 0 ..< B where keep[b] < S {
            let n = keep[b]
            let pre = tape.ssmPre.map { $0[b ..< (b + 1)] } ?? MLXArray.zeros([1, nV, dV, dK], dtype: .float32)
            if n == 0 { state[b ..< (b + 1)] = pre }
            else {
                let r = 0 ..< n
                state[b ..< (b + 1)] = gatedDeltaUpdate(q: tape.q[b ..< (b + 1), r, 0...], k: tape.k[b ..< (b + 1), r, 0...], v: tape.v[b ..< (b + 1), r, 0...],
                                                        a: tape.a[b ..< (b + 1), r, 0...], b: tape.b[b ..< (b + 1), r, 0...],
                                                        aLog: aLogEff, dtBias: dtBias, state: pre, mask: nil).1
            }
            conv[b ..< (b + 1)] = tape.convInput[b ..< (b + 1), n ..< (n + tape.convStateRows), 0...]
        }
        cache[1] = state; cache[0] = conv
        cache.prefixReplayTape = nil
    }

    func replayPrefix(cache: ArraysCache, rows n: Int) {
        guard let tape = cache.prefixReplayTape, n >= 0, n <= tape.rowCount else { return }
        let r = 0 ..< n
        if n == 0 {
            cache[1] = tape.ssmPre ?? MLXArray.zeros([1, nV, dV, dK], dtype: .float32)
        } else {
            cache[1] = gatedDeltaUpdate(q: tape.q[0..., r, 0...], k: tape.k[0..., r, 0...], v: tape.v[0..., r, 0...],
                                        a: tape.a[0..., r, 0...], b: tape.b[0..., r, 0...],
                                        aLog: aLogEff, dtBias: dtBias, state: tape.ssmPre, mask: nil).1
        }
        cache[0] = tape.convInput[0..., n ..< (n + tape.convStateRows), 0...]
        cache.prefixReplayTape = nil
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let qkvz = q8Proj(inProjQKVZ, x)
        let mixedQKV = qkvz[.ellipsis, 0 ..< convDim]
        let z = qkvz[.ellipsis, convDim...].reshaped(B, S, nV, dV)
        let ba = prefillProjection(inProjBA, x, "ba")
        let b = ba[.ellipsis, 0 ..< nV]
        let a = ba[.ellipsis, nV...]
        Q4Prof.mark("gdn.inproj", [qkvz, ba])
        let convState = cache?[0] ?? MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let ssmPre = cache?[1]
        var q: MLXArray, k: MLXArray
        let v: MLXArray
        let convInput: MLXArray
        if Q4Fused.gdnFront && S == 1 && dK == dV {
            // decode: conv + silu + head norms + conv-state shift in one kernel, reading q|k|v straight out of qkvz (no concat, no slices)
            let inv = pow(Float(dK), -0.5)
            let (qkv, newConv) = q4GDNConvNorm(qkvz: qkvz.reshaped(B * S, qkvz.dim(-1)), convState: convState, convW: conv1d.weight,
                                               convDim: convDim, kernel: convKernel, headDim: dK, keyDim: keyDim, scaleQ: inv * inv, scaleK: inv, eps: 1e-6)
            cache?[0] = newConv
            q = qkv[0..., 0 ..< keyDim].reshaped(B, S, nK, dK)
            k = qkv[0..., keyDim ..< (2 * keyDim)].reshaped(B, S, nK, dK)
            v = qkv[0..., (2 * keyDim)...].reshaped(B, S, nV, dV)
            convInput = newConv      // only read by the S > 1 tape below (never for S == 1)
        } else {
        // P019 unit 7: for a prefill chunk the concatenate + Conv1d + silu chain is three passes over an 84 MB tensor at
        // a 4096-row chunk; one kernel reads the mixed q|k|v straight out of qkvz and the K-1 cached rows instead. The
        // replay tape (partial-acceptance rollback of an MTP verify block) still needs the concatenated form, but only
        // for the small S it is ever replayed at -- a prefill chunk is never rolled back.
        let convOut: MLXArray
        if q4GDNConvPrefillOn && S > 64 && S >= convKernel - 1 && B == 1 {
            convOut = q4GDNConvPrefill(qkvz: qkvz.reshaped(B * S, qkvz.dim(-1)), convState: convState,
                                       convW: conv1d.weight, convDim: convDim, kernel: convKernel)
                .reshaped(B, S, convDim)
            // `contiguous` matters: a slice is a VIEW, so keeping it in the cache would pin the whole 84 MB qkvz buffer
            // of every layer (measured: peak +1.9 GB at a 4096-row chunk)
            convInput = contiguous(mixedQKV[0..., (S - (convKernel - 1))..., 0...])
            if let cache { cache[0] = convInput }
        } else {
            convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache {
                cache[0] = convInput[0..., (convInput.dim(1) - (convKernel - 1))..., 0...]
            }
            convOut = q4Silu(conv1d(convInput))
        }
        Q4Prof.mark("gdn.conv", [convOut])
        let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        q = parts[0].reshaped(B, S, nK, dK)
        k = parts[1].reshaped(B, S, nK, dK)
        v = parts[2].reshaped(B, S, nV, dV)
        if qScale == nil || qScale!.dtype != x.dtype {
            let invScale = pow(Float(dK), -0.5)
            qScale = MLXArray(invScale * invScale).asType(x.dtype); kScale = MLXArray(invScale).asType(x.dtype)
            eval(qScale!, kScale!)
        }
        if Q4Fused.headNorm {
            let inv = pow(Float(dK), -0.5)
            q = q4HeadNormScale(q, headDim: dK, scale: inv * inv, eps: 1e-6)
            k = q4HeadNormScale(k, headDim: dK, scale: inv, eps: 1e-6)
        } else {
            q = qScale! * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            k = kScale! * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
        }
        }
        Q4Prof.mark("gdn.headnorm", [q, k, v])
        let out: MLXArray, newState: MLXArray
        if Q4Fused.gdnGate {
            if aLogT == nil || aLogT!.dtype != x.dtype { aLogT = aLogEff.asType(x.dtype); dtBiasT = dtBias.asType(x.dtype); eval(aLogT!, dtBiasT!) }
            let (g, beta) = q4GDNGate(ba: ba, aLog: aLogT!, dtBias: dtBiasT!, nV: nV)
            (out, newState) = gatedDeltaUpdatePrecomputed(q: q, k: k, v: v, g: g, beta: beta, state: cache?[1], mask: nil)
        } else {
            (out, newState) = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLogEff, dtBias: dtBias, state: cache?[1], mask: nil)
        }
        Q4Prof.mark("gdn.recur", [out, newState])
        if let cache {
            cache[1] = newState
            cache.advance(S)
            // speculative verify blocks: keep what a partial acceptance needs to rebuild the state
            // after any prefix without re-running the projections (arena PrefixReplayTape)
            if S > 1 && S <= 64 {   // only a verify block is ever replayed; a prefill chunk's tape would hold 84 MB per layer
                cache.prefixReplayTape = ArraysCache.PrefixReplayTape(
                    convInput: convInput, q: q, k: k, v: v, a: a, b: b, g: a, beta: b, ssmPre: ssmPre,
                    mask: nil, rowCount: S, convStateRows: convKernel - 1)
            } else {
                cache.prefixReplayTape = nil
            }
        }
        // the recurrence runs in float32; the model runs in x.dtype (bf16) -- do not let fp32 leak out
        let r: MLXArray
        if Q4Fused.gdnFront {
            let n = q4GatedNormSG(out, weight: norm.weight, gate: z, headDim: dV, eps: norm.eps, sigmoidGate: norm.sigmoidGate, outDType: x.dtype)
            r = q8Proj(outProj, n.reshaped(B, S, valueDim))
        } else if Q4Fused.gatedNorm {
            let n = q4GatedNorm(out, weight: norm.weight, gate: z, headDim: dV, eps: norm.eps, sigmoidGate: norm.sigmoidGate, outDType: x.dtype)
            r = q8Proj(outProj, n.reshaped(B, S, valueDim))
        } else {
            r = q8Proj(outProj, norm(out, gate: z, outDType: x.dtype).reshaped(B, S, valueDim))
        }
        Q4Prof.mark("gdn.out", [r])
        return r
    }
}

// MARK: - MoE

final class Qwen4ExpMLP: Module, UnaryLayer {
    let hidden: Int
    /// gate_proj | up_proj concatenated on the output axis (sanitize): one qmv, then silu(g)*u in one kernel
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(dimensions: Int, hidden: Int) {
        self.hidden = hidden
        _gateUpProj.wrappedValue = Linear(dimensions, 2 * hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dimensions, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(q4SiluMulSplit(q8Proj(gateUpProj, x), hidden: hidden))
    }
}

final class Qwen4ExpSparseMoeBlock: Module {
    let topK: Int
    static let dumpRouter: String? = ProcessInfo.processInfo.environment["ENGINE_DUMP_ROUTER"]
    static let dumpRouterN: Int = Int(ProcessInfo.processInfo.environment["ENGINE_DUMP_ROUTER_N"] ?? "1") ?? 1
    nonisolated(unsafe) static var routerDumps = 0
    let numExperts: Int
    /// router `gate` [E, d] | `shared_expert_gate` [1, d] (bf16), one gemv; logits = out[..., 0..<E], shared gate = out[..., E]
    @ModuleInfo(key: "gate_sg") var gateSG: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP

    init(_ a: Qwen4ExpTextConfiguration) {
        topK = a.numExpertsPerTok
        numExperts = a.numExperts
        _gateSG.wrappedValue = Linear(a.hiddenSize, a.numExperts + 1, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(inputDims: a.hiddenSize, hiddenDims: a.moeIntermediateSize, numExperts: a.numExperts)
        if q4SiluMode == 0 { _switchMLP.wrappedValue.activationProduct = { g, u in q4SiluMul(g, u) } }   // one dispatch, no evalLock
        else if q4SiluMode == 1 { _switchMLP.wrappedValue.activationProduct = { g, u in g * sigmoid(g) * u } }
        _sharedExpert.wrappedValue = Qwen4ExpMLP(dimensions: a.hiddenSize, hidden: a.sharedExpertIntermediateSize)
        super.init()
    }

    nonisolated(unsafe) static var routerStats: Bool = ProcessInfo.processInfo.environment["ENGINE_ROUTER_STATS"] != nil
    nonisolated(unsafe) static var statPairs = 0
    nonisolated(unsafe) static var statDistinct = 0
    nonisolated(unsafe) static var statCalls = 0
    nonisolated(unsafe) static var statRows = 0
    private var gate32: MLXArray? = nil
    private var compiled: (@Sendable (MLXArray) -> MLXArray)? = nil
    private var banksCache: (Q4ExpertBank, Q4ExpertBank, Q4ExpertBank, Int)? = nil
    private var banksTried = false
    /// the quantised expert banks (weight/scales/biases) by parameter key; nil unless every bank is
    /// 4-bit g32, or -- with Q4Fused.moeExperts8 -- 8-bit g64.
    ///
    /// P025: this gate is why the fused expert path was DEAD on E8h. E8h's routed experts are 8-bit
    /// g64 and the old body accepted 4-bit g32 only, so every decode token on the operating artifact
    /// fell through to MLX gather_qmm while the project's own kernels sat unused. Measured on the
    /// artifact where they DO run (mix1v, 512): serial 57.57 -> 63.17 tok/s, +9.7% (OBS-ENG-077).
    private func expertBanks() -> (Q4ExpertBank, Q4ExpertBank, Q4ExpertBank, Int)? {
        if banksTried { return banksCache }
        banksTried = true
        let p = Dictionary(uniqueKeysWithValues: switchMLP.parameters().flattened())
        func bank(_ name: String, inDim: Int) -> Q4ExpertBank? {
            guard let w = p[name + ".weight"], let s = p[name + ".scales"], let b = p[name + ".biases"] else { return nil }
            guard w.dtype == .uint32 else { return nil }
            // 4-bit: 8 values per uint32 -> weight last dim = in/8; g32: scales last dim = in/32
            if w.dim(-1) == inDim / 8, s.dim(-1) == inDim / 32 { return Q4ExpertBank(w: w, s: s, b: b, bits: 4, group: 32) }
            // 8-bit: 4 values per uint32 -> in/4; g64: scales last dim = in/64
            if Q4Fused.moeExperts8, w.dim(-1) == inDim / 4, s.dim(-1) == inDim / 64 { return Q4ExpertBank(w: w, s: s, b: b, bits: 8, group: 64) }
            return nil
        }
        let D = gateSG.weight.dim(-1)
        guard let g = bank("gate_proj", inDim: D), let u = bank("up_proj", inDim: D) else { return nil }
        let I = g.w.dim(1)
        guard let d = bank("down_proj", inDim: I) else { return nil }
        // MIXED formats would read one bank with the other's unpack. Refuse rather than compute garbage.
        guard g.bits == u.bits, u.bits == d.bits else { return nil }
        // the no-lane variant has no 8-bit twin (q4MoEExperts precondition)
        guard g.bits == 4 || Q4Fused.moeDownLane else { return nil }
        banksCache = (g, u, d, I)
        return banksCache
    }
    /// P025: what the fused expert path actually resolves to on THIS checkpoint. Read at decode
    /// widths (rows = 1), which is the only place the path is reachable.
    func expertPathWitness() -> String {
        guard Q4Fused.moeExperts else { return "OFF (bit 256 clear)" }
        guard let b = expertBanks() else { return "OFF (banks rejected: experts are not a format the fused path accepts)" }
        return "LIVE \(b.0.bits)-bit g\(b.0.group)"
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Qwen4ExpGatedResidual.useCompile {
            if compiled == nil {
                compiled = compile(inputs: [self], shapeless: false) { [unowned self] a in self.forwardUncompiled(a) }
            }
            return compiled!(x)
        }
        return forwardUncompiled(x)
    }

    /// the fused expert path WITHOUT the combine: (yk (rows,K,D), w (rows,K), shared (lead...,D), shared gate (lead...,1)) for a consumer
    /// that folds the combine (q4HCInjectMoE); nil when the fused path does not apply
    func forwardPieces(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)? {
        guard Q4Fused.moeTopK, Q4Fused.moeExperts, Q4Fused.moeDownLane, !Qwen4ExpGatedResidual.useCompile,
              !Qwen4ExpSparseMoeBlock.routerFP32 else { return nil }
        let rows = x.size / x.dim(-1)
        guard rows <= 8, let banks = expertBanks() else { return nil }
        let gsg = routerLogits(x)
        let (idx, w, sgate) = q4MoETopK(logits: gsg, experts: numExperts, topK: topK, sharedGate: true)
        let xf = x.reshaped(rows, x.dim(-1)); let idf = idx.reshaped(rows, topK); let wf = w.reshaped(rows, topK)
        let yk = q4MoEExpertsK(x: xf, idx: idf, gate: banks.0, up: banks.1, down: banks.2, K: topK, D: x.dim(-1), I: banks.3)
        return (yk, wf, sharedExpert(x), sgate)
    }

    /// P038: the router matmul in float32 instead of the model dtype. DEFAULT OFF and it must stay
    /// off: HF runs the router Linear in the MODEL dtype and this port matches it (OBS-ENG-041 (1)).
    /// `mlx-lm` PR#1788 casts it to float32, which is the ONE known non-conformance of the parity
    /// reference. This knob makes the engine deliberately WRONG in the same way, so the parity
    /// disagreement can be attributed instead of argued.
    static let routerFP32 = ProcessInfo.processInfo.environment["ENGINE_ROUTER_FP32"] != nil

    /// NOTE, and this is why the first version of this knob read ZERO: MLX's bf16 matmul already
    /// ACCUMULATES in float32 and rounds once at the end, so computing in float32 and casting the
    /// RESULT back to bf16 reproduces the bf16 matmul bit for bit. The extra precision only changes
    /// anything if it survives into the top-k SELECTION, so the fp32 arm must return fp32.
    func routerLogits(_ x: MLXArray) -> MLXArray {
        guard Qwen4ExpSparseMoeBlock.routerFP32 else { return prefillProjection(gateSG, x, "router") }
        return matmul(x.asType(.float32), gateSG.weight.asType(.float32).transposed())
    }

    func forwardUncompiled(_ x: MLXArray) -> MLXArray {
        // router: bf16 matmul (HF runs the router Linear in the model dtype), fp32 softmax
        let gsg = routerLogits(x)
        let idx: MLXArray, w: MLXArray, sgate: MLXArray
        if Q4Fused.moeTopK, !Qwen4ExpSparseMoeBlock.routerFP32 {
            (idx, w, sgate) = q4MoETopK(logits: gsg, experts: numExperts, topK: topK, sharedGate: true)
        } else {
            let logits = gsg[.ellipsis, 0 ..< numExperts]
            sgate = gsg[.ellipsis, numExperts...]
            idx = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
            w = softmax(takeAlong(logits, idx, axis: -1).asType(.float32), axis: -1, precise: true).asType(x.dtype)
        }
        let rows = x.size / x.dim(-1)
        // P025 unit 6: ENGINE_ROUTER_STATS=1 -- at DECODE widths (2..8 rows), how many DISTINCT
        // experts does a block of rows select? The whole multi-row expert kernel is worth building
        // only if the rows SHARE experts; if all rows x topK pairs are distinct, reading each expert
        // once IS what the shipped kernel already does and there is nothing to dedup. This forces a
        // host sync on a tiny array, so it is a DIAGNOSTIC BUILD SWITCH and never on in a timed run.
        if Qwen4ExpSparseMoeBlock.routerStats, rows >= 2, rows <= 8 {
            let flat = idx.reshaped(rows * topK).asType(.int32).asArray(Int32.self)
            var seen = Set<Int32>(); for v in flat { seen.insert(v) }
            Qwen4ExpSparseMoeBlock.statPairs += flat.count
            Qwen4ExpSparseMoeBlock.statDistinct += seen.count
            Qwen4ExpSparseMoeBlock.statCalls += 1
            Qwen4ExpSparseMoeBlock.statRows = rows
            if Qwen4ExpSparseMoeBlock.statCalls % 480 == 0 {
                let p = Double(Qwen4ExpSparseMoeBlock.statPairs), d = Double(Qwen4ExpSparseMoeBlock.statDistinct)
                FileHandle.standardError.write(String(format:
                    "qwen4_exp: ROUTER STATS rows=%d calls=%d  pairs/call %.2f  distinct/call %.2f  SHARING %.2f%% (weight traffic multiplier %.3fx)\n",
                    rows, Qwen4ExpSparseMoeBlock.statCalls, p / Double(Qwen4ExpSparseMoeBlock.statCalls),
                    d / Double(Qwen4ExpSparseMoeBlock.statCalls), 100.0 * (1.0 - d / p), d / p).data(using: .utf8)!)
            }
        }
        // ENGINE_DUMP_ROUTER=path: the REAL rows-per-expert distribution of one prefill chunk,
        // written once per process. lab/moe_ragged_lab.py prices raggedness on RANDOM logits;
        // whether the trained router is that ragged is a separate, measurable question.
        if let dumpPath = Qwen4ExpSparseMoeBlock.dumpRouter, rows > 16,
           Qwen4ExpSparseMoeBlock.routerDumps < Qwen4ExpSparseMoeBlock.dumpRouterN {
            let n = Qwen4ExpSparseMoeBlock.routerDumps
            Qwen4ExpSparseMoeBlock.routerDumps += 1
            let flat = idx.reshaped(rows * topK).asType(.int32).asArray(Int32.self)
            try? Data(bytes: flat, count: flat.count * 4).write(to: URL(fileURLWithPath: "\(dumpPath).\(n)"))
            if n == 0 {
                FileHandle.standardError.write("qwen4_exp: router dump \(rows) rows x \(topK), \(Qwen4ExpSparseMoeBlock.dumpRouterN) call(s) -> \(dumpPath).N\n".data(using: .utf8)!)
            }
        }
        if Q4Fused.moeExperts && rows <= q4MoEFusedMaxRows, let banks = expertBanks() {
            let xf = x.reshaped(rows, x.dim(-1)); let idf = idx.reshaped(rows, topK); let wf = w.reshaped(rows, topK)
            if Q4Fused.moeDownLane {
                let yk = q4MoEExpertsK(x: xf, idx: idf, gate: banks.0, up: banks.1, down: banks.2, K: topK, D: x.dim(-1), I: banks.3)
                return q4MoECombine(y: yk, w: wf, shared: sharedExpert(x), gate: sgate, topK: topK, d: x.dim(-1))
            }
            let y = q4MoEExperts(x: xf, idx: idf, wts: wf, gate: banks.0, up: banks.1, down: banks.2,
                                 K: topK, D: x.dim(-1), I: banks.3).reshaped(x.shape)
            return y + sigmoid(sgate) * sharedExpert(x)
        }
        let yk = switchMLP(x, idx)
        if Q4Fused.moeCombine {
            return q4MoECombine(y: yk, w: w, shared: sharedExpert(x), gate: sgate, topK: topK, d: x.dim(-1))
        }
        let y = (yk * expandedDimensions(w, axis: -1)).sum(axis: -2)
        return y + sigmoid(sgate) * sharedExpert(x)
    }
}

// MARK: - Hyper-connections (gated residual)

final class Qwen4ExpGatedResidual: Module {
    let hc: Int
    let d: Int
    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var inject: Linear?

    init(_ a: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        hc = a.hcCount
        d = a.hiddenSize
        let hcDim = hc * d
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: a.rmsNormEps)
        _mixDown.wrappedValue = Linear(hcDim, a.hcLowrank, bias: false)
        _mixUp.wrappedValue = Linear(a.hcLowrank, hcDim, bias: false)
        if useCombine {
            _inject.wrappedValue = Linear(hcDim, hc, bias: false)
        }
        super.init()
    }

    var debugSink: ((String, MLXArray) -> Void)? = nil
    private var compiled: (@Sendable ([MLXArray]) -> [MLXArray])? = nil
    static let useCompile: Bool = ProcessInfo.processInfo.environment["ENGINE_COMPILE"] != nil   // measured slower (31.1 vs 33.3 tok/s); opt-in
    static let hcSplit: Int = Int(ProcessInfo.processInfo.environment["ENGINE_HC_SPLIT"] ?? "10") ?? 10   // mixDown split-K chunks (K = 10240 -> 1024 per chunk); 5/10/20 equal in the lab

    /// (mixed (B,S,d), inject (B,S,hc)?) ; `hyper` itself is the residual to add to
    /// true when the fused chain will run for this input (so a caller may precompute `normed` for it)
    /// true when the fused chain will run. TWO bodies now: the 8-bit g64 one, and (unit 36) a DENSE
    /// one for a full-precision checkpoint. Before the dense body existed, a bf16 hyper-connection
    /// fell back to nine MLX ops and cost +2.57 ms/token (OBS-ENG-066 (2)) -- precision and speed were
    /// in conflict here only because nobody had written it.
    var mixUpDenseFusable: Bool {
        !(mixUp is QuantizedLinear) && mixDown as? QuantizedLinear == nil
            && mixUp.weight.dim(0) == hc * d && mixUp.weight.dim(1) % 4 == 0 && (hc * d) % 256 == 0
    }
    func fusedChainActive(rows: Int) -> Bool {
        guard Q4Fused.hcFused, debugSink == nil, rows <= 16 else { return false }
        if (mixUp as? QuantizedLinear).map({ $0.bits == 8 && $0.groupSize == 64 }) == true { return true }
        return Q4Fused.hcFusedDense && mixUpDenseFusable
    }
    func callAsFunction(_ hyper: MLXArray, normed pre: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        if let pre, fusedChainActive(rows: hyper.size / (hc * d)) { return forwardFused(hyper, normed: pre) }
        if let pre, debugSink == nil, !Self.useCompile { return forwardUncompiled(hyper, normed: pre) }
        if Self.useCompile && debugSink == nil {
            if compiled == nil {
                compiled = compile(inputs: [self], shapeless: false) { [unowned self] a in
                    let r = self.forwardUncompiled(a[0])
                    return r.1 == nil ? [r.0] : [r.0, r.1!]
                }
            }
            let r = compiled!([hyper])
            return (r[0], r.count > 1 ? r[1] : nil)
        }
        return forwardUncompiled(hyper)
    }

    /// ENGINE_HC_EXTRA_HOP=n: insert n EXTRA dependent kernel boundaries into the HC chain, adding
    /// zero arithmetic and zero bytes of weight. Purpose is measurement, not optimisation: the lab
    /// says the HC block costs 58.2 us dependent against 21.4 us for the same kernels run
    /// independently, so the tax should be the BOUNDARIES. n>0 prices one boundary IN SITU, and the
    /// slope is what removing a boundary by fusion would pay back. Default 0; numerics unchanged
    /// (bf16 x + 0 is exact).
    static let extraHop: Int = Int(ProcessInfo.processInfo.environment["ENGINE_HC_EXTRA_HOP"] ?? "0") ?? 0
    nonisolated(unsafe) static var hopZero: MLXArray? = nil

    private func forwardFused(_ hyper: MLXArray, normed: MLXArray) -> (MLXArray, MLXArray?) {
        if !(mixUp is QuantizedLinear) {
            let winj = inject?.weight ?? MLXArray.zeros([hc, hc * d], dtype: hyper.dtype)
            let (mixed, injp): (MLXArray, MLXArray)
            if Q4Fused.hcSplitK, let parts = bf16ProjSplitKPartials(mixDown, normed, split: Self.hcSplit) {
                (mixed, injp) = bf16HCUpMix(down: parts, normed: normed, up: mixUp, winj: winj, hc: hc, d: d, downSplits: Self.hcSplit)
            } else {
                (mixed, injp) = bf16HCUpMix(down: mixDown(normed), normed: normed, up: mixUp, winj: winj, hc: hc, d: d)
            }
            var mixedOut = mixed
            if Self.extraHop > 0 {
                if Self.hopZero == nil { Self.hopZero = MLXArray([Float(0)]).asType(mixed.dtype) }
                for _ in 0 ..< Self.extraHop { mixedOut = mixedOut + Self.hopZero! }
            }
            return (mixedOut, injp)
        }
        let up = mixUp as! QuantizedLinear
        // mixDown is skinny (N = lowrank 320, K = hc*d = 10240): on the dependent critical path MLX's qmv takes ~28 us for 3.3 MB
        // (too few threadgroups to hide DRAM latency); split-K (20 chunks, 6400 simdgroups) takes ~13 us. Measured in lab/hc_chain_dram.py.
        // Deterministic split-K (10 K-chunks, float32 partials summed while up_mix stages silu(down)); the atomic variant made
        // MTP drafts run-to-run nondeterministic. lab/hc_chain_dram.py: 61.6 -> 41.0 us per HC (rows=1), 86.3 -> 51.8 (rows=4).
        let winj = inject?.weight ?? MLXArray.zeros([hc, hc * d], dtype: hyper.dtype)
        let (mixed, injp): (MLXArray, MLXArray)
        if Q4Fused.hcSplitK, let parts = q8ProjSplitKPartials(mixDown, normed, split: Self.hcSplit) {
            (mixed, injp) = q4HCUpMix(down: parts, normed: normed, up: up, winj: winj, hc: hc, d: d, downSplits: Self.hcSplit)
        } else {
            let down = q8Proj(mixDown, normed)
            (mixed, injp) = q4HCUpMix(down: down, normed: normed, up: up, winj: winj, hc: hc, d: d)
        }
        var mixedOut = mixed
        if Self.extraHop > 0 {
            if Self.hopZero == nil { Self.hopZero = MLXArray([Float(0)]).asType(mixed.dtype) }
            for _ in 0 ..< Self.extraHop { mixedOut = mixedOut + Self.hopZero! }   // a real dependent dispatch each
        }
        return (mixedOut, inject == nil ? nil : injp)     // injp: per-threadgroup partial inject sums, consumed by q4HCInjectP / q4HCInjectNorm
    }
    func forwardUncompiled(_ hyper: MLXArray, normed pre: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        if pre == nil, fusedChainActive(rows: hyper.size / (hc * d)) {   // decode / verify rows; prefill keeps the qmm path
            return forwardFused(hyper, normed: q4HCNorm(hyper, onePlusW: hcNorm.onePlusWeight(hyper.dtype), groups: hc, groupSize: d, eps: hcNorm.eps))
        }
        let normed = pre ?? hcNorm(hyper)
        Q4Prof.mark("hc.norm", [normed])
        debugSink?("hc_normed", normed)
        debugSink?("hc_norm_weight", expandedDimensions(hcNorm.weight, axis: 0))
        let down = prefillProjection(mixDown, normed, "hc")
        Q4Prof.mark("hc.down", [down])
        debugSink?("hc_down", down)
        if Q4Fused.hcMix && debugSink == nil {
            let u = mixUp(Q4Fused.siluDiv ? q4SiluDiv(down, by: hc) : q4Silu(down / Float(hc)))
            Q4Prof.mark("hc.up", [u])
            // P020 unit 10: the mix and the inject logits both read the SAME 84 MB `normed` at a prefill chunk.
            // One kernel reads it once and produces both. ENGINE_HC_MIX_INJECT=0 restores the two passes.
            if q4HCMixInjectOn, Q4Fused.hcInject, let inject, hyper.size / (hc * d) > 16,
               let w = inject.weight as MLXArray?, w.dtype == hyper.dtype, d % 256 == 0 {
                let (mixed, inj) = q4HCMixInject(normed: normed, u: u, winj: w, hc: hc, d: d, rows: q4HCMixInjectRows)
                Q4Prof.mark("hc.mix", [mixed, inj])
                return (mixed, inj)     // raw inject logits; the combine kernel applies 2*sigmoid(./hc)
            }
            let mixed = q4HCMix(normed: normed, u: u, hc: hc, d: d)
            Q4Prof.mark("hc.mix", [mixed])
            guard let inject else { return (mixed, nil) }
            if Q4Fused.hcInject {
                let inj = inject(normed)
                Q4Prof.mark("hc.inject", [inj])
                return (mixed, inj)     // raw inject logits; the combine kernel applies 2*sigmoid(./hc)
            }
            return (mixed, 2 * sigmoid(inject(normed) / Float(hc)))
        }
        var w = q4Silu(down / Float(hc))
        w = sigmoid(mixUp(w))
        debugSink?("hc_w", w)
        let lead = Array(w.shape.dropLast())
        w = w.reshaped(lead + [hc, d])
        let mixed = (w * normed.reshaped(lead + [hc, d])).mean(axis: -2)
        guard let inject else { return (mixed, nil) }
        let inj = 2 * sigmoid(inject(normed) / Float(hc))
        return (mixed, inj)
    }
}

// MARK: - n-gram / PLE

/// Host-side id math for the hashed n-gram table (mirrors the reference
/// `_shift_right` + mixed-id hash; int64 wrapping multiply, xor, floor mod).
public struct Qwen4ExpNgramHash {
    public let ngramSize: Int
    public let headsPerNgram: Int
    public let nHeads: Int
    public let eos: Int
    public var multipliers: [Int64]
    public var vocab: [Int64]
    public var offsets: [Int64]
    public let totalRows: Int
    public let rowsPerShard: Int
    public let nShards: Int

    private static func splitmix64(_ v0: UInt64) -> UInt64 {
        var v = v0 &+ 0x9E37_79B9_7F4A_7C15
        v = (v ^ (v >> 30)) &* 0xBF58_476D_1CE4_E5B9
        v = (v ^ (v >> 27)) &* 0x94D0_49BB_1331_11EB
        return v ^ (v >> 31)
    }
    private static func isPrime(_ v: Int) -> Bool {
        if v < 2 { return false }
        if v % 2 == 0 { return v == 2 }
        var d = 3
        while d * d <= v { if v % d == 0 { return false }; d += 2 }
        return true
    }
    private static func nthPrimeAfter(_ start: Int, _ count: Int) -> Int {
        var p = start
        for _ in 0 ..< count { p += 1; while !isPrime(p) { p += 1 } }
        return p
    }

    public init(_ a: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        ngramSize = a.ngramSize
        headsPerNgram = a.headsPerNgram
        nHeads = (a.ngramSize - 1) * a.headsPerNgram
        eos = a.eosTokenId
        let maxLong: UInt64 = (1 << 63) - 1
        let half = max(1, (maxLong / UInt64(max(a.vocabularySize, 1))) / 2)
        let baseSeed = UInt64(a.seed) &+ 10007 &* UInt64(pleLayerIndex)
        var m: [Int64] = []
        for i in 0 ..< a.ngramSize {
            let v = baseSeed &+ 0x9E37_79B9_7F4A_7C15 &* UInt64(i + 1)
            m.append(Int64(2 * (Self.splitmix64(v) % half) + 1))
        }
        multipliers = m
        var sizes: [Int64] = [], offs: [Int64] = []
        var total: Int64 = 0
        for h in 0 ..< nHeads {
            let g = pleLayerIndex * nHeads + h
            let s = Int64(Self.nthPrimeAfter(a.ngramVocabSizeBase - 1, g + 1))
            sizes.append(s); offs.append(total); total += s
        }
        vocab = sizes; offsets = offs
        let div = Int64(a.ngramDivisibleBy)
        let padded = Int((total + div - 1) / div * div)
        totalRows = padded
        nShards = a.splitNgramParts
        rowsPerShard = (padded + nShards - 1) / nShards
    }

    public func with(multipliers m: [Int64], vocab v: [Int64], offsets o: [Int64]) -> Qwen4ExpNgramHash {
        var c = self
        c.multipliers = m; c.vocab = v; c.offsets = o
        return c
    }

    @inline(__always) private static func floorMod(_ x: Int64, _ m: Int64) -> Int64 {
        let r = x % m
        return r < 0 ? r + m : r
    }

    /// Global row ids [ids.count * nHeads] for `ids`, preceded by `prev` (ngramSize-1 tokens).
    public func rowIds(prev: [Int], ids: [Int]) -> [Int64] {
        let ctx = ngramSize - 1
        precondition(prev.count == ctx)
        let all = prev + ids
        var out = [Int64](repeating: 0, count: ids.count * nHeads)
        var lastEos = -1
        for t in 0 ..< all.count {
            let tok = all[t]
            if t >= ctx {
                let segPos = t - (lastEos + 1)
                var mixed = Int64(tok) &* multipliers[0]
                let row = (t - ctx) * nHeads
                var pos = 1
                for n in 2 ... ngramSize {
                    while pos < n {
                        let shifted = (segPos >= pos && t >= pos) ? all[t - pos] : eos
                        mixed ^= Int64(shifted) &* multipliers[pos]
                        pos += 1
                    }
                    let h0 = (n - 2) * headsPerNgram
                    for h in h0 ..< h0 + headsPerNgram {
                        out[row + h] = Self.floorMod(mixed, vocab[h]) + offsets[h]
                    }
                }
            }
            if tok == eos { lastEos = t }
        }
        return out
    }
}

/// 128 embedding shards addressed by global row id; touched shards decided on the host.
final class Qwen4ExpShardedEmbedding: Module {
    @ModuleInfo(key: "shards") var shards: [Embedding]
    let rows: Int
    let dim: Int

    init(nShards: Int, rows: Int, dim: Int) {
        self.rows = rows
        self.dim = dim
        _shards.wrappedValue = (0 ..< nShards).map { _ in Embedding(embeddingCount: rows, dimensions: dim) }
        super.init()
    }

    /// The device hash lookup needs one contiguous table. Keep the public shard parameters as
    /// row views into that same allocation, rather than retaining a second full embedding table.
    /// Evaluate both concatenations and views before rebinding: no lazy graph may keep the old
    /// shard buffers alive or read a parameter after its MLXArray context has been replaced.
    func mergeQuantizedStorage() -> (weight: MLXArray, scales: MLXArray, biases: MLXArray?,
                                     groupSize: Int, bits: Int, witness: [String: Int]) {
        let q = shards.compactMap { $0 as? QuantizedEmbedding }
        precondition(!q.isEmpty && q.count == shards.count, "n-gram shards must be quantised embeddings")
        let first = q[0]
        precondition(q.allSatisfy {
            $0.groupSize == first.groupSize && $0.bits == first.bits && $0.mode == first.mode
                && $0.weight.shape == first.weight.shape && $0.scales.shape == first.scales.shape
                && $0.biases?.shape == first.biases?.shape && $0.weight.dim(0) == rows
        }, "n-gram shard storage must have uniform quantisation and row geometry")
        let logicalBytes = q.reduce(0) { $0 + $1.weight.nbytes + $1.scales.nbytes + ($1.biases?.nbytes ?? 0) }
        let activeBefore = Memory.activeMemory
        let weight = concatenated(q.map { $0.weight }, axis: 0)
        let scales = concatenated(q.map { $0.scales }, axis: 0)
        let biases = first.biases == nil ? nil : concatenated(q.map { $0.biases! }, axis: 0)
        eval([weight, scales] + (biases.map { [$0] } ?? []))
        let activeAfterMerge = Memory.activeMemory
        for (i, shard) in q.enumerated() {
            let range = (i * rows)..<((i + 1) * rows)
            var views: [String: MLXArray] = ["weight": weight[range], "scales": scales[range]]
            if let biases { views["biases"] = biases[range] }
            eval(Array(views.values))
            // Module.update mutates existing MLXArray contexts and preserves parameter keys,
            // shapes, module identity and the host diagnostic table lookup.
            try! shard.update(parameters: .unflattened(views), verify: .all)
        }
        let witness = ["shards": q.count, "logical_bytes": logicalBytes,
                       "rebound_shard_bytes": logicalBytes, "active_before_bytes": activeBefore,
                       "active_after_merge_bytes": activeAfterMerge,
                       "active_after_rebind_bytes": Memory.activeMemory,
                       "process_peak_bytes": Memory.peakMemory]
        return (weight, scales, biases, first.groupSize, first.bits, witness)
    }

    /// gids: flat host ids -> (N, dim) float32 rows in the same order
    func callAsFunction(_ gids: [Int64]) -> MLXArray {
        let shardOf = gids.map { Int($0) / rows }
        let rowOf = gids.map { Int32(Int($0) % rows) }
        var order: [Int] = Array(0 ..< gids.count)
        order.sort { shardOf[$0] < shardOf[$1] }
        var pieces: [MLXArray] = []
        var i = 0
        while i < order.count {
            let s = shardOf[order[i]]
            var j = i
            var rowsSel: [Int32] = []
            while j < order.count && shardOf[order[j]] == s { rowsSel.append(rowOf[order[j]]); j += 1 }
            pieces.append(shards[s](MLXArray(rowsSel)).asType(.float32))
            i = j
        }
        let sorted = concatenated(pieces, axis: 0)  // rows in `order`
        var inv = [Int32](repeating: 0, count: order.count)
        for (k, o) in order.enumerated() { inv[o] = Int32(k) }
        return take(sorted, MLXArray(inv), axis: 0)
    }
}

/// n-gram row ids on device: one thread per (batch, position); `hist` = [prev ctx | ids] int32,
/// int64 wrapping multiply / xor / floor-mod exactly as the host version (Qwen4ExpNgramHash.rowIds).
private let qwen4ExpNgramRowsKernel = MLXFast.metalKernel(
    name: "qwen4_exp_ngram_rows",
    inputNames: ["hist", "mults", "vocab", "offs"],
    outputNames: ["rows"],
    source: """
        uint idx = thread_position_in_grid.x;            // b * T + t
        uint b = idx / T;
        uint t = idx % T;
        uint base = b * (CTX + T);
        uint pos = t + CTX;                              // position in the history row
        long seg = CTX + 1;                              // >= any shift we use
        for (uint back = 1; back <= CTX; ++back) {
            if (hist[base + pos - back] == EOS) { seg = (long)back - 1; break; }
        }
        ulong mixed = (ulong)(long)hist[base + pos] * (ulong)mults[0];
        uint p = 1;
        for (uint n = 2; n <= NGRAM; ++n) {
            for (; p < n; ++p) {
                long shifted = (seg >= (long)p) ? (long)hist[base + pos - p] : (long)EOS;
                mixed ^= (ulong)shifted * (ulong)mults[p];
            }
            uint h0 = (n - 2) * HPN;
            for (uint hh = h0; hh < h0 + HPN; ++hh) {
                long m = (long)mixed;
                long v = vocab[hh];
                long r = m % v; if (r < 0) r += v;
                rows[idx * NHEADS + hh] = (int)(r + offs[hh]);
            }
        }
        """
)

final class Qwen4ExpNGramEmbedding: Module {
    /// Built from the config (seed) at init; REPLACED after load by the checkpoint's own
    /// `layer_multipliers` / `ngram_heads_vocab_sizes` / `ngram_heads_offsets` buffers
    /// (the shipped multipliers do NOT match a seed-0 rebuild -- see ENGINE KB).
    private(set) var hash: Qwen4ExpNgramHash
    let headDim: Int
    @ModuleInfo(key: "ngram_embedding") var table: Qwen4ExpShardedEmbedding
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ParameterInfo(key: "ngram_heads_vocab_sizes") var headVocabSizes: MLXArray
    @ParameterInfo(key: "ngram_heads_offsets") var headOffsets: MLXArray
    private var bound = false

    init(_ a: Qwen4ExpTextConfiguration, embedDim: Int, pleLayerIndex: Int) {
        hash = Qwen4ExpNgramHash(a, pleLayerIndex: pleLayerIndex)
        headDim = embedDim / hash.nHeads
        _table.wrappedValue = Qwen4ExpShardedEmbedding(nShards: hash.nShards, rows: hash.rowsPerShard, dim: headDim)
        _layerMultipliers.wrappedValue = MLXArray(hash.multipliers)
        _headVocabSizes.wrappedValue = MLXArray(hash.vocab)
        _headOffsets.wrappedValue = MLXArray(hash.offsets)
        super.init()
    }

    private func bindCheckpointBuffers() {
        if bound { return }
        bound = true
        let m = layerMultipliers.asType(.int64).asArray(Int64.self)
        let v = headVocabSizes.asType(.int64).asArray(Int64.self)
        let o = headOffsets.asType(.int64).asArray(Int64.self)
        FileHandle.standardError.write("qwen4_exp: n-gram buffers loaded dtype=\(layerMultipliers.dtype) multipliers=\(m) (config seed rebuild \(hash.multipliers))\n".data(using: .utf8)!)
        if m.count == hash.multipliers.count, v.count == hash.vocab.count, o.count == hash.offsets.count,
           m != hash.multipliers || v != hash.vocab || o != hash.offsets {
            FileHandle.standardError.write("qwen4_exp: WARNING checkpoint n-gram buffers differ from the config rebuild; using the config rebuild (int64 buffers may have been cast by the loader)\n".data(using: .utf8)!)
        }
    }

    // merged quantised table (all shards concatenated) + device-side hash buffers
    private var mergedW: MLXArray? = nil, mergedS: MLXArray? = nil, mergedB: MLXArray? = nil
    private var multsDev: MLXArray? = nil, vocabDev: MLXArray? = nil, offsDev: MLXArray? = nil
    private var groupSize = 32, bits = 4
    private var storageWitness: [String: Int] = [:]

    /// Model-level parameter/module updates recurse through this override (Module.update's
    /// child dispatch is virtual). Derived tables must be rebuilt after a normal checkpoint
    /// reload. Direct low-level mutation of a shard is not a model reload API.
    private func invalidateDerivedStorage() {
        mergedW = nil; mergedS = nil; mergedB = nil
        multsDev = nil; vocabDev = nil; offsDev = nil
        storageWitness = [:]; bound = false
    }

    @discardableResult
    override func update(parameters: ModuleParameters, verify: VerifyUpdate,
                         path: [String] = [], modulePath: [String] = []) throws -> Self {
        invalidateDerivedStorage()
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    @discardableResult
    override func update(modules: ModuleChildren, verify: VerifyUpdate,
                         path: [String] = [], modulePath: [String] = []) throws -> Self {
        invalidateDerivedStorage()
        return try super.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    func prepareStorage() -> [String: Int] {
        bindCheckpointBuffers()
        bindMerged()
        return storageWitness
    }

    private func bindMerged() {
        if mergedW != nil { return }
        let storage = table.mergeQuantizedStorage()
        mergedW = storage.weight; mergedS = storage.scales; mergedB = storage.biases
        groupSize = storage.groupSize; bits = storage.bits; storageWitness = storage.witness
        multsDev = MLXArray(hash.multipliers); vocabDev = MLXArray(hash.vocab); offsDev = MLXArray(hash.offsets)
        eval(multsDev!, vocabDev!, offsDev!)
        FileHandle.standardError.write(("qwen4_exp: n-gram storage shared "
            + storageWitness.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            + "\n").data(using: .utf8)!)
    }

    /// Row ids (B,T,nHeads) int32 on device from ids (B,T) int32 and prev (B,ctx) int32.
    func rowIdsDevice(ids: MLXArray, prev: MLXArray) -> MLXArray {
        let B = ids.dim(0), T = ids.dim(1), ctx = hash.ngramSize - 1
        let hist = concatenated([prev.asType(.int32), ids.asType(.int32)], axis: 1)
        return qwen4ExpNgramRowsKernel(
            [hist, multsDev!, vocabDev!, offsDev!],
            template: [("T", T), ("CTX", ctx), ("NGRAM", hash.ngramSize), ("HPN", hash.headsPerNgram),
                       ("NHEADS", hash.nHeads), ("EOS", hash.eos)],
            grid: (B * T, 1, 1), threadGroup: (min(B * T, 256), 1, 1),
            outputShapes: [[B, T, hash.nHeads]], outputDTypes: [.int32])[0]
    }

    /// ids (B,T) int32, prev (B,ctx) int32 -> (B,T,embedDim) float32 (device path, no host sync)
    func callAsFunction(ids: MLXArray, prev: MLXArray) -> MLXArray {
        bindCheckpointBuffers()
        bindMerged()
        let rows = rowIdsDevice(ids: ids, prev: prev)              // (B,T,nHeads)
        if let d = Qwen4ExpEnv.dumpActs, ids.dim(1) > 1 {
            let r = rows.asType(.int32).asArray(Int32.self)
            try? Data(bytes: r, count: r.count * 4).write(to: URL(fileURLWithPath: d).appendingPathComponent("ngram_rows.i32"))
        }
        let flat = rows.flattened()
        let out = dequantized(mergedW![flat], scales: mergedS![flat], biases: mergedB == nil ? nil : mergedB![flat],
                              groupSize: groupSize, bits: bits)
        return out.asType(.float32).reshaped(ids.dim(0), ids.dim(1), hash.nHeads * headDim)
    }

    /// host path kept for cross-checks (engine ngram-ids)
    func callAsFunction(ids: [[Int]], prev: [[Int]]) -> MLXArray {
        bindCheckpointBuffers()
        let B = ids.count, T = ids[0].count
        var all: [Int64] = []
        all.reserveCapacity(B * T * hash.nHeads)
        for b in 0 ..< B { all += hash.rowIds(prev: prev[b], ids: ids[b]) }
        return table(all).reshaped(B, T, hash.nHeads * headDim)
    }
}

final class Qwen4ExpPLELayer: Module {
    let d: Int
    let hc: Int
    let dilation: Int
    let stateLen: Int
    @ModuleInfo(key: "ple_embedding") var embedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ a: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        d = a.hiddenSize
        hc = a.hcCount
        let hcDim = d * hc
        dilation = a.ngramSize
        stateLen = (a.pleConvKernelSize - 1) * dilation
        _embedding.wrappedValue = Qwen4ExpNGramEmbedding(a, embedDim: a.pleEmbedDim, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(a.pleEmbedDim, hcDim, bias: false)
        _valueProj.wrappedValue = Linear(a.pleEmbedDim, d, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: a.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: a.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcDim, groupSize: d, eps: a.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(inputChannels: hcDim, outputChannels: hcDim, kernelSize: a.pleConvKernelSize,
                                      stride: 1, padding: 0, dilation: dilation, groups: hcDim, bias: false)
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let S = x.dim(1)
        let state = cache?[2] ?? MLXArray.zeros([x.dim(0), stateLen, x.dim(-1)], dtype: x.dtype)
        let full = concatenated([state, x], axis: 1)
        if let cache {
            cache[2] = full[0..., (full.dim(1) - stateLen)..., 0...]
            cache[4] = S > 1 ? full : nil          // rollback buffer for a verify block
        }
        return silu(conv1d(full[0..., (full.dim(1) - (stateLen + S))..., 0...]))
    }

    private var compiledGate: (@Sendable ([MLXArray]) -> [MLXArray])? = nil
    private func gateMath(_ emb: MLXArray, _ hidden: MLXArray) -> [MLXArray] {
        let lead = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(emb)).reshaped(lead + [hc, d])
        let value = valueProj(emb)
        let query = normQuery(hidden).reshaped(lead + [hc, d])
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(d))
        gate = sqrt(maximum(abs(gate), MLXArray(Float(1e-6)).asType(gate.dtype))) * sign(gate)   // 0-d MLXArray promotes; keep dtype
        var gated = sigmoid(gate) * expandedDimensions(value, axis: -2)
        gated = gated.reshaped(lead + [hc * d])
        return [gated, normConv(gated)]
    }

    func callAsFunction(_ hidden: MLXArray, ids: MLXArray, prev: MLXArray, cache: ArraysCache?) -> MLXArray {
        let emb = embedding(ids: ids, prev: prev).asType(hidden.dtype)
        let r: [MLXArray]
        if Qwen4ExpGatedResidual.useCompile {
            if compiledGate == nil {
                compiledGate = compile(inputs: [self], shapeless: false) { [unowned self] a in self.gateMath(a[0], a[1]) }
            }
            r = compiledGate!([emb, hidden])
        } else {
            r = gateMath(emb, hidden)
        }
        return r[0] + shortConv(r[1], cache: cache)
    }
}

// MARK: - Decoder layer

final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHC: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHC: Qwen4ExpGatedResidual

    init(_ a: Qwen4ExpTextConfiguration, layerIdx: Int) {
        isLinear = a.layerTypes[layerIdx] == "linear_attention"
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(a)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(a)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(a)
        if let pleIdx = a.pleLayerIds.firstIndex(of: layerIdx + 1) {
            _ple.wrappedValue = Qwen4ExpPLELayer(a, pleLayerIndex: pleIdx)
        }
        _attnHC.wrappedValue = Qwen4ExpGatedResidual(a)
        _mlpHC.wrappedValue = Qwen4ExpGatedResidual(a)
        super.init()
    }

    private static let combineCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) { a in
        [Qwen4ExpDecoderLayer.combineRaw(a[0], a[1], a[2])]
    }
    private static func combineRaw(_ hyper: MLXArray, _ x: MLXArray, _ inject: MLXArray) -> MLXArray {
        // hyper + (x[...,None,:] * inject[...,None]).reshape(..., hc*d)
        let lead = Array(x.shape.dropLast())
        let prod = expandedDimensions(x, axis: -2) * expandedDimensions(inject, axis: -1)
        return hyper + prod.reshaped(lead + [-1])
    }
    /// combine; when `next` (the HC that consumes the result) runs the fused chain, also emit its grouped norm
    private func combineChained(_ hyper: MLXArray, _ x: MLXArray, _ inject: MLXArray, next: Qwen4ExpGatedResidual?) -> (MLXArray, MLXArray?) {
        // P020 unit 7 (prefill): the combine wrote the 84 MB residual stream and the next hyper-connection's grouped
        // norm read it straight back. One kernel does both -- bit-identical, one fewer full pass over the stream.
        if q4HCChainPrefill, Q4Fused.hcMix, Q4Fused.hcInject, let next, inject.ndim == 2 || inject.ndim == 3,
           inject.dtype != .float32 || inject.ndim == 2,
           hyper.size / (attnHC.hc * attnHC.d) > 16,          // prefill chunks only; the decode program is untouched
           !next.fusedChainActive(rows: hyper.size / (attnHC.hc * attnHC.d)), attnHC.d % 256 == 0 {
            let (o, n) = q4HCInjectNorm2(hyper: hyper, x: x, inj: inject, nextOnePlusW: next.hcNorm.onePlusWeight(hyper.dtype),
                                         eps: next.hcNorm.eps, hc: attnHC.hc, d: attnHC.d)
            return (o, n)
        }
        if Q4Fused.hcChainNorm, let next, inject.ndim == 3, inject.dtype == .float32, inject.dim(1) == attnHC.hc * attnHC.d / 256,
           next.fusedChainActive(rows: hyper.size / (attnHC.hc * attnHC.d)) {
            let (o, n) = q4HCInjectNorm(hyper: hyper, x: x, injp: inject, nextOnePlusW: next.hcNorm.onePlusWeight(hyper.dtype), eps: next.hcNorm.eps, hc: attnHC.hc, d: attnHC.d)
            return (o, n)
        }
        return (combine(hyper, x, inject), nil)
    }
    private func combine(_ hyper: MLXArray, _ x: MLXArray, _ inject: MLXArray) -> MLXArray {
        if inject.ndim == 3 && inject.dtype == .float32 && inject.dim(1) == attnHC.hc * attnHC.d / 256 {   // fused-HC partials
            return q4HCInjectP(hyper: hyper, x: x, injp: inject, hc: attnHC.hc, d: attnHC.d)
        }
        if Q4Fused.hcMix && Q4Fused.hcInject { return q4HCInject(hyper: hyper, x: x, inj: inject, hc: attnHC.hc, d: attnHC.d) }
        return Qwen4ExpGatedResidual.useCompile ? Self.combineCompiled([hyper, x, inject])[0] : Self.combineRaw(hyper, x, inject)
    }

    /// in-situ ablation (timing only, output is garbage): ENGINE_ABL_SKIP=mixer,mlp,hc,ple  ENGINE_ABL_LAYERS=n
    static let ablSkip: Set<String> = Set((ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "").split(separator: ",").map(String.init))
    static let ablLayers: Int = Int(ProcessInfo.processInfo.environment["ENGINE_ABL_LAYERS"] ?? "") ?? Int.max
    /// P029 unit 1, the mixer sub-census. `mixer` collapses TWO different programs: the GDN recurrence on
    /// the 36 linear layers and the QSA full attention on the 12 attention layers (full_attention_interval
    /// 4). No active claim splits them at 262144, and they have opposite context behaviour -- GDN is
    /// context-FLAT, the attention carries the whole penalty. ENGINE_ABL_SKIP=gdn skips only the linear
    /// layers, =attn only the attention layers; both leave x unchanged (both blocks are d -> d) so every
    /// downstream shape and kernel is identical. `mixer` remains as the CONSISTENCY CONTROL: if the
    /// instrument is linear, (base-gdn) + (base-attn) == (base-mixer). TIMING ONLY.

    func callAsFunction(_ h: MLXArray, rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?, ids: MLXArray, prev: MLXArray, dump: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        callChained(h, rope: rope, mask: mask, cache: cache, ids: ids, prev: prev, normedIn: nil, next: nil, dump: dump).0
    }

    /// `normedIn`: this layer's attnHC grouped norm of `h`, if the previous combine produced it; `next`: the HC that
    /// consumes this layer's output (next layer's attnHC or the final mixer) -> its norm is emitted by the last combine
    func callChained(_ h: MLXArray, rope: Qwen4ExpRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                     cache: KVCache?, ids: MLXArray, prev: MLXArray, normedIn: MLXArray?, next: Qwen4ExpGatedResidual?,
                     dump: ((String, MLXArray) -> Void)? = nil,
                     pos3: MLXArray? = nil, mropeSection: [Int] = [11, 11, 10]) -> (MLXArray, MLXArray?) {
        var h = h
        var normedIn = normedIn
        let abl = Self.ablSkip
        if abl.contains("hc") {
            // no hyper-connections: plain residual on the first d lanes of the stream
            let d = attnHC.d
            var x = h[.ellipsis, 0 ..< d]
            if !abl.contains("mixer") {
                if let linearAttn { x = linearAttn(x, cache: cache as? ArraysCache) }
                else { let list = cache as? CacheList; x = selfAttn!(x, rope: rope, mask: mask, cache: list?[0], indexerCache: list?[1] as? ArraysCache, pos3: pos3, mropeSection: mropeSection) }
            }
            if !abl.contains("mlp") { x = mlp(x) }
            return (h + tiled(x, repetitions: [1, 1, attnHC.hc]), nil)
        }
        if let ple, !Qwen4ExpEnv.noPLE, !abl.contains("ple") {
            let p = ple(h, ids: ids, prev: prev, cache: cache as? ArraysCache)
            Q4Prof.mark("ple", [p])
            dump?("ple", p)
            h = h + p
            normedIn = nil                                   // the residual changed after the previous combine
        }
        var (x, inj) = attnHC(h, normed: dump == nil ? normedIn : nil)
        Q4Prof.mark("hc_attn", [x, inj!])
        dump?("attn_in", x)
        if dump != nil { FileHandle.standardError.write("dtype: h=\(h.dtype) attn_in=\(x.dtype) inj=\(inj!.dtype)\n".data(using: .utf8)!) }
        if abl.contains("mixer") {
        } else if let linearAttn {
            if !abl.contains("gdn") { x = linearAttn(x, cache: cache as? ArraysCache) }
        } else if !abl.contains("attn") {
            let list = cache as? CacheList
            x = selfAttn!(x, rope: rope, mask: mask, cache: list?[0], indexerCache: list?[1] as? ArraysCache, pos3: pos3, mropeSection: mropeSection)
            // P070: the QSA block ids this call attended through (gather path only; the mask path leaves lastTop nil)
            if dump != nil, let t = selfAttn!.indexer.lastTop { dump?("top", t.0.asType(.float32)) }
        }
        dump?("attn_out", x)   // P070: the mixer output (GDN or QSA attention) before the hyper-connection combine
        if dump != nil { FileHandle.standardError.write("dtype: block_out=\(x.dtype)\n".data(using: .utf8)!) }
        let (h1, n1) = combineChained(h, x, inj!, next: dump == nil ? mlpHC : nil)
        h = h1
        dump?("attn_res", h)   // P071: residual stream after the attention inject
        let (y, inj2) = mlpHC(h, normed: n1)
        Q4Prof.mark("hc_mlp", [h, y, inj2!])
        dump?("mlp_in", y)     // P071: the MoE input (mlp hyper-connection mix)
        if dump != nil { dump?("router", mlp.routerLogits(y)) }   // P071: router logits (S, E), dump mode only
        if Q4Fused.moeFold, dump == nil, !abl.contains("mlp"), !Q4Fused.hcChainNorm, inj2!.ndim == 3, inj2!.dtype == .float32,
           let (yk, wf, sh, sg) = mlp.forwardPieces(y) {
            // MoE combine folded into the hyper-connection inject (one dispatch less on the critical path)
            let r = q4HCInjectMoE(hyper: h, yk: yk, w: wf, shared: sh.reshaped(yk.dim(0), yk.dim(2)), gate: sg.reshaped(yk.dim(0)), injp: inj2!, hc: attnHC.hc, d: attnHC.d)
            Q4Prof.mark("moe+hc", [r])
            return (r, nil)
        }
        let m = abl.contains("mlp") ? y : mlp(y)
        Q4Prof.mark("moe", [m])
        dump?("mlp_out", m)    // P071: the MoE output (experts + shared expert)
        if dump != nil { FileHandle.standardError.write("dtype: after_attn_h=\(h.dtype) mlp_in=\(y.dtype) mlp_out=\(m.dtype)\n".data(using: .utf8)!) }
        let (r, n2) = combineChained(h, m, inj2!, next: dump == nil ? next : nil)
        if Q4Prof.active { var a = [r]; if let n2 { a.append(n2) }; Q4Prof.mark("hc_combine", a) }
        dump?("layer", r)
        return (r, n2)
    }
}

// MARK: - Model

public final class Qwen4ExpModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual
    let rope: Qwen4ExpRotary
    let hc: Int
    let ngramContext: Int
    let mropeSection: [Int]
    let eos: Int
    let faIdx: Int
    let pleIdx: Int?

    init(_ a: Qwen4ExpTextConfiguration) {
        mropeSection = a.mropeSection
        _embedTokens.wrappedValue = Embedding(embeddingCount: a.vocabularySize, dimensions: a.hiddenSize)
        _layers.wrappedValue = (0 ..< a.hiddenLayers).map { Qwen4ExpDecoderLayer(a, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(a, useCombine: false)
        rope = Qwen4ExpRotary(dim: Int(Float(a.headDim) * a.partialRotaryFactor), base: a.ropeTheta * Qwen4ExpVariants.ropeThetaMult)   // P072 arm
        hc = a.hcCount
        ngramContext = a.ngramSize - 1
        eos = a.eosTokenId
        faIdx = a.layerTypes.firstIndex(of: "full_attention") ?? 0
        pleIdx = (0 ..< a.hiddenLayers).first { a.pleLayerIds.contains($0 + 1) }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, embedsOverride: nil, pos3: nil)
    }

    /// P037 unit 3. `embedsOverride` replaces the token embedding for the whole block (the caller has
    /// already spliced the image features onto the `<|image_pad|>` rows); `pos3` is (3, B, S) mRoPE
    /// positions. BOTH nil is the shipped text path, byte for byte -- the overload above is the only
    /// thing the text path calls.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, embedsOverride: MLXArray?,
                        pos3: MLXArray?) -> MLXArray {
        Q4Prof.begin(rows: inputs.dim(1), batch: inputs.dim(0))
        var h = embedsOverride ?? embedTokens(inputs)
        Q4Prof.mark("embed", [h])
        let caches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let attnCache = (caches[faIdx] as? CacheList)?[0]
        let mask = createAttentionMask(h: h, cache: attnCache.map { [$0] })

        // n-gram context: the previous (ngramSize-1) tokens live in the PLE layer's cache slot 3
        // as an int32 array; everything stays on device (no host sync inside the step)
        let idsDev = inputs.asType(.int32)
        var prev = MLXArray.full([inputs.dim(0), ngramContext], values: MLXArray(Int32(eos)), dtype: .int32)
        if let pleIdx, let pc = caches[pleIdx] as? ArraysCache {
            if let stored = pc[3] { prev = stored }
            let hist = concatenated([prev, idsDev], axis: 1)
            pc[3] = hist[0..., (hist.dim(1) - ngramContext)...]
            pc[5] = inputs.dim(1) > 1 ? hist : nil   // rollback buffer for a verify block
        }
        // P070: ENGINE_DUMP_CALLS=1 suffixes every name with the forward-call index (`_c<n>`), so a
        // chunked prefill and the decode steps that follow stop overwriting each other, and S == 1
        // calls are dumped too. ENGINE_DUMP_LAYERS=3,7 restricts the per-layer names to those layers.
        let dumpCalls = Qwen4ExpEnv.dumpCalls
        let dumpDir = (inputs.dim(1) > 1 || dumpCalls) ? Qwen4ExpEnv.dumpActs : nil  // prefill only unless ENGINE_DUMP_CALLS
        let callIdx = Self.dumpCallCounter
        if dumpDir != nil { Self.dumpCallCounter += 1 }
        let dumpLayers: Set<Int>? = Qwen4ExpEnv.dumpLayers
        func dump(_ name: String, _ a: MLXArray) {
            guard let dumpDir else { return }
            let f = a[0].asType(.float32).asArray(Float.self)
            let fname = dumpCalls ? "\(name)_c\(callIdx).f32" : name + ".f32"
            try? Data(bytes: f, count: f.count * 4).write(to: URL(fileURLWithPath: dumpDir).appendingPathComponent(fname))
        }
        dump("embed", h)
        // P040: the layer-0 hyper-connection internals are OPT-IN (ENGINE_DUMP_HC0=1), because setting
        // `debugSink` makes `Qwen4ExpGatedResidual` take a different body -- it skips the fused hcMix
        // and the compiled path -- so a dump that sets it is no longer measuring the champion. The
        // control that found this: with the sink set, the dumped run's logits sit maxdiff 23.5 from
        // the champion's; with it off they must be 0.0, and the gate below is that number.
        if dumpDir != nil, Qwen4ExpEnv.dumpHC0 {
            layers[0].attnHC.debugSink = { name, a in dump(name + "_0", a) }
        }
        h = tiled(h, repetitions: [1, 1, hc])
        var normedNext: MLXArray? = nil
        // MTP verify blocks (2..16 rows): dispatch the graph in slices so the GPU starts the early layers while the host is still
        // building the rest (the GPU ledger showed ~2.7 ms idle per round = verify build minus draft GPU time, OBS-ENG-025)
        let every = Self.dispatchEvery
        // P095 arm (ENGINE_DISPATCH_EVERY_BATCHED=1): a batched decode step is B rows at S = 1 and
        // builds a graph B times the size of a serial one; slice its dispatch like a verify block's.
        let sliced = every > 0 && ((inputs.dim(1) >= 2 && inputs.dim(1) <= 16)
                                   || (Self.dispatchEveryBatched && inputs.dim(1) == 1 && inputs.dim(0) >= 2))
        for (i, layer) in layers.enumerated() {
            if i >= Qwen4ExpDecoderLayer.ablLayers { break }
            if sliced && i > 0 && i % every == 0 { asyncEval(h) }
            if dumpDir != nil, let ple = layer.ple {
                dump("ple_emb_\(i)", ple.embedding(ids: idsDev, prev: prev))
            }
            let nextHC: Qwen4ExpGatedResidual? = i + 1 < layers.count ? layers[i + 1].attnHC : mixer
            let (hh, nn) = layer.callChained(h, rope: rope, mask: layer.isLinear ? .none : mask, cache: caches[i], ids: idsDev, prev: prev,
                                             normedIn: normedNext, next: nextHC,
                                             dump: (dumpDir == nil || !(dumpLayers?.contains(i) ?? true)) ? nil : { name, a in dump(name + "_\(i)", a) },
                                             pos3: pos3, mropeSection: mropeSection)
            h = hh; normedNext = nn
        }
        layers[0].attnHC.debugSink = nil
        if Q4Prof.active { let r = mixer(h, normed: normedNext).0; Q4Prof.mark("final_mixer", [r]); Q4Prof.end(); return r }
        lastHidden = h
        return mixer(h, normed: normedNext).0
    }

    /// the (B,S,hc*d) residual stream of the last forward, before the final mixer (MTP input)
    private(set) var lastHidden: MLXArray? = nil
    /// P070: forward-call index for ENGINE_DUMP_CALLS
    nonisolated(unsafe) static var dumpCallCounter: Int = 0
    /// ENGINE_KV_STEP: KV cache growth step in tokens (mlx-swift-lm default 256)
    static let kvStep: Int = Int(ProcessInfo.processInfo.environment["ENGINE_KV_STEP"] ?? "256") ?? 256
    /// ENGINE_DISPATCH_EVERY=n: asyncEval the residual stream every n layers on 2..16-row blocks (0 = off)
    static let dispatchEvery: Int = Int(ProcessInfo.processInfo.environment["ENGINE_DISPATCH_EVERY"] ?? "8") ?? 8
    static let dispatchEveryBatched: Bool = (ProcessInfo.processInfo.environment["ENGINE_DISPATCH_EVERY_BATCHED"] ?? "0") != "0"
}

// MARK: - MTP head (one full-attention decoder layer over fused (embedding, trunk stream))

public final class Qwen4ExpMTP: Module {
    let d: Int
    let hc: Int
    @ModuleInfo(key: "pre_fc_norm_embedding") var preNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preNormHidden: Qwen4ExpRMSNorm
    /// One rms statistic per hyper-connection stream, as every other 10240-wide norm in this architecture takes.
    /// P020 unit 3: the flat reading was a porting GUESS (HF drops every mtp.* tensor, so nothing constrains it) and it
    /// is WRONG -- grouped raises MTP acceptance at every depth and 256k decode by +3.75% (K=2) / +4.58% (K=3) with the
    /// committed sequence identical. ENGINE_MTP_HNORM_GROUP=0 restores the flat reading.
    static let hiddenNormGrouped: Bool = ProcessInfo.processInfo.environment["ENGINE_MTP_HNORM_GROUP"] != "0"
    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual
    let rope: Qwen4ExpRotary

    init(_ a: Qwen4ExpTextConfiguration) {
        d = a.hiddenSize; hc = a.hcCount
        _preNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(dimensions: d, eps: a.rmsNormEps)
        // P020 D-D: HF carries no MTP reference, so the grouping of this 10240-wide norm is UNKNOWN by reading.
        // Every OTHER 10240-wide norm in this architecture (hc_norm, PLE norm_key/query/conv) takes ONE statistic per
        // stream of hidden_size; this one was ported flat. ENGINE_MTP_HNORM_GROUP=1 selects the grouped reading.
        // Decided by MTP acceptance, the only instrument that can see it.
        _preNormHidden.wrappedValue = Qwen4ExpRMSNorm(dimensions: hc * d,
                                                      groupSize: Qwen4ExpMTP.hiddenNormGrouped ? d : nil,
                                                      eps: a.rmsNormEps)
        _fcEmbedding.wrappedValue = Linear(d, d, bias: false)
        _fcHidden.wrappedValue = Linear(d, d, bias: false)
        var la = a
        la.hiddenLayers = 1; la.layerTypes = ["full_attention"]; la.pleLayerIds = []
        // P020 unit 4: the DRAFT head's QSA budget is a free policy knob -- the trunk verifies every draft, so the
        // committed sequence cannot change whatever the draft attends to. ENGINE_MTP_IDX_BUDGET raises it (a value
        // above the context length turns the indexer off entirely: `kvLen <= budget` short-circuits to dense causal).
        // P031: PROMOTED. The draft head's QSA budget is 512, not the trunk's 2048. Measured at
        // 262144 on the champion, four interleaved passes: MTP K=2 55.09 -> 56.40 tok/s (+2.37%) and
        // K=1 54.47 -> 55.45 (+1.80%), with ACCEPTANCE UP (0.49 -> 0.51 and 0.61 -> 0.63) and the
        // ROUND slightly cheaper. It CANNOT cost correctness -- the trunk verifies every draft -- and
        // that was CHECKED, not assumed: the committed token sequence is IDENTICAL at draft budgets
        // 2048 / 1024 / 512 / 256 / 128 at both depths. The declared gates cannot move because the
        // head's indexer returns nil below its budget at 512 and 8192, and they were re-read to prove
        // it: 512 serial -0.18%, 512 K=3 -0.48%, 8k K=3 +0.52%, all inside the paired resolution.
        // 1024 reads the same +2.35%; 256 is WORSE (-3.29%) and 128 is +1.98%, so this is a shallow
        // optimum and not a monotone trend -- do not extrapolate it. ENGINE_MTP_IDX_BUDGET overrides.
        la.indexerBudget = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_IDX_BUDGET"] ?? "") ?? 512
        _layers.wrappedValue = [Qwen4ExpDecoderLayer(la, layerIdx: 0)]
        // P032: the reuse period is a property of THIS indexer, the draft head's, and of no other.
        if let n = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_IDX_REUSE"] ?? ""), n > 1 {
            _layers.wrappedValue[0].selfAttn?.indexer.reusePeriod = n
        }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(a, useCombine: false)
        rope = Qwen4ExpRotary(dim: Int(Float(a.headDim) * a.partialRotaryFactor), base: a.ropeTheta * Qwen4ExpVariants.ropeThetaMult)   // P072 arm
        super.init()
    }

    public func newCache() -> KVCache { let kv = Qwen4ExpCacheCapacity.makeKVCache(); return CacheList(kv, ArraysCache(size: 3)) }   // indexer: [0] raw keys, [1] pooled blocks, [2] P039 full 3-axis positions (visual only)

    /// hidden (B,S,hc*d) trunk stream at positions p, tokens (B,S) = the tokens at p+1
    /// -> (mixed (B,S,d) for lm_head, stream (B,S,hc*d) for the next depth)
    public func callAsFunction(hidden: MLXArray, tokens: MLXArray, embed: Embedding, cache: KVCache?) -> (MLXArray, MLXArray) {
        let lead = Array(hidden.shape.dropLast())
        let e = fcEmbedding(preNormEmbedding(embed(tokens)))                       // (B,S,d)
        let hs = fcHidden(preNormHidden(hidden).reshaped(lead + [hc, d]))          // (B,S,hc,d)
        var h = (expandedDimensions(e, axis: -2) + hs).reshaped(lead + [hc * d])
        let attn = (cache as? CacheList)?[0]
        let mask = createAttentionMask(h: h, cache: attn.map { [$0] })
        let dummyIds = tokens.asType(.int32)
        let dummyPrev = MLXArray.zeros([hidden.dim(0), 2], dtype: .int32)
        h = layers[0](h, rope: rope, mask: mask, cache: cache, ids: dummyIds, prev: dummyPrev)
        return (mixer(h).0, h)
    }
}

public final class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: Qwen4ExpConfiguration
    @ModuleInfo(key: "model") public var model: Qwen4ExpModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "mtp") public var mtp: Qwen4ExpMTP?
    /// P037: the checkpoint's own vision tower, mounted when the config declares one. `Qwen3VLVision`
    /// is the vendored Qwen3-VL implementation and it matches this checkpoint's `vision_tower.*` tree
    /// name for name and dimension for dimension (OBS-ENG-098). Nothing on the TEXT path reads it.
    @ModuleInfo(key: "vision_tower") public var visionTower: Qwen3VLVision.VisionModel?
    /// Set at init from the config, and `sanitize` MUST agree with it: `update(parameters:verify:[.all])`
    /// fails if the tower is built and gets no weights, or if vision weights arrive with no tower.
    public let hasVision: Bool
    public static let noVision: Bool = ProcessInfo.processInfo.environment["ENGINE_NO_VISION"] != nil
    public static let wantMTP: Bool = ProcessInfo.processInfo.environment["ENGINE_MTP"] != nil
    /// true when the indexer's pooled block cache (slot 1) is a capacity buffer -- callers must NOT slice it (P020 unit 8)
    public static var indexerPooledBuffer: Bool { Qwen4ExpQSAIndexer.pooledBuffer }

    /// Materialize fixed serving storage on the model owner before admission is sized, even
    /// when the optional warm forward is disabled. No model forward or token arithmetic runs.
    /// Returned measurements are plain values; rebound bytes describe parameter ownership,
    /// while active/peak bytes are the MLX allocator's process-wide observations.
    @discardableResult
    public func prepareServingStorage() -> [String: Int] {
        let before = Memory.activeMemory
        var result = ["ngram_layers": 0, "ngram_logical_bytes": 0, "ngram_rebound_shard_bytes": 0]
        for layer in model.layers {
            guard let embedding = layer.ple?.embedding else { continue }
            let witness = embedding.prepareStorage()
            result["ngram_layers", default: 0] += 1
            result["ngram_logical_bytes", default: 0] += witness["logical_bytes", default: 0]
            result["ngram_rebound_shard_bytes", default: 0] += witness["rebound_shard_bytes", default: 0]
            result["ngram_max_active_after_merge_bytes"] = max(result["ngram_max_active_after_merge_bytes", default: 0],
                                                               witness["active_after_merge_bytes", default: 0])
        }
        result["active_before_bytes"] = before
        result["active_after_bytes"] = Memory.activeMemory
        result["process_peak_bytes"] = Memory.peakMemory
        return result
    }

    /// ACTIVATION WITNESS for the quantisation-gated fast paths (BOUND-ENG-007). The fused
    /// hyper-connection chain is worth 14.0% of serial decode and the draft-vocabulary trim needs a
    /// quantised head (OBS-ENG-061), and BOTH are gated on a cast that fails SILENTLY: a checkpoint
    /// whose config disagrees with its packing simply runs slower with nothing on stderr. Called ONCE
    /// after load, from the CLI -- deliberately not from `fusedChainActive`, because putting a mutable
    /// static on that hot path measured a repeatable -0.24% (6/6 paired, runs/p021_witness_ab.txt).
    public func fusionWitness() -> String {
        func q(_ m: Linear?) -> String {
            guard let m else { return "absent" }
            guard let qq = m as? QuantizedLinear else { return "bf16 (NOT QuantizedLinear)" }
            return "\(qq.bits)b g\(qq.groupSize)"
        }
        let l0 = model.layers.first
        let mixUp = l0?.attnHC.mixUp
        let body = l0.map { $0.attnHC.fusedChainActive(rows: 1) ? ((mixUp is QuantizedLinear) ? "ACTIVE (8-bit body)" : "ACTIVE (dense bf16 body)") : "OFF" } ?? "OFF"
        let moe = model.layers.compactMap { $0.mlp as? Qwen4ExpSparseMoeBlock }.first
        let expert = moe.map { $0.expertPathWitness() } ?? "no MoE block"
        return "qwen4_exp: fusion witness -- hc mixUp \(q(mixUp)) -> fused chain "
            + "\(body); lm_head \(q(lmHead)) -> draft-vocabulary trim "
            + "\(lmHead is QuantizedLinear ? "AVAILABLE" : "OFF"); Q4Fused.hcFused=\(Q4Fused.hcFused)"
            + "; fused experts \(expert)"
            // P030 DEFECT, found before the knob shipped: the P024 state cache is keyed by
            // sha256(model identity | M | token prefix) and this witness IS that identity -- but it did
            // not carry `indexer_budget`. Two programs with different budgets produce DIFFERENT KV and
            // pooled state from the same tokens, so a warm run would have silently served the wrong
            // program's state. Naming it here makes a mismatched rung a MISS instead of a lie.
            + "; qsa budget \(configuration.text.indexerBudget)"
            + "; prefill row projections \(prefillRowProjectionMode)"
            + (Self.mropeSelfTest ? "; " + mropeSelfTestLine() : "")
    }

    /// Read only on the model owner thread. Counts prove the arm actually dispatched.
    public func prefillProjectionWitness() -> [String: Int] { prefillRowProjectionCalls }

    /// P037 unit 2 GATE, and it is a MEASUREMENT not an argument: fed three IDENTICAL axes the mRoPE
    /// path must reproduce the shipped 1-D rotary EXACTLY. If it does not, the text path is at risk the
    /// moment anything routes through it. ENGINE_MROPE_SELFTEST=1.
    static let mropeSelfTest: Bool = ProcessInfo.processInfo.environment["ENGINE_MROPE_SELFTEST"] != nil
    func mropeSelfTestLine() -> String {
        let r = model.rope
        let sec = configuration.text.mropeSection
        let T = 777
        let p1 = Qwen4ExpRotary.positions(offset: 0, count: T)          // (1, T)
        let (c0, s0) = r.tables(p1)
        let p3 = tiled(p1[.newAxis, 0..., 0...], repetitions: [3, 1, 1])  // (3, 1, T), all axes equal
        let (c1, s1) = r.tablesMRope(p3, section: sec)
        eval(c0, s0, c1, s1)
        let dc = (abs(c0 - c1)).max().item(Float.self)
        let ds = (abs(s0 - s1)).max().item(Float.self)
        return "mrope selftest section \(sec): identical-axes max|dcos| \(dc) max|dsin| \(ds) -> " + (dc == 0 && ds == 0 ? "EXACT" : "MISMATCH")
    }

    public init(_ c: Qwen4ExpConfiguration) {
        configuration = c
        let a = c.text
        vocabularySize = a.vocabularySize
        kvHeads = a.layerTypes.map { $0 == "full_attention" ? a.kvHeads : 0 }
        _model.wrappedValue = Qwen4ExpModelInner(a)
        hasVision = (c.vision != nil) && !Self.noVision
        if let v = c.vision, !Self.noVision { _visionTower.wrappedValue = Qwen3VLVision.VisionModel(v) }
        if Self.wantMTP { _mtp.wrappedValue = Qwen4ExpMTP(a) }
        if !a.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(a.hiddenSize, a.vocabularySize, bias: false)
        }
        super.init()
    }

    /// Blocks longer than 16 rows (prefill chunks) get logits for the LAST row only: lm_head over a 4096-row chunk is a
    /// 4096 x 248320 bf16 temporary (2 GB) and 230 ms (OBS-ENG-028). Verify blocks (<= 16 rows) keep every row.
    /// ENGINE_FULL_LOGITS=1 restores full logits (the `logits` parity tool sets it).
    static let lastLogitsOnly: Bool = ProcessInfo.processInfo.environment["ENGINE_FULL_LOGITS"] == nil
    /// P095: ENGINE_ABL_SKIP=head replaces the 0.675 GB `lm_head` read with a broadcast of one hidden
    /// column to the vocabulary shape, so every consumer (argmax, sampler) still runs over [B, S, V]
    /// and the difference is the head's own cost. TIMING ONLY; the tokens are garbage.
    static let ablHead: Bool = (ProcessInfo.processInfo.environment["ENGINE_ABL_SKIP"] ?? "")
        .split(separator: ",").map(String.init).contains("head")
    /// P095 U3-B: ENGINE_HEAD_QMM_MIN=m (0 = off) -- at m <= rows < 12 the head's 0.675 GB is read ONCE
    /// through the tiled qmm instead of once per row through qmv (see q8ProjPadded).
    static let headQmmMin: Int = Int(ProcessInfo.processInfo.environment["ENGINE_HEAD_QMM_MIN"] ?? "0") ?? 0
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if Self.lastLogitsOnly && out.dim(1) > 16 { out = out[0..., (out.dim(1) - 1)..., 0...] }
        if Self.ablHead, let lmHead { return broadcast(out[.ellipsis, 0 ..< 1], to: [out.dim(0), out.dim(1), lmHead.weight.dim(0)]) }
        if let lmHead { return q8ProjPadded(lmHead, out, minRows: Self.headQmmMin) }
        return model.embedTokens.asLinear(out)
    }

    /// P037 unit 3: the multimodal entry point. `embedsOverride` is the token embedding with the image
    /// features already spliced onto the `<|image_pad|>` rows; `pos3` is (3, B, S) mRoPE positions.
    /// The text path never calls this -- it calls the two-argument overload, unchanged.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, embedsOverride: MLXArray?,
                               pos3: MLXArray?) -> MLXArray {
        var out = model(inputs, cache: cache, embedsOverride: embedsOverride, pos3: pos3)
        if Self.lastLogitsOnly && out.dim(1) > 16 { out = out[0..., (out.dim(1) - 1)..., 0...] }
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    /// token ids -> (B, S, hidden), so a caller can splice visual features before the forward
    public func embedTokenIds(_ ids: MLXArray) -> MLXArray { model.embedTokens(ids) }

    /// P037: pixels -> (N, hidden) visual tokens, N = product(grid) / merge^2. Throws if no tower.
    public func visionFeatures(pixels: MLXArray, gridTHW: [THW]) -> MLXArray? {
        guard let visionTower else { return nil }
        return visionTower(pixels, gridTHW: gridTHW).0
    }

    public func logitsFromMixed(_ mixed: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(mixed) }
        return model.embedTokens.asLinear(mixed)
    }

    /// ENGINE_MTP_DRAFT_VOCAB=N: the DRAFT head reads only the first N rows of lm_head plus every id >= vocabSpecialsFrom (the
    /// chat-template specials at the top of the table). Drafts are proposals -- the verify step still scores the full head, so
    /// the output is unchanged; a true next token outside the trimmed set only costs a rejection. lm_head is 397 MB (4-bit,
    /// 248320 rows) and streams at bandwidth (OBS-ENG-026): the draft head is the largest single item of a draft step.
    static let draftVocab: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_DRAFT_VOCAB"] ?? "98304") ?? 98304
    private var draftHead: (w: MLXArray, s: MLXArray, b: MLXArray, ids: MLXArray, bits: Int, gs: Int)? = nil
    private var draftHeadTried = false
    /// int32 [1]: argmax token of the draft head for one mixed row (B=1, S=1)
    public func draftToken(_ mixed: MLXArray) -> MLXArray {
        if !draftHeadTried {
            draftHeadTried = true
            let N = Self.draftVocab
            if N > 0, let q = lmHead as? QuantizedLinear, let b = q.biases, N < q.weight.dim(0) {
                let V = q.weight.dim(0)
                let specialsFrom = configuration.text.vocabSpecialsFrom
                var rows = Array(0 ..< N)
                if specialsFrom > N && specialsFrom < V { rows += Array(specialsFrom ..< V) }
                let idx = MLXArray(rows.map { Int32($0) })
                let w = q.weight[idx], sc = q.scales[idx], bi = b[idx]
                eval(w, sc, bi)
                draftHead = (w, sc, bi, idx, q.bits, q.groupSize)
                FileHandle.standardError.write("mtp draft head: \(rows.count) of \(V) rows (\(N) + \(rows.count - N) specials)\n".data(using: .utf8)!)
            }
        }
        guard let d = draftHead else {
            let logits = logitsFromMixed(mixed)
            draftProbSink?(softmax(logits.asType(.float32), axis: -1).max(axis: -1))
            return logits.argMax(axis: -1).asType(.int32)
        }
        let logits = quantizedMatmul(mixed, d.w, scales: d.s, biases: d.b, transpose: true, groupSize: d.gs, bits: d.bits)
        draftProbSink?(softmax(logits.asType(.float32), axis: -1).max(axis: -1))
        return d.ids[logits.argMax(axis: -1)]
    }

    /// P106 H53 (diagnostic, ENGINE_ROUND_TRACE_PROBS): receives each draft's head probability (max softmax over the
    /// draft vocabulary), lazily, next to the unchanged argmax. nil (the default) adds nothing to the graph.
    public var draftProbSink: ((MLXArray) -> Void)?

    /// trunk forward returning (logits, pre-mixer stream) -- the MTP input
    public func forwardHidden(_ inputs: MLXArray, cache: [KVCache]?) -> (MLXArray, MLXArray) {
        let logits = self(inputs, cache: cache)
        return (logits, model.lastHidden!)
    }

    /// After a multi-token block of `blockRows` tokens was consumed, rewind every cache so that only
    /// the first `n` rows of that block remain committed. No projections are recomputed.
    public func rollback(_ caches: [KVCache], blockRows: Int, keep n: Int) {
        let drop = blockRows - n
        guard drop > 0 else { return }
        for (i, c) in caches.enumerated() {
            let layer = model.layers[i]
            if let l = c as? CacheList {
                let kv = l[0] as! KVCacheSimple
                _ = kv.trim(drop)
                let idx = l[1] as! ArraysCache
                // slot 0 (raw keys) is a capacity buffer whose logical length follows the KV offset (P018); only the pooled blocks
                // built from the dropped rows must go
                let newLen = kv.offset
                if !Qwen4ExpQSAIndexer.pooledBuffer, let p = idx[1], let r = layer.selfAttn?.indexer.ratio, p.dim(1) > newLen / r { idx[1] = p[0..., 0 ..< (newLen / r), 0...] }
            } else if let a = c as? ArraysCache, let gdn = layer.linearAttn {
                gdn.replayPrefix(cache: a, rows: n)
                // ArraysCache.offset is not advanced by this fork; leave it
                if layer.ple != nil {
                    if let full = a[4] {              // [state(9) | block rows]
                        a[2] = full[0..., n ..< (n + layer.ple!.stateLen), 0...]
                    }
                    if let hist = a[5] {              // [ctx(2) | block rows]
                        a[3] = hist[0..., n ..< (n + model.ngramContext)]
                    }
                }
            }
        }
    }

    // MARK: cache snapshot / restore (speculative decoding rollback)
    /// P099: rollback after a BATCHED verify block of `blockRows` rows per row: row b keeps its first keep[b] rows.
    /// Attention rows are not trimmed (the pool's offset is a high-water mark and the per-row lengths are the
    /// caller's: `lengths[b] + keep[b]` after this); the indexer's pooled watermark is rewound per row so blocks
    /// that hold unaccepted keys are pooled again; GDN states replay per row from the tape; the PLE conv state and
    /// the n-gram history are re-sliced per row from the verify-block buffers (slots 4 / 5).
    public func rollbackRows(_ caches: [KVCache], blockRows S: Int, keep: [Int], lengths: [Int]) {
        let B = keep.count
        for (i, c) in caches.enumerated() {
            let layer = model.layers[i]
            if c is CacheList, let idx = layer.selfAttn?.indexer {
                let r = idx.ratio
                if idx.raggedPooled.count == B {
                    for b in 0 ..< B { idx.raggedPooled[b] = min(idx.raggedPooled[b], (lengths[b] + keep[b]) / r) }
                }
            } else if let a = c as? ArraysCache, let gdn = layer.linearAttn {
                if keep.contains(where: { $0 < S }) { gdn.replayPrefixRows(cache: a, keep: keep) } else { a.prefixReplayTape = nil }
                if let ple = layer.ple {
                    if let full = a[4] { a[2] = concatenated((0 ..< B).map { b in full[b ..< (b + 1), keep[b] ..< (keep[b] + ple.stateLen), 0...] }, axis: 0) }
                    if let hist = a[5] { a[3] = concatenated((0 ..< B).map { b in hist[b ..< (b + 1), keep[b] ..< (keep[b] + model.ngramContext)] }, axis: 0) }
                }
            }
        }
    }

    /// P099: the MTP head's cache for B rows (a CacheList of KVCacheSimple + the indexer's ArraysCache(3)), padded and
    /// stacked like a trunk attention layer; `unstackHeadCache` takes one row back out at its own length.
    public func stackHeadCaches(_ caches: [KVCache]) -> KVCache {
        let kvs = caches.map { ($0 as! CacheList)[0] as! KVCacheSimple }
        let idxs = caches.map { ($0 as! CacheList)[1] as! ArraysCache }
        let kv = Qwen4ExpCacheCapacity.makeKVCache()
        kv.setBuffers(keys: Self.padCat(kvs.map { $0.rawKeys! }, axis: 2), values: Self.padCat(kvs.map { $0.rawValues! }, axis: 2),
                      offset: kvs.map { $0.offset }.max()!)
        let idx = ArraysCache(size: 3)
        idx[0] = Self.padCat(idxs.compactMap { $0[0] }, axis: 1)
        if idxs.allSatisfy({ $0[1] != nil }) { idx[1] = Self.padCat(idxs.map { $0[1]! }, axis: 1) }
        return CacheList(kv, idx)
    }
    public func unstackHeadCache(_ stacked: KVCache, slot: Int, length: Int, pooledBlocks: Int) -> KVCache {
        let l = stacked as! CacheList
        let kv = l[0] as! KVCacheSimple, idx = l[1] as! ArraysCache
        let out = mtp!.newCache() as! CacheList
        (out[0] as! KVCacheSimple).state = [kv.rawKeys![slot ..< (slot + 1), 0..., 0 ..< length, 0...], kv.rawValues![slot ..< (slot + 1), 0..., 0 ..< length, 0...]]
        let oi = out[1] as! ArraysCache
        if let x = idx[0] { oi[0] = x[slot ..< (slot + 1), 0 ..< min(length, x.dim(1)), 0...] }
        if let x = idx[1], pooledBlocks > 0 { oi[1] = x[slot ..< (slot + 1), 0 ..< min(pooledBlocks, x.dim(1)), 0...] }
        return out
    }
    static func padCat(_ arrs: [MLXArray], axis: Int) -> MLXArray {
        let target = arrs.map { $0.dim(axis) }.max()!
        let padded = arrs.map { a -> MLXArray in
            if a.dim(axis) == target { return a }
            var shape = a.shape; shape[axis] = target - a.dim(axis)
            return concatenated([a, MLXArray.zeros(shape, dtype: a.dtype)], axis: axis)
        }
        return concatenated(padded, axis: 0)
    }
    /// P099: the head's per-row pooled watermark after a padded re-prime (rows past their own length must be pooled again)
    public func rewindHeadRows(lengths: [Int]) {
        guard let idx = mtp?.layers[0].selfAttn?.indexer, idx.raggedPooled.count == lengths.count else { return }
        for b in 0 ..< lengths.count { idx.raggedPooled[b] = min(idx.raggedPooled[b], lengths[b] / idx.ratio) }
    }
    /// P106 H60: the head's per-row pooled watermark, read and restored around a draft chain queued ahead of its round.
    public var headRaggedPooled: [Int] {
        get { mtp?.layers[0].selfAttn?.indexer.raggedPooled ?? [] }
        set { mtp?.layers[0].selfAttn?.indexer.raggedPooled = newValue }
    }
    public func resetHeadRaggedState() {
        mtp?.layers[0].selfAttn?.indexer.raggedPooled = []
        mtp?.layers[0].selfAttn?.indexer.lastTopRagged = nil
        mtp?.layers[0].selfAttn?.indexer.lastTop = nil
    }

    public struct CacheSnapshot { public var arrays: [[MLXArray]]; public var offsets: [Int] }

    public func snapshot(_ caches: [KVCache]) -> CacheSnapshot {
        var arrays: [[MLXArray]] = []; var offsets: [Int] = []
        for c in caches {
            if let l = c as? CacheList {
                let kv = l[0] as! KVCacheSimple
                var ist = (l[1] as! ArraysCache).state
                if !ist.isEmpty, ist[0].dim(1) > kv.offset { ist[0] = ist[0][0..., 0 ..< kv.offset, 0...] }   // logical prefix of the capacity buffer (P018)
                arrays.append(kv.state + ist); offsets.append(kv.offset)
            } else if let a = c as? ArraysCache {
                arrays.append(a.state); offsets.append(a.offset)
            }
        }
        return CacheSnapshot(arrays: arrays, offsets: offsets)
    }

    /// restore: attention KV is trimmed back to the snapshot offset (in-place buffers), the
    /// array caches (GDN conv/ssm, PLE conv/context, indexer keys) are set back to the snapshot
    public func restore(_ caches: [KVCache], _ snap: CacheSnapshot) {
        for (i, c) in caches.enumerated() {
            if let l = c as? CacheList {
                let kv = l[0] as! KVCacheSimple
                let extra = kv.offset - snap.offsets[i]
                if extra > 0 { _ = kv.trim(extra) }
                let idx = l[1] as! ArraysCache
                idx.state = Array(snap.arrays[i].dropFirst(kv.state.count))
            } else if let a = c as? ArraysCache {
                a.state = snap.arrays[i]
                a.offset = snap.offsets[i]
            }
        }
    }

    // MARK: cache export / import (P017: prefill snapshots on disk; every experiment starts from the saved state)
    /// Slot-exact export: attention KV sliced to the offset, indexer slots 0..1, GDN/PLE slots 0..5 (nil slots omitted).
    public func exportCaches(_ caches: [KVCache], prefix: String, into d: inout [String: MLXArray]) {
        for (i, c) in caches.enumerated() {
            if let l = c as? CacheList {
                let kv = l[0] as! KVCacheSimple
                let st = kv.state
                if st.count == 2 { d["\(prefix)L\(i).k"] = st[0]; d["\(prefix)L\(i).v"] = st[1] }
                let idx = l[1] as! ArraysCache
                if let x = idx[0] { d["\(prefix)L\(i).i0"] = x.dim(1) > kv.offset ? x[0..., 0 ..< kv.offset, 0...] : x }   // logical prefix of the capacity buffer (P018)
                if let x = idx[1] { d["\(prefix)L\(i).i1"] = x }
            } else if let a = c as? ArraysCache {
                for s in 0 ..< 6 { if let x = a[s] { d["\(prefix)L\(i).a\(s)"] = x } }
            }
        }
    }
    /// Inverse of exportCaches into FRESH caches (newCache()); KVCacheSimple.state's setter sets offset = keys.dim(2).
    public func importCaches(_ caches: [KVCache], prefix: String, from d: [String: MLXArray]) {
        for (i, c) in caches.enumerated() {
            if let l = c as? CacheList {
                if let k = d["\(prefix)L\(i).k"], let v = d["\(prefix)L\(i).v"] { (l[0] as! KVCacheSimple).state = [k, v] }
                let idx = l[1] as! ArraysCache
                for s in 0 ..< 2 { if let x = d["\(prefix)L\(i).i\(s)"] { idx[s] = x } }
                // P103: slot 2 (the P039 full 3-axis mRoPE positions) is NOT exported, so an import into a cache that
                // already holds tokens would leave a buffer sized for the OLD length under the NEW offset -- and the
                // realloc path copies `0 ..< offset` out of it. Clear it: it is rebuilt from the next forward's positions.
                if d["\(prefix)L\(i).i0"] != nil { idx[2] = nil }
            } else if let a = c as? ArraysCache {
                for s in 0 ..< 6 { if let x = d["\(prefix)L\(i).a\(s)"] { a[s] = x } }
            }
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            if layer.isLinear {
                // 0: conv state, 1: ssm state, 2: PLE conv state, 3: n-gram context,
                // 4: PLE full conv buffer (verify block), 5: n-gram history (verify block)
                return ArraysCache(size: 6)
            }
            // 0: attention KV, 1: indexer raw keys
            let kv = Qwen4ExpCacheCapacity.makeKVCache()
            return CacheList(kv, ArraysCache(size: 3))   // indexer: [0] raw keys, [1] pooled blocks, [2] P039 full 3-axis positions (visual only)
        }
    }

    /// P088 -- the ragged decode path keeps PER-ROW state on each indexer (`raggedPooled`, the
    /// count of blocks already pooled for row b, and `lastTopRagged`). That state belongs to a
    /// particular set of rows in a particular pool. The moment a batch is re-stacked -- a member
    /// joined, left, or the group simply changed -- row b is a DIFFERENT sequence, and carrying the
    /// old watermark means blocks that were never pooled are treated as pooled: the selection then
    /// scores stale rows and the answer degenerates. Must be called on every (re)stack.
    public func resetRaggedState() {
        for layer in model.layers {
            layer.selfAttn?.indexer.raggedPooled = []
            layer.selfAttn?.indexer.lastTopRagged = nil
            layer.selfAttn?.indexer.lastTop = nil
        }
    }

    public func makeCache() -> [KVCache] { newCache(parameters: nil) }

    /// P080 M1 -- stack B single-sequence caches into ONE batched cache whose rows may sit at
    /// DIFFERENT lengths. Every buffer is padded to the longest row and the rows are concatenated on
    /// the batch axis, so row b keeps its own keys, its own pooled blocks and its own recurrent state
    /// exactly where a serial run left them. This is what a ragged decode step then reads per row,
    /// and it is the seed of the slot pool M2 needs.
    /// P088 -- the inverse of `stackCaches`: pull row `slot` back out as a single-sequence cache of
    /// its own `length`. A batch is formed by stacking and dissolved by unstacking, so a request
    /// leaves a group with exactly the cache it would have had alone -- which is what lets it go
    /// back to the solo path (and to MTP) the moment it is the only one left.
    public func unstackRow(_ batched: [KVCache], slot: Int, length: Int, pooledBlocks: Int) -> [KVCache] {
        let out = newCache(parameters: nil)
        for (i, c) in batched.enumerated() {
            if let l = c as? CacheList {
                let kv = l[0] as! KVCacheSimple
                let idx = l[1] as! ArraysCache
                let o = out[i] as! CacheList
                if let k = kv.rawKeys, let v = kv.rawValues {
                    (o[0] as! KVCacheSimple).state = [k[slot ..< (slot + 1), 0..., 0 ..< length, 0...],
                                                      v[slot ..< (slot + 1), 0..., 0 ..< length, 0...]]
                }
                let oi = o[1] as! ArraysCache
                if let x = idx[0] { oi[0] = x[slot ..< (slot + 1), 0 ..< min(length, x.dim(1)), 0...] }
                if let x = idx[1], pooledBlocks > 0 { oi[1] = x[slot ..< (slot + 1), 0 ..< min(pooledBlocks, x.dim(1)), 0...] }
            } else if let a = c as? ArraysCache {
                let o = out[i] as! ArraysCache
                for sIdx in 0 ..< 4 { if let x = a[sIdx] { o[sIdx] = x[slot ..< (slot + 1)] } }
            }
        }
        return out
    }

    /// Metadata-only private trunk while a distinct standing pool owns its history.
    /// Owner-thread use only, AFTER stack/spec graph construction. No eval is needed here:
    /// lazy pool outputs own the input arrays until their evaluation completes. These private
    /// placeholders must be replaced by unstackRow before any non-pooled forward or export.
    public static func privateTrunkHistoryPlaceholder(_ caches: [KVCache]) -> [KVCache] {
        caches.map { cache in
            guard let list = cache as? CacheList else { return cache } // fixed GDN/PLE state unchanged
            let oldKV = list[0] as! KVCacheSimple
            let oldIndexer = list[1] as! ArraysCache
            precondition(oldIndexer[2] == nil, "pooled private history release is text-only")
            let kv = KVCacheSimple()
            kv.offset = oldKV.offset
            kv.step = oldKV.step
            kv.capacityGrowthLimit = oldKV.capacityGrowthLimit
            let indexer = ArraysCache(size: 3)
            indexer.offset = oldIndexer.offset
            let result = CacheList(kv, indexer)
            result.offset = list.offset
            return result
        }
    }

    public func stackCaches(_ per: [[KVCache]]) -> [KVCache] {
        precondition(!per.isEmpty, "stackCaches: nothing to stack")
        let L = per[0].count
        func padCat(_ arrs: [MLXArray], axis: Int) -> MLXArray {
            let target = arrs.map { $0.dim(axis) }.max()!
            let padded = arrs.map { a -> MLXArray in
                if a.dim(axis) == target { return a }
                var shape = a.shape; shape[axis] = target - a.dim(axis)
                return concatenated([a, MLXArray.zeros(shape, dtype: a.dtype)], axis: axis)
            }
            return concatenated(padded, axis: 0)
        }
        return (0 ..< L).map { i -> KVCache in
            if let first = per[0][i] as? CacheList {
                let kvs = per.map { ($0[i] as! CacheList)[0] as! KVCacheSimple }
                let idxs = per.map { ($0[i] as! CacheList)[1] as! ArraysCache }
                let kv = Qwen4ExpCacheCapacity.makeKVCache()
                kv.setBuffers(keys: padCat(kvs.map { $0.rawKeys! }, axis: 2),
                              values: padCat(kvs.map { $0.rawValues! }, axis: 2),
                              offset: kvs.map { $0.offset }.max()!)
                let idx = ArraysCache(size: 3)
                idx[0] = padCat(idxs.compactMap { $0[0] }, axis: 1)
                if idxs.allSatisfy({ $0[1] != nil }) { idx[1] = padCat(idxs.map { $0[1]! }, axis: 1) }
                precondition(idxs.allSatisfy { $0[2] == nil }, "P080 M1 is text-only (no mRoPE position history)")
                _ = first
                return CacheList(kv, idx)
            }
            let acs = per.map { $0[i] as! ArraysCache }
            let out = ArraysCache(size: 6)
            // Slots 0-3 are the running state a decode step needs (conv, ssm, PLE conv, n-gram
            // context) and every one of them is fixed-size, so they stack by concatenation. Slots 4
            // and 5 are the MTP verify block's transient buffers -- their length follows the last
            // prefill chunk, so they differ per row, and a ragged DECODE step never reads them. They
            // are deliberately dropped rather than padded: padding a conv history with zeros would
            // invent tokens.
            for slot in 0 ..< 4 where acs[0][slot] != nil {
                precondition(acs.allSatisfy { $0[slot] != nil }, "stackCaches: slot \(slot) present in some rows only")
                out[slot] = concatenated(acs.map { $0[slot]! }, axis: 0)
            }
            return out
        }
    }

    // MARK: P106 H48 -- row-resident batch pool

    /// Whether the ragged path may run row-resident under the current environment. The masked and
    /// ablation selection arms score a padded stacked view of every row, the SDPA arm reads the
    /// stacked K/V, and the replay dump saves the stacked storage; none of them is the serving
    /// program, so they keep the stacked pool.
    /// ENGINE_IDX_POOL_BUF=0 (the P020 concatenation arm) makes the single-sequence indexer read slot 1's
    /// LENGTH as its pooled count, which only the stacked unstack's exact-size slice satisfies, so it keeps the
    /// stacked pool too.
    public static var rowResidentEligible: Bool {
        Qwen4ExpQSAIndexer.pooledBuffer
            && !Qwen4ExpQSAIndexer.raggedSelectMasked && !Qwen4ExpQSAIndexer.ablTopK && !Qwen4ExpQSAIndexer.ablTopKSpread
            && !Qwen4ExpQSAIndexer.ablIdxScore && Qwen4ExpQSAIndexer.replayDumpDir == nil && !Qwen4ExpAttention.raggedSDPA
    }

    /// `stackCaches` without copying any row's growing history. Attention layers get a marker
    /// (registered against the members' own CacheLists, slot order); fixed-size state is stacked
    /// exactly as `stackCaches` stacks it. The rows' CacheLists stay the rows' own objects and are
    /// written in place by the ragged step; `unstackRowResident` hands them back.
    public func stackCachesRowResident(_ per: [[KVCache]]) -> [KVCache] { Self.stackRowResident(per) }
    /// Model-free core of `stackCachesRowResident` (the tests call it without weights).
    public static func stackRowResident(_ per: [[KVCache]]) -> [KVCache] {
        precondition(!per.isEmpty, "stackCachesRowResident: nothing to stack")
        let L = per[0].count
        return (0 ..< L).map { i -> KVCache in
            if per[0][i] is CacheList {
                let lists = per.map { $0[i] as! CacheList }
                precondition(lists.allSatisfy { ($0[1] as! ArraysCache)[2] == nil }, "P080 M1 is text-only (no mRoPE position history)")
                return rowResidentMarker(lists)
            }
            return stackFixedState(per.map { $0[i] as! ArraysCache })
        }
    }
    /// A marker CacheList for `lists` (one per row, slot order), registered for the ragged branch.
    static func rowResidentMarker(_ lists: [CacheList]) -> CacheList {
        let marker = Qwen4ExpCacheCapacity.makeKVCache()
        marker.offset = lists.map { ($0[0] as! KVCacheSimple).offset }.max()!
        Qwen4ExpBatch.rowResidentLists[ObjectIdentifier(marker)] = (marker, lists)
        return CacheList(marker, ArraysCache(size: 3))
    }
    /// The fixed-size slots 0-3 concatenated on the batch axis (`stackCaches`' ArraysCache branch).
    static func stackFixedState(_ acs: [ArraysCache]) -> ArraysCache {
        let out = ArraysCache(size: 6)
        for slot in 0 ..< 4 where acs[0][slot] != nil {
            precondition(acs.allSatisfy { $0[slot] != nil }, "stackCaches: slot \(slot) present in some rows only")
            out[slot] = concatenated(acs.map { $0[slot]! }, axis: 0)
        }
        return out
    }
    /// The rows behind a row-resident pool layer's marker (nil for a stacked layer).
    public static func rowResidentRows(_ cache: KVCache) -> [CacheList]? {
        guard let list = cache as? CacheList, let kv = list[0] as? KVCacheSimple else { return nil }
        return Qwen4ExpBatch.rowResident(kv)
    }
    /// Unregister every marker of a dissolving row-resident pool (trunk layers and/or the head cache).
    public static func releaseRowResident(_ caches: [KVCache]) {
        for c in caches {
            guard let list = c as? CacheList, let kv = list[0] as? KVCacheSimple else { continue }
            Qwen4ExpBatch.rowResidentLists[ObjectIdentifier(kv)] = nil
        }
    }
    /// Inverse of `stackCachesRowResident` for slot `slot`: the row's own attention CacheLists (from the
    /// pool's registry, the authority), their KV offset set to the committed `length` -- the pool's
    /// high-water mark may include rejected verify rows -- and fresh fixed state sliced from the stack
    /// exactly as `unstackRow` slices it. Returns the caches and whether they are the objects in `own`.
    public func unstackRowResident(_ batched: [KVCache], slot: Int, length: Int, own: [KVCache]) -> ([KVCache], Bool) {
        Self.unstackRowResidentCore(batched, slot: slot, length: length, own: own)
    }
    public static func unstackRowResidentCore(_ batched: [KVCache], slot: Int, length: Int, own: [KVCache]) -> ([KVCache], Bool) {
        var aliased = own.count == batched.count
        let out = batched.enumerated().map { (i, c) -> KVCache in
            if let rows = rowResidentRows(c) {
                let list = rows[slot]
                (list[0] as! KVCacheSimple).offset = length
                if aliased, !((own[i] as AnyObject) === list) { aliased = false }
                return list
            }
            let a = c as! ArraysCache
            let o = ArraysCache(size: 6)
            for sIdx in 0 ..< 4 { if let x = a[sIdx] { o[sIdx] = x[slot ..< (slot + 1)] } }
            return o
        }
        return (out, aliased)
    }
    /// P106 B41: a read-only view of one row-resident pool member at its committed length, for capturing a hot
    /// rung WITHOUT dissolving the pool: the attention entries are the member's own CacheLists (untouched -- no
    /// offset is written, unlike unstackRowResidentCore), the fixed state is slot `slot` of the stacked arrays.
    public static func rowResidentView(_ batched: [KVCache], slot: Int) -> [KVCache] {
        batched.map { c -> KVCache in
            if let rows = rowResidentRows(c) { return rows[slot] }
            let a = c as! ArraysCache
            let o = ArraysCache(size: 6)
            for sIdx in 0 ..< 4 { if let x = a[sIdx] { o[sIdx] = x[slot ..< (slot + 1)] } }
            return o
        }
    }
    /// The MTP head's pool, row-resident (one attention layer).
    public func stackHeadCachesRowResident(_ caches: [KVCache]) -> KVCache {
        let lists = caches.map { $0 as! CacheList }
        precondition(lists.allSatisfy { ($0[1] as! ArraysCache)[2] == nil }, "P099 head pool is text-only")
        return Self.rowResidentMarker(lists)
    }
    public func unstackHeadCacheRowResident(_ stacked: KVCache, slot: Int, length: Int) -> KVCache {
        let list = Self.rowResidentRows(stacked)![slot]
        (list[0] as! KVCacheSimple).offset = length
        return list
    }
    /// An owned copy of a head/trunk attention CacheList (capacity kept, every slot copied bit for
    /// bit). A row-resident pool writes its members' caches IN PLACE; a head cache that is also the
    /// request's retired handover owner (read at teardown when no live spec remains) must not see
    /// those writes, exactly as it never sees the stacked pool's. Growing the capacity by one step
    /// keeps the slice update a real copy (a full-range update would alias its source).
    public static func ownedAttentionCopy(_ list: CacheList) -> CacheList {
        let kv = list[0] as! KVCacheSimple, idx = list[1] as! ArraysCache
        func copied(_ x: MLXArray, axis: Int) -> MLXArray {
            var shape = x.shape; shape[axis] += 1
            let z = MLXArray.zeros(shape, dtype: x.dtype)
            if axis == 2 { z[0..., 0..., 0 ..< x.dim(2), 0...] = x } else { z[0..., 0 ..< x.dim(1), 0...] = x }
            return z
        }
        let nkv = KVCacheSimple()
        nkv.step = kv.step; nkv.capacityGrowthLimit = kv.capacityGrowthLimit
        if let k = kv.rawKeys, let v = kv.rawValues {
            nkv.setBuffers(keys: copied(k, axis: 2), values: copied(v, axis: 2), offset: kv.offset)
        } else { nkv.offset = kv.offset }
        let nidx = ArraysCache(size: 3)
        for s in 0 ..< 2 { if let x = idx[s] { nidx[s] = copied(x, axis: 1) } }
        precondition(idx[2] == nil, "row-resident copies are text-only")
        nidx.offset = idx.offset
        let out = CacheList(nkv, nidx)
        out.offset = list.offset
        // no eval: the copy is scheduled with the pool's first step; its source is retired (never written again)
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        // precision overlays: `<module>_pN.{weight,scales,biases}` replace `<module>.*`.
        // _p6 was the MIX-2 6-bit projection overlay; the tag is now generic so a variant can be
        // measured without rewriting 107 GB of shards -- the loader enumerates every .safetensors in
        // the directory, so a variant is symlinks + one overlay file + a config with the new
        // (bits, group_size) for exactly those modules. The config MUST agree: `quantize(model:)`
        // sizes the QuantizedLinear from it, and a disagreement fails update(parameters:verify:).
        let overlayTag: String? = ["_p4.", "_p6.", "_p8."].first { tag in weights.keys.contains { $0.contains(tag) } }
        let overlaid = overlayTag.map { tag in
            Set(weights.keys.filter { $0.contains(tag) }.map { String($0[..<$0.range(of: tag)!.lowerBound]) })
        } ?? []
        if let tag = overlayTag, !overlaid.isEmpty { FileHandle.standardError.write("qwen4_exp: precision overlay \(tag.dropLast()): \(overlaid.count) modules replaced\n".data(using: .utf8)!) }
        for (k0, v0) in weights {
            if let tag = overlayTag, let r = k0.range(of: tag) {
                let stem = String(k0[..<r.lowerBound]); let suffix = String(k0[r.upperBound...])
                var k = stem + "." + suffix
                if k.hasPrefix("model.language_model.") { k = "model." + k.dropFirst("model.language_model.".count) }
                else if k.hasPrefix("language_model.") { k = String(k.dropFirst("language_model.".count)) }
                if k.hasPrefix("model.mtp.") { k = String(k.dropFirst("model.".count)) }
                // the MTP block is only BUILT when ENGINE_MTP is set, and this branch used to run
                // ahead of the wantMTP filter below: an overlay that covers mtp modules then handed
                // the model parameters for a submodule that does not exist and every SERIAL run died
                // with incompatibleItems(path: ["mtp"]). Same filter, same place as the generic path.
                if k.hasPrefix("mtp."), !Self.wantMTP { continue }
                // P036 DEFECT FIX: this branch used to `continue` BEFORE the generic path's
                // `ngram_embedding.shard_i` -> `ngram_embedding.shards.i` rename, so an overlay over a
                // SHARDED module handed the model a key no submodule answers to and the load died with
                // keyNotFound(["...","ngram_embedding","shards","0","weight"]). E8h's own overlay covers
                // only experts and lm_head, which need no rename, so the defect was invisible until an
                // overlay touched the n-gram table. Same rename, same place as the generic path.
                if let r2 = k.range(of: ".ngram_embedding.shard_") {
                    k = k[..<r2.lowerBound] + ".ngram_embedding.shards." + k[r2.upperBound...]
                }
                out[k] = v0; continue
            }
            if let dot = k0.lastIndex(of: "."), overlaid.contains(String(k0[..<dot])) { continue }   // shadowed by the overlay
            var k = k0
            var v = v0
            if k.hasPrefix("mtp.") || k.hasPrefix("model.mtp.") || k.hasPrefix("language_model.mtp.") {
                if !Self.wantMTP { continue }
                if k.hasPrefix("language_model.mtp.") { k = String(k.dropFirst("language_model.".count)) }
                if k.hasPrefix("model.mtp.") { k = String(k.dropFirst("model.".count)) }
                if k.hasSuffix("mlp.experts.gate_up_proj") || k.hasSuffix("mlp.experts.down_proj") || k.hasSuffix("conv1d.weight") {
                    // fall through to the generic handling below with the mtp. prefix kept
                } else { out[k] = v; continue }
            }
            // P037: the tower is mounted iff the config declares one, and this drop must agree with
            // that or `verify: [.all]` fails -- loudly, which is the point.
            if k.contains("visual.") { continue }
            if k.hasPrefix("vision_tower.") {
                if !hasVision { continue }
                out[k] = v0; continue
            }
            if k.hasPrefix("model.language_model.") {
                k = "model." + k.dropFirst("model.language_model.".count)
            } else if k.hasPrefix("language_model.") {
                k = String(k.dropFirst("language_model.".count))
            }
            // `ngram_embedding.shard_i.*` -> `ngram_embedding.shards.i.*`
            if let r = k.range(of: ".ngram_embedding.shard_") {
                let rest = k[r.upperBound...]
                k = k[..<r.lowerBound] + ".ngram_embedding.shards." + rest
            }
            // HF fused experts: (E, 2*inter, hidden) -> gate/up ; (E, hidden, inter) -> down
            if k.hasSuffix("mlp.experts.gate_up_proj") {
                let base = String(k.dropLast("experts.gate_up_proj".count))
                let mid = v.dim(-2) / 2
                out[base + "switch_mlp.gate_proj.weight"] = v[.ellipsis, 0 ..< mid, 0...]
                out[base + "switch_mlp.up_proj.weight"] = v[.ellipsis, mid..., 0...]
                continue
            }
            if k.hasSuffix("mlp.experts.down_proj") {
                out[String(k.dropLast("experts.down_proj".count)) + "switch_mlp.down_proj.weight"] = v
                continue
            }
            // DEBUG toggles for checkpoint-convention experiments
            if ProcessInfo.processInfo.environment["ENGINE_SWAP_GATE_UP"] != nil {
                if k.contains("switch_mlp.gate_proj.") { k = k.replacingOccurrences(of: "switch_mlp.gate_proj.", with: "switch_mlp.up_proj.") }
                else if k.contains("switch_mlp.up_proj.") { k = k.replacingOccurrences(of: "switch_mlp.up_proj.", with: "switch_mlp.gate_proj.") }
            }
            // torch conv (C,1,K) -> mlx (C,K,1)
            if k.hasSuffix("conv1d.weight") && v.ndim == 3 && v.dim(1) == 1 {
                v = v.transposed(0, 2, 1)
            }
            out[k] = v
        }
        Self.mergeProjections(&out)
        return out
    }

    /// Concatenate sibling projections that share an input into one module (one qmv/gemv instead of
    /// 2-4 latency-bound ones). Exact: rows of an affine-quantised matrix are independent, so
    /// concatenating weight/scales/biases on axis 0 is the same function as running the parts.
    /// Merged modules quantize like their first constituent via `quantizationAliases`.
    static func mergeProjections(_ w: inout [String: MLXArray]) {
        let groups: [(merged: String, parts: [String])] = [
            ("self_attn.qkvi_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj"]),
            ("linear_attn.in_proj_qkvz", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]),
            ("linear_attn.in_proj_ba", ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]),
            ("mlp.shared_expert.gate_up_proj", ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"]),
            ("mlp.gate_sg", ["mlp.gate", "mlp.shared_expert_gate"]),
        ]
        var merged = 0
        for (m, parts) in groups {
            let anchors = w.keys.filter { $0.hasSuffix("." + parts[0] + ".weight") }
            for anchor in anchors {
                let prefix = String(anchor.dropLast((parts[0] + ".weight").count))   // "model.layers.3." / "mtp.layers.0."
                // P063: siblings stored unalike (Vontra 4-bit g32 keeps `mlp.gate` bf16 and quantises
                // `shared_expert_gate`) are brought to the unquantised sibling's dtype first: the
                // quantised part is dequantised in place (bits and group inferred from its packing),
                // then the group merges exactly as an all-bf16 group does. E9 stores every sibling
                // alike, so nothing changes there (logits bit-identical, checked).
                let weights = parts.compactMap { w[prefix + $0 + ".weight"] }
                if weights.count == parts.count, Set(weights.map { $0.dtype }).count > 1,
                   let plain = weights.first(where: { $0.dtype != .uint32 }) {
                    for part in parts {
                        let key = prefix + part
                        guard let packed = w[key + ".weight"], packed.dtype == .uint32,
                              let scales = w[key + ".scales"] else { continue }
                        let inDim = plain.dim(-1)
                        let bits = 32 * packed.dim(-1) / inDim
                        let group = inDim / scales.dim(-1)
                        w[key + ".weight"] = dequantized(
                            packed, scales: scales, biases: w[key + ".biases"], groupSize: group, bits: bits
                        ).asType(plain.dtype)
                        w[key + ".scales"] = nil
                        w[key + ".biases"] = nil
                        FileHandle.standardError.write("qwen4_exp: dequantised \(key) (\(bits)-bit g\(group)) to merge with its bf16 sibling\n".data(using: .utf8)!)
                    }
                }
                for suffix in ["weight", "scales", "biases"] {
                    let keys = parts.map { prefix + $0 + "." + suffix }
                    let arrays = keys.compactMap { w[$0] }
                    if arrays.isEmpty { continue }
                    guard arrays.count == parts.count else {
                        fatalError("qwen4_exp merge: \(prefix)\(m).\(suffix): \(arrays.count)/\(parts.count) parts present")
                    }
                    w[prefix + m + "." + suffix] = concatenated(arrays, axis: 0)
                    for k in keys { w[k] = nil }
                }
                quantizationAliases[prefix + m] = prefix + parts[0]
                merged += 1
            }
        }
        if merged > 0 { FileHandle.standardError.write("qwen4_exp: merged \(merged) projection groups\n".data(using: .utf8)!) }
    }

    public var loraLayers: [Module] { model.layers }
}

// MARK: - Prefill profiler (P019, INST-ENG-010): ENGINE_PREFILL_PROFILE=1 -- eval() barriers after every block of a > 16-row chunk,
// wall time between consecutive marks charged to the mark's key. The barriers serialise the graph, so the sum overstates the
// unprofiled chunk by the launch latencies; validate the sum against the unprofiled prefill rate.
public enum Q4Prof {
    public static let enabled = ProcessInfo.processInfo.environment["ENGINE_PREFILL_PROFILE"] != nil
    /// P091: the census used to be prefill-only (`S > 16`), which is why the decode step -- the thing
    /// a serving engine actually spends its life doing, and the one whose balance between attention
    /// and experts nobody here had measured -- was invisible to it. Set to 1 to include decode.
    /// The default keeps every earlier prefill number comparable.
    public static let minRows = Int(ProcessInfo.processInfo.environment["ENGINE_PROFILE_MIN_ROWS"] ?? "") ?? 17
    nonisolated(unsafe) public static var active = false
    nonisolated(unsafe) static var acc: [String: Double] = [:]
    nonisolated(unsafe) static var cnt: [String: Int] = [:]
    nonisolated(unsafe) static var last = 0.0
    nonisolated(unsafe) static var phase = "prefill"
    nonisolated(unsafe) static var forwards: [String: Int] = [:]
    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
    public static func begin(rows S: Int, batch B: Int = 1) {
        // P095: a batched decode step is B rows at S = 1; count its rows so ENGINE_PROFILE_MIN_ROWS=1
        // (P091) and the default alike see it, while the phase stays "decode".
        active = enabled && max(S, B * S) >= minRows
        // Prefill and decode are different machines -- one is compute bound on whole tiles, the
        // other reads the same weights for a single row -- so they get separate books.
        phase = S > 1 ? "prefill" : "decode"
        if active { forwards[phase, default: 0] += 1; last = now() }
    }
    public static func end() { active = false }
    /// H60: the arrays are an autoclosure -- the ~10 marks per layer built an array literal on every forward even with the
    /// profiler off.
    @inline(__always) static func mark(_ key: String, _ arrays: @autoclosure () -> [MLXArray]) {
        guard active else { return }
        eval(arrays())
        let t = now(); let k = phase + "/" + key
        acc[k, default: 0] += t - last; cnt[k, default: 0] += 1; last = t
    }
    public static func reset() { acc = [:]; cnt = [:]; forwards = [:] }
    public static func report() {
        guard enabled else { return }
        if acc.isEmpty { FileHandle.standardError.write("step profile: no marks recorded\n".data(using: .utf8)!); return }
        var lines: [String] = []
        for ph in ["prefill", "decode"] {
            let rows = acc.filter { $0.key.hasPrefix(ph + "/") }
            guard !rows.isEmpty else { continue }
            let total = rows.values.reduce(0, +)
            let n = max(1, forwards[ph] ?? 1)
            let marks = rows.reduce(0) { $0 + cnt[$1.key]! }
            lines.append(String(format: "%@ profile: %d forward(s), total %.3f s, %.3f ms per forward, %d marks (%.1f per forward)",
                                ph, n, total, 1e3 * total / Double(n), marks, Double(marks) / Double(n)))
            for (k, v) in rows.sorted(by: { $0.value > $1.value }) {
                let name = String(k.dropFirst(ph.count + 1))
                lines.append("  " + name.padding(toLength: 16, withPad: " ", startingAt: 0)
                             + String(format: " %8.3f s  %5.1f%%  n=%d  %.4f ms each  %.4f ms/forward",
                                      v, 100 * v / total, cnt[k]!, 1e3 * v / Double(cnt[k]!), 1e3 * v / Double(n)))
            }
        }
        FileHandle.standardError.write((lines.joined(separator: "\n") + "\n").data(using: .utf8)!)
    }
}
