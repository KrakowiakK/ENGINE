import Foundation
import MLX

/// P124: the OpenAI / vLLM sampling parameters beyond temperature, top_p and top_k. Before this the server accepted
/// `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` and `logit_bias` and silently ignored them.
///
/// Semantics (what vLLM and Hugging Face do):
/// - `logit_bias` {token id: bias in -100...100} is added to the logits first;
/// - `repetition_penalty` p (> 0; 1 = off) divides a positive logit by p and multiplies a negative one by p, for every token
///   seen in the prompt or the output so far;
/// - `presence_penalty` / `frequency_penalty` (-2...2; 0 = off) subtract `presence * (count > 0) + frequency * count` for
///   every token the OUTPUT holds so far (OpenAI);
/// - `min_p` (0...1; 0 = off) keeps, after temperature / top_k / top_p, only tokens with probability >= min_p x the most
///   probable one (applied by the sampler; it does nothing to a greedy request).
/// All but min_p change the logits, so they apply to greedy requests too (they can move the argmax).
public struct ServeSamplingParams: Sendable, Equatable {
    public var minP: Float = 0
    public var presencePenalty: Float = 0
    public var frequencyPenalty: Float = 0
    public var repetitionPenalty: Float = 1
    public var logitBias: [Int: Float] = [:]

    public init(minP: Float = 0, presencePenalty: Float = 0, frequencyPenalty: Float = 0, repetitionPenalty: Float = 1,
                logitBias: [Int: Float] = [:]) {
        self.minP = minP; self.presencePenalty = presencePenalty; self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty; self.logitBias = logitBias
    }

    /// Every parameter at its no-op value: the request runs exactly as before P124 (and stays batchable, with MTP).
    public var isIdentity: Bool { minP <= 0 && !adjustsLogits }
    /// Parameters that change the logits (everything except min_p).
    public var adjustsLogits: Bool {
        presencePenalty != 0 || frequencyPenalty != 0 || repetitionPenalty != 1 || logitBias.contains { $0.value != 0 }
    }

    /// Parse and validate from a request body. Returns an error message for an out-of-range or malformed value (-> 400);
    /// `vocabularySize` bounds the logit_bias token ids when known. `repeat_penalty` (llama.cpp / Ollama) is an alias.
    public static func parse(_ obj: [String: Any], vocabularySize: Int? = nil) -> (params: ServeSamplingParams, error: String?) {
        var p = ServeSamplingParams()
        // JSON true/false arrive as a CFBoolean NSNumber, which `as? Double` would read as 1/0 (and `is Bool` matches
        // every NSNumber 0 or 1), so the CF type decides
        func isBool(_ v: Any) -> Bool { (v as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false }
        func number(_ key: String) -> (Float?, String?) {
            guard let raw = obj[key], !(raw is NSNull) else { return (nil, nil) }
            if isBool(raw) { return (nil, "'\(key)' must be a number") }
            guard let d = raw as? Double, d.isFinite else { return (nil, "'\(key)' must be a number") }
            return (Float(d), nil)
        }
        let (minP, e1) = number("min_p")
        let (presence, e2) = number("presence_penalty")
        let (frequency, e3) = number("frequency_penalty")
        var (repetition, e4) = number("repetition_penalty")
        if repetition == nil && e4 == nil { (repetition, e4) = number("repeat_penalty") }
        if let e = e1 ?? e2 ?? e3 ?? e4 { return (p, e) }
        if let v = minP {
            guard (0 ... 1).contains(v) else { return (p, "'min_p' must be in [0, 1]") }
            p.minP = v
        }
        if let v = presence {
            guard (-2 ... 2).contains(v) else { return (p, "'presence_penalty' must be in [-2, 2]") }
            p.presencePenalty = v
        }
        if let v = frequency {
            guard (-2 ... 2).contains(v) else { return (p, "'frequency_penalty' must be in [-2, 2]") }
            p.frequencyPenalty = v
        }
        if let v = repetition {
            guard v > 0, v <= 10 else { return (p, "'repetition_penalty' must be in (0, 10]") }
            p.repetitionPenalty = v
        }
        if let raw = obj["logit_bias"], !(raw is NSNull) {
            guard let dict = raw as? [String: Any] else { return (p, "'logit_bias' must be an object {token_id: bias}") }
            for (k, v) in dict {
                guard let id = Int(k), id >= 0 else { return (p, "'logit_bias' key '\(k)' is not a token id") }
                if let n = vocabularySize, id >= n { return (p, "'logit_bias' token id \(id) is outside the vocabulary (\(n))") }
                guard !isBool(v), let b = v as? Double, b.isFinite, (-100 ... 100).contains(b) else {
                    return (p, "'logit_bias' value for \(k) must be a number in [-100, 100]")
                }
                if b != 0 { p.logitBias[id] = Float(b) }
            }
        }
        return (p, nil)
    }
}

/// The per-request logits processor: the dense bias / penalty vectors on the host, updated per committed token, uploaded
/// once per step (V floats; 1 MB at this model's 248 320 tokens -- small next to a decode step).
public struct ServeLogitProcessor {
    public let params: ServeSamplingParams
    private let promptIds: [Int]
    private var counts: [Int: Int] = [:]
    private var V = 0
    private var bias: MLXArray? = nil          // static: logit_bias
    private var penaltyHost: [Float] = []      // -(presence * (count > 0) + frequency * count), output tokens
    private var seenHost: [Float] = []         // 1 for every prompt or output token (repetition_penalty)
    private var penaltyArr: MLXArray? = nil
    private var seenArr: MLXArray? = nil
    private var dirty = true
    public private(set) var observed = 0

    public init(params: ServeSamplingParams, promptIds: [Int]) {
        self.params = params
        self.promptIds = promptIds
    }

    /// Record one committed output token (its count feeds presence / frequency; it joins the repetition set).
    public mutating func observe(_ token: Int) {
        guard token >= 0 else { return }
        counts[token, default: 0] += 1
        observed += 1
        if V > 0, token < V { update(token) }
    }

    private mutating func update(_ token: Int) {
        let c = Float(counts[token] ?? 0)
        penaltyHost[token] = -(params.presencePenalty * (c > 0 ? 1 : 0) + params.frequencyPenalty * c)
        seenHost[token] = 1
        dirty = true
    }

    private mutating func setUp(_ vocab: Int) {
        V = vocab
        if !params.logitBias.isEmpty {
            var b = [Float](repeating: 0, count: vocab)
            for (id, v) in params.logitBias where id < vocab { b[id] = v }
            bias = MLXArray(b)
        }
        penaltyHost = [Float](repeating: 0, count: vocab)
        seenHost = [Float](repeating: 0, count: vocab)
        for t in promptIds where t >= 0 && t < vocab { seenHost[t] = 1 }
        for t in counts.keys where t < vocab { update(t) }
        dirty = true
    }

    /// The processed logits (float32, same shape) for logits whose last axis is the vocabulary. Every row gets the same
    /// history: the caller applies it to one position at a time (the serial path).
    public mutating func apply(_ logits: MLXArray) -> MLXArray {
        let vocab = logits.dim(-1)
        if vocab != V { setUp(vocab) }
        var l = logits.asType(.float32)
        if let bias { l = l + bias }
        let usesPenalty = params.presencePenalty != 0 || params.frequencyPenalty != 0
        let usesRepetition = params.repetitionPenalty != 1
        if dirty && (usesPenalty || usesRepetition) {
            if usesPenalty { penaltyArr = MLXArray(penaltyHost) }
            if usesRepetition { seenArr = MLXArray(seenHost) }
            dirty = false
        }
        if usesRepetition, let seen = seenArr {
            let r = MLXArray(params.repetitionPenalty)
            let penalised = MLX.which(l .> 0, l / r, l * r)
            l = MLX.which(seen .> 0, penalised, l)
        }
        if usesPenalty, let pen = penaltyArr { l = l + pen }
        return l
    }

    /// min_p on sorted-or-not logits along the last axis (after temperature / top_k / top_p): -inf where the probability is
    /// below min_p x the row's most probable token. Exposed for the sampler.
    public static func applyMinP(_ l: MLXArray, minP: Float) -> MLXArray {
        guard minP > 0 else { return l }
        let p = softmax(l, axis: -1)
        let keep = p .>= (p.max(axis: -1, keepDims: true) * MLXArray(minP))
        return MLX.which(keep, l, MLXArray(-Float.infinity))
    }
}
