import Foundation

/// A reservation bound, not a claim that MLX's allocator enforces a hard memory limit.
/// Includes private KV, padded pooled KV and a simultaneous restack. Fixed/model/hot/
/// allocator/workspace reserves are subtracted by the server before constructing it.
public final class AdmissionBudget: @unchecked Sendable {
    /// Logical history bytes per token: K + V, raw indexer keys, and one pooled
    /// indexer key per compression group. The caller supplies the cache scalar
    /// width and includes any enabled draft-head attention layers. Fixed states,
    /// allocation capacity is bounded separately; reserve retains the triple-history and 512-token margins.
    /// Convert before multiplying so large geometry cannot overflow an Int.
    public static func perTokenBytes(attentionLayers: Int, kvHeads: Int, headDim: Int,
                                     elementBytes: Int = 2, indexerHeadDim: Int = 0,
                                     indexerCompressRatio: Int = 1) -> Double {
        precondition(attentionLayers >= 0 && kvHeads > 0 && headDim > 0 && elementBytes > 0
                     && indexerHeadDim >= 0 && indexerCompressRatio > 0)
        let scalarBytes = Double(elementBytes)
        let kv = 2 * Double(kvHeads) * Double(headDim) * scalarBytes
        let rawIndexer = Double(indexerHeadDim) * scalarBytes
        let pooledIndexer = rawIndexer / Double(indexerCompressRatio)
        return Double(attentionLayers) * (kv + rawIndexer + pooledIndexer)
    }

    /// Capacity envelope for any history whose largest write ends at `length`.
    /// Scalar KV can append a whole forward block; ragged KV/raw round the total.
    /// Pooled indexer growth is NOT rounded. A soft cap never hides required rows.
    /// Double arithmetic avoids Int overflow before rounding/multiplication.
    public static func historyCapacityLengths(length: Int, kvStep: Int = 256,
        indexerStep: Int = 1024, indexerCompressRatio: Int = 4,
        maxForwardRows: Int? = nil, capacityGrowthLimit: Int? = nil)
        -> (kv: Double, raw: Double, pooled: Double) {
        precondition(length > 0 && kvStep > 0 && indexerStep > 0 && indexerCompressRatio > 0)
        precondition(maxForwardRows == nil || maxForwardRows! > 0)
        precondition(capacityGrowthLimit == nil || capacityGrowthLimit! > 0)
        let n = Double(length), q = Double(kvStep), i = Double(indexerStep)
        let ratio = Double(indexerCompressRatio), old = n - 1
        let block = Double(min(length, maxForwardRows ?? length))
        func rounded(_ x: Double, _ quantum: Double) -> Double { ceil(x / quantum) * quantum }
        func bounded(_ wanted: Double, need: Double, divisor: Double = 1) -> Double {
            capacityGrowthLimit.map { max(need, min(wanted, floor(Double($0) / divisor))) } ?? wanted
        }
        let scalarKV = old + rounded(max(block, floor(old / 8)), q)
        let raggedKV = rounded(n + max(q, floor(old / 8)), q)
        let kv = bounded(max(scalarKV, raggedKV), need: n)
        let raw = bounded(rounded(n + max(i, floor(old / 8)), i), need: n)
        let blocks = floor(n / ratio)
        let pooled = blocks == 0 ? 0 : bounded(blocks + max(floor(i / ratio), floor((blocks - 1) / 8)),
                                               need: blocks, divisor: ratio)
        return (kv, raw, pooled)
    }
    public static func historyCapacityBytes(length: Int, attentionLayers: Int, kvHeads: Int,
        headDim: Int, elementBytes: Int = 2, indexerHeadDim: Int = 0,
        indexerCompressRatio: Int = 1, kvStep: Int = 256, indexerStep: Int = 1024,
        maxForwardRows: Int? = nil, capacityGrowthLimit: Int? = nil) -> Double {
        _ = perTokenBytes(attentionLayers: attentionLayers, kvHeads: kvHeads, headDim: headDim,
                          elementBytes: elementBytes, indexerHeadDim: indexerHeadDim,
                          indexerCompressRatio: indexerCompressRatio)
        let c = historyCapacityLengths(length: length, kvStep: kvStep, indexerStep: indexerStep,
            indexerCompressRatio: indexerCompressRatio, maxForwardRows: maxForwardRows,
            capacityGrowthLimit: capacityGrowthLimit)
        return Double(attentionLayers) * Double(elementBytes)
            * (2 * Double(kvHeads) * Double(headDim) * c.kv + Double(indexerHeadDim) * (c.raw + c.pooled))
    }

    private let lock = NSLock()
    private var lengths: [Int: Int] = [:]
    private var next = 0
    public let maxContext: Int
    public let maxChoices: Int
    public let capacityBytes: Double
    public let bytesPerToken: Double
    public let fixedBytesPerSequence: Double
    private let historyBytes: @Sendable (Int) -> Double
    /// nil retains the legacy fixed active-only budget. In shared mode capacityBytes
    /// includes active reservations and reclaimable hot storage; this is its idle ceiling.
    public let hotCacheCeilingBytes: Double?
    public init(maxContext: Int, maxChoices: Int = 8, capacityBytes: Double,
                bytesPerToken: Double, fixedBytesPerSequence: Double = 384e6,
                historyCapacityBytes: (@Sendable (Int) -> Double)? = nil,
                hotCacheCeilingBytes: Double? = nil) {
        precondition(maxContext > 0 && maxChoices > 0 && capacityBytes > 0 && bytesPerToken > 0)
        precondition(hotCacheCeilingBytes == nil || (hotCacheCeilingBytes!.isFinite && hotCacheCeilingBytes! >= 0))
        self.hotCacheCeilingBytes = hotCacheCeilingBytes
        self.maxContext = maxContext; self.maxChoices = maxChoices
        self.capacityBytes = capacityBytes; self.bytesPerToken = bytesPerToken
        self.fixedBytesPerSequence = fixedBytesPerSequence
        // Generic fallback: no serving cap; allow a whole-context scalar append and
        // cache quanta up to 1024. Qwen supplies exact applied geometry below.
        self.historyBytes = historyCapacityBytes ?? { length in
            Self.historyCapacityLengths(length: length, kvStep: 1024).kv * bytesPerToken
        }
    }
    public func valid(prompt: Int, output: Int, choices: Int, lookahead: Int) -> Bool {
        prompt > 0 && output > 0 && choices > 0 && choices <= maxChoices && lookahead >= 0
            && prompt <= maxContext && lookahead <= maxContext - prompt
            && output <= maxContext - prompt - lookahead
    }
    /// Called on the model owner in shared mode. The prepare callback runs BEFORE
    /// commit while this lock excludes other reserve/release transactions. It receives
    /// the prospective hot target and increase in the unchanged active bound. It must
    /// not call this AdmissionBudget recursively, and must return false if cache
    /// reclamation/physical readback cannot cover the increase. No MLX escapes here.
    public func reserve(length: Int, prepareCache: ((Double, Double) -> Bool)? = nil) -> Int? {
        lock.lock(); defer { lock.unlock() }
        guard length > 0 && length <= maxContext else { return nil }
        let paddedLength = max(length, lengths.values.max() ?? 0)
        let history = historyBytes(paddedLength)
        guard history.isFinite, history > 0 else { return nil }
        let estimate = Double(lengths.count + 1) * (3 * (history + 512 * bytesPerToken) + fixedBytesPerSequence)
        guard estimate <= capacityBytes else { return nil }
        if let ceiling = hotCacheCeilingBytes {
            guard let prepareCache,
                  prepareCache(min(ceiling, max(0, capacityBytes - estimate)),
                               max(0, estimate - activeBytesLocked())) else { return nil }
        }
        next += 1; lengths[next] = length; return next
    }
    public func release(_ id: Int) { lock.lock(); lengths[id] = nil; lock.unlock() }
    public var active: Int { lock.lock(); defer { lock.unlock() }; return lengths.count }
    private func activeBytesLocked() -> Double {
        guard let longest = lengths.values.max() else { return 0 }
        return Double(lengths.count) * (3 * (historyBytes(longest) + 512 * bytesPerToken) + fixedBytesPerSequence)
    }
    /// Plain values only; safe for publication after an owner transaction has returned.
    public func snapshot() -> [String: Double] {
        lock.lock(); defer { lock.unlock() }
        let activeBytes = activeBytesLocked()
        return ["shared": hotCacheCeilingBytes == nil ? 0 : 1,
                "capacity_bytes": capacityBytes, "active_reservation_bytes": activeBytes,
                "active_reservations": Double(lengths.count), "max_length": Double(lengths.values.max() ?? 0),
                "hot_ceiling_bytes": hotCacheCeilingBytes ?? 0,
                "hot_target_bytes": hotCacheCeilingBytes.map { min($0, max(0, capacityBytes - activeBytes)) } ?? 0]
    }
}
