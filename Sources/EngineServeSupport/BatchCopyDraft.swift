import Foundation

// T-0248 (D-0084): the host-side pieces of padded copy drafts in E9's batched MTP rounds (design C-0718). NOT wired into
// serving: runBatchedMTPRound keeps its own loop until the copy census reads GO on client traffic (D-0083, T-0218); then
// T-0247 wires these in behind a default-off switch and gates them on a weights-loaded bit-identity test.

/// One row of a batched verify block: how many of its drafts are accepted and which tokens it commits. The same rule as
/// runBatchedMTPRound (BatchedRound.swift, the per-row loop after the verify read), plus a realK cap for padded rows.
/// T-0250 correction: the cap is NOT needed for exactness -- a filler equal to the prediction was verified on exactly the
/// committed prefix, so accepting it commits what greedy decoding would (the oracle test shows both ways lossless). The
/// cap keeps a row's accounting to its real drafts (BatchCopyLookup's 'full', acceptance counters).
/// - preds: the block's K+1 predictions for this row; drafts: its K (padded) drafts; realK: how many drafts are real
/// - thinkCloseCut: the think-close id when the row has an open <think> with a bias (the round's thinkOpen && bias > 0)
public func acceptRow(preds: [Int], drafts: [Int], realK: Int, thinkCloseCut: Int? = nil,
                      stopIds: Set<Int> = []) -> (tokens: [Int], accepted: Int) {
    precondition(preds.count == drafts.count + 1, "a verify row has one prediction more than drafts")
    let k = max(0, min(realK, drafts.count))
    var a = 0
    while a < k && preds[a] == drafts[a] { a += 1 }
    if let close = thinkCloseCut, let ci = (0 ..< a).first(where: { drafts[$0] == close }) { a = ci }
    if let si = (0 ..< a).first(where: { stopIds.contains(drafts[$0]) }) { a = si }
    return (Array(drafts.prefix(a)) + [preds[a]], a)
}

/// One row's copy-mode lookup policy, the one the census prices (copyCensusWalk): a lookup round starts on an n-gram match
/// of at least minMatch tokens (2 right after a fully accepted lookup round), drafts startLen tokens, and the block doubles
/// after every fully accepted lookup round up to maxLen. The index holds the row's prompt + committed output.
public struct BatchCopyLookup {
    public let minMatch: Int, startLen: Int, maxLen: Int
    private var ix: CopyCensusIndex
    /// T-0247: tokens the index holds (prompt + committed output incl. the pending token); the caller re-syncs on mismatch
    public var contextCount: Int { ix.ctx.count }
    private var full = false, L = 0
    private var proposedLen = 0              // the L of the outstanding proposal (the census sets L to it on a hit)

    /// context: the prompt plus the first output token (the prefill's), as the census starts its walk
    public init(context: [Int], minMatch: Int, startLen: Int, maxLen: Int, window: Int = 65536) {
        self.minMatch = minMatch; self.startLen = startLen; self.maxLen = maxLen
        ix = CopyCensusIndex(Array(context.suffix(window)))
    }

    /// the lookup drafts for this row's next round, or nil (the row keeps its MTP drafts)
    public mutating func propose() -> [Int]? {
        let lNext = full ? min(maxLen, max(1, L) * 2) : startLen
        guard let hit = ix.lookup(lNext), hit.match >= (full ? 2 : minMatch) else { proposedLen = 0; return nil }
        proposedLen = lNext
        return Array(hit.cont)
    }

    /// after the round: the tokens the row committed, and whether (and how far) a lookup proposal of it was accepted
    public mutating func commit(tokens: [Int], lookup: [Int]?, accepted: Int) {
        if let lookup {
            L = proposedLen
            full = accepted == lookup.count
        } else {
            full = false; L = 0
        }
        proposedLen = 0
        ix.append(tokens)
    }
}

/// The padded verify block of one batched round with copy drafts: every row gets K drafts; realK says how many are real.
/// For the packed per-row-depth verify (forwardPacked) a row verifies realK + 1 positions: packedDepths(plan).
public struct CopyBlockPlan: Equatable {
    public let K: Int
    public let drafts: [[Int]]
    public let realK: [Int]
    public let lookupRows: [Int]
}

/// C-0684's verify fit at batch B: round ms = 47.39 + 0.278 B + 1.472 per real verify position + padMs per padded one.
/// T-0249: in the packed per-row-depth verify (Qwen4ExpModel.forwardPacked, T-0152) row-wise layers skip pads, which cost
/// ~0.397 ms each (C-0443); in a uniform [B, K+1] block a pad is a full position (padMs 1.472).
public func batchRoundMs(B: Int, realPositions: Int, pads: Int = 0, padMs: Double = 0.397) -> Double {
    47.39 + 0.278 * Double(B) + 1.472 * Double(realPositions) + padMs * Double(pads)
}
/// a uniform round: every row verifies K+1 positions
public func batchRoundMs(B: Int, K: Int) -> Double { batchRoundMs(B: B, realPositions: B * (K + 1)) }

/// Plan one round. mtpDrafts: each row's MTP drafts (all the same length, the round's MTP K); lookups: each row's lookup
/// proposal or nil. A lookup row's drafts are its lookup, truncated to maxK; every row is padded to the block's K by
/// repeating its last draft (the realK cap keeps fillers from being accepted). The round takes lookups only when the
/// expected tokens per ms beat the MTP-only round (C-0693's pricing): a lookup row is expected to commit
/// 1 + lookupAcceptRate * its drafts, an MTP row mtpTokensPerRow. padMs: what a padded position costs in the verify
/// forward that runs the block (0.397 packed per-row-depth, the forward T-0247 should use; 1.472 a uniform block).
/// T-0275: costTable (optional) replaces the linear fit with a measured packed-verify cost curve (rows -> ms) at the
/// table's batch size; the MTP-only round is priced from the same table, so the comparison stays like for like.
public func planCopyBlock(mtpDrafts: [[Int]], lookups: [[Int]?], maxK: Int = 8,
                          lookupAcceptRate: Double = 0.80, mtpTokensPerRow: Double = 2.21, padMs: Double = 0.397,
                          costTable: CopyCostTable? = nil) -> CopyBlockPlan {
    precondition(mtpDrafts.count == lookups.count && !mtpDrafts.isEmpty)
    let B = mtpDrafts.count, mtpK = mtpDrafts[0].count
    precondition(mtpDrafts.allSatisfy { $0.count == mtpK && !$0.isEmpty })
    let mtpOnly = CopyBlockPlan(K: mtpK, drafts: mtpDrafts, realK: Array(repeating: mtpK, count: B), lookupRows: [])
    let cut = lookups.map { $0.map { Array($0.prefix(maxK)) }.flatMap { $0.isEmpty ? nil : $0 } }
    let rows = cut.indices.filter { cut[$0] != nil }
    guard !rows.isEmpty else { return mtpOnly }
    let K = max(mtpK, rows.map { cut[$0]!.count }.max()!)
    let expected = cut.map { c in c.map { 1 + lookupAcceptRate * Double($0.count) } ?? mtpTokensPerRow }.reduce(0, +)
    let real = cut.map { ($0?.count ?? mtpK) + 1 }.reduce(0, +)
    let cost = costTable?.roundMs(B: B, realPositions: real, pads: B * (K + 1) - real, padMs: padMs)
        ?? batchRoundMs(B: B, realPositions: real, pads: B * (K + 1) - real, padMs: padMs)
    // the MTP-only round runs the uniform forward: price it from the table's own uniform measurement when it has one,
    // never from the packed curve (T-0286: mixing the two misprices every B other than the table's)
    let base = costTable?.baselineMs(B: B, mtpK: mtpK) ?? batchRoundMs(B: B, K: mtpK)
    guard expected / cost > Double(B) * mtpTokensPerRow / base else { return mtpOnly }
    let drafts = (0 ..< B).map { b -> [Int] in
        let d = cut[b] ?? mtpDrafts[b]
        return d + Array(repeating: d.last!, count: K - d.count)
    }
    return CopyBlockPlan(K: K, drafts: drafts, realK: (0 ..< B).map { cut[$0]?.count ?? mtpK }, lookupRows: rows)
}

/// T-0250: the per-row depths of a plan in the packed per-row-depth verify (Qwen4ExpPackedRows(depths:positions: K + 1)):
/// each row's pending token plus its real drafts
public func packedDepths(_ plan: CopyBlockPlan) -> [Int] { plan.realK.map { $0 + 1 } }

/// T-0250: split a packed verify output (per-row positions concatenated in row order, Qwen4ExpPackedRows' gather order)
/// back into rows
public func splitPacked<T>(_ flat: [T], depths: [Int]) -> [[T]] {
    precondition(flat.count == depths.reduce(0, +), "packed length \(flat.count) != sum of depths \(depths.reduce(0, +))")
    var out: [[T]] = [], i = 0
    for d in depths { out.append(Array(flat[i ..< i + d])); i += d }
    return out
}

/// T-0247: the plan without the MTP drafts on the host (in the round they are device arrays): the block width K, each
/// row's real draft count and the rows that take their lookup. Same rule and gate as planCopyBlock.
public func planCopyBlock(mtpK: Int, lookups: [[Int]?], maxK: Int = 8, lookupAcceptRate: Double = 0.80,
                          mtpTokensPerRow: Double = 2.21, padMs: Double = 0.397, costTable: CopyCostTable? = nil) -> CopyBlockPlan {
    let dummy = Array(repeating: Array(repeating: 0, count: mtpK), count: lookups.count)
    let p = planCopyBlock(mtpDrafts: dummy, lookups: lookups, maxK: maxK, lookupAcceptRate: lookupAcceptRate,
                          mtpTokensPerRow: mtpTokensPerRow, padMs: padMs, costTable: costTable)
    return CopyBlockPlan(K: p.K, drafts: [], realK: p.realK, lookupRows: p.lookupRows)
}

/// T-0266 (C-0753, C-0754): the copy gate's acceptance estimate. The first version used accepted/drafted over the whole
/// process once 200 tokens were drafted; when those read low the gate refused every lookup, no evidence arrived, and the
/// estimate never moved (a lock that lasted until the next restart). Here the counts start from a prior worth `weight`
/// drafted tokens, every copy row adds its evidence, and every round whose lookups the gate refused decays the counts, so a
/// refusing gate drifts back toward the prior and tries lookups again after roughly a hundred refused rounds.
public struct CopyRateEstimate: Equatable {
    public var drafted = 0.0, accepted = 0.0
    public let prior: Double, weight: Double, decay: Double
    public init(prior: Double = 0.8, weight: Double = 20, decay: Double = 0.98) {
        self.prior = prior; self.weight = weight; self.decay = decay
    }
    public var rate: Double { (accepted + prior * weight) / (drafted + weight) }
    public mutating func observe(drafted d: Int, accepted a: Int) { drafted += Double(d); accepted += Double(a) }
    public mutating func refused() { drafted *= decay; accepted *= decay }
}

/// T-0267 (C-0754): the plan with no cost gate -- every row with a lookup proposal takes it (truncated to maxK); the others
/// keep mtpK MTP drafts. ENGINE_BATCH_COPY_ALWAYS=1 uses it to test whether the gate's cost model refuses lookups that pay.
public func planCopyBlockAll(mtpK: Int, lookups: [[Int]?], maxK: Int = 8) -> CopyBlockPlan {
    let cut = lookups.map { $0.map { Array($0.prefix(maxK)) }.flatMap { $0.isEmpty ? nil : $0 } }
    let realK = cut.map { $0?.count ?? mtpK }
    return CopyBlockPlan(K: max(mtpK, realK.max() ?? mtpK), drafts: [], realK: realK, lookupRows: cut.indices.filter { cut[$0] != nil })
}


/// T-0275 (C-0754, C-0759): a measured verify cost curve for the copy gate -- the packed per-row-depth verify's ms per
/// round against its real rows, at the batch size it was measured at (lab/e9/windows/w_t0275_costcurve.sh writes it).
/// Text: one "rows ms" pair per line, '#' comments; at least two points. Between points the cost is interpolated
/// linearly; outside, extended with the nearest segment's slope. A batch other than the table's adds C-0684's per-row
/// term (0.278 ms per row of B), pads add padMs each, as in batchRoundMs.
public struct CopyCostTable: Equatable, Sendable {
    public let batch: Int
    public let rows: [Int]
    public let ms: [Double]
    /// T-0286 (review of T-0277): the MTP-only round (uniform K=2 block, no pads) measured at `batch`, kept apart from the
    /// packed curve; other batch sizes move it by C-0684's per-row cost of a 3-wide row (3 x 1.472 + 0.278 ms)
    public let uniform3: Double?
    public init?(batch: Int, points: [(Int, Double)], uniform3: Double? = nil) {
        let p = points.sorted { $0.0 < $1.0 }
        guard batch > 0, p.count >= 2, Set(p.map { $0.0 }).count == p.count, p.allSatisfy({ $0.1 > 0 }),
              uniform3.map({ $0 > 0 }) ?? true else { return nil }
        self.batch = batch; rows = p.map { $0.0 }; ms = p.map { $0.1 }; self.uniform3 = uniform3
    }
    /// the MTP-only round at batch B with K drafts per row: from `uniform3` when K = 2, else nil (the caller keeps the fit)
    public func baselineMs(B: Int, mtpK: Int) -> Double? {
        guard mtpK == 2, let u = uniform3 else { return nil }
        return u + (3 * 1.472 + 0.278) * Double(B - batch)
    }
    /// "batch N", optional "uniform3 MS", then "rows ms" lines (packed real rows, pads' cost removed)
    public init?(text: String) {
        var batch = 0; var u3: Double? = nil; var pts: [(Int, Double)] = []
        for raw in text.split(separator: "\n") {
            let line = raw.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init)?
                .trimmingCharacters(in: .whitespaces) ?? ""
            if line.isEmpty { continue }
            let f = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            if f.count == 2, f[0] == "batch", let b = Int(f[1]) { batch = b; continue }
            if f.count == 2, f[0] == "uniform3", let u = Double(f[1]) { u3 = u; continue }
            guard f.count == 2, let r = Int(f[0]), let m = Double(f[1]) else { return nil }
            pts.append((r, m))
        }
        self.init(batch: batch, points: pts, uniform3: u3)
    }
    /// ms of a verify whose row-wise layers run `rows` real rows, at the table's batch
    public func ms(rows r: Int) -> Double {
        var i = 0
        while i < rows.count - 2 && r > rows[i + 1] { i += 1 }
        let (r0, r1, m0, m1) = (Double(rows[i]), Double(rows[i + 1]), ms[i], ms[i + 1])
        return m0 + (m1 - m0) * (Double(r) - r0) / (r1 - r0)
    }
    public func roundMs(B: Int, realPositions: Int, pads: Int, padMs: Double) -> Double {
        ms(rows: realPositions) + 0.278 * Double(B - batch) + padMs * Double(pads)
    }
}


/// T-0290: the table measured at the batch size nearest `batch` (ties: the larger batch), or nil when there is none
public func nearestCopyCostTable(_ tables: [CopyCostTable], batch: Int) -> CopyCostTable? {
    tables.min { a, b in
        let da = abs(a.batch - batch), db = abs(b.batch - batch)
        return da != db ? da < db : a.batch > b.batch
    }
}
