import Foundation

/// Cost samples belong to one stable pool membership. Periodic exploration must count
/// plain steps too, or a losing speculative sample disables all future observation.
public struct BatchDraftPolicy {
    public private(set) var acceptance = 0.8
    public private(set) var roundMs: [Int: Double] = [:]
    public private(set) var plainMs = 0.0
    private var roundsSincePlain = 0
    private var plainSinceRound = 0
    public init() {}
    public func depth(maxK: Int, mode: String) -> Int {
        guard maxK > 0, mode != "never" else { return 0 }
        if mode == "always" { return maxK }
        // H60 gate mode `cycle:K1,K2,...`: a fixed, deterministic depth per decision (0 = a plain step), so an identity gate
        // can drive the pre-queue's discard paths (a plain step or a different K right after a round) reproducibly
        if let c = Self.cycle(mode) { return min(c[decisions % c.count], maxK) }
        // H60: at maxK <= 3 every decision is B44's (the global EMA, never restarted). Deeper drafting reads each depth's OWN
        // acceptance EMA: the H60 window-2 replay showed that restarting one shared EMA on a depth switch let a single weak
        // round declare the round a loser and trigger B44's up-to-32 plain-step streak (68 plain B5 steps in one slot)
        guard maxK > 3 else {
            return gated(acceptance >= 0.6 ? min(3, maxK) : (acceptance >= 0.45 ? min(2, maxK) : 1), acc: acceptance)
        }
        let p = impliedPNow
        let k = deep ? maxK : { let f3 = Self.fraction(p: p, K: 3); return f3 >= 0.6 ? 3 : (f3 >= 0.45 ? 2 : 1) }()
        // the depth's own EMA only while it is the depth being sampled; any other depth's EMA may be stale, so its fraction
        // is projected from the current p (a stale low EMA would otherwise re-open the plain-step streak)
        return gated(k, acc: k == lastK ? (accByK[k] ?? Self.fraction(p: p, K: k)) : Self.fraction(p: p, K: k))
    }
    /// the per-position acceptance implied by the most recently sampled depth's own EMA (B44's global EMA before any round)
    public var impliedPNow: Double {
        guard lastK > 0, let f = accByK[lastK] else { return Self.impliedP(fraction: acceptance, K: max(lastK, 3)) }
        return Self.impliedP(fraction: f, K: lastK)
    }
    private func gated(_ k: Int, acc: Double) -> Int {
        if plainMs == 0 { return roundsSincePlain >= 39 ? 0 : k }
        if roundsSincePlain >= exploreInterval(k: k, acc: acc) { return 0 }
        if plainSinceRound >= 32 { return k }
        guard let cost = roundMs[k] else { return k }
        return (1 + Double(k) * acc) * plainMs > cost ? k : 0
    }
    /// P106 H60: a forced plain step is a sample of the losing arm. H53 (OMP, B4/B5): 626 of 30400 steps, each ~0.57 of a
    /// round's tokens per ms -- ~0.5 % of batched decode. While the round's estimated rate beats the plain step's by
    /// `wideMargin` or more, the plain arm is re-sampled every `exploreEveryWide` rounds instead of every 32; the margin is
    /// re-evaluated every round from the running acceptance, so a falling acceptance returns to the 32-round cadence at once.
    public static let exploreEvery = 32, exploreEveryWide = 256
    public static let wideMargin = 1.5
    public func exploreInterval(k: Int, acc: Double? = nil) -> Int {
        guard plainMs > 0, let cost = roundMs[k], cost > 0 else { return Self.exploreEvery }
        return (1 + Double(k) * (acc ?? acceptance)) * plainMs >= Self.wideMargin * cost ? Self.exploreEveryWide : Self.exploreEvery
    }
    /// the decisions noted so far (rounds + plain steps); indexes `cycle:` modes
    public private(set) var decisions = 0
    public static func cycle(_ mode: String) -> [Int]? {
        guard mode.hasPrefix("cycle:") else { return nil }
        let ks = mode.dropFirst(6).split(separator: ",").compactMap { Int($0) }
        return ks.isEmpty || ks.contains(where: { $0 < 0 }) ? nil : ks
    }
    public static func fraction(p: Double, K: Int) -> Double {
        guard K > 0 else { return 0 }
        var s = 0.0, q = 1.0
        for _ in 0 ..< K { q *= p; s += q }
        return s / Double(K)
    }
    public mutating func noteRound(k: Int, ms: Double, acceptance acc: Double) {
        decisions += 1
        let first = roundMs.isEmpty
        // H60: the same EMA per depth. On a switch to another depth its prior is PROJECTED from the current p (never a stale
        // EMA of that depth, never the lone new sample), so one weak round moves the estimate by 20 %, as within a depth.
        let prior: Double? = first ? nil : (k == lastK ? accByK[k] : Self.fraction(p: impliedPNow, K: k))
        roundMs[k] = roundMs[k].map { 0.8 * $0 + 0.2 * ms } ?? ms
        acceptance = first ? acc : 0.8 * acceptance + 0.2 * acc          // B44's EMA over every round, unchanged
        accByK[k] = prior.map { 0.8 * $0 + 0.2 * acc } ?? acc
        lastK = k
        roundsSincePlain += 1; plainSinceRound = 0
        let p = impliedPNow
        if !deep, p >= Self.deepUp { deep = true } else if deep, p < Self.deepDown { deep = false }
    }
    public private(set) var accByK: [Int: Double] = [:]
    /// P106 H60: drafting deeper than 3 (only with --batch-mtp > 3). `acceptance` is a fraction at depth K; the
    /// per-position conditional rate p that produces it solves (p + p^2 + ... + p^K) / K = acceptance. Depth K+1 pays
    /// while p^(K+1) / (1 + p + ... + p^K) exceeds the marginal round cost (H53 B4: +12.1 ms on 86.3 -> p* ~ 0.80), so
    /// the policy goes deep at p >= 0.84 and back at p < 0.80 (hysteresis: a p at break-even does not flip every round).
    public static let deepUp = 0.84, deepDown = 0.80
    public private(set) var deep = false
    private var lastK = 0
    public static func impliedP(fraction f: Double, K: Int) -> Double {
        guard K > 1 else { return min(max(f, 0), 1) }
        var lo = 0.0, hi = 1.0
        for _ in 0 ..< 40 {
            let p = (lo + hi) / 2
            var s = 0.0, q = 1.0
            for _ in 0 ..< K { q *= p; s += q }
            if s / Double(K) < f { lo = p } else { hi = p }
        }
        return (lo + hi) / 2
    }
    public mutating func notePlain(ms: Double) {
        decisions += 1
        plainMs = plainMs == 0 ? ms : 0.8 * plainMs + 0.2 * ms
        roundsSincePlain = 0; plainSinceRound += 1
    }
}
