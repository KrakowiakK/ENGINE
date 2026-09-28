/// The prefill chunk schedule of `engine serve`, model-free so it can be tested.
///
/// `end` is Serve.swift's `chunkEnd` arithmetic moved here unchanged (P119): P093 rung alignment, H57 the QSA budget
/// cut and width-multiple ends, H50 the canonical-target cap (diagnostic mode only), P093 the one-token hot reserve.
/// `withSharedSplit` is P119's only addition to the schedule: a cold prompt's first chunk ends at the shared-prefix rung.
public enum PrefillChunkPlan {
    /// The end of the chunk that starts at `from`. `canonicalTarget` is nil unless the H25 canonical mode caps chunks.
    public static func end(from: Int, width: Int, rungAlign: Int, promptCount: Int, qsaBudgetCut: Int,
                           canonicalTarget: Int?, hotReserve: Int) -> Int {
        // H57: a chunk ends on a multiple of its own width (after the budget cut at 2048 a 4096 schedule runs
        // 2048, 4096, 8192, 12288, ...), so the prefix-step rungs (every 8192) and the H54 anchors are still captured.
        let w = max(1, width)
        let aligned = (((w > rungAlign ? ((from + w) / w) * w : from + w)) / rungAlign) * rungAlign
        var end = min(promptCount, max(aligned, from + 1))
        if qsaBudgetCut > 0, from < qsaBudgetCut, end > qsaBudgetCut { end = qsaBudgetCut }
        // H50: a wide chunk must end ON the target rather than step over it, or no certified rung is captured.
        if let target = canonicalTarget, from < target, end > target { end = target }
        if hotReserve > 0, end == promptCount, from < promptCount - hotReserve { return promptCount - hotReserve }
        return end
    }

    /// P119: a prompt must be at least rung + this long for the shared-prefix split, capture and resume. The chunk after
    /// the rung then has at least `margin - 1` rows (the last token is the one-token hot reserve's own chunk), where the
    /// cold schedule's first chunk had rung + that many. What makes the two schedules pick the same kernels is not the
    /// attention kernel but MLX's row-count thresholds in the vendored quantized matmuls (DERIVED, review F4):
    ///   - the MoE sorted gather takes `gather_qmm_rhs` only for S * topK / experts >= 4 (S * 10 / 512 >= 4: S >= 205),
    ///     below that `gather_qmv` (backend/metal/quantized.cpp, `B / E >= 4`);
    ///   - `qmm_splitk` splits K while the tile grid is under ~512 threadgroups: N = 1280 (shared-expert gate_up) for
    ///     M <= 192, N = 2560 for M <= 96 (same file, split_k = 512 / tiles).
    /// So the margin must stay >= `sharedPrefixMinTailRows + 1` = 206; 256 leaves 255 rows. (ENGINE_ROUTER_FP32 would add
    /// the fp32 steel split-K for tails <= 992 rows: the server refuses the knob with it.)
    public static let sharedPrefixMargin = 256
    /// The fewest rows the chunk after the rung may have (the MoE gather threshold above, the larger of the two).
    public static let sharedPrefixMinTailRows = 205

    /// P119: does a prompt of `promptCount` tokens get the shared-prefix treatment (split + capture)? `rung` 0 = off.
    public static func sharedSplitEligible(promptCount: Int, rung: Int, margin: Int) -> Bool {
        rung > 0 && promptCount >= rung + margin
    }

    /// P119: the chunk starting at `from` whose unsplit end is `end`. Only the first chunk of a cold prefill (from 0)
    /// that would step over the rung is cut, at the rung; `split` marks a boundary that exists only because of the cut.
    public static func withSharedSplit(from: Int, end: Int, promptCount: Int, rung: Int, margin: Int) -> (end: Int, split: Bool) {
        if sharedSplitEligible(promptCount: promptCount, rung: rung, margin: margin), from == 0, end > rung { return (rung, true) }
        return (end, false)
    }

    /// P119: is the chunk [from, to) the one whose end state is the shared-prefix rung? (The server also requires that
    /// the chunk ran B1 in this request's own body -- not a native batch, not a coalesced import.)
    public static func capturesSharedRung(from: Int, to: Int, promptCount: Int, rung: Int, margin: Int) -> Bool {
        sharedSplitEligible(promptCount: promptCount, rung: rung, margin: margin) && from == 0 && to == rung && to < promptCount
    }
}
