import XCTest
@testable import EngineServeSupport
import Qwen4Exp

/// T-0248 (D-0084, design C-0718): host-side pieces of padded batched copy drafts, not wired into serving yet.
final class BatchCopyDraftTests: XCTestCase {
    // MARK: acceptRow

    /// without padding it is runBatchedMTPRound's rule: accept the matching prefix, commit it plus the next prediction
    func testAcceptRowMatchesTheRoundsRule() {
        XCTAssertEqual(acceptRow(preds: [5, 6, 9], drafts: [5, 6], realK: 2).accepted, 2)
        XCTAssertEqual(acceptRow(preds: [5, 6, 9], drafts: [5, 6], realK: 2).tokens, [5, 6, 9])
        XCTAssertEqual(acceptRow(preds: [5, 7, 9], drafts: [5, 6], realK: 2).tokens, [5, 7])
        XCTAssertEqual(acceptRow(preds: [4, 7, 9], drafts: [5, 6], realK: 2).tokens, [4])
    }

    /// a padded row: the cap limits acceptance to the real drafts (accounting; T-0250: not needed for exactness)
    func testRealKCapStopsPaddedFillersFromBeingAccepted() {
        // real drafts [5, 6], padded to K = 5 by repeating 6; the model happens to predict 6 again after them
        let r = acceptRow(preds: [5, 6, 6, 6, 6, 1], drafts: [5, 6, 6, 6, 6], realK: 2)
        XCTAssertEqual(r.accepted, 2)
        XCTAssertEqual(r.tokens, [5, 6, 6])               // the third token is the prediction after the real drafts
        XCTAssertEqual(acceptRow(preds: [5, 6, 6, 6, 6, 1], drafts: [5, 6, 6, 6, 6], realK: 5).accepted, 5)
        XCTAssertEqual(acceptRow(preds: [5, 9], drafts: [5], realK: 0).tokens, [5])   // no real draft: just the prediction
    }

    /// the think-close and stop-id cuts apply inside the accepted prefix, as in the round
    func testThinkCloseAndStopIdsCutTheAcceptedPrefix() {
        XCTAssertEqual(acceptRow(preds: [5, 99, 7, 8], drafts: [5, 99, 7], realK: 3, thinkCloseCut: 99).tokens, [5, 99])
        XCTAssertEqual(acceptRow(preds: [5, 2, 7, 8], drafts: [5, 2, 7], realK: 3, stopIds: [2]).tokens, [5, 2])
        XCTAssertEqual(acceptRow(preds: [5, 2, 7, 8], drafts: [5, 2, 7], realK: 3, stopIds: [2]).accepted, 1)
    }

    // MARK: BatchCopyLookup == the census walk

    /// driving the per-row policy over an output gives exactly copyCensusWalk's counts (the census prices this policy)
    func testPerRowLookupPolicyReproducesTheCensusWalk() {
        var rng = SystemRandomNumberGenerator()
        func rand(_ n: Int) -> [Int] { (0 ..< n).map { _ in Int.random(in: 0 ..< 500, using: &rng) } }
        var cases: [([Int], [Int])] = [
            (Array(100 ..< 400), Array(150 ..< 250)),                                   // re-typed span
            (Array(0 ..< 300), Array(1000 ..< 1100)),                                   // nothing to copy
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97]),
        ]
        for _ in 0 ..< 20 {                                                             // partly copied, small vocab
            let p = rand(400)
            let start = Int.random(in: 0 ..< 300, using: &rng)
            cases.append((p, Array(p[start ..< start + 60]) + rand(40) + Array(p[(start + 10) ..< (start + 70)])))
        }
        // a copied span with one wrong token at every offset of the doubling blocks (rounds accepted all but the last draft)
        for off in 0 ..< 24 { cases.append((Array(0 ..< 300), Array(100 ..< (120 + off)) + [999] + Array((121 + off) ..< 200))) }
        // periodic output: lookups run into the end of the context (the continuation is shorter than the asked length)
        for period in 2 ... 7 { cases.append((Array(500 ..< 520), (0 ..< 90).map { 1000 + $0 % period })) }
        // tiny vocabularies: many n-gram repeats, continuations cut by the context's end, rounds accepted all but the last
        // draft -- the states where L and 'full' change the next proposal
        for _ in 0 ..< 300 {
            let v = Int.random(in: 2 ... 4, using: &rng)
            cases.append(((0 ..< Int.random(in: 8 ... 40, using: &rng)).map { _ in Int.random(in: 0 ..< v, using: &rng) },
                          (0 ..< Int.random(in: 20 ... 80, using: &rng)).map { _ in Int.random(in: 0 ..< v, using: &rng) }))
        }
        for (cfg, (prompt, output)) in zip([(4, 2, 8), (6, 2, 8), (10, 12, 12)].flatMap { c in cases.map { _ in c } },
                                           cases + cases + cases) {
            let census = copyCensusWalk(prompt: prompt, output: output, minMatch: cfg.0, startLen: cfg.1, maxLen: cfg.2)
            var lk = BatchCopyLookup(context: prompt + output.prefix(1), minMatch: cfg.0, startLen: cfg.1, maxLen: cfg.2)
            var mine = CopyCensusCounts()
            let end = output.count - 1
            var j = 0
            while j < end {
                if let p = lk.propose() {
                    var acc = 0
                    for t in p { guard j + 1 + acc <= end, output[j + 1 + acc] == t else { break }; acc += 1 }
                    let commit = min(acc + 1, end - j)
                    mine.lookupRounds += 1; mine.drafted += p.count; mine.accepted += acc; mine.lookupTokens += commit
                    lk.commit(tokens: Array(output[(j + 1) ... (j + commit)]), lookup: p, accepted: acc)
                    j += commit
                } else {
                    lk.commit(tokens: [output[j + 1]], lookup: nil, accepted: 0)
                    j += 1
                }
            }
            XCTAssertEqual(mine, census, "config \(cfg) prompt \(prompt.count) output \(output.count)")
        }
    }

    // MARK: planCopyBlock

    func testPlanPadsEveryRowToTheBlockAndRecordsRealK() {
        let mtp = [[1, 2], [3, 4], [5, 6]]
        let plan = planCopyBlock(mtpDrafts: mtp, lookups: [nil, [10, 11, 12, 13, 14, 15], nil])
        XCTAssertEqual(plan.K, 6)
        XCTAssertEqual(plan.lookupRows, [1])
        XCTAssertEqual(plan.realK, [2, 6, 2])
        XCTAssertEqual(plan.drafts[0], [1, 2, 2, 2, 2, 2])
        XCTAssertEqual(plan.drafts[1], [10, 11, 12, 13, 14, 15])
        XCTAssertTrue(plan.drafts.allSatisfy { $0.count == plan.K })
    }

    func testPlanCapsLookupsAtMaxKAndFallsBackWithoutLookups() {
        let mtp = [[1, 2], [3, 4]]
        XCTAssertEqual(planCopyBlock(mtpDrafts: mtp, lookups: [Array(10 ..< 30), Array(40 ..< 60)], maxK: 8).K, 8)
        let none = planCopyBlock(mtpDrafts: mtp, lookups: [nil, []])
        XCTAssertEqual(none, CopyBlockPlan(K: 2, drafts: mtp, realK: [2, 2], lookupRows: []))
    }

    /// the gate: one long lookup row among 16 MTP rows pads all 16 and does not pay; most rows copying does
    func testPlanTakesLookupsOnlyWhenTheyPayAtTheRoundsCost() {
        let mtp = Array(repeating: [1, 2], count: 16)
        let oneRow = planCopyBlock(mtpDrafts: mtp, lookups: [Array(10 ..< 18)] + Array(repeating: nil, count: 15))
        XCTAssertEqual(oneRow.K, 2)
        XCTAssertEqual(oneRow.lookupRows, [])
        let most = planCopyBlock(mtpDrafts: mtp, lookups: Array(repeating: Array(10 ..< 18), count: 12) + Array(repeating: nil, count: 4))
        XCTAssertEqual(most.K, 8)
        XCTAssertEqual(most.lookupRows.count, 12)
        // C-0684's fit: a K=2 round at B=16 is 122.5 ms
        XCTAssertEqual(batchRoundMs(B: 16, K: 2), 122.502, accuracy: 0.01)
    }

    /// T-0249: which forward runs the block decides the gate. 6 of 16 rows copying 8 drafts pay in the packed per-row-depth
    /// verify (pads 0.397 ms, C-0443) but not in a uniform block (pads cost a full 1.472 ms position)
    func testPadCostOfTheVerifyForwardDecidesTheGate() {
        let mtp = Array(repeating: [1, 2], count: 16)
        let lookups: [[Int]?] = Array(repeating: Array(10 ..< 18), count: 6) + Array(repeating: nil, count: 10)
        XCTAssertEqual(planCopyBlock(mtpDrafts: mtp, lookups: lookups).lookupRows.count, 6)               // packed (default)
        XCTAssertEqual(planCopyBlock(mtpDrafts: mtp, lookups: lookups, padMs: 1.472).lookupRows, [])     // uniform
        // the packed price of that block: 16 rows, real positions 10*3 + 6*9 = 84, pads 16*9 - 84 = 60
        XCTAssertEqual(batchRoundMs(B: 16, realPositions: 84, pads: 60), 47.39 + 4.448 + 1.472 * 84 + 0.397 * 60, accuracy: 1e-9)
    }

    // MARK: T-0250 -- packed order and the oracle round

    /// splitPacked follows Qwen4ExpPackedRows' gather order: row b's positions 0..<depth[b], rows in order
    func testSplitPackedMatchesThePackedRowsGatherOrder() {
        let depths = [3, 1, 4, 2], S = 4
        let L = Qwen4ExpPackedRows(depths: depths, positions: S)
        let rows = splitPacked(L.gather.map { Int($0) }, depths: depths)
        for (b, r) in rows.enumerated() { XCTAssertEqual(r, (0 ..< depths[b]).map { b * S + $0 }) }
        XCTAssertEqual(packedDepths(CopyBlockPlan(K: 6, drafts: [], realK: [2, 6, 0], lookupRows: [1])), [3, 7, 1])
    }

    /// The CPU stand-in for the bit-identity test: an oracle model (each verify position predicts the true next token when
    /// the drafts before it were right, garbage otherwise) drives whole multi-row requests through plan -> verify (packed or
    /// uniform padded, capped or not) -> acceptRow -> commit -> lookup update. Every row must commit exactly its true output.
    func testOracleRoundsCommitExactlyTheTrueOutputInEveryMode() {
        var rng = SystemRandomNumberGenerator()
        for mode in ["packed", "uniform-capped", "uniform-uncapped"] {
            for trial in 0 ..< 40 {
                let B = Int.random(in: 1 ... 12, using: &rng), vocab = [3, 50, 5000][trial % 3]
                var prompts: [[Int]] = [], outputs: [[Int]] = []
                for _ in 0 ..< B {
                    let p = (0 ..< Int.random(in: 20 ... 200, using: &rng)).map { _ in Int.random(in: 0 ..< vocab, using: &rng) }
                    let s0 = Int.random(in: 0 ..< p.count / 2, using: &rng)
                    let copied = Bool.random(using: &rng) ? Array(p[s0...].prefix(60)) : []
                    let o = [p.last! &+ 1] + copied + (0 ..< Int.random(in: 5 ... 60, using: &rng)).map { _ in Int.random(in: 0 ..< vocab, using: &rng) }
                    prompts.append(p); outputs.append(o)
                }
                var done = outputs.map { Array($0.prefix(1)) }                    // output[0] comes from the prefill
                var lk = (0 ..< B).map { BatchCopyLookup(context: prompts[$0] + outputs[$0].prefix(1), minMatch: 4, startLen: 2, maxLen: 8) }
                var rounds = 0
                while (0 ..< B).contains(where: { done[$0].count < outputs[$0].count }), rounds < 10_000 {
                    rounds += 1
                    let live = (0 ..< B).filter { done[$0].count < outputs[$0].count }
                    let lookups = live.map { lk[$0].propose() }
                    // MTP drafts: right with probability ~0.7, else garbage
                    let mtp = live.map { b -> [Int] in (0 ..< 2).map { i in
                        let n = done[b].count + i
                        return n < outputs[b].count && Double.random(in: 0 ..< 1, using: &rng) < 0.7 ? outputs[b][n] : -1 - i } }
                    let plan = planCopyBlock(mtpDrafts: mtp, lookups: lookups)
                    // the oracle verify: position t of row r predicts the true token after pending + drafts[0..<t] if those were right
                    func oracle(_ b: Int, _ drafts: [Int]) -> [Int] {
                        var preds: [Int] = [], ok = true
                        for t in 0 ... drafts.count {
                            let n = done[b].count + t
                            preds.append(ok && n < outputs[b].count ? outputs[b][n] : -99)
                            if t < drafts.count { ok = ok && n < outputs[b].count && drafts[t] == outputs[b][n] }
                        }
                        return preds
                    }
                    var rowPreds: [[Int]], rowDrafts: [[Int]], caps: [Int]
                    if mode == "packed" {
                        rowDrafts = live.indices.map { Array(plan.drafts[$0].prefix(plan.realK[$0])) }
                        let flat = live.indices.flatMap { oracle(live[$0], rowDrafts[$0]) }
                        rowPreds = splitPacked(flat, depths: packedDepths(plan)); caps = plan.realK
                    } else {
                        rowDrafts = plan.drafts
                        rowPreds = live.indices.map { oracle(live[$0], rowDrafts[$0]) }
                        caps = mode == "uniform-capped" ? plan.realK : Array(repeating: plan.K, count: live.count)
                    }
                    for (i, b) in live.enumerated() {
                        let r = acceptRow(preds: rowPreds[i], drafts: rowDrafts[i], realK: caps[i])
                        let room = outputs[b].count - done[b].count
                        let tokens = Array(r.tokens.prefix(room))                     // a finished request stops
                        XCTAssertEqual(tokens, Array(outputs[b][done[b].count ..< done[b].count + tokens.count]),
                                       "mode \(mode) trial \(trial) row \(b) round \(rounds)")
                        done[b] += tokens
                        let used = plan.lookupRows.contains(i) ? lookups[i] : nil
                        lk[b].commit(tokens: tokens, lookup: used, accepted: min(r.accepted, used?.count ?? 0))
                    }
                }
                XCTAssertEqual(done, outputs, "mode \(mode) trial \(trial)")
                XCTAssertLessThan(rounds, 10_000)
            }
        }
    }

    /// T-0247: the host-free plan agrees with the full plan on K, realK and lookup rows
    func testPlanWithoutHostDraftsMatchesTheFullPlan() {
        let mtp = Array(repeating: [1, 2], count: 16)
        for lookups in [Array(repeating: Array(10 ..< 18), count: 12) + Array(repeating: nil, count: 4),
                        [Array(10 ..< 18)] + Array(repeating: nil, count: 15),
                        Array(repeating: nil, count: 16)] as [[[Int]?]] {
            for pad in [0.397, 1.472] {
                let full = planCopyBlock(mtpDrafts: mtp, lookups: lookups, maxK: 7, padMs: pad)
                let lite = planCopyBlock(mtpK: 2, lookups: lookups, maxK: 7, padMs: pad)
                XCTAssertEqual([full.K] + full.realK + full.lookupRows, [lite.K] + lite.realK + lite.lookupRows)
            }
        }
        var lk = BatchCopyLookup(context: [1, 2, 3], minMatch: 4, startLen: 2, maxLen: 7)
        XCTAssertEqual(lk.contextCount, 3)
        lk.commit(tokens: [4, 5], lookup: nil, accepted: 0)
        XCTAssertEqual(lk.contextCount, 5)
    }

    // T-0266 (C-0753): the deploy's lock -- 200 drafted at 0.55 -- now recovers after refused rounds instead of holding
    func testCopyRateEstimateCannotLock() {
        var e = CopyRateEstimate()
        XCTAssertEqual(e.rate, 0.8, accuracy: 1e-12)
        e.observe(drafted: 200, accepted: 110)
        XCTAssertLessThan(e.rate, 0.6)
        let B = 18
        let lookups: [[Int]?] = (0 ..< B).map { $0 < 10 ? Array(100 ..< 107) : nil }
        func takes(_ r: Double) -> Bool {
            !planCopyBlock(mtpK: 2, lookups: lookups, maxK: 7, lookupAcceptRate: r, mtpTokensPerRow: 2.5, padMs: 0.397).lookupRows.isEmpty
        }
        XCTAssertFalse(takes(e.rate))
        var rounds = 0
        while !takes(e.rate) && rounds < 1000 { e.refused(); rounds += 1 }
        XCTAssertTrue(takes(e.rate))
        XCTAssertGreaterThan(rounds, 20)      // not after a handful: a refusing gate on extract-like text stays refusing a while
        XCTAssertLessThan(rounds, 300)        // but not for the rest of the process
        // steady evidence wins over the prior: a long run at 0.85 reads close to 0.85
        var f = CopyRateEstimate(); f.observe(drafted: 5000, accepted: 4250)
        XCTAssertEqual(f.rate, 0.85, accuracy: 0.002)
    }

    // T-0267: no gate -- every proposal is taken, truncated to maxK; MTP rows keep mtpK
    func testPlanCopyBlockAllTakesEveryProposal() {
        let p = planCopyBlockAll(mtpK: 2, lookups: [nil, [1, 2, 3, 4, 5, 6, 7, 8, 9], [4], nil], maxK: 7)
        XCTAssertEqual(p, CopyBlockPlan(K: 7, drafts: [], realK: [2, 7, 1, 2], lookupRows: [1, 2]))
        XCTAssertEqual(planCopyBlockAll(mtpK: 2, lookups: [nil, []], maxK: 7), CopyBlockPlan(K: 2, drafts: [], realK: [2, 2], lookupRows: []))
    }

    // T-0275: the measured cost curve -- parse, interpolate, extrapolate, batch offset, pads
    func testCopyCostTableParsesAndInterpolates() {
        let t = CopyCostTable(text: "# packed verify at B 19\nbatch 19\n57 130.0\n64 140.0\n68 160.0 # step\n100 200.0\n")!
        XCTAssertEqual(t.batch, 19); XCTAssertEqual(t.rows, [57, 64, 68, 100])
        XCTAssertEqual(t.ms(rows: 57), 130.0, accuracy: 1e-9)
        XCTAssertEqual(t.ms(rows: 66), 150.0, accuracy: 1e-9)                 // inside the step
        XCTAssertEqual(t.ms(rows: 84), 180.0, accuracy: 1e-9)
        XCTAssertEqual(t.ms(rows: 116), 220.0, accuracy: 1e-9)                // beyond: the last segment's slope
        XCTAssertEqual(t.ms(rows: 50), 120.0, accuracy: 1e-9)                 // below: the first segment's slope
        XCTAssertEqual(t.roundMs(B: 17, realPositions: 64, pads: 10, padMs: 0.4), 140.0 - 2 * 0.278 + 4.0, accuracy: 1e-9)
        XCTAssertNil(CopyCostTable(text: "batch 19\n57 130\n"))             // one point
        XCTAssertNil(CopyCostTable(text: "57 130\n64 140\n"))               // no batch
        XCTAssertNil(CopyCostTable(text: "batch 19\n57 130\n57 131\n"))    // duplicate rows
        XCTAssertNil(CopyCostTable(text: "batch 19\n57 x\n64 140\n"))
    }

    // a measured step the linear fit hides makes the gate refuse a round it would take on the fit
    func testCostTableStepChangesTheGate() {
        let B = 19, K = 2
        let lookups: [[Int]?] = (0 ..< B).map { $0 < 8 ? Array(0 ..< 7) : nil }      // 8*8 + 11*3 = 97 real rows
        let fit = planCopyBlock(mtpK: K, lookups: lookups, maxK: 7, lookupAcceptRate: 0.85, mtpTokensPerRow: 2.5, padMs: 0.397)
        XCTAssertEqual(fit.lookupRows, Array(0 ..< 8))
        // linear like the fit up to 64 rows, then a 30 ms step
        let stepped = CopyCostTable(batch: B, points: [(57, batchRoundMs(B: B, realPositions: 57)), (64, batchRoundMs(B: B, realPositions: 64)),
                                                       (65, batchRoundMs(B: B, realPositions: 65) + 30), (120, batchRoundMs(B: B, realPositions: 120) + 30)])!
        let withStep = planCopyBlock(mtpK: K, lookups: lookups, maxK: 7, lookupAcceptRate: 0.85, mtpTokensPerRow: 2.5, padMs: 0.397,
                                     costTable: stepped)
        XCTAssertTrue(withStep.lookupRows.isEmpty)
        // a table equal to the fit decides as the fit does
        let same = CopyCostTable(batch: B, points: [(40, batchRoundMs(B: B, realPositions: 40)), (160, batchRoundMs(B: B, realPositions: 160))])!
        XCTAssertEqual(planCopyBlock(mtpK: K, lookups: lookups, maxK: 7, lookupAcceptRate: 0.85, mtpTokensPerRow: 2.5, padMs: 0.397,
                                     costTable: same), fit)
    }

    // T-0286: the MTP-only baseline comes from the table's uniform3 (moved by a 3-wide row's cost for other B), never from
    // the packed curve; without uniform3 the gate keeps the fit's baseline
    func testCostTableBaselineIsTheUniformMeasurementNotThePackedCurve() {
        let t = CopyCostTable(text: "batch 19\nuniform3 136.6\n62 126.0\n97 180.0\n")!
        XCTAssertEqual(t.uniform3, 136.6)
        XCTAssertEqual(t.baselineMs(B: 19, mtpK: 2)!, 136.6, accuracy: 1e-9)
        XCTAssertEqual(t.baselineMs(B: 18, mtpK: 2)!, 136.6 - (3 * 1.472 + 0.278), accuracy: 1e-9)
        XCTAssertNil(t.baselineMs(B: 19, mtpK: 3))
        XCTAssertNil(CopyCostTable(text: "batch 19\n62 126.0\n97 180.0\n")!.baselineMs(B: 19, mtpK: 2))
        // a packed curve cheaper than the fit must not drag the baseline down at B 18 or B 20: the plan at B 18 and B 20
        // compares against the uniform baseline, so an MTP-only group (no lookups) stays MTP-only and a copy group's
        // decision does not flip just because B moved by one
        for B in [18, 19, 20] {
            let lk: [[Int]?] = (0 ..< B).map { $0 < 6 ? Array(0 ..< 7) : nil }
            let p = planCopyBlock(mtpK: 2, lookups: lk, maxK: 7, lookupAcceptRate: 0.85, mtpTokensPerRow: 2.5, padMs: 0.397, costTable: t)
            XCTAssertEqual(p.lookupRows, Array(0 ..< 6), "B \(B)")
        }
    }

    // T-0286: the exact file w_t0277_costcurve.sh writes parses (comment, batch, uniform3, packed points)
    func testCostTableReadsTheWindowWritersFile() {
        let t = CopyCostTable(text: "# T-0277 packed verify cost at B 19 (S 8 block, pads at 0.397 ms removed); uniform3 = the MTP round (uniform S3)\nbatch 19\nuniform3 137.00\n62 124.27\n81 156.81\n97 188.16\n")
        XCTAssertNotNil(t)
        XCTAssertEqual(t?.batch, 19); XCTAssertEqual(t?.rows, [62, 81, 97]); XCTAssertEqual(t?.uniform3, 137.0)
    }

    // T-0290: one table per measured batch size; a round uses the nearest
    func testNearestCostTableByBatch() {
        let t19 = CopyCostTable(text: "batch 19\nuniform3 136\n62 126\n97 180\n")!
        let t32 = CopyCostTable(text: "batch 32\nuniform3 190\n100 200\n200 330\n")!
        XCTAssertEqual(nearestCopyCostTable([t19, t32], batch: 18)?.batch, 19)
        XCTAssertEqual(nearestCopyCostTable([t19, t32], batch: 25)?.batch, 19)
        XCTAssertEqual(nearestCopyCostTable([t19, t32], batch: 26)?.batch, 32)     // 7 vs 6 away: 32 is nearer
        XCTAssertEqual(nearestCopyCostTable([t19, t32], batch: 30)?.batch, 32)
        XCTAssertNil(nearestCopyCostTable([], batch: 20))
        let tA = CopyCostTable(text: "batch 20\n57 1\n60 2\n")!, tB = CopyCostTable(text: "batch 24\n57 1\n60 2\n")!
        XCTAssertEqual(nearestCopyCostTable([tA, tB], batch: 22)?.batch, 24)       // a tie goes to the larger batch
    }
}
