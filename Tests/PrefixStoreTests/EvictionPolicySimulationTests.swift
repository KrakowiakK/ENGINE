import XCTest
import MLX
@testable import Qwen4Exp

/// 2026-10-01 -- replay of an agent-loop workload shaped like the operator's test agent (overlapping tasks that
/// re-send a handful of prompts sharing a long preamble, cycle after cycle, plus some follow-up turns) against the real
/// hot store, with small arrays whose bytes scale as on E9 (one GDN rung ~ 4.6k tokens of rows) and the server's rung
/// layout: the 2048 budget-cut chunk end, the reserve rung at prompt - 1 (where a repeated prompt resumes), the
/// prompt end, decode rungs every 512 generated tokens (the last two kept) and the final state. Compares the policy
/// before 2026-10-01 (per-entry rung copies, whole-entry eviction) with rung sharing + rung trim by the prefill tokens
/// the server would recompute and the store's eviction-regret counter. Opt-in (minutes): ENGINE_RUN_SIMULATIONS=1.
final class EvictionPolicySimulationTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    struct Rng { var s: UInt64; mutating func next(_ n: Int) -> Int { s = s &* 6364136223846793005 &+ 1442695040888963407; return Int((s >> 33) % UInt64(n)) } }

    private func rows(T: Int) -> [String: MLXArray] {
        ["trunk.L0.k": MLXArray.zeros([1, 1, T, 128]), "trunk.L0.v": MLXArray.zeros([1, 1, T, 128]),
         "trunk.L0.i0": MLXArray.zeros([1, T, 2]), "trunk.L0.i1": MLXArray.zeros([1, T / 4, 2])]
    }
    private func rung(_ length: Int) -> HotPrefixStore.Rung {
        HotPrefixStore.rung(fromExported: ["trunk.L1.a0": MLXArray.zeros([1, 4600, 256])], length: length, ratio: 4)
    }
    /// The server's ladder for a request whose prompt is `p` tokens and whose output brings it to `full`.
    private func ladder(prompt p: Int, full: Int) -> [HotPrefixStore.Rung] {
        var lens: [Int] = []
        if p > 2049 { lens.append(2048) }
        lens.append(p - 1); lens.append(p)
        let decode = stride(from: p + 512, to: full, by: 512).map { $0 }
        lens += decode.suffix(2); lens.append(full)
        return Array(Set(lens)).sorted().map { rung($0) }
    }

    struct Task { var prompts: [[Int]]; var answers: [[Int]]; var cyclesLeft: Int; var next = 0 }

    private func simulate(newPolicy: Bool, budgetMB: Int, seed: UInt64) -> (prompt: Int, cold: Int, followCold: Int, follow: Int, regretReq: Int, evictions: Int, trimmed: Int) {
        var rng = Rng(s: seed)
        var tokenBase = 1_000_000
        func fresh(_ n: Int) -> [Int] { defer { tokenBase += n }; return Array(tokenBase ..< tokenBase + n) }
        func newTask() -> Task {
            let pre = fresh(1000 + rng.next(2500))
            let prompts = (0 ..< (4 + rng.next(5))).map { _ in pre + fresh(50 + rng.next(550)) }
            return Task(prompts: prompts, answers: prompts.map { _ in fresh(200 + rng.next(1200)) }, cyclesLeft: 3 + rng.next(25))
        }
        let store = HotPrefixStore(capBytes: budgetMB << 20, rungStep: 512, keepDecodeRungs: 2, strictBudget: true)
        store.rungSharing = newPolicy; store.rungTrim = newPolicy
        var active = [newTask(), newTask(), newTask()]
        var promptTok = 0, cold = 0, followCold = 0, follow = 0, requests = 0
        func serve(_ prompt: [Int], _ answer: [Int]) -> Int {
            let hit = store.lookup(prompt)
            if let hit { _ = store.exported(hit, ratio: 4) }
            let got = hit?.length ?? 0
            let full = prompt + answer
            store.store(tokens: full, rows: rows(T: full.count), rowsValidTo: full.count, mtpValidTo: 0, rungs: ladder(prompt: prompt.count, full: full.count))
            return prompt.count - got
        }
        while requests < 2500 {
            let t = rng.next(active.count)
            let i = active[t].next
            let prompt = active[t].prompts[i], answer = active[t].answers[i]
            let c = serve(prompt, answer)
            promptTok += prompt.count; cold += c; requests += 1
            if rng.next(100) < 15 {   // a follow-up turn on that answer
                let p2 = prompt + answer + fresh(30 + rng.next(120))
                let c2 = serve(p2, fresh(150 + rng.next(400)))
                promptTok += p2.count; cold += c2; followCold += c2; follow += p2.count; requests += 1
            }
            active[t].next += 1
            if active[t].next == active[t].prompts.count {
                active[t].next = 0; active[t].cyclesLeft -= 1
                if active[t].cyclesLeft == 0 { active[t] = newTask() }
            }
        }
        return (promptTok, cold, followCold, follow, store.regretRequests, store.evictions, store.rungsTrimmed)
    }

    func testRungSharingAndTrimRecomputeLessOnAnAgentLoop() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ENGINE_RUN_SIMULATIONS"] == "1", "opt-in: ENGINE_RUN_SIMULATIONS=1")
        var lines: [String] = []
        var oldCold = 0, newCold = 0, oldRegret = 0, oldFollow = 0, newFollow = 0
        for budget in [500, 1000, 2000] {
            for seed: UInt64 in [1, 2, 3] {
                let o = simulate(newPolicy: false, budgetMB: budget, seed: seed)
                let n = simulate(newPolicy: true, budgetMB: budget, seed: seed)
                oldCold += o.cold; newCold += n.cold; oldRegret += o.regretReq; oldFollow += o.followCold; newFollow += n.followCold
                lines.append(String(format: "budget %4d MB seed %d | before cold %.1f%% follow-up cold %.1f%% regret %d evict %d | after cold %.1f%% follow-up cold %.1f%% regret %d evict %d trimmed %d",
                                    budget, seed, 100 * Double(o.cold) / Double(o.prompt), 100 * Double(o.followCold) / Double(max(1, o.follow)), o.regretReq, o.evictions,
                                    100 * Double(n.cold) / Double(n.prompt), 100 * Double(n.followCold) / Double(max(1, n.follow)), n.regretReq, n.evictions, n.trimmed))
            }
        }
        print(lines.joined(separator: "\n"))
        print(String(format: "TOTAL cold tokens before %d after %d (%.1f%%); follow-up cold before %d after %d", oldCold, newCold,
                     100 * Double(newCold - oldCold) / Double(max(1, oldCold)), oldFollow, newFollow))
        XCTAssertGreaterThan(oldRegret, 0, "the workload must put the store under eviction pressure")
        XCTAssertLessThan(newCold, oldCold)
    }
}
