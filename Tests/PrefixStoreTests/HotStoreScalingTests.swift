import Foundation
import XCTest
import MLX
@testable import Qwen4Exp

/// Host-cost scaling of the hot prefix store (opt-in: ENGINE_RUN_STRESS=1; CPU arrays, no model). The store's methods run on
/// the model thread, so their cost is decode time once the store is warm: this times them at realistic sizes -- E entries
/// that share a P-token preamble (an agent's system prompt and tools) plus a 512-token tail of their own, a full eviction
/// ledger -- and prints ms per call, to compare with a ~32 ms decode round.
final class HotStoreScalingTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }
    private let ratio = 4

    private func rows(T: Int) -> [String: MLXArray] {
        ["trunk.L0.k": MLXArray.zeros([1, 1, T, 1]), "trunk.L0.v": MLXArray.zeros([1, 1, T, 1]),
         "trunk.L0.i0": MLXArray.zeros([1, T, 1]), "trunk.L0.i1": MLXArray.zeros([1, max(1, T / ratio), 1])]
    }
    private func rung(_ length: Int, _ salt: Int) -> HotPrefixStore.Rung {
        var d: [String: MLXArray] = [:]
        for l in 0 ..< 36 { d["trunk.L\(l).a0"] = MLXArray([Float(length + salt)]).reshaped([1, 1]) }   // 36 head arrays, as the GDN layers
        return HotPrefixStore.rung(fromExported: d, length: length, ratio: ratio)
    }
    private func ms(_ reps: Int, _ body: () -> Void) -> Double {
        body()
        let t0 = Date(); for _ in 0 ..< reps { body() }
        return Date().timeIntervalSince(t0) * 1000 / Double(reps)
    }
    private func tokens(_ preamble: [Int], _ i: Int) -> [Int] { preamble + (0 ..< 512).map { 1_000_000 + i * 1000 + $0 } }

    /// Regret semantics (always on): an evicted entry's rung counts when the request shares its prefix up to that rung,
    /// the best such rung wins, and nothing counts when the shared prefix stops short of a rung step beyond the hit.
    func testRegretCountsTheBestEvictedRungThatSharesTheRequestsPrefix() {
        let store = HotPrefixStore(capBytes: 1 << 40, rungStep: 512, keepDecodeRungs: 4, strictBudget: true)
        _ = store.setBudgetBytes(1 << 40)
        let a = Array(0 ..< 2048)
        store.store(tokens: a, rows: rows(T: 2048), rowsValidTo: 2048, mtpValidTo: 0, rungs: [rung(512, 0), rung(1024, 0), rung(1536, 0), rung(2048, 0)])
        _ = store.setBudgetBytes(0)                       // evict it into the ledger
        XCTAssertEqual(store.count, 0)
        _ = store.setBudgetBytes(1 << 40)
        let shared1600 = Array(a[0 ..< 1600]) + [-1, -2, -3]
        XCTAssertNil(store.lookup(shared1600))
        XCTAssertEqual(store.regretRequests, 1)
        XCTAssertEqual(store.regretTokens, 1536, "the best evicted rung under the 1600-token shared prefix")
        let shared400 = Array(a[0 ..< 400]) + [-1, -2]
        XCTAssertNil(store.lookup(shared400))
        XCTAssertEqual(store.regretRequests, 1, "no evicted rung lies inside a 400-token shared prefix")
    }

    /// The eviction-regret scan with a FULL ledger (512 evicted entries sharing the preamble): runs in every non-quiet lookup.
    func testRegretScanWithAFullLedger() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ENGINE_RUN_STRESS"] == "1", "opt-in stress timing")
        for P in [1_024, 18_432, 100_000] {
            let preamble = Array(0 ..< P)
            let store = HotPrefixStore(capBytes: 1 << 40, rungStep: 512, keepDecodeRungs: 2, strictBudget: true)
            _ = store.setBudgetBytes(1 << 40)
            for i in 0 ..< 8 { let t = tokens(preamble, i); store.store(tokens: t, rows: rows(T: t.count), rowsValidTo: t.count, mtpValidTo: 0, rungs: [rung(P, i), rung(t.count, i)]) }
            _ = store.setBudgetBytes(store.chargedBytes)   // full: every further store evicts one entry into the ledger
            for i in 8 ..< 530 { let t = tokens(preamble, i); store.store(tokens: t, rows: rows(T: t.count), rowsValidTo: t.count, mtpValidTo: 0, rungs: [rung(P, i), rung(t.count, i)]) }
            let probe = tokens(preamble, 999_999)
            let quiet = ms(10) { _ = store.lookup(probe, quiet: true) }
            let loud = ms(10) { _ = store.lookup(probe) }
            print(String(format: "hot store regret: P %6d entries %d ledger %d | lookup quiet %.3f ms | lookup+regret %.3f ms",
                         P, store.count, store.snapshot()["evicted_ledger"] ?? -1, quiet, loud))
        }
    }

    func testHostCostOfTheStoreAtRealisticSizes() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ENGINE_RUN_STRESS"] == "1", "opt-in stress timing")
        for P in [1_024, 18_432, 100_000] {
            for E in [10, 70, 200] {
                let preamble = Array(0 ..< P)
                let store = HotPrefixStore(capBytes: 1 << 40, rungStep: 512, keepDecodeRungs: 2, strictBudget: true)
                _ = store.setBudgetBytes(1 << 40)
                for i in 0 ..< E {
                    let t = tokens(preamble, i)
                    store.store(tokens: t, rows: rows(T: t.count), rowsValidTo: t.count, mtpValidTo: 0,
                                rungs: [rung(P / 2, i), rung(P, i), rung(t.count - 1, i), rung(t.count, i)])
                }
                let probe = tokens(preamble, 999_999)
                let snap = ms(50) { _ = store.snapshot() }
                let look = ms(10) { _ = store.lookup(probe, quiet: true) }
                let lookRegret = ms(10) { _ = store.lookup(probe) }
                // one store into a full budget: evicts one entry first
                let budget = store.chargedBytes
                _ = store.setBudgetBytes(budget)
                var k = 0
                let storeEvict = ms(5) {
                    k += 1; let t = tokens(preamble, 10_000 + k)
                    store.store(tokens: t, rows: rows(T: t.count), rowsValidTo: t.count, mtpValidTo: 0,
                                rungs: [rung(P / 2, k), rung(P, k), rung(t.count - 1, k), rung(t.count, k)])
                }
                print(String(format: "hot store scaling: P %6d E %3d (ledger %3d) | snapshot %.3f ms | lookup quiet %.3f ms | lookup+regret %.3f ms | store+evict %.3f ms",
                             P, E, store.snapshot()["evicted_ledger"] ?? -1, snap, look, lookRegret, storeEvict))
            }
        }
    }
}
