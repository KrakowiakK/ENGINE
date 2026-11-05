// T-0186: packed prefill on the owner -- the picked mixable chunk takes the other requests' queued mixable chunks (rank
// order, within packRows rows and packMax chunks) and, when present, the batched steps; one `packedStep` call, then every
// chunk's body in order. Fake callbacks, CPU only. The layout of the packed rows is pinned at the end.
import Foundation
import MLX
import XCTest

@testable import EngineServeSupport
@testable import Qwen4Exp

private final class Rec<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    @discardableResult func with<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&v) }
    var value: T { with { $0 } }
}

final class PackedDispatchTests: XCTestCase {
    private func waitForQueued(_ count: Int, on mt: ModelThread, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while mt.phaseStats().values.reduce(0, { $0 + $1.queued }) != count {
            if ProcessInfo.processInfo.systemUptime > deadline {
                XCTFail("expected \(count) queued jobs, got \(mt.phaseStats())", file: file, line: line); return
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    /// events: "packed:<chunk keys>:<step keys>", "mixed:<key>:<step keys>", "step:<keys>", "chunk:<key>"
    private func make(events: Rec<[String]>, packRows: Int, packMax: Int = 8, maxConsecutive: Int = 4) -> ModelThread {
        ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: maxConsecutive,
                    packRows: packRows, packMax: packMax,
                    packedStep: { mixes, reqs in
                        events.with { $0.append("packed:\(mixes.map(\.key)):\(reqs.map(\.key))") }
                        return reqs.map { $0.key * 100 + 2 }
                    },
                    mixedStep: { mix, reqs in
                        events.with { $0.append("mixed:\(mix.key):\(reqs.map(\.key))") }
                        return reqs.map { $0.key * 100 + 1 }
                    }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }
            return reqs.map { $0.key * 100 }
        }
    }

    /// Queue `chunks` (key, rows) behind a held owner, release it, and return the events once every chunk body ran.
    private func runChunks(_ mt: ModelThread, _ chunks: [(Int, Int)], events: Rec<[String]>) -> [String] {
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "chunks"); done.expectedFulfillmentCount = chunks.count
        for (n, (k, rows)) in chunks.enumerated() {
            Thread { mt.prefill(mix: PrefillMix(key: k, from: 0, to: rows)) { events.with { $0.append("chunk:\(k)") } }; done.fulfill() }.start()
            waitForQueued(n + 1, on: mt)
        }
        hold.signal()
        wait(for: [done], timeout: 5)
        return events.value
    }

    func testQueuedChunksArePackedInOrderAndEveryBodyRunsAfterTheForward() {
        let events = Rec([String]())
        let mt = make(events: events, packRows: 8192)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(7, 2000), (8, 2000), (9, 2000)], events: events)
        XCTAssertEqual(ev, ["packed:[7, 8, 9]:[]", "chunk:7", "chunk:8", "chunk:9"])
        XCTAssertEqual(mt.packedStats.dispatches, 1)
        XCTAssertEqual(mt.packedStats.chunks, 3)
        XCTAssertEqual(mt.packedStats.sizes, [3: 1])
    }

    func testOffArmRunsEachChunkAlone() {
        // the reference: packRows 0 is the code path before T-0186 -- three ordinary chunk dispatches, no packed call
        let events = Rec([String]())
        let mt = make(events: events, packRows: 0)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(7, 2000), (8, 2000), (9, 2000)], events: events)
        XCTAssertEqual(ev, ["chunk:7", "chunk:8", "chunk:9"])
        XCTAssertEqual(mt.packedStats.dispatches, 0)
    }

    func testTheRowBudgetStopsAtTheFirstChunkThatDoesNotFit() {
        // 3000 + 3000 fit in 7000; the third (3000) does not, and the smaller fourth may not jump it (rank order is kept)
        let events = Rec([String]())
        let mt = make(events: events, packRows: 7000)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(1, 3000), (2, 3000), (3, 3000), (4, 500)], events: events)
        XCTAssertEqual(Array(ev.prefix(3)), ["packed:[1, 2]:[]", "chunk:1", "chunk:2"], "\(ev)")
        XCTAssertEqual(Array(ev.dropFirst(3)), ["packed:[3, 4]:[]", "chunk:3", "chunk:4"], "\(ev)")
    }

    func testPackMaxLimitsTheChunkCount() {
        let events = Rec([String]())
        let mt = make(events: events, packRows: 100_000, packMax: 2)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(1, 100), (2, 100), (3, 100)], events: events)
        XCTAssertEqual(ev, ["packed:[1, 2]:[]", "chunk:1", "chunk:2", "chunk:3"])
    }

    func testAChunkWiderThanTheBudgetRunsAlone() {
        let events = Rec([String]())
        let mt = make(events: events, packRows: 4096)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(1, 5000), (2, 1000)], events: events)
        // the picked chunk is always taken; the next one does not fit beside it
        XCTAssertEqual(ev, ["chunk:1", "chunk:2"])
    }

    func testOneRequestsChunksAreNeverPackedTogether() {
        // a request has one chunk in flight at a time; two queued jobs with the same key (a defensive case) are not packed
        let events = Rec([String]())
        let mt = make(events: events, packRows: 8192)
        defer { mt.shutdown() }
        let ev = runChunks(mt, [(5, 1000), (5, 1000), (6, 1000)], events: events)
        XCTAssertEqual(ev.first, "packed:[5, 6]:[]", "\(ev)")
        XCTAssertEqual(ev.filter { $0 == "chunk:5" }.count, 2)
    }

    func testTheBatchedStepsRideWithThePackedChunks() {
        // two decoders step twice (each resubmits right after its result); burst 1: after one decode dispatch the chunks are
        // due and carry the resubmitted steps in ONE packed call
        let events = Rec([String]())
        let mt = make(events: events, packRows: 8192, maxConsecutive: 1)
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let tokens = Rec([Int: [Int]]())
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 1); mt.enter(key: 2)
        for (n, k) in [1, 2].enumerated() {
            Thread {
                for _ in 0 ..< 2 { let r = mt.stepResult(key: k, pending: 0, length: 10); tokens.with { $0[k, default: []].append(r.token) } }
                done.fulfill()
            }.start()
            waitForQueued(n + 1, on: mt)
        }
        for k in [8, 9] {
            Thread { mt.prefill(mix: PrefillMix(key: k, from: 0, to: 1000)) { events.with { $0.append("chunk:\(k)") } }; done.fulfill() }.start()
            waitForQueued(k - 5, on: mt)
        }
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 1); mt.leave(key: 2)
        let ev = events.value
        // the two decoders resubmit in either order
        XCTAssertEqual(ev.count, 4, "\(ev)")
        XCTAssertEqual(ev.first, "step:[1, 2]", "\(ev)")
        XCTAssertTrue(["packed:[8, 9]:[1, 2]", "packed:[8, 9]:[2, 1]"].contains(ev[1]), "\(ev)")
        XCTAssertEqual(Array(ev.suffix(2)), ["chunk:8", "chunk:9"], "\(ev)")
        XCTAssertEqual(tokens.value, [1: [100, 102], 2: [200, 202]], "the second steps' tokens come from the packed call")
        XCTAssertEqual(mt.packedStats.withSteps, 1)
    }

    func testSetupClosuresRunBeforeAQueuedChunkOnlyWithPackingOn() {
        // a younger request's setup (cache lookup, MTP arming) runs ahead of the picked chunk, so its own chunk can be queued
        // in time to pack; packRows 0 keeps queue order (the reference)
        for (rows, want) in [(8192, ["setup:8", "chunk:7"]), (0, ["chunk:7", "setup:8"])] {
            let events = Rec([String]())
            let mt = make(events: events, packRows: rows)
            let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
            held.wait()
            let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 2
            Thread { mt.prefill(mix: PrefillMix(key: 7, from: 0, to: 1000)) { events.with { $0.append("chunk:7") } }; done.fulfill() }.start()
            waitForQueued(1, on: mt)
            Thread { mt.prefill(setup: true) { events.with { $0.append("setup:8") } }; done.fulfill() }.start()
            waitForQueued(2, on: mt)
            hold.signal()
            wait(for: [done], timeout: 5)
            XCTAssertEqual(events.value, want, "packRows \(rows)")
            XCTAssertEqual(mt.setupFirstCount, rows > 0 ? 1 : 0)
            mt.shutdown()
        }
    }

    /// Chunks 8 and 9 wait (mixed wait) for three decoders' steps; steps 1 and 2 arrive, then decoder 3's handover closure
    /// (after which decoder 3 submits its step). Returns the events.
    private func handoverScene(setup: Bool) -> [String] {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, mixedWait: 0.4, packRows: 8192,
                             packedStep: { mixes, reqs in
                                 events.with { $0.append("packed:\(mixes.map(\.key)):\(reqs.map(\.key).sorted())") }
                                 return reqs.map { $0.key }
                             },
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key)") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key).sorted())") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        for k in [1, 2, 3] { mt.enterDecode(key: k); mt.enter(key: k) }
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 5
        for k in [8, 9] {
            Thread { mt.prefill(mix: PrefillMix(key: k, from: 0, to: 1000)) { events.with { $0.append("chunk:\(k)") } }; done.fulfill() }.start()
            waitForQueued(k - 7, on: mt)
        }
        hold.signal()                                                  // the owner picks chunk 8 and waits for the steps
        Thread.sleep(forTimeInterval: 0.05)
        for k in [1, 2] { Thread { _ = mt.stepResult(key: k, pending: 0, length: 10); done.fulfill() }.start() }
        Thread.sleep(forTimeInterval: 0.03)
        Thread {
            mt.decode(traceLabel: "handover", traceKey: 3, setup: setup) { events.with { $0.append("handover:3") } }
            _ = mt.stepResult(key: 3, pending: 0, length: 10); done.fulfill()
        }.start()
        wait(for: [done], timeout: 5)
        for k in [1, 2, 3] { mt.leaveDecode(key: k); mt.leave(key: k) }
        return events.value
    }

    func testANewDecodersHandoverRunsFirstSoEveryStepRidesThePackedForward() {
        let ev = handoverScene(setup: true)
        XCTAssertEqual(ev, ["handover:3", "packed:[8, 9]:[1, 2, 3]", "chunk:8", "chunk:9"])
    }

    func testWithoutTheSetupFlagTheHandoverHoldsTheStepsBack() {
        // the reference: the handover is a barrier, the wait runs out short of the quorum and the chunks go without the steps
        let ev = handoverScene(setup: false)
        XCTAssertEqual(ev.first, "packed:[8, 9]:[]", "\(ev)")
        XCTAssertFalse(ev.contains { $0.hasPrefix("packed:") && $0.hasSuffix("[1, 2, 3]") }, "\(ev)")
    }

    /// T-0190 (QC audit R4): two requests' chunks in one 60 ms packed forward under a 50 ms prefill quantum. Each used
    /// 30 ms of it, so neither has used its quantum; charging each the whole forward rotated both.
    func testAPackedForwardChargesEachRequestItsShareOfTheQuantum() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true, prefillContinue: 0,
                             prefillQuantum: 0.05, packRows: 8192,
                             packedStep: { _, reqs in Thread.sleep(forTimeInterval: 0.06); return reqs.map { $0.key } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "chunks"); done.expectedFulfillmentCount = 2
        for (n, k) in [7, 8].enumerated() {
            Thread {
                mt.beginRequestPrefill()
                mt.prefill(mix: PrefillMix(key: k, from: 0, to: 1000)) { }
                mt.endRequestPrefill()
                done.fulfill()
            }.start()
            waitForQueued(n + 1, on: mt)
        }
        hold.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(mt.packedStats.dispatches, 1, "the two chunks were packed")
        XCTAssertEqual(mt.prefillOrderStats.rotations, 0, "each request used 30 of its 50 ms")
    }

    /// T-0192: request A (older) has its page chunk queued; request B (younger) has its vision tower queued, then a setup
    /// closure, then its chunk. With a tower lookahead of 1 the owner runs B's tower first, holds while B's setup and chunk
    /// arrive, and packs A and B into one forward; with 0 (today) A's chunk goes alone, B's later.
    private func lookaheadScene(_ k: Int) -> (events: [String], stats: (towers: Int, holds: Int, packed: Int)) {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true, prefillContinue: 0,
                             packRows: 8192, towerLookahead: k,
                             packedStep: { mixes, reqs in events.with { $0.append("packed:\(mixes.map(\.key))") }; return reqs.map { $0.key } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "requests"); done.expectedFulfillmentCount = 2
        Thread {
            mt.beginRequestPrefill()
            mt.prefill(mix: PrefillMix(key: 1, from: 0, to: 2000)) { events.with { $0.append("chunk:1") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        waitForQueued(1, on: mt)
        Thread {
            mt.beginRequestPrefill()
            mt.prefill(lookahead: true) { events.with { $0.append("tower:2") }; Thread.sleep(forTimeInterval: 0.005) }
            Thread.sleep(forTimeInterval: 0.002)          // the request thread's gap between its tower and its next job
            mt.prefill(setup: true) { events.with { $0.append("setup:2") } }
            mt.prefill(mix: PrefillMix(key: 2, from: 0, to: 2000)) { events.with { $0.append("chunk:2") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        waitForQueued(2, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        return (events.value, mt.lookaheadStats)
    }

    func testATowerLookaheadPacksTheNextPageWithTheOlderChunk() {
        let r = lookaheadScene(1)
        XCTAssertEqual(r.events, ["tower:2", "setup:2", "packed:[1, 2]", "chunk:1", "chunk:2"], "\(r.events)")
        XCTAssertEqual(r.stats.towers, 1); XCTAssertEqual(r.stats.packed, 1)
    }

    func testWithoutLookaheadTheOlderChunkGoesAlone() {
        // the reference (today's order): A's chunk first and alone, then B's tower, setup and chunk
        let r = lookaheadScene(0)
        XCTAssertEqual(r.events, ["chunk:1", "tower:2", "setup:2", "chunk:2"], "\(r.events)")
        XCTAssertEqual(r.stats.towers, 0)
    }

    /// T-0192 (P251 QC-3 rollback, C-0620): after a lookahead hold, the packed page chunks may still wait for the decoders'
    /// steps (ENGINE_MIXED_AFTER_CONTINUE, as after a continuation wait). The steps arrive 50 ms after release.
    private func lookaheadWithDecodersScene(afterContinue: Bool) -> [String] {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true, prefillContinue: 0,
                             mixedWait: 0.4, mixedAfterContinue: afterContinue, packRows: 8192, towerLookahead: 1,
                             packedStep: { mixes, reqs in
                                 events.with { $0.append("packed:\(mixes.map(\.key)):\(reqs.map(\.key).sorted())") }
                                 return reqs.map { $0.key } },
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key)") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key).sorted())") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        for k in [8, 9] { mt.enterDecode(key: k); mt.enter(key: k) }
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
        Thread {
            mt.beginRequestPrefill()
            mt.prefill(mix: PrefillMix(key: 1, from: 0, to: 2000)) { events.with { $0.append("chunk:1") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        waitForQueued(1, on: mt)
        Thread {
            mt.beginRequestPrefill()
            mt.prefill(lookahead: true) { events.with { $0.append("tower:2") }; Thread.sleep(forTimeInterval: 0.005) }
            mt.prefill(mix: PrefillMix(key: 2, from: 0, to: 2000)) { events.with { $0.append("chunk:2") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        waitForQueued(2, on: mt)
        hold.signal()
        Thread.sleep(forTimeInterval: 0.05)
        for k in [8, 9] { Thread { _ = mt.stepResult(key: k, pending: 0, length: 10); done.fulfill() }.start() }
        wait(for: [done], timeout: 5)
        for k in [8, 9] { mt.leaveDecode(key: k); mt.leave(key: k) }
        return events.value
    }

    func testAfterALookaheadHoldThePackedPagesStillWaitForTheSteps() {
        let ev = lookaheadWithDecodersScene(afterContinue: true)
        XCTAssertTrue(ev.contains("packed:[1, 2]:[8, 9]"), "the packed pages carried the decoders' steps: \(ev)")
    }

    func testWithoutTheAllowanceThePackedPagesGoWithoutTheSteps() {
        // the reference: no mixed wait after the hold (the P251 build behaved this way with the allowance on)
        let ev = lookaheadWithDecodersScene(afterContinue: false)
        XCTAssertTrue(ev.contains("packed:[1, 2]:[]"), "\(ev)")
    }

    func testLayoutSplitsChunksThenTheVerifyBlock() {
        Device.setDefault(device: .cpu)
        let L = Qwen4ExpPackedLayout(chunks: [3, 2], batch: 2, positions: 2)
        XCTAssertEqual(L.total, 9)
        XCTAssertEqual(L.segments, [3, 2, 4])
        let a = MLXArray((0 ..< 9).map { Float($0) }).reshaped([1, 9, 1])
        let (cs, v) = L.split(a)
        XCTAssertEqual(cs.map { $0.reshaped([-1]).asArray(Float.self) }, [[0, 1, 2], [3, 4]])
        XCTAssertEqual(v!.shape, [2, 2, 1])
        XCTAssertEqual(v!.reshaped([-1]).asArray(Float.self), [5, 6, 7, 8])
        XCTAssertEqual(L.join(cs, v).reshaped([-1]).asArray(Float.self), (0 ..< 9).map { Float($0) })
        let T = Qwen4ExpPackedLayout(chunks: [4])
        XCTAssertEqual(T.segments, [4]); XCTAssertNil(T.split(MLXArray.zeros([1, 4, 1])).verify)
    }
}
