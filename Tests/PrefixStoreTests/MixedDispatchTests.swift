// T-0144: the owner's mixed dispatch -- a mixable prefill chunk taken together with the queued batched steps, the steps'
// round run through `mixedStep` (whose forward also computes the chunk), then the chunk's own body. Fake callbacks, CPU only.
import Foundation
import XCTest

@testable import EngineServeSupport

private final class Rec<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    @discardableResult func with<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&v) }
    var value: T { with { $0 } }
}

final class MixedDispatchTests: XCTestCase {
    private func waitForQueued(_ count: Int, on mt: ModelThread, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while mt.phaseStats().values.reduce(0, { $0 + $1.queued }) != count {
            if ProcessInfo.processInfo.systemUptime > deadline {
                XCTFail("expected \(count) queued jobs, got \(mt.phaseStats())", file: file, line: line); return
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    /// events: "step:<keys>" (an ordinary batched step), "mixed:<chunk key>:<from>:<keys>", "chunk:<key>" (a chunk body)
    private func make(events: Rec<[String]>, mixed: Bool, keep: Double = 0, wait: TimeInterval = 0.005,
                      share: Double = 0.5) -> ModelThread {
        ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, decodeAffinity: 0.002, decodeShare: share,
                    decodeShareMinWork: 0, mixedWait: wait, mixedDecodeShareKeep: keep,
                    mixedStep: mixed ? { mix, reqs in
                        events.with { $0.append("mixed:\(mix.key):\(mix.from):\(reqs.map(\.key))") }
                        Thread.sleep(forTimeInterval: 0.003)
                        return reqs.map { $0.key * 100 + 1 }
                    } : nil) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }
            Thread.sleep(forTimeInterval: 0.001)
            return reqs.map { $0.key * 100 }
        }
    }

    func testAMixableChunkTakesTheQueuedStepsAndItsBodyRunsAfterTheirRound() {
        let events = Rec([String]())
        let mt = make(events: events, mixed: true)
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let tokens = Rec([Int: Int]())
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 3
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 1); mt.enter(key: 2)
        for k in [1, 2] {
            Thread { let r = mt.stepResult(key: k, pending: 0, length: 10); tokens.with { $0[k] = r.token }; done.fulfill() }.start()
            waitForQueued(k, on: mt)
        }
        Thread { mt.prefill(mix: PrefillMix(key: 9, from: 0, to: 4096)) { events.with { $0.append("chunk:9") } }; done.fulfill() }.start()
        waitForQueued(3, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 1); mt.leave(key: 2)
        // decode has priority while its burst lasts: an ordinary step first, then the chunk with the resubmitted steps
        let ev = events.value
        XCTAssertTrue(ev.contains("mixed:9:0:[1, 2]") || ev.contains("step:[1, 2]"), "\(ev)")
        if let m = ev.firstIndex(where: { $0.hasPrefix("mixed:9") }) {
            XCTAssertEqual(ev[m + 1], "chunk:9", "the chunk body runs right after the mixed round: \(ev)")
        }
        XCTAssertEqual(ev.filter { $0 == "chunk:9" }.count, 1)
        XCTAssertEqual(Set(tokens.value.keys), [1, 2])
    }

    func testWithStepsQueuedAndNoDecodePriorityTheChunkIsMixed() {
        // burst 1: after one decode dispatch the prefill is due; the steps resubmitted meanwhile ride with it
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 1,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key):\(reqs.map(\.key))") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 3
        mt.enterDecode(key: 1); mt.enter(key: 1)
        Thread { _ = mt.stepResult(key: 1, pending: 0, length: 1); _ = mt.stepResult(key: 1, pending: 0, length: 2); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill(mix: PrefillMix(key: 7, from: 0, to: 2048)) { events.with { $0.append("chunk:7") } }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        Thread { mt.prefill { events.with { $0.append("plain") } }; done.fulfill() }.start()
        waitForQueued(3, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leave(key: 1)
        let ev = events.value
        XCTAssertEqual(ev.first, "step:[1]", "\(ev)")
        XCTAssertTrue(ev.contains("mixed:7:[1]"), "the second step rode with the chunk: \(ev)")
        XCTAssertEqual(mt.mixedStats.dispatches, 1); XCTAssertEqual(mt.mixedStats.rows, 1)
    }

    func testWithoutStepsOrWithoutTheCallbackTheChunkIsAnOrdinaryPrefill() {
        for mixed in [true, false] {
            let events = Rec([String]())
            let mt = make(events: events, mixed: mixed)
            mt.prefill(mix: PrefillMix(key: 3, from: 0, to: 10)) { events.with { $0.append("chunk:3") } }
            XCTAssertEqual(events.value, ["chunk:3"])
            XCTAssertEqual(mt.mixedStats.dispatches, 0)
            XCTAssertEqual(mt.mixedStats.waits, 0, "no decoder registered: no hold for steps")
            mt.shutdown()
        }
    }

    func testCancelledStepsAreNotCarriedAndTheChunkStillRuns() {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 1,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(reqs.map(\.key))") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 3
        let cancelled = Rec([StepResult]())
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 2)
        // a decode closure first so the prefill is due after it (burst 1), then a live step and a cancelled one
        Thread { mt.decode { events.with { $0.append("decode") } }; done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill(mix: PrefillMix(key: 5, from: 0, to: 8)) { events.with { $0.append("chunk:5") } }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        Thread { let r = mt.stepResult(key: 2, pending: 0, length: 1, cancelled: { true }); cancelled.with { $0.append(r) }; done.fulfill() }.start()
        waitForQueued(3, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 2)
        XCTAssertEqual(cancelled.value.first?.cancelled, true)
        XCTAssertFalse(events.value.contains { $0.contains("2]") && $0.hasPrefix("mixed") }, "\(events.value)")
        XCTAssertTrue(events.value.contains("chunk:5"), "\(events.value)")
    }

    /// C-0425: a partial step set is never mixed -- with two batched rows and only one step queued (the other never comes)
    /// the chunk runs as an ordinary prefill after the bounded wait, and the step keeps its own path.
    func testAPartialStepSetIsNotMixed() {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 1, mixedWait: 0.01,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(reqs.map(\.key))") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 3
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 1); mt.enter(key: 2)
        Thread { mt.decode { events.with { $0.append("decode") } }; done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill(mix: PrefillMix(key: 5, from: 0, to: 8)) { events.with { $0.append("chunk:5") } }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        Thread { _ = mt.stepResult(key: 1, pending: 0, length: 1); done.fulfill() }.start()
        waitForQueued(3, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 1); mt.leave(key: 2)
        let ev = events.value
        XCTAssertFalse(ev.contains { $0.hasPrefix("mixed") }, "\(ev)")
        XCTAssertTrue(ev.contains("chunk:5") && ev.contains("step:[1]"), "\(ev)")
        XCTAssertEqual(mt.mixedStats.dispatches, 0)
        XCTAssertGreaterThanOrEqual(mt.mixedStats.waits, 1)
    }

    // MARK: T-0148 -- a lone row's serial MTP round rides with a chunk

    /// queue after the held owner: a plain decode closure X (uses the burst of 1), the chunk, then the lone row's round D
    private func serialRun(mixable: Bool, callback: Bool, steppableRows: [Int] = [], steps: [Int] = []) -> ([String], ModelThread) {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 2, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 1, mixedWait: 0.01,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(reqs.map(\.key))") }; return reqs.map { $0.key } },
                             mixedSerial: callback ? { mix, round in events.with { $0.append("serial:\(mix.key)") }; round() } : nil) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let jobs = 3 + steps.count
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = jobs
        mt.enterDecode(key: 7)
        for k in steppableRows { mt.enter(key: k); mt.enterDecode(key: k) }
        var queued = 0
        Thread { mt.decode { events.with { $0.append("X") } }; done.fulfill() }.start()
        queued += 1; waitForQueued(queued, on: mt)
        for k in steps {
            Thread { _ = mt.stepResult(key: k, pending: 0, length: 1); done.fulfill() }.start()
            queued += 1; waitForQueued(queued, on: mt)
        }
        Thread { mt.prefill(mix: PrefillMix(key: 9, from: 0, to: 64)) { events.with { $0.append("chunk:9") } }; done.fulfill() }.start()
        queued += 1; waitForQueued(queued, on: mt)
        Thread { mt.decode(mixable: mixable) { events.with { $0.append("round:7") } }; done.fulfill() }.start()
        queued += 1; waitForQueued(queued, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 7)
        for k in steppableRows { mt.leave(key: k); mt.leaveDecode(key: k) }
        return (events.value, mt)
    }

    func testALoneMixableRoundRidesWithTheChunkAndRunsBeforeItsBody() {
        let (ev, mt) = serialRun(mixable: true, callback: true)
        defer { mt.shutdown() }
        XCTAssertEqual(ev, ["X", "serial:9", "round:7", "chunk:9"])
        XCTAssertEqual(mt.mixedStats.serial, 1); XCTAssertEqual(mt.mixedStats.dispatches, 1)
    }

    func testANonMixableRoundOrNoCallbackIsNeverCarried() {
        for (mixable, callback) in [(false, true), (true, false)] {
            let (ev, mt) = serialRun(mixable: mixable, callback: callback)
            XCTAssertFalse(ev.contains { $0.hasPrefix("serial") }, "mixable \(mixable) callback \(callback): \(ev)")
            XCTAssertEqual(Set(ev), ["X", "chunk:9", "round:7"])
            XCTAssertEqual(mt.mixedStats.serial, 0)
            mt.shutdown()
        }
    }

    func testWithBatchedRowsTheQuorumPathIsUsedNotTheSerialOne() {
        let (ev, mt) = serialRun(mixable: true, callback: true, steppableRows: [1, 2], steps: [1, 2])
        defer { mt.shutdown() }
        XCTAssertFalse(ev.contains { $0.hasPrefix("serial") }, "\(ev)")
        XCTAssertTrue(ev.contains("mixed:[1, 2]") || ev.contains("step:[1, 2]"), "\(ev)")
        XCTAssertEqual(mt.mixedStats.serial, 0)
    }

    /// keep = 0: after a mixed dispatch no separate decode round runs while another mixable chunk is queued -- the decoders
    /// advance inside the chunks. keep = 1: the separate share returns.
    func testTheDecodeShareAfterAMixedDispatchFollowsKeep() {
        func run(keep: Double) -> [String] {
            let events = Rec([String]())
            let mt = make(events: events, mixed: true, keep: keep, wait: 0.05, share: 0.5)
            defer { mt.shutdown() }
            let decoding = Rec(true), started = DispatchSemaphore(value: 0)
            let done = expectation(description: "decoder and chunks"); done.expectedFulfillmentCount = 2
            mt.enterDecode(key: 1); mt.enter(key: 1)
            Thread {
                started.signal()
                while decoding.value { _ = mt.stepResult(key: 1, pending: 0, length: 1) }
                done.fulfill()
            }.start()
            started.wait()
            Thread.sleep(forTimeInterval: 0.02)
            Thread {
                for c in 0 ..< 4 {
                    mt.prefill(mix: PrefillMix(key: 9, from: c * 100, to: c * 100 + 100, continues: c < 3)) {
                        events.with { $0.append("chunk") }; Thread.sleep(forTimeInterval: 0.02)
                    }
                }
                decoding.with { $0 = false }
                done.fulfill()
            }.start()
            wait(for: [done], timeout: 10)
            mt.leaveDecode(key: 1); mt.leave(key: 1)
            return events.value
        }
        func between(_ ev: [String]) -> [Int] {
            // ordinary steps between consecutive chunk bodies
            var out: [Int] = [], n = 0, seen = false
            for e in ev {
                if e == "chunk" { if seen { out.append(n) }; seen = true; n = 0 } else if e.hasPrefix("step") { n += 1 }
            }
            return out
        }
        let closed = run(keep: 0), open = run(keep: 1)
        XCTAssertTrue(closed.filter { $0.hasPrefix("mixed") }.count >= 3, "\(closed)")
        XCTAssertTrue(between(closed).allSatisfy { $0 == 0 }, "keep 0: no separate step between chunks: \(closed)")
        XCTAssertGreaterThan(between(open).reduce(0, +), 0, "keep 1: the share lets separate steps run: \(open)")
    }

    // MARK: T-0185

    /// the first test's scene (two queued steps, a queued mixable chunk, decode share 0.5 and a decode burst left): today an
    /// ordinary step may run first; with mixedAll the chunk takes the steps before any separate round
    private func sceneMixAll(_ mixAll: Bool) -> [String] {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4, decodeAffinity: 0.002,
                             decodeShare: 0.5, decodeShareMinWork: 0, mixedWait: 0.02, mixedAll: mixAll,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key):\(reqs.map(\.key))") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 3
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 1); mt.enter(key: 2)
        for k in [1, 2] {
            Thread { _ = mt.stepResult(key: k, pending: 0, length: 10); done.fulfill() }.start()
            waitForQueued(k, on: mt)
        }
        Thread { mt.prefill(mix: PrefillMix(key: 9, from: 0, to: 2048)) { events.with { $0.append("chunk:9") } }; done.fulfill() }.start()
        waitForQueued(3, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 1); mt.leave(key: 2)
        return events.value
    }

    func testMixAllTakesTheChunkWithTheStepsBeforeAnySeparateRound() {
        XCTAssertEqual(Array(sceneMixAll(true).prefix(2)), ["mixed:9:[1, 2]", "chunk:9"])
    }

    func testWithoutMixAllTheDecodeBurstRunsAnOrdinaryStepFirst() {
        // the shipped rule, the reference: decode has priority while its burst and share last
        XCTAssertEqual(sceneMixAll(false).first, "step:[1, 2]")
    }

    /// C-0587's scene: under arrival order a younger request's job is queued while the oldest request is between jobs, so
    /// the owner holds for the oldest (continuation wait); the oldest's chunk arrives, then the decoders' steps a little
    /// later. The re-entered pick used to forbid the mixed wait; with mixedAfterContinue the chunk still waits for them.
    private func sceneAfterContinue(_ fix: Bool) -> [String] {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true, prefillContinue: 0.3,
                             mixedWait: 0.3, mixedAfterContinue: fix,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key):\(reqs.map(\.key))") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key))") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
        mt.enterDecode(key: 1); mt.enterDecode(key: 2); mt.enter(key: 1); mt.enter(key: 2)
        let aStamped = DispatchSemaphore(value: 0), aGo = DispatchSemaphore(value: 0)
        Thread {                                                     // the oldest request, between its jobs at first
            mt.beginRequestPrefill(); aStamped.signal(); aGo.wait()
            mt.prefill(mix: PrefillMix(key: 5, from: 0, to: 2048)) { events.with { $0.append("chunk:5") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        aStamped.wait()
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        Thread { mt.beginRequestPrefill(); mt.prefill { events.with { $0.append("young") } }; mt.endRequestPrefill(); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        hold.signal()                                                // the owner now holds for the oldest (A)
        Thread.sleep(forTimeInterval: 0.05); aGo.signal()            // A's chunk arrives inside the continuation wait
        Thread.sleep(forTimeInterval: 0.08)                          // the decoders' steps arrive inside the mixed wait
        for k in [1, 2] { Thread { _ = mt.stepResult(key: k, pending: 0, length: 10); done.fulfill() }.start() }
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leaveDecode(key: 2); mt.leave(key: 1); mt.leave(key: 2)
        return events.value
    }

    func testAfterAContinuationWaitTheChunkStillWaitsForTheSteps() {
        // the two steps are submitted from two threads at once: either order
        let first = sceneAfterContinue(true).first
        XCTAssertTrue(first == "mixed:5:[1, 2]" || first == "mixed:5:[2, 1]", "\(first ?? "-")")
    }

    func testWithoutTheFixTheChunkAfterAContinuationWaitRunsAlone() {
        // C-0587, the regression reference
        XCTAssertEqual(sceneAfterContinue(false).first, "chunk:5")
    }

    /// T-0190 (QC audit D1): chunk 9 waits for three decoders' steps; steps 1 and 2 arrive, then decoder 3 queues a decode
    /// closure (e.g. its step installer) and only after it its step. The closure is a barrier the steps cannot pass and only
    /// the owner can run it, so the wait could only run out.
    private func unblockScene(_ unblock: Bool) -> [String] {
        let events = Rec([String]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, mixedWait: 0.4, mixedUnblock: unblock,
                             mixedStep: { mix, reqs in events.with { $0.append("mixed:\(mix.key):\(reqs.map(\.key).sorted())") }; return reqs.map { $0.key } }) { reqs in
            events.with { $0.append("step:\(reqs.map(\.key).sorted())") }; return reqs.map { $0.key }
        }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        for k in [1, 2, 3] { mt.enterDecode(key: k); mt.enter(key: k) }
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
        Thread { mt.prefill(mix: PrefillMix(key: 9, from: 0, to: 1000)) { events.with { $0.append("chunk:9") } }; done.fulfill() }.start()
        waitForQueued(1, on: mt)
        let t0 = Date()
        hold.signal()
        Thread.sleep(forTimeInterval: 0.05)
        for k in [1, 2] { Thread { _ = mt.stepResult(key: k, pending: 0, length: 10); done.fulfill() }.start() }
        Thread.sleep(forTimeInterval: 0.03)
        Thread {
            mt.decode(traceLabel: "install", traceKey: 3) { events.with { $0.append("closure:3") } }
            _ = mt.stepResult(key: 3, pending: 0, length: 10); done.fulfill()
        }.start()
        wait(for: [done], timeout: 5)
        events.with { $0.append(String(format: "elapsed:%.2f", Date().timeIntervalSince(t0))) }
        for k in [1, 2, 3] { mt.leaveDecode(key: k); mt.leave(key: k) }
        XCTAssertEqual(mt.unblockBreaksCount, unblock ? 1 : 0)
        return events.value
    }

    func testUnblockRunsTheClosureAndTheChunkThenTakesEveryStep() {
        let ev = unblockScene(true)
        XCTAssertEqual(Array(ev.prefix(3)), ["closure:3", "mixed:9:[1, 2, 3]", "chunk:9"], "\(ev)")
        let elapsed = Double(ev.last!.dropFirst(8))!
        XCTAssertLessThan(elapsed, 0.3, "the wait stopped at the closure instead of running out (0.4 s): \(ev)")
    }

    func testWithoutUnblockTheWaitRunsOutAndTheChunkGoesWithoutTheSteps() {
        // the reference: the pre-T-0190 behaviour
        let ev = unblockScene(false)
        XCTAssertEqual(ev.first, "chunk:9", "\(ev)")
        XCTAssertFalse(ev.contains { $0.hasPrefix("mixed:9") }, "\(ev)")
    }
}
