// T-0191 (QC backlog): the owner's counters count what their names say. Each test fails with the pre-T-0191 counting
// (noted per test) -- a counter that lies is how a feature that never ran passed for "on" (C-0595, C-0606).
import Foundation
import XCTest

@testable import EngineServeSupport

private final class Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    @discardableResult func with<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&v) }
    var value: T { with { $0 } }
}

final class QCCounterTests: XCTestCase {
    private func waitForQueued(_ count: Int, on mt: ModelThread, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while mt.phaseStats().filter({ $0.key != "mixed" }).values.reduce(0, { $0 + $1.queued }) != count {
            if ProcessInfo.processInfo.systemUptime > deadline {
                XCTFail("expected \(count) queued jobs, got \(mt.phaseStats())", file: file, line: line); return
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    /// A mixed dispatch carries a prefill chunk and a decode step. The step is a DECODE job (it used to be charged to
    /// prefill: 41 % of "prefill jobs" in t0241), and the dispatch is a "mixed" dispatch.
    func testStepsRidingInAMixedDispatchAreDecodeJobs() {
        // a long mixed wait: it ends the moment the resubmitted step is queued, so the test does not depend on a 5 ms window
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 1, mixedWait: 5,
                             mixedStep: { _, reqs in reqs.map { $0.key } }) { reqs in reqs.map { $0.key } }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 2
        mt.enterDecode(key: 1); mt.enter(key: 1)
        Thread { _ = mt.stepResult(key: 1, pending: 0, length: 1); _ = mt.stepResult(key: 1, pending: 0, length: 2); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill(mix: PrefillMix(key: 7, from: 0, to: 2048)) { }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leave(key: 1)
        XCTAssertEqual(mt.mixedStats.dispatches, 1, "the second step rode with the chunk")
        let p = mt.phaseStats()
        XCTAssertEqual(p["prefill"]?.jobs, 1, "only the chunk is a prefill job: \(p)")
        XCTAssertEqual(p["decode"]?.jobs, 2, "both steps are decode jobs, the riding one too: \(p)")
        XCTAssertEqual(p["mixed"]?.dispatches, 1, "the dispatch that carried both is a mixed dispatch: \(p)")
        XCTAssertEqual(p["prefill"]?.dispatches, 0, "no prefill-only dispatch ran: \(p)")
    }

    /// One sequential prefill caller (20 ms chunks) and a decoder: with a 0.5 share the decoder gets steps past its burst,
    /// each counted as a share grant; with the share off there are none.
    func testShareGrantsCountTheDecodeDispatchesTheShareAloneAllowed() {
        func run(share: Double) -> (grants: Int, decodes: Int) {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 2,
                                 decodeAffinity: 1.0, decodeShare: share) { $0.map(\.key) }
            defer { mt.shutdown() }
            let decoding = Box(true), started = DispatchSemaphore(value: 0)
            let done = expectation(description: "callers"); done.expectedFulfillmentCount = 2
            Thread {
                var first = true
                while decoding.value {
                    mt.prefill { Thread.sleep(forTimeInterval: 0.02) }
                    if first { first = false; started.signal() }
                }
                done.fulfill()
            }.start()
            XCTAssertEqual(started.wait(timeout: .now() + 5), .success)
            Thread {
                mt.enterDecode()
                for _ in 0 ..< 30 { mt.decode { Thread.sleep(forTimeInterval: 0.001) }; Thread.sleep(forTimeInterval: 0.001) }
                mt.leaveDecode()
                decoding.with { $0 = false }
                done.fulfill()
            }.start()
            wait(for: [done], timeout: 30)
            return (mt.decodeShareStats.grants, mt.phaseStats()["decode"]?.jobs ?? 0)
        }
        let on = run(share: 0.5), off = run(share: 0)
        XCTAssertGreaterThan(on.grants, 0, "a 20 ms chunk at share 0.5 grants steps past the burst of 2")
        XCTAssertLessThan(on.grants, on.decodes)
        XCTAssertEqual(off.grants, 0, "no share, no grants")
    }

    /// An affinity wait ended by a SETUP decode closure (packing's handover) did not get the decoder a round: the next
    /// dispatch is the setup. It counts as preempted, not as a hit (the pre-T-0191 counter counted any decode job).
    func testAnAffinityWaitEndedByASetupClosureIsPreemptedNotAHit() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4,
                             decodeAffinity: 1.0, packRows: 8192, packedStep: { _, reqs in reqs.map { $0.key } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        let decoding = Box(true), started = DispatchSemaphore(value: 0)
        let done = expectation(description: "callers"); done.expectedFulfillmentCount = 2
        Thread {
            var first = true
            while decoding.value {
                mt.prefill { Thread.sleep(forTimeInterval: 0.01) }
                if first { first = false; started.signal() }
            }
            done.fulfill()
        }.start()
        XCTAssertEqual(started.wait(timeout: .now() + 5), .success)
        Thread {
            mt.enterDecode()
            // after each ordinary decode the owner waits for this decoder; half the time what comes is a setup closure
            for i in 0 ..< 24 {
                if i % 2 == 0 { mt.decode { } } else { mt.decode(traceLabel: "handover", setup: true) { } }
                Thread.sleep(forTimeInterval: 0.002)
            }
            mt.leaveDecode()
            decoding.with { $0 = false }
            done.fulfill()
        }.start()
        wait(for: [done], timeout: 30)
        let w = mt.decodeAffinityStats, s = mt.decodeShareStats
        XCTAssertGreaterThan(s.preempted, 0, "waits ended by a setup closure are preempted: waits \(w.waits) hits \(w.hits)")
        XCTAssertGreaterThan(w.hits, 0, "waits ended by an ordinary decode are hits")
        XCTAssertLessThanOrEqual(w.hits + s.preempted, w.waits)
    }

    /// Packing's setup-first: a 10 ms setup closure picked while a batched step waits is recorded as long and as ahead of
    /// steps -- the witness for "setup closures are short" (they run before everything else).
    func testSetupFirstRecordsLongSetupsAheadOfWaitingSteps() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, packRows: 8192,
                             packedStep: { _, reqs in reqs.map { $0.key } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 2
        mt.enterDecode(key: 1); mt.enter(key: 1)
        Thread { _ = mt.stepResult(key: 1, pending: 0, length: 1); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill(setup: true) { Thread.sleep(forTimeInterval: 0.01) }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        hold.signal()
        wait(for: [done], timeout: 5)
        mt.leaveDecode(key: 1); mt.leave(key: 1)
        let st = mt.setupFirstStats
        XCTAssertEqual(st.long, 1, "\(st)")
        XCTAssertEqual(st.aheadOfSteps, 1, "\(st)")
        XCTAssertGreaterThanOrEqual(st.maxSeconds, 0.009, "\(st)")
    }
}
