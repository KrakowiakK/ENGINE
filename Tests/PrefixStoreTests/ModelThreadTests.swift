// P089 U10 -- the owner thread, tested with a fake step. Every property here is one whose failure is
// silent in production: work done on the wrong thread corrupts the heap minutes later in an
// unrelated frame, a lost wakeup hangs a client, and a job counted twice hands a request somebody
// else's token.
import Foundation
import XCTest

@testable import EngineServeSupport

/// A lock-guarded box. Swift 6 refuses a mutable capture in an escaping `Thread` closure, and a
/// box makes the synchronisation explicit anyway.
private final class Shared<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    @discardableResult func with<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&v) }
    var value: T { with { $0 } }
}

final class ModelThreadTests: XCTestCase {
    /// Build a thread whose "batched step" records what it was handed and answers key*1000 + size.
    private func makeThread(minBatch: Int, window: TimeInterval = 0.05, maxBatch: Int = 8,
                            onBatch: (([StepRequest]) -> Void)? = nil) -> ModelThread {
        ModelThread(minBatch: minBatch, gatherWindow: window, maxBatch: maxBatch) { reqs in
            onBatch?(reqs)
            return reqs.map { $0.key * 1000 + reqs.count }
        }
    }

    // MARK: the ownership rule -- the reason this type exists

    func testEveryJobRunsOnOneThreadAndItIsNotTheCaller() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let threads = Shared(Set<ObjectIdentifier>())
        let sawCaller = Shared(false)
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 8
        for _ in 0 ..< 8 {
            Thread {
                let caller = ObjectIdentifier(Thread.current)
                for _ in 0 ..< 50 {
                    mt.exclusive {
                        let here = ObjectIdentifier(Thread.current)
                        threads.with { $0.insert(here) }
                        if here == caller { sawCaller.with { $0 = true } }
                    }
                }
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 60)
        XCTAssertEqual(threads.value.count, 1, "work ran on \(threads.value.count) threads; the whole point is one")
        XCTAssertFalse(sawCaller.value, "a job ran on the submitting thread")
    }

    func testNoTwoJobsOverlap() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let inside = Shared(0), maxInside = Shared(0)
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 6
        for _ in 0 ..< 6 {
            Thread {
                for _ in 0 ..< 40 {
                    mt.exclusive {
                        let n = inside.with { $0 += 1; return $0 }
                        maxInside.with { $0 = max($0, n) }
                        Thread.sleep(forTimeInterval: 0.0005)
                        inside.with { $0 -= 1 }
                    }
                }
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 90)
        XCTAssertEqual(maxInside.value, 1, "two jobs were inside the owner at once")
    }

    func testIsOwnerIsTrueInsideAndAReentrantCallDoesNotDeadlock() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        XCTAssertFalse(mt.isOwner)
        let nested: Int = mt.exclusive {
            XCTAssertTrue(mt.isOwner)
            return mt.exclusive { 42 }      // a job that submits more work runs it inline
        }
        XCTAssertEqual(nested, 42)
    }

    func testResultsComeBackToTheRightCaller() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 16
        for i in 0 ..< 16 {
            Thread {
                for _ in 0 ..< 25 {
                    XCTAssertEqual(mt.exclusive { i * 7 }, i * 7, "a caller was handed another's result")
                }
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 60)
    }

    func testJobsAreServedInArrivalOrder() {
        // FIFO is the property an agent fleet feels: a long prefill (submitted chunk by chunk)
        // cannot starve a short request, because each chunk goes to the back of the queue.
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let order = Shared([Int]())
        let blocked = expectation(description: "owner busy")
        let release = expectation(description: "may finish")
        Thread { mt.exclusive { blocked.fulfill(); self.wait(for: [release], timeout: 20) } }.start()
        wait(for: [blocked], timeout: 20)          // the owner is now occupied; everyone queues behind
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 8
        for i in 0 ..< 8 {
            Thread { mt.exclusive { order.with { $0.append(i) } }; done.fulfill() }.start()
            Thread.sleep(forTimeInterval: 0.02)    // a clear arrival order
        }
        release.fulfill()
        wait(for: [done], timeout: 30)
        XCTAssertEqual(order.value, Array(0 ..< 8), "jobs were not served in arrival order")
    }

    // MARK: batching -- what replaced the rendezvous

    func testEveryStepIsRunExactlyOnceAndReturnsItsOwnToken() {
        let seen = Shared([Int: Int]())
        let mt = makeThread(minBatch: 4, window: 0.2) { reqs in
            seen.with { d in for r in reqs { d[r.key, default: 0] += 1 } }
        }
        defer { mt.shutdown() }
        for _ in 0 ..< 4 { mt.enter() }
        let rounds = 30
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 4
        for k in 1 ... 4 {
            Thread {
                for _ in 0 ..< rounds {
                    let tok = mt.step(key: k, pending: 7, length: 4096)
                    XCTAssertEqual(tok / 1000, k, "row \(k) was handed another row's token")
                }
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 90)
        for _ in 0 ..< 4 { mt.leave() }
        let counts = seen.value
        for k in 1 ... 4 { XCTAssertEqual(counts[k], rounds, "row \(k) ran \(counts[k] ?? 0) times, expected \(rounds)") }
    }

    func testAGroupFormsWhenAQuorumIsDeclared() {
        let widest = Shared(0)
        let mt = makeThread(minBatch: 3, window: 1.0) { reqs in
            widest.with { $0 = max($0, reqs.count) }
        }
        defer { mt.shutdown() }
        for _ in 0 ..< 3 { mt.enter() }
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 3
        for k in 1 ... 3 {
            Thread {
                Thread.sleep(forTimeInterval: Double(k) * 0.03)     // a spread in host time
                _ = mt.step(key: k, pending: 1, length: 4096)
                done.fulfill()
            }.start()
        }
        wait(for: [done], timeout: 30)
        XCTAssertEqual(widest.value, 3, "the gather window must coalesce a declared quorum")
    }

    /// P089 -- a lone eligible row used to pay the whole gather window on EVERY token: 25 ms per step
    /// caps decode at 40 tok/s before the step itself is counted, and the P078 fleet measured 14.4
    /// tok/s on a row that had the GPU almost to itself. The window may only be paid when `minBatch`
    /// rows have actually declared themselves steppable.
    func testALoneSteppableRowDoesNotPayTheGatherWindow() {
        let mt = makeThread(minBatch: 4, window: 1.0)
        defer { mt.shutdown() }
        mt.enter()                               // one row is steppable; the other three are not
        defer { mt.leave() }
        let t0 = Date()
        _ = mt.step(key: 1, pending: 1, length: 4096)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 0.2,
                          "a row with no possible company waited for the window anyway")
    }

    func testTheGroupIsCappedAndTheOverflowIsServedNext() {
        let widest = Shared(0)
        let mt = makeThread(minBatch: 2, window: 0.1, maxBatch: 3) { reqs in
            widest.with { $0 = max($0, reqs.count) }
        }
        defer { mt.shutdown() }
        for _ in 0 ..< 6 { mt.enter() }
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 6
        for k in 1 ... 6 {
            Thread { _ = mt.step(key: k, pending: 1, length: 4096); done.fulfill() }.start()
        }
        wait(for: [done], timeout: 30)
        for _ in 0 ..< 6 { mt.leave() }
        XCTAssertLessThanOrEqual(widest.value, 3, "a group wider than maxBatch was handed out")
        XCTAssertGreaterThanOrEqual(widest.value, 2, "nothing coalesced at all")
    }

    /// Coalescing may only look at the FRONT of the queue. Taking step jobs from behind an exclusive
    /// job would reorder the queue and lose the fairness the previous test fixes.
    func testAnExclusiveJobIsNotOvertakenByLaterSteps() {
        let order = Shared([String]())
        let mt = makeThread(minBatch: 2, window: 0.1) { reqs in
            order.with { $0.append("step" + reqs.map { String($0.key) }.joined()) }
        }
        defer { mt.shutdown() }
        for _ in 0 ..< 3 { mt.enter() }
        let blocked = expectation(description: "busy"); let release = expectation(description: "go")
        Thread { mt.exclusive { blocked.fulfill(); self.wait(for: [release], timeout: 20) } }.start()
        wait(for: [blocked], timeout: 20)
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 3
        Thread { _ = mt.step(key: 1, pending: 1, length: 4096); done.fulfill() }.start()
        Thread.sleep(forTimeInterval: 0.05)
        Thread { mt.exclusive { order.with { $0.append("excl") } }; done.fulfill() }.start()
        Thread.sleep(forTimeInterval: 0.05)
        Thread { _ = mt.step(key: 2, pending: 1, length: 4096); done.fulfill() }.start()
        Thread.sleep(forTimeInterval: 0.05)
        release.fulfill()
        wait(for: [done], timeout: 30)
        for _ in 0 ..< 3 { mt.leave() }
        XCTAssertEqual(order.value, ["step1", "excl", "step2"],
                       "the queue was reordered: \(order.value)")
    }

    func testEnterAndLeaveBalanceAndNeverGoNegative() {
        let mt = makeThread(minBatch: 2)
        defer { mt.shutdown() }
        XCTAssertEqual(mt.steppableCount, 0)
        mt.enter(); mt.enter()
        XCTAssertEqual(mt.steppableCount, 2)
        mt.leave(); mt.leave(); mt.leave()          // one extra: a double-unregister must not underflow
        XCTAssertEqual(mt.steppableCount, 0)
        let done = expectation(description: "done"); done.expectedFulfillmentCount = 16
        for _ in 0 ..< 16 {
            Thread { for _ in 0 ..< 200 { mt.enter(); mt.leave() }; done.fulfill() }.start()
        }
        wait(for: [done], timeout: 60)
        XCTAssertEqual(mt.steppableCount, 0, "the counter must survive concurrent registration")
    }
}


extension ModelThreadTests {
    /// Observe actual enqueue completion while the owner is held. Ordering assertions below do
    /// not depend on a sleep being long enough for a newly started submitter to reach the queue.
    private func waitForQueued(_ count: Int, on mt: ModelThread,
                               file: StaticString = #filePath, line: UInt = #line) {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while mt.phaseStats().values.reduce(0, { $0 + $1.queued }) != count {
            if ProcessInfo.processInfo.systemUptime > deadline {
                XCTFail("expected \(count) queued jobs, got \(mt.phaseStats())", file: file, line: line)
                return
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    func testDecodePassesPrefillAndPhaseClosuresRemainOnOwner() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let done = expectation(description: "phase jobs"); done.expectedFulfillmentCount = 4
        for (i, label) in ["p0", "d0", "p1", "d1"].enumerated() {
            Thread {
                let work = {
                    XCTAssertTrue(mt.isOwner)
                    order.with { $0.append(label) }
                    return mt.exclusive { mt.prefill { mt.decode { 42 } } }
                }
                let result = label.hasPrefix("p") ? mt.prefill(work) : mt.decode(work)
                XCTAssertEqual(result, 42)
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["d0", "d1", "p0", "p1"])
        let stats = mt.phaseStats()
        XCTAssertEqual(stats["decode"]?.jobs, 2)
        XCTAssertEqual(stats["decode"]?.dispatches, 2)
        XCTAssertEqual(stats["prefill"]?.jobs, 2)
        XCTAssertEqual(stats["control"]?.jobs, 1, "nested closures execute inline")
        for phase in ["control", "prefill", "decode"] {
            XCTAssertEqual(stats[phase]?.queued, 0)
            XCTAssertGreaterThan(stats[phase]!.queueWaitSeconds, 0)
            XCTAssertGreaterThan(stats[phase]!.workSeconds, 0)
            XCTAssertLessThanOrEqual(stats[phase]!.maxWorkSeconds, stats[phase]!.workSeconds)
        }
    }

    func testControlBarrierSeparatesPriorityRegions() {
        let mt = makeThread(minBatch: 1)
        defer { mt.shutdown() }
        let order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let done = expectation(description: "barrier jobs"); done.expectedFulfillmentCount = 5
        for (i, label) in ["p0", "d0", "control", "p1", "d1"].enumerated() {
            Thread {
                let work = { order.with { $0.append(label) } }
                if label == "control" { mt.exclusive(work) }
                else if label.hasPrefix("p") { mt.prefill(work) }
                else { mt.decode(work) }
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["d0", "p0", "control", "d1", "p1"])
    }

    func testWaitingPrefillProgressIsBoundedByDecodeDispatchCount() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             maxConsecutiveDecodeDispatches: 2) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let labels = ["p0", "p1"] + (0..<6).map { "d\($0)" }
        let done = expectation(description: "bounded jobs"); done.expectedFulfillmentCount = labels.count
        for (i, label) in labels.enumerated() {
            Thread {
                let work = { order.with { $0.append(label) } }
                if label.hasPrefix("p") { mt.prefill(work) } else { mt.decode(work) }
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["d0", "d1", "p0", "d2", "d3", "p1", "d4", "d5"])
    }

    /// Decode affinity: a decoder that resubmits 1 ms after each result, while two prefill callers keep a chunk waiting.
    /// Off (the shipped behaviour) it gets one dispatch per prefill; on, it gets the whole burst before the next prefill.
    func testDecodeAffinityLetsADecoderUseItsBurstWhilePrefillWaits() {
        func run(affinity: TimeInterval) -> (runs: [Int], stats: (waits: Int, hits: Int)) {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4,
                                 decodeAffinity: affinity) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Shared([Character]()), decoding = Shared(true)
            let done = expectation(description: "decoder and prefill callers"); done.expectedFulfillmentCount = 3
            for _ in 0..<2 {
                Thread {
                    while decoding.value { mt.prefill { order.with { $0.append("p") }; Thread.sleep(forTimeInterval: 0.01) } }
                    done.fulfill()
                }.start()
            }
            waitForQueued(1, on: mt)
            Thread {
                mt.enterDecode()
                for _ in 0..<16 { mt.decode { order.with { $0.append("d") } }; Thread.sleep(forTimeInterval: 0.001) }
                mt.leaveDecode()
                decoding.with { $0 = false }
                done.fulfill()
            }.start()
            wait(for: [done], timeout: 20)
            // lengths of the decode runs that sit between two prefills
            let text = String(order.value), parts = text.split(separator: "p", omittingEmptySubsequences: false)
            let interior = parts.dropFirst().dropLast().map(\.count).filter { $0 > 0 }
            return (interior, mt.decodeAffinityStats)
        }
        // thread start-up can leave a moment with no prefill queued, so the shape is asserted on most runs, the burst
        // bound on every run
        let off = run(affinity: 0)
        XCTAssertFalse(off.runs.isEmpty)
        XCTAssertTrue(off.runs.allSatisfy { $0 <= 4 }, "the burst bound holds: \(off.runs)")
        XCTAssertGreaterThanOrEqual(off.runs.filter { $0 == 1 }.count * 10, off.runs.count * 7,
                                    "without affinity a resubmitting decoder gets one dispatch per prefill: \(off.runs)")
        XCTAssertEqual(off.stats.waits, 0); XCTAssertEqual(off.stats.hits, 0)
        let on = run(affinity: 1.0)
        XCTAssertFalse(on.runs.isEmpty)
        XCTAssertTrue(on.runs.allSatisfy { $0 <= 4 }, "the burst bound holds: \(on.runs)")
        XCTAssertGreaterThanOrEqual(on.runs.filter { $0 == 4 }.count * 10, on.runs.count * 7,
                                    "with affinity it uses the burst of 4 between prefills: \(on.runs)")
        XCTAssertGreaterThan(on.stats.hits, 0); XCTAssertLessThanOrEqual(on.stats.hits, on.stats.waits)
    }

    /// The server's shape: ONE long prefill that resubmits its next chunk only after the previous ends, and one decoder.
    /// Without affinity the decoder gets one dispatch per chunk; with it, the burst -- even though no prefill is queued at
    /// the moment the first decode after a chunk is taken (the case the first version missed).
    func testDecodeAffinityWorksWithASequentialPrefillCaller() {
        func run(affinity: TimeInterval) -> [Int] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4,
                                 decodeAffinity: affinity) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Shared([Character]()), decoding = Shared(true), started = DispatchSemaphore(value: 0)
            let done = expectation(description: "decoder and one prefill caller"); done.expectedFulfillmentCount = 2
            Thread {
                // resubmits at once, but only after its chunk ended: at the decode taken right after a chunk the next chunk
                // is usually not queued yet, which is what made the first version's counter read 0
                var first = true
                while decoding.value {
                    mt.prefill { order.with { $0.append("p") }; Thread.sleep(forTimeInterval: 0.01) }
                    if first { first = false; started.signal() }
                }
                done.fulfill()
            }.start()
            XCTAssertEqual(started.wait(timeout: .now() + 5), .success)
            Thread {
                mt.enterDecode()
                for _ in 0..<16 { mt.decode { order.with { $0.append("d") } }; Thread.sleep(forTimeInterval: 0.001) }
                mt.leaveDecode()
                decoding.with { $0 = false }
                done.fulfill()
            }.start()
            wait(for: [done], timeout: 20)
            let parts = String(order.value).split(separator: "p", omittingEmptySubsequences: false)
            return parts.dropFirst().dropLast().map(\.count).filter { $0 > 0 }
        }
        let off = run(affinity: 0), on = run(affinity: 1.0)
        XCTAssertFalse(off.isEmpty); XCTAssertFalse(on.isEmpty)
        XCTAssertGreaterThanOrEqual(off.filter { $0 == 1 }.count * 10, off.count * 7, "without affinity: \(off)")
        XCTAssertGreaterThanOrEqual(on.filter { $0 >= 4 }.count * 10, on.count * 7, "with affinity the burst is used: \(on)")
    }

    /// Time share: with one sequential prefill caller and a decoder that resubmits 1 ms after each 1 ms step, a long chunk
    /// (20 ms, share 0.5) leaves room for several decode steps per chunk, a short chunk (1 ms) for none -- the short-context
    /// behaviour stays the shipped one, where a fixed burst of 16 had starved prefill.
    func testDecodeShareScalesTheBurstWithThePrefillChunk() {
        func run(chunk: TimeInterval, minWork: Double = 0) -> [Int] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4,
                                 decodeAffinity: 1.0, decodeShare: 0.5, decodeShareMinWork: minWork) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Shared([Character]()), decoding = Shared(true), started = DispatchSemaphore(value: 0)
            let done = expectation(description: "decoder and one prefill caller"); done.expectedFulfillmentCount = 2
            Thread {
                var first = true
                while decoding.value {
                    mt.prefill { order.with { $0.append("p") }; Thread.sleep(forTimeInterval: chunk) }
                    if first { first = false; started.signal() }
                }
                done.fulfill()
            }.start()
            XCTAssertEqual(started.wait(timeout: .now() + 5), .success)
            Thread {
                mt.enterDecode()
                for _ in 0..<30 { mt.decode { order.with { $0.append("d") }; Thread.sleep(forTimeInterval: 0.001) }; Thread.sleep(forTimeInterval: 0.001) }
                mt.leaveDecode()
                decoding.with { $0 = false }
                done.fulfill()
            }.start()
            wait(for: [done], timeout: 30)
            let parts = String(order.value).split(separator: "p", omittingEmptySubsequences: false)
            return parts.dropFirst().dropLast().map(\.count).filter { $0 > 0 }
        }
        let long = run(chunk: 0.02), short = run(chunk: 0.001)
        XCTAssertFalse(long.isEmpty); XCTAssertFalse(short.isEmpty)
        XCTAssertTrue(long.allSatisfy { $0 <= 64 }, "the share has a hard cap: \(long)")
        XCTAssertGreaterThanOrEqual(long.sorted()[long.count / 2], 6, "a 20 ms chunk leaves room for ~10 steps of 1 ms, past the burst of 4: \(long)")
        XCTAssertGreaterThanOrEqual(short.filter { $0 == 1 }.count * 10, short.count * 7, "a 1 ms chunk leaves none: \(short)")
        // below the minimum dispatch length the share stays shut even for the 20 ms chunk
        let gated = run(chunk: 0.02, minWork: 0.05)
        XCTAssertGreaterThanOrEqual(gated.filter { $0 == 1 }.count * 10, gated.count * 7, "a chunk under decodeShareMinWork opens no share: \(gated)")
    }

    func testBatchGatherCanPassPrefillsButNotDecodeClosures() {
        let order = Shared([String]())
        let mt = ModelThread(minBatch: 2, gatherWindow: 1, maxBatch: 8) { reqs in
            order.with { $0.append("batch" + reqs.map { String($0.key) }.joined()) }
            return reqs.map(\.key)
        }
        defer { mt.shutdown() }
        mt.enter(); mt.enter(); mt.enter()
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let labels = ["p0", "1", "p1", "2", "decode", "3"]
        let done = expectation(description: "batch jobs"); done.expectedFulfillmentCount = labels.count
        for (i, label) in labels.enumerated() {
            Thread {
                if let key = Int(label) { XCTAssertEqual(mt.step(key: key, pending: 0, length: 4096), key) }
                else if label == "decode" { mt.decode { order.with { $0.append(label) } } }
                else { mt.prefill { order.with { $0.append(label) } } }
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["batch12", "decode", "batch3", "p0", "p1"])
        XCTAssertEqual(mt.phaseStats()["decode"]?.jobs, 4)
        XCTAssertEqual(mt.phaseStats()["decode"]?.dispatches, 3)
        XCTAssertEqual(mt.sizeHistogram, [2: 1, 1: 1])
    }

    func testFIFOControlArmKeepsPhaseJobsInArrivalOrder() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             phaseScheduling: false) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let labels = ["p0", "d0", "p1", "d1"]
        let done = expectation(description: "FIFO jobs"); done.expectedFulfillmentCount = labels.count
        for (i, label) in labels.enumerated() {
            Thread {
                let work = { order.with { $0.append(label) } }
                if label.hasPrefix("p") { mt.prefill(work) } else { mt.decode(work) }
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, labels)
        XCTAssertEqual(mt.phaseStats()["prefill"]?.jobs, 2)
        XCTAssertEqual(mt.phaseStats()["decode"]?.jobs, 2)
    }

    func testDecoderCountIsIndependentOfBatchEligibilityAndDoesNotUnderflow() {
        let mt = makeThread(minBatch: 4)
        defer { mt.shutdown() }
        mt.enterDecode(); mt.enterDecode(); mt.enter()
        XCTAssertEqual(mt.decoderCount, 2)
        XCTAssertEqual(mt.steppableCount, 1)
        mt.leaveDecode(); mt.leaveDecode(); mt.leaveDecode()
        XCTAssertEqual(mt.decoderCount, 0)
        XCTAssertEqual(mt.steppableCount, 1)
        mt.leave()
    }

    func testPrefillBatchPreparesOnOwnerThenUsesOnlyEligibleRowsAndRunsEveryBody() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             prefillMaxBatch: 4, prefillTokenBudget: 4096) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Shared([String]()), batched = Shared(Set<Int>())
        let cancelled = Shared(false)
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let done = expectation(description: "prepared prefill"); done.expectedFulfillmentCount = 4
        for key in 0..<4 {
            Thread {
                let result = mt.prefill(key: key, position: 0, group: "offset0-width1024", tokens: 1024, prepare: {
                    XCTAssertTrue(mt.isOwner)
                    order.with { $0.append("prepare\(key)") }
                    // First row cancelled while queued; last row can reuse a prefix.
                    return !(key == 0 && cancelled.value) && key != 3
                }, batch: { keys in
                    XCTAssertTrue(mt.isOwner)
                    XCTAssertEqual(key, 1, "the first eligible row owns the callback")
                    XCTAssertEqual(keys, [1, 2])
                    batched.with { $0.formUnion(keys) }
                    order.with { $0.append("batch12") }
                }) {
                    XCTAssertTrue(mt.isOwner)
                    order.with { $0.append("body\(key)") }
                    return batched.value.contains(key) ? key + 10 : -key - 1
                }
                XCTAssertEqual(result, key == 1 || key == 2 ? key + 10 : -key - 1)
                done.fulfill()
            }.start()
            waitForQueued(key + 1, on: mt)
        }
        cancelled.with { $0 = true }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["prepare0", "prepare1", "prepare2", "prepare3", "batch12",
                                    "body0", "body1", "body2", "body3"])
        let stats = mt.prefillBatchStats()
        XCTAssertEqual(stats.gatheredGroups, 1)
        XCTAssertEqual(stats.gatheredSizeHistogram, [4: 1])
        XCTAssertEqual(stats.executedBatches, 1)
        XCTAssertEqual(stats.executedJobs, 2)
        XCTAssertEqual(stats.executedSizeHistogram, [2: 1])
        XCTAssertEqual(stats.filteredJobs, 2)
        XCTAssertEqual(mt.phaseStats()["prefill"]?.jobs, 4)
        XCTAssertEqual(mt.phaseStats()["prefill"]?.dispatches, 1)
    }

    func testPrefillBudgetIncompatibleGroupAndControlBarrierBoundGathering() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             prefillMaxBatch: 4, prefillTokenBudget: 3072) { $0.map(\.key) }
        defer { mt.shutdown() }
        let batches = Shared([[Int]]()), order = Shared([Int]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let done = expectation(description: "bounded prefill"); done.expectedFulfillmentCount = 9
        for key in 1...9 {
            Thread {
                if key == 7 {
                    mt.exclusive { order.with { $0.append(key) } }
                } else {
                    mt.prefill(key: key, position: 0, group: key == 5 ? "other" : "same",
                               tokens: key == 3 ? 2048 : 1024, prepare: { true }, batch: { keys in
                        batches.with { $0.append(keys) }
                    }) { order.with { $0.append(key) } }
                }
                done.fulfill()
            }.start()
            waitForQueued(key, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(batches.value, [[1, 2], [3, 4], [8, 9]])
        XCTAssertEqual(order.value, Array(1...9), "the token cap and control barrier constrain these groups")
    }

    func testPrefillWidthCapAndSingleEligibleFallback() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             prefillMaxBatch: 2, prefillTokenBudget: 16384) { $0.map(\.key) }
        defer { mt.shutdown() }
        let batches = Shared([[Int]]()), bodies = Shared([Int]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let done = expectation(description: "capped prefill"); done.expectedFulfillmentCount = 4
        for key in 0..<4 {
            Thread {
                mt.prefill(key: key, position: 0, group: "same", tokens: 1024, prepare: { key != 3 }, batch: { keys in
                    batches.with { $0.append(keys) }
                }) { bodies.with { $0.append(key) } }
                done.fulfill()
            }.start()
            waitForQueued(key + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(batches.value, [[0, 1]], "one remaining eligible row must use its ordinary body")
        XCTAssertEqual(bodies.value, [0, 1, 2, 3])
        let stats = mt.prefillBatchStats()
        XCTAssertEqual(stats.gatheredSizeHistogram, [2: 2])
        XCTAssertEqual(stats.executedSizeHistogram, [2: 1])
        XCTAssertEqual(stats.filteredJobs, 1)
    }

    func testFairPrefillGroupCanSkipDecodeWithoutCrossingControl() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             maxConsecutiveDecodeDispatches: 1,
                             prefillMaxBatch: 4, prefillTokenBudget: 4096) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let labels = ["p0", "d0", "p1", "d1", "p2", "control", "p3"]
        let done = expectation(description: "phase group"); done.expectedFulfillmentCount = labels.count
        for (i, label) in labels.enumerated() {
            Thread {
                let work = { order.with { $0.append(label) } }
                if label == "control" { mt.exclusive(work) }
                else if label.hasPrefix("d") { mt.decode(work) }
                else {
                    let key = Int(label.dropFirst())!
                    mt.prefill(key: key, position: 0, group: "same", tokens: 1024, prepare: { true }, batch: { keys in
                        XCTAssertEqual(keys, [0, 1, 2])
                        order.with { $0.append("batch012") }
                    }, work)
                }
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["d0", "batch012", "p0", "p1", "p2", "d1", "control", "p3"])
    }

    func testPrefillFIFOArmDoesNotGatherAcrossDecodeAndDefaultBatchingIsInert() {
        for (enabled, width) in [(false, 4), (true, 1)] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                                 phaseScheduling: enabled, prefillMaxBatch: width) { $0.map(\.key) }
            let order = Shared([String]())
            let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
            busy.wait()
            let labels = ["p0", "decode", "p1"]
            let done = expectation(description: "isolated prefill \(enabled)"); done.expectedFulfillmentCount = labels.count
            for (i, label) in labels.enumerated() {
                Thread {
                    let work = { order.with { $0.append(label) } }
                    if label == "decode" { mt.decode(work) }
                    else {
                        mt.prefill(key: i, position: (2 - i) * 1024, group: "same", tokens: 1024, prepare: { true }, batch: { _ in
                            XCTFail("these chunks must run separately")
                        }, work)
                    }
                    done.fulfill()
                }.start()
                waitForQueued(i + 1, on: mt)
            }
            release.signal()
            wait(for: [done], timeout: 5)
            XCTAssertEqual(order.value, enabled ? ["decode", "p0", "p1"] : labels)
            XCTAssertEqual(mt.prefillBatchStats().executedBatches, 0)
            mt.shutdown()
        }
    }

    func testLiveDecoderTightensAggregatePrefillBudgetWithoutChangingChunks() {
        for (hasDecoder, chunkTokens, expectedGroups) in [
            (false, 1024, [[0, 1, 2, 3]]),
            (true, 1024, []),
            (true, 512, [[0, 1], [2, 3]])
        ] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                                 prefillMaxBatch: 4, prefillTokenBudget: 4096,
                                 prefillDecodeTokenBudget: 1024) { $0.map(\.key) }
            if hasDecoder { mt.enterDecode() }
            let groups = Shared([[Int]]()), bodies = Shared([Int]())
            let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
            busy.wait()
            let done = expectation(description: "prefill decode cap \(hasDecoder) \(chunkTokens)")
            done.expectedFulfillmentCount = 4
            for key in 0..<4 {
                Thread {
                    mt.prefill(key: key, position: 0, group: "same", tokens: chunkTokens, prepare: { true }, batch: { keys in
                        groups.with { $0.append(keys) }
                    }) { bodies.with { $0.append(key) } }
                    done.fulfill()
                }.start()
                waitForQueued(key + 1, on: mt)
            }
            release.signal()
            wait(for: [done], timeout: 5)
            XCTAssertEqual(groups.value, expectedGroups)
            XCTAssertEqual(bodies.value, [0, 1, 2, 3])
            XCTAssertEqual(mt.phaseStats()["prefill"]?.dispatches,
                           hasDecoder ? (chunkTokens == 1024 ? 4 : 2) : 1)
            if hasDecoder { mt.leaveDecode() }
            mt.shutdown()
        }
    }

    func testPrefillAlignmentGathersNonadjacentPeersAfterBoundedDecodeBurst() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             prefillMaxBatch: 4) { $0.map(\.key) }
        defer { mt.shutdown() }
        let batches = Shared([[Int]]()), order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let positions = [2048, 0, 1024, 0, 2048]
        let done = expectation(description: "aligned jobs"); done.expectedFulfillmentCount = 11
        for (key, position) in positions.enumerated() {
            Thread {
                mt.prefill(key: key, position: position, group: "at\(position)", tokens: 1024,
                           prepare: { true }, batch: { keys in batches.with { $0.append(keys) } }) {
                    order.with { $0.append("p\(key)") }
                }
                done.fulfill()
            }.start()
            waitForQueued(key + 1, on: mt)
        }
        for key in 0..<6 {
            Thread { mt.decode { order.with { $0.append("d\(key)") } }; done.fulfill() }.start()
            waitForQueued(positions.count + key + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(batches.value, [[1, 3], [0, 4]])
        XCTAssertEqual(order.value, ["d0", "d1", "d2", "d3", "p1", "p3", "d4", "d5", "p2", "p0", "p4"])
        XCTAssertEqual(mt.prefillBatchStats().reorderedDispatches, 2)
        XCTAssertEqual(mt.prefillBatchStats().forcedOldestDispatches, 0)
        XCTAssertEqual(mt.phaseStats()["prefill"]?.dispatches, 3)
    }

    func testPrefillAlignmentStopsAtPlainPrefillAndControlBarriers() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             prefillMaxBatch: 4) { $0.map(\.key) }
        defer { mt.shutdown() }
        let batches = Shared([[Int]]()), order = Shared([Int]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let positions = [2048, -1, 1024, 1024, -2, 0, 0]
        let done = expectation(description: "alignment barriers"); done.expectedFulfillmentCount = positions.count
        for (key, position) in positions.enumerated() {
            Thread {
                let body = { order.with { $0.append(key) } }
                if position == -1 { mt.prefill(body) }
                else if position == -2 { mt.exclusive(body) }
                else {
                    mt.prefill(key: key, position: position, group: "at\(position)", tokens: 1024,
                               prepare: { true }, batch: { keys in batches.with { $0.append(keys) } }, body)
                }
                done.fulfill()
            }.start()
            waitForQueued(key + 1, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, Array(positions.indices))
        XCTAssertEqual(batches.value, [[2, 3], [5, 6]])
        XCTAssertEqual(mt.prefillBatchStats().reorderedDispatches, 0)
    }

    func testPrefillAlignmentPreservesSameKeyOrderAndUniqueBatchMembership() {
        // Production callers submit synchronously. Exercise malformed concurrent use too:
        // an older job for the same key must run first, even if its position is greater, and
        // two jobs for one key must never both enter the same native batch.
        let cases: [(jobs: [(Int, Int)], order: [Int], batches: [[Int]])] = [
            ([(7, 2048), (8, 1024), (7, 0), (9, 0)], [3, 1, 0, 2], []),
            ([(7, 0), (7, 0), (8, 0)], [0, 2, 1], [[7, 8]])
        ]
        for fixture in cases {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                                 prefillMaxBatch: 4) { $0.map(\.key) }
            defer { mt.shutdown() }
            let batches = Shared([[Int]]()), order = Shared([Int]())
            let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
            busy.wait()
            let done = expectation(description: "same-key jobs"); done.expectedFulfillmentCount = fixture.jobs.count
            for (index, job) in fixture.jobs.enumerated() {
                let (key, position) = job
                Thread {
                    mt.prefill(key: key, position: position, group: "at\(position)", tokens: 1024,
                               prepare: { true }, batch: { keys in
                        XCTAssertEqual(Set(keys).count, keys.count)
                        batches.with { $0.append(keys) }
                    }) { order.with { $0.append(index) } }
                    done.fulfill()
                }.start()
                waitForQueued(index + 1, on: mt)
            }
            release.signal()
            wait(for: [done], timeout: 5)
            XCTAssertEqual(order.value, fixture.order)
            XCTAssertEqual(batches.value, fixture.batches)
        }
    }

    func testPrefillReorderBoundSurvivesNewArrivalsAndInterveningDecode() {
        for limit in [2, 4] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                                 maxConsecutivePrefillReorders: limit, prefillMaxBatch: 4) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Shared([String]())
            let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
            busy.wait()
            let entered = (0..<limit).map { _ in DispatchSemaphore(value: 0) }
            let releaseChunk = (0..<limit).map { _ in DispatchSemaphore(value: 0) }
            let done = expectation(description: "reorder bound \(limit)")
            done.expectedFulfillmentCount = 1 + 2 * (limit + 1)
            Thread {
                mt.prefill(key: 0, position: 100_000, group: "oldest", tokens: 1024,
                           prepare: { true }, batch: { _ in XCTFail("no compatible peer") }) {
                    order.with { $0.append("oldest") }
                }
                done.fulfill()
            }.start()
            waitForQueued(1, on: mt)
            func submitLow(_ n: Int) {
                Thread {
                    mt.prefill(key: n, position: 0, group: "fresh\(n)", tokens: 1024,
                               prepare: { true }, batch: { _ in XCTFail("no compatible peer") }) {
                        order.with { $0.append("p\(n)") }
                        if n <= limit {
                            entered[n - 1].signal()
                            XCTAssertEqual(releaseChunk[n - 1].wait(timeout: .now() + 5), .success)
                        }
                    }
                    done.fulfill()
                }.start()
            }
            func submitDecode(_ n: Int) {
                Thread { mt.decode { order.with { $0.append("d\(n)") } }; done.fulfill() }.start()
            }
            submitLow(1)
            waitForQueued(2, on: mt)
            submitDecode(1)
            waitForQueued(3, on: mt)
            release.signal()
            for n in 1...limit {
                XCTAssertEqual(entered[n - 1].wait(timeout: .now() + 5), .success)
                // The oldest is still queued. Insert a fresh low-position job while the
                // preceding chunk is on the owner, then decode between the two chunks.
                submitLow(n + 1)
                waitForQueued(2, on: mt)
                submitDecode(n + 1)
                waitForQueued(3, on: mt)
                releaseChunk[n - 1].signal()
            }
            wait(for: [done], timeout: 5)
            var expected = (1...limit).flatMap { ["d\($0)", "p\($0)"] }
            expected += ["d\(limit + 1)", "oldest", "p\(limit + 1)"]
            XCTAssertEqual(order.value, expected)
            XCTAssertEqual(mt.prefillBatchStats().reorderedDispatches, limit)
            XCTAssertEqual(mt.prefillBatchStats().forcedOldestDispatches, 1)
        }
    }

    func testPrefillAlignmentDoesNotWaitForCompletedCallersToSubmitSuccessors() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 10, maxBatch: 8,
                             prefillMaxBatch: 4) { $0.map(\.key) }
        defer { mt.shutdown() }
        let batches = Shared([[Int]]()), order = Shared([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        let aheadEntered = DispatchSemaphore(value: 0), releaseAhead = DispatchSemaphore(value: 0)
        let parked = DispatchSemaphore(value: 0), submitNext = DispatchSemaphore(value: 0)
        let done = expectation(description: "known caller wake gap"); done.expectedFulfillmentCount = 4
        Thread {
            mt.prefill(key: 0, position: 5120, group: "at5120", tokens: 1024,
                       prepare: { true }, batch: { _ in XCTFail("the ahead row is alone") }) {
                order.with { $0.append("ahead") }
                aheadEntered.signal()
                XCTAssertEqual(releaseAhead.wait(timeout: .now() + 5), .success)
            }
            done.fulfill()
        }.start()
        waitForQueued(1, on: mt)
        for key in 1...3 {
            Thread {
                mt.prefill(key: key, position: 0, group: "at0", tokens: 1024,
                           prepare: { true }, batch: { keys in batches.with { $0.append(keys) } }) {
                    order.with { $0.append("first\(key)") }
                }
                // Hold completed callers outside the owner to reproduce a scheduling delay.
                parked.signal()
                XCTAssertEqual(submitNext.wait(timeout: .now() + 5), .success)
                mt.prefill(key: key, position: 1024, group: "at1024", tokens: 1024,
                           prepare: { true }, batch: { keys in batches.with { $0.append(keys) } }) {
                    order.with { $0.append("next\(key)") }
                }
                done.fulfill()
            }.start()
            waitForQueued(key + 1, on: mt)
        }
        release.signal()
        XCTAssertEqual(aheadEntered.wait(timeout: .now() + 5), .success,
                       "ready work runs without waiting for successors that have not been submitted")
        for _ in 1...3 { XCTAssertEqual(parked.wait(timeout: .now() + 5), .success) }
        XCTAssertEqual(order.value, ["first1", "first2", "first3", "ahead"])
        XCTAssertEqual(batches.value, [[1, 2, 3]])
        XCTAssertEqual(mt.phaseStats()["prefill"]?.queued, 0)
        for _ in 1...3 { submitNext.signal() }
        waitForQueued(3, on: mt)
        releaseAhead.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(batches.value.count, 2)
        XCTAssertEqual(Set(batches.value.dropFirst().first ?? []), Set([1, 2, 3]))
        XCTAssertEqual(mt.prefillBatchStats().executedSizeHistogram, [3: 2],
                       "ready-queue alignment alone cannot promise a B4 rendezvous")
    }

    func testCancellationWhileQueuedSkipsForwardAndSnapshot() {
        let cancelled = Shared(false), seen = Shared([Int]())
        let mt = makeThread(minBatch: 1) { reqs in seen.with { $0 += reqs.map(\.key) } }
        defer { mt.shutdown() }
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let done = expectation(description: "cancelled step")
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        Thread {
            let r = mt.stepResult(key: 7, pending: 1, length: 4096, cancelled: { cancelled.value })
            XCTAssertTrue(r.cancelled)
            done.fulfill()
        }.start()
        Thread.sleep(forTimeInterval: 0.03)
        cancelled.with { $0 = true }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(seen.value, [])
        XCTAssertEqual(mt.phaseStats()["decode"]?.cancelledJobs, 1)
        XCTAssertEqual(mt.step(key: 8, pending: 1, length: 4096), 8001)
    }
    func testExclusiveBarrierEndsGatherImmediately() {
        let mt = makeThread(minBatch: 4, window: 2)
        defer { mt.shutdown() }
        for _ in 0..<4 { mt.enter() }
        let stepDone = expectation(description: "step")
        Thread { _ = mt.step(key: 1, pending: 0, length: 4096); stepDone.fulfill() }.start()
        Thread.sleep(forTimeInterval: 0.03)
        let t = ProcessInfo.processInfo.systemUptime
        mt.exclusive { }
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - t, 0.5,
                          "an exclusive barrier makes gathering impossible; do not wait two seconds")
        wait(for: [stepDone], timeout: 1)
    }
    func testQueuedTokensAreSnapshottedBeforeOwnerRunsAnotherJob() {
        let queue = Shared([11, 12, 13])
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 1, snapshot: { _, token in
            let q = queue.with { value in let copy = value; value.removeAll(); return copy }
            return StepResult(token: token, queued: q, hasSpec: true)
        }) { _ in [10] }
        defer { mt.shutdown() }
        let result = mt.stepResult(key: 1, pending: 0, length: 4096)
        mt.exclusive { queue.with { $0 = [99] } }
        XCTAssertEqual(result, StepResult(token: 10, queued: [11, 12, 13], hasSpec: true))
    }
}

extension ModelThreadTests {
    func testOwnerTraceNamesActualRungBarrierAndPreservesQueueOrder() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let holder = Shared<ModelThread?>(nil)
        let mt = ModelThread(minBatch: 4, gatherWindow: 0.02, maxBatch: 8, ownerTrace: { event in
            XCTAssertTrue(holder.value!.isOwner)
            _ = holder.value!.steppableCount // observer must be outside cond, or this deadlocks
            events.with { $0.append(event) }
        }) { $0.map(\.key) }
        holder.with { $0 = mt }; defer { mt.shutdown() }
        for key in 1...4 { mt.enter(key: key); mt.enterDecode(key: key) }
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start(); busy.wait()
        let done = expectation(description: "traced jobs"); done.expectedFulfillmentCount = 3
        Thread { _ = mt.step(key: 1, pending: 11, length: 8512); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.decode(traceLabel: "hot_rung", traceKey: 2) {}; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        Thread { _ = mt.step(key: 3, pending: 33, length: 8509); done.fulfill() }.start()
        waitForQueued(3, on: mt); release.signal(); wait(for: [done], timeout: 5)
        let dispatches = events.value.filter { $0.event == "dispatch" }
        let event = try XCTUnwrap(dispatches.first { $0.selected.first?.key == 1 })
        XCTAssertEqual(event.gatherExit, "barrier_incomplete")
        XCTAssertEqual(event.gatherWaitCount, 0)
        XCTAssertEqual(event.quorum, 4)
        XCTAssertEqual(event.registeredKeys, [1,2,3,4]); XCTAssertEqual(event.decoderKeys, [1,2,3,4])
        XCTAssertTrue(event.identityValid)
        XCTAssertEqual(event.queue.compactMap(\.key), [1,2,3])
        XCTAssertEqual(event.exitBarrier?.label, "hot_rung"); XCTAssertEqual(event.exitBarrier?.key, 2)
        XCTAssertEqual(event.readyPrefixKeys, [1])
        let rung = try XCTUnwrap(dispatches.first { $0.selected.first?.label == "hot_rung" })
        XCTAssertEqual(rung.selected.first?.jobID, event.exitBarrier?.jobID)
        XCTAssertGreaterThan(rung.dispatchSequence, event.dispatchSequence)
        XCTAssertEqual(event.json["gather_window_ms"] as? Double, 20)
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: event.json))
    }

    func testOwnerTraceDoesNotBlameRungWhenQuorumWasReady() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let mt = ModelThread(minBatch: 4, gatherWindow: 1, maxBatch: 8,
                             ownerTrace: { e in events.with { $0.append(e) } }) { $0.map(\.key) }
        defer { mt.shutdown() }; for k in 1...4 { mt.enter(key: k) }
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start(); busy.wait()
        let done = expectation(description: "quorum before rung"); done.expectedFulfillmentCount = 5
        for k in 1...4 {
            Thread { _ = mt.step(key: k, pending: k, length: 8512); done.fulfill() }.start()
            waitForQueued(k, on: mt)
        }
        Thread { mt.decode(traceLabel: "hot_rung", traceKey: 1) {}; done.fulfill() }.start()
        waitForQueued(5, on: mt); release.signal(); wait(for: [done], timeout: 5)
        let e = try XCTUnwrap(events.value.first { $0.event == "dispatch" && $0.selected.count == 4 })
        XCTAssertEqual(e.gatherExit, "quorum_and_barrier")
        XCTAssertEqual(e.exitReadyCount, 4); XCTAssertEqual(e.exitBarrier?.label, "hot_rung")
        XCTAssertEqual(e.gatherWaitCount, 0)
    }

    func testOwnerTraceSeparatesControlBarrierAndActualCancellation() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let seen = Shared([Int]())
        let mt = ModelThread(minBatch: 2, gatherWindow: 1, maxBatch: 8,
                             ownerTrace: { e in events.with { $0.append(e) } }) { reqs in
            seen.with { $0 += reqs.map(\.key) }; return reqs.map(\.key)
        }
        defer { mt.shutdown() }; mt.enter(key: 1); mt.enter(key: 2)
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start(); busy.wait()
        let done = expectation(description: "cancel then control"); done.expectedFulfillmentCount = 2
        Thread {
            XCTAssertTrue(mt.stepResult(key: 1, pending: 1, length: 8192, cancelled: { true }).cancelled)
            done.fulfill()
        }.start(); waitForQueued(1, on: mt)
        Thread { mt.exclusive {}; done.fulfill() }.start(); waitForQueued(2, on: mt)
        release.signal(); wait(for: [done], timeout: 5)
        let e = try XCTUnwrap(events.value.first { $0.event == "dispatch" && $0.selected.first?.key == 1 })
        XCTAssertEqual(e.gatherExit, "barrier_incomplete"); XCTAssertEqual(e.exitBarrier?.phase, "control")
        let live = try XCTUnwrap(events.value.first { $0.event == "step_live" })
        XCTAssertEqual(live.liveKeys, []); XCTAssertEqual(live.filteredKeys, [1]); XCTAssertEqual(seen.value, [])
        XCTAssertEqual(live.dispatchSequence, e.dispatchSequence)
    }

    func testOwnerTraceDuplicateAndUnderflowOnlyInvalidateInstrument() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             ownerTrace: { e in events.with { $0.append(e) } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        mt.enter(key: 7); mt.enter(key: 7)
        XCTAssertEqual(mt.steppableCount, 2, "diagnostics must not change production counter semantics")
        mt.leave(key: 7); mt.leave(key: 7); mt.leave(key: 7)
        mt.enterDecode(key: 8); mt.enterDecode(key: 8)
        XCTAssertEqual(mt.decoderCount, 2)
        mt.leaveDecode(key: 8); mt.leaveDecode(key: 8); mt.leaveDecode(key: 8)
        XCTAssertEqual(mt.steppableCount, 0); XCTAssertEqual(mt.decoderCount, 0)
        mt.exclusive {}
        XCTAssertFalse(try XCTUnwrap(events.value.last).identityValid)
    }

    func testOwnerTraceQueueSnapshotIsBoundedWithoutDroppingJobs() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8,
                             ownerTrace: { e in events.with { $0.append(e) } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start(); busy.wait()
        let done = expectation(description: "bounded trace queue"); done.expectedFulfillmentCount = 65
        for i in 1...65 {
            Thread { mt.decode(traceLabel: "hot_rung", traceKey: i) {}; done.fulfill() }.start()
            waitForQueued(i, on: mt)
        }
        release.signal(); wait(for: [done], timeout: 5)
        let e = try XCTUnwrap(events.value.first { $0.queueTotal == 65 })
        XCTAssertEqual(e.queue.count, 64); XCTAssertEqual(e.json["queue_truncated"] as? Bool, true)
        XCTAssertEqual(events.value.filter { $0.selected.first?.label == "hot_rung" }.count, 65)
    }

    func testOwnerTraceDeadlineAndBelowMinimumAreNotBarrierReasons() throws {
        let events = Shared([ModelOwnerTraceEvent]())
        let mt = ModelThread(minBatch: 4, gatherWindow: 0, maxBatch: 8,
                             ownerTrace: { e in events.with { $0.append(e) } }) { $0.map(\.key) }
        defer { mt.shutdown() }
        for k in 1...4 { mt.enter(key: k) }
        _ = mt.step(key: 1, pending: 1, length: 8192)
        for k in 2...4 { mt.leave(key: k) }
        _ = mt.step(key: 1, pending: 2, length: 8193)
        let d = events.value.filter { $0.event == "dispatch" }
        XCTAssertEqual(d.map(\.gatherExit), ["deadline", "below_min"])
        XCTAssertTrue(d.allSatisfy { $0.exitBarrier == nil && $0.gatherWaitCount == 0 })
    }
}


extension ModelThreadTests {
    func testOwnerTraceBufferCapsActualStorageAndDoesNotConstructDroppedRecords() {
        let buffer = ModelOwnerTraceBuffer<Int>()
        var built = 0
        func build(_ i: Int) -> Int { built += 1; return i }
        for i in 0..<8195 { buffer.append(build(i)) }
        XCTAssertEqual(buffer.records.count, 8192)
        XCTAssertEqual(buffer.records, Array(0..<8192))
        XCTAssertEqual(buffer.dropped, 3)
        XCTAssertEqual(built, 8192, "post-cap records must not be formatted or retained")
        buffer.removeAll()
        XCTAssertTrue(buffer.records.isEmpty)
        XCTAssertEqual(buffer.dropped, 3, "flush must preserve the incomplete witness")
    }
}
