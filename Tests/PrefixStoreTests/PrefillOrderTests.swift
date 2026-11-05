// T-0141 (C-0407): ENGINE_PREFILL_ORDER=arrival. A request's prefill-phase owner jobs carry its arrival stamp and the
// owner serves the oldest request first, holding briefly for its next job, so a burst of prompts runs to completion one
// after another instead of round-robin (where every prompt's first token waited for the whole burst's prefill).
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

final class PrefillOrderTests: XCTestCase {
    private func makeThread(arrival: Bool, continueWait: TimeInterval = 0.05, burst: Int = 4,
                            share: Double = 0) -> ModelThread {
        ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: burst,
                    decodeShare: share, prefillOrderArrival: arrival, prefillContinue: continueWait) { $0.map(\.key) }
    }

    private func waitForQueued(_ count: Int, on mt: ModelThread, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while mt.phaseStats().values.reduce(0, { $0 + $1.queued }) != count {
            if ProcessInfo.processInfo.systemUptime > deadline {
                XCTFail("expected \(count) queued jobs, got \(mt.phaseStats())", file: file, line: line); return
            }
            Thread.sleep(forTimeInterval: 0.001)
        }
    }

    /// The server's shape: `requests` callers, each stamped in arrival order, each submitting `chunks` sequential prefill
    /// jobs (resubmitting only after the previous ran). The owner is held until every caller has queued its first job.
    private func burst(arrival: Bool, requests: [String], chunks: Int, work: TimeInterval = 0.005,
                       gap: TimeInterval = 0.0003) -> [String] {
        let mt = makeThread(arrival: arrival)
        defer { mt.shutdown() }
        let order = Box([String]())
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "burst"); done.expectedFulfillmentCount = requests.count
        for (i, name) in requests.enumerated() {
            let stamped = DispatchSemaphore(value: 0)
            Thread {
                mt.beginRequestPrefill()
                stamped.signal()
                for c in 1 ... chunks {
                    mt.prefill { order.with { $0.append("\(name)\(c)") }; Thread.sleep(forTimeInterval: work) }
                    Thread.sleep(forTimeInterval: gap)
                }
                mt.endRequestPrefill()
                done.fulfill()
            }.start()
            stamped.wait()
            waitForQueued(i + 1, on: mt)       // each caller's first job (the held exclusive left the queue)
        }
        hold.signal()
        wait(for: [done], timeout: 20)
        return order.value
    }

    func testFifoIsRoundRobinAcrossABurst() {
        // the shipped behaviour, the regression reference: every prompt advances one job per cycle
        XCTAssertEqual(burst(arrival: false, requests: ["A", "B", "C"], chunks: 3),
                       ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"])
    }

    func testArrivalRunsEachRequestToCompletionInArrivalOrder() {
        XCTAssertEqual(burst(arrival: true, requests: ["A", "B", "C"], chunks: 3),
                       ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"])
    }

    func testArrivalHoldsOnlyUpToTheContinueWaitAndNotAfterTheRequestEnds() {
        let mt = makeThread(arrival: true, continueWait: 0.02)
        defer { mt.shutdown() }
        let order = Box([(String, TimeInterval)]())
        let t0 = ProcessInfo.processInfo.systemUptime
        @Sendable func note(_ s: String) { order.with { $0.append((s, ProcessInfo.processInfo.systemUptime - t0)) } }
        let aFirstDone = DispatchSemaphore(value: 0), bStamped = DispatchSemaphore(value: 0)
        let done = expectation(description: "two callers"); done.expectedFulfillmentCount = 2
        // A (oldest) runs one job, then does 300 ms of work off the owner (tokenising, image decode) before its next
        Thread {
            mt.beginRequestPrefill()
            mt.prefill { note("A1") }
            aFirstDone.signal()
            Thread.sleep(forTimeInterval: 0.3)
            mt.prefill { note("A2") }
            mt.endRequestPrefill()
            done.fulfill()
        }.start()
        aFirstDone.wait()
        Thread {
            bStamped.signal()
            mt.beginRequestPrefill()
            for i in 1 ... 3 { mt.prefill { note("B\(i)") } }
            mt.endRequestPrefill()
            done.fulfill()
        }.start()
        bStamped.wait()
        wait(for: [done], timeout: 10)
        let seq = order.value.map(\.0), at = Dictionary(uniqueKeysWithValues: order.value)
        XCTAssertEqual(seq, ["A1", "B1", "B2", "B3", "A2"], "the younger request is served while the oldest is away")
        XCTAssertLessThan(at["B3"]!, 0.25, "each hold is bounded by the 20 ms continue wait: \(order.value)")
        let s = mt.prefillOrderStats
        XCTAssertGreaterThanOrEqual(s.waits, 1)
        // after A ended, B alone is the oldest: no further waits beyond B's own (its jobs resubmit within the wait)
        let waitsBefore = s.waits
        Thread {
            mt.beginRequestPrefill(); mt.endRequestPrefill()       // a request that ends without a job: no stale stamp
        }.start()
        Thread.sleep(forTimeInterval: 0.05)
        mt.prefill { note("U") }                                    // an unstamped job: nothing active, no hold
        XCTAssertEqual(mt.prefillOrderStats.waits, waitsBefore)
    }

    func testEndingTheRequestReleasesAHoldImmediately() {
        let mt = makeThread(arrival: true, continueWait: 2.0)
        defer { mt.shutdown() }
        let aRan = DispatchSemaphore(value: 0), bQueued = DispatchSemaphore(value: 0)
        let bRanAt = Box(TimeInterval(0)), aEndedAt = Box(TimeInterval(0))
        let done = expectation(description: "two callers"); done.expectedFulfillmentCount = 2
        Thread {
            mt.beginRequestPrefill()
            mt.prefill { aRan.signal() }
            bQueued.wait()
            Thread.sleep(forTimeInterval: 0.05)
            aEndedAt.with { $0 = ProcessInfo.processInfo.systemUptime }
            mt.endRequestPrefill()                                 // cancelled: no further job
            done.fulfill()
        }.start()
        aRan.wait()
        Thread {
            mt.beginRequestPrefill()
            Thread { Thread.sleep(forTimeInterval: 0.01); bQueued.signal() }.start()
            mt.prefill { bRanAt.with { $0 = ProcessInfo.processInfo.systemUptime } }
            mt.endRequestPrefill()
            done.fulfill()
        }.start()
        wait(for: [done], timeout: 10)
        XCTAssertGreaterThan(bRanAt.value, 0)
        XCTAssertLessThan(bRanAt.value - aEndedAt.value, 0.5, "B waited for the 2 s hold instead of A's end")
    }

    func testControlBarrierAndUnstampedJobsKeepTheirPlace() {
        let mt = makeThread(arrival: true, continueWait: 0.05)
        defer { mt.shutdown() }
        let order = Box([String]())
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
        // stamps: A (oldest) then B; queue after the held control: B, U (unstamped), X (control), A
        let aStamped = DispatchSemaphore(value: 0), aGo = DispatchSemaphore(value: 0)
        Thread {
            mt.beginRequestPrefill(); aStamped.signal(); aGo.wait()
            mt.prefill { order.with { $0.append("A") } }
            mt.endRequestPrefill(); done.fulfill()
        }.start()
        aStamped.wait()
        Thread { mt.beginRequestPrefill(); mt.prefill { order.with { $0.append("B") } }; mt.endRequestPrefill(); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        Thread { mt.prefill { order.with { $0.append("U") } }; done.fulfill() }.start()
        waitForQueued(2, on: mt)
        Thread { mt.exclusive { order.with { $0.append("X") } }; done.fulfill() }.start()
        waitForQueued(3, on: mt)
        aGo.signal()
        waitForQueued(4, on: mt)
        hold.signal()
        wait(for: [done], timeout: 10)
        // U (unstamped) first, then B: the oldest A is queued, but behind the control barrier, which nothing crosses
        XCTAssertEqual(order.value, ["U", "B", "X", "A"])
    }

    /// C-0419: with controlPassesPrefill a control job passes queued prefill jobs, never a decode job; off, it stays a
    /// barrier (the shipped order).
    func testControlPassesQueuedPrefillButNotDecode() {
        func run(passes: Bool) -> [String] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, controlPassesPrefill: passes) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Box([String]())
            let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
            held.wait()
            let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
            let jobs: [(String, (ModelThread, @escaping () -> Void) -> Void)] = [
                ("P1", { m, w in m.prefill(w) }), ("D1", { m, w in m.decode(w) }),
                ("X", { m, w in m.exclusive(w) }), ("P2", { m, w in m.prefill(w) })]
            for (i, (name, submit)) in jobs.enumerated() {
                Thread { submit(mt) { order.with { $0.append(name) } }; done.fulfill() }.start()
                waitForQueued(i + 1, on: mt)
            }
            hold.signal()
            wait(for: [done], timeout: 5)
            return order.value
        }
        XCTAssertEqual(run(passes: false), ["D1", "P1", "X", "P2"])
        XCTAssertEqual(run(passes: true), ["D1", "X", "P1", "P2"])
    }

    /// The W1c trace (C-0419): the oldest request's next chunk queued behind a control job, younger chunks in front of it.
    /// Barrier mode serves the younger ones first (the breakdown seen in W1a); with control passing prefill the control job
    /// runs and the oldest is served next.
    func testArrivalSeesTheOldestBehindAControlJobWhenControlPassesPrefill() {
        func run(passes: Bool) -> [String] {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true,
                                 prefillContinue: 0.05, controlPassesPrefill: passes) { $0.map(\.key) }
            defer { mt.shutdown() }
            let order = Box([String]())
            let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
            Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
            held.wait()
            let done = expectation(description: "jobs"); done.expectedFulfillmentCount = 4
            let aStamped = DispatchSemaphore(value: 0), aGo = DispatchSemaphore(value: 0)
            Thread {
                mt.beginRequestPrefill(); aStamped.signal(); aGo.wait()
                mt.prefill { order.with { $0.append("A") } }; mt.endRequestPrefill(); done.fulfill()
            }.start()
            aStamped.wait()
            for (i, name) in ["B", "C"].enumerated() {
                let stamped = DispatchSemaphore(value: 0)
                Thread {
                    mt.beginRequestPrefill(); stamped.signal()
                    mt.prefill { order.with { $0.append(name) } }; mt.endRequestPrefill(); done.fulfill()
                }.start()
                stamped.wait()
                waitForQueued(i + 1, on: mt)
            }
            Thread { mt.exclusive { order.with { $0.append("X") } }; done.fulfill() }.start()
            waitForQueued(3, on: mt)
            aGo.signal()
            waitForQueued(4, on: mt)
            hold.signal()
            wait(for: [done], timeout: 5)
            return order.value
        }
        XCTAssertEqual(run(passes: false), ["B", "C", "X", "A"])
        XCTAssertEqual(run(passes: true), ["X", "A", "B", "C"])
    }

    func testDecodeBurstBoundStillHoldsWithArrivalOrder() {
        // the same shape as testWaitingPrefillProgressIsBoundedByDecodeDispatchCount, prefills stamped in reverse
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 2,
                             prefillOrderArrival: true, prefillContinue: 0) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Box([String]())
        let busy = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { busy.signal(); release.wait() } }.start()
        busy.wait()
        // p1 is stamped first (older) but queued second
        let p1Stamped = DispatchSemaphore(value: 0), p1Go = DispatchSemaphore(value: 0)
        let labels = ["p0", "p1"] + (0 ..< 6).map { "d\($0)" }
        let done = expectation(description: "jobs"); done.expectedFulfillmentCount = labels.count
        Thread {
            mt.beginRequestPrefill(); p1Stamped.signal(); p1Go.wait()
            mt.prefill { order.with { $0.append("p1") } }; mt.endRequestPrefill(); done.fulfill()
        }.start()
        p1Stamped.wait()
        Thread { mt.beginRequestPrefill(); mt.prefill { order.with { $0.append("p0") } }; mt.endRequestPrefill(); done.fulfill() }.start()
        waitForQueued(1, on: mt)
        p1Go.signal()
        waitForQueued(2, on: mt)
        for (i, label) in labels.dropFirst(2).enumerated() {
            Thread { mt.decode { order.with { $0.append(label) } }; done.fulfill() }.start()
            waitForQueued(3 + i, on: mt)
        }
        release.signal()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(order.value, ["d0", "d1", "p1", "d2", "d3", "p0", "d4", "d5"])
    }

    /// The quantum: a request whose prefill dispatches used the quantum goes behind the others in prefill.
    func testAHugePromptIsRotatedBehindTheBurstAfterItsQuantum() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true,
                             prefillContinue: 0.05, prefillQuantum: 0.05) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Box([String]())
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "two callers"); done.expectedFulfillmentCount = 2
        for (i, (name, jobs)) in [("A", 4), ("B", 2)].enumerated() {
            let stamped = DispatchSemaphore(value: 0)
            Thread {
                mt.beginRequestPrefill(); stamped.signal()
                for c in 1 ... jobs { mt.prefill { order.with { $0.append("\(name)\(c)") }; Thread.sleep(forTimeInterval: 0.03) } }
                mt.endRequestPrefill(); done.fulfill()
            }.start()
            stamped.wait()
            waitForQueued(i + 1, on: mt)
        }
        hold.signal()
        wait(for: [done], timeout: 10)
        // A's first two jobs use 60 ms >= the 50 ms quantum: A goes behind B, which runs to completion
        XCTAssertEqual(order.value, ["A1", "A2", "B1", "B2", "A3", "A4"])
        XCTAssertGreaterThanOrEqual(mt.prefillOrderStats.rotations, 1)
    }

    /// No starvation and no reordering inside a request, over random bursts: callers with 1-4 jobs, random off-owner gaps
    /// (some longer than the continue wait), a decoder running throughout.
    func testRandomBurstsAllFinishInPerRequestOrder() {
        var rng = SystemRandomNumberGenerator()
        for _ in 0 ..< 20 {
            let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, maxConsecutiveDecodeDispatches: 4,
                                 decodeAffinity: 0.002, decodeShare: 0.5, prefillOrderArrival: true,
                                 prefillContinue: 0.003) { $0.map(\.key) }
            let order = Box([(Int, Int)]())
            let n = Int.random(in: 2 ... 6, using: &rng)
            let plans = (0 ..< n).map { _ in (jobs: Int.random(in: 1 ... 4, using: &rng),
                                             gap: Double.random(in: 0 ... 0.006, using: &rng),
                                             delay: Double.random(in: 0 ... 0.01, using: &rng)) }
            let done = expectation(description: "callers"); done.expectedFulfillmentCount = n + 1
            let decoding = Box(true)
            Thread {
                mt.enterDecode()
                while decoding.value { mt.decode { Thread.sleep(forTimeInterval: 0.0005) } }
                mt.leaveDecode(); done.fulfill()
            }.start()
            let finished = Box(0)
            for (r, plan) in plans.enumerated() {
                Thread {
                    Thread.sleep(forTimeInterval: plan.delay)
                    mt.beginRequestPrefill()
                    for j in 0 ..< plan.jobs {
                        mt.prefill { order.with { $0.append((r, j)) }; Thread.sleep(forTimeInterval: 0.001) }
                        Thread.sleep(forTimeInterval: plan.gap)
                    }
                    mt.endRequestPrefill()
                    if finished.with({ $0 += 1; return $0 }) == n { decoding.with { $0 = false } }
                    done.fulfill()
                }.start()
            }
            wait(for: [done], timeout: 20)
            mt.shutdown()
            let seen = order.value
            XCTAssertEqual(seen.count, plans.reduce(0) { $0 + $1.jobs }, "every job ran once")
            for r in 0 ..< n {
                XCTAssertEqual(seen.filter { $0.0 == r }.map(\.1), Array(0 ..< plans[r].jobs), "request \(r) kept its order")
            }
        }
    }

    // MARK: T-0178 (C-0574) -- an image request's vision tower is an owner prefill job submitted before its choice loop

    /// The server's image-request shape: each request submits its TOWER (a prefill job), then takes its arrival stamp
    /// (the choice loop), allocates its sequence (a control job) and prefills `chunks` chunks. `stampTower` takes the stamp
    /// before the tower, as ENGINE_VISION_ARRIVAL_STAMP=1 does. Production settings: arrival order, control passes prefill.
    private func visionBurst(stampTower: Bool, requests: [String], chunks: Int = 2) -> [String] {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true,
                             prefillContinue: 0.05, controlPassesPrefill: true) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Box([String]())
        let hold = DispatchSemaphore(value: 0), held = DispatchSemaphore(value: 0)
        Thread { mt.exclusive { held.signal(); hold.wait() } }.start()
        held.wait()
        let done = expectation(description: "vision burst"); done.expectedFulfillmentCount = requests.count
        for (i, name) in requests.enumerated() {
            Thread {
                if stampTower { mt.beginRequestPrefill() }
                mt.prefill { order.with { $0.append("T\(name)") }; Thread.sleep(forTimeInterval: 0.005) }   // the tower
                mt.beginRequestPrefill()                                   // the choice loop (idempotent if stamped)
                mt.exclusive { order.with { $0.append("S\(name)") } }     // sequence allocation
                for c in 1 ... chunks {
                    mt.prefill { order.with { $0.append("\(name)\(c)") }; Thread.sleep(forTimeInterval: 0.005) }
                }
                mt.endRequestPrefill()
                done.fulfill()
            }.start()
            waitForQueued(i + 1, on: mt)                                   // every request's tower is queued behind the hold
        }
        hold.signal()
        wait(for: [done], timeout: 20)
        return order.value
    }

    func testUnstampedTowersOfABurstAllRunBeforeTheFirstPrefill() {
        // the shipped behaviour (C-0574), the regression reference: unstamped jobs rank first
        // (a request's control job may pass in between, but no prompt chunk starts until every tower ran)
        let o = visionBurst(stampTower: false, requests: ["A", "B", "C"])
        print("T-0178 unstamped order: \(o)")
        let lastTower = o.lastIndex(where: { $0.hasPrefix("T") })!, firstChunk = o.firstIndex(where: { $0.hasSuffix("1") && !$0.hasPrefix("T") && !$0.hasPrefix("S") })!
        XCTAssertGreaterThan(firstChunk, lastTower, "a prompt chunk ran before the last tower: \(o)")
    }

    func testStampedTowerRunsItsRequestsPrefillBeforeTheNextTower() {
        XCTAssertEqual(visionBurst(stampTower: true, requests: ["A", "B", "C"]),
                       ["TA", "SA", "A1", "A2", "TB", "SB", "B1", "B2", "TC", "SC", "C1", "C2"])
    }

    func testAReleasedTowerStampDoesNotHoldTheNextRequest() {
        // a request that fails after its tower (500, cancel) releases its stamp; the next request's jobs must not wait
        // out the continuation window for the dead request's next job
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8, prefillOrderArrival: true,
                             prefillContinue: 0.3, controlPassesPrefill: true) { $0.map(\.key) }
        defer { mt.shutdown() }
        let order = Box([String]())
        let aStamped = DispatchSemaphore(value: 0), aDone = DispatchSemaphore(value: 0)
        Thread {
            mt.beginRequestPrefill(); aStamped.signal()
            mt.prefill { order.with { $0.append("TA") } }
            mt.endRequestPrefill()                                         // the tower failed: early exit
            aDone.signal()
        }.start()
        aStamped.wait(); aDone.wait()
        let t0 = ProcessInfo.processInfo.systemUptime
        Thread { mt.beginRequestPrefill(); mt.prefill { order.with { $0.append("B1") } }; mt.prefill { order.with { $0.append("B2") } }
                 mt.endRequestPrefill() }.start()
        let deadline = t0 + 5
        while order.value.count < 3 && ProcessInfo.processInfo.systemUptime < deadline { Thread.sleep(forTimeInterval: 0.001) }
        XCTAssertEqual(order.value, ["TA", "B1", "B2"])
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - t0, 0.25, "B waited out a released request's continuation window")
    }
}
