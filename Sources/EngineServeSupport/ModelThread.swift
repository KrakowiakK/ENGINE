import Foundation

/// One queued decode step, as the owner thread sees it: a key naming the sequence, plus the two
/// plain numbers a batched forward needs. No MLX type crosses this boundary.
public struct StepRequest: Sendable {
    public let key: Int
    public let pending: Int        // the token to forward
    public let length: Int         // committed context length, i.e. this row's offset
    public init(key: Int, pending: Int, length: Int) {
        self.key = key; self.pending = pending; self.length = length
    }
}

/// P089 U10 -- ONE OWNER THREAD FOR EVERY MLX CALL.
///
/// P087's ticket gate made the model single-threaded *while a forward was running*, which is not the
/// same invariant. The modules are process-global instances with mutable state (the QSA indexer
/// keeps its last top-k on `self`), and every MLXArray a request holds is freed by whichever thread
/// drops the last reference -- so a connection thread doing `pending = next` calls `mlx_array_free`
/// while another thread is inside the model. That is the shape of both crashes P089 recorded:
/// SIGABRT in `mlx_array_free` ("pointer being freed was not allocated") and SIGSEGV in
/// `swift_release_dealloc`. A gate cannot fix it, because the dangerous operation is a
/// DEALLOCATION, and deallocation is not something a caller opts into.
///
/// So the rule stops being "hold the gate across model calls" and becomes an ownership rule:
/// **every MLXArray is created, mutated and destroyed on this thread, and a connection thread holds
/// no MLX-typed value at all.** Connection threads submit closures and receive plain data back --
/// which is why `step` takes and returns `Int`, and why `exclusive`'s result type must be plain.
///
/// The thread is also where BATCHING belongs. With one owner there is no leader to elect and no
/// rendezvous to get wrong: coalescing is just "look at the front of my own queue". That deletes
/// `BatchHub` along with its whole class of defect -- a follower released without being served, a
/// follower reading its row back outside the gate.
///
/// Control jobs remain FIFO barriers. Between barriers, ready decode can pass prefill chunks;
/// a bounded decode burst guarantees that a waiting prefill still makes progress.
/// A one-slot inbox for `exclusive`'s return value. A nested generic class is not allowed, and the
/// value has to outlive the closure that produced it, so it gets its own type.
public struct StepResult: Sendable, Equatable {
    public let token: Int
    public let queued: [Int]
    public let hasSpec: Bool
    public let cancelled: Bool
    public init(token: Int, queued: [Int] = [], hasSpec: Bool = false, cancelled: Bool = false) {
        self.token = token; self.queued = queued; self.hasSpec = hasSpec; self.cancelled = cancelled
    }
}

private final class ResultBox<T> { var value: T? = nil }

/// Host-side owner-thread measurements. Work time includes the closure/callback and owner snapshot;
/// it is not GPU execution time. Queue wait is summed per job, work time once per dispatch.
public struct ModelPhaseStats: Sendable, Equatable {
    public fileprivate(set) var jobs = 0
    public fileprivate(set) var dispatches = 0
    public fileprivate(set) var cancelledJobs = 0
    public fileprivate(set) var queueWaitSeconds: TimeInterval = 0
    public fileprivate(set) var maxQueueWaitSeconds: TimeInterval = 0
    public fileprivate(set) var workSeconds: TimeInterval = 0
    public fileprivate(set) var maxWorkSeconds: TimeInterval = 0
    public fileprivate(set) var queued = 0
}

/// Selection and execution are separate: cancellation or prefix reuse can remove selected rows
/// before a native prefill batch runs. Histograms contain groups of two or more rows only.
public struct ModelPrefillBatchStats: Sendable, Equatable {
    public fileprivate(set) var gatheredGroups = 0
    public fileprivate(set) var gatheredSizeHistogram: [Int: Int] = [:]
    public fileprivate(set) var executedBatches = 0
    public fileprivate(set) var executedJobs = 0
    public fileprivate(set) var executedSizeHistogram: [Int: Int] = [:]
    public fileprivate(set) var filteredJobs = 0
    /// Lower-position dispatches that omitted the oldest queued prefill, and dispatches forced
    /// back to that oldest job after the configured reorder bound was reached.
    public fileprivate(set) var reorderedDispatches = 0
    public fileprivate(set) var forcedOldestDispatches = 0
}

/// Owner-local diagnostic storage. Callers supply plain records; the rejected record is never
/// constructed. The fixed cap bounds both retained entries and post-cap instrumentation work.
public final class ModelOwnerTraceBuffer<Record> {
    public private(set) var records: [Record] = []
    public private(set) var dropped = 0
    public init() {}
    public func append(_ record: @autoclosure () -> Record) {
        guard records.count < 8192 else { dropped += 1; return }
        records.append(record())
    }
    public func removeAll() { records.removeAll() }
}

/// Bounded plain snapshots for the opt-in owner-order instrument. No tensor or closure escapes.
public struct ModelOwnerJobTrace: Sendable {
    public let jobID: Int
    public let phase: String
    public let label: String
    public let key: Int?
    public let pending: Int?
    public let length: Int?
    public var json: [String: Any] {
        ["job_id": jobID, "phase": phase, "kind": pending == nil ? "closure" : "step",
         "label": label, "key": key as Any? ?? NSNull(),
         "pending": pending as Any? ?? NSNull(), "length": length as Any? ?? NSNull()]
    }
}

public struct ModelOwnerTraceEvent: Sendable {
    public fileprivate(set) var event = "dispatch"
    public fileprivate(set) var dispatchSequence = 0
    public fileprivate(set) var selected: [ModelOwnerJobTrace] = []
    public fileprivate(set) var queue: [ModelOwnerJobTrace] = []
    public fileprivate(set) var queueTotal = 0
    public fileprivate(set) var registeredKeys: [Int] = []
    public fileprivate(set) var decoderKeys: [Int] = []
    public fileprivate(set) var steppable = 0
    public fileprivate(set) var decoders = 0
    public fileprivate(set) var quorum = 0
    public fileprivate(set) var minBatch = 0, maxBatch = 0
    public fileprivate(set) var gatherWindowMS: Double = 0
    public fileprivate(set) var identityValid = true
    public fileprivate(set) var phaseScheduling = true
    public fileprivate(set) var decodeBurst = 0
    public fileprivate(set) var prefillWaiting = false
    public fileprivate(set) var gatherExit = "not_step"
    public fileprivate(set) var gatherWaitCount = 0
    public fileprivate(set) var exitReadyCount = 0
    public fileprivate(set) var exitSteppable = 0
    public fileprivate(set) var exitBarrier: ModelOwnerJobTrace? = nil
    public fileprivate(set) var readyPrefixKeys: [Int] = []
    public fileprivate(set) var finalBarrier: ModelOwnerJobTrace? = nil
    public fileprivate(set) var liveKeys: [Int] = []
    public fileprivate(set) var filteredKeys: [Int] = []
    public var json: [String: Any] {
        if event == "step_live" {
            return ["event": event, "dispatch_seq": dispatchSequence,
                    "selected_keys": selected.compactMap(\.key), "live_keys": liveKeys, "filtered_keys": filteredKeys]
        }
        var d: [String: Any] = ["event": event, "dispatch_seq": dispatchSequence,
            "selected_jobs": selected.map(\.json), "queue": queue.map(\.json), "queue_total": queueTotal,
            "queue_truncated": queueTotal > queue.count, "registered_keys": registeredKeys, "decoder_keys": decoderKeys,
            "registered_steppable": steppable, "decoders": decoders, "quorum": quorum,
            "key_counts_valid": identityValid, "phase_scheduler": phaseScheduling, "decode_burst": decodeBurst,
            "prefill_waiting": prefillWaiting, "gather_exit": gatherExit, "gather_wait_count": gatherWaitCount]
        d["min_batch"] = minBatch; d["max_batch"] = maxBatch; d["gather_window_ms"] = gatherWindowMS
        d["gather_exit_ready_count"] = exitReadyCount; d["gather_exit_steppable"] = exitSteppable
        d["gather_exit_barrier"] = exitBarrier?.json ?? NSNull()
        d["ready_prefix_keys"] = readyPrefixKeys; d["all_queued_step_keys"] = queue.filter { $0.pending != nil }.compactMap(\.key)
        d["final_barrier"] = finalBarrier?.json ?? NSNull()
        return d
    }
}

public final class ModelThread: @unchecked Sendable {
    private enum Phase: String, CaseIterable { case control, prefill, decode }

    private struct PrefillWork {
        let key: Int
        let position: Int
        let group: String
        let tokens: Int
        let prepare: () -> Bool
        let batch: ([Int]) -> Void
    }

    /// One unit of work. A closure runs alone; decode requests can coalesce without crossing a
    /// control barrier or another decode closure. All arguments and results crossing the owner
    /// boundary are plain Sendable values.
    private final class Job {
        let phase: Phase
        let body: (() -> Void)?
        let req: StepRequest?
        let prefill: PrefillWork?
        let cancelled: () -> Bool
        var traceID = 0
        var traceLabel = ""
        var traceKey: Int? = nil
        var enqueuedAt: TimeInterval = 0
        var result = StepResult(token: -1)
        var done = false
        init(phase: Phase, body: (() -> Void)?, req: StepRequest?, prefill: PrefillWork? = nil,
             cancelled: @escaping () -> Bool = { false }) {
            self.phase = phase; self.body = body; self.req = req; self.prefill = prefill; self.cancelled = cancelled
        }
    }

    private let cond = NSCondition()
    private var queue: [Job] = []
    private var steppable = 0              // rows that could join a batched step right now
    private var decoders = 0               // all decode-phase rows, including solo/MTP/logprobs
    private var stopping = false
    private var owner: Thread? = nil
    private let minBatch: Int
    private let maxBatch: Int
    private let gatherWindow: TimeInterval
    private let phaseScheduling: Bool
    private let maxConsecutiveDecodeDispatches: Int
    private let maxConsecutivePrefillReorders: Int
    private let prefillMaxBatch: Int
    private let prefillTokenBudget: Int
    private let prefillDecodeTokenBudget: Int
    private var consecutiveDecodeDispatches = 0
    private var consecutivePrefillReorders = 0
    private let runBatch: ([StepRequest]) -> [Int]
    private let snapshot: ((StepRequest, Int) -> StepResult)?
    private let ownerTrace: ((ModelOwnerTraceEvent) -> Void)?
    private var traceJobID = 0, traceDispatchID = 0
    private var traceRegistered: Set<Int> = [], traceDecoders: Set<Int> = []
    private var traceKeysValid = true
    private var traceSelection: ModelOwnerTraceEvent? = nil

    /// Instrument: are groups forming, and how wide?
    private var recordedStepCount = 0
    private var recordedSizeHistogram: [Int: Int] = [:]
    private var recordedPhases: [Phase: ModelPhaseStats] = [:]
    private var recordedPrefillBatches = ModelPrefillBatchStats()
    public var stepCount: Int { cond.lock(); defer { cond.unlock() }; return recordedStepCount }
    public var sizeHistogram: [Int: Int] { cond.lock(); defer { cond.unlock() }; return recordedSizeHistogram }

    /// - minBatch: below this many steppable rows the owner never lingers for company.
    /// - maxBatch: the widest group it will run; the overflow is served by the next round.
    /// - maxConsecutiveDecodeDispatches: while prefill is waiting, at most this many decode
    ///   dispatches may run before a prefill dispatch. One batched step is one dispatch.
    /// - maxConsecutivePrefillReorders: when prefill batching is enabled, lower-position chunks
    ///   may pass the oldest prefill this many times before it must run. Decode does not reset
    ///   this bound. Plain prefill closures and control jobs end an alignment region.
    /// - prefillMaxBatch: widest compatible prefill group; one disables prefill batching.
    /// - prefillTokenBudget: aggregate token cap for a group. An individually oversized chunk
    ///   still runs its ordinary body, alone. Prefill batching adds no gathering delay.
    /// - prefillDecodeTokenBudget: tighter aggregate cap while any request is decoding. This
    ///   limits prefill grouping without changing a request's chunk width or arithmetic.
    /// - runBatch: performs one decode step for the whole group and returns one token per request,
    ///   in the order given. Called ON the owner thread.
    public init(minBatch: Int, gatherWindow: TimeInterval, maxBatch: Int,
                phaseScheduling: Bool = true,
                maxConsecutiveDecodeDispatches: Int = 4,
                maxConsecutivePrefillReorders: Int = 4,
                prefillMaxBatch: Int = 1,
                prefillTokenBudget: Int = 4096,
                prefillDecodeTokenBudget: Int = 1024,
                snapshot: ((StepRequest, Int) -> StepResult)? = nil,
                ownerTrace: ((ModelOwnerTraceEvent) -> Void)? = nil,
                runBatch: @escaping ([StepRequest]) -> [Int]) {
        self.minBatch = max(1, minBatch)
        self.maxBatch = max(self.minBatch, maxBatch)
        self.gatherWindow = gatherWindow
        self.phaseScheduling = phaseScheduling
        self.maxConsecutiveDecodeDispatches = max(1, maxConsecutiveDecodeDispatches)
        self.maxConsecutivePrefillReorders = max(1, maxConsecutivePrefillReorders)
        self.prefillMaxBatch = max(1, prefillMaxBatch)
        self.prefillTokenBudget = max(1, prefillTokenBudget)
        self.prefillDecodeTokenBudget = max(1, prefillDecodeTokenBudget)
        self.runBatch = runBatch
        self.snapshot = snapshot
        self.ownerTrace = ownerTrace
        let t = Thread { [weak self] in self?.loop() }
        t.name = "engine.model"
        t.stackSize = 16 << 20            // the model's call chain is deep; the 512k default is not enough
        t.start()
        // The owner must exist before the first caller submits, or `isOwner` is answered against a
        // nil and a re-entrant `exclusive` would deadlock against itself.
        cond.lock()
        while owner == nil { cond.wait() }
        cond.unlock()
    }

    /// True when the CALLER is the owner thread. Callers assert the ownership rule with it.
    public var isOwner: Bool { Thread.current === owner }

    /// Run `body` on the owner thread and return its result. `body` must not return, or otherwise
    /// let escape, any MLX-typed value: that is the entire point of this type.
    @discardableResult
    public func exclusive<T: Sendable>(_ body: () -> T) -> T {
        run(.control, body)
    }

    /// One bounded prefill chunk. Decode can overtake it only before the next control barrier.
    /// As for `exclusive`, no MLX-typed value may escape the owner closure.
    @discardableResult
    public func prefill<T: Sendable>(_ body: () -> T) -> T {
        run(.prefill, body)
    }

    /// Submit a chunk that may share native prefill with compatible chunks. `group` names the
    /// complete compatibility class (model, offset, width and any relevant execution options).
    /// `position` is the chunk's starting offset; the phase scheduler can favor lagging rows
    /// without interpreting the compatibility string. Each key retains its own arrival order.
    /// On the owner, every selected `prepare` runs first. With at least two eligible rows, the
    /// first eligible job's `batch` receives their keys in queue order. Every selected `body`
    /// then runs, including filtered rows: it consumes prepared results or performs its ordinary
    /// coalescing/cancellation/fallback path. Callbacks must leave all MLX values on the owner.
    @discardableResult
    public func prefill<T: Sendable>(key: Int, position: Int, group: String, tokens: Int,
                                     prepare: () -> Bool, batch: ([Int]) -> Void,
                                     _ body: () -> T) -> T {
        precondition(tokens > 0, "a prefill chunk must contain at least one token")
        precondition(position >= 0, "a prefill chunk must have a nonnegative position")
        if isOwner { _ = prepare(); return body() }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(prepare) { prepareBody -> T in
            withoutActuallyEscaping(batch) { batchBody -> T in
                withoutActuallyEscaping(body) { escaped -> T in
                    let work = PrefillWork(key: key, position: position, group: group, tokens: tokens,
                                           prepare: prepareBody, batch: batchBody)
                    submitAndWait(Job(phase: .prefill, body: { box.value = escaped() },
                                      req: nil, prefill: work))
                    return box.value!
                }
            }
        }
    }

    /// A serial/speculative decode round, with the same priority as a batched decode request.
    @discardableResult
    public func decode<T: Sendable>(traceLabel: String = "decode", traceKey: Int? = nil, _ body: () -> T) -> T {
        run(.decode, traceLabel: traceLabel, traceKey: traceKey, body)
    }

    private func run<T: Sendable>(_ phase: Phase, traceLabel: String = "", traceKey: Int? = nil, _ body: () -> T) -> T {
        if isOwner { return body() }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(body) { escaped -> T in
            let job = Job(phase: phase, body: { box.value = escaped() }, req: nil)
            job.traceLabel = traceLabel; job.traceKey = traceKey
            submitAndWait(job)
            return box.value!
        }
    }

    /// Run one decode step for `key`, possibly batched with the steps queued beside it, and return
    /// the token sampled for THIS row. Returns when this row's step has actually been run -- never
    /// merely when someone else's has.
    public func step(key: Int, pending: Int, length: Int) -> Int {
        stepResult(key: key, pending: pending, length: length).token
    }

    public func stepResult(key: Int, pending: Int, length: Int, cancelled: @escaping () -> Bool = { false }) -> StepResult {
        precondition(!isOwner, "a decode step may not be submitted from the owner thread")
        let job = Job(phase: .decode, body: nil, req: StepRequest(key: key, pending: pending, length: length), cancelled: cancelled)
        submitAndWait(job)
        return job.result
    }

    /// A row declares itself able to join a batched step. The owner lingers for company only while
    /// at least `minBatch` rows have declared it -- otherwise a lone eligible request pays the whole
    /// gather window on every token, which measured 14.4 tok/s where the step alone allows ~90.
    public func enter(key: Int? = nil) {
        cond.lock(); steppable += 1
        if ownerTrace != nil { if let key { if !traceRegistered.insert(key).inserted { traceKeysValid = false } } else { traceKeysValid = false } }
        cond.broadcast(); cond.unlock()
    }
    public func leave(key: Int? = nil) {
        cond.lock(); steppable = max(0, steppable - 1)
        if ownerTrace != nil { if let key { if traceRegistered.remove(key) == nil { traceKeysValid = false } } else { traceKeysValid = false } }
        cond.broadcast(); cond.unlock()
    }
    public var steppableCount: Int { cond.lock(); defer { cond.unlock() }; return steppable }

    /// Independent of batch eligibility: short, forced, logprob and solo-MTP decoders also need
    /// latency protection from prefill. Callers pair these on their decode-phase lifetime.
    public func enterDecode(key: Int? = nil) {
        cond.lock(); decoders += 1
        if ownerTrace != nil { if let key { if !traceDecoders.insert(key).inserted { traceKeysValid = false } } else { traceKeysValid = false } }
        cond.broadcast(); cond.unlock()
    }
    public func leaveDecode(key: Int? = nil) {
        cond.lock(); decoders = max(0, decoders - 1)
        if ownerTrace != nil { if let key { if traceDecoders.remove(key) == nil { traceKeysValid = false } } else { traceKeysValid = false } }
        cond.broadcast(); cond.unlock()
    }
    public var decoderCount: Int { cond.lock(); defer { cond.unlock() }; return decoders }

    public func phaseStats() -> [String: ModelPhaseStats] {
        cond.lock(); defer { cond.unlock() }
        return Dictionary(uniqueKeysWithValues: Phase.allCases.map { phase in
            var stats = recordedPhases[phase, default: ModelPhaseStats()]
            stats.queued = queue.reduce(0) { $0 + ($1.phase == phase ? 1 : 0) }
            return (phase.rawValue, stats)
        })
    }

    public func prefillBatchStats() -> ModelPrefillBatchStats {
        cond.lock(); defer { cond.unlock() }
        return recordedPrefillBatches
    }

    public func statsLine() -> String {
        cond.lock(); defer { cond.unlock() }
        let sizes = recordedSizeHistogram.sorted { $0.key < $1.key }.map { "B\($0.key)=\($0.value)" }.joined(separator: " ")
        let phases = Phase.allCases.map { phase in
            let s = recordedPhases[phase, default: ModelPhaseStats()]
            return "\(phase.rawValue)=\(s.jobs)/\(s.dispatches) wait_ms=\(String(format: "%.1f", s.queueWaitSeconds * 1000)) work_ms=\(String(format: "%.1f", s.workSeconds * 1000))"
        }.joined(separator: "; ")
        let ps = recordedPrefillBatches
        let prefillSizes = ps.executedSizeHistogram.sorted { $0.key < $1.key }.map { "B\($0.key)=\($0.value)" }.joined(separator: " ")
        return "batched steps \(recordedStepCount), sizes: \(sizes); phases jobs/dispatches: \(phases); prefill batches \(ps.executedBatches) jobs \(ps.executedJobs) gathered \(ps.gatheredGroups) filtered \(ps.filteredJobs), sizes: \(prefillSizes)"
    }

    /// Stop the owner thread once the queue drains. Only for tests: the server's thread lives as
    /// long as the process.
    public func shutdown() { cond.lock(); stopping = true; cond.broadcast(); cond.unlock() }

    private func submitAndWait(_ job: Job) {
        cond.lock()
        if ownerTrace != nil { traceJobID += 1; job.traceID = traceJobID }
        job.enqueuedAt = ProcessInfo.processInfo.systemUptime
        queue.append(job)
        cond.broadcast()
        while !job.done { cond.wait() }
        cond.unlock()
    }

    private func loop() {
        cond.lock()
        owner = Thread.current
        cond.broadcast()
        while true {
            while queue.isEmpty && !stopping {
                consecutivePrefillReorders = 0
                cond.wait()
            }
            if queue.isEmpty { cond.unlock(); return }          // stopping, and drained
            if ownerTrace != nil { traceDispatchID += 1; captureTraceLocked(reason: "not_step") }
            let batch = takeFrontLocked()
            var selectedTrace = traceSelection
            if ownerTrace != nil { selectedTrace?.selected = batch.map(traceJob) }
            let phase = batch[0].phase
            let startedAt = ProcessInfo.processInfo.systemUptime
            var metrics = recordedPhases[phase, default: ModelPhaseStats()]
            metrics.jobs += batch.count
            metrics.dispatches += 1
            for job in batch {
                let wait = max(0, startedAt - job.enqueuedAt)
                metrics.queueWaitSeconds += wait
                metrics.maxQueueWaitSeconds = max(metrics.maxQueueWaitSeconds, wait)
            }
            if phase == .prefill, batch.count > 1, batch[0].prefill != nil {
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                let eligible = batch.compactMap { job -> PrefillWork? in
                    guard let work = job.prefill, work.prepare() else { return nil }
                    return work
                }
                if eligible.count >= 2 { eligible[0].batch(eligible.map(\.key)) }
                for job in batch { job.body!() }
                cond.lock()
                if batch.count >= 2 {
                    recordedPrefillBatches.gatheredGroups += 1
                    recordedPrefillBatches.gatheredSizeHistogram[batch.count, default: 0] += 1
                }
                recordedPrefillBatches.filteredJobs += batch.count - eligible.count
                if eligible.count >= 2 {
                    recordedPrefillBatches.executedBatches += 1
                    recordedPrefillBatches.executedJobs += eligible.count
                    recordedPrefillBatches.executedSizeHistogram[eligible.count, default: 0] += 1
                }
            } else if batch.count == 1, let body = batch[0].body {
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                body()
                cond.lock()
            } else {
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                // Re-check on the owner: the client may have left while waiting in FIFO.
                let live = batch.filter { job in
                    if job.cancelled() { job.result = StepResult(token: -1, cancelled: true); return false }
                    return true
                }
                if var event = selectedTrace {
                    event.event = "step_live"; event.liveKeys = live.map { $0.req!.key }
                    event.filteredKeys = batch.filter { $0.result.cancelled }.map { $0.req!.key }
                    ownerTrace?(event)
                }
                let tokens = live.isEmpty ? [] : runBatch(live.map { $0.req! })
                // Snapshot and drain while still on the owner, BEFORE another job can dissolve
                // the pool. Only immutable Sendable values return to connection threads.
                let results = live.enumerated().map { i, j in
                    let token = i < tokens.count ? tokens[i] : -1
                    return snapshot?(j.req!, token) ?? StepResult(token: token)
                }
                cond.lock()
                metrics.cancelledJobs += batch.count - live.count
                if !live.isEmpty { recordedStepCount += 1; recordedSizeHistogram[live.count, default: 0] += 1 }
                for (j, result) in zip(live, results) { j.result = result }
            }
            let work = max(0, ProcessInfo.processInfo.systemUptime - startedAt)
            metrics.workSeconds += work
            metrics.maxWorkSeconds = max(metrics.maxWorkSeconds, work)
            recordedPhases[phase] = metrics
            for j in batch { j.done = true }
            cond.broadcast()
        }
    }

    private func traceJob(_ job: Job) -> ModelOwnerJobTrace {
        ModelOwnerJobTrace(jobID: job.traceID, phase: job.phase.rawValue,
                          label: job.req != nil ? "step" : (job.traceLabel.isEmpty ? job.phase.rawValue : job.traceLabel),
                          key: job.req?.key ?? job.prefill?.key ?? job.traceKey,
                          pending: job.req?.pending, length: job.req?.length ?? job.prefill?.position)
    }

    /// Called only while cond is held; no observer, serialization or I/O happens under the lock.
    private func captureTraceLocked(reason: String, waits: Int = 0, exitReady: Int = 0,
                                    exitSteppable: Int? = nil, exitBarrier: ModelOwnerJobTrace? = nil) {
        guard ownerTrace != nil else { return }
        var e = ModelOwnerTraceEvent()
        e.dispatchSequence = traceDispatchID; e.queueTotal = queue.count
        e.queue = queue.prefix(64).map(traceJob)
        e.registeredKeys = traceRegistered.sorted(); e.decoderKeys = traceDecoders.sorted()
        e.steppable = steppable; e.decoders = decoders; e.quorum = min(steppable, maxBatch)
        e.minBatch = minBatch; e.maxBatch = maxBatch; e.gatherWindowMS = gatherWindow * 1000
        e.identityValid = traceKeysValid && traceRegistered.count == steppable && traceDecoders.count == decoders
        e.phaseScheduling = phaseScheduling; e.decodeBurst = consecutiveDecodeDispatches
        e.prefillWaiting = queue.prefix { $0.phase != .control }.contains { $0.phase == .prefill }
        e.gatherExit = reason; e.gatherWaitCount = waits; e.exitReadyCount = exitReady
        e.exitSteppable = exitSteppable ?? steppable; e.exitBarrier = exitBarrier
        e.finalBarrier = queue.first { $0.phase == .control || ($0.phase == .decode && $0.body != nil) }.map(traceJob)
        for job in queue {
            if job.phase == .control || (job.phase == .decode && job.body != nil) { break }
            if let req = job.req { e.readyPrefixKeys.append(req.key) }
            if e.readyPrefixKeys.count >= maxBatch { break }
        }
        traceSelection = e
    }

    /// Called and returned with the lock held. Only the prefix before the first control job can
    /// be reordered. Bounded decode priority and bounded prefill alignment prevent starvation.
    private func takeFrontLocked() -> [Job] {
        if !phaseScheduling { return takeFIFOFrontLocked() }
        if queue[0].phase == .control {
            consecutiveDecodeDispatches = 0
            consecutivePrefillReorders = 0
            return [queue.removeFirst()]
        }
        let barrier = queue.firstIndex { $0.phase == .control } ?? queue.count
        let prefill = queue[..<barrier].firstIndex { $0.phase == .prefill }
        let decode = queue[..<barrier].firstIndex { $0.phase == .decode }
        if prefill == nil { consecutivePrefillReorders = 0 }
        if let prefill, decode == nil || consecutiveDecodeDispatches >= maxConsecutiveDecodeDispatches {
            consecutiveDecodeDispatches = 0
            return takeAlignedPrefillGroupLocked(at: prefill)
        }
        // There is a decode before the barrier, otherwise the prefill branch above handled it.
        let first = decode!
        consecutiveDecodeDispatches = prefill == nil ? 0 : consecutiveDecodeDispatches + 1
        if queue[first].body != nil { return [queue.remove(at: first)] }
        func candidates() -> (indices: [Int], blocked: Bool) {
            var indices: [Int] = []
            for (i, job) in queue.enumerated() {
                if job.phase == .control || (job.phase == .decode && job.body != nil) {
                    return (indices, true)
                }
                if job.req != nil {
                    indices.append(i)
                    if indices.count == maxBatch { return (indices, false) }
                }
            }
            return (indices, false)
        }
        var traceReason = minBatch > 1 ? "below_min" : "no_gather", traceWaits = 0
        var traceExitReady = 0, traceExitSteppable = steppable
        var traceExitBarrier: ModelOwnerJobTrace? = nil
        if minBatch > 1 && steppable >= minBatch {
            let deadline = Date().addingTimeInterval(gatherWindow)
            traceReason = "deadline"
            while steppable >= minBatch && Date() < deadline {
                let ready = candidates()
                if ready.indices.count >= min(steppable, maxBatch) || ready.blocked {
                    if ownerTrace != nil {
                        let quorum = ready.indices.count >= min(steppable, maxBatch)
                        traceReason = quorum ? (ready.blocked ? "quorum_and_barrier" : "quorum") : "barrier_incomplete"
                        traceExitReady = ready.indices.count; traceExitSteppable = steppable
                        if ready.blocked, let job = queue.first(where: { $0.phase == .control || ($0.phase == .decode && $0.body != nil) }) {
                            traceExitBarrier = traceJob(job)
                        }
                    }
                    break
                }
                // Control/decode closures end gathering; a leaving row shrinks the quorum.
                if ownerTrace != nil { traceWaits += 1 }
                if !cond.wait(until: deadline) { traceReason = "wait_timeout"; break }
            }
            if traceReason == "deadline", steppable < minBatch { traceReason = "below_min" }
        }
        let indices = candidates().indices
        if ownerTrace != nil {
            if traceReason != "quorum" && traceReason != "quorum_and_barrier" && traceReason != "barrier_incomplete" {
                traceExitReady = indices.count; traceExitSteppable = steppable
            }
            captureTraceLocked(reason: traceReason, waits: traceWaits, exitReady: traceExitReady,
                               exitSteppable: traceExitSteppable, exitBarrier: traceExitBarrier)
        }
        // A prefill can arrive during gathering. This dispatch then counts towards its bound.
        if prefill == nil {
            let end = queue.firstIndex { $0.phase == .control } ?? queue.count
            consecutiveDecodeDispatches = queue[..<end].contains { $0.phase == .prefill } ? 1 : 0
        }
        let taken = indices.map { queue[$0] }
        for i in indices.reversed() { queue.remove(at: i) }
        return taken
    }

    /// Favor the least advanced ready row, then gather compatible peers even when other chunk
    /// jobs separate them. Plain closures have no offset/ownership metadata and remain barriers
    /// for prefill reordering. There is deliberately no wait for callers to submit a next chunk.
    private func takeAlignedPrefillGroupLocked(at oldest: Int) -> [Job] {
        guard prefillMaxBatch > 1, queue[oldest].prefill != nil else {
            consecutivePrefillReorders = 0
            return [queue.remove(at: oldest)]
        }
        var candidates: [Int] = []
        var seenKeys = Set<Int>()
        for i in oldest..<queue.count {
            let job = queue[i]
            if job.phase == .control { break }
            if job.phase == .decode { continue }
            guard let work = job.prefill else { break }
            // Callers normally have only one outstanding chunk per key. Enforce that order
            // even if two callers accidentally submit the same key concurrently.
            if seenKeys.insert(work.key).inserted { candidates.append(i) }
        }
        var first = oldest
        for i in candidates where queue[i].prefill!.position < queue[first].prefill!.position {
            first = i
        }
        if first != oldest, consecutivePrefillReorders >= maxConsecutivePrefillReorders {
            first = oldest
            recordedPrefillBatches.forcedOldestDispatches += 1
        }
        if first == oldest {
            consecutivePrefillReorders = 0
        } else {
            consecutivePrefillReorders += 1
            recordedPrefillBatches.reorderedDispatches += 1
        }
        let initial = queue[first].prefill!
        let tokenBudget = decoders > 0 ? min(prefillTokenBudget, prefillDecodeTokenBudget) : prefillTokenBudget
        guard initial.tokens <= tokenBudget else { return [queue.remove(at: first)] }
        var indices: [Int] = []
        var tokens = 0
        for i in candidates {
            let work = queue[i].prefill!
            guard work.group == initial.group, work.position == initial.position else { continue }
            if indices.count == prefillMaxBatch || work.tokens > tokenBudget - tokens { break }
            indices.append(i)
            tokens += work.tokens
        }
        let taken = indices.map { queue[$0] }
        for i in indices.reversed() { queue.remove(at: i) }
        return taken
    }

    /// Matched diagnostic control: the original FIFO policy, with identical phase accounting.
    private func takeFIFOFrontLocked() -> [Job] {
        if queue[0].phase == .prefill { return takePrefillGroupLocked(at: 0, maySkipDecode: false) }
        if queue[0].body != nil { return [queue.removeFirst()] }
        func frontRun() -> Int {
            var n = 0
            while n < queue.count && n < maxBatch && queue[n].req != nil { n += 1 }
            return n
        }
        if minBatch > 1 && steppable >= minBatch {
            let deadline = Date().addingTimeInterval(gatherWindow)
            while steppable >= minBatch && frontRun() < min(steppable, maxBatch) && Date() < deadline {
                if frontRun() < queue.count { break }
                if !cond.wait(until: deadline) { break }
            }
        }
        let n = frontRun()
        captureTraceLocked(reason: "fifo_not_in_scope", exitReady: n)
        let taken = Array(queue.prefix(n))
        queue.removeFirst(n)
        return taken
    }

    /// Original contiguous gathering for the diagnostic FIFO arm. Keep its incompatible-job
    /// stop and token cap unchanged so offset alignment only applies to the phase arm.
    private func takePrefillGroupLocked(at first: Int, maySkipDecode: Bool) -> [Job] {
        let tokenBudget = decoders > 0 ? min(prefillTokenBudget, prefillDecodeTokenBudget) : prefillTokenBudget
        guard prefillMaxBatch > 1, let initial = queue[first].prefill,
              initial.tokens <= tokenBudget else { return [queue.remove(at: first)] }
        var indices = [first]
        var tokens = initial.tokens
        for i in (first + 1)..<queue.count {
            if indices.count == prefillMaxBatch { break }
            let job = queue[i]
            if job.phase == .control { break }
            if job.phase == .decode {
                if maySkipDecode { continue }
                break
            }
            guard let work = job.prefill, work.group == initial.group,
                  work.tokens <= tokenBudget - tokens else { break }
            indices.append(i)
            tokens += work.tokens
        }
        let taken = indices.map { queue[$0] }
        for i in indices.reversed() { queue.remove(at: i) }
        return taken
    }
}

/// Live requests, so the server can refuse rather than run out of memory.
public final class ActiveCount: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    public init() {}
    public func enter(max: Int) -> Bool { lock.lock(); defer { lock.unlock() }; if n >= max { return false }; n += 1; return true }
    public func leave() { lock.lock(); n -= 1; lock.unlock() }
    public var current: Int { lock.lock(); defer { lock.unlock() }; return n }
}
nonisolated(unsafe) public let activeRequests = ActiveCount()
