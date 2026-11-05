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

/// T-0144: what a mixable prefill chunk tells the owner: the sequence and the chunk's token range [from, to).
public struct PrefillMix: Sendable, Equatable {
    public let key: Int, from: Int, to: Int
    /// more chunks of this prompt follow (the owner then holds briefly for the next one instead of a lone decode round)
    public let continues: Bool
    public init(key: Int, from: Int, to: Int, continues: Bool = false) {
        self.key = key; self.from = from; self.to = to; self.continues = continues
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
        /// T-0141: the submitting request's arrival stamp (`beginRequestPrefill`), read from the caller's thread at submit
        var order: Int? = nil
        /// T-0144: a prefill chunk that may ride in one forward with the queued verify steps
        var mix: PrefillMix? = nil
        var mixKey: Int? { mix?.key }
        /// T-0148: a decode closure (a lone row's serial MTP round) that may ride with a mixable chunk via `mixedSerial`
        var mixDecode = false
        /// T-0186: a short per-request setup closure ahead of the request's first chunk (cache lookup, MTP arming) or a new
        /// decoder's batch handover; with packing on it is dispatched before any chunk is picked, so the younger requests'
        /// chunks get queued and can pack, and the batched steps are not held behind a decode closure
        var setup = false
        /// T-0192: an image request's vision tower; with packing and ENGINE_PACK_TOWER_LOOKAHEAD it may run before an older
        /// request's chunk so the younger request's chunk is queued in time to pack with it
        var lookahead = false
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
    /// Decode affinity (see `decodeAffinity`): how often the owner waited for a decoder's next job, and how often it came.
    private var decodeAffinityWaits = 0, decodeAffinityHits = 0
    /// Decode dispatches since the last prefill dispatch, whether or not a prefill was queued at the time -- the burst the
    /// affinity wait counts. `consecutiveDecodeDispatches` restarts when no prefill is queued, and a lone long prefill
    /// resubmits its next chunk only after the previous one ends, so it read 0 at almost every decision: the first
    /// version waited 6 times in a whole 300k prefill (night A/B 2026-10-03).
    private var decodesSincePrefill = 0
    /// Time share (see `decodeShare`): the owner-measured work of the last prefill dispatch and the decode work since.
    private var lastPrefillWork = 0.0, decodeWorkSincePrefill = 0.0
    private let decodeShare: Double, decodeShareMinWork: Double
    private static let decodeShareCap = 64
    private let decodeAffinity: TimeInterval
    /// T-0122: ENGINE_DECODE_AFFINITY_FALLBACK=1 (see `affinityOpen` in takeFrontLocked); default off, the shipped trade
    private let decodeAffinityFallback: Bool = ProcessInfo.processInfo.environment["ENGINE_DECODE_AFFINITY_FALLBACK"] == "1"
    private var consecutivePrefillReorders = 0
    /// T-0141 (C-0407): ENGINE_PREFILL_ORDER=arrival. Every owner job a request submits during its prefill phase carries
    /// the request's arrival stamp; the prefill branch takes the oldest stamp first (unstamped jobs before all), so a burst
    /// runs its prompts to completion one after another instead of round-robin. A request resubmits only after its job
    /// ran, so with the oldest request in prefill and none of its jobs queued the owner waits up to `prefillContinue`
    /// for it. Off (the default) the FIFO order is untouched.
    /// A request whose prefill dispatches have used `prefillQuantum` of owner time goes behind every request then in
    /// prefill (round-robin with that quantum), so one huge prompt cannot hold the burst behind it for minutes.
    private let prefillOrderArrival: Bool
    private let prefillContinue: TimeInterval
    private let prefillQuantum: TimeInterval
    private var nextPrefillStamp = 0, nextPrefillRank = 0
    /// request stamp -> (rank: lower runs first, work: owner seconds of its prefill dispatches since its last rotation)
    private var activePrefill: [Int: (rank: Int, work: TimeInterval)] = [:]
    private var prefillQuantumRotations = 0
    /// T-0143 (C-0419): a control job may pass queued PREFILL jobs (never a decode job or decode closure). Control jobs in
    /// serving are short owner closures (sequence allocation, end-of-request statistics: 3-16 ms of work) that otherwise
    /// waited behind seconds of prefill chunks -- 207 ms mean, 15 s max on live traffic -- holding a finished request's
    /// response, and acting as a barrier that hid the oldest request's next chunk from the arrival order.
    private let controlPassesPrefill: Bool
    private var controlPassedPrefill = 0
    /// T-0200: a control job may also pass queued batched decode STEPS (never a decode closure). In the gather loop a control
    /// job ends the gathering (`barrier_incomplete`), so each one that lands while a round's steps arrive split the decode
    /// pool into two rounds: 31% of the >=8-decoder dispatches in W1 traces were such partial rounds, with ~7 control jobs
    /// per request plus every /v1/engine/sessions poll. A control job's submitting thread is blocked on it, so it has no step
    /// queued: running it first is the order an earlier arrival would have had. The steps then gather as one round.
    private let controlPassesDecode: Bool
    /// T-0207: a decode closure that arrives during the gathering of batched steps runs first (default off: a barrier)
    private let decodeClosurePasses: Bool
    private var decodeClosurePassed = 0
    private var controlPassedDecode = 0
    /// T-0144 (D-0043, C-0420): the mixed step. A prefill chunk submitted with a `mixKey`, when chosen while batched decode
    /// steps are queued in front of the next barrier, is dispatched TOGETHER with them: `mixedStep(key, steps)` runs the
    /// steps' round with one forward over [the chunk ; the verify rows] (the caller leaves the chunk's result for its body),
    /// then the chunk's own body runs. Decode then has had its round: the separate decode share after a mixed dispatch is
    /// `mixedDecodeShareKeep` of the usual one (0: none, the steps advance once per chunk).
    private let mixedStep: ((PrefillMix, [StepRequest]) -> [Int])?
    private let mixedWait: TimeInterval
    /// T-0185 (C-0587): after an arrival-order continuation wait (or an affinity wait), the re-entered pick still lets the
    /// mixed step wait for the decoders' steps (before, mayWait=false also forbade it and 9 of 32 burst chunks ran alone)
    private let mixedAfterContinue: Bool
    /// T-0185 (C-0585/C-0586): while a mixable chunk is queued and batched rows exist, the chunk (with every queued step) is
    /// taken before any separate decode round -- the expert weights are read once for prefill and decode
    private let mixedAll: Bool
    private let mixedDecodeShareKeep: Double
    private var mixedDispatches = 0, mixedRows = 0, mixedWaits = 0, mixedWaitHits = 0
    /// T-0184: why a chunk with a mix key ran without the decoders -- [lone: steppable < minBatch (no serial rider),
    /// barrier: fewer steps than the quorum before a control job / decode closure while more steps queue behind it,
    /// short: fewer steps queued than the quorum (some decoders had not resubmitted)]
    private var mixedMissLone = 0, mixedMissBarrier = 0, mixedMissShort = 0
    /// T-0185: picks where mix-all took the chunk although a decode round was queued and the share would have run it
    private var mixedAllForced = 0
    public var mixedAllForcedCount: Int { cond.lock(); defer { cond.unlock() }; return mixedAllForced }
    /// T-0186: packed prefill. When the owner picks a mixable chunk (alone or with the batched steps), it also takes the
    /// mixable chunks of OTHER requests queued before the next barrier, in prefill rank order, while the chunks' rows stay
    /// within `packRows` and their count within `packMax`; `packedStep(mixes, steps)` then computes all of them -- and the
    /// steps' round when there are steps -- in one forward, leaving each chunk's result for its body, and every chunk's body
    /// runs after it in rank order. packRows 0 = off.
    private let packRows: Int
    private let packMax: Int
    private let packedStep: (([PrefillMix], [StepRequest]) -> [Int])?
    private var packedDispatches = 0, packedChunks = 0, packedWithSteps = 0, setupFirst = 0, setupBreaks = 0
    public var setupBreaksCount: Int { cond.lock(); defer { cond.unlock() }; return setupBreaks }
    private var setupResume = false
    // T-0191 (QC counters), owner state consumed once per dispatch in loop()
    private var recordedMixed = ModelPhaseStats()
    private var pendingAffinityWait = false, decodeAffinityPreempted = 0
    private var pendingShareGrant = false, decodeShareGrants = 0, decodeShareGrantsAfterMixed = 0, lastPrefillMixed = false
    private var pendingSetupFirst: Bool? = nil
    private var setupFirstWork = 0.0, setupFirstMaxWork = 0.0, setupFirstLong = 0, setupFirstAheadOfSteps = 0, setupFirstWorkAheadOfSteps = 0.0
    private var pendingForcedMix = false
    private var prefillContinueEnded = 0, mixedAfterContinueWaits = 0, mixedAfterContinueHits = 0
    /// T-0190 (QC audit D1): a decode closure queued ahead of the batched steps (a new decoder's step installer, hot_rung,
    /// prearm/rearm, another row's serial round) is a barrier the steps cannot pass, and only the owner can run it -- so the
    /// mixed wait could only run out (25 ms) and the chunk went without the decoders. With `mixedUnblock` the wait stops at
    /// once, the closure runs, and the prefill pick resumes with the steps. Control jobs are left alone (ordering barriers).
    private let mixedUnblock: Bool
    private var unblockBreaks = 0
    /// T-0192: up to this many younger requests' vision towers run before an older request's mixable chunk is picked, so their
    /// chunks are queued in time to pack with it (a page's chunk otherwise waits only for its own tower and packs alone). 0 = off.
    private let towerLookahead: Int
    private let lookaheadHoldWindow: TimeInterval = 0.05
    private var lookaheadSinceChunk = 0, lookaheadTowers = 0, lookaheadHolds = 0, lookaheadPacked = 0
    /// after a lookahead tower: its request (stamp) is between its tower and its chunk; hold the chunk pick until that chunk is
    /// queued (its setups run first, setup-first) or the deadline passes -- otherwise the older chunk is picked in the gap
    private var lookaheadStamp: Int? = nil
    private var lookaheadDeadline = Date.distantPast
    public var lookaheadStats: (towers: Int, holds: Int, packed: Int) {
        cond.lock(); defer { cond.unlock() }; return (lookaheadTowers, lookaheadHolds, lookaheadPacked)
    }
    public var lookaheadTowersCount: Int { cond.lock(); defer { cond.unlock() }; return lookaheadTowers }
    /// T-0190 (W87 bisect): the pre-R3 accounting, where a setup closure counted as the last prefill dispatch and so shrank
    /// the decode share to ~0 until the next chunk -- an accidental decode throttle during waves; for the A/B only
    private let setupResetsShare: Bool
    public var unblockBreaksCount: Int { cond.lock(); defer { cond.unlock() }; return unblockBreaks }
    public var setupFirstCount: Int { cond.lock(); defer { cond.unlock() }; return setupFirst }
    private var packedSizes: [Int: Int] = [:]
    public var packedStats: (dispatches: Int, chunks: Int, withSteps: Int, sizes: [Int: Int]) {
        cond.lock(); defer { cond.unlock() }; return (packedDispatches, packedChunks, packedWithSteps, packedSizes)
    }
    public var mixedMissStats: (lone: Int, barrier: Int, short: Int) {
        cond.lock(); defer { cond.unlock() }; return (mixedMissLone, mixedMissBarrier, mixedMissShort)
    }
    /// T-0148 (D-0046): below the batch minimum a decoding row runs serial MTP rounds as decode closures and has no step to
    /// ride with a chunk. `mixedSerial(mix, round)` runs such a closure (submitted with `mixable`) with its verify forward
    /// replaced once by one over [chunk ; its block] -- the caller installs and clears that -- then the chunk's body runs.
    private let mixedSerial: ((PrefillMix, () -> Void) -> Void)?
    private var mixedSerialDispatches = 0
    /// after a mixed dispatch whose prompt continues (keep 0): until this deadline a decode step does not run ALONE ahead
    /// of the prompt's next chunk -- the chunk is resubmitted a moment later and the steps ride in it (a lone verify round
    /// re-reads every expert for a few rows, the cost the mixed step exists to avoid)
    private var mixedHoldUntil: Date? = nil
    private var mixedHolds = 0, mixedHoldHits = 0
    private var prefillContinueWaits = 0, prefillContinueHits = 0, prefillArrivalReorders = 0
    private static let prefillStampKey = "engine.modelthread.prefillStamp"
    /// T-0143 diagnostic (ENGINE_PREFILL_ORDER_TRACE=FILE): one line per owner dispatch -- uptime, phase, the stamp(s) run,
    /// work seconds, decoders, and for a prefill dispatch under arrival order how it was chosen (the oldest request in
    /// prefill, whether one of its jobs was queued, whether the owner was allowed to hold, the queued prefill stamps).
    /// Written on the owner after the work, outside the lock; off (nil) costs nothing.
    private let prefillOrderTrace: FileHandle? = {
        guard let path = ProcessInfo.processInfo.environment["ENGINE_PREFILL_ORDER_TRACE"] else { return nil }
        FileManager.default.createFile(atPath: path, contents: nil)
        return FileHandle(forWritingAtPath: path)
    }()
    private var prefillOrderNote = ""
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
    /// - decodeAffinity: right after a decode dispatch, with prefill waiting, the decode burst not used up and a decoder
    ///   registered, wait up to this long for that decoder's next job before the prefill runs (0 = never wait). A
    ///   decoding row submits its next step a moment after its last result, so without the wait it got ONE dispatch per
    ///   prefill chunk: 2.4-3.0 tok/s for a short chat while a 1M-token document prefilled in ~1.3 s chunks (2026-10-02).
    /// - decodeShare: 0 keeps the count rules above. Above 0, while a prefill waits, decode may run (and the affinity wait
    ///   may hold the owner for it) as long as its work since the last prefill dispatch stays under decodeShare x that
    ///   prefill dispatch's work, up to 64 dispatches; past the share the count bound applies as before. A long-context
    ///   chunk (~1.3 s at 1M) then leaves room for a decoder's rounds, while a short prompt's chunk leaves none -- a fixed
    ///   burst of 16 starved short-context prefill (N=8 aggregate 158.1 vs 169.7 tok/s, TTFT p50 4.72 vs 1.19 s).
    /// - decodeShareMinWork: the share opens only after a prefill dispatch at least this long. Share 0.5 on every dispatch
    ///   kept short-context aggregate but raised TTFT p50 at N=16 by ~1 s; short prompts' chunks (0.1-0.3 s) stay shipped.
    /// - runBatch: performs one decode step for the whole group and returns one token per request,
    ///   in the order given. Called ON the owner thread.
    public init(minBatch: Int, gatherWindow: TimeInterval, maxBatch: Int,
                phaseScheduling: Bool = true,
                maxConsecutiveDecodeDispatches: Int = 4,
                maxConsecutivePrefillReorders: Int = 4,
                prefillMaxBatch: Int = 1,
                prefillTokenBudget: Int = 4096,
                prefillDecodeTokenBudget: Int = 1024,
                decodeAffinity: TimeInterval = 0,
                decodeShare: Double = 0,
                decodeShareMinWork: Double = 0,
                prefillOrderArrival: Bool = false,
                prefillContinue: TimeInterval = 0.002,
                prefillQuantum: TimeInterval = 15,
                controlPassesPrefill: Bool = false,
                controlPassesDecode: Bool = false,
                decodeClosurePasses: Bool = false,
                mixedWait: TimeInterval = 0.005,
                mixedAfterContinue: Bool = false,
                mixedAll: Bool = false,
                mixedDecodeShareKeep: Double = 0,
                packRows: Int = 0,
                packMax: Int = 8,
                mixedUnblock: Bool = false,
                towerLookahead: Int = 0,
                setupResetsShare: Bool = false,
                packedStep: (([PrefillMix], [StepRequest]) -> [Int])? = nil,
                mixedStep: ((PrefillMix, [StepRequest]) -> [Int])? = nil,
                mixedSerial: ((PrefillMix, () -> Void) -> Void)? = nil,
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
        self.decodeAffinity = max(0, decodeAffinity)
        self.decodeShare = max(0, decodeShare)
        self.decodeShareMinWork = max(0, decodeShareMinWork)
        self.prefillOrderArrival = prefillOrderArrival
        self.prefillContinue = max(0, prefillContinue)
        self.prefillQuantum = prefillQuantum > 0 ? prefillQuantum : .infinity
        self.controlPassesPrefill = controlPassesPrefill
        self.controlPassesDecode = controlPassesDecode
        self.decodeClosurePasses = decodeClosurePasses
        self.mixedStep = mixedStep
        self.mixedSerial = mixedSerial
        self.mixedWait = max(0, mixedWait)
        self.mixedAfterContinue = mixedAfterContinue
        self.mixedAll = mixedAll
        self.mixedDecodeShareKeep = min(1, max(0, mixedDecodeShareKeep))
        self.packRows = max(0, packRows)
        self.mixedUnblock = mixedUnblock
        self.towerLookahead = max(0, towerLookahead)
        self.setupResetsShare = setupResetsShare
        self.packMax = max(1, packMax)
        self.packedStep = packedStep
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

    /// T-0192: a vision tower closure that the owner may run ahead of an older request's chunk (see `towerLookahead`).
    @discardableResult
    public func prefill<T: Sendable>(lookahead: Bool, _ body: () -> T) -> T {
        if isOwner || !lookahead { return prefill(body) }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(body) { escaped -> T in
            let job = Job(phase: .prefill, body: { box.value = escaped() }, req: nil)
            job.order = Self.currentPrefillStamp
            job.lookahead = true
            submitAndWait(job)
            return box.value!
        }
    }

    /// T-0186: a prefill-phase setup closure (see `Job.setup`); without packing an ordinary prefill closure.
    @discardableResult
    public func prefill<T: Sendable>(setup: Bool, _ body: () -> T) -> T {
        if isOwner || !setup { return prefill(body) }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(body) { escaped -> T in
            let job = Job(phase: .prefill, body: { box.value = escaped() }, req: nil)
            job.order = Self.currentPrefillStamp
            job.setup = true
            submitAndWait(job)
            return box.value!
        }
    }

    /// T-0144: one prefill chunk (`mix`: its sequence key and token range) that the owner may dispatch together with the queued batched decode
    /// steps (see `mixedStep`); without a `mixedStep` callback it is an ordinary prefill closure.
    @discardableResult
    public func prefill<T: Sendable>(mix: PrefillMix, _ body: () -> T) -> T {
        if isOwner { return body() }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(body) { escaped -> T in
            let job = Job(phase: .prefill, body: { box.value = escaped() }, req: nil)
            job.order = Self.currentPrefillStamp
            job.mix = mix
            submitAndWait(job)
            return box.value!
        }
    }

    /// (mixed dispatches, verify rows carried, waits for steps, waits that got them); all 0 without `mixedStep`
    public var mixedStats: (dispatches: Int, rows: Int, waits: Int, hits: Int, holds: Int, holdHits: Int, serial: Int) {
        cond.lock(); defer { cond.unlock() }
        return (mixedDispatches, mixedRows, mixedWaits, mixedWaitHits, mixedHolds, mixedHoldHits, mixedSerialDispatches)
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
                    let job = Job(phase: .prefill, body: { box.value = escaped() }, req: nil, prefill: work)
                    job.order = Self.currentPrefillStamp
                    submitAndWait(job)
                    return box.value!
                }
            }
        }
    }

    /// A serial/speculative decode round, with the same priority as a batched decode request.
    @discardableResult
    public func decode<T: Sendable>(traceLabel: String = "decode", traceKey: Int? = nil, mixable: Bool = false,
                                    setup: Bool = false, _ body: () -> T) -> T {
        run(.decode, traceLabel: traceLabel, traceKey: traceKey, mixDecode: mixable, setup: setup, body)
    }

    private func run<T: Sendable>(_ phase: Phase, traceLabel: String = "", traceKey: Int? = nil, mixDecode: Bool = false,
                                  setup: Bool = false, _ body: () -> T) -> T {
        if isOwner { return body() }
        let box = ResultBox<T>()
        return withoutActuallyEscaping(body) { escaped -> T in
            let job = Job(phase: phase, body: { box.value = escaped() }, req: nil)
            job.traceLabel = traceLabel; job.traceKey = traceKey
            job.mixDecode = mixDecode && phase == .decode
            job.setup = setup
            // every phase carries it: only prefill jobs are ranked by it, but the continuation wait must see the request's
            // control job (seq allocation) queued behind a barrier as 'its next job is queued'
            job.order = Self.currentPrefillStamp
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

    /// T-0141: start the calling thread's request prefill phase. Prefill jobs this thread submits until
    /// `endRequestPrefill` carry a fresh arrival stamp. Stamps are issued in call order, so call it when the request
    /// arrives at the model (before its first owner job). Idempotent per thread: a second call keeps the first stamp.
    @discardableResult
    public func beginRequestPrefill() -> Int {
        let dict = Thread.current.threadDictionary
        if let stamp = dict[Self.prefillStampKey] as? Int { return stamp }
        cond.lock()
        let stamp = nextPrefillStamp
        nextPrefillStamp += 1
        activePrefill[stamp] = (nextPrefillRank, 0)
        nextPrefillRank += 1
        cond.unlock()
        dict[Self.prefillStampKey] = stamp
        return stamp
    }

    /// T-0141: the calling thread's request left its prefill phase (first token, error or cancellation). Idempotent.
    public func endRequestPrefill() {
        let dict = Thread.current.threadDictionary
        guard let stamp = dict[Self.prefillStampKey] as? Int else { return }
        dict.removeObject(forKey: Self.prefillStampKey)
        cond.lock(); activePrefill[stamp] = nil; cond.broadcast(); cond.unlock()
    }

    private static var currentPrefillStamp: Int? { Thread.current.threadDictionary[prefillStampKey] as? Int }

    /// (continuation waits, hits, dispatches where arrival order took a prefill other than the queue's first, quantum
    /// rotations); all 0 off
    public var prefillOrderStats: (waits: Int, hits: Int, reorders: Int, rotations: Int, controlPassed: Int) {
        cond.lock(); defer { cond.unlock() }
        return (prefillContinueWaits, prefillContinueHits, prefillArrivalReorders, prefillQuantumRotations, controlPassedPrefill)
    }
    /// T-0200: control jobs run ahead of queued batched decode steps (0 with controlPassesDecode off)
    public var controlPassedDecodeCount: Int { cond.lock(); defer { cond.unlock() }; return controlPassedDecode }
    /// T-0207: decode closures run ahead of queued batched decode steps (0 with decodeClosurePasses off)
    public var decodeClosurePassedCount: Int { cond.lock(); defer { cond.unlock() }; return decodeClosurePassed }

    /// The rank a queued job is served by: unstamped and finished-request jobs first (they are short owner closures),
    /// then the request ranks.
    private func prefillRankLocked(_ job: Job) -> Int {
        guard let stamp = job.order, let entry = activePrefill[stamp] else { return Int.min }
        return entry.rank
    }
    /// (waits, hits) of the decode-affinity wait; both stay 0 while `decodeAffinity` is 0.
    public var decodeAffinityStats: (waits: Int, hits: Int) { cond.lock(); defer { cond.unlock() }; return (decodeAffinityWaits, decodeAffinityHits) }
    /// T-0191: affinity waits whose next dispatch was not a decode round; decode dispatches granted by the time share alone
    /// (the burst was used up and a prefill waited), and how many of those followed a mixed prefill dispatch
    public var decodeShareStats: (preempted: Int, grants: Int, grantsAfterMixed: Int) {
        cond.lock(); defer { cond.unlock() }; return (decodeAffinityPreempted, decodeShareGrants, decodeShareGrantsAfterMixed)
    }
    /// T-0191: setup-first dispatches (packing): total and max work seconds, how many took over 5 ms, and how many (and how
    /// much work) ran while batched steps were queued
    public var setupFirstStats: (seconds: Double, maxSeconds: Double, long: Int, aheadOfSteps: Int, secondsAheadOfSteps: Double) {
        cond.lock(); defer { cond.unlock() }
        return (setupFirstWork, setupFirstMaxWork, setupFirstLong, setupFirstAheadOfSteps, setupFirstWorkAheadOfSteps)
    }
    /// T-0191: continuation waits that ended because the request left prefill (not because its next job came);
    /// mixed waits entered right after a continuation or affinity wait (ENGINE_MIXED_AFTER_CONTINUE) and those that got the steps
    public var continueEndedCount: Int { cond.lock(); defer { cond.unlock() }; return prefillContinueEnded }
    public var mixedAfterContinueStats: (waits: Int, hits: Int) {
        cond.lock(); defer { cond.unlock() }; return (mixedAfterContinueWaits, mixedAfterContinueHits)
    }

    /// control/prefill/decode: jobs, queue waits and cancellations of that phase's jobs, and the dispatches and work of
    /// single-phase dispatches; "mixed": the dispatches and work of dispatches that carried prefill AND decode jobs (T-0191)
    public func phaseStats() -> [String: ModelPhaseStats] {
        cond.lock(); defer { cond.unlock() }
        var out = Dictionary(uniqueKeysWithValues: Phase.allCases.map { phase in
            var stats = recordedPhases[phase, default: ModelPhaseStats()]
            stats.queued = queue.reduce(0) { $0 + ($1.phase == phase ? 1 : 0) }
            return (phase.rawValue, stats)
        })
        out["mixed"] = recordedMixed
        return out
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

    // T-0164 instrument (ENGINE_OWNER_WAIT_STATS=1): wall time the owner thread spends in each of its waits (it dispatches
    // no GPU work meanwhile), logged every 10 s with the dispatch work time of the same window
    private static let waitStatsOn = ProcessInfo.processInfo.environment["ENGINE_OWNER_WAIT_STATS"] == "1"
    private var waitStats: [String: (n: Int, s: Double)] = [:]
    private var waitStatsSince = ProcessInfo.processInfo.systemUptime
    private var waitStatsWork = 0.0
    @discardableResult
    private func ownerWait(_ reason: String, until: Date? = nil) -> Bool {
        let t0 = ProcessInfo.processInfo.systemUptime
        let r: Bool
        if let until { r = cond.wait(until: until) } else { cond.wait(); r = true }
        if Self.waitStatsOn {
            let dt = ProcessInfo.processInfo.systemUptime - t0
            let e = waitStats[reason] ?? (0, 0)
            waitStats[reason] = (e.n + 1, e.s + dt)
        }
        return r
    }
    private func waitStatsMaybeLog(work: Double) {
        guard Self.waitStatsOn else { return }
        waitStatsWork += work
        let now = ProcessInfo.processInfo.systemUptime
        guard now - waitStatsSince >= 10 else { return }
        let span = now - waitStatsSince
        let parts = waitStats.sorted { $0.key < $1.key }.map { String(format: "%@ %.0f ms/s (n %d)", $0.key, $0.value.s / span * 1e3, $0.value.n) }
        FileHandle.standardError.write(String(format: "engine owner waits over %.1f s: dispatch work %.0f ms/s; ", span, waitStatsWork / span * 1e3)
            .appending(parts.joined(separator: ", ")).appending("\n").data(using: .utf8)!)
        waitStats = [:]; waitStatsWork = 0; waitStatsSince = now
    }

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
                ownerWait("empty")
            }
            if queue.isEmpty { cond.unlock(); return }          // stopping, and drained
            if ownerTrace != nil { traceDispatchID += 1; captureTraceLocked(reason: "not_step") }
            let batch = takeFrontLocked()
            var selectedTrace = traceSelection
            if ownerTrace != nil { selectedTrace?.selected = batch.map(traceJob) }
            let phase = batch[0].phase
            let startedAt = ProcessInfo.processInfo.systemUptime
            // T-0191 (QC counters): each job's count and queue wait go to ITS phase -- the decode steps riding in a mixed
            // or packed dispatch used to be charged to prefill (41 % of "prefill jobs" in t0241); the dispatch and its work
            // go to "mixed" when the batch carries both phases
            for job in batch {
                let wait = max(0, startedAt - job.enqueuedAt)
                var m = recordedPhases[job.phase, default: ModelPhaseStats()]
                m.jobs += 1
                m.queueWaitSeconds += wait
                m.maxQueueWaitSeconds = max(m.maxQueueWaitSeconds, wait)
                recordedPhases[job.phase] = m
            }
            let carriesBoth = batch.contains { $0.phase == .prefill } && batch.contains { $0.phase == .decode }
            var cancelledSteps = 0
            let wasAffinityWait = pendingAffinityWait
            pendingAffinityWait = false
            var mixedRan = false
            let packCount = batch.prefix(while: { $0.mix != nil && $0.req == nil && !$0.mixDecode }).count
            if packCount >= 2, let packedStep {
                precondition(batch.dropFirst(packCount).allSatisfy { $0.req != nil }, "a packed dispatch carries chunks, then batched steps")
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                let chunks = Array(batch.prefix(packCount))
                let live = batch.dropFirst(packCount).filter { job in
                    if job.cancelled() { job.result = StepResult(token: -1, cancelled: true); return false }
                    return true
                }
                // one forward for every chunk (and the steps' round), the snapshot before anything can touch the pool, then
                // each chunk's own body in rank order (each consumes its prepared forward or computes its chunk itself)
                let results: [StepResult] = autoreleasepool {
                    let tokens = packedStep(chunks.map { $0.mix! }, live.map { $0.req! })
                    let r = live.enumerated().map { i, j in
                        let token = i < tokens.count ? tokens[i] : -1
                        return snapshot?(j.req!, token) ?? StepResult(token: token)
                    }
                    for c in chunks { c.body!() }
                    return r
                }
                cond.lock()
                cancelledSteps += batch.count - packCount - live.count
                packedDispatches += 1; packedChunks += packCount; packedSizes[packCount, default: 0] += 1
                if !live.isEmpty {
                    recordedStepCount += 1; recordedSizeHistogram[live.count, default: 0] += 1
                    mixedDispatches += 1; mixedRows += live.count; mixedRan = true; packedWithSteps += 1
                }
                for (j, result) in zip(live, results) { j.result = result }
            } else if batch.count == 2, let mix = batch[0].mix, batch[1].req == nil, batch[1].mixDecode, let mixedSerial {
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                // the lone row's serial round first (its verify forward also computes the chunk), then the chunk's body
                autoreleasepool {
                    mixedSerial(mix) { batch[1].body!() }
                    batch[0].body!()
                }
                cond.lock()
                mixedDispatches += 1; mixedRows += 1; mixedSerialDispatches += 1; mixedRan = true
            } else if batch.count > 1, let mix = batch[0].mix, let mixedStep {
                precondition(batch.dropFirst().allSatisfy { $0.req != nil }, "a mixed step carries batched steps only (T-0148)")
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                let steps = Array(batch.dropFirst())
                let live = steps.filter { job in
                    if job.cancelled() { job.result = StepResult(token: -1, cancelled: true); return false }
                    return true
                }
                // the steps' round first (its forward also computes the chunk, left for the chunk's body), the snapshot
                // before anything else can touch the pool, then the chunk's own body
                let results: [StepResult] = autoreleasepool {
                    let tokens = live.isEmpty ? [] : mixedStep(mix, live.map { $0.req! })
                    let r = live.enumerated().map { i, j in
                        let token = i < tokens.count ? tokens[i] : -1
                        return snapshot?(j.req!, token) ?? StepResult(token: token)
                    }
                    batch[0].body!()
                    return r
                }
                cond.lock()
                cancelledSteps += steps.count - live.count
                if !live.isEmpty {
                    recordedStepCount += 1; recordedSizeHistogram[live.count, default: 0] += 1
                    mixedDispatches += 1; mixedRows += live.count; mixedRan = true
                }
                for (j, result) in zip(live, results) { j.result = result }
            } else if phase == .prefill, batch.count > 1, batch[0].prefill != nil {
                cond.unlock()
                if let selectedTrace { ownerTrace?(selectedTrace) }
                // Every job runs inside its own autorelease pool: this thread lives as long as the server and never drains
                // one by itself, so Foundation/Metal objects autoreleased during a job (strings above all) piled up for the
                // process lifetime -- ~986k CFStrings after one morning of batched traffic (2026-10-03, T-0050).
                let eligible: [PrefillWork] = autoreleasepool {
                    let eligible = batch.compactMap { job -> PrefillWork? in
                        guard let work = job.prefill, work.prepare() else { return nil }
                        return work
                    }
                    if eligible.count >= 2 { eligible[0].batch(eligible.map(\.key)) }
                    for job in batch { job.body!() }
                    return eligible
                }
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
                autoreleasepool { body() }
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
                // Snapshot and drain while still on the owner, BEFORE another job can dissolve
                // the pool. Only immutable Sendable values return to connection threads.
                let results: [StepResult] = autoreleasepool {
                    let tokens = live.isEmpty ? [] : runBatch(live.map { $0.req! })
                    return live.enumerated().map { i, j in
                        let token = i < tokens.count ? tokens[i] : -1
                        return snapshot?(j.req!, token) ?? StepResult(token: token)
                    }
                }
                cond.lock()
                cancelledSteps += batch.count - live.count
                if !live.isEmpty { recordedStepCount += 1; recordedSizeHistogram[live.count, default: 0] += 1 }
                for (j, result) in zip(live, results) { j.result = result }
            }
            let work = max(0, ProcessInfo.processInfo.systemUptime - startedAt)
            if let trace = prefillOrderTrace {
                let stamps = batch.map { $0.order.map(String.init) ?? "-" }.joined(separator: ",")
                // T-0183: what the dispatch carried -- kind, batched decode steps, the prefill chunk's tokens (a PrefillMix)
                let steps = batch.filter { $0.req != nil }.count
                let chunk = batch.compactMap { $0.req == nil ? $0.mix.map { $0.to - $0.from } : nil }.reduce(0, +)
                let kind = packCount >= 2 ? (mixedRan ? "packed-mixed" : "packed")
                    : (mixedRan ? "mixed" : (steps > 0 ? "step" : (batch[0].mix != nil ? "chunk" : "closure")))
                let line = String(format: "%.4f %@ n=%d stamps=%@ work=%.4f decoders=%d kind=%@ steps=%d chunk=%d %@\n", startedAt, phase.rawValue, batch.count,
                                  stamps, work, decoders, kind, steps, chunk, phase == .prefill ? prefillOrderNote : "")
                prefillOrderNote = ""
                cond.unlock(); trace.write(Data(line.utf8)); cond.lock()
            }
            // T-0190 (QC audit R3): a setup closure is not prefill work -- it must not reset the decode share's yardstick
            if phase == .prefill, !batch[0].setup || setupResetsShare { lastPrefillWork = work; decodeWorkSincePrefill = 0 }
            if mixedRan {
                // the decoders had their round inside the chunk: only `mixedDecodeShareKeep` of the separate share remains
                decodeWorkSincePrefill = decodeShare * lastPrefillWork * (1 - mixedDecodeShareKeep)
                if mixedDecodeShareKeep == 0 {
                    consecutiveDecodeDispatches = maxConsecutiveDecodeDispatches
                    if batch[0].mix?.continues == true { mixedHoldUntil = Date().addingTimeInterval(mixedWait) }
                }
            }
            if phase == .prefill, prefillOrderArrival {
                // T-0190 (QC audit R4): a packed dispatch charges each of its requests its share of the forward, not the
                // whole forward (N packed requests used to use up the quantum N times too fast); steps riding along are
                // decode work and charge nothing here
                let stamps = Set(batch.filter { $0.req == nil }.compactMap(\.order))
                for stamp in stamps where activePrefill[stamp] != nil {
                    activePrefill[stamp]!.work += work / Double(stamps.count)
                    if activePrefill[stamp]!.work >= prefillQuantum {
                        activePrefill[stamp] = (nextPrefillRank, 0)      // behind every request now in prefill
                        nextPrefillRank += 1
                        prefillQuantumRotations += 1
                    }
                }
            }
            else if phase == .decode { decodeWorkSincePrefill += work }
            if cancelledSteps > 0 { recordedPhases[.decode, default: ModelPhaseStats()].cancelledJobs += cancelledSteps }
            var metrics = carriesBoth ? recordedMixed : recordedPhases[phase, default: ModelPhaseStats()]
            metrics.dispatches += 1
            metrics.workSeconds += work
            waitStatsMaybeLog(work: work)
            metrics.maxWorkSeconds = max(metrics.maxWorkSeconds, work)
            if carriesBoth { recordedMixed = metrics } else { recordedPhases[phase] = metrics }
            // T-0191: an affinity wait was "hit" only if the dispatch it led to really was a decode round
            if wasAffinityWait {
                // a decode round alone, or the waited-for steps riding in a mixed/packed forward (review of b2eeed2)
                if (phase == .decode && !batch[0].setup) || mixedRan { decodeAffinityHits += 1 } else { decodeAffinityPreempted += 1 }
            }
            // T-0191: a decode dispatch the time share alone allowed (the burst was used up, a prefill was waiting)
            if pendingShareGrant {
                if phase == .decode { decodeShareGrants += 1; if lastPrefillMixed { decodeShareGrantsAfterMixed += 1 } }
                pendingShareGrant = false
            }
            if phase == .prefill, !batch[0].setup { lastPrefillMixed = mixedRan }
            // T-0192: a chunk dispatch ends the lookahead allowance; a lookahead tower starts the hold for its request's chunk
            if batch.contains(where: { $0.mix != nil && $0.req == nil }) {
                if lookaheadSinceChunk > 0, packCount >= 2 { lookaheadPacked += 1 }
                lookaheadSinceChunk = 0; lookaheadStamp = nil
            }
            if batch[0].lookahead, lookaheadSinceChunk > 0 {
                lookaheadStamp = batch[0].order; lookaheadDeadline = Date().addingTimeInterval(lookaheadHoldWindow)
            }
            // T-0191: setup-first dispatches (packing) -- how long they really are, and whether steps were waiting behind them
            if let ahead = pendingSetupFirst {
                setupFirstWork += work; setupFirstMaxWork = max(setupFirstMaxWork, work)
                if work > 0.005 { setupFirstLong += 1 }
                if ahead { setupFirstAheadOfSteps += 1; setupFirstWorkAheadOfSteps += work }
                pendingSetupFirst = nil
            }
            // T-0191: mix-all counted once per dispatch that really mixed because of it (was: every takeFrontLocked entry)
            if pendingForcedMix { if mixedRan { mixedAllForced += 1 }; pendingForcedMix = false }
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
    private func takeFrontLocked(mayWait: Bool = true, mixWait: Bool? = nil) -> [Job] {
        if !phaseScheduling { return takeFIFOFrontLocked() }
        if controlPassesPrefill || controlPassesDecode, let c = queue.firstIndex(where: { $0.phase == .control }), c > 0,
           !queue[..<c].contains(where: { $0.phase == .decode && (!controlPassesDecode || $0.body != nil) }),
           controlPassesPrefill || !queue[..<c].contains(where: { $0.phase == .prefill }) {
            if queue[..<c].contains(where: { $0.phase == .decode }) { controlPassedDecode += 1 } else { controlPassedPrefill += 1 }
            consecutiveDecodeDispatches = 0
            consecutivePrefillReorders = 0
            decodesSincePrefill = 0
            return [queue.remove(at: c)]
        }
        if queue[0].phase == .control {
            consecutiveDecodeDispatches = 0
            consecutivePrefillReorders = 0
            decodesSincePrefill = 0
            return [queue.removeFirst()]
        }
        let barrier = queue.firstIndex { $0.phase == .control } ?? queue.count
        let prefill = queue[..<barrier].firstIndex { $0.phase == .prefill }
        let decode = queue[..<barrier].firstIndex { $0.phase == .decode }
        if prefill == nil { consecutivePrefillReorders = 0; setupResume = false }
        // T-0186: with packing on, the queued setup closures (oldest first) run before anything else is picked -- a
        // request's prefill setup (so its chunk gets queued in time to pack) and a new decoder's batch handover (a decode
        // closure is a barrier to the batched steps, which then could not ride the packed forward; C-0596). Both are short.
        if packRows > 0, let s = queue[..<barrier].indices.filter({ queue[$0].setup && queue[$0].body != nil })
            .min(by: { prefillRankLocked(queue[$0]) < prefillRankLocked(queue[$1]) }) {
            setupFirst += 1
            pendingSetupFirst = queue[..<barrier].contains { $0.req != nil }      // T-0191: steps waiting behind it?
            return [queue.remove(at: s)]
        }
        // the time share: decode work since the last prefill dispatch under decodeShare x that dispatch's work
        if let until = mixedHoldUntil {
            if prefill == nil, decode != nil, mayWait, Date() < until {
                mixedHolds += 1
                while !stopping {
                    let end = queue.firstIndex { $0.phase == .control } ?? queue.count
                    if queue[..<end].contains(where: { $0.phase == .prefill }) { mixedHoldHits += 1; break }
                    if !ownerWait("mixed_hold", until: until) { break }
                }
                mixedHoldUntil = nil
                return takeFrontLocked(mayWait: false)
            }
            if prefill != nil || Date() >= until { mixedHoldUntil = nil }
        }
        let shareOpen = decodeShare > 0 && lastPrefillWork >= decodeShareMinWork && decodesSincePrefill < Self.decodeShareCap
            && decodeWorkSincePrefill < decodeShare * lastPrefillWork
        let forceMix = mixedAll && mixedStep != nil && steppable >= minBatch
            && queue[..<barrier].contains(where: { $0.phase == .prefill && $0.mixKey != nil })
        let resume = setupResume && (packRows > 0 || mixedUnblock)
        let burstSpent = consecutiveDecodeDispatches >= maxConsecutiveDecodeDispatches
        if let prefill, decode == nil || forceMix || resume || (burstSpent && !shareOpen) {
            setupResume = false
            // T-0191: mix-all counts only a dispatch it alone sent to prefill (decode queued, no resume, share/burst allowed decode)
            pendingForcedMix = forceMix && decode != nil && !resume && !(burstSpent && !shareOpen)
            // T-0122 (C-0215): decodeAffinityFallback -- with the time share closed (a short prefill dispatch), the affinity wait
            // falls back to the count rule (the burst of maxConsecutiveDecodeDispatches) instead of not waiting at all
            let affinityOpen = decodeShare > 0 ? (shareOpen || (decodeAffinityFallback && decodesSincePrefill < maxConsecutiveDecodeDispatches))
                                               : decodesSincePrefill < maxConsecutiveDecodeDispatches
            if mayWait, decode == nil, decodeAffinity > 0, decoders > 0, decodesSincePrefill > 0, affinityOpen {
                decodeAffinityWaits += 1
                let deadline = Date().addingTimeInterval(decodeAffinity)
                while !stopping {
                    let end = queue.firstIndex { $0.phase == .control } ?? queue.count
                    // T-0191: a decode job came; whether the wait paid off is decided by the dispatch that follows (loop())
                    if queue[..<end].contains(where: { $0.phase == .decode }) { pendingAffinityWait = true; break }
                    if !ownerWait("affinity", until: deadline) { break }
                }
                return takeFrontLocked(mayWait: false, mixWait: mixedAfterContinue ? true : nil)       // the queue may have changed while the lock was released
            }
            if prefillOrderArrival {
                if prefillOrderTrace != nil {
                    let oldest = activePrefill.min(by: { $0.value.rank < $1.value.rank })?.key
                    let queued = queue[..<barrier].filter { $0.phase == .prefill }.map { $0.order.map(String.init) ?? "-" }
                    prefillOrderNote = "oldest=\(oldest.map(String.init) ?? "-") oldest_queued=\(oldest.map { o in queue.contains { $0.order == o } } ?? false) "
                        + "may_wait=\(mayWait) barrier=\(barrier)/\(queue.count) queued_prefill=[\(queued.joined(separator: ","))]"
                }
                // the oldest request still in prefill: when none of its jobs is queued it is between two of them (it
                // resubmits only after the last one ran), so hold the owner for it briefly instead of serving a younger one
                if mayWait, prefillContinue > 0, let oldest = activePrefill.min(by: { $0.value.rank < $1.value.rank })?.key,
                   !queue.contains(where: { $0.order == oldest }) {
                    prefillContinueWaits += 1
                    let deadline = Date().addingTimeInterval(prefillContinue)
                    while !stopping {
                        if queue.contains(where: { $0.order == oldest }) { prefillContinueHits += 1; break }
                        if activePrefill[oldest] == nil { prefillContinueEnded += 1; break }   // T-0191: it left prefill
                        if !ownerWait("prefill_continue", until: deadline) { break }
                    }
                    return takeFrontLocked(mayWait: false, mixWait: mixedAfterContinue ? true : nil)     // the queue may have changed while the lock was released
                }
                if let st = lookaheadStamp, activePrefill[st] != nil, Date() < lookaheadDeadline,
                   !queue.contains(where: { $0.order == st && $0.mix != nil }) {
                    lookaheadHolds += 1
                    _ = ownerWait("lookahead", until: min(lookaheadDeadline, Date().addingTimeInterval(0.005)))
                    // the same allowance as the continuation wait (C-0587): the chunk picked after the hold may still wait
                    // for the decoders' steps -- without it the packed pages went without them (the P251 QC-3 rollback)
                    return takeFrontLocked(mayWait: false, mixWait: mixedAfterContinue ? true : nil)
                }
                lookaheadStamp = nil
                let chosen = arrivalPrefillIndexLocked(before: barrier) ?? prefill
                // T-0192: before an older request's chunk, run a younger request's queued vision tower (at most towerLookahead
                // per chunk dispatch), so that request's chunk is queued in time to pack with the older one
                if packRows > 0, towerLookahead > 0, lookaheadSinceChunk < towerLookahead, queue[chosen].mix != nil {
                    let r0 = prefillRankLocked(queue[chosen])
                    if let t = queue[..<barrier].indices.filter({ queue[$0].lookahead && prefillRankLocked(queue[$0]) > r0 })
                        .min(by: { prefillRankLocked(queue[$0]) < prefillRankLocked(queue[$1]) }) {
                        lookaheadSinceChunk += 1; lookaheadTowers += 1
                        return [queue.remove(at: t)]
                    }
                }
                if chosen != prefill { prefillArrivalReorders += 1 }
                consecutiveDecodeDispatches = 0
                decodesSincePrefill = 0
                if let mixed = takeMixedLocked(at: chosen, mayWait: mixWait ?? mayWait, viaContinue: mixWait == true && !mayWait) { return packLocked(mixed) }
                return packLocked(takeAlignedPrefillGroupLocked(at: chosen, reorder: false))
            }
            consecutiveDecodeDispatches = 0
            decodesSincePrefill = 0
            if let mixed = takeMixedLocked(at: prefill, mayWait: mixWait ?? mayWait, viaContinue: mixWait == true && !mayWait) { return packLocked(mixed) }
            return packLocked(takeAlignedPrefillGroupLocked(at: prefill))
        }
        // There is a decode before the barrier, otherwise the prefill branch above handled it.
        let first = decode!
        // T-0191: the burst was used up and a prefill waited -- this decode dispatch exists only because the time share is open
        if prefill != nil, burstSpent, shareOpen { pendingShareGrant = true }
        decodesSincePrefill += 1
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
                // T-0200: a control job that arrived during the gathering runs now and the gathering starts again, instead of
                // the steps queued so far leaving as a partial round (only control jobs: a decode closure stays a barrier)
                // T-0207: a decode closure likewise, with decodeClosurePasses: its submitting thread is blocked in it, so it has no
                // step in this round, and a closure queued ahead of the steps already runs first today
                if controlPassesDecode || decodeClosurePasses, ready.blocked, ready.indices.count < min(steppable, maxBatch),
                   let c = queue.firstIndex(where: { $0.phase == .control || ($0.phase == .decode && $0.body != nil) }),
                   queue[c].phase == .control ? controlPassesDecode : decodeClosurePasses {
                    if queue[c].phase == .control { controlPassedDecode += 1 } else { decodeClosurePassed += 1 }
                    consecutiveDecodeDispatches = 0
                    consecutivePrefillReorders = 0
                    decodesSincePrefill = 0
                    return [queue.remove(at: c)]
                }
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
                if !ownerWait("gather", until: deadline) { traceReason = "wait_timeout"; break }
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
    /// T-0144: the chosen prefill job together with the batched steps queued before the next barrier, when it is a mixable
    /// chunk and a `mixedStep` exists; nil = dispatch it as an ordinary prefill. With decoders registered but no step queued
    /// yet (a decoder resubmits a moment after its result) the owner may hold up to `mixedWait` for them.
    private func takeMixedLocked(at index: Int, mayWait: Bool, viaContinue: Bool = false) -> [Job]? {
        guard mixedStep != nil || mixedSerial != nil, queue[index].mixKey != nil else { return nil }
        if steppable < minBatch {
            // T-0148: no batched rows -- a lone row's serial MTP round may ride instead
            guard mixedSerial != nil else { if decoders > 0 { mixedMissLone += 1 }; return nil }
            func serialIndex() -> Int? {
                for (i, job) in queue.enumerated() {
                    if job.phase == .control { return nil }
                    if job.phase == .decode, job.body != nil, job.mixDecode { return i }
                }
                return nil
            }
            if let d = serialIndex() { return removeMixedLocked(chunk: index, steps: [d]) }
            guard mayWait, mixedWait > 0, decoders > 0 else { return nil }
            mixedWaits += 1
            let key = queue[index].mixKey
            let deadline = Date().addingTimeInterval(mixedWait)
            while !stopping {
                // T-0191: a hit when the wait got the chunk a rider -- the serial one here, or (steppable reached minBatch)
                // the full batched set the re-pick below takes; a re-pick without riders is the batched path's miss
                if serialIndex() != nil { mixedWaitHits += 1; break }
                if steppable >= minBatch { break }
                if !ownerWait("mixed_wait", until: deadline) { break }
            }
            guard let again = queue.firstIndex(where: { $0.mixKey == key && $0.phase == .prefill }) else { return takeFrontLocked(mayWait: false) }
            let handover = serialIndex() == nil && steppable >= minBatch
            if let r = takeMixedLocked(at: again, mayWait: false) {
                if handover, r.count > 1 { mixedWaitHits += 1 }
                return r
            }
            return [queue.remove(at: again)]
        }
        guard mixedStep != nil else { mixedMissLone += 1; return nil }
        // C-0425: only the FULL set of batched rows may ride -- a partial set changes the pool's membership, which
        // dissolves and re-stacks it and runs a lone step without an MTP round (the forward is not consumed). Rows below
        // the batch minimum decode serially and cannot ride at all.
        guard steppable >= minBatch else { return nil }
        func quorum() -> Int { min(steppable, maxBatch) }
        func stepIndices() -> [Int] {
            var out: [Int] = []
            for (i, job) in queue.enumerated() {
                if job.phase == .control || (job.phase == .decode && job.body != nil) { break }
                if job.req != nil { out.append(i); if out.count == maxBatch { break } }
            }
            return out
        }
        let steps = stepIndices()
        if steps.count >= quorum(), quorum() > 0 { return removeMixedLocked(chunk: index, steps: steps) }
        guard mayWait, mixedWait > 0 else {
            if queue.filter({ $0.req != nil }).count > steps.count { mixedMissBarrier += 1 } else { mixedMissShort += 1 }
            return nil
        }
        mixedWaits += 1
        let key = queue[index].mixKey
        let deadline = Date().addingTimeInterval(mixedWait)
        if viaContinue { mixedAfterContinueWaits += 1 }      // T-0191: the wait ENGINE_MIXED_AFTER_CONTINUE allows
        // T-0190 (QC audit D2): only before the barrier -- setup-first dispatches from there, so a setup behind a control
        // job used to end the wait and skip it without being run
        func barrierIndex() -> Int { queue.firstIndex { $0.phase == .control } ?? queue.count }
        func setupQueued() -> Bool { packRows > 0 && queue[..<barrierIndex()].contains { $0.setup && $0.body != nil } }
        /// T-0190 (D1): the decode closure the steps are stuck behind, if any (before the first control job)
        func blocker() -> Int? {
            guard mixedUnblock else { return nil }
            for (i, job) in queue.enumerated() {
                if job.phase == .control { return nil }
                if job.phase == .decode && job.body != nil { return i }
            }
            return nil
        }
        while !stopping {
            if steppable < minBatch { break }
            if stepIndices().count >= quorum() { mixedWaitHits += 1; if viaContinue { mixedAfterContinueHits += 1 }; break }
            if setupQueued() { break }                 // T-0186: a new decoder's handover arrived; it runs first
            if blocker() != nil { break }              // T-0190: only the owner can run it; waiting cannot help
            if !ownerWait("mixed_wait", until: deadline) { break }
        }
        // T-0186: the handover first (the prefill branch dispatches it), then the chunk is picked again with the steps
        if setupQueued(), stepIndices().count < quorum() {
            setupBreaks += 1; setupResume = true         // the prefill pick stands: it is taken up again after the setup
            return takeFrontLocked(mayWait: false)
        }
        if let b = blocker(), stepIndices().count < quorum() {
            unblockBreaks += 1; setupResume = true       // the closure now, then the same prefill pick with the steps
            return [queue.remove(at: b)]
        }
        // the queue may have changed: find the chunk again (the same job: its caller is blocked on it)
        guard let again = queue.firstIndex(where: { $0.mixKey == key && $0.phase == .prefill }) else { return takeFrontLocked(mayWait: false) }
        let now = stepIndices()
        if steppable >= minBatch, now.count >= quorum(), quorum() > 0 { return removeMixedLocked(chunk: again, steps: now) }
        // T-0184: classify the miss -- are more step requests queued behind a control job / decode closure?
        if steppable < minBatch { mixedMissLone += 1 }
        else if queue.filter({ $0.req != nil }).count > now.count { mixedMissBarrier += 1 }
        else { mixedMissShort += 1 }
        prefillOrderNote += " miss lone=\(mixedMissLone) barrier=\(mixedMissBarrier) short=\(mixedMissShort) steps_before_barrier=\(now.count) steps_queued=\(queue.filter { $0.req != nil }.count) quorum=\(quorum())"
        return [queue.remove(at: again)]        // no full set: an ordinary prefill, the steps keep their own gather
    }

    /// T-0186: add the other requests' queued mixable chunks to a dispatch whose first job is a mixable chunk (see packRows).
    /// A serial rider (T-0148) or a native prefill group is left as it is.
    private func packLocked(_ taken: [Job]) -> [Job] {
        guard packRows > 0, packedStep != nil, let first = taken.first, first.mix != nil, first.prefill == nil,
              !taken.contains(where: { $0.mixDecode }) else { return taken }
        // a recursive pick may already have packed: count every chunk taken so far
        let have = taken.prefix(while: { $0.mix != nil && $0.req == nil })
        var rows = have.reduce(0) { $0 + $1.mix!.to - $1.mix!.from }
        var keys = Set(have.map { $0.mix!.key })
        let barrier = queue.firstIndex { $0.phase == .control } ?? queue.count
        let candidates = (0 ..< barrier).filter { queue[$0].phase == .prefill && queue[$0].mix != nil && queue[$0].prefill == nil }
            .map { (i: $0, rank: prefillRankLocked(queue[$0])) }
            .sorted { $0.rank != $1.rank ? $0.rank < $1.rank : $0.i < $1.i }
        var extra: [Int] = []
        for c in candidates {
            if have.count + extra.count >= packMax { break }
            let m = queue[c.i].mix!
            guard !keys.contains(m.key) else { continue }
            if rows + (m.to - m.from) > packRows { break }        // rank order is kept: no later chunk jumps a larger one
            rows += m.to - m.from; keys.insert(m.key); extra.append(c.i)
        }
        guard !extra.isEmpty else { return taken }
        let jobs = extra.map { queue[$0] }
        for i in extra.sorted(by: >) { queue.remove(at: i) }
        return Array(have) + jobs + taken.dropFirst(have.count)
    }

    private func removeMixedLocked(chunk: Int, steps: [Int]) -> [Job] {
        let taken = [queue[chunk]] + steps.map { queue[$0] }
        for i in ([chunk] + steps).sorted(by: >) { queue.remove(at: i) }
        return taken
    }

    /// T-0141: the prefill job before `barrier` with the smallest arrival stamp; unstamped jobs (no request prefill phase)
    /// come first, in queue order. Ties keep queue order, so one request's jobs stay in its own submission order.
    private func arrivalPrefillIndexLocked(before barrier: Int) -> Int? {
        var best: Int? = nil, bestRank = Int.max
        for i in 0 ..< barrier where queue[i].phase == .prefill {
            let r = prefillRankLocked(queue[i])
            if best == nil || r < bestRank { best = i; bestRank = r }
        }
        return best
    }

    private func takeAlignedPrefillGroupLocked(at oldest: Int, reorder: Bool = true) -> [Job] {
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
        for i in candidates where reorder && queue[i].prefill!.position < queue[first].prefill!.position {
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
                if !ownerWait("gather", until: deadline) { break }
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
nonisolated(unsafe) public let activeRequests = RequestGate()   // P116: FIFO wait when --queue-max > 0
