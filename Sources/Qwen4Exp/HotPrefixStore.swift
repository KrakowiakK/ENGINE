// P093 -- the HOT prefix store: finished sequences kept in GPU memory, resumable at any rung.
//
// The disk store (PrefixStore.swift) is what makes a coding agent's 18k-token fixed head free across
// sessions. It is the wrong instrument for the turn-to-turn continuation of ONE conversation, and the
// live server showed exactly how (runs/p093):
//
//   - a hit costs a disk read of the whole prefix, twice (the trunk import and the MTP arming each
//     called `load`), and the read competes for the page cache with a 190 GB process: 0.5 s at 20k
//     tokens, 1.0-1.5 s at 90k, and 4-6 s outliers on a 160-token delta;
//   - rungs sit at multiples of 512, so every turn re-prefills up to 511 tokens it had already seen;
//   - nothing is written DURING decode, so the reasoning a request generates -- the bulk of an
//     agent's tokens, and per this checkpoint's own chat template part of the next prompt -- is
//     prefilled again from scratch on the next turn.
//
// This store keeps, per finished sequence, ONE copy of the position-indexed rows (attention K/V,
// indexer keys, pooled blocks -- append-only, so row r is the same bytes at every later length) and a
// LADDER of rungs, each just the fixed-size non-row state (the GDN recurrent state, the PLE/n-gram
// context: ~115 MB) at a particular length: the end of the prefill, every `rungStep` tokens of
// decode, and the final state. The next prompt is matched by its longest common token prefix, and
// the highest rung at or below it is materialised by SLICING the shared rows to that length. So a
// continuation that re-tokenises identically resumes at the final state with zero re-prefill, and
// one that drifts (a tool call whose parameters re-render in a different order) resumes at the last
// rung before the drift instead of at the start of the turn.
//
// Ownership: every method runs on the model thread (the server asserts it); nothing here locks.
import Foundation
import Darwin
import MLX
import MLXLMCommon

public final class HotPrefixStore {
    /// A construction certificate, never inferred from a divisible length. The next token
    /// is the actual last shifted MTP input, which may be absent from Entry.tokens.
    public struct CanonicalPrefillOrigin: Equatable, Sendable {
        public enum Kind: Int, Sendable { case coldOrCertifiedB1FullWidthV1 = 1 }
        public let kind: Kind
        public let width: Int
        public let mtpNextToken: Int?
        public init(width: Int, mtpNextToken: Int?) {
            precondition(width > 0)
            self.kind = .coldOrCertifiedB1FullWidthV1
            self.width = width; self.mtpNextToken = mtpNextToken
        }
    }

    /// Plain owner bookkeeping. Only actual regular B1 calls or a certified import
    /// advance this lineage; tail/decode/native work cannot launder a modulo boundary.
    public struct CanonicalPrefillTrajectory {
        public let width: Int
        /// P106 H50: the widest B1 chunk that advances the lineage. `width` (the default) is the H25 rule;
        /// a multiple of `width` is admitted only where the caller has made wide chunks compute the
        /// width-sensitive projections in `width`-row blocks, which is what makes their state equal.
        public let maxChunk: Int
        public private(set) var boundary = 0
        public private(set) var valid = true
        public init(width: Int, maxChunk: Int? = nil) {
            precondition(width > 0)
            self.width = width; self.maxChunk = maxChunk ?? width
            precondition(self.maxChunk >= width && self.maxChunk % width == 0)
        }
        public mutating func invalidate() { valid = false }
        public mutating func resume(length: Int, origin: CanonicalPrefillOrigin) -> Bool {
            guard origin.width == width, length > 0, length % width == 0 else { valid = false; return false }
            boundary = length; valid = true; return true
        }
        public mutating func advance(from: Int, to: Int, batch: Int, interior: Bool,
                                     mtpNextToken: Int?) -> CanonicalPrefillOrigin? {
            let span = to - from
            guard valid, from == boundary, span > 0, span % width == 0, span <= maxChunk, batch == 1, interior else {
                boundary = to; valid = false; return nil
            }
            boundary = to
            return CanonicalPrefillOrigin(width: width, mtpNextToken: mtpNextToken)
        }
    }

    /// The non-row state at one length. `head` holds the `a0..a5` arrays of every linear layer,
    /// keyed exactly as `exportCaches` keys them.
    public struct Rung {
        public let length: Int
        public let head: [String: MLXArray]
        public let bytes: Int
        public let canonical: CanonicalPrefillOrigin?
        public init(length: Int, head: [String: MLXArray], bytes: Int, canonical: CanonicalPrefillOrigin? = nil) {
            self.length = length; self.head = head; self.bytes = bytes; self.canonical = canonical
        }
        public var metadata: [String: Int] {
            ["length": length, "origin": canonical?.kind.rawValue ?? 0,
             "width": canonical?.width ?? 0, "mtp_next_token": canonical?.mtpNextToken ?? -1]
        }
        public func withCanonical(_ value: CanonicalPrefillOrigin) -> Rung {
            Rung(length: length, head: head, bytes: bytes, canonical: value)
        }
    }

    final class Entry {
        let id: Int
        let tokens: [Int]
        /// `trunk.L{i}.k/.v/.i0/.i1` and, when the sequence decoded with MTP, `mtp.L0.*`: the RAW
        /// buffers (capacity included) plus the logical length they are valid to.
        let rows: [String: MLXArray]
        let rowsValidTo: Int
        /// The MTP head's rows are valid to here (0: the sequence never drafted). A request that
        /// joined a batch retired its draft state part-way, so this can be short of the trunk.
        let mtpValidTo: Int
        let rungs: [Rung]           // ascending by length; geometry is immutable after construction
        var lastUse: Date
        let rowBytes: Int
        /// P103: a PARTIAL entry written mid-prefill so that identical prompts arriving together share one prefill
        /// (P100's coalescing). It carries one rung and, from the plain path, no MTP rows -- so it must never
        /// supersede a finished entry of the same conversation, and it must not outlive the request that wrote it.
        let inFlight: Bool
        let owner: Int
        let bytes: Int
        let chargedBytes: Int
        /// P119: a SHARED-PREFIX entry (`sharedEntries`, never in `entries`): exactly `sharedPrefixLength` tokens, rows
        /// sliced to them, one rung at that length. `sharedMTPNextToken` is the MTP head's shifted input at the rung
        /// (the prompt's token at index `sharedPrefixLength`): the head's last row was computed from it, so the head
        /// rows serve only a prompt with the same next token. nil: captured without the head (trunk only).
        let shared: Bool
        let sharedMTPNextToken: Int?
        let sharedHash: UInt64
        var sharedUse = 0            // LRU tick among the shared entries (deterministic, unlike Date)
        init(tokens: [Int], rows: [String: MLXArray], rowsValidTo: Int, mtpValidTo: Int, rungs: [Rung], inFlight: Bool = false, owner: Int = -1, id: Int = 0,
             shared: Bool = false, sharedMTPNextToken: Int? = nil, sharedHash: UInt64 = 0) {
            self.id = id
            self.tokens = tokens; self.rows = rows; self.rowsValidTo = rowsValidTo; self.mtpValidTo = mtpValidTo
            self.inFlight = inFlight; self.owner = owner
            self.shared = shared; self.sharedMTPNextToken = sharedMTPNextToken; self.sharedHash = sharedHash
            self.rungs = rungs; self.lastUse = Date()
            let rowBytes = rows.values.reduce(0) { $0 + $1.nbytes }
            self.rowBytes = rowBytes
            self.bytes = rowBytes + rungs.reduce(0) { $0 + $1.bytes }
            self.chargedBytes = HotPrefixStore.charge(rows: rows, rungs: rungs, tokenCount: tokens.count)
        }
    }

    public let capBytes: Int
    /// Opt-in shared RAM policy; the legacy fixed-cap behavior stays unchanged.
    public let strictBudget: Bool
    public private(set) var budgetBytes: Int
    public private(set) var rejectedStores = 0, preCopyEvictions = 0, copyAttempts = 0
    /// Contiguous may share an input with 16 KiB slack. A new Metal copy rounds
    /// to a VM page and can reuse a buffer <2 pages larger (BufferCache); charge
    /// three pages conservatively, including already exactly packed arrays.
    private static let backingSlack = max(16_384, 3 * Int(getpagesize()))
    /// Optional owner readback guard before strict cache copy construction. It must
    /// not retain MLX inputs; the server uses it to refuse unreclaimed allocations.
    public var permitAllocation: ((Int) -> Bool)?
    public let rungStep: Int
    /// Decode rungs kept per sequence besides the prefill-end and final ones: a sliding window, so
    /// a 100k-token response does not pin 200 recurrent states.
    public let keepDecodeRungs: Int
    /// An older entry within this many tokens of being a prefix of a new one is replaced by it.
    /// P094: the slack is RELATIVE (at most a quarter of the old entry) and small. An absolute 64
    /// let any new store evict every entry shorter than 64 tokens, and -- under a coding agent whose
    /// sessions share an 18k-token preamble -- let a fresh session evict another session whose own
    /// turn was under 64 tokens: the two are not the same conversation, and the rule could not tell.
    public var supersedeSlack = 16
    private(set) var entries: [Entry] = []
    private var nextEntryID = 1
    /// Diagnostics for the last lookup when `debug` is on: per entry, its tokens (a COW reference),
    /// its rung lengths, and the common prefix with the prompt. The server decodes the drift point.
    public var debug = false
    public private(set) var lastDebug: [(tokens: [Int], rungs: [Int], common: Int, mtpValidTo: Int)] = []
    /// P106 D50: the query `lastDebug` was computed for. Any other request's lookup (coalescing, a sibling)
    /// overwrites `lastDebug` between a lookup and its reader's next owner job, so a reader checks this first.
    public private(set) var lastDebugQuery: [Int] = []
    public private(set) var hits = 0, misses = 0, stores = 0, evictions = 0

    /// P100: evict a DOMINATED entry first -- one whose tokens are (at least 90%) a prefix of a newer entry, i.e. the
    /// earlier turn of the same conversation -- and only then the least recently used. Under a subagent fan-out the
    /// parent sits idle while its subagents store turn after turn; plain LRU made the parent the victim.
    public var evictAffinity = true
    /// P100: in-flight prefill entries stored / resumed (coalesced sibling prefills)
    public private(set) var coalescedStores = 0, coalescedHits = 0
    /// P103: coalesce materialises into a cache that has already forwarded chunks (the only non-fresh import in the
    /// server; the disk-resume + coalesce interleaving reaches it on the FIRST loop iteration), and partial entries reaped.
    public var coalescedIntoNonFresh = 0
    public private(set) var inFlightReaped = 0
    public init(capBytes: Int, rungStep: Int = 512, keepDecodeRungs: Int = 2, strictBudget: Bool = false) {
        precondition(capBytes >= 0)
        self.capBytes = capBytes; self.budgetBytes = capBytes; self.strictBudget = strictBudget
        self.rungStep = max(1, rungStep); self.keepDecodeRungs = max(0, keepDecodeRungs)
    }

    public var enabled: Bool { capBytes > 0 }
    /// Includes the P119 shared-prefix entries: their bytes are charged to this store like any entry's.
    public var totalBytes: Int { entries.reduce(0) { $0 + $1.bytes } + sharedEntries.reduce(0) { $0 + $1.bytes } }
    /// Ordinary entries only (finished and in-flight); the P119 shared-prefix entries are `sharedCount`.
    public var count: Int { entries.count }
    private static func charge(rows: [String: MLXArray], rungs: [Rung], tokenCount: Int) -> Int {
        rows.values.reduce(0) { $0 + $1.nbytes + backingSlack }
            + rungs.reduce(0) { total, rung in
                total + rung.head.values.reduce(0) { $0 + $1.nbytes + backingSlack }
            } + tokenCount * MemoryLayout<Int>.stride
    }
    public var chargedBytes: Int {
        entries.reduce(0) { $0 + $1.chargedBytes } + sharedChargedBytes
    }
    /// Model owner only. Shrinks before new allocations, including to zero entries.
    /// A failed reclaim is a refused admission, never permission to overshoot.
    /// P119: the shared-prefix entries beyond their protected share of the NEW budget go first (least recently used);
    /// the rest are reclaimed only once no ordinary entry is left to evict -- an admission is never refused to keep them.
    @discardableResult
    public func setBudgetBytes(_ target: Int) -> Bool {
        precondition(strictBudget && target >= 0)
        budgetBytes = min(capBytes, target)
        trimSharedToShare()
        evict(to: budgetBytes, charged: true, keepOne: false, includeShared: true)
        return chargedBytes <= budgetBytes
    }
    /// Caller publishes these plain values from the owner; HTTP never reads arrays.
    /// P119: `logical_bytes`/`charged_bytes` include the shared-prefix entries (they are charged to this store) while
    /// `entries`/`rungs`/`inflight_entries` count ordinary entries only; with the knob on the `shared_*` keys say how much
    /// of the charge is theirs and how many ordinary entries their captures evicted (also counted in `evictions` and
    /// `pre_copy_evictions`). Knob off: exactly the pre-P119 keys.
    public func snapshot() -> [String: Int] {
        var d = ["ceiling_bytes": capBytes, "target_bytes": budgetBytes, "logical_bytes": totalBytes,
                 "charged_bytes": chargedBytes, "entries": count, "rungs": entries.reduce(0) { $0 + $1.rungs.count },
                 "strict_budget": strictBudget ? 1 : 0, "rejected_stores": rejectedStores,
                 "pre_copy_evictions": preCopyEvictions, "copy_attempts": copyAttempts,
                 "inflight_entries": entries.filter { $0.inFlight }.count]
        if sharedPrefixEnabled {
            d["shared_entries"] = sharedEntries.count; d["shared_logical_bytes"] = sharedLogicalBytes
            d["shared_charged_bytes"] = sharedChargedBytes; d["shared_capture_evictions"] = sharedCaptureEvictions
            d["shared_copy_attempts"] = sharedCopyAttempts
        }
        return d
    }
    public func statsLine() -> String {
        let line = String(format: "hot prefix store: %d sequence(s), %.1f GB of %.1f GB, hits %d, misses %d, stores %d, evictions %d, coalesced stores %d hits %d (%d into a non-fresh cache), in-flight reaped %d",
               entries.count, Double(totalBytes) / 1e9, Double(capBytes) / 1e9, hits, misses, stores, evictions, coalescedStores, coalescedHits, coalescedIntoNonFresh, inFlightReaped)
        guard sharedPrefixEnabled else { return line }
        return line + String(format: "; shared prefix rungs %d of %d at %d (%.2f GB), first sightings %d admissions %d, stores %d upgrades %d dedup %d hits %d evictions %d/%d rejected %d",
                             sharedEntries.count, sharedPrefixCapacity, sharedPrefixLength, Double(sharedLogicalBytes) / 1e9,
                             sharedFirstSightings, sharedAdmissions,
                             sharedStores, sharedUpgrades, sharedDedupSkips, sharedHits, sharedLRUEvictions, sharedForcedEvictions, sharedRejected)
    }

    /// P106 H54 anchors (store-time only): when an entry is STORED, besides the first two and the last `keepDecodeRungs + 1`
    /// rungs it keeps, for every `anchorStep`-wide band inside the last `anchorWindow` tokens, the highest rung of that band
    /// -- mostly the previous turns' end rungs the entry inherited from the conversation it supersedes. H53 (B41, real
    /// OMP): a client edit 4.5k tokens before the end of a 91k conversation resumed at 38753 of 86738 common tokens (48k
    /// re-prefilled, TTFT 61.7 s) because retention had pruned every middle rung. Anchors are chosen among rungs the store
    /// already holds or the live request already has: they never enlarge a LIVE ladder, so the per-row live-rung reserve
    /// (`retainedRungLimit`) is unchanged -- H58: counting them there starved the shared hot store at eight 258k rows (D52).
    /// The stored rungs are charged to the entry like any other. 0 = off.
    public var anchorStep = 0
    public var anchorWindow = 0
    var anchorSlots: Int { anchorStep > 0 && anchorWindow > 0 ? anchorWindow / anchorStep + 1 : 0 }

    /// The first two captures plus the last `keepDecodeRungs + 1` captures: the LIVE limit (admission reserves it per row).
    /// The extra last entry is needed even when final store cannot capture a new final rung.
    public var retainedRungLimit: Int { keepDecodeRungs + 3 }

    /// Shared final/live selector. Equal-length captures remain distinct and keep input order.
    /// This accepts an unpruned history at any cutoff; a previously pruned live history must
    /// first pass `retainedLiveRungs(_:validTo:)` before a caller stores it at that cutoff.
    /// `anchors` (store-time selections only) adds the H54 band anchors.
    public func retainedRungs(_ rungs: [Rung], validTo: Int? = nil, anchors: Bool = false) -> [Rung] {
        let ordered = rungs.enumerated()
            .filter { indexed in validTo.map { indexed.element.length <= $0 } ?? true }
            .sorted {
                $0.element.length == $1.element.length ? $0.offset < $1.offset
                    : $0.element.length < $1.element.length
            }.map { $0.element }
        guard ordered.count > retainedRungLimit else { return ordered }
        let tail = ordered.count - (keepDecodeRungs + 1)
        guard anchors, anchorSlots > 0, let top = ordered.last?.length else {
            return Array(ordered.prefix(2)) + Array(ordered.suffix(keepDecodeRungs + 1))
        }
        // per band inside the window, the highest middle rung (the last index of that band among the middle ones)
        var pick: [Int: Int] = [:]
        for i in 2 ..< tail where ordered[i].length >= top - anchorWindow { pick[ordered[i].length / anchorStep] = i }
        let keep = Set(pick.values)
        return ordered.enumerated().filter { i, _ in i < 2 || i >= tail || keep.contains(i) }.map { $0.element }
    }

    /// Bound a monotone live history after append, without evaluating or copying tensor data.
    /// A backwards capture/cutoff is refused: discarded middle entries cannot be recovered.
    /// The caller must make refusal observable and decline a final store that fails this guard.
    public func retainedLiveRungs(_ rungs: [Rung], validTo: Int? = nil) -> [Rung]? {
        guard zip(rungs, rungs.dropFirst()).allSatisfy({ $0.0.length <= $0.1.length }),
              validTo.map({ limit in rungs.last.map { $0.length <= limit } ?? true }) ?? true else { return nil }
        return retainedRungs(rungs, validTo: validTo)
    }

    // MARK: capture

    /// Which exported keys are position-indexed rows, and by what divisor of the length.
    static func rowDivisor(_ name: String, ratio: Int) -> Int? {
        if name.hasSuffix(".k") || name.hasSuffix(".v") || name.hasSuffix(".i0") { return 1 }
        if name.hasSuffix(".i1") { return ratio }
        return nil
    }
    static func rowAxis(_ name: String) -> Int { name.hasSuffix(".k") || name.hasSuffix(".v") ? 2 : 1 }

    /// Detach views that would retain much larger backing allocations. In this vendored MLX,
    /// contiguous copies when backing bytes exceed logical bytes by more than 16 KiB, even for
    /// an already contiguous slice. Evaluation is essential: a pending copy still holds its
    /// input graph. Compact arrays are reused, so capturing and later storing a rung does not
    /// copy its state twice. Logical counters do not describe allocator rounding;
    /// strict-budget charges separately include both sharing and allocator slack.
    private static func compact(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        let result = arrays.mapValues { contiguous($0) }
        eval(Array(result.values))
        return result
    }

    /// The non-row state of `caches` right now, as a rung at `length`. The recurrent state is
    /// replaced by each step; compacting also prevents a B1 snapshot from retaining a B8 pool.
    public func captureRung(_ caches: [KVCache], length: Int, model: Qwen4ExpModel, canonical: CanonicalPrefillOrigin? = nil) -> Rung {
        var d: [String: MLXArray] = [:]
        model.exportCaches(caches, prefix: "trunk.", into: &d)
        return Self.rung(fromExported: d, length: length, ratio: model.configuration.text.indexerCompressRatio, canonical: canonical)
    }
    /// Model-free core of `captureRung`, for the tests.
    public static func rung(fromExported d: [String: MLXArray], length: Int, ratio: Int, canonical: CanonicalPrefillOrigin? = nil) -> Rung {
        let head = compact(d.filter { rowDivisor($0.key, ratio: ratio) == nil })
        return Rung(length: length, head: head, bytes: head.values.reduce(0) { $0 + $1.nbytes }, canonical: canonical)
    }

    /// The raw row buffers of `caches` (and of the MTP cache, when there is one).
    private func captureRows(_ caches: [KVCache], mtp: KVCache?, model: Qwen4ExpModel) -> [String: MLXArray] {
        var rows: [String: MLXArray] = [:]
        func take(_ cs: [KVCache], prefix: String) {
            for (i, c) in cs.enumerated() {
                guard let l = c as? CacheList else { continue }
                let kv = l[0] as! KVCacheSimple
                if let k = kv.rawKeys, let v = kv.rawValues { rows["\(prefix)L\(i).k"] = k; rows["\(prefix)L\(i).v"] = v }
                let idx = l[1] as! ArraysCache
                if let x = idx[0] { rows["\(prefix)L\(i).i0"] = x }
                if let x = idx[1] { rows["\(prefix)L\(i).i1"] = x }
            }
        }
        take(caches, prefix: "trunk.")
        if let mtp { take([mtp], prefix: "mtp.") }
        return rows
    }

    /// Store a finished sequence. `tokens` must be EXACTLY the tokens whose state the caches hold --
    /// no pending token, nothing rolled back -- and `rungs` are the rungs captured along the way
    /// (the final rung is captured here). An earlier entry whose tokens are a prefix of these is
    /// superseded: a conversation keeps one entry, not one per turn.
    public func store(tokens: [Int], caches: [KVCache], mtp: KVCache?, rungs: [Rung], model: Qwen4ExpModel, inFlight: Bool = false, owner: Int = -1, finalCanonical: CanonicalPrefillOrigin? = nil) {
        guard enabled, !tokens.isEmpty else { return }
        // The caller's claim is checked against the caches themselves: a final rung is only taken
        // when the attention offset IS the token count. Otherwise the earlier rungs (valid at their
        // own lengths, rows being append-only) are kept and the mismatch is reported, never stored.
        let trunkLen = (caches.first { $0 is CacheList } as? CacheList).map { ($0[0] as! KVCacheSimple).offset } ?? -1
        var final: Rung? = nil
        if trunkLen == tokens.count {
            if strictBudget {
                // Metadata only: the core below reclaims room BEFORE compacting final state.
                var d: [String: MLXArray] = [:]
                model.exportCaches(caches, prefix: "trunk.", into: &d)
                let head = d.filter { Self.rowDivisor($0.key, ratio: model.configuration.text.indexerCompressRatio) == nil }
                final = Rung(length: tokens.count, head: head, bytes: head.values.reduce(0) { $0 + $1.nbytes }, canonical: finalCanonical)
            } else { final = captureRung(caches, length: tokens.count, model: model, canonical: finalCanonical) }
        }
        else { FileHandle.standardError.write("engine: hot store: cache holds \(trunkLen) tokens but \(tokens.count) were claimed -- final rung not taken\n".data(using: .utf8)!) }
        var mtpValid = 0
        if let mtp, let kv = (mtp as? CacheList)?[0] as? KVCacheSimple { mtpValid = min(kv.offset, tokens.count) }
        store(tokens: tokens, rows: captureRows(caches, mtp: mtp, model: model), rowsValidTo: min(trunkLen, tokens.count),
              mtpValidTo: mtpValid, rungs: rungs + (final.map { [$0] } ?? []), inFlight: inFlight, owner: owner)
    }

    /// Model-free core of `store`: rows are the raw buffers, valid to `rowsValidTo`; rungs at any
    /// lengths (ascending or not); `mtpValidTo` 0 when the sequence has no MTP rows.
    public func store(tokens: [Int], rows: [String: MLXArray], rowsValidTo: Int, mtpValidTo: Int, rungs: [Rung], inFlight: Bool = false, owner: Int = -1) {
        guard enabled, !tokens.isEmpty else { return }
        let length = tokens.count
        var ladder = retainedRungs(rungs, validTo: min(length, rowsValidTo), anchors: true)
        guard !ladder.isEmpty else { return }
        func commonPrefix(_ old: Entry) -> Int {
            let n = min(old.tokens.count, length)
            var common = 0
            while common < n && old.tokens[common] == tokens[common] { common += 1 }
            return common
        }
        func superseded(_ old: Entry) -> Bool {
            guard !inFlight, !old.inFlight else { return false }
            return old.tokens.count - commonPrefix(old) <= min(supersedeSlack, old.tokens.count / 4)
        }
        // P106 B41: the turn that supersedes a conversation's previous entry inherits that entry's rungs inside
        // their common prefix -- valid for the new rows by construction (same tokens, same state). Without it a
        // conversation kept only its latest turn's rungs, so the early rungs another conversation could share
        // (a system prompt) vanished at the next turn.
        if !inFlight {
            let own = Set(ladder.map(\.length))
            let carried = entries.filter(superseded).flatMap { old -> [Rung] in
                let c = commonPrefix(old)
                return old.rungs.filter { $0.length <= c && !own.contains($0.length) }
            }
            if !carried.isEmpty { ladder = retainedRungs(carried + ladder, validTo: min(length, rowsValidTo), anchors: true) }
        }
        if strictBudget {
            let incoming = Self.charge(rows: rows, rungs: ladder, tokenCount: tokens.count)
            // Refuse before deleting a useful predecessor or constructing copy graphs.
            // P119: the shared-prefix entries stay (ordinary pressure does not evict them), so they are not room.
            guard incoming <= budgetBytes - sharedChargedBytes else { rejectedStores += 1; return }
            entries.removeAll(where: superseded)
            let before = evictions
            evict(to: budgetBytes - incoming, charged: true, keepOne: false)
            preCopyEvictions += evictions - before
            guard chargedBytes <= budgetBytes - incoming,
                  permitAllocation?(incoming) ?? true else { rejectedStores += 1; return }
        }
        copyAttempts += 1
        // Both decode unstacking and native prefill produce views into multi-row pools. A single
        // retained row must not pin all other rows while being charged only its own nbytes.
        // In-flight entries need the same rule: native prefill can supply their views too.
        let ownedRows = Self.compact(rows)
        ladder = ladder.map { rung in
            let head = Self.compact(rung.head)
            return Rung(length: rung.length, head: head, bytes: head.values.reduce(0) { $0 + $1.nbytes }, canonical: rung.canonical)
        }
        let e = Entry(tokens: tokens, rows: ownedRows, rowsValidTo: rowsValidTo, mtpValidTo: mtpValidTo, rungs: ladder, inFlight: inFlight, owner: owner, id: nextEntryID)
        nextEntryID += 1
        // Superseded: an older entry that is a prefix of this one, or nearly so -- a previous turn
        // whose tokens the new prompt re-rendered with a different last token or two (an empty
        // think block, a re-ordered tool call) is still the same conversation, one turn behind.
        // P103: an IN-FLIGHT store supersedes nothing. It is a prefix of a prompt still being read, it carries one
        // rung, and from the plain path no MTP rows at all -- and a chunk end that lands within `supersedeSlack` of a
        // finished entry's length would have removed that entry's whole ladder and its draft state, pinning the
        // conversation's next turn to serial decode with no rung to resume from. Only a finished store supersedes.
        if !strictBudget { entries.removeAll(where: superseded) }
        entries.append(e)
        if inFlight { coalescedStores += 1 } else { stores += 1 }
        evictIfOver()
    }

    /// P103: drop the partial entries a finished (or abandoned) request left behind. A request that ends without
    /// reaching its final `store` -- an error, a disconnected client -- used to leave its in-flight entries in the
    /// store for good, each pinning that request's whole KV buffer with no way to tell them apart.
    @discardableResult
    public func reapInFlight(owner: Int) -> Int {
        let before = entries.count
        entries.removeAll { $0.inFlight && $0.owner == owner }
        let n = before - entries.count
        inFlightReaped += n
        return n
    }

    private func evictIfOver() {
        evict(to: strictBudget ? budgetBytes : capBytes, charged: strictBudget, keepOne: !strictBudget)
    }

    private func evict(to target: Int, charged: Bool, keepOne: Bool, includeShared: Bool = false) {
        while (charged ? chargedBytes : totalBytes) > target {
            guard entries.count > (keepOne ? 1 : 0) else {
                // P119: shared-prefix entries are never victims of ordinary pressure (a store, the fixed cap); only a
                // budget squeeze that the ordinary entries cannot meet reaches them (after `trimSharedToShare`), least
                // recently used first. With none stored this is exactly the old loop condition.
                guard includeShared, let i = lruSharedIndex() else { break }
                sharedEntries.remove(at: i); sharedForcedEvictions += 1
                continue
            }
            var victim: Int? = nil
            if evictAffinity {
                // dominated: a NEWER entry (appended later) shares at least 90% of this entry's own tokens
                let dominated = entries.indices.filter { i in
                    let a = entries[i].tokens
                    return (i + 1 ..< entries.count).contains { j in
                        let b = entries[j].tokens; let n = min(a.count, b.count)
                        var p = 0
                        while p < n && a[p] == b[p] { p += 1 }
                        return p * 10 >= a.count * 9
                    }
                }
                victim = dominated.min { entries[$0].lastUse < entries[$1].lastUse }
            }
            let i = victim ?? entries.indices.min { entries[$0].lastUse < entries[$1].lastUse }!
            entries.remove(at: i); evictions += 1
        }
        // Only the legacy fixed-cap policy retains one oversized entry.
    }

    // MARK: P119 shared-prefix rungs

    /// P119 -- ONE rung per distinct prompt head, shared by every cold prompt that begins with it.
    ///
    /// A client that opens many conversations with the same system prompt (aider: all 352 main requests of a run share
    /// 1321 tokens after the chat template) resumes from nothing today: the ordinary rungs sit at the prompt-prefix step
    /// (8192), the reserve/prompt-end/decode rungs of earlier conversations all lie past the shared head, so no rung is
    /// at or below the common prefix. A shared-prefix rung is the state at exactly `sharedPrefixLength` tokens (1024: a
    /// width-canonical boundary, H50b/H57), captured by the server from a cold B1 prefill whose first chunk ends there.
    ///
    /// It lives OUTSIDE `entries`: never superseded, never dominated, never an eviction victim of ordinary pressure (a
    /// store making room, the fixed cap) -- but only inside its PROTECTED SHARE: all shared entries together are charged
    /// at most `budgetBytes / sharedPrefixShareDivisor` (a capture makes room from the least recently used shared
    /// entries; a budget squeeze trims them to the share of the new budget first). Their bytes are charged to this
    /// store (`totalBytes`, `chargedBytes`), so the ordinary entries get that much less room; a strict-budget squeeze
    /// (`setBudgetBytes`, admission) that the ordinary entries cannot meet evicts the rest too, least recently used
    /// first. At most `sharedPrefixCapacity` are kept, LRU among them. Identity = the exact first `sharedPrefixLength`
    /// token ids (hash + length, then the ids) plus, for an entry that carries the MTP head's rows, the head's shifted
    /// input token (see `Entry.sharedMTPNextToken`); a head-carrying capture supersedes a trunk-only one of the same ids.
    /// A head earns a capture only on evidence of reuse (`admitSharedPrefix`: its second cold sighting).
    /// 0 length or 0 capacity = off: nothing is stored or recorded and every decision is the pre-P119 one.
    public var sharedPrefixLength = 0
    public var sharedPrefixCapacity = 0
    /// The shortest prompt a shared rung is captured from or resumes (0: any prompt longer than the rung). The server
    /// sets rung + PrefillChunkPlan.sharedPrefixMargin: the same prompts that get the split, so the chunk after the rung
    /// (resumed or split) has at least margin - 1 rows, above every row-count kernel threshold it would otherwise cross.
    public var sharedPrefixMinPrompt = 0
    /// The shared entries' protected share of the budget: at most `budgetBytes / divisor` charged (0 or 1: the whole budget).
    public var sharedPrefixShareDivisor = 8
    /// How many distinct heads' first sightings are remembered (LRU) for `admitSharedPrefix`.
    public var sharedPrefixSightingCapacity = 64
    public var sharedPrefixEnabled: Bool { enabled && sharedPrefixLength > 0 && sharedPrefixCapacity > 0 }
    private func sharedPromptEligible(_ count: Int) -> Bool { count > sharedPrefixLength && count >= sharedPrefixMinPrompt }
    private(set) var sharedEntries: [Entry] = []
    private var sharedTick = 0
    /// head hash -> tick of its last cold sighting (plain values; a hash collision can only admit one needless capture,
    /// never a wrong resume: the entries themselves are matched by the exact ids)
    private var sharedSightings: [UInt64: Int] = [:]
    private var sightingTick = 0
    public private(set) var sharedStores = 0, sharedUpgrades = 0, sharedDedupSkips = 0, sharedHits = 0
    public private(set) var sharedLRUEvictions = 0, sharedForcedEvictions = 0, sharedRejected = 0, sharedMismatches = 0
    public private(set) var sharedFirstSightings = 0, sharedAdmissions = 0, sharedCaptureEvictions = 0, sharedCopyAttempts = 0
    public var sharedCount: Int { sharedEntries.count }
    public var sharedLogicalBytes: Int { sharedEntries.reduce(0) { $0 + $1.bytes } }
    public var sharedChargedBytes: Int { sharedEntries.reduce(0) { $0 + $1.chargedBytes } }
    /// The protected share of the current budget (legacy mode: of the fixed cap, which is its budget), in the mode's own
    /// accounting: charged bytes under the strict budget, logical bytes under the legacy cap (as each mode evicts).
    public var sharedShareBytes: Int { sharedPrefixShareDivisor > 1 ? budgetBytes / sharedPrefixShareDivisor : budgetBytes }
    private func shareCost(_ e: Entry) -> Int { strictBudget ? e.chargedBytes : e.bytes }
    private func lruSharedIndex() -> Int? {
        sharedEntries.indices.min { sharedEntries[$0].sharedUse < sharedEntries[$1].sharedUse }
    }
    /// A squeeze first takes back what the shared entries hold beyond their share of the new budget (LRU first).
    private func trimSharedToShare() {
        while sharedChargedBytes > sharedShareBytes, let i = lruSharedIndex() {
            sharedEntries.remove(at: i); sharedForcedEvictions += 1
        }
    }

    /// P119 admission (review F2/F7: no capture without evidence of reuse). Called by the server for a request whose
    /// prefill starts COLD (no hot hit) and is eligible for the split: records the sighting of its head and answers
    /// whether the head has earned the split + capture -- a cold request with the same first `sharedPrefixLength` ids was
    /// seen before (within the last `sharedPrefixSightingCapacity` distinct heads), or a shared entry of these ids already
    /// exists (an MTP upgrade of a trunk-only one, another next token). A first sighting costs nothing: the request keeps
    /// the unsplit schedule and no ~173 MB copy is made, so traffic whose prompts all start differently (one-off clients,
    /// cold-nonce instruments) never pays for rungs nobody resumes. Knob off: false, nothing recorded.
    public func admitSharedPrefix(prompt: [Int]) -> Bool {
        guard sharedPrefixEnabled, sharedPromptEligible(prompt.count) else { return false }
        let h = Self.prefixHash(prompt[0 ..< sharedPrefixLength])
        let seen = sharedSightings[h] != nil || !sharedMatches(prompt).isEmpty
        sightingTick += 1; sharedSightings[h] = sightingTick
        if sharedSightings.count > max(1, sharedPrefixSightingCapacity),
           let oldest = sharedSightings.min(by: { $0.value < $1.value })?.key {
            sharedSightings.removeValue(forKey: oldest)
        }
        if seen { sharedAdmissions += 1 } else { sharedFirstSightings += 1 }
        return seen
    }

    public enum SharedPrefixDecision: String, Sendable { case disabled, ineligible, deduplicated, store, upgrade }
    public enum SharedPrefixOutcome: String, Sendable { case disabled, ineligible, deduplicated, stored, upgraded, rejected, mismatch }

    /// FNV-1a over the token ids' 64-bit patterns: deterministic across processes (Swift's Hasher is seeded per run).
    public static func prefixHash<C: Collection>(_ tokens: C) -> UInt64 where C.Element == Int {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for t in tokens {
            var v = UInt64(bitPattern: Int64(t))
            for _ in 0 ..< 8 { h ^= v & 0xff; h = h &* 0x0000_0100_0000_01b3; v >>= 8 }
        }
        return h
    }
    private func touchShared(_ e: Entry) { sharedTick += 1; e.sharedUse = sharedTick }
    /// A duplicate capture copies nothing; it refreshes the entries that cover it.
    private func recordSharedDedup(_ prompt: [Int], next: Int?) {
        sharedDedupSkips += 1
        sharedMatches(prompt).filter { next == nil || $0.sharedMTPNextToken == next }.forEach(touchShared)
    }
    /// The shared entries whose tokens are exactly `prompt`'s first `sharedPrefixLength` (hash and length first).
    private func sharedMatches(_ prompt: [Int]) -> [Entry] {
        let R = sharedPrefixLength
        guard R > 0, prompt.count >= R, !sharedEntries.isEmpty else { return [] }
        let head = prompt[0 ..< R], h = Self.prefixHash(head)
        return sharedEntries.filter { $0.tokens.count == R && $0.sharedHash == h && $0.tokens.elementsEqual(head) }
    }

    /// What capturing `prompt`'s shared-prefix rung would do; `withMTP`: the capture carries the head's rows (their
    /// shifted input is `prompt[sharedPrefixLength]`). Pure: the caller checks this BEFORE exporting anything.
    public func sharedPrefixDecision(prompt: [Int], withMTP: Bool) -> SharedPrefixDecision {
        guard sharedPrefixEnabled else { return .disabled }
        let R = sharedPrefixLength
        guard sharedPromptEligible(prompt.count) else { return .ineligible }
        let next: Int? = withMTP ? prompt[R] : nil
        let matches = sharedMatches(prompt)
        // a trunk-only capture is covered by ANY entry of the same ids; a head capture by one with the same next token
        if matches.contains(where: { next == nil || $0.sharedMTPNextToken == next }) { return .deduplicated }
        if next != nil, matches.contains(where: { $0.sharedMTPNextToken == nil }) { return .upgrade }
        return .store
    }

    /// Model-free core of `captureSharedPrefix`. `rows` are the RAW row buffers (capacity included) valid to at least
    /// `sharedPrefixLength` positions; they are sliced to exactly that many and compacted like every stored row (a
    /// strided slice is copied out; rows are append-only, so a kept view's bytes never change). `rung` must be at that length.
    /// `mtpNextToken` nil stores the trunk only (any `mtp.` rows are dropped); otherwise it must be `prompt[R]` and the
    /// head rows must be present. The strict budget is checked BEFORE any copy, as for an ordinary store.
    @discardableResult
    public func storeSharedPrefix(prompt: [Int], rows: [String: MLXArray], mtpNextToken: Int?, rung: Rung, ratio: Int) -> SharedPrefixOutcome {
        let decision = sharedPrefixDecision(prompt: prompt, withMTP: mtpNextToken != nil)
        switch decision {
        case .disabled: return .disabled
        case .ineligible: return .ineligible
        case .deduplicated: recordSharedDedup(prompt, next: mtpNextToken); return .deduplicated
        case .store, .upgrade: break
        }
        let R = sharedPrefixLength
        guard rung.length == R, mtpNextToken == nil || mtpNextToken == prompt[R] else { sharedMismatches += 1; return .mismatch }
        var sliced: [String: MLXArray] = [:]
        for (k, v) in rows {
            guard let div = Self.rowDivisor(k, ratio: ratio) else { continue }
            if mtpNextToken == nil, k.hasPrefix("mtp.") { continue }
            let n = R / div, ax = Self.rowAxis(k)
            guard v.ndim > ax, v.dim(ax) >= n else { sharedMismatches += 1; return .mismatch }
            sliced[k] = ax == 2 ? v[0..., 0..., 0 ..< n, 0...] : v[0..., 0 ..< n, 0...]
        }
        guard sliced.keys.contains(where: { $0.hasPrefix("trunk.") }),
              mtpNextToken == nil || sliced.keys.contains(where: { $0.hasPrefix("mtp.") }) else { sharedMismatches += 1; return .mismatch }
        let incoming = Self.charge(rows: sliced, rungs: [rung], tokenCount: R)
        // The protected share bounds every mode (review F1: legacy included, where the fixed cap is the budget). Refused
        // before touching anything: even with every other shared entry gone the new one would not fit in it.
        let share = sharedShareBytes
        let incomingShare = strictBudget ? incoming : sliced.values.reduce(0) { $0 + $1.nbytes } + rung.bytes
        guard incomingShare <= share else { sharedRejected += 1; return .rejected }
        // Whom this store replaces: the trunk-only entry of the same ids it upgrades; then, least recently used first,
        // shared entries until the count is under the capacity and the charge fits the share (ordinary entries are
        // never displaced to keep a shared one). Terminates: with every other shared entry replaced, incoming <= share.
        var replaced = decision == .upgrade ? sharedMatches(prompt).filter { $0.sharedMTPNextToken == nil } : []
        let upgradedCount = replaced.count
        func isReplaced(_ e: Entry) -> Bool { replaced.contains { $0 === e } }
        var kept = sharedEntries.reduce(0) { $0 + shareCost($1) } - replaced.reduce(0) { $0 + shareCost($1) }
        while sharedEntries.count - replaced.count >= sharedPrefixCapacity || kept + incomingShare > share,
              let lru = sharedEntries.filter({ !isReplaced($0) }).min(by: { $0.sharedUse < $1.sharedUse }) {
            replaced.append(lru); kept -= shareCost(lru)
        }
        // Strict: kept + incoming <= share <= budgetBytes, so evicting ordinary entries can always make the room; the
        // physical guard below can still refuse.
        sharedEntries.removeAll(where: isReplaced)
        if strictBudget {
            // Room is made the way an ordinary store makes it (ordinary entries only); then the physical guard.
            let before = evictions
            evict(to: budgetBytes - incoming, charged: true, keepOne: false)
            preCopyEvictions += evictions - before; sharedCaptureEvictions += evictions - before
            guard chargedBytes <= budgetBytes - incoming, permitAllocation?(incoming) ?? true else {
                // Review F3: a refused capture must not lose what it would have replaced. The replaced entries are still
                // resident (no allocation) and were charged before this call, which only lowered the total since.
                // (The ordinary entries evicted above stay evicted, as for a refused ordinary store.)
                sharedEntries.append(contentsOf: replaced); sharedEntries.sort { $0.id < $1.id }
                sharedRejected += 1; return .rejected
            }
        }
        sharedLRUEvictions += replaced.count - upgradedCount
        // Review F4: release the replaced entries' arrays before the copies below allocate (no transient 2x).
        replaced.removeAll()
        sharedCopyAttempts += 1
        let ownedRows = Self.compact(sliced)
        let head = Self.compact(rung.head)
        let stored = Rung(length: R, head: head, bytes: head.values.reduce(0) { $0 + $1.nbytes })
        let e = Entry(tokens: Array(prompt[0 ..< R]), rows: ownedRows, rowsValidTo: R, mtpValidTo: mtpNextToken == nil ? 0 : R,
                      rungs: [stored], id: nextEntryID, shared: true, sharedMTPNextToken: mtpNextToken,
                      sharedHash: Self.prefixHash(prompt[0 ..< R]))
        nextEntryID += 1
        touchShared(e)
        sharedEntries.append(e)
        if decision == .upgrade { sharedUpgrades += 1 } else { sharedStores += 1 }
        if !strictBudget {                       // the legacy fixed cap: ordinary entries make the room
            let before = evictions
            evictIfOver()
            sharedCaptureEvictions += evictions - before
        }
        return decision == .upgrade ? .upgraded : .stored
    }

    /// P119: store the shared-prefix rung of `prompt` from `caches` holding EXACTLY its first `sharedPrefixLength`
    /// tokens (and from `mtp`, the draft head's cache, when the request primes one: it must hold as many). Nothing is
    /// copied for a duplicate, and in strict mode nothing before the budget check. The caller guarantees the geometry
    /// (a B1 chunk [0, sharedPrefixLength) of a cold prefill, width-canonical projections): the store cannot see it.
    @discardableResult
    public func captureSharedPrefix(prompt: [Int], caches: [KVCache], mtp: KVCache?, model: Qwen4ExpModel) -> SharedPrefixOutcome {
        let decision = sharedPrefixDecision(prompt: prompt, withMTP: mtp != nil)
        switch decision {
        case .disabled: return .disabled
        case .ineligible: return .ineligible
        case .deduplicated: recordSharedDedup(prompt, next: mtp != nil ? prompt[sharedPrefixLength] : nil); return .deduplicated
        case .store, .upgrade: break
        }
        let R = sharedPrefixLength
        let trunkLen = (caches.first { $0 is CacheList } as? CacheList).map { ($0[0] as! KVCacheSimple).offset } ?? -1
        let mtpLen = ((mtp as? CacheList)?[0] as? KVCacheSimple)?.offset
        guard trunkLen == R, mtp == nil || mtpLen == R else {
            sharedMismatches += 1
            FileHandle.standardError.write("engine: shared prefix rung: cache holds \(trunkLen) (head \(mtpLen ?? -1)) tokens, not \(R) -- not stored\n".data(using: .utf8)!)
            return .mismatch
        }
        let ratio = model.configuration.text.indexerCompressRatio
        var d: [String: MLXArray] = [:]
        model.exportCaches(caches, prefix: "trunk.", into: &d)
        // Metadata only (no copy yet): the core compacts after its budget check.
        let head = d.filter { Self.rowDivisor($0.key, ratio: ratio) == nil }
        let rung = Rung(length: R, head: head, bytes: head.values.reduce(0) { $0 + $1.nbytes })
        return storeSharedPrefix(prompt: prompt, rows: captureRows(caches, mtp: mtp, model: model),
                                 mtpNextToken: mtp != nil ? prompt[R] : nil, rung: rung, ratio: ratio)
    }

    /// Plain owner counters for the server's witness (GET /v1/engine/sessions `shared_prefix_rung`).
    public func sharedPrefixSnapshot() -> [String: Int] {
        ["enabled": sharedPrefixEnabled ? 1 : 0, "length": sharedPrefixLength, "capacity": sharedPrefixCapacity,
         "entries": sharedEntries.count, "entries_with_mtp": sharedEntries.filter { $0.mtpValidTo > 0 }.count,
         "stores": sharedStores, "upgrades": sharedUpgrades, "dedup_skips": sharedDedupSkips, "hits": sharedHits,
         "lru_evictions": sharedLRUEvictions, "forced_evictions": sharedForcedEvictions, "rejected": sharedRejected,
         "mismatches": sharedMismatches, "bytes": sharedLogicalBytes, "charged_bytes": sharedChargedBytes,
         "share_bytes": sharedPrefixEnabled ? sharedShareBytes : 0, "share_divisor": sharedPrefixShareDivisor,
         "first_sightings": sharedFirstSightings, "admissions": sharedAdmissions, "sightings": sharedSightings.count,
         "capture_evictions": sharedCaptureEvictions, "copy_attempts": sharedCopyAttempts]
    }

    // MARK: lookup / materialise

    private static func metadata(_ entry: Entry) -> [String: Int] {
        var d = ["id": entry.id, "tokens": entry.tokens.count, "rows_valid_to": entry.rowsValidTo,
                 "mtp_valid_to": entry.mtpValidTo, "in_flight": entry.inFlight ? 1 : 0]
        if entry.shared { d["shared_prefix"] = 1 }
        return d
    }
    public struct Hit {
        let entry: Entry
        public let rung: Int
        public let length: Int
        public var hasMTP: Bool { length <= entry.mtpValidTo }
        /// P119: the hit is a shared-prefix rung.
        public var isSharedPrefix: Bool { entry.shared }
        /// Value contains fixed heads only: callers must not retain Hit/Entry after import.
        public var selectedRung: Rung { entry.rungs[rung] }
        public var entryMetadata: [String: Int] { HotPrefixStore.metadata(entry) }
        public var entryRungMetadata: [[String: Int]] { entry.rungs.map { $0.metadata } }
    }
    /// Diagnostic only, owner-called after a successful completed store. No Entry/MLX escapes.
    public func completedMetadata(tokens: [Int]) -> ([String: Int], [[String: Int]])? {
        guard let e = entries.last(where: { !$0.inFlight && $0.tokens == tokens }) else { return nil }
        return (Self.metadata(e), e.rungs.map { $0.metadata })
    }

    /// The best resume point for `ids`: the highest rung at or below the longest common prefix, and
    /// strictly short of the prompt (a full-length hit has no chunk left to prefill and nowhere to
    /// take its first logits from). `minLength` refuses hits that would not beat another source.
    /// With `preferMTP`, a rung the MTP rows cover wins over a higher one they do not: re-priming
    /// the draft head costs a full prefill, so a serial resume would pin the whole conversation's
    /// later turns to serial decode.
    public func lookup(_ ids: [Int], minLength: Int = 1, preferMTP: Bool = false, quiet: Bool = false,
                       canonicalWidth: Int? = nil, diagnostic: (([String: Int]) -> Void)? = nil) -> Hit? {
        guard enabled else { return nil }
        /// P119: a shared-prefix entry is a candidate only where resuming from it cannot cost the request its draft
        /// head: for an MTP request it needs its head rows AND the same shifted input token (resuming without the head
        /// would pin an MTP request to serial decode, which costs far more than the prefill it saves); a request that does
        /// not want MTP takes any. Never in the canonical diagnostic mode: it carries no certificate.
        func sharedEligible(_ e: Entry) -> Bool {
            let R = e.tokens.count
            guard canonicalWidth == nil, sharedPromptEligible(ids.count) else { return false }
            return !preferMTP || (e.mtpValidTo >= R && e.sharedMTPNextToken == ids[R])
        }
        func best(_ candidates: [Entry], mtpOnly: Bool) -> Hit? {
            var best: Hit? = nil
            for e in candidates {
                let n = min(e.tokens.count, ids.count)
                var p = 0
                while p < n && e.tokens[p] == ids[p] { p += 1 }
                var limit = min(p, ids.count - 1, e.rowsValidTo)
                if mtpOnly { limit = min(limit, e.mtpValidTo) }
                guard limit >= minLength else { continue }
                for (ri, r) in e.rungs.enumerated().reversed() where r.length <= limit {
                    if let width = canonicalWidth {
                        let reason: Int
                        if r.length < minLength { reason = 5 }
                        else if r.canonical == nil { reason = 1 }
                        else if r.canonical!.width != width { reason = 2 }
                        else if mtpOnly && (r.canonical!.mtpNextToken == nil || r.length > e.mtpValidTo) { reason = 4 }
                        else if mtpOnly && r.canonical!.mtpNextToken != ids[r.length] { reason = 3 }
                        else { reason = 0 }
                        if let diagnostic {
                            var d = r.metadata
                            d.merge(["entry_id": e.id, "entry_tokens": e.tokens.count,
                                     "in_flight": e.inFlight ? 1 : 0, "common": p, "limit": limit,
                                     "mtp_valid_to": e.mtpValidTo, "query_next_token": ids[r.length],
                                     "rejection": reason]) { _, new in new }
                            diagnostic(d)
                        }
                        // Do not break on a newer incompatible rung: a lower certificate may fit.
                        if reason != 0 { continue }
                    }
                    if r.length >= minLength, best == nil || r.length > best!.length { best = Hit(entry: e, rung: ri, length: r.length) }
                    break
                }
            }
            return best
        }
        // In canonical mode a requested head needs its matching shifted input and coverage;
        // a plain/legacy fallback would silently change the requested prefill trajectory.
        var found = canonicalWidth != nil ? best(entries, mtpOnly: preferMTP)
            : ((preferMTP ? best(entries, mtpOnly: true) : nil) ?? best(entries, mtpOnly: false))
        // P119 (review F1): a shared-prefix rung only where it beats that decision -- strictly longer than whatever the
        // ordinary entries give, or as long and carrying the draft head the ordinary hit lacks (an MTP request; the
        // shared candidate then always has it). Otherwise ties go to the ordinary rung, and a longer ordinary hit is never
        // displaced: an MTP request whose only ordinary hit is a 50k-token entry without draft rows (the serial fallback)
        // keeps resuming at 50k instead of re-prefilling from the rung, exactly as without the knob. Nothing stored
        // (knob off): today's decision.
        if canonicalWidth == nil, !sharedEntries.isEmpty,
           let s = best(sharedEntries.filter(sharedEligible), mtpOnly: preferMTP) {
            let o = found?.length ?? 0
            if s.length > o || (s.length == o && preferMTP && s.hasMTP && found?.hasMTP == false) { found = s }
        }
        if quiet { if found != nil { coalescedHits += 1 }; return found }
        if found != nil { hits += 1 } else { misses += 1 }
        if debug {
            lastDebugQuery = ids
            lastDebug = entries.map { e in
                let n = min(e.tokens.count, ids.count)
                var p = 0
                while p < n && e.tokens[p] == ids[p] { p += 1 }
                return (e.tokens, e.rungs.map { $0.length }, p, e.mtpValidTo)
            }
        }
        return found
    }

    /// Fill FRESH caches with the state at `hit.length`: rows sliced to the length, the rung's head
    /// arrays as they were. Returns false (and touches nothing) if the entry cannot serve it.
    @discardableResult
    public func materialize(_ hit: Hit, into caches: [KVCache], mtp: KVCache?, model: Qwen4ExpModel) -> Bool {
        guard let d = exported(hit, ratio: model.configuration.text.indexerCompressRatio) else { return false }
        model.importCaches(caches, prefix: "trunk.", from: d)
        if let mtp, hit.hasMTP { model.importCaches([mtp], prefix: "mtp.", from: d) }
        return true
    }

    /// Model-free core of `materialize`: the exported-format dictionary for `hit`, or nil.
    public func exported(_ hit: Hit, ratio: Int) -> [String: MLXArray]? {
        let e = hit.entry
        let M = hit.length
        guard M <= e.rowsValidTo, hit.rung < e.rungs.count, e.rungs[hit.rung].length == M else { return nil }
        var d = e.rungs[hit.rung].head
        for (k, v) in e.rows {
            guard let div = Self.rowDivisor(k, ratio: ratio) else { continue }
            if !hit.hasMTP, k.hasPrefix("mtp.") { continue }
            let n = M / div
            let ax = Self.rowAxis(k)
            guard v.dim(ax) >= n else { return nil }
            d[k] = ax == 2 ? v[0..., 0..., 0 ..< n, 0...] : v[0..., 0 ..< n, 0...]
        }
        e.lastUse = Date()
        if e.shared { sharedHits += 1; touchShared(e) }
        // A short imported prefix must not keep a full long donor alive after that
        // donor is evicted and its cache charge released. Evaluate the detaching
        // copies on this owner before returning them to any live request cache.
        return strictBudget ? Self.compact(d) : d
    }
}
