import Foundation
import XCTest
import MLX
@testable import EngineServeSupport
@testable import Qwen4Exp

/// P119 -- shared-prefix rungs: the store side (dedupe, lookup/resume selection, LRU bound, accounting, knob-off
/// negative control) on the model-free core, and the server's chunk schedule (PrefillChunkPlan, the real arithmetic
/// Serve.swift runs) for the width-canonical argument: the knob only cuts a cold prompt's first chunk at 1024.
final class SharedPrefixRungTests: XCTestCase {
    private let ratio = 4
    private let R = 16          // the store is length-agnostic; the server admits only 1024
    private let slack = max(16_384, 3 * Int(getpagesize()))
    private let margin = PrefillChunkPlan.sharedPrefixMargin     // the server's own constant (review F4), not a copy

    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    private func address(_ array: MLXArray) -> UInt {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    }
    private func same(_ a: MLXArray, _ b: MLXArray) -> Bool { a.shape == b.shape && arrayEqual(a, b).item(Bool.self) }

    /// RAW row buffers of capacity `cap` (>= the valid length), in the exported key format, distinguished by `salt`.
    private func rows(cap: Int, salt: Float, mtp: Bool) -> [String: MLXArray] {
        var d: [String: MLXArray] = [:]
        for l in [0, 2] {
            d["trunk.L\(l).k"] = MLXArray((0 ..< (2 * cap * 3)).map { Float($0) + salt }).reshaped([1, 2, cap, 3])
            d["trunk.L\(l).v"] = MLXArray((0 ..< (2 * cap * 3)).map { Float($0) * 2 + salt }).reshaped([1, 2, cap, 3])
            d["trunk.L\(l).i0"] = MLXArray((0 ..< (cap * 2)).map { Float($0) + salt }).reshaped([1, cap, 2])
            d["trunk.L\(l).i1"] = MLXArray((0 ..< ((cap / ratio) * 2)).map { Float($0) + salt }).reshaped([1, cap / ratio, 2])
        }
        if mtp {
            d["mtp.L0.k"] = MLXArray((0 ..< (2 * cap * 3)).map { Float($0) + 100 + salt }).reshaped([1, 2, cap, 3])
            d["mtp.L0.v"] = MLXArray((0 ..< (2 * cap * 3)).map { Float($0) + 200 + salt }).reshaped([1, 2, cap, 3])
        }
        return d
    }
    private func rung(_ length: Int, salt: Float) -> HotPrefixStore.Rung {
        HotPrefixStore.rung(fromExported: ["trunk.L1.a0": MLXArray([Float(length) + salt]).reshaped([1, 1]),
                                           "trunk.L1.a1": MLXArray((0 ..< 6).map { Float($0) + salt }).reshaped([1, 2, 3])],
                            length: length, ratio: ratio)
    }
    /// `shareDivisor` 1: the whole budget is the shared entries' share (tests about other rules); the server's is 8.
    private func store(capacity: Int = 4, cap: Int = 1 << 30, strict: Bool = false, shareDivisor: Int = 1) -> HotPrefixStore {
        let s = HotPrefixStore(capBytes: cap, rungStep: 4, strictBudget: strict)
        s.sharedPrefixLength = R; s.sharedPrefixCapacity = capacity; s.sharedPrefixShareDivisor = shareDivisor
        return s
    }
    /// A prompt whose first R tokens are selected by `head` (distinct heads never share a token), then `tail`.
    private func prompt(head: Int, tail: [Int]) -> [Int] { (0 ..< R).map { $0 + head * 1000 } + tail }
    @discardableResult
    private func capture(_ s: HotPrefixStore, _ p: [Int], mtp: Bool, salt: Float = 0) -> HotPrefixStore.SharedPrefixOutcome {
        s.storeSharedPrefix(prompt: p, rows: rows(cap: 40, salt: salt, mtp: mtp), mtpNextToken: mtp ? p[R] : nil,
                            rung: rung(R, salt: salt), ratio: ratio)
    }

    // MARK: store

    func testDedupeStoresOncePerDistinctPrefix() {
        let s = store()
        let a1 = prompt(head: 1, tail: [7, 8, 9]), a2 = prompt(head: 1, tail: [7, 55, 66, 77])
        XCTAssertEqual(capture(s, a1, mtp: true), .stored)
        XCTAssertEqual(s.sharedPrefixDecision(prompt: a2, withMTP: true), .deduplicated, "decided before any export or copy")
        XCTAssertEqual(capture(s, a2, mtp: true), .deduplicated, "same head, same next token, different tail")
        XCTAssertEqual(capture(s, a2, mtp: false), .deduplicated, "a trunk-only capture is covered by the head-carrying entry")
        var b = a1; b[R - 1] = -5                       // differs only in the head's last token
        XCTAssertEqual(capture(s, b, mtp: true), .stored)
        XCTAssertEqual(s.sharedCount, 2)
        XCTAssertEqual(s.sharedStores, 2); XCTAssertEqual(s.sharedDedupSkips, 2)
        XCTAssertEqual(s.count, 0, "shared entries are not ordinary entries")
        XCTAssertEqual(capture(s, Array(a1.prefix(R)), mtp: false), .ineligible, "a prompt must extend past the rung")
        XCTAssertEqual(s.sharedCount, 2)
        // exact-ids identity: hash (deterministic FNV-1a, length-sensitive) + length, then the ids themselves
        XCTAssertEqual(HotPrefixStore.prefixHash([Int]()), 0xcbf2_9ce4_8422_2325)
        XCTAssertNotEqual(HotPrefixStore.prefixHash([0]), HotPrefixStore.prefixHash([0, 0]))
        XCTAssertEqual(s.sharedEntries[0].sharedHash, HotPrefixStore.prefixHash(a1.prefix(R)))
        XCTAssertEqual(s.sharedEntries.map(\.tokens), [Array(a1.prefix(R)), Array(b.prefix(R))])
    }

    func testHeadIdentityIncludesTheShiftedInputAndAHeadCaptureUpgradesATrunkOnlyOne() throws {
        let s = store()
        let p = prompt(head: 2, tail: [7, 1, 2])
        XCTAssertEqual(capture(s, p, mtp: false), .stored)
        XCTAssertEqual(s.sharedEntries[0].mtpValidTo, 0)
        XCTAssertNil(s.sharedEntries[0].rows["mtp.L0.k"], "a trunk-only capture drops the head rows")
        XCTAssertNil(s.lookup(p + [3], preferMTP: true), "an MTP request is not resumed without its head")
        XCTAssertEqual(capture(s, p, mtp: true), .upgraded)
        XCTAssertEqual(s.sharedCount, 1)
        XCTAssertEqual(s.sharedEntries[0].mtpValidTo, R)
        XCTAssertEqual(s.sharedEntries[0].sharedMTPNextToken, 7)
        var q = p; q[R] = 8                               // same head, different shifted input
        XCTAssertEqual(capture(s, q, mtp: true, salt: 3), .stored, "the head's last row was computed from the next token")
        XCTAssertEqual(capture(s, q, mtp: false), .deduplicated)
        XCTAssertEqual(s.sharedCount, 2)
        XCTAssertEqual(s.sharedUpgrades, 1); XCTAssertEqual(s.sharedStores, 2)
        // each MTP request is served by the entry whose next token is its own
        XCTAssertEqual(try XCTUnwrap(s.lookup(q + [1], preferMTP: true)).selectedRung.head["trunk.L1.a0"]?.item(Float.self), Float(R) + 3)
        XCTAssertEqual(try XCTUnwrap(s.lookup(p + [1], preferMTP: true)).selectedRung.head["trunk.L1.a0"]?.item(Float.self), Float(R))
        XCTAssertTrue(try XCTUnwrap(s.lookup(p + [1], preferMTP: true)).hasMTP)
        // a head capture must carry head rows and name the prompt's own next token
        XCTAssertEqual(s.storeSharedPrefix(prompt: prompt(head: 9, tail: [7]), rows: rows(cap: 40, salt: 0, mtp: false),
                                           mtpNextToken: 7, rung: rung(R, salt: 0), ratio: ratio), .mismatch)
        XCTAssertEqual(s.storeSharedPrefix(prompt: prompt(head: 9, tail: [7]), rows: rows(cap: 40, salt: 0, mtp: true),
                                           mtpNextToken: 6, rung: rung(R, salt: 0), ratio: ratio), .mismatch)
        XCTAssertEqual(s.storeSharedPrefix(prompt: prompt(head: 9, tail: [7]), rows: rows(cap: 40, salt: 0, mtp: true),
                                           mtpNextToken: 7, rung: rung(R + 4, salt: 0), ratio: ratio), .mismatch)
        XCTAssertEqual(s.sharedMismatches, 3)
    }

    func testLookupResumesAtTheRungWithExactDetachedRowsAndNeverCostsTheHead() throws {
        let s = store()
        let donor = prompt(head: 3, tail: [7, 100, 101])
        let full = rows(cap: 40, salt: 5, mtp: true)
        eval(Array(full.values))
        XCTAssertEqual(s.storeSharedPrefix(prompt: donor, rows: full, mtpNextToken: 7, rung: rung(R, salt: 5), ratio: ratio), .stored)
        let e = try XCTUnwrap(s.sharedEntries.first)
        // stored rows: exactly R positions (R / ratio pooled blocks), equal to the prefix; the strided K/V slices are
        // copied out of the capacity buffers (`compact`: a view with < 16 KiB of excess backing may be kept, as for any entry)
        XCTAssertEqual(e.rows["trunk.L0.k"]?.shape, [1, 2, R, 3])
        XCTAssertEqual(e.rows["trunk.L2.i0"]?.shape, [1, R, 2])
        XCTAssertEqual(e.rows["trunk.L2.i1"]?.shape, [1, R / ratio, 2])
        XCTAssertEqual(e.rows["mtp.L0.v"]?.shape, [1, 2, R, 3])
        for (k, v) in e.rows {
            let src = try XCTUnwrap(full[k])
            if k.hasSuffix(".k") || k.hasSuffix(".v") { XCTAssertNotEqual(address(v), address(src), "\(k) must not alias the live buffer") }
            let n = R / (HotPrefixStore.rowDivisor(k, ratio: ratio) ?? 1)
            XCTAssertTrue(same(v, HotPrefixStore.rowAxis(k) == 2 ? src[0..., 0..., 0 ..< n, 0...] : src[0..., 0 ..< n, 0...]), k)
        }
        // a later cold prompt with the same head (and next token) resumes AT the rung, with the head
        let later = prompt(head: 3, tail: [7, 200, 201, 202])
        let h = try XCTUnwrap(s.lookup(later, preferMTP: true))
        XCTAssertEqual(h.length, R); XCTAssertTrue(h.hasMTP); XCTAssertTrue(h.isSharedPrefix)
        XCTAssertEqual(h.entryMetadata["shared_prefix"], 1)
        let d = try XCTUnwrap(s.exported(h, ratio: ratio))
        XCTAssertTrue(same(d["trunk.L0.k"]!, e.rows["trunk.L0.k"]!))
        XCTAssertTrue(same(d["mtp.L0.k"]!, e.rows["mtp.L0.k"]!))
        XCTAssertEqual(d["trunk.L1.a0"]?.item(Float.self), Float(R) + 5, "the rung's own fixed-size state")
        XCTAssertEqual(s.sharedHits, 1)
        // a plain request resumes there whatever its next token
        var plain = later; plain[R] = 99
        XCTAssertEqual(s.lookup(plain)?.length, R)
        XCTAssertEqual(s.lookup(plain)?.isSharedPrefix, true)
        // an MTP request whose next token differs gets NO hit, never a head-less one (serial decode costs more)
        XCTAssertNil(s.lookup(plain, preferMTP: true))
        // strictly short of the prompt; a head that differs anywhere misses; never in the canonical diagnostic mode
        XCTAssertNil(s.lookup(Array(later.prefix(R))))
        var drift = later; drift[3] = -1
        XCTAssertNil(s.lookup(drift))
        XCTAssertNil(s.lookup(later, preferMTP: true, canonicalWidth: 1024))
        XCTAssertNil(s.lookup(later, minLength: R + 1), "minLength still refuses a hit that does not beat another source")
    }

    func testPromptsShorterThanTheMinimumNeitherCaptureNorResume() {
        // the server sets rung + 256: a resumed request's first chunk past the rung is then as wide as a donor's
        let s = store()
        s.sharedPrefixMinPrompt = R + 4
        XCTAssertEqual(capture(s, prompt(head: 1, tail: [7, 1, 2]), mtp: true), .ineligible)       // R + 3 tokens
        XCTAssertEqual(capture(s, prompt(head: 1, tail: [7, 1, 2, 3]), mtp: true), .stored)        // R + 4
        XCTAssertNil(s.lookup(prompt(head: 1, tail: [7, 1, 2]), preferMTP: true))
        XCTAssertNil(s.lookup(prompt(head: 1, tail: [7, 1, 2])))
        XCTAssertEqual(s.lookup(prompt(head: 1, tail: [7, 9, 9, 9]), preferMTP: true)?.length, R)
        XCTAssertEqual(s.sharedStores, 1)
    }

    func testOrdinaryEntriesWinLongerAndEqualRungsAndLoseShorterOnes() throws {
        let s = store()
        let p = prompt(head: 4, tail: [7] + Array(1 ... 17))
        capture(s, p, mtp: true)
        // another conversation shares only 10 tokens with a rung at 8: the shared rung is higher
        let other = Array(p.prefix(10)) + [555, 556, 557, 558, 559, 560]
        s.store(tokens: other, rows: rows(cap: 16, salt: 1, mtp: true), rowsValidTo: 16, mtpValidTo: 16,
                rungs: [rung(8, salt: 1), rung(16, salt: 1)])
        XCTAssertEqual(s.lookup(p, preferMTP: true)?.length, R)
        XCTAssertEqual(s.lookup(p, preferMTP: true)?.isSharedPrefix, true)
        // an earlier turn of the same conversation with a rung past R wins by length
        s.store(tokens: Array(p.prefix(R + 8)), rows: rows(cap: R + 8, salt: 2, mtp: true), rowsValidTo: R + 8, mtpValidTo: R + 8,
                rungs: [rung(R + 8, salt: 2)])
        let longer = try XCTUnwrap(s.lookup(p, preferMTP: true))
        XCTAssertEqual(longer.length, R + 8); XCTAssertFalse(longer.isSharedPrefix)
        // an ordinary rung at exactly R ties and keeps winning (ordinary entries are scanned first)
        let s2 = store()
        capture(s2, p, mtp: true)
        s2.store(tokens: Array(p.prefix(R)) + [777], rows: rows(cap: R + 1, salt: 3, mtp: true), rowsValidTo: R + 1,
                 mtpValidTo: R + 1, rungs: [rung(R, salt: 3)])
        let tie = try XCTUnwrap(s2.lookup(p, preferMTP: true))
        XCTAssertEqual(tie.length, R); XCTAssertFalse(tie.isSharedPrefix)
        XCTAssertEqual(tie.selectedRung.head["trunk.L1.a0"]?.item(Float.self), Float(R) + 3)
        // a finished store never supersedes or inherits from a shared entry
        s2.store(tokens: p, rows: rows(cap: p.count, salt: 4, mtp: true), rowsValidTo: p.count, mtpValidTo: p.count,
                 rungs: [rung(p.count, salt: 4)])
        XCTAssertEqual(s2.sharedCount, 1)
        XCTAssertFalse(s2.entries.contains { $0.shared })
    }

    func testLRUBoundAmongSharedRungs() {
        let s = store(capacity: 2)
        let a = prompt(head: 5, tail: [7, 1]), b = prompt(head: 6, tail: [7, 1]), c = prompt(head: 7, tail: [7, 1])
        XCTAssertEqual(capture(s, a, mtp: true), .stored)
        XCTAssertEqual(capture(s, b, mtp: true), .stored)
        // A is resumed after B was stored: B is now the least recently used
        let hit = s.lookup(prompt(head: 5, tail: [7, 9, 9]), preferMTP: true)!
        XCTAssertNotNil(s.exported(hit, ratio: ratio))
        XCTAssertEqual(capture(s, c, mtp: true), .stored)
        XCTAssertEqual(s.sharedCount, 2)
        XCTAssertNotNil(s.lookup(a, preferMTP: true)); XCTAssertNil(s.lookup(b, preferMTP: true)); XCTAssertNotNil(s.lookup(c, preferMTP: true))
        XCTAssertEqual(s.sharedLRUEvictions, 1)
        // a duplicate capture refreshes its entry: C was stored last, but A's dedupe makes C the LRU
        XCTAssertEqual(capture(s, a, mtp: true), .deduplicated)
        XCTAssertEqual(capture(s, b, mtp: true), .stored)
        XCTAssertNotNil(s.lookup(a, preferMTP: true)); XCTAssertNil(s.lookup(c, preferMTP: true))
        XCTAssertEqual(s.sharedLRUEvictions, 2)
        // an upgrade replaces its own trunk-only entry, not the LRU
        let t = store(capacity: 2)
        capture(t, a, mtp: false); capture(t, b, mtp: true)
        XCTAssertEqual(capture(t, a, mtp: true), .upgraded)
        XCTAssertEqual(t.sharedCount, 2); XCTAssertEqual(t.sharedLRUEvictions, 0)
        XCTAssertNotNil(t.lookup(b, preferMTP: true))
    }

    // MARK: accounting

    func testAccountingIsChargedToTheStoreAndOrdinaryPressureNeverEvictsIt() throws {
        let s = store()
        capture(s, prompt(head: 8, tail: [7, 1]), mtp: true)
        let e = try XCTUnwrap(s.sharedEntries.first)
        // logical: rows (2 trunk layers x (k 384 + v 384 + i0 128 + i1 32) + head k/v 2 x 384 = 2624) + rung (4 + 24)
        XCTAssertEqual(s.sharedLogicalBytes, 2624 + 28)
        XCTAssertEqual(s.totalBytes, s.sharedLogicalBytes)
        // charged: every array plus its backing slack, plus the token ids (as for an ordinary entry)
        XCTAssertEqual(s.sharedChargedBytes, (e.rows.count + e.rungs[0].head.count) * slack + 2652 + R * MemoryLayout<Int>.stride)
        XCTAssertEqual(s.chargedBytes, s.sharedChargedBytes)
        let snap = s.sharedPrefixSnapshot()
        XCTAssertEqual(snap["bytes"], s.sharedLogicalBytes); XCTAssertEqual(snap["charged_bytes"], s.sharedChargedBytes)
        XCTAssertEqual(snap["entries"], 1); XCTAssertEqual(snap["entries_with_mtp"], 1); XCTAssertEqual(snap["enabled"], 1)
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: snap))
        // the legacy fixed cap: a store that overflows it evicts ordinary entries (keeping one), never the shared rung
        let cap = s.sharedLogicalBytes + 3000
        let legacy = store(cap: cap)
        capture(legacy, prompt(head: 8, tail: [7, 1]), mtp: true)
        for i in 0 ..< 4 {
            legacy.store(tokens: Array((100 * i + 50) ..< (100 * i + 66)), rows: rows(cap: 16, salt: Float(i), mtp: false),
                         rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: Float(i))])
        }
        XCTAssertEqual(legacy.sharedCount, 1)
        XCTAssertGreaterThan(legacy.evictions, 0)
        XCTAssertLessThanOrEqual(legacy.totalBytes, cap)
        // a cap below the shared rung plus one entry: the legacy policy keeps its one oversized entry (as today) and
        // the shared rung is still not a victim -- the overshoot is the legacy cap's own one entry PLUS the shared
        // entries, which the protected share bounds (cap / divisor; here the whole cap, see the share tests). The server
        // refuses the knob without the strict budget, so this mode is the store's contract only.
        let tight = store(cap: s.sharedLogicalBytes + 1000)
        capture(tight, prompt(head: 8, tail: [7, 1]), mtp: true)
        tight.store(tokens: Array(50 ..< 66), rows: rows(cap: 16, salt: 0, mtp: false), rowsValidTo: 16, mtpValidTo: 0,
                    rungs: [rung(16, salt: 0)])
        XCTAssertEqual(tight.sharedCount, 1); XCTAssertEqual(tight.count, 1); XCTAssertEqual(tight.sharedForcedEvictions, 0)
    }

    func testStrictBudgetKeepsTheSharedRungUnderOrdinaryPressureAndASqueezeTheOrdinaryEntriesCanMeet() throws {
        let probe = store()
        capture(probe, prompt(head: 8, tail: [7, 1]), mtp: true)
        let shared = probe.sharedChargedBytes
        let ordinary: (HotPrefixStore, Int) -> Void = { s, i in
            s.store(tokens: Array((100 * i + 50) ..< (100 * i + 66)), rows: ["trunk.L0.k": MLXArray.zeros([1, 2, 16, 3])],
                    rowsValidTo: 16, mtpValidTo: 0, rungs: [self.rung(16, salt: Float(i))])
        }
        let one = HotPrefixStore(capBytes: 1 << 30, strictBudget: true); ordinary(one, 0)
        let entry = one.chargedBytes
        // room for the shared rung and two ordinary entries
        let s = store(cap: shared + 2 * entry + 10, strict: true)
        XCTAssertEqual(capture(s, prompt(head: 8, tail: [7, 1]), mtp: true), .stored)
        for i in 0 ..< 5 { ordinary(s, i) }
        XCTAssertEqual(s.sharedCount, 1, "ordinary stores make room from ordinary entries only")
        XCTAssertEqual(s.count, 2)
        XCTAssertLessThanOrEqual(s.chargedBytes, s.budgetBytes)
        // an ordinary store that could only fit in the shared rung's bytes is refused before any copy
        let copies = s.copyAttempts, rejected = s.rejectedStores
        let width = (entry + shared / 2 + 394) / 128 + 1   // charge in (2 entries + 10, shared + 2 entries + 10]: fits only in the shared bytes
        s.store(tokens: Array(5000 ..< 5016), rows: ["trunk.L0.k": MLXArray.zeros([1, 2, 16, width])],
                rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 9)])
        XCTAssertEqual(s.rejectedStores, rejected + 1); XCTAssertEqual(s.copyAttempts, copies)
        XCTAssertEqual(s.count, 2, "and nothing was evicted for it")
        // a squeeze the ordinary entries can meet keeps the shared rung (the 12.9 GB case: 8 clamped rows)
        XCTAssertTrue(s.setBudgetBytes(shared + 1))
        XCTAssertEqual(s.count, 0); XCTAssertEqual(s.sharedCount, 1); XCTAssertEqual(s.sharedForcedEvictions, 0)
        XCTAssertNotNil(s.lookup(prompt(head: 8, tail: [7, 2]), preferMTP: true))
        // a squeeze below it reclaims it: an admission is never refused to keep a shared rung
        XCTAssertTrue(s.setBudgetBytes(shared - 1))
        XCTAssertEqual(s.sharedCount, 0); XCTAssertEqual(s.sharedForcedEvictions, 1); XCTAssertEqual(s.chargedBytes, 0)
        // a shared capture that could not fit even alone is refused before any copy, keeping what is stored
        XCTAssertEqual(capture(s, prompt(head: 9, tail: [7, 1]), mtp: true), .rejected)
        XCTAssertEqual(s.sharedRejected, 1); XCTAssertEqual(s.sharedCount, 0)
        // and the physical guard refuses a shared copy like an ordinary one
        XCTAssertTrue(s.setBudgetBytes(s.capBytes))
        s.permitAllocation = { _ in false }
        XCTAssertEqual(capture(s, prompt(head: 9, tail: [7, 1]), mtp: true), .rejected)
        XCTAssertEqual(s.sharedCount, 0)
        s.permitAllocation = nil
        XCTAssertEqual(capture(s, prompt(head: 9, tail: [7, 1]), mtp: true), .stored)
    }

    func testARefusedCaptureLosesNothingItWouldHaveReplaced() throws {
        // Review F3: the replaced entries used to be dropped before the physical guard; a refusal then lost them.
        // (a) an upgrade refused: the trunk-only rung it would have replaced still serves plain requests
        let s = store(strict: true)
        let p = prompt(head: 2, tail: [7, 1, 2])
        XCTAssertEqual(capture(s, p, mtp: false), .stored)
        s.permitAllocation = { _ in false }
        XCTAssertEqual(capture(s, p, mtp: true), .rejected)
        XCTAssertEqual(s.sharedCount, 1); XCTAssertEqual(s.sharedUpgrades, 0); XCTAssertEqual(s.sharedLRUEvictions, 0)
        XCTAssertEqual(s.sharedEntries.first?.mtpValidTo, 0)
        XCTAssertEqual(s.lookup(p + [3])?.length, R, "the plain hit it had before")
        s.permitAllocation = nil
        XCTAssertEqual(capture(s, p, mtp: true), .upgraded)
        XCTAssertEqual(s.sharedCount, 1); XCTAssertEqual(s.sharedEntries.first?.mtpValidTo, R)
        // (b) at capacity, refused: the LRU entry it would have displaced stays, in its place
        let t = store(capacity: 2, strict: true)
        let a = prompt(head: 5, tail: [7, 1]), b = prompt(head: 6, tail: [7, 1]), c = prompt(head: 7, tail: [7, 1])
        capture(t, a, mtp: true); capture(t, b, mtp: true)
        let before = t.sharedEntries.map(\.tokens), charged = t.chargedBytes
        t.permitAllocation = { _ in false }
        XCTAssertEqual(capture(t, c, mtp: true), .rejected)
        XCTAssertEqual(t.sharedEntries.map(\.tokens), before); XCTAssertEqual(t.chargedBytes, charged)
        XCTAssertEqual(t.sharedLRUEvictions, 0); XCTAssertEqual(t.sharedRejected, 1); XCTAssertEqual(t.sharedCopyAttempts, 2)
        XCTAssertNotNil(t.lookup(a, preferMTP: true)); XCTAssertNotNil(t.lookup(b, preferMTP: true))
        t.permitAllocation = nil
        XCTAssertEqual(capture(t, c, mtp: true), .stored)
        XCTAssertEqual(t.sharedLRUEvictions, 1); XCTAssertEqual(t.sharedCount, 2)
    }

    // MARK: review fixes (F1 selection, F2 admission, share bound, witness)

    func testASharedRungNeverDisplacesALongerOrdinaryHitEvenOneWithoutTheHead() throws {
        // Review F1: an MTP request whose only ordinary hit is long but head-less (stored after a logprobs / hard-guard /
        // serial turn) used to fall back to it; the always-present shared rung must not turn that into a re-prefill.
        let p = prompt(head: 4, tail: [7] + Array(1 ... 40))
        func withConversation(_ s: HotPrefixStore) {
            s.store(tokens: Array(p.prefix(R + 30)), rows: rows(cap: R + 30, salt: 2, mtp: false), rowsValidTo: R + 30,
                    mtpValidTo: 0, rungs: [rung(R + 24, salt: 2), rung(R + 30, salt: 2)])
        }
        let s = store(); capture(s, p, mtp: true); withConversation(s)
        let today = HotPrefixStore(capBytes: 1 << 30, rungStep: 4); withConversation(today)
        let h = try XCTUnwrap(s.lookup(p, preferMTP: true))
        XCTAssertEqual(h.length, R + 30); XCTAssertFalse(h.isSharedPrefix); XCTAssertFalse(h.hasMTP)
        XCTAssertEqual(h.length, today.lookup(p, preferMTP: true)?.length, "the pre-P119 decision")
        XCTAssertEqual(s.lookup(p)?.length, R + 30)
        // shorter head-less fallback (below the rung): the shared rung is longer AND carries the head -- it wins
        let u = store(); capture(u, p, mtp: true)
        u.store(tokens: Array(p.prefix(12)) + [900, 901, 902, 903], rows: rows(cap: 16, salt: 3, mtp: false), rowsValidTo: 16,
                mtpValidTo: 0, rungs: [rung(12, salt: 3)])
        XCTAssertEqual(u.lookup(p, preferMTP: true)?.isSharedPrefix, true)
        // equal length, the ordinary one head-less: the shared rung wins the tie for an MTP request only
        let v = store(); capture(v, p, mtp: true)
        v.store(tokens: Array(p.prefix(R)) + [900, 901], rows: rows(cap: R + 2, salt: 4, mtp: false), rowsValidTo: R + 2,
                mtpValidTo: 0, rungs: [rung(R, salt: 4)])
        let tie = try XCTUnwrap(v.lookup(p, preferMTP: true))
        XCTAssertTrue(tie.isSharedPrefix); XCTAssertTrue(tie.hasMTP)
        XCTAssertEqual(v.lookup(p)?.isSharedPrefix, false, "a plain request keeps the ordinary rung on a tie")
        // an ordinary MTP rung shorter than R loses to the shared rung (which is what the rung is for)
        let w = store(); capture(w, p, mtp: true)
        w.store(tokens: Array(p.prefix(8)) + [900, 901], rows: rows(cap: 10, salt: 5, mtp: true), rowsValidTo: 10,
                mtpValidTo: 10, rungs: [rung(8, salt: 5)])
        XCTAssertEqual(w.lookup(p, preferMTP: true)?.length, R)
    }

    func testAHeadEarnsTheSplitAndCaptureOnlyOnItsSecondColdSighting() {
        // Review F2/F7: no split, no copy, no eviction for a head nobody has asked for twice.
        let s = store()
        s.sharedPrefixSightingCapacity = 2
        let a1 = prompt(head: 1, tail: [7, 1]), a2 = prompt(head: 1, tail: [8, 2, 3])     // same head, other tails
        let b = prompt(head: 2, tail: [7, 1]), c = prompt(head: 3, tail: [7, 1])
        XCTAssertFalse(s.admitSharedPrefix(prompt: a1), "first sighting: the unsplit schedule, nothing stored")
        XCTAssertTrue(s.admitSharedPrefix(prompt: a2), "second: the head has a reuse")
        XCTAssertFalse(s.admitSharedPrefix(prompt: b))
        XCTAssertEqual(s.sharedFirstSightings, 2); XCTAssertEqual(s.sharedAdmissions, 1)
        XCTAssertEqual(s.sharedCount, 0, "admission alone stores nothing")
        // bounded: two remembered heads; a third forgets the least recently seen (a: seen at tick 2, b at 3)
        XCTAssertFalse(s.admitSharedPrefix(prompt: c))
        XCTAssertFalse(s.admitSharedPrefix(prompt: a1), "a was forgotten")
        XCTAssertTrue(s.admitSharedPrefix(prompt: c))
        XCTAssertEqual(s.sharedPrefixSnapshot()["sightings"], 2)
        // a shared entry of these ids admits whatever the sightings remember (an MTP upgrade, another next token)
        let t = store(); t.sharedPrefixSightingCapacity = 1
        capture(t, b, mtp: false)
        XCTAssertFalse(t.admitSharedPrefix(prompt: c))           // b's sighting is not what admits it below
        XCTAssertTrue(t.admitSharedPrefix(prompt: prompt(head: 2, tail: [9, 9])))
        // ineligible prompts are neither recorded nor admitted; the head must match exactly
        t.sharedPrefixMinPrompt = R + 4
        XCTAssertFalse(t.admitSharedPrefix(prompt: prompt(head: 5, tail: [1, 2, 3])))
        XCTAssertFalse(t.admitSharedPrefix(prompt: prompt(head: 5, tail: [1, 2, 3])))
        var drift = prompt(head: 6, tail: [1, 2, 3, 4]); XCTAssertFalse(t.admitSharedPrefix(prompt: drift))
        drift[R - 1] = -9; XCTAssertFalse(t.admitSharedPrefix(prompt: drift), "one token of the head differs: a new head")
        // knob off: never admitted, nothing recorded
        let off = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        XCTAssertFalse(off.admitSharedPrefix(prompt: a1)); XCTAssertFalse(off.admitSharedPrefix(prompt: a1))
        XCTAssertEqual(off.sharedFirstSightings + off.sharedAdmissions, 0); XCTAssertEqual(off.sharedPrefixSnapshot()["sightings"], 0)
    }

    func testTheSharedEntriesStayInsideTheirShareOfTheBudget() throws {
        // Review F2 (memory): shared rungs are protected from ordinary pressure only inside budget / divisor, so no MAX
        // can crowd out the conversations; a capture makes room from LRU shared entries, a squeeze trims them first.
        let probe = store(); capture(probe, prompt(head: 8, tail: [7, 1]), mtp: true)
        let one = probe.sharedChargedBytes
        let ordinary: (HotPrefixStore, Int) -> Void = { s, i in
            s.store(tokens: Array((100 * i + 50) ..< (100 * i + 66)), rows: ["trunk.L0.k": MLXArray.zeros([1, 2, 16, 3])],
                    rowsValidTo: 16, mtpValidTo: 0, rungs: [self.rung(16, salt: Float(i))])
        }
        // divisor 8, budget 8 x 2.5 rungs: the share holds two rungs although the capacity is 8
        let s = store(capacity: 8, cap: 8 * (2 * one + one / 2), strict: true, shareDivisor: 8)
        XCTAssertEqual(s.sharedShareBytes, 2 * one + one / 2)
        for head in 1 ... 3 { XCTAssertEqual(capture(s, prompt(head: head, tail: [7, 1]), mtp: true), .stored) }
        XCTAssertEqual(s.sharedCount, 2); XCTAssertEqual(s.sharedLRUEvictions, 1)
        XCTAssertNil(s.lookup(prompt(head: 1, tail: [7, 2]), preferMTP: true), "head 1 was the least recently used")
        XCTAssertLessThanOrEqual(s.sharedChargedBytes, s.sharedShareBytes)
        // ordinary stores overfill the other 7/8 (they evict each other) and never touch the shared entries
        let evictionsBefore = s.evictions
        for i in 0 ..< 100 { ordinary(s, i) }
        XCTAssertGreaterThan(s.evictions, evictionsBefore)
        XCTAssertEqual(s.sharedCount, 2); XCTAssertGreaterThan(s.count, 0); XCTAssertEqual(s.sharedForcedEvictions, 0)
        XCTAssertLessThanOrEqual(s.chargedBytes, s.budgetBytes)
        XCTAssertGreaterThanOrEqual(s.budgetBytes - s.sharedChargedBytes, s.budgetBytes / 8 * 7)
        // a squeeze whose share holds one rung trims the LRU one FIRST (ordinary entries keep the rest of the budget)
        let ordinaryBefore = s.count
        XCTAssertTrue(s.setBudgetBytes(8 * (one + one / 2)))
        XCTAssertEqual(s.sharedCount, 1); XCTAssertEqual(s.sharedForcedEvictions, 1)
        XCTAssertNotNil(s.lookup(prompt(head: 3, tail: [7, 2]), preferMTP: true), "the most recently used rung is kept")
        XCTAssertGreaterThan(s.count, 0); XCTAssertLessThanOrEqual(s.count, ordinaryBefore)
        XCTAssertLessThanOrEqual(s.chargedBytes, s.budgetBytes)
        // a rung that alone exceeds the share is refused before anything is touched
        XCTAssertTrue(s.setBudgetBytes(8 * (one - 1)))
        XCTAssertEqual(s.sharedCount, 0)
        let (copies, lru) = (s.sharedCopyAttempts, s.sharedLRUEvictions)
        XCTAssertEqual(capture(s, prompt(head: 9, tail: [7, 1]), mtp: true), .rejected)
        XCTAssertEqual(s.sharedCopyAttempts, copies); XCTAssertEqual(s.sharedLRUEvictions, lru)
        // legacy mode (review F1): the share is of the fixed cap, in its logical bytes -- a rung is never stored on top
        let logical = probe.sharedLogicalBytes
        let legacy = store(capacity: 8, cap: 8 * (logical + logical / 2), shareDivisor: 8)
        XCTAssertEqual(capture(legacy, prompt(head: 1, tail: [7, 1]), mtp: true), .stored)
        XCTAssertEqual(capture(legacy, prompt(head: 2, tail: [7, 1]), mtp: true), .stored)
        XCTAssertEqual(legacy.sharedCount, 1); XCTAssertEqual(legacy.sharedLRUEvictions, 1)
        XCTAssertLessThanOrEqual(legacy.sharedLogicalBytes, legacy.capBytes / 8)
        let small = store(cap: 8 * (logical - 1), shareDivisor: 8)
        XCTAssertEqual(capture(small, prompt(head: 1, tail: [7, 1]), mtp: true), .rejected, "alone over the share: not stored")
        XCTAssertEqual(small.totalBytes, 0)
    }

    func testTheHotCacheSnapshotSaysWhatIsSharedAndIsTodaysWithTheKnobOff() throws {
        // Review F5: the shared bytes are in logical/charged_bytes while entries/rungs count ordinary entries only.
        let todayKeys: Set<String> = ["ceiling_bytes", "target_bytes", "logical_bytes", "charged_bytes", "entries", "rungs",
                                      "strict_budget", "rejected_stores", "pre_copy_evictions", "copy_attempts", "inflight_entries"]
        XCTAssertEqual(Set(HotPrefixStore(capBytes: 1 << 30, strictBudget: true).snapshot().keys), todayKeys)
        let s = store(cap: 1 << 30, strict: true)
        capture(s, prompt(head: 1, tail: [7, 1]), mtp: true)
        s.store(tokens: Array(50 ..< 66), rows: rows(cap: 16, salt: 0, mtp: false), rowsValidTo: 16, mtpValidTo: 0,
                rungs: [rung(16, salt: 0)])
        let snap = s.snapshot()
        XCTAssertEqual(Set(snap.keys), todayKeys.union(["shared_entries", "shared_logical_bytes", "shared_charged_bytes",
                                                        "shared_capture_evictions", "shared_copy_attempts"]))
        XCTAssertEqual(snap["entries"], 1); XCTAssertEqual(snap["shared_entries"], 1)
        XCTAssertEqual(snap["charged_bytes"]! - snap["shared_charged_bytes"]!, s.entries[0].chargedBytes)
        XCTAssertEqual(snap["logical_bytes"]! - snap["shared_logical_bytes"]!, s.entries[0].bytes)
        XCTAssertEqual(snap["copy_attempts"], 1); XCTAssertEqual(snap["shared_copy_attempts"], 1)
        // ordinary entries a capture evicts are counted as such
        let t = store(cap: 1 << 30, strict: true)
        t.store(tokens: Array(50 ..< 66), rows: rows(cap: 16, salt: 0, mtp: false), rowsValidTo: 16, mtpValidTo: 0,
                rungs: [rung(16, salt: 0)])
        let probe = store(); capture(probe, prompt(head: 1, tail: [7, 1]), mtp: true)
        XCTAssertTrue(t.setBudgetBytes(t.chargedBytes + probe.sharedChargedBytes - 1))
        XCTAssertEqual(capture(t, prompt(head: 1, tail: [7, 1]), mtp: true), .stored)
        XCTAssertEqual(t.count, 0); XCTAssertEqual(t.snapshot()["shared_capture_evictions"], 1)
        XCTAssertEqual(t.sharedPrefixSnapshot()["capture_evictions"], 1)
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: t.sharedPrefixSnapshot()))
    }

    // MARK: knob off (negative control)

    func testKnobOffStoresNothingAndEveryOrdinaryDecisionIsTodays() {
        // The same ordinary script on (a) today's store, (b) a store with the knob on but no capture yet, and (c) a store
        // whose knob is off but is asked to capture: every lookup, eviction, budget decision and counter must agree.
        func run(_ s: HotPrefixStore, captureShared: Bool) -> [String] {
            var log: [String] = []
            if captureShared {
                let outcome = s.storeSharedPrefix(prompt: prompt(head: 1, tail: [7, 1]), rows: rows(cap: 40, salt: 0, mtp: true),
                                                  mtpNextToken: 7, rung: rung(R, salt: 0), ratio: ratio)
                log.append("capture \(outcome.rawValue)")
            }
            let convo = prompt(head: 1, tail: [7] + Array(1 ... 23))          // shares the head the capture would have
            s.store(tokens: Array(convo.prefix(24)), rows: rows(cap: 24, salt: 1, mtp: true), rowsValidTo: 24, mtpValidTo: 24,
                    rungs: [rung(8, salt: 1), rung(24, salt: 1)])
            s.store(tokens: Array(convo.prefix(32)), rows: rows(cap: 32, salt: 2, mtp: true), rowsValidTo: 32, mtpValidTo: 20,
                    rungs: [rung(28, salt: 2), rung(32, salt: 2)])
            s.store(tokens: Array(600 ..< 640), rows: rows(cap: 40, salt: 3, mtp: false), rowsValidTo: 40, mtpValidTo: 0,
                    rungs: [rung(40, salt: 3)], inFlight: true, owner: 3)
            for q in [prompt(head: 1, tail: [7, 1, 2]), prompt(head: 1, tail: [7] + Array(1 ... 30)), convo + [5],
                      prompt(head: 2, tail: [7, 1]), Array(600 ..< 650)] {
                for mtp in [false, true] {
                    let h = s.lookup(q, preferMTP: mtp)
                    log.append("lookup \(q.count) \(mtp) -> \(h?.length ?? -1) \(h?.hasMTP ?? false)")
                }
            }
            log.append("budget \(s.setBudgetBytes(s.chargedBytes / 2)) count \(s.count) evictions \(s.evictions)")
            log.append("reap \(s.reapInFlight(owner: 3))")
            log.append("stats \(s.hits) \(s.misses) \(s.stores) \(s.coalescedStores) \(s.rejectedStores)")
            return log
        }
        let today = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        let armed = store(cap: 1 << 30, strict: true)
        let off = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        XCTAssertFalse(off.sharedPrefixEnabled)
        let reference = run(today, captureShared: false)
        XCTAssertEqual(run(armed, captureShared: false), reference, "knob on, nothing captured: today's decisions")
        XCTAssertEqual(run(off, captureShared: true), ["capture disabled"] + reference, "knob off: a capture is a no-op")
        XCTAssertEqual(off.sharedCount, 0); XCTAssertEqual(off.sharedPrefixSnapshot()["enabled"], 0)
        XCTAssertEqual(today.snapshot(), off.snapshot())
        XCTAssertFalse(off.admitSharedPrefix(prompt: prompt(head: 1, tail: [7, 1])))
        XCTAssertEqual(off.sharedPrefixSnapshot()["sightings"], 0)
        // armed but nothing captured: the hot_cache witness differs only by the shared_* keys (all zero)
        let extra = armed.snapshot().filter { today.snapshot()[$0.key] == nil }
        XCTAssertEqual(Set(extra.keys), ["shared_entries", "shared_logical_bytes", "shared_charged_bytes",
                                         "shared_capture_evictions", "shared_copy_attempts"])
        XCTAssertTrue(extra.values.allSatisfy { $0 == 0 })
        // capacity 0 is off too
        let zero = store(capacity: 0)
        XCTAssertEqual(capture(zero, prompt(head: 1, tail: [7, 1]), mtp: true), .disabled)
        // and the server's schedule: rung 0 never cuts or captures
        for n in [1, 1024, 1025, 1280, 2049, 5000, 131_073] {
            for from in [0, 1, 1024, 2048] where from < n {
                for width in [512, 1024, 2048, 4096] {
                    let end = PrefillChunkPlan.end(from: from, width: width, rungAlign: 1, promptCount: n, qsaBudgetCut: 2048,
                                                   canonicalTarget: nil, hotReserve: 1)
                    let plan = PrefillChunkPlan.withSharedSplit(from: from, end: end, promptCount: n, rung: 0, margin: margin)
                    XCTAssertEqual(plan.end, end); XCTAssertFalse(plan.split)
                    XCTAssertFalse(PrefillChunkPlan.capturesSharedRung(from: from, to: end, promptCount: n, rung: 0, margin: margin))
                }
            }
        }
    }

    // MARK: the server's schedule (width-canonical argument)

    private struct Chunk: Equatable { let from: Int, to: Int, split: Bool }
    /// The production geometry: rungAlign 1 (no disk cache), QSA budget cut 2048, one-token hot reserve.
    private func schedule(from start: Int, prompt n: Int, width: Int, rung: Int) -> [Chunk] {
        var out: [Chunk] = [], c = start
        while c < n {
            let end = PrefillChunkPlan.end(from: c, width: width, rungAlign: 1, promptCount: n, qsaBudgetCut: 2048,
                                           canonicalTarget: nil, hotReserve: 1)
            let plan = PrefillChunkPlan.withSharedSplit(from: c, end: end, promptCount: n, rung: rung, margin: margin)
            XCTAssertGreaterThan(plan.end, c)
            out.append(Chunk(from: c, to: plan.end, split: plan.split)); c = plan.end
        }
        return out
    }

    func testTheKnobOnlyCutsAColdPromptsFirstChunkAt1024AndTheResumeIsTheColdScheduleAfterIt() {
        for n in [1280, 1321, 1500, 2047, 2048, 2049, 2050, 2304, 3000, 4097, 5000, 8193, 17000, 70001, 258_089] {
            // alone: 4096 chunks, the first cut at the QSA budget
            let off = schedule(from: 0, prompt: n, width: 4096, rung: 0)
            let on = schedule(from: 0, prompt: n, width: 4096, rung: 1024)
            XCTAssertEqual(on[0], Chunk(from: 0, to: 1024, split: true), "n \(n)")
            XCTAssertEqual(Array(on.dropFirst()), [Chunk(from: 1024, to: off[0].to, split: false)] + off.dropFirst(), "n \(n)")
            XCTAssertLessThanOrEqual(off[0].to, 2048, "the unsplit first chunk never passes the budget (H57)")
            // the width-canonical law's geometry: in the dense region (to the 2048 budget) every chunk is B1-width <= 2048
            // and every interior end is 1024-aligned; only the prompt tail (the reserve split) is not
            for c in on where c.to < n - 1 { XCTAssertEqual(c.to % 1024, 0, "n \(n) chunk \(c)") }
            for c in on where c.to <= 2048 { XCTAssertLessThanOrEqual(c.to - c.from, 2048) }
            // a request resumed from the shared rung runs exactly the knob-on cold schedule after its first chunk
            XCTAssertEqual(schedule(from: 1024, prompt: n, width: 4096, rung: 1024), Array(on.dropFirst()), "n \(n)")
            // the rung is captured at exactly one chunk end: [0, 1024)
            XCTAssertEqual(on.filter { PrefillChunkPlan.capturesSharedRung(from: $0.from, to: $0.to, promptCount: n, rung: 1024, margin: margin) },
                           [on[0]])
            // shared regime: [0, 1024) is already the natural first chunk -- nothing is cut, the rung is still captured
            let sharedOff = schedule(from: 0, prompt: n, width: 1024, rung: 0)
            let sharedOn = schedule(from: 0, prompt: n, width: 1024, rung: 1024)
            XCTAssertEqual(sharedOn, sharedOff, "n \(n)")
            XCTAssertTrue(PrefillChunkPlan.capturesSharedRung(from: sharedOn[0].from, to: sharedOn[0].to, promptCount: n, rung: 1024, margin: margin))
            // B40-style 2048 alone: the same single cut
            let off2048 = schedule(from: 0, prompt: n, width: 2048, rung: 0)
            let on2048 = schedule(from: 0, prompt: n, width: 2048, rung: 1024)
            XCTAssertEqual(Array(on2048.dropFirst()), [Chunk(from: 1024, to: off2048[0].to, split: false)] + off2048.dropFirst())
        }
        // below rung + margin: nothing is cut or captured (the split would cost one more MoE sweep for little reuse)
        for n in [2, 1024, 1025, 1100, 1279] {
            XCTAssertEqual(schedule(from: 0, prompt: n, width: 4096, rung: 1024), schedule(from: 0, prompt: n, width: 4096, rung: 0))
            XCTAssertFalse(PrefillChunkPlan.sharedSplitEligible(promptCount: n, rung: 1024, margin: margin))
        }
        // a resumed request (from > 0) is never cut
        XCTAssertEqual(PrefillChunkPlan.withSharedSplit(from: 512, end: 2048, promptCount: 5000, rung: 1024, margin: margin).end, 2048)
        // review F4: the chunk after the rung never has fewer rows than MLX's row-count kernel thresholds (the MoE
        // gather S >= 205, qmm split-K M >= 193): the shortest eligible prompt leaves exactly margin - 1 rows there
        XCTAssertGreaterThanOrEqual(margin - 1, PrefillChunkPlan.sharedPrefixMinTailRows)
        XCTAssertGreaterThanOrEqual(PrefillChunkPlan.sharedPrefixMinTailRows * 10 / 512, 4)
        XCTAssertLessThan((PrefillChunkPlan.sharedPrefixMinTailRows - 1) * 10 / 512, 4)
        for width in [1024, 2048, 4096] {
            let on = schedule(from: 0, prompt: 1024 + margin, width: width, rung: 1024)
            XCTAssertEqual(on[1], Chunk(from: 1024, to: 1024 + margin - 1, split: false))
            XCTAssertEqual(on.map { $0.to - $0.from }.min(), 1, "only the one-token reserve chunk is narrower")
        }
    }

    func testChunkEndArithmeticIsServesOwn() {
        // Pins PrefillChunkPlan.end to the pre-P119 Serve.swift chunkEnd on the geometries the gates know (H57/B43 table).
        let e = { (from: Int, n: Int, w: Int, align: Int, target: Int?) in
            PrefillChunkPlan.end(from: from, width: w, rungAlign: align, promptCount: n, qsaBudgetCut: 2048, canonicalTarget: target, hotReserve: 1)
        }
        XCTAssertEqual(e(0, 131_072, 4096, 1, nil), 2048)          // the budget cut
        XCTAssertEqual(e(2048, 131_072, 4096, 1, nil), 4096)       // width-aligned, not 6144 (the withdrawn first B43 freeze)
        XCTAssertEqual(e(4096, 131_072, 4096, 1, nil), 8192)
        XCTAssertEqual(e(0, 5000, 1024, 1, nil), 1024)
        XCTAssertEqual(e(4096, 5000, 4096, 1, nil), 4999)          // the one-token hot reserve
        XCTAssertEqual(e(4999, 5000, 4096, 1, nil), 5000)
        XCTAssertEqual(e(0, 5000, 4096, 512, nil), 2048)           // disk rung alignment
        XCTAssertEqual(e(700, 5000, 1024, 512, nil), 1024)
        XCTAssertEqual(e(0, 5000, 4096, 1, 1024), 1024)            // H25 canonical target (diagnostic)
    }
}
