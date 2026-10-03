import XCTest
import MLX
@testable import Qwen4Exp

/// P093 -- the hot prefix store's model-free core: what it keeps, what it resumes from, and that a
/// materialised rung is the rows sliced to its length under the rung's own head arrays.
final class HotPrefixStoreTests: XCTestCase {
    private let ratio = 4

    override func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    private func address(_ array: MLXArray) -> UInt {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    }

    func testFinalAndInflightStoresDetachB8ViewsWithExactBytes() throws {
        for inFlight in [false, true] {
            let length = 1024
            let pool = MLXArray((0..<(8 * 2 * length * 16)).map { Float($0) }).reshaped([8, 2, length, 16])
            let headPool = MLXArray((0..<(8 * 4 * 16 * 32)).map { Float($0) / 4 }).reshaped([8, 4, 16, 32])
            let row = pool[3..<4], head = headPool[3..<4]
            eval(row, head)
            let tokens = Array(0..<length)
            let rawRung = HotPrefixStore.Rung(length: length, head: ["trunk.L1.a1": head], bytes: head.nbytes)
            // Control: the old Entry construction retains row views and charges only nbytes.
            let control = HotPrefixStore.Entry(tokens: tokens, rows: ["trunk.L0.k": row],
                                               rowsValidTo: length, mtpValidTo: 0, rungs: [rawRung])
            XCTAssertEqual(address(try XCTUnwrap(control.rows["trunk.L0.k"])),
                           address(pool) + UInt(3 * row.nbytes))
            XCTAssertEqual(control.rowBytes * 8, pool.nbytes)
            let rowBytes = row.asData().data, headBytes = head.asData().data
            let store = HotPrefixStore(capBytes: 1 << 24)
            store.store(tokens: tokens, rows: control.rows, rowsValidTo: length, mtpValidTo: 0,
                        rungs: [rawRung], inFlight: inFlight, owner: 7)
            let stored = try XCTUnwrap(store.entries.first)
            let storedRow = try XCTUnwrap(stored.rows["trunk.L0.k"])
            let storedHead = try XCTUnwrap(stored.rungs.first?.head["trunk.L1.a1"])
            XCTAssertNotEqual(address(storedRow), address(row), "retaining a B1 view would pin the B8 row buffer")
            XCTAssertNotEqual(address(storedHead), address(head), "the final rung also must detach from its B8 state")
            XCTAssertEqual(storedRow.asData().data, rowBytes)
            XCTAssertEqual(storedHead.asData().data, headBytes)
            XCTAssertEqual(store.totalBytes, row.nbytes + head.nbytes)
            XCTAssertEqual(stored.inFlight, inFlight)
            let hit = try XCTUnwrap(store.lookup(tokens + [9999]))
            let restored = try XCTUnwrap(store.exported(hit, ratio: ratio))
            XCTAssertEqual(restored["trunk.L0.k"]!.asData().data, rowBytes)
            XCTAssertEqual(restored["trunk.L1.a1"]!.asData().data, headBytes)
            // Keep the control and original allocations alive throughout pointer comparisons.
            XCTAssertEqual(control.rows["trunk.L0.k"]!.asData().data, rowBytes)
            XCTAssertEqual(pool.dim(0), 8); XCTAssertEqual(headPool.dim(0), 8)
        }
    }

    func testCapturedRungDetachesImmediatelyAndStoreReusesItsCompactBacking() throws {
        let pool = MLXArray((0..<(8 * 4 * 16 * 32)).map(Float.init)).reshaped([8, 4, 16, 32])
        let view = pool[2..<3]
        let captured = HotPrefixStore.rung(fromExported: ["trunk.L1.a1": view], length: 8, ratio: ratio)
        let compactHead = try XCTUnwrap(captured.head["trunk.L1.a1"])
        XCTAssertNotEqual(address(compactHead), address(view))
        XCTAssertEqual(compactHead.asData().data, view.asData().data)
        let store = HotPrefixStore(capBytes: 1 << 20)
        store.store(tokens: Array(0..<8), rows: [:], rowsValidTo: 8, mtpValidTo: 0, rungs: [captured])
        let storedHead = try XCTUnwrap(store.entries.first?.rungs.first?.head["trunk.L1.a1"])
        XCTAssertEqual(address(storedHead), address(compactHead), "an already compact rung must not be copied twice")
        XCTAssertEqual(storedHead.asData().data, view.asData().data)
        XCTAssertEqual(pool.dim(0), 8)
    }

    /// rows for a sequence of `T` tokens with a distinguishing `salt`, in the exported key format
    private func rows(T: Int, salt: Float, mtp: Bool) -> [String: MLXArray] {
        var d: [String: MLXArray] = [:]
        for l in [0, 2] {
            d["trunk.L\(l).k"] = MLXArray((0 ..< (2 * T * 3)).map { Float($0) + salt }).reshaped([1, 2, T, 3])
            d["trunk.L\(l).v"] = MLXArray((0 ..< (2 * T * 3)).map { Float($0) * 2 + salt }).reshaped([1, 2, T, 3])
            d["trunk.L\(l).i0"] = MLXArray((0 ..< (T * 2)).map { Float($0) + salt }).reshaped([1, T, 2])
            d["trunk.L\(l).i1"] = MLXArray((0 ..< ((T / ratio) * 2)).map { Float($0) + salt }).reshaped([1, T / ratio, 2])
        }
        if mtp {
            d["mtp.L0.k"] = MLXArray((0 ..< (2 * T * 3)).map { Float($0) + 100 + salt }).reshaped([1, 2, T, 3])
            d["mtp.L0.v"] = MLXArray((0 ..< (2 * T * 3)).map { Float($0) + 200 + salt }).reshaped([1, 2, T, 3])
        }
        return d
    }
    private func rung(_ length: Int, salt: Float) -> HotPrefixStore.Rung {
        HotPrefixStore.rung(fromExported: ["trunk.L1.a0": MLXArray([Float(length) + salt]).reshaped([1, 1]),
                                           "trunk.L1.a1": MLXArray((0 ..< 6).map { Float($0) + salt }).reshaped([1, 2, 3]),
                                           "trunk.L0.k": MLXArray.zeros([1, 2, 5, 3])],   // a row array must NOT be part of a rung
                            length: length, ratio: ratio)
    }

    func testRungKeepsOnlyHeadArrays() {
        let r = rung(8, salt: 0)
        XCTAssertEqual(Set(r.head.keys), ["trunk.L1.a0", "trunk.L1.a1"])
        XCTAssertEqual(r.bytes, 4 + 24)
    }

    func testLookupTakesHighestRungUnderCommonPrefixAndShortOfPrompt() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, keepDecodeRungs: 16)
        let toks = Array(0 ..< 24)
        store.store(tokens: toks, rows: rows(T: 24, salt: 0, mtp: true), rowsValidTo: 24, mtpValidTo: 24,
                    rungs: [rung(10, salt: 0), rung(16, salt: 0), rung(20, salt: 0), rung(24, salt: 0)])
        // identical continuation: the final rung
        XCTAssertEqual(store.lookup(toks + [99, 100])?.length, 24)
        // drift at token 18: the rung at 16
        var drift = toks; drift[18] = -1
        XCTAssertEqual(store.lookup(drift + [7])?.length, 16)
        // the prompt IS the sequence: the rung must be strictly short of it
        XCTAssertEqual(store.lookup(toks)?.length, 20)
        // a prompt shorter than the first rung: nothing
        XCTAssertNil(store.lookup(Array(0 ..< 9)))
        // minLength refuses a hit that would not beat the caller's other source
        XCTAssertNil(store.lookup(drift + [7], minLength: 17))
        XCTAssertEqual(store.lookup(drift + [7], minLength: 16)?.length, 16)
        // unrelated prompt: nothing
        XCTAssertNil(store.lookup([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]))
    }

    func testPreferMTPTakesTheCoveredRungOverAHigherOne() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        let toks = Array(0 ..< 24)
        // MTP rows valid to 16 (a batch handover retired the draft state there)
        store.store(tokens: toks, rows: rows(T: 24, salt: 0, mtp: true), rowsValidTo: 24, mtpValidTo: 16,
                    rungs: [rung(10, salt: 0), rung(16, salt: 0), rung(20, salt: 0), rung(24, salt: 0)])
        let plain = store.lookup(toks + [1])
        XCTAssertEqual(plain?.length, 24); XCTAssertEqual(plain?.hasMTP, false)
        let mtp = store.lookup(toks + [1], preferMTP: true)
        XCTAssertEqual(mtp?.length, 16); XCTAssertEqual(mtp?.hasMTP, true)
        // a sequence that never drafted still serves a preferMTP lookup, without MTP
        let store2 = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        store2.store(tokens: toks, rows: rows(T: 24, salt: 0, mtp: false), rowsValidTo: 24, mtpValidTo: 0, rungs: [rung(24, salt: 0)])
        let h = store2.lookup(toks + [1], preferMTP: true)
        XCTAssertEqual(h?.length, 24); XCTAssertEqual(h?.hasMTP, false)
    }

    func testExportedSlicesRowsToTheRungAndOmitsMTPRowsWhenNotCovered() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        let toks = Array(0 ..< 24)
        let full = rows(T: 24, salt: 3, mtp: true)
        store.store(tokens: toks, rows: full, rowsValidTo: 24, mtpValidTo: 16,
                    rungs: [rung(16, salt: 16), rung(24, salt: 24)])
        let h = store.lookup(toks + [1], preferMTP: true)!
        let d = store.exported(h, ratio: ratio)!
        XCTAssertEqual(d["trunk.L0.k"]!.shape, [1, 2, 16, 3])
        XCTAssertEqual(d["trunk.L2.i0"]!.shape, [1, 16, 2])
        XCTAssertEqual(d["trunk.L2.i1"]!.shape, [1, 4, 2])
        XCTAssertEqual(d["mtp.L0.k"]!.shape, [1, 2, 16, 3])
        XCTAssertEqual(d["trunk.L1.a0"]!.item(Float.self), 16 + 16)          // the rung's OWN head
        // the sliced rows are the prefix of the full rows, byte for byte
        let want = full["trunk.L0.k"]![0..., 0..., 0 ..< 16, 0...]
        XCTAssertTrue(MLX.allClose(d["trunk.L0.k"]!, want).item(Bool.self))
        // the uncovered final rung: no MTP rows in the export
        let h24 = store.lookup(toks + [1])!
        let d24 = store.exported(h24, ratio: ratio)!
        XCTAssertNil(d24["mtp.L0.k"])
        XCTAssertEqual(d24["trunk.L0.v"]!.shape, [1, 2, 24, 3])
        XCTAssertEqual(d24["trunk.L1.a0"]!.item(Float.self), 24 + 24)
    }

    func testLadderKeepsFirstLastAndASlidingWindow() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 1, keepDecodeRungs: 2)
        let toks = Array(0 ..< 40)
        store.store(tokens: toks, rows: rows(T: 40, salt: 0, mtp: false), rowsValidTo: 40, mtpValidTo: 0,
                    rungs: (1 ... 10).map { rung($0 * 4, salt: 0) })          // 4, 8, ..., 40
        // P100: the first TWO rungs (the reserve and the prompt-end ones in the server) are kept, then the window, then the last
        XCTAssertEqual(store.entries[0].rungs.map { $0.length }, [4, 8, 32, 36, 40])
        // rungs past the rows' validity are dropped, and a store with none left is refused
        let s2 = HotPrefixStore(capBytes: 1 << 30, rungStep: 1)
        s2.store(tokens: toks, rows: rows(T: 40, salt: 0, mtp: false), rowsValidTo: 20, mtpValidTo: 0,
                 rungs: [rung(8, salt: 0), rung(24, salt: 0), rung(40, salt: 0)])
        XCTAssertEqual(s2.entries[0].rungs.map { $0.length }, [8])
        XCTAssertEqual(s2.lookup(toks + [1])?.length, 8)
        s2.store(tokens: Array(100 ..< 140), rows: rows(T: 40, salt: 0, mtp: false), rowsValidTo: 40, mtpValidTo: 0, rungs: [rung(60, salt: 0)])
        XCTAssertEqual(s2.count, 1)
    }

    func testNextTurnSupersedesItsPredecessorAndLRUEvictsUnderTheCap() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        let a = Array(0 ..< 16)
        store.store(tokens: a, rows: rows(T: 16, salt: 0, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 0)])
        // the next turn of the same conversation replaces it
        let b = a + Array(50 ..< 58)
        store.store(tokens: b, rows: rows(T: 24, salt: 1, mtp: false), rowsValidTo: 24, mtpValidTo: 0, rungs: [rung(16, salt: 1), rung(24, salt: 1)])
        XCTAssertEqual(store.count, 1)
        XCTAssertEqual(store.lookup(b + [1])?.length, 24)
        // a different conversation coexists
        let c = Array(200 ..< 216)
        store.store(tokens: c, rows: rows(T: 16, salt: 2, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 2)])
        XCTAssertEqual(store.count, 2)
        // a cap one byte below two entries evicts the least recently used (b, stored first)
        let probe = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        probe.store(tokens: b, rows: rows(T: 24, salt: 1, mtp: false), rowsValidTo: 24, mtpValidTo: 0, rungs: [rung(24, salt: 1)])
        probe.store(tokens: c, rows: rows(T: 16, salt: 2, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 2)])
        let tight = HotPrefixStore(capBytes: probe.totalBytes - 1, rungStep: 4)
        tight.store(tokens: b, rows: rows(T: 24, salt: 1, mtp: false), rowsValidTo: 24, mtpValidTo: 0, rungs: [rung(24, salt: 1)])
        tight.store(tokens: c, rows: rows(T: 16, salt: 2, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 2)])
        XCTAssertEqual(tight.count, 1)
        XCTAssertEqual(tight.evictions, 1)
        XCTAssertNotNil(tight.lookup(c + [1]))
        XCTAssertNil(tight.lookup(b + [1]))
        // a single entry over the cap is kept: it is the conversation in progress
        let tiny = HotPrefixStore(capBytes: 1, rungStep: 4)
        tiny.store(tokens: c, rows: rows(T: 16, salt: 2, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 2)])
        XCTAssertEqual(tiny.count, 1)
        // disabled store: nothing in, nothing out
        let off = HotPrefixStore(capBytes: 0)
        off.store(tokens: c, rows: rows(T: 16, salt: 2, mtp: false), rowsValidTo: 16, mtpValidTo: 0, rungs: [rung(16, salt: 2)])
        XCTAssertEqual(off.count, 0); XCTAssertNil(off.lookup(c + [1]))
    }

    /// 2026-09-30: eviction affinity must not treat a recently used sibling as dominated. Two prompts share a long
    /// preamble (>= 90% of the first), the first keeps serving hits after the second was stored, and a third store
    /// makes room: the idle sibling goes, the reused one stays. A previous turn nobody resumes is still dominated.
    func testDominatedEvictionSparesASiblingThatKeepsServingHits() {
        let pre = Array(0 ..< 360)
        let a = pre + Array(900 ..< 920)            // 380 tokens, 360 shared with b (94.7%, like the logged 2377/2504)
        let b = pre + Array(700 ..< 720)            // 20 differing tokens: more than supersedeSlack (16), so not a turn
        let c = Array(3000 ..< 3380)
        let probe = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        probe.store(tokens: a, rows: rows(T: 380, salt: 0, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 0)])
        probe.store(tokens: b, rows: rows(T: 380, salt: 1, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 1)])
        let store = HotPrefixStore(capBytes: probe.totalBytes + 1, rungStep: 4)   // room for two entries, not three
        store.store(tokens: a, rows: rows(T: 380, salt: 0, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 0)])
        store.store(tokens: b, rows: rows(T: 380, salt: 1, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 1)])
        // a keeps being used after b was stored
        let hit = store.lookup(a + [1])
        XCTAssertEqual(hit?.length, 380)
        XCTAssertNotNil(store.exported(hit!, ratio: ratio))
        store.store(tokens: c, rows: rows(T: 380, salt: 2, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 2)])
        XCTAssertEqual(store.evictions, 1)
        XCTAssertEqual(store.lookup(a + [1])?.length, 380, "the reused sibling must survive")
        XCTAssertNil(store.lookup(b + [1]).flatMap { $0.length == 380 ? $0 : nil }, "the idle sibling is the victim")
        // without the later hit, a is dominated by b and goes first (P100 unchanged)
        let idle = HotPrefixStore(capBytes: probe.totalBytes + 1, rungStep: 4)
        idle.store(tokens: a, rows: rows(T: 380, salt: 0, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 0)])
        idle.store(tokens: b, rows: rows(T: 380, salt: 1, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 1)])
        idle.store(tokens: c, rows: rows(T: 380, salt: 2, mtp: false), rowsValidTo: 380, mtpValidTo: 0, rungs: [rung(380, salt: 2)])
        XCTAssertEqual(idle.lookup(b + [1])?.length, 380)
        XCTAssertNil(idle.lookup(a + [1]).flatMap { $0.length == 380 ? $0 : nil })
    }

    /// 2026-10-01 RUNG SHARING: siblings with the same token prefix hold ONE rung object for that prefix, it is
    /// charged once, a lookup still resumes from it, and evicting one sibling leaves the other its rung.
    func testSiblingsShareThePrefixRungAndItIsChargedOnce() {
        let pre = Array(0 ..< 360)
        let a = pre + Array(900 ..< 920), b = pre + Array(700 ..< 720)
        func store(_ s: HotPrefixStore, _ t: [Int], _ salt: Float) {
            s.store(tokens: t, rows: rows(T: 380, salt: salt, mtp: false), rowsValidTo: 380, mtpValidTo: 0,
                    rungs: [rung(360, salt: salt), rung(380, salt: salt)])
        }
        let solo = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        store(solo, a, 0)
        let both = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        store(both, a, 0); store(both, b, 1)
        XCTAssertEqual(both.rungDedupHits, 1, "b's prefix rung at 360 is a's")
        let snap = both.snapshot()
        XCTAssertEqual(snap["rung_refs"], 4); XCTAssertEqual(snap["rung_unique"], 3)
        // two entries cost less than twice one: the shared rung is charged once
        XCTAssertLessThan(both.chargedBytes, 2 * solo.chargedBytes)
        // b still resumes at the shared prefix rung and at its own end
        XCTAssertEqual(both.lookup(pre + [5, 6])?.length, 360)
        XCTAssertEqual(both.lookup(b + [1])?.length, 380)
        // a different prefix never shares (hash, then tokens)
        let other = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        store(other, a, 0); store(other, Array(5000 ..< 5380), 2)
        XCTAssertEqual(other.rungDedupHits, 0)
        // evicting a keeps b's shared rung alive and charged
        let tight = HotPrefixStore(capBytes: both.chargedBytes + 1, rungStep: 4, strictBudget: true)
        store(tight, a, 0); store(tight, b, 1)
        store(tight, Array(8000 ..< 8380), 3)
        XCTAssertGreaterThan(tight.evictions, 0)
        if tight.lookup(b + [1]) != nil { XCTAssertEqual(tight.lookup(pre + [5, 6])?.length, 360) }
    }

    /// 2026-10-01 RUNG TRIM: under pressure the store first drops rungs between an entry's reserve (prompt - 1) and
    /// final rungs that never served a hit -- the prompt end and the decode window -- before evicting any entry; the
    /// reserve, the prefix rungs below it, a used rung and the final state stay.
    func testPressureTrimsUnusedDecodeRungsBeforeEvictingAnEntry() {
        let a = Array(0 ..< 400), b = Array(1000 ..< 1400)
        func ladder(_ salt: Float) -> [HotPrefixStore.Rung] { [100, 199, 200, 300, 400].map { rung($0, salt: salt) } }   // prefix, reserve, prompt end, decode, final
        let probe = HotPrefixStore(capBytes: 1 << 30, rungStep: 4, strictBudget: true)
        probe.store(tokens: a, rows: rows(T: 400, salt: 0, mtp: false), rowsValidTo: 400, mtpValidTo: 0, rungs: ladder(0))
        let one = probe.chargedBytes
        // room for two entries minus one rung: storing b must trim instead of evicting a
        let store = HotPrefixStore(capBytes: 2 * one - 1, rungStep: 4, keepDecodeRungs: 8, strictBudget: true)
        store.store(tokens: a, rows: rows(T: 400, salt: 0, mtp: false), rowsValidTo: 400, mtpValidTo: 0, rungs: ladder(0))
        XCTAssertEqual(store.lookup(Array(a[0 ..< 300]) + [7])?.length, 300)              // a's decode rung serves a hit: used
        _ = store.exported(store.lookup(Array(a[0 ..< 300]) + [7])!, ratio: ratio)
        store.store(tokens: b, rows: rows(T: 400, salt: 1, mtp: false), rowsValidTo: 400, mtpValidTo: 0, rungs: ladder(1))
        XCTAssertEqual(store.evictions, 0, "a trim made the room")
        XCTAssertEqual(store.rungsTrimmed, 1)
        XCTAssertEqual(store.count, 2)
        // a lost its unused prompt-end rung (200), kept reserve 199, prefix 100, used 300 and final 400
        XCTAssertEqual(store.lookup(Array(a[0 ..< 250]) + [7])?.length, 199)
        XCTAssertEqual(store.lookup(Array(a[0 ..< 300]) + [7])?.length, 300)
        XCTAssertEqual(store.lookup(a + [7])?.length, 400)
        XCTAssertEqual(store.lookup(Array(b[0 ..< 250]) + [7])?.length, 200)              // b is whole
        // with trim off the same pressure evicts the older entry instead
        let off = HotPrefixStore(capBytes: 2 * one - 1, rungStep: 4, keepDecodeRungs: 8, strictBudget: true)
        off.rungTrim = false
        off.store(tokens: a, rows: rows(T: 400, salt: 0, mtp: false), rowsValidTo: 400, mtpValidTo: 0, rungs: ladder(0))
        off.store(tokens: b, rows: rows(T: 400, salt: 1, mtp: false), rowsValidTo: 400, mtpValidTo: 0, rungs: ladder(1))
        XCTAssertEqual(off.evictions, 1); XCTAssertEqual(off.count, 1)
    }

    /// P106 B41 (H50d): the next turn of a conversation supersedes its previous entry; it must INHERIT the old
    /// entry's rungs inside their common prefix (the early "prefix" rungs another conversation can share), and a
    /// different conversation sharing only that prefix must then still resume there.
    func testSupersedingTurnInheritsPrefixRungs() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 4)
        let turn1 = Array(0 ..< 24)
        store.store(tokens: turn1, rows: rows(T: 24, salt: 1, mtp: false), rowsValidTo: 24, mtpValidTo: 0,
                    rungs: [rung(8, salt: 1), rung(16, salt: 1), rung(24, salt: 1)])
        let turn2 = turn1 + Array(100 ..< 112)           // the same conversation, one turn later
        store.store(tokens: turn2, rows: rows(T: 36, salt: 1, mtp: false), rowsValidTo: 36, mtpValidTo: 0,
                    rungs: [rung(28, salt: 1), rung(36, salt: 1)])
        XCTAssertEqual(store.count, 1, "turn 2 supersedes turn 1")
        XCTAssertEqual(store.entries[0].rungs.map(\.length), [8, 16, 24, 28, 36])
        // another conversation shares only the first 10 tokens: it resumes at the inherited prefix rung
        let other = Array(0 ..< 10) + [900, 901, 902]
        XCTAssertEqual(store.lookup(other)?.length, 8)
        // a rung past the common prefix of the superseded entry is never inherited
        let fork = Array(0 ..< 20) + Array(500 ..< 504)   // diverges at 20 < 24: not superseded (slack 16 < ... ) or partial
        store.store(tokens: fork, rows: rows(T: 24, salt: 2, mtp: false), rowsValidTo: 24, mtpValidTo: 0, rungs: [rung(24, salt: 2)])
        for e in store.entries { XCTAssertTrue(e.rungs.allSatisfy { $0.length <= e.tokens.count }) }
    }

    /// P106 H54 (H53 replay at small scale): a conversation's turn-end rungs, inherited through supersede, are kept as
    /// store-time anchors, so a client edit a few turns back resumes at the last turn end before the edit -- not at the
    /// entry's first rungs. Scale: rungStep 1, anchor band 8, window 32; each turn appends 6 tokens.
    func testEditAFewTurnsBackResumesAtAnInheritedTurnEnd() {
        let store = HotPrefixStore(capBytes: 1 << 30, rungStep: 1, keepDecodeRungs: 2)
        store.anchorStep = 8; store.anchorWindow = 32
        var tokens = Array(0 ..< 40)
        store.store(tokens: tokens, rows: rows(T: 40, salt: 1, mtp: false), rowsValidTo: 40, mtpValidTo: 0,
                    rungs: [rung(8, salt: 1), rung(16, salt: 1), rung(39, salt: 1), rung(40, salt: 1)])
        for turn in 1 ... 8 {                                    // eight more turns of 6 tokens: ends 46, 52, ..., 88
            tokens += Array((1000 * turn) ..< (1000 * turn + 6))
            let n = tokens.count
            store.store(tokens: tokens, rows: rows(T: n, salt: 1, mtp: false), rowsValidTo: n, mtpValidTo: 0,
                        rungs: [rung(n - 1, salt: 1), rung(n, salt: 1)])
        }
        XCTAssertEqual(store.count, 1)
        let kept = store.entries[0].rungs.map(\.length)
        XCTAssertTrue(kept.contains(70) && kept.contains(76), "turn ends inherited as band anchors: \(kept)")
        // the client edits a token 11 positions before the end (inside the turn that ended at 82): resume at 76, not 16
        var edited = tokens; edited[tokens.count - 11] = -7
        XCTAssertEqual(store.lookup(edited)?.length, 76)
        store.anchorStep = 0
    }

    /// P106 H50: the canonical lineage admits wide B1 chunks only when asked (maxChunk), only in whole
    /// multiples of the lattice width, and never a native (batch > 1), tail or misaligned call.
    func testCanonicalTrajectoryWideChunks() {
        var h25 = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024)
        XCTAssertEqual(h25.maxChunk, 1024)
        XCTAssertNil(h25.advance(from: 0, to: 2048, batch: 1, interior: true, mtpNextToken: 7))
        XCTAssertFalse(h25.valid)

        var t = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024, maxChunk: 4096)
        XCTAssertEqual(t.advance(from: 0, to: 4096, batch: 1, interior: true, mtpNextToken: 7)?.width, 1024)
        XCTAssertNotNil(t.advance(from: 4096, to: 5120, batch: 1, interior: true, mtpNextToken: nil))
        XCTAssertNotNil(t.advance(from: 5120, to: 8192, batch: 1, interior: true, mtpNextToken: nil))
        XCTAssertEqual(t.boundary, 8192)
        XCTAssertNil(t.advance(from: 8192, to: 16384, batch: 1, interior: true, mtpNextToken: nil))   // wider than maxChunk
        XCTAssertFalse(t.valid)

        var misaligned = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024, maxChunk: 4096)
        XCTAssertNil(misaligned.advance(from: 0, to: 3000, batch: 1, interior: true, mtpNextToken: nil))
        XCTAssertNil(misaligned.advance(from: 3000, to: 4096, batch: 1, interior: true, mtpNextToken: nil))  // lineage stays broken

        var native = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024, maxChunk: 4096)
        XCTAssertNil(native.advance(from: 0, to: 1024, batch: 2, interior: true, mtpNextToken: nil))
        var tail = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024, maxChunk: 4096)
        XCTAssertNil(tail.advance(from: 0, to: 2048, batch: 1, interior: false, mtpNextToken: nil))
        var resumed = HotPrefixStore.CanonicalPrefillTrajectory(width: 1024, maxChunk: 4096)
        XCTAssertTrue(resumed.resume(length: 3072, origin: .init(width: 1024, mtpNextToken: nil)))
        XCTAssertNotNil(resumed.advance(from: 3072, to: 7168, batch: 1, interior: true, mtpNextToken: nil))
        XCTAssertFalse(resumed.resume(length: 3000, origin: .init(width: 1024, mtpNextToken: nil)))
    }
}
