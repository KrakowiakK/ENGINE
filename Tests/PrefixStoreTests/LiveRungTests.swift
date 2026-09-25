import XCTest
import MLX
@testable import Qwen4Exp

/// H9: compare online retention against the OLD final-store selector on a full history.
/// Tiny CPU arrays carry different bytes at duplicate lengths; no model is loaded.
final class LiveRungTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    private func rung(_ length: Int, identity: Int, shared: MLXArray? = nil) -> HotPrefixStore.Rung {
        var head = ["trunk.L1.a1": MLXArray([Int32(identity), Int32(length)]).reshaped([1, 2])]
        if let shared { head["trunk.L1.a2"] = shared }
        if identity % 3 == 0 {
            head["trunk.L1.a4"] = MLXArray([Int32(identity), -1, -2]).reshaped([1, 3])
            head["trunk.L1.a5"] = MLXArray([Int32(identity + 1)]).reshaped([1, 1])
        }
        return HotPrefixStore.Rung(length: length, head: head,
                                   bytes: head.values.reduce(0) { $0 + $1.nbytes })
    }

    /// Deliberately retain the old head + middle + final formulation as the oracle.
    private func oldFinal(_ all: [HotPrefixStore.Rung], keep: Int, validTo: Int) -> [HotPrefixStore.Rung] {
        var ladder = all.filter { $0.length <= validTo }.sorted { $0.length < $1.length }
        if ladder.count > keep + 3 {
            let head = Array(ladder.prefix(2)), last = ladder[ladder.count - 1]
            let middle = ladder[2 ..< (ladder.count - 1)].suffix(keep)
            ladder = head + Array(middle) + [last]
        }
        return ladder
    }

    private func assertRungsEqual(_ actual: [HotPrefixStore.Rung], _ expected: [HotPrefixStore.Rung],
                                  file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.map(\.length), expected.map(\.length), file: file, line: line)
        XCTAssertEqual(actual.map(\.bytes), expected.map(\.bytes), file: file, line: line)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(Set(a.head.keys), Set(b.head.keys), file: file, line: line)
            for key in b.head.keys {
                XCTAssertEqual(a.head[key]?.asData().data, b.head[key]?.asData().data, file: file, line: line)
            }
        }
    }

    func testEveryMonotonePrefixMatchesUnprunedOracleIncludingDuplicateFinals() throws {
        let shared = MLXArray([Int32(17), 23]).reshaped([1, 2])
        for keep in [0, 1, 2, 16] {
            for prefillCount in 0...2 {
                let store = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: keep)
                var full: [HotPrefixStore.Rung] = [], live: [HotPrefixStore.Rung] = []
                let prefill = (0..<prefillCount).map { rung(8 + $0, identity: 100 + $0, shared: shared) }
                // Three captures at each length have distinct bytes and must not be deduplicated.
                let captures = prefill + (0..<64).map { rung(10 + $0 / 3, identity: $0, shared: shared) }
                for capture in captures {
                    full.append(capture)
                    live = try XCTUnwrap(store.retainedLiveRungs(live + [capture]))
                    XCTAssertLessThanOrEqual(live.count, store.retainedRungLimit)
                    assertRungsEqual(live, oldFinal(full, keep: keep, validTo: capture.length))
                    // Omitted final, equal-length new final, and higher-length new final.
                    for final in [nil, rung(capture.length, identity: 1000), rung(capture.length + 1, identity: 1001)] as [HotPrefixStore.Rung?] {
                        let addition = final.map { [$0] } ?? []
                        let cutoff = final?.length ?? capture.length
                        let selected = try XCTUnwrap(store.retainedLiveRungs(live + addition, validTo: cutoff))
                        assertRungsEqual(selected, oldFinal(full + addition, keep: keep, validTo: cutoff))
                    }
                    // Gross logical bytes count shared arrays per rung, not unique allocations.
                    XCTAssertEqual(live.reduce(0) { $0 + $1.bytes },
                        live.flatMap { $0.head.values }.reduce(0) { $0 + $1.nbytes })
                }
            }
        }
    }

    func testFinalLookupAndExportMatchOracleAtEveryCommonPrefix() throws {
        for keep in [0, 1, 2, 16] {
            let tokens = Array(0..<40)
            let rows = ["trunk.L0.k": MLXArray((0..<40).map(Int32.init)).reshaped([1, 1, 40, 1])]
            let full = (0..<64).map { rung(1 + $0 / 2, identity: $0) }
            let online = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: keep)
            var live: [HotPrefixStore.Rung] = []
            for capture in full { live = try XCTUnwrap(online.retainedLiveRungs(live + [capture])) }
            for final in [nil, rung(32, identity: 1000), rung(40, identity: 1001)] as [HotPrefixStore.Rung?] {
                let addition = final.map { [$0] } ?? []
                let oracle = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: keep)
                let candidate = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: keep)
                oracle.store(tokens: tokens, rows: rows, rowsValidTo: 40, mtpValidTo: 0,
                             rungs: oldFinal(full + addition, keep: keep, validTo: 40))
                candidate.store(tokens: tokens, rows: rows, rowsValidTo: 40, mtpValidTo: 0, rungs: live + addition)
                assertRungsEqual(candidate.entries[0].rungs, oracle.entries[0].rungs)
                for common in 0...tokens.count {
                    let query = Array(tokens.prefix(common)) + [9999]
                    let a = candidate.lookup(query), b = oracle.lookup(query)
                    XCTAssertEqual(a?.length, b?.length)
                    if let a, let b {
                        let actual = try XCTUnwrap(candidate.exported(a, ratio: 4))
                        let expected = try XCTUnwrap(oracle.exported(b, ratio: 4))
                        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
                        for key in expected.keys { XCTAssertEqual(actual[key]?.asData().data, expected[key]?.asData().data) }
                    }
                }
            }
        }
    }

    /// H54 (H53 finding): at store time, the highest rung of every anchor band inside the tail window survives retention.
    func testStoreTimeAnchorsKeepOneRungPerBandInTheTailWindow() {
        let store = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: 2)
        store.anchorStep = 8; store.anchorWindow = 32
        let full = (1...100).map { rung($0, identity: $0) }
        XCTAssertEqual(store.retainedRungs(full, anchors: true).map(\.length), [1, 2, 71, 79, 87, 95, 97, 98, 99, 100])
        XCTAssertEqual(store.retainedRungs(full).map(\.length), [1, 2, 98, 99, 100], "live / anchor-free selection = legacy")
        store.anchorStep = 0
        XCTAssertEqual(store.retainedRungs(full, anchors: true).map(\.length), [1, 2, 98, 99, 100], "anchors off = legacy")
    }

    /// D52: anchors never enlarge a LIVE ladder, so the per-row live-rung reserve the admission budget charges is unchanged.
    func testAnchorsDoNotChangeTheLiveLimitOrLivePruning() throws {
        let store = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: 2)
        let legacyLimit = store.retainedRungLimit
        store.anchorStep = 8; store.anchorWindow = 32
        XCTAssertEqual(store.retainedRungLimit, legacyLimit)
        var live: [HotPrefixStore.Rung] = []
        for i in 1...100 {
            live = try XCTUnwrap(store.retainedLiveRungs(live + [rung(i, identity: i)]))
            XCTAssertLessThanOrEqual(live.count, legacyLimit)
        }
        XCTAssertEqual(live.map(\.length), [1, 2, 98, 99, 100])
    }

    func testBackwardsCaptureAndCutoffAreExplicitlyRefused() throws {
        let store = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: 2)
        let full = (1...10).map { rung($0, identity: $0) }
        let live = try XCTUnwrap(store.retainedLiveRungs(full))
        XCTAssertEqual(live.map(\.length), [1, 2, 8, 9, 10])
        XCTAssertNil(store.retainedLiveRungs(live, validTo: 7))
        XCTAssertNil(store.retainedLiveRungs(live + [rung(9, identity: 100)]))
        XCTAssertNotNil(store.retainedLiveRungs(live + [rung(10, identity: 100)], validTo: 10))
        // General final selection retains legacy cutoff behavior on UNPRUNED input.
        assertRungsEqual(store.retainedRungs(full, validTo: 7), oldFinal(full, keep: 2, validTo: 7))
        XCTAssertEqual(store.retainedRungs(full, validTo: 7).map(\.length), [1, 2, 5, 6, 7])
        XCTAssertEqual(store.retainedRungs(live, validTo: 7).map(\.length), [1, 2],
                       "the caller must not bypass the live cutoff guard")
        XCTAssertEqual(try XCTUnwrap(store.retainedLiveRungs([])).count, 0)
    }

    func testUnorderedFinalInputHasStableDuplicateOrderingAndCutoff() {
        let store = HotPrefixStore(capBytes: 1 << 20, keepDecodeRungs: 1)
        let lengths = [4, 1, 2, 4, 2, 6, 4, 5]
        let full = lengths.enumerated().map { rung($0.element, identity: $0.offset) }
        assertRungsEqual(store.retainedRungs(full, validTo: 4), oldFinal(full, keep: 1, validTo: 4))
        XCTAssertNil(store.retainedLiveRungs(full))
    }

    func testSixtyFourLiveSnapshotsKeepFiveIncludingOptionalRollbackArrays() throws {
        let store = HotPrefixStore(capBytes: 1 << 20)
        let shared = MLXArray([Int32(11)])
        let full = (0..<64).map { rung(100 + $0 * 512, identity: $0, shared: shared) }
        var live: [HotPrefixStore.Rung] = []
        for capture in full { live = try XCTUnwrap(store.retainedLiveRungs(live + [capture])) }
        XCTAssertEqual(live.count, 5)
        assertRungsEqual(live, [full[0], full[1], full[61], full[62], full[63]])
        XCTAssertNotNil(live.last?.head["trunk.L1.a4"])
        XCTAssertNotNil(live.last?.head["trunk.L1.a5"])
        XCTAssertEqual(live.reduce(0) { $0 + $1.bytes },
                       [0, 1, 61, 62, 63].reduce(0) { $0 + full[$1].bytes })
    }
}
