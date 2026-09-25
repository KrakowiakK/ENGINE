// P085/P086 -- the shared prefix store's contract, in tests rather than in a benchmark.
//
// This store is the piece most likely to be wrong in a way no benchmark notices: a wrong HIT does not
// crash, it returns a plausible answer computed from somebody else's state. So the properties that
// make it safe are pinned here -- the key is content-addressed (that is what makes it shareable), a
// reassembled rung is byte-identical to what was written, a missing block is a clean MISS rather than
// a partial restore, and eviction never eats the prefill that is currently running.
import Foundation
import MLX
import XCTest

@testable import Qwen4Exp

final class PrefixStoreTests: XCTestCase {
    var dir: URL!
    var cfg: StateCacheConfig!
    let savedBlockRows = StateCache.blockRows

    override func setUp() {
        super.setUp()
        // The store's contract is about bytes and files, not about kernels, so the suite runs on the
        // CPU device: a test bundle does not sit next to the engine's mlx.metallib and has no
        // business needing a GPU to prove that a block round-trips.
        Device.setDefault(device: Device.cpu)
        dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("prefixstore-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        StateCache.blockRows = 4                       // tiny blocks so the arithmetic is visible
        cfg = StateCacheConfig(dir: dir, identity: "ident-A", step: 8, writeExact: false, maxBytes: 0)
    }
    override func tearDown() {
        StateCache.blockRows = savedBlockRows
        try? FileManager.default.removeItem(at: dir)
        super.tearDown()
    }

    // rows [0, T) of a state that looks like the real one: KV on axis 2, indexer on axis 1, and a
    // fixed-size array standing in for the GDN recurrent state
    private func state(rows T: Int, salt: Float = 0) -> [String: MLXArray] {
        let kv = MLXArray((0 ..< (2 * T * 3)).map { Float($0) + salt }).reshaped([1, 2, T, 3])
        let idx = MLXArray((0 ..< (T * 2)).map { Float($0) * 2 + salt }).reshaped([1, T, 2])
        return ["trunk.L0.k": kv, "trunk.L0.v": kv + MLXArray(Float(0.5)),
                "trunk.L0.i0": idx,
                "trunk.L1.a1": MLXArray((0 ..< 6).map { Float($0) + salt }).reshaped([1, 2, 3]),
                "T": MLXArray(Int32(T))]
    }
    private func assertSame(_ a: [String: MLXArray], _ b: [String: MLXArray],
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Set(a.keys), Set(b.keys), "array names differ", file: file, line: line)
        for (k, v) in a {
            guard let w = b[k] else { continue }
            XCTAssertEqual(v.shape, w.shape, "shape of \(k)", file: file, line: line)
            let d = (v.asType(.float32) - w.asType(.float32)).abs().max().item(Float.self)
            XCTAssertEqual(d, 0, "values of \(k)", file: file, line: line)
        }
    }
    private func files(_ prefix: String) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasPrefix(prefix) }
    }

    // MARK: the sharing property

    func testKeyIsContentAddressedSoSessionsShare() {
        let a = Array(0 ..< 32), b = Array(0 ..< 32)
        XCTAssertEqual(StateCache.key(identity: "i", tokens: a, count: 16),
                       StateCache.key(identity: "i", tokens: b, count: 16),
                       "two sessions with the same prefix must land on the same key -- that IS the sharing")
        var c = a; c[20] = 999                          // differs only AFTER the prefix
        XCTAssertEqual(StateCache.key(identity: "i", tokens: a, count: 16),
                       StateCache.key(identity: "i", tokens: c, count: 16),
                       "a difference past the prefix must not change the key")
        var d = a; d[3] = 999                           // differs INSIDE the prefix
        XCTAssertNotEqual(StateCache.key(identity: "i", tokens: a, count: 16),
                          StateCache.key(identity: "i", tokens: d, count: 16))
        XCTAssertNotEqual(StateCache.key(identity: "i", tokens: a, count: 16),
                          StateCache.key(identity: "j", tokens: a, count: 16),
                          "a different model identity must never collide")
        XCTAssertNotEqual(StateCache.key(identity: "i", tokens: a, count: 16),
                          StateCache.key(identity: "i", tokens: a, count: 8))
    }

    // MARK: round trip

    func testRungReassemblesExactly() {
        let toks = Array(0 ..< 64)
        let M = 16                                      // 4 whole blocks of 4
        let d = state(rows: M)
        StateCache.write(cfg, toks, M, d)
        let u = StateCache.url(cfg, toks, M)
        XCTAssertTrue(FileManager.default.fileExists(atPath: u.path))
        XCTAssertEqual(files("blk_").count, 4, "16 rows in blocks of 4")
        guard let back = StateCache.load(cfg, toks, M, u) else { return XCTFail("reassembly returned nil") }
        assertSame(d, back)
    }

    func testTailRowsShorterThanABlockSurvive() {
        let toks = Array(0 ..< 64)
        let M = 18                                      // 4 whole blocks + 2 rows
        let d = state(rows: M)
        StateCache.write(cfg, toks, M, d)
        XCTAssertEqual(files("blk_").count, 4)
        guard let back = StateCache.load(cfg, toks, M, StateCache.url(cfg, toks, M)) else {
            return XCTFail("reassembly returned nil")
        }
        assertSame(d, back)
    }

    func testBlocksAreDeduplicatedAcrossRungs() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 8, state(rows: 8))
        XCTAssertEqual(files("blk_").count, 2)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        XCTAssertEqual(files("blk_").count, 4,
                       "the second rung must add only its NEW blocks -- the first two are the same bytes")
    }

    func testSecondSessionWithTheSamePrefixWritesNoNewBlocks() {
        let a = Array(0 ..< 64)
        var b = a; b[40] = 12345                        // same first 16 tokens, different later
        StateCache.write(cfg, a, 16, state(rows: 16))
        let after = files("blk_").count
        StateCache.write(cfg, b, 16, state(rows: 16))
        XCTAssertEqual(files("blk_").count, after, "a second session on the same prefix must add nothing")
    }

    // MARK: failure is a clean miss, never a partial restore

    func testMissingBlockIsACleanMiss() {
        let toks = Array(0 ..< 64)
        let M = 16
        StateCache.write(cfg, toks, M, state(rows: M))
        let victim = dir.appendingPathComponent(files("blk_").sorted()[0])
        try? FileManager.default.removeItem(at: victim)
        XCTAssertNil(StateCache.load(cfg, toks, M, StateCache.url(cfg, toks, M)),
                     "a rung missing a block must return nil, not a short state")
    }

    func testLookupTakesTheLongestStoredPrefix() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 8, state(rows: 8))
        StateCache.write(cfg, toks, 16, state(rows: 16))
        guard let (_, M) = StateCache.lookup(cfg, Array(toks.prefix(20))) else { return XCTFail("no hit") }
        XCTAssertEqual(M, 16, "the deeper rung answers with less prefill left to do")
    }

    func testLookupOnAnUnrelatedPromptMisses() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        XCTAssertNil(StateCache.lookup(cfg, Array((1000 ..< 1064))), "no shared prefix, no hit")
    }

    // MARK: eviction

    private func age(_ name: String, seconds: Double) {
        let u = dir.appendingPathComponent(name)
        try? FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-seconds)], ofItemAtPath: u.path)
    }
    private func totalBytes() -> Double {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).reduce(0.0) {
            $0 + Double(((try? FileManager.default.attributesOfItem(
                atPath: dir.appendingPathComponent($1).path))?[.size] as? Int) ?? 0)
        }
    }

    func testEvictionDropsTheLeastRecentlyUsedFirst() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        let blocks = files("blk_").sorted()
        XCTAssertEqual(blocks.count, 4)
        for (i, b) in blocks.enumerated() { age(b, seconds: 10_000 - Double(i) * 1000) }   // [0] oldest
        for h in files("st_") { age(h, seconds: 500) }
        let before = totalBytes()
        StateCache.evictIfOver(cfg, cap: before * 0.5, graceSeconds: 1)
        XCTAssertLessThan(totalBytes(), before, "something must go when the store is over cap")
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent(blocks[0]).path),
                       "the least recently used entry goes first")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent(blocks[3]).path),
                      "the most recently used entry stays")
    }

    func testEvictionIsANoOpUnderTheCap() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        let before = totalBytes(), names = Set(files(""))
        StateCache.evictIfOver(cfg, cap: before * 10, graceSeconds: 0)
        XCTAssertEqual(totalBytes(), before)
        XCTAssertEqual(Set(files("")), names)
    }

    func testGraceWindowProtectsThePrefillInFlight() {
        // every entry was written seconds ago, which is exactly the state a long prefill is in when
        // its own later rungs trigger eviction. Nothing may be dropped, or the prefill eats itself.
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        let before = totalBytes()
        StateCache.evictIfOver(cfg, cap: before * 0.1, graceSeconds: 3600)
        XCTAssertEqual(totalBytes(), before, "a hot working set must not be evicted even when over cap")
    }

    func testDisabledCapNeverEvicts() {
        let toks = Array(0 ..< 64)
        StateCache.write(cfg, toks, 16, state(rows: 16))
        let before = totalBytes()
        StateCache.evictIfOver(cfg, cap: 0, graceSeconds: 0)
        XCTAssertEqual(totalBytes(), before)
    }

    func testReadTouchesEntriesSoLRUSeesUse() throws {
        let toks = Array(0 ..< 64)
        let M = 16
        StateCache.write(cfg, toks, M, state(rows: M))
        for f in files("") { age(f, seconds: 5000) }
        _ = StateCache.load(cfg, toks, M, StateCache.url(cfg, toks, M))
        for f in files("") {
            let a = try FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(f).path)
            let used = (a[.modificationDate] as? Date) ?? .distantPast
            XCTAssertLessThan(Date().timeIntervalSince(used), 60,
                              "\(f) was read, so LRU must see it as recently used")
        }
    }
}
