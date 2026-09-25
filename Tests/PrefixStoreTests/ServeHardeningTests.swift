import Foundation
import Darwin
import XCTest
@testable import EngineServeSupport
@testable import Qwen4Exp

final class ServeHardeningTests: XCTestCase {
    private func pair() throws -> [Int32] {
        var fds: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0 else { throw HTTPTransport.Failure.disconnected }
        fds.forEach { HTTPTransport.configure($0) }
        return fds
    }
    func testHTTPRejectsAmbiguousNegativeOversizedAndTruncatedLengths() throws {
        for headers in ["Content-Length: -1", "Content-Length: 99999999999999999999",
                        "Content-Length: 9", "Content-Length: 1\r\nContent-Length: 1",
                        "Transfer-Encoding: chunked", "Expect: 100-continue"] {
            let f = try pair(); defer { f.forEach { close($0) } }
            HTTPTransport.writeAll(f[1], Data("POST / HTTP/1.1\r\n\(headers)\r\n\r\nx".utf8))
            shutdown(f[1], SHUT_WR)
            XCTAssertThrowsError(try HTTPTransport.readRequest(f[0], maxBody: 8, timeout: 0.1), headers)
        }
        let f = try pair(); defer { f.forEach { close($0) } }
        HTTPTransport.writeAll(f[1], Data("POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nx".utf8))
        shutdown(f[1], SHUT_WR)
        XCTAssertThrowsError(try HTTPTransport.readRequest(f[0], timeout: 0.1))
    }
    func testHTTPReadsExactBodyAndDetectsClosedPeerAndWriteFailure() throws {
        let f = try pair(); defer { close(f[0]) }
        HTTPTransport.writeAll(f[1], Data("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}".utf8))
        let request = try HTTPTransport.readRequest(f[0])
        XCTAssertEqual(request.path, "/v1/chat/completions")
        XCTAssertEqual(request.body, Data("{}".utf8))
        XCTAssertFalse(HTTPTransport.peerClosed(f[0]))
        close(f[1])
        // EOF alone does not distinguish close() from shutdown(SHUT_WR).
        XCTAssertFalse(HTTPTransport.heartbeat(f[0]))
        XCTAssertFalse(HTTPTransport.writeAll(f[0], Data("x".utf8)))
    }
    func testCompleteRequestWithHalfCloseStillReceivesResponse() throws {
        let f = try pair(); defer { f.forEach { close($0) } }
        HTTPTransport.writeAll(f[1], Data("POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}".utf8))
        shutdown(f[1], SHUT_WR)
        XCTAssertEqual(try HTTPTransport.readRequest(f[0]).body, Data("{}".utf8))
        XCTAssertFalse(HTTPTransport.peerClosed(f[0]))
        XCTAssertTrue(HTTPTransport.heartbeat(f[0]))
        XCTAssertTrue(HTTPTransport.writeAll(f[0], Data("ok".utf8)))
        var bytes = [UInt8](repeating: 0, count: 3)
        XCTAssertEqual(read(f[1], &bytes, bytes.count), 3)
        XCTAssertEqual(bytes, Array("\nok".utf8))
    }
    func testHTTPDeadlineAndHeaderBound() throws {
        let f = try pair(); defer { f.forEach { close($0) } }
        XCTAssertThrowsError(try HTTPTransport.readRequest(f[0], timeout: 0.02))
        HTTPTransport.writeAll(f[1], Data(repeating: 65, count: 64))
        XCTAssertThrowsError(try HTTPTransport.readRequest(f[0], maxHeaders: 32, timeout: 0.1))
    }
    func testContextAdmissionIncludesOutputLookaheadAndPaddedPool() {
        let b = AdmissionBudget(maxContext: 1000, capacityBytes: 20_000, bytesPerToken: 1, fixedBytesPerSequence: 0)
        XCTAssertTrue(b.valid(prompt: 990, output: 6, choices: 8, lookahead: 4))
        XCTAssertFalse(b.valid(prompt: 990, output: 7, choices: 8, lookahead: 4))
        XCTAssertFalse(b.valid(prompt: 1, output: Int.max, choices: 1, lookahead: 4))
        XCTAssertFalse(b.valid(prompt: 1, output: 1, choices: 9, lookahead: 4))
        let a = b.reserve(length: 1000)!, c = b.reserve(length: 1)!
        XCTAssertNil(b.reserve(length: 1), "each row is padded to the longest admitted row")
        b.release(a); b.release(c)
        XCTAssertEqual(b.active, 0)
        XCTAssertNotNil(b.reserve(length: 1000))
    }
    func testE9AdmissionIncludesRawAndPooledIndexerHistory() {
        // E9 config: twelve trunk full-attention layers plus one MTP layer;
        // bf16 K/V with two 256-wide heads and 128-wide indexer keys, pooled 4:1.
        let bytes = AdmissionBudget.perTokenBytes(attentionLayers: 13, kvHeads: 2, headDim: 256,
                                                  indexerHeadDim: 128, indexerCompressRatio: 4)
        XCTAssertEqual(bytes, 30_784)
        let capacity: @Sendable (Int) -> Double = {
            AdmissionBudget.historyCapacityBytes(length: $0, attentionLayers: 13, kvHeads: 2, headDim: 256,
                indexerHeadDim: 128, indexerCompressRatio: 4, capacityGrowthLimit: 262_144)
        }
        let fullB8Bound = 197_126_455_296.0
        XCTAssertEqual(capacity(262_144), 8_069_840_896)
        XCTAssertEqual(8 * (3 * (262_144.0 + 512) * bytes + 384e6), fullB8Bound)
        let exact = AdmissionBudget(maxContext: 262_144, capacityBytes: fullB8Bound, bytesPerToken: bytes,
                                    historyCapacityBytes: capacity)
        let below = AdmissionBudget(maxContext: 262_144, capacityBytes: fullB8Bound - 1, bytesPerToken: bytes,
                                    historyCapacityBytes: capacity)
        for _ in 0..<7 {
            XCTAssertNotNil(exact.reserve(length: 262_144))
            XCTAssertNotNil(below.reserve(length: 262_144))
        }
        XCTAssertNotNil(exact.reserve(length: 262_144))
        XCTAssertNil(below.reserve(length: 262_144), "the eighth full context must include pooled indexer storage")
        XCTAssertEqual(exact.active, 8)
        XCTAssertEqual(below.active, 7)
    }
    func testCapacityEnvelopePreservesQuantaAndSoftLimit() {
        let full = AdmissionBudget.historyCapacityLengths(length: 262_144, maxForwardRows: 1024)
        XCTAssertEqual(full.kv, 294_912); XCTAssertEqual(full.raw, 294_912); XCTAssertEqual(full.pooled, 73_727)
        for n in [1, 3, 4, 255, 256, 257, 1023, 1024, 1025, 262_143, 262_144] {
            let uncapped = AdmissionBudget.historyCapacityLengths(length: n, maxForwardRows: 1024)
            let capped = AdmissionBudget.historyCapacityLengths(length: n, maxForwardRows: 1024, capacityGrowthLimit: 262_144)
            XCTAssertGreaterThanOrEqual(capped.kv, Double(n)); XCTAssertLessThanOrEqual(capped.kv, 262_144)
            XCTAssertGreaterThanOrEqual(capped.raw, Double(n)); XCTAssertLessThanOrEqual(capped.raw, 262_144)
            XCTAssertGreaterThanOrEqual(capped.pooled, Double(n / 4)); XCTAssertLessThanOrEqual(capped.pooled, 65_536)
            XCTAssertEqual(capped.kv, min(262_144, uncapped.kv))
            XCTAssertEqual(capped.raw, min(262_144, uncapped.raw))
        }
        let beyond = AdmissionBudget.historyCapacityLengths(length: 262_148, capacityGrowthLimit: 262_144)
        XCTAssertEqual(beyond.kv, 262_148); XCTAssertEqual(beyond.raw, 262_148); XCTAssertEqual(beyond.pooled, 65_537)
        // Existing context validation remains a hard serving gate despite soft allocation limits.
        let b = AdmissionBudget(maxContext: 262_144, capacityBytes: 1e15, bytesPerToken: 30_784)
        XCTAssertFalse(b.valid(prompt: 258_045, output: 4096, choices: 8, lookahead: 4))
        XCTAssertNil(b.reserve(length: 262_145))
    }
    func testE9StableFixedStateGeometry() throws {
        let a = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(Qwen4ExpCacheCapacity.fixedStateBytes(a, mtp: true), 115_662_856)
        XCTAssertLessThan(3 * Qwen4ExpCacheCapacity.fixedStateBytes(a, mtp: true), 384e6)
    }

    func testPerTokenBytesSupportsStandardKVAndScalarWidth() {
        XCTAssertEqual(AdmissionBudget.perTokenBytes(attentionLayers: 32, kvHeads: 8, headDim: 128), 131_072)
        XCTAssertEqual(AdmissionBudget.perTokenBytes(attentionLayers: 32, kvHeads: 8, headDim: 128,
                                                     elementBytes: 4), 262_144)
        XCTAssertEqual(AdmissionBudget.perTokenBytes(attentionLayers: 0, kvHeads: 1, headDim: 1), 0)
    }
    func testPerTokenBytesDoesNotTruncatePooledSlopeOrOverflowIntegerProducts() {
        let fractional = AdmissionBudget.perTokenBytes(attentionLayers: 1, kvHeads: 1, headDim: 1,
                                                        indexerHeadDim: 1, indexerCompressRatio: 3)
        XCTAssertEqual(fractional, 6 + 2.0 / 3, accuracy: 1e-12)
        let huge = AdmissionBudget.perTokenBytes(attentionLayers: 1, kvHeads: Int.max, headDim: Int.max,
                                                 elementBytes: 8)
        XCTAssertTrue(huge.isFinite)
        XCTAssertGreaterThan(huge, Double(Int.max))
    }
    /// H60: the plain arm is re-sampled every 256 rounds while the round wins by >= 1.5x (H53 OMP: B4 K3 86.3 ms at
    /// acceptance ~0.76 vs plain B4 46.2 ms), every 32 otherwise, and the cadence follows the running acceptance at once.
    func testDraftPolicyExploresRarelyOnlyWhileTheRoundWinsWide() {
        var p = BatchDraftPolicy()
        p.noteRound(k: 3, ms: 86.3, acceptance: 0.76)
        p.notePlain(ms: 46.2)
        XCTAssertEqual(p.exploreInterval(k: 3), BatchDraftPolicy.exploreEveryWide)
        var plain = 0
        for _ in 0 ..< 1024 {
            let k = p.depth(maxK: 3, mode: "auto")
            if k == 0 { plain += 1; p.notePlain(ms: 46.2) } else { p.noteRound(k: k, ms: 86.3, acceptance: 0.76) }
        }
        XCTAssertEqual(plain, 1024 / (BatchDraftPolicy.exploreEveryWide + 1))
        // acceptance falls: (1 + 3a) * 46.2 < 1.5 * 86.3 for a < 0.6 -> back to 32 on the next decision
        for _ in 0 ..< 20 { p.noteRound(k: 3, ms: 86.3, acceptance: 0.45) }
        XCTAssertEqual(p.exploreInterval(k: 3), BatchDraftPolicy.exploreEvery)
        // no plain sample yet, or no round cost: the old cadence
        XCTAssertEqual(BatchDraftPolicy().exploreInterval(k: 3), BatchDraftPolicy.exploreEvery)
        XCTAssertEqual(p.exploreInterval(k: 2), BatchDraftPolicy.exploreEvery)
        XCTAssertEqual(p.depth(maxK: 3, mode: "always"), 3)
    }
    /// H60: depth > 3 only with --batch-mtp > 3, entered at an implied per-position acceptance >= 0.84 and left below 0.80;
    /// at maxK 3 the policy's choices are B44's.
    func testDraftPolicyGoesDeepOnlyAboveBreakEvenWithHysteresis() {
        XCTAssertEqual(BatchDraftPolicy.impliedP(fraction: (0.87 + 0.87 * 0.87 + 0.87 * 0.87 * 0.87) / 3, K: 3), 0.87, accuracy: 1e-6)
        XCTAssertEqual(BatchDraftPolicy.impliedP(fraction: 0.7, K: 1), 0.7, accuracy: 1e-12)
        var p = BatchDraftPolicy()
        p.noteRound(k: 3, ms: 86.3, acceptance: 0.758)          // H53: p ~ 0.87 per position
        p.notePlain(ms: 46.2)
        XCTAssertTrue(p.deep)
        XCTAssertEqual(p.depth(maxK: 4, mode: "auto"), 4)
        XCTAssertEqual(p.depth(maxK: 3, mode: "auto"), 3)       // maxK 3: never deeper
        // at K4, p 0.82 (between the thresholds): stays deep
        let f82 = (0.82 + pow(0.82, 2) + pow(0.82, 3) + pow(0.82, 4)) / 4
        for _ in 0 ..< 30 { p.noteRound(k: 4, ms: 98, acceptance: f82) }
        XCTAssertTrue(p.deep)
        // p 0.75: leaves
        let f75 = (0.75 + pow(0.75, 2) + pow(0.75, 3) + pow(0.75, 4)) / 4
        for _ in 0 ..< 30 { p.noteRound(k: 4, ms: 98, acceptance: f75) }
        XCTAssertFalse(p.deep)
        XCTAssertLessThan(p.depth(maxK: 4, mode: "auto"), 4)
        // maxK 3 never records a depth > 3, so its acceptance EMA mixes K 1..3 exactly as B44's did
        var q = BatchDraftPolicy()
        for (k, a) in [(3, 0.7), (2, 0.5), (3, 0.65), (1, 0.4)] { q.noteRound(k: k, ms: 80, acceptance: a) }
        var e = 0.7; for a in [0.5, 0.65, 0.4] { e = 0.8 * e + 0.2 * a }
        XCTAssertEqual(q.acceptance, e, accuracy: 1e-12)
    }
    /// H60 review: in deep mode the depth is maxK for every p in the hysteresis band -- the review's case (maxK 5,
    /// p 0.83: the K5 fraction 0.592 < 0.6) alternated 5, 2, 5, 2; leaving deep lands on the K3-equivalent rule, not K2.
    func testDraftPolicyDeepModeDoesNotThrashAgainstShallowThresholds() {
        var p = BatchDraftPolicy()
        p.noteRound(k: 3, ms: 86.3, acceptance: BatchDraftPolicy.fraction(p: 0.87, K: 3))
        p.notePlain(ms: 46.2)
        XCTAssertTrue(p.deep)
        var ks: [Int] = []
        for _ in 0 ..< 40 {
            let k = p.depth(maxK: 5, mode: "auto")
            ks.append(k)
            if k == 0 { p.notePlain(ms: 46.2) } else { p.noteRound(k: k, ms: 86.3 + 12.1 * Double(k - 3), acceptance: BatchDraftPolicy.fraction(p: 0.83, K: k)) }
        }
        XCTAssertFalse(ks.contains(2), "deep mode must not fall to K2 inside the band: \(ks)")
        XCTAssertEqual(Set(ks.filter { $0 > 0 }), [5])
        for _ in 0 ..< 30 { p.noteRound(k: 5, ms: 110, acceptance: BatchDraftPolicy.fraction(p: 0.78, K: 5)) }
        XCTAssertFalse(p.deep)
        p.notePlain(ms: 46.2)                                   // (70 rounds since the last plain sample would force one)
        XCTAssertEqual(p.depth(maxK: 5, mode: "auto"), 3, "p 0.78 -> K3 fraction 0.62 -> the K3 rule")
        XCTAssertEqual(BatchDraftPolicy.fraction(p: 0.5, K: 2), 0.375, accuracy: 1e-12)
    }
    /// H60 window 2: one weak round after a depth switch must not open a plain-step streak (the replay slot with 68 plain
    /// B5 steps). With per-depth EMAs a single bad K4 sample moves the K4 estimate by 20 %, not to the sample.
    func testDraftPolicyOneWeakRoundDoesNotOpenAPlainStreak() {
        var p = BatchDraftPolicy()
        for _ in 0 ..< 10 { p.noteRound(k: 3, ms: 93.0, acceptance: BatchDraftPolicy.fraction(p: 0.87, K: 3)) }
        p.notePlain(ms: 55.0)
        XCTAssertTrue(p.deep)
        XCTAssertEqual(p.depth(maxK: 4, mode: "auto"), 4)
        p.noteRound(k: 4, ms: 104.0, acceptance: 0.15)          // one very weak K4 round
        var plains = 0
        for _ in 0 ..< 20 {
            let k = p.depth(maxK: 4, mode: "auto")
            if k == 0 { plains += 1; p.notePlain(ms: 55.0) } else { p.noteRound(k: k, ms: 104.0, acceptance: BatchDraftPolicy.fraction(p: 0.87, K: k)) }
        }
        XCTAssertEqual(plains, 0)
        // and a depth not sampled for long is judged by the current p, not by its stale EMA
        var q = BatchDraftPolicy()
        q.noteRound(k: 4, ms: 104.0, acceptance: 0.10)          // an old, bad K4 estimate
        for _ in 0 ..< 30 { q.noteRound(k: 3, ms: 93.0, acceptance: BatchDraftPolicy.fraction(p: 0.88, K: 3)) }
        q.notePlain(ms: 55.0)
        XCTAssertTrue(q.deep)
        XCTAssertEqual(q.depth(maxK: 4, mode: "auto"), 4)
    }
    /// H60 gate mode: `cycle:3,2,0` repeats its depths per decision (0 = plain), capped by maxK; malformed -> not a cycle.
    func testDraftPolicyCycleModeIsDeterministic() {
        XCTAssertEqual(BatchDraftPolicy.cycle("cycle:3,2,0,3"), [3, 2, 0, 3])
        XCTAssertNil(BatchDraftPolicy.cycle("auto")); XCTAssertNil(BatchDraftPolicy.cycle("cycle:")); XCTAssertNil(BatchDraftPolicy.cycle("cycle:3,-1"))
        var p = BatchDraftPolicy()
        var ks: [Int] = []
        for _ in 0 ..< 8 {
            let k = p.depth(maxK: 3, mode: "cycle:3,2,0,4")
            ks.append(k)
            if k == 0 { p.notePlain(ms: 40) } else { p.noteRound(k: k, ms: 80, acceptance: 0.7) }
        }
        XCTAssertEqual(ks, [3, 2, 0, 3, 3, 2, 0, 3])
        XCTAssertEqual(p.depth(maxK: 3, mode: "never"), 0)
    }
    func testDraftPolicyRetriesAfterLosingSampleAndCanRecover() {
        var p = BatchDraftPolicy()
        p.noteRound(k: 3, ms: 100, acceptance: 0.8)
        p.notePlain(ms: 10)
        XCTAssertEqual(p.depth(maxK: 3, mode: "auto"), 0)
        var probes = 0
        for _ in 0..<1000 {
            let k = p.depth(maxK: 3, mode: "auto")
            if k == 0 { p.notePlain(ms: 10) }
            else { probes += 1; p.noteRound(k: k, ms: 10, acceptance: 1) }
        }
        XCTAssertGreaterThan(probes, 100)
        XCTAssertEqual(p.depth(maxK: 3, mode: "never"), 0)
        XCTAssertEqual(p.depth(maxK: 3, mode: "always"), 3)
    }
    func testCacheIdentityIncludesSameSizedWeightsConfigurationAndProgram() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        let weights = dir.appendingPathComponent("model.safetensors")
        try Data([1, 2, 3]).write(to: weights)
        let a = try StateCache.identity(dir: dir, witness: "same", program: "build-a")
        try Data([1, 4, 3]).write(to: weights)
        let b = try StateCache.identity(dir: dir, witness: "same", program: "build-a")
        XCTAssertNotEqual(a, b)
        XCTAssertNotEqual(b, try StateCache.identity(dir: dir, witness: "same", program: "build-b"))
        try Data("{\"x\":1}".utf8).write(to: dir.appendingPathComponent("config.json"))
        XCTAssertNotEqual(b, try StateCache.identity(dir: dir, witness: "same", program: "build-a"))
        try FileManager.default.removeItem(at: weights)
        XCTAssertThrowsError(try StateCache.identity(dir: dir, witness: "same", program: "build-a"))
    }
    func testFileDigestPreservesAllChunksAndPartialFinalRead() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        try Data(repeating: 0x5a, count: (9 << 20) + 17).write(to: file)
        XCTAssertEqual(try StateCache.fileDigest(file), "40de94802bc49423852af0e17ace444e2129d4200ba9effda23df1efdb98b5b1")
    }
    func testIndexedKeysEqualDirectKeysAtEveryRungAndBlock() {
        let tokens = Array(0..<8197)
        let config = StateCacheConfig(dir: URL(fileURLWithPath: "/tmp"), identity: "test", step: 512, writeExact: false)
        let indexed = config.indexed(for: tokens)
        for n in Array(stride(from: 512, through: tokens.count, by: 512)) + [tokens.count, 17] {
            XCTAssertEqual(StateCache.url(indexed, tokens, n), StateCache.url(config, tokens, n))
            XCTAssertEqual(StateCache.blockURL(indexed, tokens, n), StateCache.blockURL(config, tokens, n))
        }
    }
    func testSplitDelimitersStopsUnicodeAndToolMarkersNeverLeak() {
        for marker in ["</think>", "<tool_call>", "🖥️STOP"] {
            var stream = DelimiterStream([marker])
            var out = "", rest = ""
            for scalar in ("hello " + marker + "world").unicodeScalars {
                let d = stream.append(String(scalar)); out += d.text; rest += d.remainder
            }
            XCTAssertEqual(out, "hello "); XCTAssertEqual(rest, "world")
            XCTAssertEqual(stream.matched, 0)
        }
        var s = DelimiterStream(["STOP"])
        XCTAssertEqual(s.append("ok ST").text, "ok ")
        XCTAssertEqual(s.finish(), "ST")
    }
    func testXMLStreamHandlesSplitTagsAndLargeParameterOnce() {
        var parser = XMLToolStream()
        let value = String(repeating: "zażółć 🖥️\n", count: 10_000)
        let xml = "<function=write><parameter=path>\na.swift\n</parameter><parameter=text>\n" + value + "\n</parameter></function></tool_call>"
        var events: [XMLToolStream.Event] = []
        for char in xml { events += parser.append(String(char)) }
        XCTAssertEqual(events, [.open("write"), .parameter("path", "a.swift"), .parameter("text", value), .close])
    }
    func testDetokenizerDeltaIsAppendOnlyAcrossUTF8AndCommits() {
        let text = String(repeating: "Zażółć 🖥️👩🏽‍💻\r\n", count: 100)
        let bytes = Array(text.utf8)
        var d = IncrementalDetokenizer(window: 8) { String(decoding: $0.map(UInt8.init), as: UTF8.self) }
        var emitted = ""
        for byte in bytes { emitted += d.appendDelta(Int(byte)) }
        XCTAssertEqual(emitted, text)
        XCTAssertEqual(d.text, text)
    }
    func testDetokenizerFlushPreservesEveryTruncatedUTF8SuffixAndLiteralReplacement() {
        let bytes = Array("Zażółć 🖥️ 👩🏽‍💻 \u{FFFD}\u{FFFD}".utf8)
        for n in 0...bytes.count {
            var d = IncrementalDetokenizer(window: 2) { String(decoding: $0.map(UInt8.init), as: UTF8.self) }
            var wire = ""
            for byte in bytes.prefix(n) { wire += d.appendDelta(Int(byte)) }
            wire += d.finishDelta()
            XCTAssertEqual(wire, String(decoding: bytes.prefix(n), as: UTF8.self), "prefix \(n)")
            XCTAssertEqual(d.finishDelta(), "", "flush must be idempotent")
        }
    }
}
