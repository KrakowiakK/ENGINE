import XCTest
import Foundation
import Tokenizers
@testable import EngineServeSupport

/// P093 -- the incremental detokeniser must return, after every token, exactly what a whole-output
/// decode returns. First against a synthetic byte-level decoder that splits UTF-8 sequences across
/// tokens on purpose, then against the checkpoint's own tokenizer on random ids (skipped when the
/// weights directory is not on this machine).
final class IncrementalDetokenizerTests: XCTestCase {
    /// A byte-level "tokenizer": token id i is byte i (0..<256); ids >= 256 are special tokens
    /// rendered literally, splitting the byte stream like the real ByteLevel decoder does.
    private static func syntheticDecode(_ ids: [Int]) -> String {
        var out = ""
        var run: [UInt8] = []
        func flush() { if !run.isEmpty { out += String(decoding: run, as: UTF8.self); run = [] } }
        for i in ids {
            if i >= 256 { flush(); out += "<|s\(i - 256)|>" } else { run.append(UInt8(i)) }
        }
        flush()
        return out
    }

    /// Deterministic PRNG (SplitMix64): a failure names its seed and reproduces. H53 (2026-09-23): this test drew its
    /// random tail from SystemRandomNumberGenerator, so it failed once in a gate run and could not be replayed.
    struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// The fixed prefix (emoji, CJK, combining marks, a stray continuation byte, specials) + `n` seeded random bytes.
    static func syntheticStream(seed: UInt64, n: Int = 2000) -> [Int] {
        var bytes: [Int] = []
        let sample = "Hi 👋🏽 there — 日本語のテキスト, ñ, e\u{0301}, 🖥️ done.\n"
        bytes += Array(sample.utf8).map(Int.init)
        bytes += [0x80, 0xE2]                          // invalid lone continuation, then a truncated lead
        bytes += Array("tail 🎉".utf8).map(Int.init)
        bytes += [256, 257]                            // specials
        bytes += Array("after".utf8).map(Int.init)
        var rng = SplitMix64(state: seed)
        for _ in 0 ..< n { bytes.append(Int(UInt8.random(in: 0 ... 255, using: &rng))) }
        return bytes
    }

    /// First divergence of the incremental text from a whole decode, or nil.
    static func firstDivergence(_ bytes: [Int], window: Int) -> (at: Int, got: String, want: String)? {
        var inc = IncrementalDetokenizer(window: window, decode: syntheticDecode)
        var whole: [Int] = []
        for b in bytes {
            whole.append(b)
            let got = inc.append(b)
            let want = syntheticDecode(whole)
            if got != want { return (whole.count, got, want) }
        }
        return nil
    }

    /// D51 (H53): a truncated UTF-8 sequence right after a Prepend scalar (U+0600 ARABIC NUMBER SIGN) decodes to
    /// U+0600 U+FFFD -- ONE grapheme, so a Character-level `hasSuffix("\u{FFFD}")` missed it and the commit point
    /// advanced into the middle of the sequence. Every window must keep the stream equal to a whole decode.
    func testPrependScalarBeforeTruncatedSequence() {
        var bytes = Array("pad ".utf8).map(Int.init)
        for _ in 0 ..< 12 { bytes += [0xD8, 0x80, 0xE2, 0x82, 0xAC] + Array(" x".utf8).map(Int.init) }   // U+0600 then the euro sign, split below
        for window in [1, 2, 3, 5, 8] {
            XCTAssertNil(Self.firstDivergence(bytes, window: window), "window \(window)")
        }
        var inc = IncrementalDetokenizer(window: 1, decode: Self.syntheticDecode)
        var streamed = ""
        for b in bytes { streamed += inc.appendDelta(b) }
        streamed += inc.finishDelta()
        XCTAssertEqual(streamed, Self.syntheticDecode(bytes), "the append-only deltas must add up to the whole decode")
    }

    func testSyntheticStreamMatchesWholeDecode() {
        for seed in UInt64(0) ..< 64 {
            let bytes = Self.syntheticStream(seed: seed)
            for window in [1, 2, 5, 32] {
                // One autorelease pool per window: String(decoding:) repairs invalid UTF-8 through errors the
                // standard library throws and catches internally, and XCTest's Swift will-throw observer records an
                // (autoreleased) call stack for every one of them. Without a pool they pile up until the test ends
                // -- a debug run passed 216 GB of RSS before it was stopped. The engine process has no such observer.
                let diverged: Bool = autoreleasepool {
                    if let d = Self.firstDivergence(bytes, window: window) {
                        XCTFail("seed \(seed) window \(window) diverged at token \(d.at): tail got \(Array(d.got.unicodeScalars.suffix(6)).map { String($0.value, radix: 16) }) want \(Array(d.want.unicodeScalars.suffix(6)).map { String($0.value, radix: 16) })")
                        return true
                    }
                    var inc = IncrementalDetokenizer(window: window, decode: Self.syntheticDecode)
                    for b in bytes { _ = inc.append(b) }
                    XCTAssertGreaterThan(inc.committed, 0, "the commit point never advanced")
                    // the commit point defers while the cut lands inside a multi-byte sequence; on random bytes
                    // every lead byte at the cut defers it once more, so the lag is bounded loosely, not at 2w+1
                    XCTAssertLessThanOrEqual(bytes.count - inc.committed, 2 * window + 64)
                    return false
                }
                if diverged { return }
            }
        }
    }

    func testRealTokenizerRandomIdsMatchWholeDecode() async throws {
        // the checkpoint's own tokenizer: ENGINE_TEST_MODEL, else the repository's weights/e9
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let dir = ProcessInfo.processInfo.environment["ENGINE_TEST_MODEL"].map { URL(fileURLWithPath: $0) }
            ?? repoRoot.appendingPathComponent("weights/e9")
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("checkpoint tokenizer not on this machine")
        }
        let tok = try await AutoTokenizer.from(modelFolder: dir)
        let decode: ([Int]) -> String = { tok.decode(tokens: $0, skipSpecialTokens: false) }
        // random ids split multi-byte sequences constantly; mix in the specials the server relies on
        let specials = ["<think>", "</think>", "<|im_end|>", "<tool_call>", "</tool_call>"].compactMap { tok.convertTokenToId($0) }
        XCTAssertFalse(specials.isEmpty)
        var rng = SystemRandomNumberGenerator()
        var ids: [Int] = tok.encode(text: "Hello 👋🏽 世界 🖥️\n</think>\n\nanswer", addSpecialTokens: false)
        for i in 0 ..< 3000 {
            ids.append(i % 97 == 0 ? specials[i % specials.count] : Int.random(in: 0 ..< 151_000, using: &rng))
        }
        var inc = IncrementalDetokenizer(window: 32, decode: decode)
        var whole: [Int] = []
        for t in ids {
            whole.append(t)
            let got = inc.append(t)
            if whole.count % 7 == 0 || whole.count < 200 {          // the whole decode is the slow part
                let want = decode(whole)
                XCTAssertEqual(got, want, "diverged at token \(whole.count)")
                if got != want { return }
            }
        }
        XCTAssertEqual(inc.text, decode(whole))
        XCTAssertLessThanOrEqual(whole.count - inc.committed, 65)
    }
}
