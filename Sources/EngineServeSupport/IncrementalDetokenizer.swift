import Foundation

/// P093 -- detokenise in O(1) per token instead of O(n).
///
/// The server used to call `tokenizer.decode(out)` on the WHOLE output every step. That is O(n) per
/// token, so a response costs O(n^2) host time, and the host time is on the critical path of a solo
/// request (the model thread idles while the connection thread detokenises). MEASURED on the live
/// server: 108.5 tok/s over the first 1000 tokens of a response falling linearly to 83.5 tok/s by
/// token 13000, at a context the sweep shows costs nothing (82 tok/s at 4k and at 16k). A 16.6k-token
/// agent turn ran at 42 tok/s where its neighbours ran at 70.
///
/// Why it was whole-text: a byte-level BPE token can end mid-UTF-8-sequence, and `decode` of a
/// prefix that ends there carries a trailing U+FFFD that vanishes once the sequence completes. A
/// naive per-token decode would emit that U+FFFD as text. This type keeps the guarantee and drops
/// the cost:
///
///   - `committedText` is `decode(tokens[0 ..< committed])`, and it is only ever advanced to a cut
///     whose decode has NO trailing U+FFFD. The ByteLevel decoder joins the bytes of a run of tokens
///     and decodes them as one UTF-8 stream, so a prefix that ends on a code-point boundary decodes
///     to exactly the same text alone as it does inside the whole -- `decode(A + B) == decode(A) +
///     decode(B)` whenever `decode(A)` ends cleanly. Added (special) tokens split the byte stream on
///     both sides identically, so they are boundaries too.
///   - The tail past the commit point is re-decoded every step, and it is at most `2 * window`
///     tokens long, so the per-token cost is bounded by the window rather than by the output.
///
/// The test suite proves the equality against the checkpoint's own tokenizer on random token ids
/// (which split multi-byte sequences constantly) and on emoji / CJK / special-token streams.
public struct IncrementalDetokenizer {
    public let decode: ([Int]) -> String
    public let window: Int
    public private(set) var tokens: [Int] = []
    public private(set) var committed = 0
    public private(set) var committedText = ""

    /// - window: the tail is never committed closer than this many tokens to the end, and the
    ///   commit is attempted once the tail is twice this long, so a decode touches at most
    ///   `2 * window` tokens per step.
    public init(window: Int = 32, decode: @escaping ([Int]) -> String) {
        self.window = max(1, window)
        self.decode = decode
    }

    /// Append one token and return the full text so far -- byte-for-byte what `decode(tokens)`
    /// would return.
    private var emittedTailBytes = 0

    public mutating func append(_ token: Int) -> String {
        _ = appendDelta(token)
        return text
    }

    /// Append-only UTF-8 delta. Work is bounded by the uncommitted token window;
    /// callers need not copy or scan the full response to stream one new token.
    public mutating func appendDelta(_ token: Int) -> String {
        tokens.append(token)
        var tail = decode(Array(tokens[committed...]))
        // D51: compare SCALARS, not Characters -- after a Prepend scalar (U+0600...) a trailing U+FFFD joins the same
        // grapheme, so `hasSuffix("\u{FFFD}")` is false and `removeLast()` would drop the Prepend scalar too.
        while tail.unicodeScalars.last == "\u{FFFD}" { tail.unicodeScalars.removeLast() }
        let bytes = Array(tail.utf8)
        let delta = String(decoding: bytes.dropFirst(min(emittedTailBytes, bytes.count)), as: UTF8.self)
        emittedTailBytes = bytes.count
        if tokens.count - committed > 2 * window {
            let cut = tokens.count - window
            let head = decode(Array(tokens[committed..<cut]))
            if head.unicodeScalars.last != "\u{FFFD}", head.utf8.count <= emittedTailBytes {
                committedText.append(head)
                committed = cut
                emittedTailBytes -= head.utf8.count
            }
        }
        return delta
    }

    /// At EOS/length, a remaining incomplete UTF-8 sequence is real replacement text.
    /// Flush it once; an actual trailing U+FFFD must also survive the wire response.
    public mutating func finishDelta() -> String {
        let bytes = Array(decode(Array(tokens[committed...])).utf8)
        let delta = String(decoding: bytes.dropFirst(min(emittedTailBytes, bytes.count)), as: UTF8.self)
        emittedTailBytes = bytes.count
        return delta
    }

    /// The full text without appending (same value the last `append` returned).
    public var text: String { committedText + decode(Array(tokens[committed...])) }
}
