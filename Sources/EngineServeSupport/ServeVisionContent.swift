import Foundation

/// 2026-10-09 (T-0053): image input on the serve path -- the parts of it that are plain data, so they are tested.
///
/// This checkpoint's chat template (`render_content` in chat_template.jinja) renders an image item as
/// `<|vision_start|><|image_pad|><|vision_end|>` and a text item as its text, concatenated in item order. Flattening the
/// content parts to that same string keeps the serve path's contract with the template (one string per message, which is
/// also what the template-token cache keys on) and puts each image's placeholder exactly where the template puts it.
/// Before this, the parser kept only the text parts: an image was dropped without a word and the model answered as if
/// none had been sent (C-0163).
public enum ServeVisionContent {
    public static let imagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case video
        case imageInSystemMessage
        case missingImageSource(part: Int)
        case unsupportedImageSource(String)
        case badBase64
        case tooLarge(bytes: Int, limit: Int)
        case fetchFailed(String)
        case fetchDisabled

        public var description: String {
            switch self {
            case .video: return "video input is not supported by this server (send frames as images)"
            case .imageInSystemMessage: return "a system message cannot contain images (the chat template refuses it)"
            case .missingImageSource(let i): return "content part \(i) is an image without an image_url / url / base64 source"
            case .unsupportedImageSource(let s): return "unsupported image source \"\(s.prefix(32))\": send a data: URL (base64) or an http(s) URL"
            case .badBase64: return "the image data URL is not valid base64"
            case .tooLarge(let b, let l): return "the image is \(b) bytes, over the \(l)-byte limit"
            case .fetchFailed(let why): return "could not fetch the image: \(why)"
            case .fetchDisabled: return "this server does not fetch image URLs (ENGINE_VISION_FETCH=0); send the image as a data: URL"
            }
        }
    }

    /// The source of an image part, or nil when the part is not an image. The template treats a part as an image when it
    /// has an `image` or `image_url` key or `type == "image"`; the OpenAI forms (`image_url` as an object or a string,
    /// `input_image`) and the Anthropic base64 `source` are read here as well.
    static func imageSource(_ p: [String: Any], index: Int) throws -> String? {
        let type = (p["type"] as? String) ?? ""
        let isImage = ["image", "image_url", "input_image"].contains(type) || p["image"] != nil || p["image_url"] != nil
        guard isImage else { return nil }
        if let o = p["image_url"] as? [String: Any], let u = o["url"] as? String { return u }
        if let u = p["image_url"] as? String { return u }
        if let u = p["image"] as? String { return u }
        if let u = p["url"] as? String { return u }
        if let s = p["source"] as? [String: Any] {
            if (s["type"] as? String) == "base64", let d = s["data"] as? String {
                return "data:\((s["media_type"] as? String) ?? "image/png");base64,\(d)"
            }
            if let u = s["url"] as? String { return u }
        }
        throw Failure.missingImageSource(part: index)
    }

    /// content parts -> the string the template renders for them, plus the image sources in order. Parts that are
    /// neither text nor an image are dropped, as before; a text-only array gives exactly the old joined text.
    public static func flatten(_ parts: [[String: Any]], role: String) throws -> (text: String, images: [String]) {
        var text = ""
        var images: [String] = []
        for (i, p) in parts.enumerated() {
            let type = (p["type"] as? String) ?? ""
            if type == "video" || type == "video_url" || p["video"] != nil || p["video_url"] != nil { throw Failure.video }
            if let src = try imageSource(p, index: i) {
                if role == "system" { throw Failure.imageInSystemMessage }
                images.append(src)
                text += imagePlaceholder
            } else if let t = p["text"] as? String {
                text += t
            }
        }
        return (text, images)
    }

    /// `data:[<mediatype>];base64,<data>` -> bytes.
    public static func decodeDataURL(_ s: String, limit: Int) throws -> Data {
        guard s.hasPrefix("data:"), let comma = s.firstIndex(of: ",") else { throw Failure.unsupportedImageSource(s) }
        let meta = s[s.index(s.startIndex, offsetBy: 5) ..< comma]
        guard meta.split(separator: ";").contains("base64") else { throw Failure.badBase64 }
        guard let d = Data(base64Encoded: String(s[s.index(after: comma)...]), options: .ignoreUnknownCharacters), !d.isEmpty else {
            throw Failure.badBase64
        }
        guard d.count <= limit else { throw Failure.tooLarge(bytes: d.count, limit: limit) }
        return d
    }

    /// An image source -> bytes: a data URL, or an http(s) URL fetched with a timeout when `allowFetch`.
    public static func load(_ source: String, limit: Int, allowFetch: Bool, timeout: TimeInterval = 30) throws -> Data {
        if source.hasPrefix("data:") { return try decodeDataURL(source, limit: limit) }
        guard let url = URL(string: source), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw Failure.unsupportedImageSource(source)
        }
        guard allowFetch else { throw Failure.fetchDisabled }
        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.setValue("ENGINE/1 (image fetch)", forHTTPHeaderField: "User-Agent")
        let done = DispatchSemaphore(value: 0)
        let box = FetchBox()
        let task = URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { done.signal() }
            if let err { box.result = .failure(Failure.fetchFailed(err.localizedDescription)); return }
            if let h = resp as? HTTPURLResponse, !(200 ..< 300).contains(h.statusCode) {
                box.result = .failure(Failure.fetchFailed("HTTP \(h.statusCode)")); return
            }
            guard let data, !data.isEmpty else { box.result = .failure(Failure.fetchFailed("empty body")); return }
            box.result = data.count > limit ? .failure(Failure.tooLarge(bytes: data.count, limit: limit)) : .success(data)
        }
        task.resume()
        if done.wait(timeout: .now() + timeout + 5) == .timedOut {
            task.cancel()
            throw Failure.fetchFailed("timed out after \(Int(timeout)) s")
        }
        return try box.result.get()
    }

    private final class FetchBox: @unchecked Sendable {
        var result: Result<Data, Error> = .failure(Failure.fetchFailed("no response"))
    }

    // MARK: - the token layout

    /// One image's span in the expanded prompt: the first `<|image_pad|>` row and the POST-MERGE grid (t, h, w).
    public struct Span: Equatable, Sendable {
        public let start: Int, t: Int, h: Int, w: Int
        public init(start: Int, t: Int, h: Int, w: Int) { self.start = start; self.t = t; self.h = h; self.w = w }
        public var tokens: Int { t * h * w }
    }

    public enum LayoutFailure: Error, Equatable, CustomStringConvertible {
        case placeholderCount(found: Int, images: Int)
        public var description: String {
            switch self {
            case .placeholderCount(let f, let n):
                return "the rendered prompt has \(f) image placeholder token(s) for \(n) image(s); remove literal <|image_pad|> text from the messages"
            }
        }
    }

    /// Expands each `<|image_pad|>` of the rendered prompt into its image's token count (t * h * w of the post-merge grid),
    /// in order, and returns the expanded ids with each image's span. The count of placeholders must equal the count of
    /// images: a literal `<|image_pad|>` typed into a message would otherwise take an image's place.
    public static func expand(ids: [Int], padId: Int, grids: [(t: Int, h: Int, w: Int)]) throws -> (ids: [Int], spans: [Span]) {
        let pads = ids.indices.filter { ids[$0] == padId }
        guard pads.count == grids.count else { throw LayoutFailure.placeholderCount(found: pads.count, images: grids.count) }
        var out: [Int] = []
        out.reserveCapacity(ids.count + grids.reduce(0) { $0 + $1.t * $1.h * $1.w })
        var spans: [Span] = []
        var k = 0
        for id in ids {
            if id == padId {
                let g = grids[k]; k += 1
                spans.append(Span(start: out.count, t: g.t, h: g.h, w: g.w))
                out.append(contentsOf: repeatElement(padId, count: g.t * g.h * g.w))
            } else {
                out.append(id)
            }
        }
        return (out, spans)
    }

    /// The prefix store's key for row `row` of an image whose content hashes to `hash`: in [2^60, 2^61), above every
    /// token id, so a stored state is found again only for the same image at the same place -- never for another image,
    /// whose pad rows carry the same token id.
    public static func keyId(hash: UInt64, row: Int) -> Int {
        // splitmix64 of (hash, row): every bit of the hash reaches the 60 kept bits
        var z = hash &+ UInt64(truncatingIfNeeded: row) &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Int(truncatingIfNeeded: (z >> 4) | (1 << 60))
    }

    /// `ids` with every image span's rows replaced by the image's keys (the model input stays `ids`).
    public static func keyed(ids: [Int], spans: [Span], hashes: [UInt64]) -> [Int] {
        precondition(spans.count == hashes.count, "one hash per image span")
        var out = ids
        for (k, sp) in spans.enumerated() {
            for j in 0 ..< sp.tokens { out[sp.start + j] = keyId(hash: hashes[k], row: j) }
        }
        return out
    }

    /// The first 8 bytes of a SHA-256 digest as a big-endian integer.
    public static func hash64<D: Sequence>(digest: D) -> UInt64 where D.Element == UInt8 {
        digest.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// The mRoPE position the first token after the prompt takes, minus the prompt length. HF's `get_rope_index` gives text
    /// one position per token and a span max(t, h, w) positions, so a prompt of `total` tokens ends at
    /// total + sum(max(t, h, w) - t*h*w) and every later token's position is its index plus this (negative) delta.
    public static func ropeDelta(spans: [Span]) -> Int {
        spans.reduce(0) { $0 + max($1.t, max($1.h, $1.w)) - $1.tokens }
    }
}
