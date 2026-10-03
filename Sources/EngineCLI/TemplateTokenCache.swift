// P106 B32 (H42): segment id cache for chat-template encoding.
//
// applyChatTemplate = Jinja render (~1.5% of cost) + BPE encode (~97%) of a ~1 MB
// rendered string (H40/H41 MEASURED). The vendored tokenizer splits the rendered
// string on every added token before pre-tokenization/BPE, so the ids of the full
// string equal the concatenation of per-segment ids (H41-SEGMENT-EXACT on all 34
// production-shaped payloads). This cache keys each text segment by SHA256 of its
// UTF-8 bytes so repeated turns and fan-out shared prefixes skip the encode.
//
// Concurrency: one NSCondition's lock covers lookup/insert/bookkeeping only;
// encoding always happens outside it. A call claims every unclaimed segment key in
// a single critical section, encodes its claims, inserts, broadcasts — then emits.
// Waiters hold no in-flight claims while blocked, so waits cannot deadlock; if a
// needed key vanishes (eviction) or never lands, the waiter claims and encodes it
// itself.
import Foundation
import EngineServeSupport
import CryptoKit
import MLXLMCommon
import Tokenizers
import Jinja

// Same attribute allowlist as Tokenizers.applyChatTemplate (specialTokenAttributes).
private let engineSpecialTokenAttributes: Set<String> = [
    "bos_token", "eos_token", "unk_token", "sep_token", "pad_token",
    "cls_token", "mask_token", "additional_special_tokens",
]

// Selects the chat template string exactly as Tokenizers.applyChatTemplate does for a nil
// chatTemplate argument: config chat_template as array (tool_use when tools, else default)
// or string.
func engineSelectChatTemplate(config: [String: Any], tools: [[String: any Sendable]]?) throws -> String {
    guard let raw = config["chat_template"], !(raw is NSNull) else {
        throw EngineError.invalid("tokenizer_config.json has no chat_template")
    }
    if let arr = raw as? [[String: Any]] {
        var dict: [String: String] = [:]
        for item in arr {
            if let name = item["name"] as? String, let t = item["template"] as? String { dict[name] = t }
        }
        if let tools, !tools.isEmpty, let t = dict["tool_use"] { return t }
        if let t = dict["default"] { return t }
    } else if let s = raw as? String {
        return s
    }
    throw EngineError.invalid("no usable chat_template in tokenizer_config.json")
}

// Replicates Tokenizers.applyChatTemplate's context construction and template.render:
// messages, add_generation_prompt (true -- the (messages:tools:additionalContext:) bridge
// the server calls uses it), tools, additionalContext, then the special-token attributes
// from tokenizer_config.json. Verified byte-identical ids on all 34 payloads in H41.
final class ChatTemplateRender: @unchecked Sendable {
    let config: [String: Any]
    private var templates: [String: Jinja.Template] = [:]
    private let templateLock = NSLock()

    init(modelDir: URL) throws {
        let data = try Data(contentsOf: modelDir.appendingPathComponent("tokenizer_config.json"))
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EngineError.invalid("tokenizer_config.json is not an object")
        }
        config = obj
    }

    private func compiled(_ source: String) throws -> Jinja.Template {
        templateLock.lock()
        if let t = templates[source] { templateLock.unlock(); return t }
        templateLock.unlock()
        let t = try Jinja.Template(source)
        templateLock.lock(); templates[source] = t; templateLock.unlock()
        return t
    }

    func render(messages: [Message], tools: [Message]?, extra: [String: any Sendable]) throws -> String {
        let template = try compiled(engineSelectChatTemplate(config: config, tools: tools))
        var context: [String: Jinja.Value] = [
            "messages": .array(try messages.map { try Jinja.Value(any: $0) }),
            "add_generation_prompt": .boolean(true),
        ]
        if let tools { context["tools"] = .array(try tools.map { try Jinja.Value(any: $0) }) }
        for (key, value) in extra { context[key] = try Jinja.Value(any: value) }
        for (key, value) in config where engineSpecialTokenAttributes.contains(key) && !(value is NSNull) {
            if let s = value as? String {
                context[key] = .string(s)
            } else if let d = value as? [String: Any] {
                // addedTokenAsString: dict form serializes as {content: "..."} (+ flags).
                if let c = d["content"] as? String { context[key] = .string(c) }
            } else if let a = value as? [String] {
                context[key] = .array(a.map { .string($0) })
            } else {
                context[key] = try Jinja.Value(any: value)
            }
        }
        return try template.render(context)
    }
}

// Replicates PreTrainedTokenizer's addedTokensRegex construction from tokenizer.json:
// added tokens sorted by descending length, each a capture group wrapped in \s* when the
// tokenizer.json entry carries lstrip/rstrip.
func engineAddedTokenRegex(_ modelDir: URL) throws -> (NSRegularExpression?, Set<String>) {
    let data = try Data(contentsOf: modelDir.appendingPathComponent("tokenizer.json"))
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw EngineError.invalid("tokenizer.json is not an object")
    }
    var added: [(content: String, prefix: Bool, suffix: Bool)] = []
    for t in obj["added_tokens"] as? [[String: Any]] ?? [] {
        guard let content = t["content"] as? String else { continue }
        added.append((content, t["lstrip"] as? Bool ?? false, t["rstrip"] as? Bool ?? false))
    }
    let sorted = added.sorted { $0.content.count > $1.content.count }
    let pattern = sorted.map {
        "\($0.prefix ? #"\s*"# : "")(\(NSRegularExpression.escapedPattern(for: $0.content)))\($0.suffix ? #"\s*"# : "")"
    }.joined(separator: "|")
    return (try? NSRegularExpression(pattern: pattern, options: []), Set(added.map { $0.content }))
}

// Replicates String.split(by: captureRegex) from Tokenizers: unmatched text between matches,
// plus the last capture group's range for each match (with one group, the token itself).
func engineSplitAddedTokens(_ text: String, by regex: NSRegularExpression) -> [String] {
    let selfRange = NSRange(text.startIndex..<text.endIndex, in: text)
    let matches = regex.matches(in: text, options: [], range: selfRange)
    if matches.isEmpty { return [text] }
    var result: [String] = []
    var start = text.startIndex
    for match in matches {
        guard let matchRange = Range(match.range, in: text) else { continue }
        if start < matchRange.lowerBound {
            result.append(String(text[start..<matchRange.lowerBound]))
        }
        start = matchRange.upperBound
        for r in (0..<match.numberOfRanges).reversed() {
            if let sepRange = Range(match.range(at: r), in: text) {
                result.append(String(text[sepRange]))
                break
            }
        }
    }
    if start < text.endIndex { result.append(String(text[start...])) }
    return result
}

final class TemplateTokenCache: @unchecked Sendable {
    struct Stats: Sendable {
        var entries = 0
        var bytes = 0
        var hits = 0            // calls whose every text segment was already cached (no wait)
        var misses = 0          // calls that encoded at least one segment
        var inflightWaits = 0   // calls that blocked on another call's in-flight segment
        var evictions = 0
        var segmentHits = 0     // per-segment granularity, for diagnostics
        var segmentMisses = 0
    }
    struct CallDetail {
        var sections = 0
        var textSegments = 0
        var cachedSegments = 0
        var encodedSegments = 0
        var waited = false
        var encoded = false
        var missedKeys: [String] = []
    }

    private let capacityBytes: Int
    private let tokenizer: any MLXLMCommon.Tokenizer
    private let regex: NSRegularExpression?
    private let addedSet: Set<String>
    private let cond = NSCondition()        // its lock is the single bookkeeping lock
    private var table: [String: [Int32]] = [:]
    private var lru = KeyLRU()              // use order, oldest first; O(1) per touch (EngineServeSupport)
    private var bytesUsed = 0
    private var inFlight: Set<String> = []
    private var stat = Stats()

    init(capacityBytes: Int, tokenizer: any MLXLMCommon.Tokenizer, modelDir: URL) throws {
        self.capacityBytes = capacityBytes
        self.tokenizer = tokenizer
        (regex, addedSet) = try engineAddedTokenRegex(modelDir)
    }

    func stats() -> Stats {
        cond.lock(); defer { cond.unlock() }
        var s = stat; s.entries = table.count; s.bytes = bytesUsed
        return s
    }

    private func segmentKey(_ s: String) -> String {
        SHA256.hash(data: Data(s.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // Segment a rendered string exactly as encode() does: added-token sections get a
    // nil key, text sections get their SHA256 key. Exposed for the --cache-check gate.
    func sectionKeys(rendered: String) -> (sections: [String], keys: [String?]) {
        let sections = regex.map { engineSplitAddedTokens(rendered, by: $0) } ?? [rendered]
        var keys: [String?] = []
        keys.reserveCapacity(sections.count)
        for s in sections { keys.append(addedSet.contains(s) ? nil : segmentKey(s)) }
        return (sections, keys)
    }

    // Caller must hold cond's lock. Inserts ids, clears the in-flight mark, evicts
    // least-recently-used entries over capacity, and wakes waiters.
    private func insertLocked(_ key: String, ids: [Int]) {
        let entry = ids.map { Int32(clamping: $0) }
        table[key] = entry
        lru.touch(key)
        bytesUsed += 4 * entry.count
        inFlight.remove(key)
        while bytesUsed > capacityBytes, let evict = lru.popOldest() {
            if let e = table.removeValue(forKey: evict) {
                bytesUsed -= 4 * e.count
                stat.evictions += 1
            }
        }
        cond.broadcast()
    }

    func encode(rendered: String) -> ([Int], CallDetail) {
        let (sections, keys) = sectionKeys(rendered: rendered)
        var detail = CallDetail(sections: sections.count)
        var keyText: [String: String] = [:]
        for (i, k) in keys.enumerated() {
            if let k { keyText[k] = sections[i] }
        }
        detail.textSegments = keys.reduce(0) { $0 + ($1 == nil ? 0 : 1) }

        // Pass 1: claim every unclaimed key in one critical section.
        cond.lock()
        var mineOrder: [String] = []
        var mine = Set<String>()
        for case let k? in keys where !mine.contains(k) {
            if table[k] == nil && !inFlight.contains(k) {
                inFlight.insert(k); mine.insert(k); mineOrder.append(k)
            }
        }
        if !mine.isEmpty {
            stat.misses += 1; detail.encoded = true
            stat.segmentMisses += mine.count; detail.missedKeys = mineOrder
        }
        cond.unlock()

        // Pass 2: encode claimed segments outside the lock, then insert + wake.
        var encoded: [String: [Int]] = [:]
        for k in mineOrder {
            encoded[k] = tokenizer.encode(text: keyText[k] ?? "", addSpecialTokens: false)
        }
        if !mineOrder.isEmpty {
            cond.lock()
            for k in mineOrder { insertLocked(k, ids: encoded[k] ?? []) }
            cond.unlock()
        }
        detail.encodedSegments = mine.count

        // Pass 3: emit in section order; wait on other callers' claims; take over
        // any key that vanished (evicted) or never landed.
        var out: [Int] = []
        out.reserveCapacity(rendered.utf8.count / 4)
        for (i, s) in sections.enumerated() {
            guard let k = keys[i] else {
                if let id = tokenizer.convertTokenToId(s) {
                    out.append(id)
                } else {
                    out.append(contentsOf: tokenizer.encode(text: s, addSpecialTokens: false))
                }
                continue
            }
            if let e = encoded[k] {
                out.append(contentsOf: e)
                continue
            }
            cond.lock()
            while true {
                if let cached = table[k] {
                    stat.segmentHits += 1
                    detail.cachedSegments += 1
                    lru.touch(k)
                    cond.unlock()
                    out.append(contentsOf: cached.map { Int($0) })
                    break
                }
                if inFlight.contains(k) {
                    if !detail.waited { stat.inflightWaits += 1; detail.waited = true }
                    cond.wait()
                    continue
                }
                // Key vanished without an owner (e.g. evicted): claim and encode it here.
                inFlight.insert(k)
                if !detail.encoded { stat.misses += 1; detail.encoded = true }
                stat.segmentMisses += 1; detail.missedKeys.append(k)
                detail.encodedSegments += 1
                cond.unlock()
                let ids = tokenizer.encode(text: s, addSpecialTokens: false)
                cond.lock()
                insertLocked(k, ids: ids)
                cond.unlock()
                out.append(contentsOf: ids)
                break
            }
        }
        if !detail.encoded && !detail.waited && detail.textSegments > 0 {
            cond.lock(); stat.hits += 1; cond.unlock()
        }
        return (out, detail)
    }
}
