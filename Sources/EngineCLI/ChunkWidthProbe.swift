// P106 H50 diagnostic, not a serving path: is the state at a 1024-aligned prefill boundary independent
// of how [0, L) was chunked? Canonical prefix reuse (H24/H25) certifies a rung only when every chunk
// before it was exactly 1024 wide, because a different chunking was assumed to round differently. This
// measures it: each arm prefills the same L tokens through the serving B1 calls (forwardHidden, then the
// MTP head over the shifted tokens, one eval per chunk) and hashes the logical state of every trunk layer
// and of the head. Arms can switch `Qwen4ExpPrefillWidth.canonicalRows` in-process, so a knob A/B shares
// one process, one model load and one corpus.
//
//   engine batchprobe --model DIR --prompt FILE --prompt-tokens L --chunk-width-probe 1
//       --arms "1024/4096/4096@1024/4096,1024,3072,2048@1024/1024"
// An arm is a comma list of widths used cyclically; "@R" sets canonicalRows = R for that arm (default 0).
// Every width must be a multiple of 1024 and L a multiple of 1024, so every arm crosses the same
// 1024-aligned boundaries and ends exactly at L. Repeat an arm to witness run-to-run determinism.
import Foundation
import CryptoKit
import MLX
import MLXLMCommon
import Qwen4Exp

func h50ChunkWidthProbe(_ o: Options, model qm: Qwen4ExpModel, corpus: [Int], corpusSHA256: String) throws {
    let L = try o.int("--prompt-tokens", 16384)
    let armSpec = o.string("--arms", "1024/4096/4096@1024/1024")
    guard L % 1024 == 0, corpus.count >= L + 1, let mtp = qm.mtp else {
        throw EngineError.invalid("H50 probe needs L % 1024 == 0, L + 1 corpus tokens and the MTP head")
    }
    struct Arm { let label: String; let widths: [Int]; let rows: Int }
    let arms: [Arm] = try armSpec.split(separator: "/").map { raw in
        let parts = raw.split(separator: "@")
        let widths = parts[0].split(separator: ",").compactMap { Int($0) }
        let rows = parts.count > 1 ? Int(parts[1]) ?? -1 : 0
        guard !widths.isEmpty, widths.allSatisfy({ $0 > 0 && $0 % 1024 == 0 }), rows >= 0 else {
            throw EngineError.invalid("H50 arm '\(raw)': widths must be positive multiples of 1024")
        }
        return Arm(label: String(raw), widths: widths, rows: rows)
    }
    let ratio = qm.configuration.text.indexerCompressRatio
    let savedRows = Qwen4ExpPrefillWidth.canonicalRows
    Qwen4ExpCacheCapacity.configureGrowthLimit(262144)
    defer { Qwen4ExpCacheCapacity.configureGrowthLimit(nil); Qwen4ExpPrefillWidth.canonicalRows = savedRows }
    let xs = MLXArray(corpus.prefix(L + 1).map { Int32($0) })[.newAxis]

    func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    func arrayDigest(_ x: MLXArray) -> String { digest(x.asData(access: .copy).data) }
    // Per layer, one digest per named part: the logical extent only, never capacity padding.
    func layerDigests(_ caches: [KVCache]) throws -> [[String: String]] {
        try caches.enumerated().map { layer, cache in
            if let list = cache as? CacheList {
                guard let kv = list[0] as? KVCacheSimple, let idx = list[1] as? ArraysCache, kv.offset == L else {
                    throw EngineError.invalid("H50 INSTRUMENTFAIL: attention state at layer \(layer)")
                }
                // An attention layer without an indexer state says so rather than hashing nothing.
                return ["kind": "attn", "keys": arrayDigest(kv.state[0]), "values": arrayDigest(kv.state[1]),
                        "indexer_raw": idx[0].map { arrayDigest($0[0..., 0 ..< L, 0...]) } ?? "absent",
                        "indexer_pooled": idx[1].map { arrayDigest($0[0..., 0 ..< (L / ratio), 0...]) } ?? "absent"]
            }
            guard let a = cache as? ArraysCache else { throw EngineError.invalid("H50 INSTRUMENTFAIL: cache type at layer \(layer)") }
            var parts: [String: String] = ["kind": "fixed"]
            for slot in 0 ..< 4 { if let x = a[slot] { parts["slot\(slot)"] = arrayDigest(x) } }
            guard parts.count > 1 else { throw EngineError.invalid("H50 INSTRUMENTFAIL: empty fixed state at layer \(layer)") }
            return parts
        }
    }
    func joined(_ layers: [[String: String]]) -> String {
        digest(Data(layers.map { $0.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",") }
            .joined(separator: "\n").utf8))
    }

    var results: [[String: Any]] = []
    var reference: (trunk: [[String: String]], head: [String: String])? = nil
    for (armIndex, arm) in arms.enumerated() {
        Qwen4ExpPrefillWidth.canonicalRows = arm.rows
        let blockedBefore = Qwen4ExpPrefillWidth.blockedCalls
        var cache: [KVCache]? = qm.newCache(parameters: nil)
        var mtpCache: KVCache? = mtp.newCache()
        var chunks: [Int] = []
        var chunkSeconds: [Double] = []
        let t0 = Date()
        var c0 = 0
        while c0 < L {
            let w = min(arm.widths[chunks.count % arm.widths.count], L - c0)
            let c1 = c0 + w
            let tc = Date()
            // The serving B1 prime: trunk forward, then the head over the next tokens (all interior: L < corpus).
            let (_, h) = qm.forwardHidden(xs[0..., c0 ..< c1], cache: cache!)
            let toks = xs[0..., (c0 + 1) ..< (c1 + 1)]
            let (_, S) = mtp(hidden: h, tokens: toks, embed: qm.model.embedTokens, cache: mtpCache!)
            eval(S, cache!.flatMap { $0.state }, mtpCache!.state)
            chunkSeconds.append(Date().timeIntervalSince(tc))
            chunks.append(w)
            c0 = c1
        }
        let wall = Date().timeIntervalSince(t0)
        let trunk = try layerDigests(cache!)
        let head = try layerDigests([mtpCache!])[0]
        var row: [String: Any] = [
            "arm": armIndex, "label": arm.label, "canonical_rows": arm.rows,
            "blocked_projection_calls": Qwen4ExpPrefillWidth.blockedCalls - blockedBefore,
            "chunks": chunks.count, "distinct_widths": Array(Set(chunks)).sorted(),
            "wall_s": wall, "tok_per_s": Double(L) / wall,
            "chunk_s_median": chunkSeconds.sorted()[chunkSeconds.count / 2],
            "trunk_digest": joined(trunk), "head_digest": joined([head]),
        ]
        if let ref = reference {
            let differing = zip(ref.trunk, trunk).enumerated().filter { $0.element.0 != $0.element.1 }.map { $0.offset }
            row["identical_to_arm0"] = differing.isEmpty && ref.head == head
            row["trunk_layers_differing"] = differing
            if let first = differing.first {
                row["first_differing_layer"] = first
                row["first_differing_layer_kind"] = trunk[first]["kind"] ?? ""
                row["first_differing_parts"] = trunk[first].keys.filter { $0 != "kind" && ref.trunk[first][$0] != trunk[first][$0] }.sorted()
            }
            row["head_identical"] = ref.head == head
            row["head_differing_parts"] = head.keys.filter { $0 != "kind" && ref.head[$0] != head[$0] }.sorted()
        } else {
            reference = (trunk, head)
            row["layer_kinds"] = trunk.map { $0["kind"] ?? "" }
        }
        results.append(row)
        FileHandle.standardError.write("engine H50: arm \(armIndex) \(arm.label): \(chunks.count) chunks, \(String(format: "%.1f", wall)) s, \(String(format: "%.0f", Double(L) / wall)) tok/s, identical \(row["identical_to_arm0"] ?? "reference")\n".data(using: .utf8)!)
        cache = nil; mtpCache = nil
        Memory.clearCache()
    }
    let out: [String: Any] = [
        "instrument": "H50 chunk-width state probe", "prompt_tokens": L, "corpus_sha256": corpusSHA256,
        "corpus_tokens_sha256": digest(corpus.prefix(L + 1).map { Int32($0).littleEndian }.withUnsafeBytes { Data($0) }),
        "indexer_ratio": ratio, "arms": results,
        "all_identical": results.dropFirst().allSatisfy { ($0["identical_to_arm0"] as? Bool) == true },
    ]
    let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys, .prettyPrinted])
    print(String(data: data, encoding: .utf8)!)
}
