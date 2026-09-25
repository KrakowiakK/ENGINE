// engine -- ENGINE's model-agnostic runtime driver.
//
//   engine generate --model DIR --prompt TEXT [--mtp K] [--state-cache DIR] [--state-cache-step N]
//   engine bench --model DIR [--prompt FILE] [--prompt-tokens N] [--decode M]
//                [--repeats R] [--prefill-step S] [--json OUT] [--mtp K] [--snapshot-dir DIR]
//
// Loads ANY model the vendored mlx-swift-lm factory knows (config.json
// model_type: laguna, qwen3_5, qwen3_5_text, ...) from a local directory and
// measures, greedy and deterministic:
//   prefill  : N prompt tokens through the model to the first generated token
//              (time-to-first-token, tokens/s = N / ttft)
//   decode   : M single-token steps, each synchronised on the sampled token
//              (tokens/s = M / wall)
// One unmeasured warm-up run compiles kernels and shapes. Every measured run
// starts from a fresh KV cache. The JSON report records the model identity,
// the exact token counts, every run, the medians, the first generated tokens
// (so two runs of the same binary can be checked for identity), peak GPU
// memory and the host.
//
// This is the generic path (mlx-swift-lm modules). The arena's optimised Qwen
// path (Qwen35FastEngine + MTP) is reached through `mlxfast-swift` and its
// runtime worker; the two are different programs and are reported as such.
import Foundation
import AVFoundation
import CoreMedia
import CryptoKit
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM
import MLXHuggingFace
import Tokenizers
import Qwen4Exp
import MLXVLM
import CoreImage

installCrashTrace()

struct Options {
    var verb = ""
    var values: [String: String] = [:]
    init(_ argv: [String]) throws {
        var args = argv.dropFirst()
        guard let v = args.first else { throw EngineError.usage }
        verb = v
        args = args.dropFirst()
        var it = args.makeIterator()
        while let a = it.next() {
            guard a.hasPrefix("--"), let val = it.next() else { throw EngineError.usage }
            values[a] = val
        }
    }
    func string(_ k: String, _ d: String) -> String { values[k] ?? d }
    func int(_ k: String, _ d: Int) throws -> Int {
        guard let s = values[k] else { return d }
        guard let i = Int(s), i > 0 else { throw EngineError.invalid("\(k) must be a positive integer") }
        return i
    }
    /// P106 B40: for options whose 0 is a documented value (adaptive chunk, off). `int` refuses 0, so an explicit
    /// `--prefill-chunk 0` -- the adaptive policy the launcher names -- could never be written (D46).
    func nonNegativeInt(_ k: String, _ d: Int) throws -> Int {
        guard let s = values[k] else { return d }
        guard let i = Int(s), i >= 0 else { throw EngineError.invalid("\(k) must be a non-negative integer") }
        return i
    }
}

enum EngineError: Error, CustomStringConvertible {
    case usage
    case invalid(String)
    var description: String {
        switch self {
        case .usage:
            return """
            usage: engine bench --model DIR [--prompt FILE] [--prompt-tokens 512] [--decode 128]
                                [--repeats 3] [--prefill-step 4096] [--json OUT]
                   engine logits --model DIR --ids FILE.json --out PREFIX   (parity fixtures)
                   engine ngram-ids --config DIR --ids 1,2,3 [--prev 0,0]  (n-gram row ids)
                   engine generate ... --mtp K   (speculative decoding with the checkpoint's MTP head; ENGINE_MTP=1)
            """
        case .invalid(let s): return s
        }
    }
}

struct RunResult: Codable {
    let prefill_seconds: Double
    let decode_seconds: Double
    let prefill_tps: Double
    let decode_tps: Double
    let generated_head: [Int]
    let generated_ids: [Int]
}

struct Report: Codable {
    let engine: String
    let model_dir: String
    let model_type: String
    let quantization: String
    let prompt_file: String
    let prompt_tokens: Int
    let decode_tokens: Int
    let prefill_step: Int
    let repeats: Int
    let load_seconds: Double
    let runs: [RunResult]
    let median_prefill_tps: Double
    let median_decode_tps: Double
    let tokens_identical_across_runs: Bool
    let peak_gpu_memory_bytes: Int
    let host: String
    let timestamp: String
}

final class FailureBox: @unchecked Sendable { var error: Error? = nil }

/// Phase anchors on the mach-uptime axis (DispatchTime.now().uptimeNanoseconds),
/// the same axis the GPU ledger (INST-QWEN-054 patch, MLX_GPU_LEDGER) stamps
/// command buffers on. Enabled by ENGINE_ANCHORS=<path>; one line per phase:
///   anchor: run=<r> phase=<prefill|decode> step=<i> t0=<ns> t1=<ns>
final class Anchors: @unchecked Sendable {
    static let shared = Anchors()
    let fh: FileHandle?
    init() {
        if let p = ProcessInfo.processInfo.environment["ENGINE_ANCHORS"] {
            FileManager.default.createFile(atPath: p, contents: nil)
            fh = FileHandle(forWritingAtPath: p)
        } else { fh = nil }
    }
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    func write(run: Int, phase: String, step: Int, t0: UInt64, t1: UInt64) {
        guard let fh else { return }
        fh.write("anchor: run=\(run) phase=\(phase) step=\(step) t0=\(t0) t1=\(t1)\n".data(using: .utf8)!)
    }
}

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    if s.isEmpty { return .nan }
    return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
}

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(cString: buf)
}

func configFields(_ dir: URL) -> (String, String) {
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return ("UNKNOWN", "UNKNOWN") }
    let mt = obj["model_type"] as? String ?? "UNKNOWN"
    var q = "none"
    if let qq = obj["quantization"] as? [String: Any] {
        let mode = qq["mode"] as? String ?? "affine"
        let bits = qq["bits"].map { "\($0)" } ?? "?"
        let gs = qq["group_size"].map { "\($0)" } ?? "?"
        q = "\(mode) bits=\(bits) group_size=\(gs)"
    }
    return (mt, q)
}

/// One greedy generation: prefill N prompt tokens, then M decode steps.
/// P018: when DIR/<base>_<N>.safetensors is missing, the largest DIR/<base>_<M>.safetensors with M < N (same prompt file, a prefix of
/// the same token stream) is a valid starting state: load it and prefill rows M..<N. Prefill-at-depth costs seconds instead of minutes.
func partialSnapshot(for url: URL, tokens N: Int) -> (URL, Int)? {
    let name = url.deletingPathExtension().lastPathComponent
    guard let us = name.lastIndex(of: "_") else { return nil }
    let base = String(name[..<us])
    let dir = url.deletingLastPathComponent()
    guard let items = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return nil }
    var best: (URL, Int)? = nil
    for f in items where f.hasPrefix(base + "_") && f.hasSuffix(".safetensors") {
        let mid = f.dropFirst(base.count + 1).dropLast(".safetensors".count)
        if let m = Int(String(mid)), N > m, (best?.1 ?? 0) < m { best = (dir.appendingPathComponent(f), m) }
    }
    return best
}

// MARK: - P024: the content-addressed prefix state cache
//
// TTFT at 262144 is 226.3 s on E8h and unit 1 measured 92.2% of it as O(T) per-token work at the
// roofs P020 already bounded, so there is no long-context lever inside the prefill: the whole
// depth-dependent term is 7.8%. What CAN be removed is paying it twice. A deployment "na granicy
// pelnego kontekstu" sends the same document turn after turn with a different question appended,
// and the trunk+MTP state after the first M tokens is a deterministic function of (model,
// tokens[0..<M]) -- so it can be stored and reused.
//
// KEYED BY CONTENT, NOT BY FILENAME. P017/P018 key a snapshot `<prompt-basename>_<N>`, which
// ASSERTS that the stored state is a prefix of the new token stream. That is safe for a bench
// fixture and unsafe for a cache: a wrong hit is a wrong answer, not a slow one. Here the key is a
// hash over the model identity and the exact token prefix, so a hit is a PROOF.
//
// WHY THE STATE IS BIT-IDENTICAL AND NOT MERELY CLOSE. A fresh prefill of T tokens issues chunks
// [0,W), [W,2W), ...; a continuation from a rung at M issues [M,M+W), ... Every rung is a multiple
// of the chunk width W, so the continuation's chunk sequence is a SUFFIX of the fresh one: the same
// rows, in the same order, against the same cache state, through the same kernels. Nothing is
// re-associated. `step % W == 0` is a correctness precondition, not a tuning knob, and is enforced.

func runOnce(model: any LanguageModel, tokens: [Int], decodeSteps: Int, prefillStep: Int, run: Int = -1, snapshotURL: URL? = nil)
    throws -> RunResult
{
    let cache = model.newCache(parameters: nil)
    let sampler = ArgMaxSampler()
    var state: LMOutput.State? = nil
    func step(_ prev: LMInput.Text) -> MLXArray {
        let r = model(prev[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        state = r.state
        return sampler.sample(logits: r.logits[0..., -1, 0...])
    }
    let input = LMInput(tokens: MLXArray(tokens))
    var generated: [Int] = []

    let t0 = Date()
    let a0 = Anchors.now()
    var y: LMInput.Text
    if let url = snapshotURL, FileManager.default.fileExists(atPath: url.path), let qm = model as? Qwen4ExpModel {
        // P017: start from the saved prefill state (trunk caches + n0); the load is charged to ttft, never to decode
        let d = try loadArrays(url: url)
        precondition(d["T"]!.item(Int32.self) == Int32(tokens.count), "snapshot \(url.lastPathComponent) was built for a different prompt length")
        qm.importCaches(cache, prefix: "trunk.", from: d)
        y = .init(tokens: d["n0"]!.reshaped(1))
        eval(cache.flatMap { $0.state }); eval(y.tokens)
    } else if let url = snapshotURL, let (purl, M) = partialSnapshot(for: url, tokens: tokens.count), let qm = model as? Qwen4ExpModel {
        // P018: continue the prefill from the largest shorter snapshot of the same prompt (rows M..<N in prefillStep chunks)
        let d = try loadArrays(url: purl)
        qm.importCaches(cache, prefix: "trunk.", from: d)
        eval(cache.flatMap { $0.state })
        prefillRowsCharged = tokens.count - M
        FileHandle.standardError.write("engine bench: continuing prefill from \(purl.lastPathComponent): rows \(M)..<\(tokens.count)\n".data(using: .utf8)!)
        let tc = Date()
        var c0 = M
        var last = MLXArray(0)
        while c0 < tokens.count {
            let c1 = min(tokens.count, c0 + max(1, prefillStep))
            let r = model(LMInput.Text(tokens: MLXArray(tokens[c0 ..< c1].map { Int32($0) }))[text: .newAxis], cache: cache, state: state)
            state = r.state; last = r.logits
            if c1 < tokens.count { asyncEval(last) }
            c0 = c1
        }
        y = .init(tokens: sampler.sample(logits: last[0..., -1, 0...]))
        eval(y.tokens)
        prefillContinuedSeconds = Date().timeIntervalSince(tc)
        FileHandle.standardError.write(String(format: "engine bench: continued prefill %d rows in %.2f s (%.1f tok/s), peak %.1f GB\n", tokens.count - M, prefillContinuedSeconds, Double(tokens.count - M) / prefillContinuedSeconds, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
        var dd: [String: MLXArray] = ["T": MLXArray(Int32(tokens.count)), "n0": y.tokens]
        qm.exportCaches(cache, prefix: "trunk.", into: &dd)
        try save(arrays: dd, url: url)
        FileHandle.standardError.write("engine bench: wrote prefill snapshot \(url.lastPathComponent) (\(dd.count) arrays)\n".data(using: .utf8)!)
    } else {
        switch try model.prepare(input, cache: cache, windowSize: prefillStep) {
        case .tokens(let rest):
            y = .init(tokens: step(rest))
        case .logits(let out):
            y = .init(tokens: sampler.sample(logits: out.logits[0..., -1, 0...]))
        }
        eval(y.tokens)
        if let url = snapshotURL, let qm = model as? Qwen4ExpModel {
            var d: [String: MLXArray] = ["T": MLXArray(Int32(tokens.count)), "n0": y.tokens]
            qm.exportCaches(cache, prefix: "trunk.", into: &d)
            try save(arrays: d, url: url)
            FileHandle.standardError.write("engine bench: wrote prefill snapshot \(url.lastPathComponent) (\(d.count) arrays)\n".data(using: .utf8)!)
        }
    }
    let prefillSeconds = Date().timeIntervalSince(t0)
    Anchors.shared.write(run: run, phase: "prefill", step: 0, t0: a0, t1: Anchors.now())
    generated.append(y.tokens.item(Int.self))

    let d0 = Date()
    let stepTiming = ProcessInfo.processInfo.environment["ENGINE_STEP_TIMING"] != nil
    var buildNs: [Double] = [], evalNs: [Double] = [], asyncNs: [Double] = [], waitNs: [Double] = []
    let pipelined = ProcessInfo.processInfo.environment["ENGINE_NO_PIPELINE"] == nil
    let threaded = ProcessInfo.processInfo.environment["ENGINE_BUILD_THREAD"] != nil
    if threaded {
        // build step i+1's graph on a second thread while the main thread encodes/dispatches step i
        let q = DispatchQueue(label: "engine.build", qos: .userInteractive)
        var pending: MLXArray = step(y)          // graph of step 0 (built here)
        var buildNs2: [Double] = []
        for i in 0..<decodeSteps {
            let tok = pending
            let s0 = Anchors.now()
            let sem = DispatchSemaphore(value: 0)
            var next: MLXArray? = nil
            let yNext = LMInput.Text(tokens: tok)
            if i + 1 < decodeSteps {
                q.async { let b0 = Anchors.now(); next = step(yNext); buildNs2.append(Double(Anchors.now() - b0)); sem.signal() }
            } else { sem.signal() }
            if ProcessInfo.processInfo.environment["ENGINE_BUILD_THREAD"] == "2" { sem.wait(); sem.signal() }  // serialise: build first, then encode
            asyncEval(tok)                       // encode + dispatch step i on this thread
            let s1 = Anchors.now()
            generated.append(y.tokens.item(Int.self))   // step i-1's token (already computed)
            let s2 = Anchors.now()
            sem.wait()
            if let n = next { pending = n }
            y = yNext
            if stepTiming && i >= 8 { buildNs.append(Double(s1 - s0)); evalNs.append(Double(s2 - s1)) }
            Anchors.shared.write(run: run, phase: "decode", step: i, t0: s0, t1: Anchors.now())
        }
        generated.append(y.tokens.item(Int.self))
        if stepTiming && !buildNs.isEmpty {
            let me = buildNs.reduce(0, +) / Double(buildNs.count) / 1e6, mw = evalNs.reduce(0, +) / Double(evalNs.count) / 1e6
            let mb = buildNs2.isEmpty ? 0 : buildNs2.reduce(0, +) / Double(buildNs2.count) / 1e6
            FileHandle.standardError.write(String(format: "step timing (threaded build): asyncEval %.2f ms  item wait %.2f ms  [background build %.2f ms]  per step\n", me, mw, mb).data(using: .utf8)!)
        }
        let decodeSeconds = Date().timeIntervalSince(d0)
        return RunResult(prefill_seconds: prefillSeconds, decode_seconds: decodeSeconds,
                         prefill_tps: Double(tokens.count) / prefillSeconds, decode_tps: Double(decodeSteps) / decodeSeconds,
                         generated_head: Array(generated.prefix(32)), generated_ids: generated)
    }
    // P017 teacher-forced argmax check (INST-ENG-009): ENGINE_FORCE_TOKENS=file feeds the file's tokens instead of the model's own
    // argmax (so two arms see the SAME inputs at every step); ENGINE_ARGMAX_OUT=file records "argmax margin" per step, where margin =
    // top1 - top2 logit. Without FORCE_TOKENS the recorded argmax is the generated sequence itself.
    let env = ProcessInfo.processInfo.environment
    let forced: [Int]? = env["ENGINE_FORCE_TOKENS"].flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }
        .map { $0.split(whereSeparator: { $0 == "\n" || $0 == "," || $0 == " " }).compactMap { Int($0.split(separator: " ").first ?? "") } }
    let argmaxOut = env["ENGINE_ARGMAX_OUT"]
    var argmaxLines: [String] = []
    if forced != nil || argmaxOut != nil {
        precondition(!threaded, "forced mode: unset ENGINE_BUILD_THREAD")
        var cur = y
        // P018: ENGINE_FORCE_FROM_START=1 -- the FIRST input is forced too (forced[0]), so arms whose n0 differ still see identical
        // inputs at every step; step i then reads forced[i] and feeds forced[i+1]
        let fromStart = env["ENGINE_FORCE_FROM_START"] != nil
        var forced = forced
        if fromStart, let f = forced, !f.isEmpty { cur = .init(tokens: MLXArray([Int32(f[0])])); forced = Array(f.dropFirst()) }
        let blockRows = Int(env["ENGINE_FORCE_BLOCK"] ?? "1") ?? 1      // > 1: feed the forced tokens in blocks of S rows (the verify's path)
        if blockRows > 1, let forced {
            var i = 0
            while i < min(decodeSteps, forced.count) {
                let S = min(blockRows, forced.count - i)
                let ids = [cur.tokens.item(Int.self)] + Array(forced[i ..< (i + S - 1)])   // rows at positions i..i+S-1
                let r = model(LMInput.Text(tokens: MLXArray(ids.map { Int32($0) }))[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
                state = r.state
                for s in 0 ..< S {
                    let row = r.logits[0, s, 0...].asType(.float32)
                    let top2 = MLX.top(row, k: 2)
                    let a = row.argMax(axis: -1).item(Int.self)
                    argmaxLines.append("\(a) \(abs((top2[0] - top2[1]).item(Float.self)))")
                    generated.append(a)
                }
                cur = .init(tokens: MLXArray([Int32(forced[i + S - 1])]))
                i += S
            }
        } else {
        for i in 0..<decodeSteps {
            let r = model(cur[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
            state = r.state
            let row = r.logits[0, -1, 0...].asType(.float32)
            let top2 = MLX.top(row, k: 2)
            let am = row.argMax(axis: -1)
            let margin = abs((top2[0] - top2[1]).item(Float.self))
            let a = am.item(Int.self)
            argmaxLines.append("\(a) \(margin)")
            generated.append(a)
            let next = forced.map { $0.count > i ? $0[i] : a } ?? a
            cur = .init(tokens: MLXArray([Int32(next)]))
        }
        }
        if let f = argmaxOut { try? argmaxLines.joined(separator: "\n").write(toFile: f, atomically: true, encoding: .utf8) }
        let decodeSeconds = Date().timeIntervalSince(d0)
        return RunResult(prefill_seconds: prefillSeconds, decode_seconds: decodeSeconds,
                         prefill_tps: Double(tokens.count) / prefillSeconds, decode_tps: Double(decodeSteps) / decodeSeconds,
                         generated_head: Array(generated.prefix(32)), generated_ids: generated)
    }
    for i in 0..<decodeSteps {
        let s0 = Anchors.now()
        let tok = step(y)              // graph construction only (lazy)
        let s1 = Anchors.now()
        if pipelined {
            // TokenIterator pattern: dispatch step i, then read step i-1's token (already done)
            let a0 = Anchors.now()
            asyncEval(tok)
            let a1 = Anchors.now()
            generated.append(y.tokens.item(Int.self))
            let a2 = Anchors.now()
            if stepTiming && i >= 8 { asyncNs.append(Double(a1 - a0)); waitNs.append(Double(a2 - a1)) }
        } else {
            generated.append(tok.item(Int.self))   // .item synchronises the step
        }
        y = .init(tokens: tok)
        let s2 = Anchors.now()
        if stepTiming && i >= 8 { buildNs.append(Double(s1 - s0)); evalNs.append(Double(s2 - s1)) }
        Anchors.shared.write(run: run, phase: "decode", step: i, t0: s0, t1: s2)
    }
    if pipelined { generated.append(y.tokens.item(Int.self)) }
    if stepTiming && !buildNs.isEmpty {
        let mb = buildNs.reduce(0, +) / Double(buildNs.count) / 1e6, me = evalNs.reduce(0, +) / Double(evalNs.count) / 1e6
        let ma = asyncNs.isEmpty ? 0 : asyncNs.reduce(0, +) / Double(asyncNs.count) / 1e6
        let mw = waitNs.isEmpty ? 0 : waitNs.reduce(0, +) / Double(waitNs.count) / 1e6
        FileHandle.standardError.write(String(format: "step timing (steps 8..): graph build %.2f ms  eval+sync %.2f ms  [asyncEval %.2f ms, item wait %.2f ms]  per step\n", mb, me, ma, mw).data(using: .utf8)!)
    }
    let decodeSeconds = Date().timeIntervalSince(d0)

    let pRows = prefillRowsCharged > 0 ? prefillRowsCharged : tokens.count
    let pSecs = prefillRowsCharged > 0 ? prefillContinuedSeconds : prefillSeconds
    prefillRowsCharged = 0
    return RunResult(
        prefill_seconds: prefillSeconds,
        decode_seconds: decodeSeconds,
        prefill_tps: Double(pRows) / pSecs,
        decode_tps: Double(decodeSteps) / decodeSeconds,
        generated_head: Array(generated.prefix(32)), generated_ids: generated)
}

/// P016 (OBS-ENG-029): MLX's residency set is dormant by default (wired limit 0), so the Metal driver wires every resource per
/// command buffer; after a long prefill the first MTP round paid ~1 s of "Wire Memory" work (1339 events + the 25.6 GB n-gram
/// table). With a wired limit, every buffer allocated from here on is committed to an MTLResidencySet once. Default: 75% of
/// physical memory (MLX refuses anything above the device's recommended working set); ENGINE_WIRED_LIMIT_GB overrides, 0 disables.
@discardableResult
func applyWiredLimit() -> [String: String] {
    // P018 (OBS-ENG-031 (6)): ENGINE_CACHE_LIMIT_GB caps MLX's buffer cache (the allocator held 307 GB of freed buffers after a 256k
    // prefill); probes the round-1 host stall hypothesis
    // Default 32 GB (OBS-ENG-032 (6)): the unlimited cache reached 307 GB after a 256k prefill and the first decode round then
    // stalled 4.2 s (cache + active 473 GB > the 412 GB wired limit); a 32 GB limit costs the prefill nothing (345.4 vs 346.7 s)
    // and the stall drops to 1.5 s. 0 = unlimited.
    let cgb = Double(ProcessInfo.processInfo.environment["ENGINE_CACHE_LIMIT_GB"] ?? "32") ?? 32
    if cgb > 0 {
        Memory.cacheLimit = Int(cgb * 1e9)
        FileHandle.standardError.write(String(format: "engine: MLX cache limit %.0f GB\n", cgb).data(using: .utf8)!)
    }
    let env = ProcessInfo.processInfo.environment["ENGINE_WIRED_LIMIT_GB"]
    // macOS 27 regressed with the large residency set; use the observed serving
    // profile by default on that OS. The operator can explicitly select either arm.
    let defaultGB = ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 27 ? 0 : Double(ProcessInfo.processInfo.physicalMemory) * 0.75 / 1e9
    var gb = env.flatMap { Double($0) } ?? defaultGB
    if gb <= 0 { FileHandle.standardError.write("engine: wired limit disabled\n".data(using: .utf8)!); return ["requested_gb": String(gb), "state": "DISABLED_BY_CONFIG"] }
    var ok = Memory.setWiredLimitRaw(Int(gb * 1e9))
    if !ok && env == nil { gb *= 0.6; ok = Memory.setWiredLimitRaw(Int(gb * 1e9)) }   // a smaller machine: retry at 45% of RAM
    FileHandle.standardError.write(String(format: "engine: wired limit %.0f GB -> %@\n", gb, ok ? "applied" : "REFUSED").data(using: .utf8)!)
    return ["requested_gb": env ?? String(defaultGB), "attempted_gb": String(gb), "state": ok ? "APPLIED" : "REQUESTED_BUT_NOT_APPLIED"]
}

func bench(_ o: Options) async throws {
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("bench requires --model DIR") }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    let promptFile = o.string("--prompt", "bench/prompts/english_longcopy_512.txt")
    let promptTokens = try o.int("--prompt-tokens", 512)
    let decodeSteps = try o.int("--decode", 128)
    let repeats = try o.int("--repeats", 3)
    let prefillStep = try o.int("--prefill-step", 4096)
    let jsonOut = o.values["--json"]

    let (modelType, quant) = configFields(dir)
    FileHandle.standardError.write("engine bench: loading \(dir.path) (model_type=\(modelType), \(quant))\n".data(using: .utf8)!)
    let l0 = Date()
    // ENGINE's own model families, registered ahead of the vendored factory's lookup
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    if let m = context.model as? Qwen4ExpModel { FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!) }
    let loadSeconds = Date().timeIntervalSince(l0)
    FileHandle.standardError.write(String(format: "engine bench: loaded in %.1f s, active GPU memory %.2f GB\n", loadSeconds, Double(GPU.activeMemory) / 1e9).data(using: .utf8)!)

    var text = try String(contentsOfFile: promptFile, encoding: .utf8)
    var tokens = context.tokenizer.encode(text: text, addSpecialTokens: false)
    while tokens.count < promptTokens {          // short prompt: repeat the text
        text += "\n" + text
        tokens = context.tokenizer.encode(text: text, addSpecialTokens: false)
    }
    tokens = Array(tokens.prefix(promptTokens))
    // P017: --snapshot-dir DIR -> DIR/<prompt-basename>_<N>.safetensors holds the prefill (+ MTP prime) state; built by the
    // MTP path (--mtp K), reused by both paths; the load is reported as ttft
    if let f = ProcessInfo.processInfo.environment["ENGINE_DUMP_TOKENS"] {
        // P018: write the prompt's token ids (one per line) and exit -- the forced-token files of INST-ENG-009 must come from the
        // engine's own tokenizer
        try tokens.map(String.init).joined(separator: "\n").write(toFile: f, atomically: true, encoding: .utf8)
        FileHandle.standardError.write("engine bench: wrote \(tokens.count) token ids to \(f)\n".data(using: .utf8)!)
        exit(0)
    }
    var snapshotURL: URL? = nil
    if let sd = o.values["--snapshot-dir"] {
        try FileManager.default.createDirectory(atPath: sd, withIntermediateDirectories: true)
        let base = URL(fileURLWithPath: promptFile).deletingPathExtension().lastPathComponent
        snapshotURL = URL(fileURLWithPath: sd).appendingPathComponent("\(base)_\(tokens.count).safetensors")
        FileHandle.standardError.write("engine bench: snapshot \(snapshotURL!.path) \(FileManager.default.fileExists(atPath: snapshotURL!.path) ? "(load)" : "(will be written)")\n".data(using: .utf8)!)
    }

    if let kStr = o.values["--mtp"], let K = Int(kStr), K >= 1, let qm = context.model as? Qwen4ExpModel {
        // speculative decode benchmark: same prompt, `decodeSteps` tokens per run, wall over the loop
        var runs: [RunResult] = []
        var accs: [Double] = []
        // warm-up: from the snapshot when there is one (the 64-token prefix never reaches the long-context kernels, so their JIT
        // compile landed in run 1 -- OBS-ENG-034), else a 64-token prefix
        if let s = snapshotURL, FileManager.default.fileExists(atPath: s.path) { _ = speculativeGenerate(model: qm, prompt: tokens, maxTokens: 8, depth: K, snapshotURL: s) }
        else { _ = speculativeGenerate(model: qm, prompt: Array(tokens.prefix(64)), maxTokens: 8, depth: K) }
        for i in 0..<repeats {
            var ttft = 0.0
            let g0 = Date()
            let (toks, st) = speculativeGenerate(model: qm, prompt: tokens, maxTokens: decodeSteps, depth: K, onPrefill: { ttft = Date().timeIntervalSince(g0) }, snapshotURL: snapshotURL)
            let total = Date().timeIntervalSince(g0)
            let r = RunResult(prefill_seconds: ttft, decode_seconds: total - ttft, prefill_tps: st.prefillRows > 0 ? Double(st.prefillRows) / st.prefillSeconds : Double(tokens.count) / ttft,
                              decode_tps: Double(toks.count) / (total - ttft), generated_head: Array(toks.prefix(32)), generated_ids: toks)
            runs.append(r); accs.append(Double(st.accepted) / Double(max(1, st.drafted)))
            FileHandle.standardError.write(String(format: "engine bench[mtp K=%d]: run %d  prefill %.1f tok/s  decode %.2f tok/s over %d tokens  acceptance %.2f  tokens/round %.2f\n",
                K, i + 1, r.prefill_tps, r.decode_tps, toks.count, accs.last!, Double(st.committed) / Double(max(1, st.rounds))).data(using: .utf8)!)
        }
        Q4Prof.report()
        let report = Report(
            engine: "ENGINE engine bench v0 (mlx-swift-lm generic path, greedy, MTP K=\(K))",
            model_dir: dir.path, model_type: modelType, quantization: quant,
            prompt_file: promptFile, prompt_tokens: tokens.count, decode_tokens: decodeSteps,
            prefill_step: prefillStep, repeats: repeats, load_seconds: loadSeconds, runs: runs,
            median_prefill_tps: median(runs.map { $0.prefill_tps }),
            median_decode_tps: median(runs.map { $0.decode_tps }),
            tokens_identical_across_runs: Set(runs.map { $0.generated_ids }).count == 1,
            peak_gpu_memory_bytes: GPU.peakMemory,
            host: sysctlString("machdep.cpu.brand_string") + " / " + sysctlString("kern.osproductversion"),
            timestamp: ISO8601DateFormatter().string(from: Date()))
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(report)
        if let jsonOut { try data.write(to: URL(fileURLWithPath: jsonOut)) }
        print(String(data: data, encoding: .utf8)!)
        return
    }

    // warm-up at the MEASURED shapes (full prompt, a few decode steps):
    // compiles kernels and shapes; never reported
    _ = try runOnce(model: context.model, tokens: tokens, decodeSteps: 8, prefillStep: prefillStep, snapshotURL: snapshotURL)
    Q4Prof.report(); Q4Prof.reset()      // a snapshot continuation prefills in the warm-up: report it before the reset
    GPU.resetPeakMemory()

    var runs: [RunResult] = []
    for i in 0..<repeats {
        let r = try runOnce(model: context.model, tokens: tokens, decodeSteps: decodeSteps, prefillStep: prefillStep, run: i, snapshotURL: snapshotURL)
        runs.append(r)
        FileHandle.standardError.write(String(format: "engine bench: run %d  prefill %.1f tok/s (%d tok in %.3f s)  decode %.2f tok/s (%d steps in %.3f s)\n",
            i + 1, r.prefill_tps, tokens.count, r.prefill_seconds, r.decode_tps, decodeSteps, r.decode_seconds).data(using: .utf8)!)
    }
    let identical = Set(runs.map { $0.generated_ids }).count == 1
    Q4Prof.report()
    let report = Report(
        engine: "ENGINE engine bench v0 (mlx-swift-lm generic path, greedy)",
        model_dir: dir.path, model_type: modelType, quantization: quant,
        prompt_file: promptFile, prompt_tokens: tokens.count, decode_tokens: decodeSteps,
        prefill_step: prefillStep, repeats: repeats, load_seconds: loadSeconds, runs: runs,
        median_prefill_tps: median(runs.map { $0.prefill_tps }),
        median_decode_tps: median(runs.map { $0.decode_tps }),
        tokens_identical_across_runs: identical,
        peak_gpu_memory_bytes: GPU.peakMemory,
        host: sysctlString("machdep.cpu.brand_string") + " / " + sysctlString("kern.osproductversion"),
        timestamp: ISO8601DateFormatter().string(from: Date()))
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try enc.encode(report)
    if let jsonOut {
        try data.write(to: URL(fileURLWithPath: jsonOut))
    }
    print(String(data: data, encoding: .utf8)!)
}

/// engine logits: prefill logits for a token list (all positions, float32 raw),
/// then 8 greedy decode steps through the cache; for parity against the Python reference.
func logitsDump(_ o: Options) async throws {
    setenv("ENGINE_FULL_LOGITS", "1", 1)   // parity tool: logits for every position

    guard let modelPath = o.values["--model"], let idsPath = o.values["--ids"], let outPrefix = o.values["--out"] else {
        throw EngineError.invalid("logits requires --model DIR --ids FILE.json --out PREFIX")
    }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    let ids = try JSONDecoder().decode([Int].self, from: Data(contentsOf: URL(fileURLWithPath: idsPath)))
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    let l0 = Date()
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    if let m = context.model as? Qwen4ExpModel { FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!) }
    FileHandle.standardError.write(String(format: "engine logits: loaded in %.1f s, active %.1f GB\n", Date().timeIntervalSince(l0), Double(GPU.activeMemory) / 1e9).data(using: .utf8)!)
    let model = context.model
    let cache = model.newCache(parameters: nil)
    let x = MLXArray(ids.map { Int32($0) })[.newAxis]
    let t0 = Date()
    let logits = model(x, cache: cache).asType(.float32)
    eval(logits)
    let dt = Date().timeIntervalSince(t0)
    let flat = logits[0].asArray(Float.self)
    try Data(bytes: flat, count: flat.count * 4).write(to: URL(fileURLWithPath: outPrefix + "_logits.f32"))
    var gen: [Int] = []
    var y = logits[0..., -1, 0...].argMax(axis: -1)  // (1)
    for _ in 0 ..< 8 {
        gen.append(y.item(Int.self))
        let l = model(y[.newAxis], cache: cache)
        y = l[0..., -1, 0...].argMax(axis: -1)
        eval(y)
    }
    let meta: [String: Any] = ["prompt_tokens": ids.count, "vocab": logits.dim(-1), "prefill_seconds": dt,
                               "greedy_8": gen, "peak_gpu_gb": Double(GPU.peakMemory) / 1e9]
    try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: outPrefix + "_meta.json"))
    print(String(data: try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys]), encoding: .utf8)!)
}

/// engine nll: teacher-forced per-position NLL over a token id list, computed ON DEVICE in chunks.
///
/// Why this exists rather than `engine logits` + numpy: a precision decision is a 0.001-0.01 nat
/// question (OBS-ENG-065) and resolving 0.004 nats needs ~26 000 positions, whose float32 logits are
/// 26 GB PER MODEL. Only the per-position NLL is ever used, and that is 4 bytes a position. Chunking
/// also keeps the logits tensor at chunk x vocab instead of S x vocab, so the pass fits beside a
/// 309 GB bf16 model. Output: `<out>` = float32[S-1] NLL, `<out>.argmax` = int32[S-1] argmax ids.
func nllDump(_ o: Options) async throws {
    setenv("ENGINE_FULL_LOGITS", "1", 1)
    guard let modelPath = o.values["--model"], let idsPath = o.values["--ids"], let outPath = o.values["--out"] else {
        throw EngineError.invalid("nll requires --model DIR --ids FILE.json --out FILE")
    }
    let chunk = try o.int("--chunk", 1024)
    let tailSerial = try o.int("--tail-serial", 0)   // P070: feed the last N ids one at a time (the S == 1 decode path)
    // P076 DECISIVENESS PROBE: --track-id N writes, per position, the log-probability of token N and its
    // rank in the distribution. With N = 248069 (`</think>`) this is the model's readiness to STOP THINKING
    // at every point of a chain it actually produced -- measurable without generating anything, so an
    // attention arm can be scored on decisiveness in one teacher-forced pass instead of a 6-minute sample.
    let trackId = try o.int("--track-id", -1)
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    let ids = try JSONDecoder().decode([Int].self, from: Data(contentsOf: URL(fileURLWithPath: idsPath)))
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    if let m = context.model as? Qwen4ExpModel { FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!) }
    let model = context.model
    let cache = model.newCache(parameters: nil)
    var nll: [Float] = [], am: [Int32] = []
    var trackLp: [Float] = [], trackRank: [Int32] = []
    let t0 = Date()
    var i = 0
    while i < ids.count {
        let hi = i >= ids.count - tailSerial ? i + 1 : min(i + chunk, max(i + 1, ids.count - tailSerial))
        let x = MLXArray(ids[i ..< hi].map { Int32($0) })[.newAxis]
        let l = model(x, cache: cache)[0].asType(.float32)        // (rows, V)
        // targets for rows i..<hi are ids[i+1...hi]; the LAST row of the LAST chunk has no target
        let n = hi - i - (hi == ids.count ? 1 : 0)
        if n > 0 {
            let tgt = MLXArray(ids[(i + 1) ... (i + n)].map { Int32($0) })
            let lp = l[0 ..< n, 0...]
            let lse = logSumExp(lp, axis: -1)
            let picked = takeAlong(lp, tgt[0..., .newAxis], axis: -1).squeezed(axis: -1)
            let d = lse - picked
            let a = lp.argMax(axis: -1).asType(.int32)
            eval(d, a)
            nll.append(contentsOf: d.asArray(Float.self))
            am.append(contentsOf: a.asArray(Int32.self))
            if trackId >= 0 {
                let col = lp[0..., trackId ..< (trackId + 1)].squeezed(axis: -1)      // (rows)
                let tlp = col - lse
                let rank = (lp .> expandedDimensions(col, axis: -1)).asType(.int32).sum(axis: -1)
                eval(tlp, rank)
                trackLp.append(contentsOf: tlp.asArray(Float.self))
                trackRank.append(contentsOf: rank.asArray(Int32.self))
            }
        }
        i = hi
    }
    try Data(bytes: nll, count: nll.count * 4).write(to: URL(fileURLWithPath: outPath))
    if trackId >= 0 {
        try Data(bytes: trackLp, count: trackLp.count * 4).write(to: URL(fileURLWithPath: outPath + ".track\(trackId).lp"))
        try Data(bytes: trackRank, count: trackRank.count * 4).write(to: URL(fileURLWithPath: outPath + ".track\(trackId).rank"))
        let best = trackLp.max() ?? -.infinity
        let top10 = trackRank.filter { $0 < 10 }.count
        FileHandle.standardError.write(String(format: "engine nll: track id %d -- max logprob %.4f (p=%.5f), positions in top-10 %d, top-1 %d\n",
            trackId, best, exp(best), top10, trackRank.filter { $0 == 0 }.count).data(using: .utf8)!)
    }
    try Data(bytes: am, count: am.count * 4).write(to: URL(fileURLWithPath: outPath + ".argmax"))
    let mean = nll.reduce(0, +) / Float(nll.count)
    let agree = zip(am, ids.dropFirst()).filter { Int($0.0) == $0.1 }.count
    FileHandle.standardError.write(String(format: "engine nll: %d positions, mean %.5f, ppl %.4f, top-1 %.2f%%, %.1f s, peak %.1f GB\n",
        nll.count, mean, exp(mean), 100.0 * Double(agree) / Double(nll.count), Date().timeIntervalSince(t0), Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
    print(String(format: "{\"positions\":%d,\"nll\":%.6f,\"ppl\":%.6f,\"top1\":%.6f}", nll.count, mean, exp(mean), Double(agree) / Double(nll.count)))
}

/// engine generate: greedy text generation with the chat template applied by the tokenizer
/// when --chat is given (otherwise raw prompt text). For eyeballing coherence.

// MARK: - P037: images

/// The checkpoint's `preprocessor_config.json` uses the NEWER Qwen-VL convention -- `size.shortest_edge`
/// / `size.longest_edge`, both in PIXELS -- while `Qwen3VLProcessorConfiguration` decodes `min_pixels`
/// / `max_pixels`. Decoding it raw would silently fall back to the library defaults (3136 / 12845056)
/// instead of the checkpoint's 65536 / 16777216, which changes the resize target, the patch grid and
/// therefore the NUMBER OF IMAGE TOKENS. Map it explicitly.
func enginePreprocessorConfig(_ modelDir: String) throws -> Qwen3VLProcessorConfiguration {
    let url = URL(fileURLWithPath: modelDir).appendingPathComponent("preprocessor_config.json")
    guard let raw = try? Data(contentsOf: url),
          var obj = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any] else {
        throw EngineError.invalid("--image needs preprocessor_config.json in the model directory")
    }
    if obj["min_pixels"] == nil || obj["max_pixels"] == nil, let size = obj["size"] as? [String: Any] {
        if let s = size["shortest_edge"] as? Int { obj["min_pixels"] = s }
        if let l = size["longest_edge"] as? Int { obj["max_pixels"] = l }
    }
    let fixed = try JSONSerialization.data(withJSONObject: obj)
    return try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: fixed)
}

let engineVisionStartId = 248053, engineImagePadId = 248056, engineVisionEndId = 248054
/// P044 unit 2: `<|video_pad|>` = the config's `video_token_id`. A video is ONE placeholder, exactly
/// like an image; what differs is the grid, which carries t = frames / temporal_patch_size (read off
/// `QwenVL.patchify`, not assumed -- HF describes a t=1+timestamps convention for a DIFFERENT variant).
let engineVideoPadId = 248057
/// Frames per second sampled from a video. The vendored `prepare` uses 2; exposed because the patch
/// cap binds the TOTAL and fps is the only knob that moves a video's cost without re-encoding it.
let engineVideoFPS: Int = Int(ProcessInfo.processInfo.environment["ENGINE_VIDEO_FPS"] ?? "2") ?? 2

/// (3, 1, T) mRoPE positions for a sequence carrying N vision spans, transcribed from HF's
/// `Qwen3VLModel.get_rope_index`: walk the token stream; for each span emit `arange(text_len)` on all
/// three axes offset by `st_idx`, then the (t, h, w) grid indices offset by `text_len + st_idx`, where
/// **`st_idx` is max(all previous positions) + 1**. For ONE span that reduces to `start + max(t,h,w)`,
/// which is what P037's single-image version did -- so this generalises it rather than replacing it,
/// and the one-span case must reproduce the old positions exactly (P044 gate (a)).
///
/// `spans` is (padStart, t, h, w) per image, in token order, with the POST-MERGE grid.
func engineVisionPositions(total: Int, spans: [(start: Int, t: Int, h: Int, w: Int)]) -> MLXArray {
    var T = [Int32](repeating: 0, count: total)
    var H = [Int32](repeating: 0, count: total)
    var W = [Int32](repeating: 0, count: total)
    var i = 0            // cursor in the token stream
    var next: Int32 = 0  // max(previous positions) + 1
    for sp in spans {
        while i < sp.start {                       // text before this span
            T[i] = next; H[i] = next; W[i] = next; next += 1; i += 1
        }
        let base = next
        for tt in 0 ..< sp.t { for hh in 0 ..< sp.h { for ww in 0 ..< sp.w {
            if i >= total { break }
            T[i] = base + Int32(tt); H[i] = base + Int32(hh); W[i] = base + Int32(ww); i += 1
        } } }
        next = base + Int32(max(max(sp.t, sp.h), sp.w))
    }
    while i < total { T[i] = next; H[i] = next; W[i] = next; next += 1; i += 1 }
    return MLXArray(T + H + W).reshaped(3, 1, total)
}

/// P039 unit 2 -- how many patches an image will cost, computed from the processor's own sizing math
/// WITHOUT loading weights or touching pixels. mlx-vlm exposes the same thing as
/// `estimate_num_image_tokens` and documents why: it is "cheap enough to run per candidate image when
/// sizing a prompt budget or choosing a max_pixels cap". Transcribed from
/// `QwenVL.targetSize` (= `image_processing_qwen2_vl.smart_resize`), which is what the real
/// preprocessing will run a moment later, so this is the same number and not an estimate of it.
func engineImageGrid(width: Int, height: Int, cfg: Qwen3VLProcessorConfiguration) -> (h: Int, w: Int, patches: Int) {
    let factor = cfg.patchSize * cfg.mergeSize
    var hBar = max(factor, Int((Float(height) / Float(factor)).rounded()) * factor)
    var wBar = max(factor, Int((Float(width) / Float(factor)).rounded()) * factor)
    if hBar * wBar > cfg.maxPixels {
        let beta = (Float(height * width) / Float(cfg.maxPixels)).squareRoot()
        hBar = Int((Float(height) / beta / Float(factor)).rounded(.down)) * factor
        wBar = Int((Float(width) / beta / Float(factor)).rounded(.down)) * factor
    } else if hBar * wBar < cfg.minPixels {
        let beta = (Float(cfg.minPixels) / Float(height * width)).squareRoot()
        hBar = Int((Float(height) * beta / Float(factor)).rounded(.up)) * factor
        wBar = Int((Float(width) * beta / Float(factor)).rounded(.up)) * factor
    }
    hBar = (hBar / factor) * factor; wBar = (wBar / factor) * factor
    let gh = hBar / cfg.patchSize, gw = wBar / cfg.patchSize
    return (gh, gw, gh * gw)
}

/// P039 unit 2: the vision tower's cost is superlinear in the patch count and the machine has no
/// backstop -- `ENGINE_GPU_MEMORY_LIMIT_GB` only moves MLX's cache-GC threshold (`block_limit_` feeds
/// `gc_limit_`), it does NOT make an allocation fail, so nothing below this line can save the host.
/// The refusal has to happen HERE, before 195 GB of weights are resident. Default 65536 is the
/// checkpoint's own `max_pixels / patch^2`, i.e. this cap only binds when the config would not.
let engineVisionMaxPatches = Int(ProcessInfo.processInfo.environment["ENGINE_VISION_MAX_PATCHES"] ?? "65536") ?? 65536

/// P037 unit 3 -- the image path. Deliberately a SEPARATE function: the text path above is a
/// bit-identical champion and nothing here is allowed to reach into it.
func generateWithImage(_ o: Options, imagePath: String) async throws {
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("generate requires --model DIR") }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    engineStopIdSet = engineStopIds(dir.path)
    let promptText = o.values["--prompt"] ?? "Describe this image."
    let maxTokens = try o.int("--tokens", 128)
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    // P039 unit 2: size the image BEFORE the weights. P037's guard sat AFTER `load` AND after the whole
    // vision tower had run -- its comment claimed "before 195 GB of weights get touched" and that was
    // simply false. An over-large image therefore died inside the tower with the model resident, which
    // is what took the host down.
    // P044: `--image` takes a COMMA-SEPARATED list. The cap is applied to the TOTAL patch count, not
    // per image: the tower attends over the concatenation of every image's patches, so OBS-ENG-111's
    // superlinear term is over the sum. Sizing every image before any weight loads, as P039 requires.
    var paths: [String] = []
    var isVideo: [Bool] = []
    for p in imagePath.split(separator: ",").map({ String($0).trimmingCharacters(in: .whitespaces) }) where !p.isEmpty {
        paths.append(p); isVideo.append(false)
    }
    for p in (o.values["--video"] ?? "").split(separator: ",").map({ String($0).trimmingCharacters(in: .whitespaces) }) where !p.isEmpty {
        paths.append(p); isVideo.append(true)
    }
    guard !paths.isEmpty else { throw EngineError.invalid("--image/--video: no path given") }
    let pcfgEarly = try enginePreprocessorConfig(dir.path)
    var totalPatches = 0
    for (i, pth) in paths.enumerated() {
        if isVideo[i] {
            // Priced from duration x fps and the frame's natural size -- the same arithmetic the
            // decoder will run, without decoding a single frame. `patchify` pads the frame count up to
            // temporal_patch_size and then gridT = frames / temporal_patch_size.
            let asset = AVURLAsset(url: URL(fileURLWithPath: pth))
            let dur = CMTimeGetSeconds(asset.duration)
            guard let track = asset.tracks(withMediaType: .video).first, dur > 0 else {
                throw EngineError.invalid("--video: cannot read a video track from \(pth)")
            }
            let sz = track.naturalSize.applying(track.preferredTransform)
            let g = engineImageGrid(width: abs(Int(sz.width)), height: abs(Int(sz.height)), cfg: pcfgEarly)
            let tp = pcfgEarly.temporalPatchSize
            var frames = max(1, Int((dur * Double(engineVideoFPS)).rounded(.up)))
            if frames % tp != 0 { frames += tp - frames % tp }
            let gridT = frames / tp
            let patches = gridT * g.patches
            totalPatches += patches
            FileHandle.standardError.write((
                "engine video: \(pth) \(abs(Int(sz.width)))x\(abs(Int(sz.height))) \(String(format: "%.2f", dur)) s "
                + "at \(engineVideoFPS) fps -> \(frames) frames, grid t=\(gridT) -> \(patches) patches, "
                + "\(patches / (pcfgEarly.mergeSize * pcfgEarly.mergeSize)) visual tokens\n"
                ).data(using: .utf8)!)
        } else {
            guard let ce = CIImage(contentsOf: URL(fileURLWithPath: pth)) else {
                throw EngineError.invalid("--image: cannot read \(pth)")
            }
            let e = ce.extent
            let g = engineImageGrid(width: Int(e.width), height: Int(e.height), cfg: pcfgEarly)
            totalPatches += g.patches
            FileHandle.standardError.write((
                "engine image: \(pth) \(Int(e.width))x\(Int(e.height)) -> \(g.patches) patches, "
                + "\(g.patches / (pcfgEarly.mergeSize * pcfgEarly.mergeSize)) visual tokens\n"
                ).data(using: .utf8)!)
        }
    }
    FileHandle.standardError.write((
        "engine vision: \(paths.count) medium(s), \(totalPatches) patches total (cap \(engineVisionMaxPatches))\n"
        ).data(using: .utf8)!)
    if totalPatches > engineVisionMaxPatches {
        throw EngineError.invalid(
            "--image: these \(paths.count) image(s) would cost \(totalPatches) vision patches in total, over the "
            + "\(engineVisionMaxPatches) cap (ENGINE_VISION_MAX_PATCHES). The vision tower attends over the "
            + "CONCATENATION, so its cost grows superlinearly in the SUM, and MLX's memory limit cannot make an "
            + "allocation fail (REFUT-ENG-025). Refused here, with no weights loaded, rather than by the "
            + "operating system. Downscale, pass fewer images, or raise the cap deliberately.")
    }

    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    guard let m = context.model as? Qwen4ExpModel else { throw EngineError.invalid("--image needs the qwen4_exp model") }
    guard m.hasVision else { throw EngineError.invalid("--image: this build has no vision tower (ENGINE_NO_VISION set, or the config declares none)") }
    FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!)

    // 1. pixels -- ONE preprocess call per image. `preprocess(images:)` is NOT multi-image: it resizes
    // every entry to the FIRST one's target and returns a single grid, which is the VIDEO convention
    // (frames stacked into t). Distinct images must be preprocessed separately and their patch rows
    // concatenated, with one THW per image -- which is exactly what the tower's `gridTHW: [THW]` and
    // its `cumulativeSequenceLengths` already expect.
    let pcfg = try enginePreprocessorConfig(dir.path)
    let proc = Qwen3VLProcessor(pcfg, tokenizer: context.tokenizer)
    let merge = pcfg.mergeSize
    var pixList: [MLXArray] = []
    var thws: [THW] = []
    for (i, pth) in paths.enumerated() {
        if isVideo[i] {
            // The vendored `Qwen3VLProcessor.prepare` video branch, inlined: sample the frames with
            // AVFoundation, resize every frame to the FIRST frame's target (so one grid covers them
            // all), then `patchify` with temporal_patch_size -- which is where t > 1 comes from.
            var resized: CGSize = .zero
            let seq = try await MediaProcessing.asProcessedSequence(
                .url(URL(fileURLWithPath: pth)), samplesPerSecond: engineVideoFPS
            ) { frame in
                let processed = MediaProcessing.apply(frame.frame, processing: nil)
                if resized == .zero {
                    let sz = processed.extent.size
                    let (hh, ww) = try QwenVL.targetSize(
                        height: Int(sz.height), width: Int(sz.width),
                        factor: pcfg.patchSize * pcfg.mergeSize,
                        minPixels: pcfg.minPixels, maxPixels: pcfg.maxPixels)
                    resized = CGSize(width: ww, height: hh)
                }
                let f = processed.toSRGB()
                    .resampled(to: resized, method: .bicubic)
                    .normalized(mean: pcfg.imageMeanTuple, std: pcfg.imageStdTuple)
                return VideoFrame(frame: f, timeStamp: frame.timeStamp)
            }
            let (px, th) = try QwenVL.patchify(
                images: seq.frames,
                mergeSize: pcfg.mergeSize, patchSize: pcfg.patchSize,
                temporalPatchSize: pcfg.temporalPatchSize)
            pixList.append(px); thws.append(th)
            FileHandle.standardError.write(
                "engine video: \(pth) -> \(seq.frames.count) frames, pixels \(px.shape), grid t=\(th.t) h=\(th.h) w=\(th.w)\n"
                    .data(using: .utf8)!)
        } else {
            guard let ci = CIImage(contentsOf: URL(fileURLWithPath: pth)) else {
                throw EngineError.invalid("--image: cannot read \(pth)")
            }
            let (px, th) = try proc.preprocess(images: [ci], processing: nil)
            pixList.append(px); thws.append(th)
            FileHandle.standardError.write(
                "engine image: \(pth) -> pixels \(px.shape), grid t=\(th.t) h=\(th.h) w=\(th.w), merge \(merge)\n"
                    .data(using: .utf8)!)
        }
    }
    let pixels = pixList.count == 1 ? pixList[0] : concatenated(pixList, axis: 0)

    // 2. visual tokens. With N images the tower runs N attention SEGMENTS -- the branch P039 built and
    // that had never executed until this unit; ENGINE_VISION_WITNESS=1 prints the segment count.
    guard let feats = m.visionFeatures(pixels: pixels, gridTHW: thws) else {
        throw EngineError.invalid("--image: no vision tower mounted")
    }
    eval(feats)
    let patchTotal = thws.reduce(0) { $0 + $1.t * $1.h * $1.w }
    FileHandle.standardError.write(String(format:
        "engine vision: tower done -- %d medium(s), %d patches, active %.2f GB, peak %.2f GB\n",
        thws.count, patchTotal, Double(GPU.activeMemory) / 1e9, Double(GPU.peakMemory) / 1e9)
        .data(using: .utf8)!)
    let nVis = feats.dim(0)
    let perImage = thws.map { ($0.t * $0.h * $0.w) / (merge * merge) }
    let expected = perImage.reduce(0, +)
    guard nVis == expected else {
        throw EngineError.invalid("--image: the tower returned \(nVis) visual tokens but the grids predict \(expected). A silent mismatch here shifts every downstream position.")
    }
    FileHandle.standardError.write("engine vision: \(nVis) visual tokens \(perImage), feature dim \(feats.dim(1))\n".data(using: .utf8)!)

    // 3. the token sequence: the template emits ONE placeholder PER IMAGE, the caller expands each
    let eff = o.values["--reasoning-effort"]
    // One placeholder per medium, in the order the media were given: images first, then videos, which
    // is the order `paths` carries. A video is `<|video_pad|>` and is otherwise treated identically --
    // the template emits one placeholder for it too (`chat_template.jinja`).
    let padIds = isVideo.map { $0 ? engineVideoPadId : engineImagePadId }
    let content = zip(paths, isVideo).map { _, v in
        "<|vision_start|>" + (v ? "<|video_pad|>" : "<|image_pad|>") + "<|vision_end|>\n"
    }.joined() + promptText
    var ids = try context.tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": content]], tools: nil,
        additionalContext: eff.map { ["reasoning_effort": $0 as any Sendable] })
    let padSet: Set<Int> = [engineImagePadId, engineVideoPadId]
    var padPositions: [Int] = []
    for (i, id) in ids.enumerated() where padSet.contains(id) { padPositions.append(i) }
    guard padPositions.count == paths.count,
          zip(padPositions, padIds).allSatisfy({ ids[$0.0] == $0.1 }) else {
        throw EngineError.invalid("--image/--video: expected \(paths.count) vision placeholders in template order \(padIds), found \(padPositions.map { ids[$0] })")
    }
    // Expand each placeholder to its own count, LAST FIRST so the earlier indices stay valid.
    for (k, at) in padPositions.enumerated().reversed() {
        ids.replaceSubrange(at ... at, with: Array(repeating: padIds[k], count: perImage[k]))
    }
    // Re-derive each span's start AFTER expansion; they are the first row of each run of pads.
    var spans: [(start: Int, t: Int, h: Int, w: Int)] = []
    var cursor = 0
    for (k, thw) in thws.enumerated() {
        guard let at = ids[cursor...].firstIndex(of: padIds[k]) else {
            throw EngineError.invalid("--image/--video: lost vision span \(k) after expansion")
        }
        spans.append((start: at, t: thw.t, h: thw.h / merge, w: thw.w / merge))
        cursor = at + perImage[k]
    }
    FileHandle.standardError.write("engine vision: prompt tokens \(ids.count), spans \(spans.map { "\($0.start)..<\($0.start + $0.t * $0.h * $0.w)" })\n".data(using: .utf8)!)
    // P039 unit 3: the scope limit is LIFTED. A visual prompt above the indexer budget now ropes the
    // pooled block starts by INDEXING the mRoPE table at those rows, which is what the reference does
    // (`full_cos.index_select(0, group_starts)`); the full 3-axis positions ride in indexer cache slot
    // 2, as HF binds them to its own cache for the same reason.
    let visBudget = m.configuration.text.indexerBudget
    if ids.count >= visBudget {
        FileHandle.standardError.write((
            "engine vision: prompt \(ids.count) tokens is ABOVE the QSA indexer budget \(visBudget) -- "
            + "the sparse path runs, with pooled block starts roped from the mRoPE table (P039 unit 3)\n"
            ).data(using: .utf8)!)
    }

    // 4. splice each image's merger rows onto ITS OWN span
    let xs = MLXArray(ids.map { Int32($0) })[.newAxis]
    var embeds = m.embedTokenIds(xs)
    let featsCast = feats.asType(embeds.dtype)[.newAxis]
    var fRow = 0
    for (k, sp) in spans.enumerated() {
        let n = perImage[k]
        embeds[0..., sp.start ..< (sp.start + n), 0...] = featsCast[0..., fRow ..< (fRow + n), 0...]
        fRow += n
    }

    // 5. mRoPE positions over every span (HF `get_rope_index`)
    let pos3 = engineVisionPositions(total: ids.count, spans: spans)
    var nextPos = Int32(pos3[0, 0, ids.count - 1].item(Int32.self)) + 1

    // 6. prefill + decode
    let cache = m.newCache(parameters: nil)
    let t0 = Date()
    let logits = m(xs, cache: cache, embedsOverride: embeds, pos3: pos3)
    var y = logits[0..., -1, 0...].argMax(axis: -1)
    eval(y)
    let ttft = Date().timeIntervalSince(t0)
    var out: [Int] = []
    let d0 = Date()
    var pending = y
    for _ in 0 ..< maxTokens {
        let p1 = MLXArray([nextPos, nextPos, nextPos]).reshaped(3, 1, 1)
        let lg = m(pending[.newAxis], cache: cache, embedsOverride: nil, pos3: p1)[0..., -1, 0...]
        let next = engineSampler.sample(lg)
        asyncEval(next)
        let t = pending.item(Int.self)
        out.append(t)
        nextPos += 1
        if engineStopIdSet.contains(t) || (context.tokenizer.eosTokenId.map { t == $0 } ?? false) { break }
        pending = next
    }
    let dec = Date().timeIntervalSince(d0)
    print(context.tokenizer.decode(tokenIds: out))
    FileHandle.standardError.write(String(format: "\nengine vision: ttft %.2f s, decode %.2f tok/s over %d tokens, peak %.1f GB\n",
        ttft, Double(out.count) / dec, out.count, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
}

/// P051 -- `--chat-file PATH`'s JSON shape: an array of turns. `content` is either plain text or the
/// same content-item array HF's own messages use (`{"type": "text"|"image"|"video", ...}`); an
/// image/video item also carries `path` (never sent to the template -- only `type` is, matching what
/// `render_content` actually reads).
private struct ChatFileItem: Decodable {
    let type: String
    let text: String?
    let path: String?
}
private enum ChatFileContent: Decodable {
    case text(String)
    case items([ChatFileItem])
    init(from decoder: Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .text(s)
        } else {
            self = .items(try c.decode([ChatFileItem].self))
        }
    }
}
private struct ChatFileTurn: Decodable {
    let role: String
    let content: ChatFileContent
    let reasoning_content: String?
}

/// P051 -- multi-turn conversations with media anywhere, not only the single (only) turn
/// `generateWithImage`/`--chat` can build. A SEPARATE function, same reason as
/// `generateWithImage`'s own comment: nothing here reaches into the bit-identical single-turn paths.
/// The content items pass through to `applyChatTemplate` in HF's own shape, so the REAL
/// `chat_template.jinja` renders the vision placeholders and the per-role `<|im_start|>`/`<|im_end|>`
/// wrapping (including `preserve_thinking` over `reasoning_content`) -- this function does not
/// reimplement any of that by hand, only the media-loading and pad-expansion machinery
/// `generateWithImage` already has, which is already generic over WHICH message a medium sits in.
func generateWithChatFile(_ o: Options, chatFilePath: String) async throws {
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("generate requires --model DIR") }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    engineStopIdSet = engineStopIds(dir.path)
    let maxTokens = try o.int("--tokens", 128)
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    let turns = try JSONDecoder().decode(
        [ChatFileTurn].self, from: Data(contentsOf: URL(fileURLWithPath: chatFilePath)))
    guard !turns.isEmpty else { throw EngineError.invalid("--chat-file: no turns in \(chatFilePath)") }

    // Flatten every image/video reference across every turn, in document order.
    var paths: [String] = []
    var isVideo: [Bool] = []
    for turn in turns {
        guard case .items(let items) = turn.content else { continue }
        for item in items where item.type == "image" || item.type == "video" {
            guard let p = item.path else {
                throw EngineError.invalid("--chat-file: a \(item.type) item needs \"path\"")
            }
            paths.append(p); isVideo.append(item.type == "video")
        }
    }
    guard !paths.isEmpty else { throw EngineError.invalid("--chat-file: no image/video item in any turn") }

    // 0. size every medium BEFORE the weights load (P039 unit 2's rule, unchanged).
    let pcfgEarly = try enginePreprocessorConfig(dir.path)
    var totalPatches = 0
    for (i, pth) in paths.enumerated() {
        if isVideo[i] {
            let asset = AVURLAsset(url: URL(fileURLWithPath: pth))
            let dur = CMTimeGetSeconds(asset.duration)
            guard let track = asset.tracks(withMediaType: .video).first, dur > 0 else {
                throw EngineError.invalid("--chat-file: cannot read a video track from \(pth)")
            }
            let sz = track.naturalSize.applying(track.preferredTransform)
            let g = engineImageGrid(width: abs(Int(sz.width)), height: abs(Int(sz.height)), cfg: pcfgEarly)
            let tp = pcfgEarly.temporalPatchSize
            var frames = max(1, Int((dur * Double(engineVideoFPS)).rounded(.up)))
            if frames % tp != 0 { frames += tp - frames % tp }
            totalPatches += (frames / tp) * g.patches
        } else {
            guard let ce = CIImage(contentsOf: URL(fileURLWithPath: pth)) else {
                throw EngineError.invalid("--chat-file: cannot read \(pth)")
            }
            let e = ce.extent
            totalPatches += engineImageGrid(width: Int(e.width), height: Int(e.height), cfg: pcfgEarly).patches
        }
    }
    FileHandle.standardError.write((
        "engine vision: \(paths.count) medium(s) across \(turns.count) turn(s), \(totalPatches) patches total (cap \(engineVisionMaxPatches))\n"
        ).data(using: .utf8)!)
    if totalPatches > engineVisionMaxPatches {
        throw EngineError.invalid(
            "--chat-file: these \(paths.count) media would cost \(totalPatches) vision patches in total, over the "
            + "\(engineVisionMaxPatches) cap (ENGINE_VISION_MAX_PATCHES).")
    }

    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    guard let m = context.model as? Qwen4ExpModel else { throw EngineError.invalid("--chat-file needs the qwen4_exp model") }
    guard m.hasVision else { throw EngineError.invalid("--chat-file: this build has no vision tower") }
    FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!)

    // 1. pixels, one preprocess call per medium -- identical to `generateWithImage`'s step 1.
    let pcfg = try enginePreprocessorConfig(dir.path)
    let proc = Qwen3VLProcessor(pcfg, tokenizer: context.tokenizer)
    let merge = pcfg.mergeSize
    var pixList: [MLXArray] = []
    var thws: [THW] = []
    for (i, pth) in paths.enumerated() {
        if isVideo[i] {
            var resized: CGSize = .zero
            let seq = try await MediaProcessing.asProcessedSequence(
                .url(URL(fileURLWithPath: pth)), samplesPerSecond: engineVideoFPS
            ) { frame in
                let processed = MediaProcessing.apply(frame.frame, processing: nil)
                if resized == .zero {
                    let sz = processed.extent.size
                    let (hh, ww) = try QwenVL.targetSize(
                        height: Int(sz.height), width: Int(sz.width),
                        factor: pcfg.patchSize * pcfg.mergeSize,
                        minPixels: pcfg.minPixels, maxPixels: pcfg.maxPixels)
                    resized = CGSize(width: ww, height: hh)
                }
                let f = processed.toSRGB()
                    .resampled(to: resized, method: .bicubic)
                    .normalized(mean: pcfg.imageMeanTuple, std: pcfg.imageStdTuple)
                return VideoFrame(frame: f, timeStamp: frame.timeStamp)
            }
            let (px, th) = try QwenVL.patchify(
                images: seq.frames,
                mergeSize: pcfg.mergeSize, patchSize: pcfg.patchSize,
                temporalPatchSize: pcfg.temporalPatchSize)
            pixList.append(px); thws.append(th)
        } else {
            guard let ci = CIImage(contentsOf: URL(fileURLWithPath: pth)) else {
                throw EngineError.invalid("--chat-file: cannot read \(pth)")
            }
            let (px, th) = try proc.preprocess(images: [ci], processing: nil)
            pixList.append(px); thws.append(th)
        }
    }
    let pixels = pixList.count == 1 ? pixList[0] : concatenated(pixList, axis: 0)
    guard let feats = m.visionFeatures(pixels: pixels, gridTHW: thws) else {
        throw EngineError.invalid("--chat-file: no vision tower mounted")
    }
    eval(feats)
    let nVis = feats.dim(0)
    let perImage = thws.map { ($0.t * $0.h * $0.w) / (merge * merge) }
    let expected = perImage.reduce(0, +)
    guard nVis == expected else {
        throw EngineError.invalid("--chat-file: the tower returned \(nVis) visual tokens but the grids predict \(expected)")
    }

    // 2. render the FULL conversation through the REAL template, one message per turn -- content
    // items pass through as HF shapes them; `render_content` is what turns those into
    // `<|vision_start|>...<|vision_end|>` placeholders and, per OBS-ENG-127, no ordinal label unless
    // `add_vision_id` is explicitly asked for (this function never sets it, matching the reference
    // default and every other CLI path).
    let eff = o.values["--reasoning-effort"]
    var messages: [[String: any Sendable]] = []
    for turn in turns {
        var msg: [String: any Sendable] = ["role": turn.role]
        switch turn.content {
        case .text(let s):
            msg["content"] = s
        case .items(let items):
            msg["content"] = items.map { item -> [String: any Sendable] in
                item.type == "text" ? ["type": "text", "text": item.text ?? ""] : ["type": item.type]
            }
        }
        if let rc = turn.reasoning_content { msg["reasoning_content"] = rc }
        messages.append(msg)
    }
    var ids = try context.tokenizer.applyChatTemplate(
        messages: messages, tools: nil,
        additionalContext: eff.map { ["reasoning_effort": $0 as any Sendable] })
    // P051 falsifier: ENGINE_DUMP_CHAT_IDS=path writes the PRE-expansion ids (one row per pad, not yet
    // expanded to its patch count) and exits -- this is the exact quantity to diff against the
    // reference checkpoint's own AutoTokenizer.apply_chat_template on the equivalent HF-shaped messages.
    if let f = ProcessInfo.processInfo.environment["ENGINE_DUMP_CHAT_IDS"] {
        try ids.map(String.init).joined(separator: "\n").write(toFile: f, atomically: true, encoding: .utf8)
        FileHandle.standardError.write("engine chat-file: wrote \(ids.count) pre-expansion token ids to \(f)\n".data(using: .utf8)!)
        exit(0)
    }

    // 3. pad scanning, expansion, spans -- identical machinery to `generateWithImage`'s steps 3-5,
    // generic over a flat ids/thws/perImage sequence regardless of which message the media sat in.
    let padIds = isVideo.map { $0 ? engineVideoPadId : engineImagePadId }
    let padSet: Set<Int> = [engineImagePadId, engineVideoPadId]
    var padPositions: [Int] = []
    for (i, id) in ids.enumerated() where padSet.contains(id) { padPositions.append(i) }
    guard padPositions.count == paths.count,
          zip(padPositions, padIds).allSatisfy({ ids[$0.0] == $0.1 }) else {
        throw EngineError.invalid("--chat-file: expected \(paths.count) vision placeholders across the conversation, found \(padPositions.map { ids[$0] })")
    }
    for (k, at) in padPositions.enumerated().reversed() {
        ids.replaceSubrange(at ... at, with: Array(repeating: padIds[k], count: perImage[k]))
    }
    var spans: [(start: Int, t: Int, h: Int, w: Int)] = []
    var cursor = 0
    for (k, thw) in thws.enumerated() {
        guard let at = ids[cursor...].firstIndex(of: padIds[k]) else {
            throw EngineError.invalid("--chat-file: lost vision span \(k) after expansion")
        }
        spans.append((start: at, t: thw.t, h: thw.h / merge, w: thw.w / merge))
        cursor = at + perImage[k]
    }
    FileHandle.standardError.write("engine vision: prompt tokens \(ids.count), spans \(spans.map { "\($0.start)..<\($0.start + $0.t * $0.h * $0.w)" })\n".data(using: .utf8)!)
    let visBudget = m.configuration.text.indexerBudget
    if ids.count >= visBudget {
        FileHandle.standardError.write((
            "engine vision: prompt \(ids.count) tokens is ABOVE the QSA indexer budget \(visBudget) -- the sparse path runs\n"
            ).data(using: .utf8)!)
    }

    // 4. splice, 5. positions, 6. prefill+decode -- unchanged from `generateWithImage`.
    let xs = MLXArray(ids.map { Int32($0) })[.newAxis]
    var embeds = m.embedTokenIds(xs)
    let featsCast = feats.asType(embeds.dtype)[.newAxis]
    var fRow = 0
    for (k, sp) in spans.enumerated() {
        let n = perImage[k]
        embeds[0..., sp.start ..< (sp.start + n), 0...] = featsCast[0..., fRow ..< (fRow + n), 0...]
        fRow += n
    }
    let pos3 = engineVisionPositions(total: ids.count, spans: spans)
    var nextPos = Int32(pos3[0, 0, ids.count - 1].item(Int32.self)) + 1
    let cache = m.newCache(parameters: nil)
    let t0 = Date()
    let logits = m(xs, cache: cache, embedsOverride: embeds, pos3: pos3)
    var y = logits[0..., -1, 0...].argMax(axis: -1)
    eval(y)
    let ttft = Date().timeIntervalSince(t0)
    var out: [Int] = []
    let d0 = Date()
    var pending = y
    for _ in 0 ..< maxTokens {
        let p1 = MLXArray([nextPos, nextPos, nextPos]).reshaped(3, 1, 1)
        let lg = m(pending[.newAxis], cache: cache, embedsOverride: nil, pos3: p1)[0..., -1, 0...]
        let next = engineSampler.sample(lg)
        asyncEval(next)
        let t = pending.item(Int.self)
        out.append(t)
        nextPos += 1
        if engineStopIdSet.contains(t) || (context.tokenizer.eosTokenId.map { t == $0 } ?? false) { break }
        pending = next
    }
    let dec = Date().timeIntervalSince(d0)
    print(context.tokenizer.decode(tokenIds: out))
    FileHandle.standardError.write(String(format: "\nengine vision: ttft %.2f s, decode %.2f tok/s over %d tokens, peak %.1f GB\n",
        ttft, Double(out.count) / dec, out.count, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
}

func generateText(_ o: Options) async throws {
    if let cf = o.values["--chat-file"] {
        try await generateWithChatFile(o, chatFilePath: cf); return
    }
    if o.values["--image"] != nil || o.values["--video"] != nil {
        try await generateWithImage(o, imagePath: o.values["--image"] ?? ""); return
    }
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("generate requires --model DIR") }
    let dir_model = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    engineStopIdSet = engineStopIds(dir_model.path)
    if let v = o.values["--think-budget"], let n = Int(v) { engineThinkBudget = n }
    if let v = o.values["--think-nudge"], let n = Int(v) { engineThinkNudgeAt = n }
    // P077 soft stop: --think-bias-start A --think-bias-full B --think-bias-max C
    if let v = o.values["--think-bias-max"], let f = Float(v) {
        engineThinkBias.maxBias = f
        engineThinkBias.start = (o.values["--think-bias-start"].flatMap { Int($0) }) ?? 2000
        engineThinkBias.full = (o.values["--think-bias-full"].flatMap { Int($0) }) ?? 8000
        engineThinkBias.deadline = (o.values["--think-bias-deadline"].flatMap { Int($0) }) ?? 0
        let dl = engineThinkBias.deadline > 0 ? ", deadline \(engineThinkBias.deadline) (close guaranteed)" : ""
        FileHandle.standardError.write("engine: soft think bias ramp \(engineThinkBias.start) -> \(engineThinkBias.full) tokens, max +\(f) on </think>\(dl)\n".data(using: .utf8)!)
    }
    if o.values["--sample"] != nil || o.values["--temp"] != nil {
        var sp = EngineSampler(temp: 1.0, topP: 0.95, topK: 20)     // the checkpoint's own settings
        if let v = o.values["--temp"], let f = Float(v) { sp.temp = f }
        if let v = o.values["--top-p"], let f = Float(v) { sp.topP = f }
        if let v = o.values["--top-k"], let n = Int(v) { sp.topK = n }
        engineSampler = sp
        if let v = o.values["--seed"], let n = UInt64(v) { MLXRandom.seed(n) }
        FileHandle.standardError.write("engine generate: sampling temp \(sp.temp) top_p \(sp.topP) top_k \(sp.topK)\n".data(using: .utf8)!)
    }
    if !engineStopIdSet.isEmpty {
        FileHandle.standardError.write("engine generate: stop ids \(engineStopIdSet.sorted()) (from generation_config.json)\n".data(using: .utf8)!)
    }
    let dir = dir_model
    let promptText = o.values["--prompt"] ?? "The capital of France is"
    let maxTokens = try o.int("--tokens", 64)
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    if let m = context.model as? Qwen4ExpModel { FileHandle.standardError.write((m.fusionWitness() + "\n").data(using: .utf8)!) }
    var ids: [Int]
    // P024: a full-context turn cannot be typed on a command line. --prompt-file FILE reads the
    // document, --prompt-tokens N truncates it to an exact token count (so two turns can be given a
    // provably identical prefix), and --append TEXT is the question that turn adds after it.
    if let pf = o.values["--prompt-file"] {
        let text = try String(contentsOf: URL(fileURLWithPath: pf), encoding: .utf8)
        ids = context.tokenizer.encode(text: text, addSpecialTokens: false)
        if let n = try? o.int("--prompt-tokens", 0), n > 0 {
            guard ids.count >= n else { throw EngineError.invalid("--prompt-file has \(ids.count) tokens, fewer than --prompt-tokens \(n)") }
            ids = Array(ids[0 ..< n])
        }
        if let ap = o.values["--append"] { ids += context.tokenizer.encode(text: ap, addSpecialTokens: false) }
        if let flip = try? o.int("--flip-token", 0), flip > 0 {
            // NEGATIVE CONTROL: corrupt ONE token inside the shared prefix. A content-addressed cache
            // must MISS here; a filename-keyed one would hit and answer from the wrong state.
            ids[flip - 1] = ids[flip - 1] == 100 ? 101 : 100
            FileHandle.standardError.write("engine generate: NEGATIVE CONTROL -- token \(flip) altered\n".data(using: .utf8)!)
        }
    } else if o.values["--chat"] != nil {
        // P033 unit 5 -- MEASURED HAZARD, and it is the template's DEFAULT. `chat_template.jinja`
        // resolves `reasoning_effort|default('xhigh')` and at that setting injects "think carefully
        // through the task, validate key assumptions, consider plausible alternatives". On an
        // eight-part research question that instruction has no stopping rule: greedy at xhigh ran
        // **16 000 tokens WITHOUT EVER CLOSING `</think>`**, while `medium` closed at 1109 thinking
        // tokens and `low` at 420, both producing a complete answer, on the SAME attention path and
        // the SAME decoding. The vendor's default is kept -- deviating from it silently is the
        // operator's call, not this engine's -- but it is now selectable and the hazard is recorded.
        let eff = o.values["--reasoning-effort"]
        if let eff, !["xhigh", "medium", "low"].contains(eff) {
            throw EngineError.invalid("--reasoning-effort must be xhigh, medium or low")
        }
        ids = try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": promptText]], tools: nil,
            additionalContext: eff.map { ["reasoning_effort": $0 as any Sendable] })
    } else {
        ids = context.tokenizer.encode(text: promptText, addSpecialTokens: false)
    }
    engineNudgeIds = context.tokenizer.encode(
        text: o.values["--think-nudge-text"]
            ?? "\n\nOK, I have enough material. Let me stop analysing and write the final answer now.\n",
        addSpecialTokens: false)
    engineForceQueue = []
    // the thinking block is open iff the prompt ends inside one: the generation prompt emits `<think>`
    engineThinkOpen = ids.lastIndex(of: engineThinkOpenId).map { open in
        (ids.lastIndex(of: engineThinkCloseId) ?? -1) < open } ?? false
    engineThinkForced = false; engineGenCount = 0
    FileHandle.standardError.write("engine generate: prompt tokens \(ids.count), think block \(engineThinkOpen ? "OPEN" : "closed"), budget \(engineThinkBudget)\n".data(using: .utf8)!)
    // P024: --state-cache DIR keys the prefill state by a hash of (model identity, token prefix), so a
    // turn that re-sends the same document pays only for the rows the cache does not already cover.
    // The step MUST be a multiple of the prefill chunk width: a rung at a non-multiple would make the
    // continuation issue a chunk sequence a fresh prefill never issues, and the state would no longer
    // be the state. See the StateCache comment.
    var stateCache: StateCacheConfig? = nil
    if let sd = o.values["--state-cache"] {
        let W = Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "4096") ?? 4096
        let step = try o.int("--state-cache-step", 32768)
        guard step % W == 0 else { throw EngineError.invalid("--state-cache-step must be a multiple of the prefill chunk width (\(W))") }
        let dir = URL(fileURLWithPath: sd)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let witness = (context.model as? Qwen4ExpModel)?.fusionWitness() ?? ""
        let exact = o.values["--state-cache-exact"] != nil
        stateCache = StateCacheConfig(dir: dir, identity: try StateCache.identity(dir: dir_model, witness: witness), step: step, writeExact: exact).indexed(for: ids)
        FileHandle.standardError.write("engine generate: state cache \(dir.path) step \(step) exact-entry \(exact ? "on" : "off")\n".data(using: .utf8)!)
    }
    if let kStr = o.values["--mtp"], let K = Int(kStr), K >= 1, let m = context.model as? Qwen4ExpModel {
        let g0 = Date()
        var ttft = 0.0
        let (toks, st) = speculativeGenerate(model: m, prompt: ids, maxTokens: maxTokens, depth: K, onPrefill: { ttft = Date().timeIntervalSince(g0) }, stateCache: stateCache)
        let total = Date().timeIntervalSince(g0)
        print(context.tokenizer.decode(tokenIds: toks))
        FileHandle.standardError.write(String(format: "\nengine generate[mtp K=%d]: ttft %.2f s, decode %.2f tok/s over %d tokens; rounds %d, accepted %d/%d drafts (%.2f), %.2f tokens/round, replays %d\n",
            K, ttft, Double(toks.count) / (total - ttft), toks.count, st.rounds, st.accepted, st.drafted, Double(st.accepted) / Double(max(1, st.drafted)), Double(st.committed) / Double(max(1, st.rounds)), st.replays).data(using: .utf8)!)
        FileHandle.standardError.write("  K histogram: \(st.kHistogram.sorted { $0.key < $1.key }.map { "K\($0.key)=\($0.value)" }.joined(separator: " "))  (adaptive: \(DepthController.enabled))\n".data(using: .utf8)!)
        if st.gated > 0 { FileHandle.standardError.write("  margin gate: \(st.gated) of \(st.rounds) rounds recomputed serially (\(String(format: "%.1f", 100 * Double(st.gated) / Double(max(1, st.rounds))))%)\n".data(using: .utf8)!) }
        if !st.marginHist.isEmpty {
            // rows by top-2 gap, in 0.1-logit buckets; bucket 0 is everything under 0.1 logits
            let tot = st.marginHist.values.reduce(0, +)
            let under = { (t: Int) in st.marginHist.filter { $0.key < t }.values.reduce(0, +) }
            let pct = { (n: Int) in String(format: "%.3f%%", 100 * Double(n) / Double(tot)) }
            FileHandle.standardError.write("  verify rows \(tot) by top-2 margin: <0.01 logit \(under(10)) \(pct(under(10))), <0.05 \(under(50)) \(pct(under(50))), <0.1 \(under(100)) \(pct(under(100))), <0.5 \(under(500)) \(pct(under(500))), <1.0 \(under(1000)) \(pct(under(1000)))\n".data(using: .utf8)!)
        }
        print("TOKENS:", toks.prefix(24).map(String.init).joined(separator: ","))
        return
    }
    let model = context.model
    let cache = model.newCache(parameters: nil)
    let t0 = Date()
    // P024: chunk the serial prefill. One block over the whole prompt builds an S x kv mask -- 3.2 GB
    // at 8k and 824 GB at 128k (the MTP path has chunked since P017); unchunked, `engine generate`
    // could not reach full context at all.
    let genChunk = try o.int("--prefill-step", Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "4096") ?? 4096)
    let xs = MLXArray(ids.map { Int32($0) })[.newAxis]
    var lastLogits = MLXArray(0)
    var gc0 = 0
    while gc0 < ids.count {
        let gc1 = min(ids.count, gc0 + max(1, genChunk))
        lastLogits = model(xs[0..., gc0 ..< gc1], cache: cache)
        if gc1 < ids.count { asyncEval(lastLogits) }
        gc0 = gc1
    }
    var y = lastLogits[0..., -1, 0...].argMax(axis: -1)
    eval(y)
    let ttft = Date().timeIntervalSince(t0)
    var out: [Int] = []
    let d0 = Date()
    // pipelined like `engine bench`: build + dispatch step i+1 (lazy on the pending token) before reading token i,
    // so the host's graph build overlaps the GPU (43.8 -> ~55 tok/s on a chat prompt)
    var pending = y
    for _ in 0 ..< maxTokens {
        engineGenCount = out.count
        let logits = model(pending[.newAxis], cache: cache)[0..., -1, 0...]
        // P078: the loop is pipelined one token ahead -- `pending` is sampled here and only read back
        // on the NEXT iteration -- so a `</think>` sampled at step i was still unseen when step i+1
        // applied the ramp, and the ramp then closed the block a SECOND time. MEASURED 1 run in 116:
        // "</think></think>", which in the API puts a stray tag at the head of `content`. Reading the
        // sampled id now costs the sync the next iteration was going to pay anyway, and only while
        // the ramp is live.
        let next = engineForcedClose() ?? engineSampler.sample(engineApplyThinkBias(logits))
        asyncEval(next)
        if engineThinkOpen, engineThinkBias.active, next.item(Int.self) == engineThinkCloseId { engineThinkOpen = false }
        let t = pending.item(Int.self)
        out.append(t)
        if t == engineThinkCloseId { engineThinkOpen = false }
        if engineStopIdSet.contains(t) || (context.tokenizer.eosTokenId.map { t == $0 } ?? false) { break }
        pending = next
    }
    let dec = Date().timeIntervalSince(d0)
    print(context.tokenizer.decode(tokenIds: out))
    FileHandle.standardError.write(String(format: "\nengine generate: ttft %.2f s (%.0f tok/s prefill), decode %.2f tok/s over %d tokens, peak %.1f GB\n",
        ttft, Double(ids.count) / ttft, Double(out.count) / dec, out.count, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
    print("TOKENS:", out.prefix(24).map(String.init).joined(separator: ","))
}

// MARK: - MTP speculative decoding (greedy, exact: every committed token is the trunk's own argmax)

struct SpecStats { var rounds = 0, drafted = 0, accepted = 0, committed = 0, replays = 0, gated = 0; var kHistogram: [Int: Int] = [:]; var prefillRows = 0; var prefillSeconds = 0.0; var marginHist: [Int: Int] = [:] }
/// P018: rows actually prefilled by the last serial run (continuation from a shorter snapshot) and their wall time; 0 = the whole prompt
nonisolated(unsafe) var prefillRowsCharged = 0
nonisolated(unsafe) var prefillContinuedSeconds = 0.0

/// Adaptive draft depth as an empirical bandit: per-K EMA of committed tokens per round divided by
/// the measured round cost (1.12 + 0.22*K decode steps); explores K+-1 every 8 rounds; switches on a
/// >3% expected gain. Acceptance is bursty (runs of accepted drafts), so a geometric model underrates
/// deep K; empirical tokens/round per K does not.
struct DepthController {
    let kMax: Int
    var k: Int
    var ema: [Double]
    var seen: [Int]
    var round = 0
    var mean: [Double]
    var n: [Int]
    static let enabled = ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPTIVE"] != nil   // three controllers measured below the best fixed K (OBS-ENG-019); opt-in
    /// P020 unit 25: the round cost per K, re-fitted from measured round times on this champion --
    /// 512: 16.6 + 4.3K ms, 8192: 18.9 + 4.3K, 262144: 20.8 + 5.5K, i.e. c1/c0 = 0.23-0.26 against the
    /// old model's 0.196. Understating the marginal draft cost is what made the UCB controller sit on
    /// K=4 at 512, where rate(3) and rate(4) are within 1% under the old constants and 2% apart under
    /// these. ENGINE_MTP_ADAPT_C0 / _C1 override.
    static let c0: Double = Double(ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_C0"] ?? "1.0") ?? 1.0
    static let c1: Double = Double(ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_C1"] ?? "0.25") ?? 0.25
    /// P020 unit 24 fixes, both default ON when the controller is enabled; =0 restores the old behaviour
    static let blendFirst: Bool = (ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_BLEND"] ?? "1") != "0"
    static let exploreAlways: Bool = (ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_EXPLORE"] ?? "1") != "0"
    static let explorePeriod: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_PERIOD"] ?? "16") ?? 16
    /// ENGINE_MTP_ADAPT_UCB=0 restores the EMA controller; ENGINE_MTP_ADAPT_C tunes the confidence width
    static let ucb: Bool = (ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_UCB"] ?? "1") != "0"
    static let ucbC: Double = Double(ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_C"] ?? "0.02") ?? 0.3
    static let n0: Int = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_ADAPT_N0"] ?? "0") ?? 16
    init(kMax: Int) {
        self.kMax = kMax; self.k = Self.enabled ? min(kMax, 3) : kMax
        // optimistic prior: assume the first draft accepts 0.8, geometric beyond
        ema = (0...kMax).map { K in (1 - pow(0.8, Double(K + 1))) / 0.2 }
        seen = Array(repeating: 0, count: kMax + 1)
        mean = (0...kMax).map { K in (1 - pow(0.8, Double(K + 1))) / 0.2 }   // the same optimistic prior, as one pseudo-sample
        n = Array(repeating: 1, count: kMax + 1)
        // P020 unit 26: anchor the prior on the SHIPPED default depth. Absent data K=3 is the best guess
        // (it wins at 512 and 8192), so give it N0 pseudo-samples at a rate that ties the runner-up: the
        // controller then stays there until real evidence moves it, which is what removes the learning
        // transient that cost 4.9% on a 128-token run. ENGINE_MTP_ADAPT_N0=0 restores the flat prior.
        let anchor = min(kMax, 3)
        if Self.n0 > 0 && anchor >= 1 {
            n[anchor] = Self.n0
            // a rate just above the best alternative prior, so the anchor holds without being unbeatable
            var rival = 0.0
            for K in 1...kMax where K != anchor { rival = max(rival, mean[K] / Self.cost(K)) }
            mean[anchor] = max(mean[anchor], rival * Self.cost(anchor) * 1.02)
        }
    }
    static func cost(_ K: Int) -> Double { c0 + c1 * Double(K) }
    func rate(_ K: Int) -> Double { ema[K] / Self.cost(K) }
    /// P020 unit 25: a UCB1 bandit over K. OBS-ENG-053 (3c) showed the EMA controller THRASHES -- it has
    /// no notion of confidence, so a 3% threshold on a bursty process makes the argmax jump (K1=24 K2=28
    /// K3=6 K4=22 where K=3 is best). Here each depth keeps a true MEAN and a visit count, the shallower
    /// depths are credited from every deeper round (free, and exact: running at d and accepting a tells
    /// you min(a,K)+1 for every K < d), and the choice is `rate + c*sqrt(ln round / n)`, so a depth is
    /// left only once its mean is known well enough to justify it.
    mutating func observeUCB(accepted a: Int, drafted d: Int) {
        round += 1
        func sample(_ K: Int, _ committed: Double) {
            n[K] += 1
            mean[K] += (committed - mean[K]) / Double(n[K])
        }
        sample(d, Double(a + 1))
        for K in 1 ..< d { sample(K, Double(min(a, K) + 1)) }
        var best = 1, bestScore = -1.0
        let lr = log(Double(max(round, 2)))
        for K in 1...kMax {
            let bonus = Self.ucbC * (n[K] > 0 ? sqrt(lr / Double(n[K])) : 1e3)
            let sc = mean[K] / Self.cost(K) + bonus
            if sc > bestScore { bestScore = sc; best = K }
        }
        k = best
    }
    mutating func observe(accepted a: Int, drafted d: Int) {
        guard Self.enabled, d >= 1, d <= kMax else { return }
        if Self.ucb { observeUCB(accepted: a, drafted: d); return }
        round += 1
        let committed = Double(a + 1)
        // P020 unit 24: BLEND the first observation instead of REPLACING the prior. Acceptance is bursty,
        // so one unlucky first round at the starting depth used to overwrite ema[d] outright and sink it
        // below every shallower depth -- which the credit loop below then kept refreshing every round while
        // the poisoned depth was explored at most three times ever. That asymmetry is why the controller
        // collapsed to a low K at every context (OBS-ENG-053 (3)).
        ema[d] = (Self.blendFirst || seen[d] > 0) ? 0.75 * ema[d] + 0.25 * committed : committed
        seen[d] += 1
        // credit shallower depths with what this round shows (a+1 committed of d drafts): min(a, K)+1
        for K in 1 ..< d { let c = Double(min(a, K) + 1); ema[K] = seen[K] == 0 ? c : 0.9 * ema[K] + 0.1 * c }
        var best = k
        for K in 1...kMax where rate(K) > 1.03 * rate(best) { best = K }
        if round % Self.explorePeriod == 0 {       // explore a neighbour -- ALWAYS, so a poisoned estimate recovers
            let up = (round / Self.explorePeriod) % 2 == 0        // deterministic, so timing is reproducible
            let cand = up ? min(kMax, best + 1) : max(1, best - 1)
            if Self.exploreAlways || seen[cand] < 3 { best = cand }
        }
        k = best
    }
}

/// One speculative generation. Returns (tokens, stats). `depth` = drafts per round (K).
/// P033 DEFECT FIX -- THE ENGINE NEVER STOPPED. The checkpoint's `generation_config.json` declares
/// **two** stop ids, `eos_token_id: [248046, 248044]`, and the chat-completion one is
/// `<|im_end|>` = 248046. The serial path checked only `tokenizer.eosTokenId` -- a SINGLE id, and not
/// that one -- and the SPECULATIVE path, which is the deployment mode, checked NONE. Measured: on
/// "Ile to jest 2 + 2?" the model answers "2 + 2 = 4" and emits `<|im_end|>` after ~60 tokens, and
/// the engine then ran to the full 3000-token budget, hallucinating an empty user turn and answering
/// it over and over -- 110 `<|im_end|>` and 110 `<|im_start|>` in ONE reply; another prompt reached
/// 664. That is what "the model overthinks" looks like from outside. It is ours, not the model's.
nonisolated(unsafe) var engineStopIdSet: Set<Int> = []

/// P033 unit 2 -- THE ENGINE COULD NOT SERVE THIS MODEL AS CONFIGURED. It had exactly one sampler,
/// `ArgMaxSampler()`, while the checkpoint's `generation_config.json` says
/// `do_sample: true, temperature 1.0, top_k 20, top_p 0.95`. This is that decoding.
/// DEFAULT IS GREEDY (temp = 0) so every existing measurement in this KB stays reproducible;
/// `--sample` adopts the checkpoint's own settings and `--temp/--top-p/--top-k` override them.
/// ORDER: temperature, then top-k, then top-p (nucleus) -- the order the reference stacks use. The
/// nucleus keeps the token that CROSSES the threshold (`cum - p < topP`), so top_p = 0 still leaves
/// exactly one token and never an empty support.
struct EngineSampler {
    nonisolated(unsafe) static var witnessed = false
    /// P042: restore the pre-unit-4 full-sort sampler, for a paired cost difference on one binary.
    static let fullSortArm: Bool = ProcessInfo.processInfo.environment["ENGINE_SAMPLER_FULLSORT"] == "1"
    var temp: Float = 0, topP: Float = 1, topK: Int = 0
    /// P087: a PER-REQUEST random key. `MLXRandom.categorical` without one draws from the process's
    /// global stream, which is fine while exactly one request exists and meaningless the moment two
    /// interleave: their draws alternate, so a client's `seed` no longer determines its own tokens.
    /// With a key the stream belongs to the request, and a seeded request is reproducible whatever
    /// else the server is doing. nil keeps the old global behaviour for the single-sequence tools.
    var key: MLXArray? = nil
    var isGreedy: Bool { temp <= 0 }
    /// split the request's key so successive draws differ; returns the key to use for this draw
    private mutating func nextKey() -> MLXArray? {
        guard let k = key else { return nil }
        let pair = MLXRandom.split(key: k)
        key = pair.0
        return pair.1
    }
    /// the row's next draw key, split off the request's stream (nil = the global stream) -- P095 U3-K
    mutating func drawKey() -> MLXArray? { nextKey() }
    mutating func sample(_ logits: MLXArray) -> MLXArray {
        if isGreedy { return logits.argMax(axis: -1) }
        var l = logits.asType(.float32) / temp
        let V = l.dim(-1)
        // P033 unit 4. THE FIRST VERSION COST 3.6% OF A ROUND IN THE SAMPLER ITSELF: a full `sorted`
        // for top-k AND a full `argSort` for top-p, both over a **248 320**-entry vocabulary, three
        // positions per round. Nothing outside the top k can survive either filter, so the whole
        // decision lives in k = 20 values: ONE O(V) `argPartition` lifts them out, everything after
        // is a 20-wide sort, and the sampled index is mapped back. Identical semantics -- temperature,
        // then top-k, then a nucleus that keeps the token CROSSING the threshold.
        // P042 unit 1: the PRE-unit-4 body, kept behind ENGINE_SAMPLER_FULLSORT=1 so unit 4's saving
        // can be measured as a DIFFERENCE on one binary instead of inferred from a code comment.
        // Faithful to what OBS-ENG-093 (4) priced: a full `sorted` over V for top-k, then the full
        // `argSort` top-p path below. Semantics are identical, so the two arms must emit the SAME
        // tokens under the same seed -- and that identity is the check that this is a cost comparison
        // and not two different programs.
        if Self.fullSortArm {
            if !EngineSampler.witnessed {
                EngineSampler.witnessed = true
                FileHandle.standardError.write(
                    "sampler: topK=\(topK) of V=\(V) -> FULL SORT path (pre-P033-unit-4)\n".data(using: .utf8)!)
            }
            if topK > 0 && topK < V {
                let asc = sorted(l, axis: -1)                                  // full V
                let kth = asc[.ellipsis, (V - topK) ..< (V - topK + 1)]         // the k-th largest
                l = MLX.which(l .< kth, MLXArray(-Float.infinity), l)
            }
        } else if topK > 0 && topK < V {
            // P042 ACTIVATION_WITNESS: this struct has two paths and only this one carries unit 4's
            // argPartition. Witness which ran rather than assume it from the flags.
            if !EngineSampler.witnessed {
                EngineSampler.witnessed = true
                FileHandle.standardError.write(
                    "sampler: topK=\(topK) of V=\(V) -> argPartition path (P033 unit 4), k-wide sort\n"
                        .data(using: .utf8)!)
            }
            let part = argPartition(-l, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]  // top-k ids, unordered
            var vals = takeAlong(l, part, axis: -1)                                      // (..., k)
            let ord = argSort(-vals, axis: -1)                                           // descending, k-wide
            vals = takeAlong(vals, ord, axis: -1)
            let ids = takeAlong(part, ord, axis: -1)
            if topP > 0 && topP < 1 {
                let p = softmax(vals, axis: -1)
                vals = MLX.which((p.cumsum(axis: -1) - p) .< MLXArray(topP), vals, MLXArray(-Float.infinity))
            }
            let pick = MLXRandom.categorical(vals, axis: -1, key: nextKey())             // index WITHIN k
            return takeAlong(ids, pick[.ellipsis, .newAxis], axis: -1).squeezed(axis: -1)
        }
        if topP > 0 && topP < 1 {                                                        // no top-k: full path
            let p = softmax(l, axis: -1)
            let idx = argSort(-p, axis: -1)
            let ps = takeAlong(p, idx, axis: -1)
            let keepSorted = (ps.cumsum(axis: -1) - ps) .< MLXArray(topP)
            var keep = MLXArray.zeros(keepSorted.shape, type: Bool.self)
            keep = putAlong(keep, idx, values: keepSorted, axis: -1)
            l = MLX.which(keep, l, MLXArray(-Float.infinity))
        }
        return MLXRandom.categorical(l, axis: -1, key: nextKey())
    }
}
nonisolated(unsafe) var engineSampler = EngineSampler()

// P033 unit 6 -- THE THINKING BUDGET, and it is the actual repair rather than a workaround.
// MEASURED: on an eight-part research question the template's DEFAULT `reasoning_effort = xhigh`
// produced **16 000 tokens without ever emitting `</think>`**, on dense attention and on sparse, at
// MTP and serially -- all four identical, so it is neither the QSA indexer nor speculative decoding.
// The same prompt at `medium` closes at 1109 thinking tokens and at `low` at 420, on the SAME
// attention path and the SAME decoding. What xhigh lacks is a stopping rule, so the serving layer
// supplies one: if `</think>` has not been emitted after `budget` generated tokens, the NEXT token
// is forced to be `</think>` (248069) and the model answers from there.
// IT SITS AT THE SAMPLE SITE ON PURPOSE, which makes it correct in BOTH paths with one mechanism:
// serially the forced id becomes the next input; in the verify it makes every predicted position the
// close token, so the round accepts zero drafts and commits exactly it, and the existing rollback
// machinery -- which already handles a zero-accept round -- keeps the caches consistent.
let engineThinkCloseId = 248069           // </think>

/// P077 SOFT STOP -- the alternative to truncating the thinking block.
///
/// MEASURED (P076): while the think block is open the model puts p <= 3e-5 on `</think>` at EVERY
/// position of a 19k chain, and 0.99996 at the single position where it decides to stop. It is not
/// hesitating; the stop token is simply never a contender until its self-imposed verification plan
/// ends. A hard budget therefore has to cut MID-DERIVATION, which is what makes an early cut risky.
///
/// This adds a bias to the `</think>` logit that RAMPS with the length of the thinking, and it is
/// self-targeting: at the ~50 positions per chain where the model already ranks `</think>` 2nd-10th
/// (they are sentence and paragraph ends, never mid-expression) a modest bias wins, while mid-derivation
/// positions sit thousands of ranks down and are untouched by the same bias. The model then closes at
/// ITS OWN next natural boundary instead of being cut, and no text is injected.
struct EngineThinkBias {
    var start = 0, full = 0, maxBias: Float = 0
    /// P078, and it is the AGENTIC guarantee: past `full` the ramp keeps climbing, quadratically, so
    /// that by `deadline` the close token outranks anything the model can produce and the block ENDS.
    /// It is still not a truncation -- the ramp reaches a coercive height gradually, so in practice the
    /// model closes at one of its own sentence boundaries thousands of tokens before the deadline
    /// (MEASURED). 0 = off, and then a chain can in principle run to the caller's token cap.
    var deadline = 0
    /// P076 measured the worst case: at mid-derivation positions `</think>` sits ~24 logits under the
    /// argmax. 40 is above every gap observed on the six chains, so the deadline is a real bound.
    static let deadlineTop: Float = 40
    func bias(_ generated: Int) -> Float {
        guard maxBias > 0, full > start else { return 0 }
        if generated <= start { return 0 }
        let f = Float(min(generated, full) - start) / Float(full - start)
        var b = maxBias * f
        if deadline > full, generated > full {
            let g = Float(min(generated, deadline) - full) / Float(deadline - full)
            b += g * g * EngineThinkBias.deadlineTop
        }
        return b
    }
    var active: Bool { maxBias > 0 && full > start }
}
nonisolated(unsafe) var engineThinkBias = EngineThinkBias()

/// Adds the ramped bias to the `</think>` logit while the think block is open. Nothing else is touched,
/// so with the knob unset the returned array is the input array.
func engineApplyThinkBias(_ logits: MLXArray) -> MLXArray {
    guard engineThinkOpen, engineThinkBias.active else { return logits }
    let b = engineThinkBias.bias(engineGenCount)
    guard b > 0 else { return logits }
    let l = logits
    let col = l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
    l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(b).asType(l.dtype)
    return l
}
/// The MTP verify block carries K+1 logit rows; row j continues a prefix of `generated0 + j` tokens,
/// so each row gets exactly the bias serial decoding would have applied at that position. Without this
/// the soft stop was silently INERT under `--mtp` (P078: the knob had no effect at any ramp height).
func engineApplyThinkBiasBlock(_ logits: MLXArray, generated0: Int) -> MLXArray {
    guard engineThinkOpen, engineThinkBias.active else { return logits }
    let rows = logits.dim(0)
    var bs = [Float](repeating: 0, count: rows)
    for j in 0 ..< rows { bs[j] = engineThinkBias.bias(generated0 + j) }
    guard bs.contains(where: { $0 > 0 }) else { return logits }
    let l = logits
    let col = l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)]
    l[0..., engineThinkCloseId ..< (engineThinkCloseId + 1)] = col + MLXArray(bs).reshaped([rows, 1]).asType(l.dtype)
    return l
}
let engineThinkOpenId  = 248068           // <think>
nonisolated(unsafe) var engineThinkBudget = 0        // HARD close. Default OFF -- it truncates.
// DEFAULT OFF, and that is a CORRECTION of my own default. MEASURED: on "sum the digits of 60!" a
// nudge at 4096 cut the chain at 4117 and the answer was WRONG; at 8192 it was right; with the nudge
// OFF the model thinks 11 793 tokens, TERMINATES ON ITS OWN and is right. And the prompt this guard
// was built for -- the eight-part CRISPRi question that "never closed" -- **closes on its own at
// 27 189 thinking tokens** and returns a complete answer. **There was no non-termination: every
// earlier observation of it was the caller's token cap.** So the cue is an opt-in LATENCY guard for
// someone who cannot wait, never a default, because on this model it can only cost correctness.
nonisolated(unsafe) var engineThinkNudgeAt = 0
nonisolated(unsafe) var engineThinkOpen = false
nonisolated(unsafe) var engineGenCount = 0
nonisolated(unsafe) var engineThinkForced = false
nonisolated(unsafe) var engineForceQueue: [Int] = []
nonisolated(unsafe) var engineNudgeIds: [Int] = []

/// P033 unit 7 -- THE NUDGE, and it replaces the budget because the budget was the wrong repair.
/// MEASURED, and it retires my own first reading: this is NOT a repetition loop. The runs that fail
/// to terminate under SAMPLING carry **2.5-3.3% 8-gram repetition and a 12.7-14.3% peak cycle-rate,
/// BELOW a healthy corpus whose worst cases are 14.8% and 18.8%** -- a correct, terminating LCS
/// answer repeats MORE than the failure does. Nothing is cycling. The model writes varied, coherent
/// text and simply never DECIDES to stop, because the template's `xhigh` instruction has no stopping
/// criterion. So the repair must restore the DECISION, not truncate the thought.
/// A short cue is injected into the thinking stream; the model then writes its own wrap-up and emits
/// `</think>` ITSELF. CONTROLLED: the same 4000-token thinking prefix WITHOUT the cue still runs to
/// 12 000 without closing, WITH it closes and answers in 9 126. Thinking is never cut.
func engineForcedClose() -> MLXArray? {
    if !engineForceQueue.isEmpty { return MLXArray([Int32(engineForceQueue.removeFirst())]) }
    guard engineThinkOpen, !engineThinkForced else { return nil }
    if engineThinkNudgeAt > 0, engineGenCount >= engineThinkNudgeAt, !engineNudgeIds.isEmpty {
        engineThinkForced = true                       // once per generation
        engineForceQueue = Array(engineNudgeIds.dropFirst())
        FileHandle.standardError.write("engine: think nudge at \(engineGenCount) tokens (</think> still unemitted)\n".data(using: .utf8)!)
        return MLXArray([Int32(engineNudgeIds[0])])
    }
    if engineThinkBudget > 0, engineGenCount >= engineThinkBudget {
        engineThinkForced = true; engineThinkOpen = false
        FileHandle.standardError.write("engine: HARD think budget \(engineThinkBudget) -- forcing </think>\n".data(using: .utf8)!)
        return MLXArray([Int32(engineThinkCloseId)])
    }
    return nil
}

func engineStopIds(_ modelDir: String) -> Set<Int> {
    var ids: Set<Int> = []
    if let d = try? Data(contentsOf: URL(fileURLWithPath: modelDir).appendingPathComponent("generation_config.json")),
       let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
        if let n = j["eos_token_id"] as? Int { ids.insert(n) }
        if let a = j["eos_token_id"] as? [Int] { ids.formUnion(a) }
    }
    return ids
}

func speculativeGenerate(model: Qwen4ExpModel, prompt: [Int], maxTokens: Int, depth K: Int,
                         onPrefill: (() -> Void)? = nil, snapshotURL: URL? = nil,
                         stateCache: StateCacheConfig? = nil) -> ([Int], SpecStats) {
    guard let mtp = model.mtp else { fatalError("model has no MTP head (run with ENGINE_MTP=1 and an MTP-packed checkpoint)") }
    let embed = model.model.embedTokens
    var cache = model.newCache(parameters: nil)
    let mtpCache = mtp.newCache()
    var stats = SpecStats()
    var ctl = DepthController(kMax: K)
    if !DepthController.enabled { ctl.k = K }
    if let url = snapshotURL, FileManager.default.fileExists(atPath: url.path) {
        // P017: saved prefill + prime state (trunk caches, MTP caches, n0, d1, Slast); the load is time-to-first-token
        let d = try! loadArrays(url: url)
        precondition(d["T"]!.item(Int32.self) == Int32(prompt.count), "snapshot \(url.lastPathComponent) was built for a different prompt length")
        precondition(d["d1"] != nil, "snapshot \(url.lastPathComponent) has no MTP state: build it with --mtp K --snapshot-dir")
        model.importCaches(cache, prefix: "trunk.", from: d)
        model.importCaches([mtpCache], prefix: "mtp.", from: d)
        var n0 = d["n0"]!.item(Int.self)
        var out: [Int] = [n0]
        var d1Arr = d["d1"]!, Slast = d["Slast"]!
        eval(cache.flatMap { $0.state }); eval(mtpCache.state); eval(d1Arr, Slast)
        onPrefill?()
        return speculativeLoop(model: model, mtp: mtp, embed: embed, cache: &cache, mtpCache: mtpCache, stats: &stats, ctl: &ctl,
                               n0: &n0, out: &out, d1Arr: &d1Arr, Slast: &Slast, maxTokens: maxTokens)
    }
    // --- prefill trunk in 512-row chunks (one block over the whole prompt builds S x kv masks/scores: 3.2 GB at 8k, 824 GB at 128k);
    //     the hidden states of every prompt row are kept for the MTP prime
    let x = MLXArray(prompt.map { Int32($0) })[.newAxis]
    let trunkChunk = Int(ProcessInfo.processInfo.environment["ENGINE_PREFILL_CHUNK"] ?? "4096") ?? 4096
    let primeChunk = Int(ProcessInfo.processInfo.environment["ENGINE_MTP_PRIME_CHUNK"] ?? "4096") ?? 4096
    var hParts: [MLXArray] = []
    var logits = MLXArray(0)
    var start = 0
    // P024: the longest stored prefix of THIS token stream. An exact-length hit carries n0/d1/Slast
    // and answers with no prefill at all; a rung hit restores both caches and the loop below prefills
    // only the rows the rung does not cover.
    if let sc = stateCache, let (u, M) = StateCache.lookup(sc, prompt), let d = StateCache.load(sc, prompt, M, u) {
        let t0 = Date()
        model.importCaches(cache, prefix: "trunk.", from: d)
        model.importCaches([mtpCache], prefix: "mtp.", from: d)
        eval(cache.flatMap { $0.state }); eval(mtpCache.state)
        if M == prompt.count, let dd1 = d["d1"], let dsl = d["Slast"], let dn0 = d["n0"] {
            var n0 = dn0.item(Int.self)
            var out: [Int] = [n0]
            var d1Arr = dd1, Slast = dsl
            eval(d1Arr, Slast)
            FileHandle.standardError.write(String(format: "engine: state cache HIT at %d/%d (exact) in %.2f s -- no prefill\n", M, prompt.count, Date().timeIntervalSince(t0)).data(using: .utf8)!)
            stats.prefillRows = 0; stats.prefillSeconds = Date().timeIntervalSince(t0)
            onPrefill?()
            return speculativeLoop(model: model, mtp: mtp, embed: embed, cache: &cache, mtpCache: mtpCache, stats: &stats, ctl: &ctl,
                                   n0: &n0, out: &out, d1Arr: &d1Arr, Slast: &Slast, maxTokens: maxTokens)
        }
        start = M
        FileHandle.standardError.write(String(format: "engine: state cache HIT at %d/%d in %.2f s -- prefilling %d rows\n", M, prompt.count, Date().timeIntervalSince(t0), prompt.count - M).data(using: .utf8)!)
    } else if stateCache != nil {
        FileHandle.standardError.write("engine: state cache MISS -- full prefill\n".data(using: .utf8)!)
    }
    if let url = snapshotURL, let (purl, M) = partialSnapshot(for: url, tokens: prompt.count) {
        // P018: continue from the largest shorter snapshot of the same prompt (trunk + MTP caches), prefill and prime rows M..<T
        let d = try! loadArrays(url: purl)
        precondition(d["d1"] != nil, "snapshot \(purl.lastPathComponent) has no MTP state: build it with --mtp K --snapshot-dir")
        model.importCaches(cache, prefix: "trunk.", from: d)
        model.importCaches([mtpCache], prefix: "mtp.", from: d)
        eval(cache.flatMap { $0.state }); eval(mtpCache.state)
        start = M
        FileHandle.standardError.write("engine: continuing prefill from \(purl.lastPathComponent): rows \(M)..<\(prompt.count)\n".data(using: .utf8)!)
    }
    let tPrefill0 = Date()
    var n0 = 0
    var mixed = MLXArray(0), S = MLXArray(0)
    if let sc = stateCache, trunkChunk == primeChunk {
        // P024: trunk chunk and MTP prime for the SAME rows in ONE iteration, so both caches can be
        // exported together at a rung. The chunk boundaries are identical to the two-phase form
        // below (same width, same start) and the MTP head reads only `hidden` and its own cache, so
        // this reorders two independent cache updates and nothing else. The state is bit-identical;
        // it is checked against the two-phase form by the generated head, not asserted.
        var c0 = start
        while c0 < prompt.count {
            let c1 = min(prompt.count, c0 + trunkChunk)
            let (lg, h) = model.forwardHidden(x[0..., c0 ..< c1], cache: cache)
            var toks = Array(prompt[(c0 + 1) ..< min(c1 + 1, prompt.count)])
            if c1 == prompt.count { n0 = lg[0..., -1, 0...].argMax(axis: -1).item(Int.self); toks.append(n0) }
            (mixed, S) = mtp(hidden: h, tokens: MLXArray(toks.map { Int32($0) })[.newAxis], embed: embed, cache: mtpCache)
            if c1 < prompt.count { asyncEval(S) }
            if c1 < prompt.count, c1 % sc.step == 0 {
                eval(cache.flatMap { $0.state }); eval(mtpCache.state)
                var d: [String: MLXArray] = ["T": MLXArray(Int32(c1))]
                model.exportCaches(cache, prefix: "trunk.", into: &d)
                model.exportCaches([mtpCache], prefix: "mtp.", into: &d)
                StateCache.write(sc, prompt, c1, d)
            }
            c0 = c1
        }
    } else {
        var c0 = start
        while c0 < prompt.count {
            let c1 = min(prompt.count, c0 + max(1, trunkChunk))
            let (lg, h) = model.forwardHidden(x[0..., c0 ..< c1], cache: cache)
            hParts.append(h); logits = lg
            if c1 < prompt.count { asyncEval(h) }
            c0 = c1
        }
        let H = hParts.count == 1 ? hParts[0] : concatenated(hParts, axis: 1)
        n0 = logits[0..., -1, 0...].argMax(axis: -1).item(Int.self)         // token at T
        let primeTokens = MLXArray((Array(prompt.dropFirst(start + 1)) + [n0]).map { Int32($0) })[.newAxis]   // tokens at p+1 for p in start..<T
        // prime the MTP head over the prompt in chunks (like the trunk prefill): one 4096-row block builds an S x kv sparse mask and an
        // S x nBlocks argPartition -- 1.0 s at 4k context (OBS-ENG-028); the chunked prime is charged to ttft by the eval below
        let Tp = H.dim(1)
        var p0 = 0
        while p0 < Tp {
            let p1 = min(Tp, p0 + max(1, primeChunk))
            (mixed, S) = mtp(hidden: H[0..., p0 ..< p1, 0...], tokens: primeTokens[0..., p0 ..< p1], embed: embed, cache: mtpCache)
            if p1 < Tp { asyncEval(S) }
            p0 = p1
        }
    }
    var out: [Int] = [n0]
    var d1Arr = model.draftToken(mixed[0..., -1, 0...])
    var Slast = S[0..., (S.dim(1) - 1)..., 0...]
    eval(d1Arr, Slast)                                   // the prime belongs to time-to-first-token, not to the decode rate
    stats.prefillRows = prompt.count - start; stats.prefillSeconds = Date().timeIntervalSince(tPrefill0)
    if let url = snapshotURL {
        var d: [String: MLXArray] = ["T": MLXArray(Int32(prompt.count)), "n0": MLXArray(Int32(n0)), "d1": d1Arr, "Slast": Slast]
        model.exportCaches(cache, prefix: "trunk.", into: &d)
        model.exportCaches([mtpCache], prefix: "mtp.", into: &d)
        try! save(arrays: d, url: url)
        FileHandle.standardError.write("engine: wrote prefill+prime snapshot \(url.lastPathComponent) (\(d.count) arrays)\n".data(using: .utf8)!)
    }
    // The exact-length entry answers a byte-identical repeat with no prefill at all, but it costs a
    // whole state write on EVERY turn -- 8.19 GB and 1.5 s at 262144, which measured 44% of the warm
    // TTFT itself. A question-answering deployment never repeats a prompt exactly, so it is opt-in.
    if let sc = stateCache, sc.writeExact {
        eval(cache.flatMap { $0.state }); eval(mtpCache.state)
        var d: [String: MLXArray] = ["T": MLXArray(Int32(prompt.count)), "n0": MLXArray(Int32(n0)), "d1": d1Arr, "Slast": Slast]
        model.exportCaches(cache, prefix: "trunk.", into: &d)
        model.exportCaches([mtpCache], prefix: "mtp.", into: &d)
        StateCache.write(sc, prompt, prompt.count, d)
    }
    if ProcessInfo.processInfo.environment["ENGINE_CLEAR_CACHE_AFTER_PREFILL"] != nil {
        let t0 = Date(); Memory.clearCache()
        FileHandle.standardError.write(String(format: "engine: cleared the MLX buffer cache after the prefill in %.1f ms\n", Date().timeIntervalSince(t0) * 1e3).data(using: .utf8)!)
    }
    if ProcessInfo.processInfo.environment["ENGINE_MTP_ROUND_TRACE"] != nil {
        FileHandle.standardError.write(String(format: "after prefill+prime: GPU active %.2f GB cache %.2f GB peak %.2f GB\n", Double(GPU.activeMemory) / 1e9, Double(GPU.cacheMemory) / 1e9, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!)
    }
    onPrefill?()
    return speculativeLoop(model: model, mtp: mtp, embed: embed, cache: &cache, mtpCache: mtpCache, stats: &stats, ctl: &ctl,
                           n0: &n0, out: &out, d1Arr: &d1Arr, Slast: &Slast, maxTokens: maxTokens)
}

/// The MTP decode loop after the prefill/prime (or after a snapshot load): drafts, verify, rollback, append.
func speculativeLoop(model: Qwen4ExpModel, mtp: Qwen4ExpMTP, embed: Embedding, cache: inout [KVCache], mtpCache: KVCache,
                     stats: inout SpecStats, ctl: inout DepthController, n0: inout Int, out: inout [Int],
                     d1Arr: inout MLXArray, Slast: inout MLXArray, maxTokens: Int) -> ([Int], SpecStats) {
    var rollbackAfterReprime = false
    if let nStr = ProcessInfo.processInfo.environment["ENGINE_MTP_DRAFT_BENCH"], let N = Int(nStr) {
        // instrument: N chained draft steps (MTP layer + head + argmax), GPU-bound loop, timed end to end
        let noHead = ProcessInfo.processInfo.environment["ENGINE_MTP_DRAFT_NOHEAD"] != nil
        eval(d1Arr, Slast)
        var tok = d1Arr; var Sl = Slast
        for rep in 0 ..< 3 {
            let t0 = Date()
            for _ in 0 ..< N {
                let (m, s2) = mtp(hidden: Sl, tokens: tok[.newAxis], embed: embed, cache: mtpCache)
                let last = m[0..., -1, 0...]
                tok = noHead ? (last[0..., 0] .> 0).asType(.int32) : model.draftToken(last)
                Sl = s2
            }
            eval(tok)
            let dt = Date().timeIntervalSince(t0)
            FileHandle.standardError.write(String(format: "draft bench: %d chained draft steps in %.1f ms = %.3f ms/step (noHead=%d) [rep %d]\n", N, dt * 1e3, dt * 1e3 / Double(N), noHead ? 1 : 0, rep).data(using: .utf8)!)
        }
        return (out, stats)
    }
    let specNoEarly = ProcessInfo.processInfo.environment["ENGINE_MTP_NO_EARLY"] != nil
    let specTiming = ProcessInfo.processInfo.environment["ENGINE_STEP_TIMING"] != nil
    let specTrace = ProcessInfo.processInfo.environment["ENGINE_MTP_ROUND_TRACE"] != nil
    let specReplay = ProcessInfo.processInfo.environment["ENGINE_MTP_REPLAY"] != nil
    let specCheck = ProcessInfo.processInfo.environment["ENGINE_MTP_CHECK"] != nil
    // P078 -- MTP EXACTNESS. The verify block (S = K+1 rows) and serial decode (S = 1) are the same
    // math in different GEMM shapes, so their logits differ by float noise. Where the trunk's top-1
    // and top-2 are closer together than that noise, the block's argmax is NOT serial's, and from
    // there the two trajectories are different texts. `ENGINE_MTP_MARGIN_GATE t` refuses to trust a
    // row whose margin is under t logits: the round is cut there and that one position is recomputed
    // in the S = 1 form, which is by construction what serial decoding would have produced.
    let specMarginGate = Float(ProcessInfo.processInfo.environment["ENGINE_MTP_MARGIN_GATE"] ?? "") ?? 0
    let specMarginDump = ProcessInfo.processInfo.environment["ENGINE_MTP_MARGIN_DUMP"] != nil
    // INSTRUMENT (P078): per round, compute row 0 BOTH ways from the same cache state -- once as a
    // serial S = 1 forward, once as row 0 of the K+1 verify block -- and report how far apart they
    // are and how often their argmax differs. This is the quantity the gate threshold has to clear.
    let specRow0 = ProcessInfo.processInfo.environment["ENGINE_MTP_ROW0_SERIAL"] != nil
    var row0Rounds = 0, row0Disagree = 0
    var row0MaxAbs: Float = 0, row0SumAbs: Float = 0
    var row0DisagreeMargins: [Float] = []
    defer {
        if specRow0, row0Rounds > 0 {
            FileHandle.standardError.write(String(format: "mtp row0 serial-vs-block over %d rounds: argmax differs %d (%.3f%%), mean |d logit| at the serial argmax %.2e, max %.2e; margins at disagreement (milli-logits): %@\n",
                row0Rounds, row0Disagree, 100 * Double(row0Disagree) / Double(row0Rounds),
                Double(row0SumAbs) / Double(row0Rounds), Double(row0MaxAbs),
                row0DisagreeMargins.prefix(40).map { String(format: "%.0f", $0 * 1000) }.joined(separator: ",")).data(using: .utf8)!)
        }
    }
    var specBuildNs = 0.0, specWaitNs = 0.0, specTailNs = 0.0
    var roundStart = Anchors.now()
    defer {
        if specTiming && stats.rounds > 0 {
            let n = Double(stats.rounds)
            FileHandle.standardError.write(String(format: "mtp round timing: host build (draft+verify graphs) %.2f ms  sync wait %.2f ms  tail (rollback+append build) %.2f ms  per round over %d rounds\n",
                                                  specBuildNs / n / 1e6, specWaitNs / n / 1e6, specTailNs / n / 1e6, stats.rounds).data(using: .utf8)!)
        }
    }
    while out.count < maxTokens {
        roundStart = Anchors.now()
        let T = (cache.first { $0 is CacheList } as! CacheList)[0].offset      // committed length incl. prompt
        let K = ctl.k
        stats.kHistogram[K, default: 0] += 1
        // --- draft chain: stays lazy on device (no per-draft sync)
        var draftArrs: [MLXArray] = [d1Arr]
        for _ in 1 ..< K {
            let (m, s2) = mtp(hidden: Slast, tokens: draftArrs.last![.newAxis], embed: embed, cache: mtpCache)
            draftArrs.append(model.draftToken(m[0..., -1, 0...]))
            Slast = s2
        }
        // start the draft chain on the GPU now, so it runs while the host builds the (much larger) verify graph
        if !specNoEarly { asyncEval(draftArrs.last!) }
        let tDraft = Anchors.now()
        if stats.rounds == 0, let sl = ProcessInfo.processInfo.environment["ENGINE_MTP_ROUND1_SLEEP"], let secs = Double(sl) {
            eval(draftArrs.last!)                        // drafts done; the profiler window starts after this sleep
            FileHandle.standardError.write("ROUND1_SLEEP_START\n".data(using: .utf8)!); Thread.sleep(forTimeInterval: secs)
            FileHandle.standardError.write("ROUND1_SLEEP_END\n".data(using: .utf8)!)
        }
        // --- verify: trunk over [n0, d1..dK]; ONE host sync per round (preds + drafts together)
        // the snapshot is only consumed by ENGINE_MTP_REPLAY / ENGINE_MTP_CHECK; holding references to every cache array forces the
        // verify's cache writes to allocate fresh buffers instead of updating in place (OBS-ENG-028: a ~1 s driver stall in round 1)
        let snap: Qwen4ExpModel.CacheSnapshot? = (specReplay || specCheck) ? model.snapshot(cache) : nil
        var row0Serial: (tok: Int, top1: Float, top2: Float)? = nil
        let snapRow0: Qwen4ExpModel.CacheSnapshot? = specRow0 ? model.snapshot(cache) : nil
        if specRow0 {
            // one S = 1 forward, then the SAME tape rollback the round already uses to undo rejected
            // rows (keeping a snapshot per round instead grows the cache arrays until the process dies)
            let (sl, _) = model.forwardHidden(MLXArray([Int32(n0)])[.newAxis], cache: cache)
            let r = sl[0, -1]
            let t1 = r.max().item(Float.self)
            let am = r.argMax().item(Int.self)
            let oh = MLXArray(Int32(0) ..< Int32(r.dim(-1))) .== MLXArray(Int32(am))
            let t2 = MLX.which(oh, MLXArray(-Float.infinity).asType(r.dtype), r).max().item(Float.self)
            row0Serial = (am, t1, t2)
            // DEFECT in my first cut of this instrument, and it invalidated its first numbers: the GDN
            // replay tape is only written for blocks (S > 1), so `rollback` after an S == 1 forward
            // silently leaves the recurrent state advanced and the block then ran from a CORRUPTED
            // state (mean |d| 0.59 logits, max 6.6 -- all of it the instrument's own damage).
            // A snapshot taken and released around the probe is the only correct undo here.
            model.restore(cache, snapRow0!)
        }
        let tSnap = Anchors.now()
        let block = concatenated([MLXArray([Int32(n0)])] + draftArrs, axis: 0)[.newAxis]
        let (vl, vh) = model.forwardHidden(block, cache: cache)
        // P033 unit 2: SPECULATIVE SAMPLING, and it is exact in distribution. Draw the target token at
        // every verify position, accept a draft only where it EQUALS that draw, and commit the draw
        // on the first mismatch: every committed token is then a genuine sample from the trunk's
        // distribution, exactly as every committed token was the trunk's argmax before. It is not the
        // OPTIMAL rejection scheme -- that one accepts more often -- so acceptance is expected to
        // fall, and measuring how far is the point.
        engineGenCount = out.count
        let forced = engineForcedClose()
        let vlb = engineApplyThinkBiasBlock(vl[0], generated0: out.count)
        let predsArr = (forced.map { MLXArray.full([vlb.dim(-2)], values: $0) }
                        ?? (engineSampler.isGreedy ? vlb.argMax(axis: -1)
                            : engineSampler.sample(vlb.reshaped(-1, vlb.dim(-1))))).asType(.int32)
        var both = concatenated([predsArr, concatenated(draftArrs, axis: 0)], axis: 0)
        let wantMargins = specMarginGate > 0 || specMarginDump
        if wantMargins {
            // top-2 gap per row, in milli-logits, carried in the one host sync the round already pays
            let V = vlb.dim(-1)
            let oneHot = MLXArray(Int32(0) ..< Int32(V))[.newAxis, 0...] .== predsArr[0..., .newAxis]
            let m1 = vlb.max(axis: -1)
            let m2 = MLX.which(oneHot, MLXArray(-Float.infinity).asType(vlb.dtype), vlb).max(axis: -1)
            both = concatenated([both, ((m1 - m2) * 1000).asType(.int32)], axis: 0)
        }
        let tBuilt = Anchors.now()
        if specTrace && stats.rounds < 2 { FileHandle.standardError.write(String(format: "  round %d phases: drafts+dispatch %.1f ms  snapshot %.1f ms  verify build %.1f ms\n", stats.rounds + 1, Double(tDraft - roundStart) / 1e6, Double(tSnap - tDraft) / 1e6, Double(tBuilt - tSnap) / 1e6).data(using: .utf8)!) }
        let bothHost = both.asArray(Int32.self).map { Int($0) }
        let tSynced = Anchors.now()
        if specTiming { specBuildNs += Double(tBuilt - roundStart); specWaitNs += Double(tSynced - tBuilt) }
        if specTrace { FileHandle.standardError.write(String(format: "round %3d K=%d T=%d build %.1f ms wait %.1f ms  [GPU active %.2f GB cache %.2f GB peak %.2f GB]\n", stats.rounds + 1, K, T, Double(tBuilt - roundStart) / 1e6, Double(tSynced - tBuilt) / 1e6, Double(GPU.activeMemory) / 1e9, Double(GPU.cacheMemory) / 1e9, Double(GPU.peakMemory) / 1e9).data(using: .utf8)!) }
        if let rs = row0Serial {
            let r = vl[0][0]                              // the UNBIASED block row 0, same as the serial row
            let b1 = r.max().item(Float.self)
            let bam = r.argMax().item(Int.self)
            let oh = MLXArray(Int32(0) ..< Int32(r.dim(-1))) .== MLXArray(Int32(bam))
            _ = MLX.which(oh, MLXArray(-Float.infinity).asType(r.dtype), r).max().item(Float.self)
            row0Rounds += 1
            // same TOKEN in both forms: the serial argmax's logit, block vs serial. (Comparing the
            // two forms' top-2 VALUES compares different tokens whenever the argmax moved.)
            let d = abs(r[rs.tok].item(Float.self) - rs.top1)
            row0SumAbs += d; row0MaxAbs = max(row0MaxAbs, d)
            if bam != rs.tok { row0Disagree += 1; row0DisagreeMargins.append(rs.top1 - rs.top2) }
        }
        let preds = Array(bothHost.prefix(K + 1))
        let drafts = Array(bothHost[(K + 1) ..< (2 * K + 1)])
        let marginsMilli = wantMargins ? Array(bothHost.suffix(K + 1)) : []
        if specMarginDump { for m in marginsMilli { stats.marginHist[min(max(m, 0), 2000), default: 0] += 1 } }
        var a = 0
        // P033 unit 7 DEFECT, found by reading the injected text: when a token is FORCED, every
        // verify position carries the SAME id, so a draft that happens to equal it gets ACCEPTED and
        // the forced token is committed twice -- the nudge came out as "the the final final answer
        // answer". A forced round must commit EXACTLY ONE token, so accept nothing.
        if forced == nil { while a < K && preds[a] == drafts[a] { a += 1 } }
        // P078: `</think>` inside the ACCEPTED prefix ends the round there. The rows after it were
        // biased as if the block were still open (the host cannot know where the close lands until
        // after the argmax), and committing them could emit a second `</think>` into the answer.
        // P094: ONLY when a bias actually reached those rows. With the ramp at zero (no soft stop, or
        // thinking shorter than `start`) the rows after the close are exactly what serial decoding
        // would have produced, and cutting them cost one extra round per reply -- the champion gate
        // (512, K=3, 128 tokens) read -2.5% for it while the committed text stayed identical.
        if engineThinkOpen, engineThinkBias.active,
           let ci = (0 ..< a).first(where: { drafts[$0] == engineThinkCloseId }),
           ((ci + 1) ... K).contains(where: { engineThinkBias.bias(out.count + $0) > 0 }) { a = ci }
        // P078 MARGIN GATE. Every committed token of the round is read off a block-form row; where the
        // top-2 gap is under the block-vs-serial noise, that row's argmax is not decidable in this
        // form. Cut the round at the first such row and recompute exactly that position serially --
        // one extra S = 1 forward, and the committed token is then serial's by construction.
        var bonus = preds[a]
        var gatedHidden: MLXArray? = nil
        if specMarginGate > 0, forced == nil,
           let j = (0 ... a).first(where: { Float(marginsMilli[$0]) / 1000 < specMarginGate }) {
            stats.gated += 1
            model.rollback(cache, blockRows: K + 1, keep: j)          // undo the block back to row j's input
            let tokIn = j == 0 ? n0 : drafts[j - 1]
            let (sl, sh) = model.forwardHidden(MLXArray([Int32(tokIn)])[.newAxis], cache: cache)
            let slb = engineApplyThinkBiasBlock(sl[0], generated0: out.count + j)
            let st = (engineSampler.isGreedy ? slb.argMax(axis: -1)
                      : engineSampler.sample(slb.reshaped(-1, slb.dim(-1)))).asType(.int32)
            a = j
            bonus = st.item(Int.self)
            gatedHidden = sh
        }
        let newTokens = Array(drafts.prefix(a)) + [bonus]
        stats.rounds += 1; stats.drafted += K; stats.accepted += a; stats.committed += newTokens.count
        ctl.observe(accepted: a, drafted: K)
        // --- trunk state: keep if everything accepted, else roll back and replay the accepted prefix
        let Hacc = vh                                                 // positions T..T+K (we use T..T+a)
        if gatedHidden != nil {
            // the gate already rolled the trunk back and forwarded the one serial row; nothing to undo
        } else if a < K {
            let env = ProcessInfo.processInfo.environment
            if env["ENGINE_MTP_REPLAY"] != nil {
                model.restore(cache, snap!); stats.replays += 1
                let replay = MLXArray(([n0] + Array(drafts.prefix(a))).map { Int32($0) })[.newAxis]
                _ = model.forwardHidden(replay, cache: cache)
            } else if env["ENGINE_MTP_CHECK"] != nil {
                // self-check: tape rollback vs snapshot+replay on the same caches
                model.rollback(cache, blockRows: K + 1, keep: a + 1)
                let tapeStates = model.snapshot(cache)
                model.restore(cache, snap!)
                let replay = MLXArray(([n0] + Array(drafts.prefix(a))).map { Int32($0) })[.newAxis]
                _ = model.forwardHidden(replay, cache: cache)
                let replayStates = model.snapshot(cache)
                var worst: [(Double, String)] = []
                for (li, (ta, ra)) in zip(tapeStates.arrays, replayStates.arrays).enumerated() {
                    for (si, (x, y)) in zip(ta, ra).enumerated() {
                        if x.shape != y.shape { worst.append((Double.infinity, "layer \(li) slot \(si) shape \(x.shape) vs \(y.shape)")); continue }
                        let d = abs(x.asType(.float32) - y.asType(.float32)).max().item(Float.self)
                        let m = abs(y.asType(.float32)).max().item(Float.self)
                        worst.append((Double(d) / Double(max(m, 1e-6)), "layer \(li) slot \(si) max|d| \(d) refmax \(m)"))
                    }
                }
                worst.sort { $0.0 > $1.0 }
                FileHandle.standardError.write("mtp-check round \(stats.rounds) K=\(K) keep=\(a + 1): offsets tape \(tapeStates.offsets.prefix(4)) replay \(replayStates.offsets.prefix(4)); worst: \(worst.prefix(4).map { $0.1 })\n".data(using: .utf8)!)
            } else if serialReprimeFirst {
                rollbackAfterReprime = true                           // H60: built after the head re-prime is queued
            } else {
                model.rollback(cache, blockRows: K + 1, keep: a + 1)  // tape-based, no trunk re-forward
            }
        }
        // --- MTP cache: drop the temporary draft entries, append true entries for T..T+a
        let mtpKV = (mtpCache as! CacheList)[0] as! KVCacheSimple
        let extra = mtpKV.offset - T
        if extra > 0 { _ = mtpKV.trim(extra) }
        let idxc = (mtpCache as! CacheList)[1] as! ArraysCache
        // indexer slot 0 is a capacity buffer whose logical length follows the KV offset (P018): no slice
        if !Qwen4ExpModel.indexerPooledBuffer, let p = idxc[1] { let r = model.configuration.text.indexerCompressRatio; if p.dim(1) > T / r { idxc[1] = p[0..., 0 ..< (T / r), 0...] } }
        let nextTokens = MLXArray(newTokens.map { Int32($0) })[.newAxis]   // tokens at p+1 for p in T..T+a
        // gated rounds take the accepted rows from the block and the LAST row from the serial forward
        let HaccUsed = gatedHidden.map { a == 0 ? $0 : concatenated([Hacc[0..., 0 ..< a, 0...], $0], axis: 1) }
                       ?? Hacc[0..., 0 ..< (a + 1), 0...]
        let (m2, s3) = mtp(hidden: HaccUsed, tokens: nextTokens, embed: embed, cache: mtpCache)
        d1Arr = model.draftToken(m2[0..., -1, 0...])
        asyncEval(d1Arr)                                  // dispatch the append + next d1 now; the round's sync comes with the next verify
        if rollbackAfterReprime { model.rollback(cache, blockRows: K + 1, keep: a + 1); rollbackAfterReprime = false }
        Slast = s3[0..., (s3.dim(1) - 1)..., 0...]
        // P033: a round commits up to K+1 tokens at once, so a stop id can land anywhere inside the
        // block. Truncate AT it and stop -- appending the whole block would emit tokens the model
        // placed after its own end-of-turn.
        // The thinking block closes the moment `</think>` is COMMITTED, wherever in the block it sits.
        // (This clearing sat inside the stop-id branch below by a bad patch anchor, so it only ever
        // fired at end-of-turn and the nudge then triggered in the middle of a finished ANSWER.)
        if newTokens.contains(engineThinkCloseId) { engineThinkOpen = false }
        if let cut = newTokens.firstIndex(where: { engineStopIdSet.contains($0) }) {
            out += newTokens[...cut]
            break
        }
        out += newTokens
        n0 = bonus
        if specTiming { specTailNs += Double(Anchors.now() - tSynced) }
    }
    return (Array(out.prefix(maxTokens)), stats)
}

/// P080 -- THE QUESTION CONTINUOUS BATCHING TURNS ON, asked before anything is built: on a 512-expert
/// MoE at decode, does putting B sequences in one forward amortise the expert-weight reads, or is the
/// step already compute-bound so a batch buys nothing? `engine batchprobe --batch B` prefills B rows
/// -- DIFFERENT slices of the corpus, so the rows route to different experts as real traffic would --
/// and then decodes S steps with a [B, 1] block, reporting aggregate and per-sequence tok/s.
/// It measures ONLY equal-length rows: the KV cache carries one shared `offset` for the batch, which
/// is exactly why ragged continuous batching is a build rather than a wiring job.
func batchProbe(_ o: Options) async throws {
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("batchprobe requires --model DIR") }
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    let promptFile = o.string("--prompt", "bench/prompts/english_longcopy_512.txt")
    let promptTokens = try o.int("--prompt-tokens", 2048)
    let B = try o.int("--batch", 1)
    let decodeSteps = try o.int("--decode", 64)
    let prefillStep = try o.int("--prefill-step", 4096)
    let repeats = try o.int("--repeats", 2)
    let prefillCapacityLimit = try o.int("--prefill-capacity-limit", 0)
    guard B > 0, promptTokens > 0, decodeSteps > 0, prefillStep > 0, prefillCapacityLimit >= 0 else {
        throw EngineError.invalid("batchprobe requires positive batch/prompt/decode/step and a nonnegative capacity limit")
    }
    if prefillCapacityLimit > 0 {
        guard o.values["--prefill-check"] != nil else {
            throw EngineError.invalid("--prefill-capacity-limit is only supported by --prefill-check")
        }
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        let text = config["text_config"] as? [String: Any] ?? config
        let native = text["max_position_embeddings"] as? Int ?? 0
        guard native > 0, prefillCapacityLimit <= native, promptTokens <= prefillCapacityLimit,
              decodeSteps <= prefillCapacityLimit - promptTokens else {
            throw EngineError.invalid("prefill-check prompt+decode must fit the explicit capacity limit and native context")
        }
    }
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    guard let model = context.model as? Qwen4ExpModel else { throw EngineError.invalid("batchprobe needs the qwen4_exp model") }
    FileHandle.standardError.write(String(format: "engine batchprobe: loaded, active %.1f GB\n", Double(GPU.activeMemory) / 1e9).data(using: .utf8)!)

    var text = try String(contentsOfFile: promptFile, encoding: .utf8)
    var all = context.tokenizer.encode(text: text, addSpecialTokens: false)
    // H20 isolated ownership diagnostic: fixed real Serve groups, not batch-versus-solo quality.
    // Dispatch before generic B disjoint-window expansion; this fixture needs only 15388 input tokens.
    if o.values["--ownership-fixed-schedule"] == "1" {
        guard B == 8, promptTokens == 8192, prefillStep == 1024, decodeSteps == 128 else {
            throw EngineError.invalid("H20 requires --batch 8 --prompt-tokens 8192 --prefill-step 1024 --decode 128")
        }
        let required = promptTokens + 7 * prefillStep + 4 * 7
        while all.count < required { text += "\n" + text; all = context.tokenizer.encode(text: text, addSpecialTokens: false) }
        let sourceBytes = try Data(contentsOf: URL(fileURLWithPath: promptFile))
        let sourceSHA = SHA256.hash(data: sourceBytes).map { String(format: "%02x", $0) }.joined()
        try h20FixedServeSchedule(o, model: model, corpus: Array(all.prefix(required)), corpusSHA256: sourceSHA)
        return
    }
    if o.values["--ownership-shared-boundary"] == "1" {
        let sharedRows = try o.int("--ownership-shared-rows", 8)
        guard (2...8).contains(sharedRows), B == sharedRows,
              [8192, 261600].contains(promptTokens), prefillStep == 1024, decodeSteps == 128 else {
            throw EngineError.invalid("H35 geometry mismatch")
        }
        while all.count < promptTokens + 1 { text += "\n" + text; all = context.tokenizer.encode(text: text, addSpecialTokens: false) }
        let sourceBytes = try Data(contentsOf: URL(fileURLWithPath: promptFile))
        let sourceSHA = SHA256.hash(data: sourceBytes).map { String(format: "%02x", $0) }.joined()
        try h35SharedBoundary(o, model: model, corpus: Array(all.prefix(promptTokens + 1)), corpusSHA256: sourceSHA)
        return
    }
    if o.values["--chunk-width-probe"] == "1" {
        while all.count < promptTokens + 1 { text += "\n" + text; all = context.tokenizer.encode(text: text, addSpecialTokens: false) }
        let sourceBytes = try Data(contentsOf: URL(fileURLWithPath: promptFile))
        let sourceSHA = SHA256.hash(data: sourceBytes).map { String(format: "%02x", $0) }.joined()
        try h50ChunkWidthProbe(o, model: model, corpus: Array(all.prefix(promptTokens + 1)), corpusSHA256: sourceSHA)
        return
    }
    if o.values["--ownership-shared-timing"] == "1" {
        let sharedRows = try o.int("--ownership-shared-rows", 8)
        guard sharedRows == 8, B == 8, (8192...261600).contains(promptTokens),
              prefillStep == 1024, decodeSteps == 128 else {
            throw EngineError.invalid("H43 geometry mismatch")
        }
        while all.count < promptTokens + 1 { text += "\n" + text; all = context.tokenizer.encode(text: text, addSpecialTokens: false) }
        let sourceBytes = try Data(contentsOf: URL(fileURLWithPath: promptFile))
        let sourceSHA = SHA256.hash(data: sourceBytes).map { String(format: "%02x", $0) }.joined()
        try h43SharedTiming(o, model: model, corpus: Array(all.prefix(promptTokens + 1)), corpusSHA256: sourceSHA)
        return
    }
    while all.count < promptTokens * B + B * 64 { text += "\n" + text; all = context.tokenizer.encode(text: text, addSpecialTokens: false) }
    // row r takes its own window, so the rows do not share a routing pattern
    var rows: [[Int32]] = []
    for r in 0 ..< B {
        let start = (r * promptTokens) % max(1, all.count - promptTokens)
        rows.append(all[start ..< (start + promptTokens)].map { Int32($0) })
    }
    let xsAll = MLXArray(rows.flatMap { $0 }).reshaped([B, promptTokens])

    // P106: isolate native batched PREFILL from the already-known batch-decode
    // rounding. Both arms materialize every chunk and decode each row alone.
    if o.values["--prefill-check"] != nil {
        // This probe runs serially on its one consumer task. The reference keeps
        // the existing uncapped layout; only the native candidate applies H7.
        Qwen4ExpCacheCapacity.configureGrowthLimit(nil)
        defer { Qwen4ExpCacheCapacity.configureGrowthLimit(nil) }
        var capacityWitness: [[String: Any]] = []
        func recordCapacity(_ cache: [KVCache], arm: String, row: Int, stage: String) {
            let describe: (MLXArray) -> [String: Any] = {
                ["shape": $0.shape, "dtype": String(describing: $0.dtype), "nbytes": $0.nbytes]
            }
            let layers: [[String: Any]] = cache.enumerated().compactMap { i, value in
                guard let list = value as? CacheList, let kv = list[0] as? KVCacheSimple,
                      let idx = list[1] as? ArraysCache else { return nil }
                return ["layer": i, "offset": kv.offset,
                        "kv_growth_limit": kv.capacityGrowthLimit.map { $0 as Any } ?? NSNull(),
                        "kv_capacity": [kv.rawKeys, kv.rawValues].compactMap { $0 }.map(describe),
                        "indexer_raw": idx[0].map { describe($0) as Any } ?? NSNull(),
                        "indexer_pooled": idx[1].map { describe($0) as Any } ?? NSNull()]
            }
            capacityWitness.append(["arm": arm, "row": row, "stage": stage,
                "applied_growth_limit": Qwen4ExpCacheCapacity.growthLimit.map { $0 as Any } ?? NSNull(),
                "layers": layers, "active_bytes": Memory.activeMemory, "peak_bytes": Memory.peakMemory])
        }
        func prefill(_ xs: MLXArray) -> ([KVCache], MLXArray, Double) {
            let cache = model.newCache(parameters: nil)
            var last = MLXArray(0)
            let start = Date()
            for lo in stride(from: 0, to: promptTokens, by: prefillStep) {
                last = model(xs[0..., lo ..< min(promptTokens, lo + prefillStep)], cache: cache)
                eval(last, cache.flatMap { $0.state })
                let done = min(promptTokens, lo + prefillStep)
                if done == promptTokens || done % 16384 == 0 {
                    FileHandle.standardError.write(Data("prefill-check B=\(xs.dim(0)) tokens=\(done)/\(promptTokens) active_bytes=\(Memory.activeMemory) peak_bytes=\(Memory.peakMemory)\n".utf8))
                }

            }
            return (cache, last[0..., -1, 0...].asType(.float32), Date().timeIntervalSince(start))
        }
        func decode(_ cache: [KVCache], _ logits: MLXArray) -> [Int] {
            var y = logits.argMax(axis: -1).item(Int.self)
            var ids = [y]
            for step in 1 ..< max(1, decodeSteps) {
                let lg = model(MLXArray([Int32(y)]).reshaped([1, 1]), cache: cache)
                y = lg[0..., -1, 0...].argMax(axis: -1).item(Int.self)
                ids.append(y)
                if (step + 1) % 16 == 0 || step + 1 == decodeSteps {
                    FileHandle.standardError.write(Data("decode-check tokens=\(step + 1)/\(decodeSteps) active_bytes=\(Memory.activeMemory)\n".utf8))
                }
            }
            return ids
        }
        var serialIDs: [[Int]] = [], serialLogits: [MLXArray] = []
        var serialSeconds = 0.0
        for r in 0 ..< B {
            let (cache, logits, seconds) = prefill(xsAll[r ..< r + 1])
            serialSeconds += seconds
            serialLogits.append(logits)
            recordCapacity(cache, arm: "serial_uncapped", row: r, stage: "prefill")
            serialIDs.append(decode(cache, logits))
            recordCapacity(cache, arm: "serial_uncapped", row: r, stage: "decode")
        }
        Qwen4ExpCacheCapacity.configureGrowthLimit(prefillCapacityLimit > 0 ? prefillCapacityLimit : nil)
        let (batched, logits, seconds) = prefill(xsAll)
        recordCapacity(batched, arm: "native_candidate", row: -1, stage: "prefill")
        var batchIDs: [[Int]] = [], errors: [Float] = []
        for r in 0 ..< B {
            let rowLogits = logits[r ..< r + 1]
            errors.append(abs(rowLogits - serialLogits[r]).max().item(Float.self))
            let cache = model.unstackRow(batched, slot: r, length: promptTokens,
                                        pooledBlocks: promptTokens / model.configuration.text.indexerCompressRatio)
            batchIDs.append(decode(cache, rowLogits))
            recordCapacity(cache, arm: "native_candidate", row: r, stage: "decode")
        }
        let rowTokenHashes = rows.map { row in
            row.withUnsafeBytes { bytes in SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined() }
        }
        let report: [String: Any] = ["instrument": "native-prefill-solo-decode-v1",
            "batch": B, "prompt_tokens": promptTokens, "chunk": prefillStep,
            "decode": decodeSteps, "serial_prefill_seconds": serialSeconds,
            "batch_prefill_seconds": seconds, "last_logit_max_abs": errors,
            "serial_ids": serialIDs, "batch_ids": batchIDs,
            "identical_rows": zip(serialIDs, batchIDs).filter { $0 == $1 }.count,
            "prefill_capacity_limit_requested": prefillCapacityLimit,
            "capacity_witness": capacityWitness,
            "prompt_row_token_sha256": rowTokenHashes,
            "prompt_row_token_encoding": "int32-little-endian",
            "prefill_projection_witness": model.prefillProjectionWitness(),
            "peak_bytes": GPU.peakMemory]
        print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), encoding: .utf8)!)
        return
    }

    // ---- P080 U0: THE LOCKSTEP INSTRUMENT.
    // The free-running test below lets each side commit its OWN argmax, so the first flip
    // desynchronises the two token histories and every later comparison measures two different
    // texts -- which is how one late divergence was mistaken for a verdict three times. Here both
    // sides are fed the SAME token and the SERIAL side steers, so the histories are identical by
    // construction and what is compared is numbers: per (row, step) the two argmaxes, the serial
    // top-2 margin, |d| at the serial argmax, and max|d| over the vocabulary -- the quantities
    // P078 used to settle the analogous MTP question.
    if o.values["--ragged"] != nil, o.values["--lockstep"] != nil {
        let steps = try o.int("--ragged-steps", 128)
        let stride = try o.int("--ragged-stride", 512)
        let arm = o.values["--arm"] ?? "sweep"
        var lens: [Int] = (o.values["--ragged-lengths"].map { $0.split(separator: ",").compactMap { Int($0) } })
            ?? (0 ..< B).map { promptTokens + $0 * stride }
        if arm == "order" { lens.reverse() }
        let nRows = lens.count
        precondition(all.count > lens.max()! + steps + 8, "corpus too short for these lengths")

        func prefillOne(_ n: Int) -> ([KVCache], Int32) {
            let cache = model.newCache(parameters: nil)
            let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
            var last = MLXArray(0); var c0 = 0
            while c0 < n {
                let c1 = min(n, c0 + max(1, prefillStep))
                last = model(ids[0..., c0 ..< c1], cache: cache)
                if c1 < n { asyncEval(last) }
                c0 = c1
            }
            let y = last[0..., -1, 0...].argMax(axis: -1)
            eval(y, cache.flatMap { $0.state })
            return (cache, y.item(Int32.self))
        }
        /// argmax, top-1 value, second-best value -- on one float32 row
        func top2(_ v: MLXArray) -> (am: Int, t1: Float, t2: Float) {
            let am = v.argMax().item(Int.self)
            let t1 = v.max().item(Float.self)
            let oh = MLXArray(Int32(0) ..< Int32(v.dim(-1))) .== MLXArray(Int32(am))
            let t2 = MLX.which(oh, MLXArray(-Float.infinity), v).max().item(Float.self)
            return (am, t1, t2)
        }

        struct LockstepResult {
            var comparisons = 0, disagreements = 0
            var sumAbsAtArgmax = 0.0, maxAbsAtArgmax: Float = 0
            var maxAbsAnywhere: Float = 0
            var marginsAtDisagreement: [Float] = []
            var perRowLogits: [[[Float]]] = []          // arm=partner/order: the batched row's top-1 values
            var firstDisagreement: [Int: Int] = [:]     // row -> step
        }

        /// One lockstep run over `useLens`: returns the comparison statistics. `trackTop1` records
        /// row `track`'s batched top-1 value per step, which is what the partner and order arms
        /// compare across two runs.
        func lockstep(_ useLens: [Int], track: Int? = nil) -> (LockstepResult, [Float]) {
            let nb = useLens.count
            var solo: [[KVCache]] = [], stackSrc: [[KVCache]] = [], tok: [Int32] = []
            for r in 0 ..< nb {
                let (c, y0) = prefillOne(useLens[r]); solo.append(c); tok.append(y0)
                let (c2, _) = prefillOne(useLens[r]); stackSrc.append(c2)
            }
            let pool = model.stackCaches(stackSrc)
            var at = useLens
            var res = LockstepResult()
            var tracked: [Float] = []
            for step in 0 ..< steps {
                let lgB = Qwen4ExpBatch.with(at) {
                    model(MLXArray(tok)[0..., .newAxis], cache: pool)[0..., -1, 0...]
                }.asType(.float32)
                eval(lgB)
                for b in 0 ..< nb {
                    let lgS = model(MLXArray([tok[b]])[0..., .newAxis], cache: solo[b])[0..., -1, 0...].asType(.float32)
                    eval(lgS)
                    let rowS = lgS[0], rowB = lgB[b]
                    let (aS, _, t2S) = top2(rowS)
                    let aB = rowB.argMax().item(Int.self)
                    let marginS = rowS[aS].item(Float.self) - t2S
                    let dAt = abs(rowB[aS] - rowS[aS]).item(Float.self)
                    let dMax = abs(rowB - rowS).max().item(Float.self)
                    res.comparisons += 1
                    res.sumAbsAtArgmax += Double(dAt)
                    res.maxAbsAtArgmax = max(res.maxAbsAtArgmax, dAt)
                    res.maxAbsAnywhere = max(res.maxAbsAnywhere, dMax)
                    if aB != aS {
                        res.disagreements += 1
                        res.marginsAtDisagreement.append(marginS)
                        if res.firstDisagreement[b] == nil { res.firstDisagreement[b] = step }
                    }
                    if let t = track, t == b { tracked.append(rowB[aS].item(Float.self)) }
                    tok[b] = Int32(aS)                 // THE SERIAL SIDE STEERS -- histories stay identical
                }
                at = at.map { $0 + 1 }
            }
            return (res, tracked)
        }

        func report(_ tag: String, _ r: LockstepResult) {
            let pct = 100 * Double(r.disagreements) / Double(max(1, r.comparisons))
            let mean = r.sumAbsAtArgmax / Double(max(1, r.comparisons))
            let margins = r.marginsAtDisagreement.prefix(40).map { String(format: "%.0f", $0 * 1000) }.joined(separator: ",")
            FileHandle.standardError.write(String(format:
                "ragged lockstep %@: comparisons %d, argmax differs %d (%.3f%%), mean |d| at the serial argmax %.3e, max %.3e, max anywhere %.3e; margins at disagreement (milli-logits): %@\n",
                tag, r.comparisons, r.disagreements, pct, mean, Double(r.maxAbsAtArgmax), Double(r.maxAbsAnywhere), margins).data(using: .utf8)!)
            for (row, st) in r.firstDisagreement.sorted(by: { $0.key < $1.key }) {
                FileHandle.standardError.write("    row \(row) first differs at step \(st)\n".data(using: .utf8)!)
            }
        }

        if arm == "partner" {
            // THE DECISIVE ARM: the same row, batched once with a LONGER partner and once with a
            // SHORTER one. A row's logits cannot depend on its partner's length unless a quantity
            // that is shared across the batch (nBlocksMax, kvLenMax, the pooled watermark, kcap)
            // is leaking into it -- which separates "shared-max defect" from "bf16 tie" in one run.
            let target = lens[0]
            let delta = try o.int("--partner-delta", 1024)
            let (rLong, tLong) = lockstep([target, target + delta], track: 0)
            let (rShort, tShort) = lockstep([target, max(2049, target - delta)], track: 0)
            report("partner-longer  lens=[\(target),\(target + delta)]", rLong)
            report("partner-shorter lens=[\(target),\(max(2049, target - delta))]", rShort)
            var diverged = -1
            var maxGap: Float = 0
            for i in 0 ..< min(tLong.count, tShort.count) {
                let g = abs(tLong[i] - tShort[i])
                maxGap = max(maxGap, g)
                if g != 0 && diverged < 0 { diverged = i }
            }
            let verdict = maxGap == 0 ? "PARTNER-INVARIANT (no shared-max leak)" : "PARTNER-DEPENDENT (shared-max leak)"
            FileHandle.standardError.write("  partner arm: max|d| between the two runs on the tracked row = \(maxGap), first at step \(diverged) -- \(verdict)\n".data(using: .utf8)!)
            print(String(format: "{\"arm\": \"partner\", \"target\": %d, \"delta\": %d, \"steps\": %d, \"max_gap\": %.6e, \"first_gap_step\": %d, \"partner_invariant\": %@}",
                target, delta, steps, Double(maxGap), diverged, maxGap == 0 ? "true" : "false"))
            return
        }

        // `--track-row N --track-out FILE` records row N's batched top-1 value (at the SERIAL
        // argmax) per step. The partner test is then two of these in SEPARATE processes, diffed:
        // running two lockstep passes inside one process traps (exit 133) once a dozen caches and
        // two pools are alive at once, and a fresh process per measurement is the more trustworthy
        // comparison anyway.
        let trackRow = (o.values["--track-row"].flatMap { Int($0) })
        let (res, tracked) = lockstep(lens, track: trackRow)
        if let out = o.values["--track-out"] {
            let body = "[" + tracked.map { String(format: "%.9e", $0) }.joined(separator: ",") + "]"
            try body.write(toFile: out, atomically: true, encoding: .utf8)
            FileHandle.standardError.write("  tracked row \(trackRow!) -> \(out) (\(tracked.count) steps)\n".data(using: .utf8)!)
        }
        report("\(arm) B=\(nRows) lens=\(lens)", res)
        let worstMargin = res.marginsAtDisagreement.map { abs($0) }.max() ?? 0
        // one bf16 ulp at logit magnitudes of 10-20 is 0.0625-0.125; above 0.25 it is not rounding
        let verdict = res.disagreements == 0 ? "IDENTICAL" : (worstMargin <= 0.25 ? "TIE" : "DEFECT")
        FileHandle.standardError.write("  verdict: \(verdict) (worst margin at a disagreement \(worstMargin) logits)\n".data(using: .utf8)!)
        print(String(format: "{\"arm\": \"%@\", \"batch\": %d, \"lengths\": %@, \"steps\": %d, \"comparisons\": %d, \"disagreements\": %d, \"mean_abs_at_argmax\": %.6e, \"max_abs_anywhere\": %.6e, \"worst_margin\": %.6f, \"verdict\": \"%@\"}",
            arm, nRows, String(describing: lens), steps, res.comparisons, res.disagreements,
            res.sumAbsAtArgmax / Double(max(1, res.comparisons)), Double(res.maxAbsAnywhere), Double(worstMargin), verdict))
        return
    }

    // ---- P080 M1: the RAGGED proof. B rows at DIFFERENT lengths in one decode step must produce,
    // token for token, what each row produces on its own. Nothing less is evidence.
    if o.values["--ragged"] != nil, let ks = o.values["--mtp"], let KB = Int(ks), KB >= 1, let qm = model as? Qwen4ExpModel {
        // P099 U2: BATCHED MTP rounds on the pool -- every row primed like a lone request, then K-draft rounds over all
        // rows at once (greedy), rolled back per row. Reports tokens/s, acceptance, ms per round, and each row's
        // committed stream against the lone-request MTP rounds of the same row (`speculativeGenerate`, greedy).
        let steps = try o.int("--ragged-steps", 64)                      // committed tokens per row to reach
        let stride = try o.int("--ragged-stride", 512)
        let lens: [Int] = (o.values["--ragged-lengths"].map { $0.split(separator: ",").compactMap { Int($0) } })
            ?? (0 ..< B).map { promptTokens + $0 * stride }
        precondition(lens.count == B, "--ragged-lengths must list one length per row")
        precondition(all.count > lens.max()! + 8, "corpus too short")
        var ref: [[Int]] = []
        for r in 0 ..< B {
            let (toks, st) = speculativeGenerate(model: qm, prompt: Array(all[0 ..< lens[r]]), maxTokens: steps + KB + 1, depth: KB)
            ref.append(toks)
            FileHandle.standardError.write("  solo row \(r) len \(lens[r]): \(toks.prefix(8)) acceptance \(String(format: "%.2f", Double(st.accepted) / Double(max(1, st.drafted))))\n".data(using: .utf8)!)
        }
        var caches: [[KVCache]] = []; var specs: [BatchRowSpec] = []; var pending: [Int] = []
        for r in 0 ..< B {
            let cache = model.newCache(parameters: nil)
            let (n0, sp) = primeRowSpec(qm, ids: Array(all[0 ..< lens[r]]), cache: cache, chunk: max(1, prefillStep))
            caches.append(cache); specs.append(sp); pending.append(n0)
        }
        let batched = qm.stackCaches(caches); qm.resetRaggedState()
        let pool = stackSpecPool(qm, specs)
        var lengths = lens
        var got: [[Int]] = (0 ..< B).map { [pending[$0]] }
        var rounds = 0, drafted = 0, accepted = 0
        let rowsIn = (0 ..< B).map { _ in BatchRoundRow() }
        let t0 = Date()
        while got.map({ $0.count }).min()! < steps + 1 {
            let out = runBatchedMTPRound(qm, caches: batched, spec: pool, pending: pending, lengths: lengths, K: KB, rows: rowsIn)
            for b in 0 ..< B {
                got[b] += out[b].tokens
                lengths[b] += out[b].accepted + 1
                pending[b] = out[b].tokens.last!
                drafted += KB; accepted += out[b].accepted
            }
            rounds += 1
        }
        let secs = Date().timeIntervalSince(t0)
        var identical = 0
        for b in 0 ..< B {
            let n = min(steps, got[b].count, ref[b].count)
            if Array(got[b].prefix(n)) == Array(ref[b].prefix(n)) { identical += 1 }
            else {
                let d = (0 ..< n).first { got[b][$0] != ref[b][$0] } ?? 0
                FileHandle.standardError.write("  ROW \(b) DIFFERS at token \(d): batched \(got[b][max(0, d - 2) ..< min(n, d + 4)]) vs solo \(ref[b][max(0, d - 2) ..< min(n, d + 4)])\n".data(using: .utf8)!)
            }
        }
        let committed = got.reduce(0) { $0 + $1.count - 1 }
        print(String(format: "{\"batched_mtp\": %d, \"batch\": %d, \"lengths\": %@, \"rounds\": %d, \"ms_per_round\": %.2f, \"acceptance\": %.3f, \"tokens_per_round\": %.2f, \"rows_identical_to_solo\": %d, \"decode_tps_total\": %.2f}",
            KB, B, String(describing: lens), rounds, secs / Double(rounds) * 1e3, Double(accepted) / Double(max(1, drafted)), Double(committed) / Double(max(1, rounds)), identical, Double(committed) / secs))
        return
    }
    if o.values["--ragged"] != nil, let vrs = o.values["--verify-rows"], let VS = Int(vrs), VS > 1 {
        // P099 (INST-ENG-043): the batched VERIFY block priced -- each step forwards VS teacher-forced tokens per
        // row (the row's own corpus continuation, so acceptance is 100% by construction and the step cost is what
        // is measured), no rollback. Row 0 at B = 1 is checked against the serial S = VS block on the same tokens.
        let steps = try o.int("--ragged-steps", 16)
        let stride = try o.int("--ragged-stride", 512)
        let lens: [Int] = (o.values["--ragged-lengths"].map { $0.split(separator: ",").compactMap { Int($0) } })
            ?? (0 ..< B).map { promptTokens + $0 * stride }
        precondition(lens.count == B, "--ragged-lengths must list one length per row")
        let maxLen = lens.max()!
        precondition(all.count > maxLen + VS * steps + 8, "corpus too short for the verify blocks")
        func prefill(_ n: Int) -> [KVCache] {
            let cache = model.newCache(parameters: nil)
            let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
            var last = MLXArray(0); var c0 = 0
            while c0 < n {
                let c1 = min(n, c0 + max(1, prefillStep))
                last = model(ids[0..., c0 ..< c1], cache: cache)
                if c1 < n { asyncEval(last) }
                c0 = c1
            }
            eval(last, cache.flatMap { $0.state })
            return cache
        }
        // the serial reference: row 0 alone, VS tokens per forward (the MTP verify block's own path)
        let refCache = prefill(lens[0])
        var refLogits: [MLXArray] = []
        var at0 = lens[0]
        for _ in 0 ..< steps {
            let blk = MLXArray(all[at0 ..< (at0 + VS)].map { Int32($0) }).reshaped([1, VS])
            let lg = model(blk, cache: refCache); eval(lg); refLogits.append(lg); at0 += VS
        }
        var caches: [[KVCache]] = []
        for r in 0 ..< B { caches.append(prefill(lens[r])) }
        let batched = model.stackCaches(caches)
        var at = lens
        var maxd: Float = 0
        var outs: [MLXArray] = []
        let t0 = Date()
        for st in 0 ..< steps {
            let rowsNow = at
            let blk = MLXArray((0 ..< B).flatMap { b in all[at[b] ..< (at[b] + VS)].map { Int32($0) } }).reshaped([B, VS])
            let lg = Qwen4ExpBatch.with(rowsNow) { model(blk, cache: batched) }
            asyncEval(lg); outs.append(lg)
            at = at.map { $0 + VS }
            if st == steps - 1 { eval(lg) }
        }
        let secs = Date().timeIntervalSince(t0)
        for (st, lg) in outs.enumerated() { maxd = max(maxd, abs(lg[0 ..< 1] - refLogits[st]).max().item(Float.self)) }
        print(String(format: "{\"verify_rows\": %d, \"batch\": %d, \"lengths\": %@, \"steps\": %d, \"ms_per_step\": %.2f, \"row0_maxd_vs_serial\": %.3e, \"tokens_per_s_teacher_forced\": %.2f}",
            VS, B, String(describing: lens), steps, secs / Double(steps) * 1e3, maxd, Double(B * VS * steps) / secs))
        return
    }
    if o.values["--ragged"] != nil {
        let steps = try o.int("--ragged-steps", 24)
        let stride = try o.int("--ragged-stride", 512)
        // P095: explicit lengths (equal lengths = the ragged machinery without the length spread)
        let lens: [Int] = (o.values["--ragged-lengths"].map { $0.split(separator: ",").compactMap { Int($0) } })
            ?? (0 ..< B).map { promptTokens + $0 * stride }
        precondition(lens.count == B, "--ragged-lengths must list one length per row")
        let maxLen = lens.max()!
        precondition(all.count > maxLen + 8, "corpus too short for the ragged lengths")
        func prefill(_ n: Int) -> ([KVCache], Int32) {
            let cache = model.newCache(parameters: nil)
            let ids = MLXArray(all[0 ..< n].map { Int32($0) }).reshaped([1, n])
            var last = MLXArray(0); var c0 = 0
            while c0 < n {
                let c1 = min(n, c0 + max(1, prefillStep))
                last = model(ids[0..., c0 ..< c1], cache: cache)
                if c1 < n { asyncEval(last) }
                c0 = c1
            }
            let y = last[0..., -1, 0...].argMax(axis: -1)
            eval(y, cache.flatMap { $0.state })
            return (cache, y.item(Int32.self))
        }
        // reference: every row on its own, the ordinary single-sequence program
        var ref: [[Int]] = []
        for r in 0 ..< B {
            let (cache, y0) = prefill(lens[r])
            var y = MLXArray([y0]); var toks: [Int] = []
            for _ in 0 ..< steps {
                toks.append(y.item(Int.self))
                y = model(y[0..., .newAxis], cache: cache)[0..., -1, 0...].argMax(axis: -1)
                eval(y)
            }
            ref.append(toks)
            FileHandle.standardError.write("  ref row \(r) len \(lens[r]): \(toks.prefix(8))\n".data(using: .utf8)!)
        }
        // batched: the same prefills, stacked, then ragged steps over all rows at once
        var caches: [[KVCache]] = []; var firsts: [Int32] = []
        for r in 0 ..< B { let (c, y0) = prefill(lens[r]); caches.append(c); firsts.append(y0) }
        let batched = model.stackCaches(caches)
        var y = MLXArray(firsts)
        var at = lens
        // P095: PIPELINED like the uniform loop -- `asyncEval` per step, the token history read once at
        // the end. The first form read `y` on the host every step (a full sync), so the "ragged tax"
        // it reported charged the model for the probe's own round trips.
        var ys: [MLXArray] = [y]
        let t0 = Date()
        for _ in 0 ..< steps {
            let rowsNow = at
            let lg = Qwen4ExpBatch.with(rowsNow) { model(y[0..., .newAxis], cache: batched)[0..., -1, 0...] }
            y = lg.argMax(axis: -1)
            asyncEval(y)
            ys.append(y)
            at = at.map { $0 + 1 }
        }
        eval(y)
        let secs = Date().timeIntervalSince(t0)
        var got: [[Int]] = Array(repeating: [], count: B)
        for st in ys.prefix(steps) { let cur = st.asArray(Int32.self); for b in 0 ..< B { got[b].append(Int(cur[b])) } }
        var bad = 0
        for b in 0 ..< B where got[b] != ref[b] {
            bad += 1
            let d = zip(got[b], ref[b]).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? 0
            FileHandle.standardError.write("  ROW \(b) len \(lens[b]) DIFFERS at step \(d): batched \(got[b].prefix(8)) vs serial \(ref[b].prefix(8))\n".data(using: .utf8)!)
        }
        print(String(format: "{\"ragged\": true, \"batch\": %d, \"lengths\": %@, \"steps\": %d, \"rows_identical\": %d, \"rows_differing\": %d, \"decode_tps_total\": %.2f}",
            B, String(describing: lens), steps, B - bad, bad, Double(B * steps) / secs))
        return
    }

    // P083 -- THE COST OF A VERIFY-SHAPED BLOCK AT B ROWS. MTP under batching lives or dies on one
    // number: does a [B, K+1] trunk forward cost about what a [B, 1] one costs (so the drafts ride
    // along free, as they do at B = 1), or does it cost K+1 times as much (so batching has already
    // spent the amortisation MTP wanted)? This forwards a [B, W] block repeatedly and reports the
    // wall time per step; nothing is sampled, because only the cost is in question.
    if let wStr = o.values["--verify-width"], let W = Int(wStr), W >= 1 {
        let cache = model.newCache(parameters: nil)
        var c0 = 0
        while c0 < promptTokens {
            let c1 = min(promptTokens, c0 + max(1, prefillStep))
            _ = model(xsAll[0..., c0 ..< c1], cache: cache)
            c0 = c1
        }
        eval(cache.flatMap { $0.state })
        let block = MLXArray((0 ..< (B * W)).map { Int32(1000 + $0 % 500) }).reshaped([B, W])
        var last = model(block, cache: cache); eval(last)                  // warm the shape
        let reps = max(4, decodeSteps / max(1, W))
        let t0 = Date()
        for _ in 0 ..< reps { last = model(block, cache: cache) }
        eval(last)
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(reps)
        print(String(format: "{\"batch\": %d, \"verify_width\": %d, \"ms_per_step\": %.3f, \"rows_per_step\": %d, \"ms_per_row\": %.4f, \"peak_gpu_gb\": %.2f}",
                     B, W, ms, B * W, ms / Double(B * W), Double(GPU.peakMemory) / 1e9))
        return
    }

    func once() -> (prefill: Double, decode: Double, head: [Int]) {
        let cache = model.newCache(parameters: nil)
        let t0 = Date()
        var last = MLXArray(0)
        var c0 = 0
        while c0 < promptTokens {
            let c1 = min(promptTokens, c0 + max(1, prefillStep))
            last = model(xsAll[0..., c0 ..< c1], cache: cache)
            if c1 < promptTokens { asyncEval(last) }
            c0 = c1
        }
        var y = last[0..., -1, 0...].argMax(axis: -1)      // (B,)
        eval(y)
        let prefillSec = Date().timeIntervalSince(t0)
        let d0 = Date()
        var headTokens: [Int] = []
        for s in 0 ..< decodeSteps {
            let lg = model(y[0..., .newAxis], cache: cache)[0..., -1, 0...]
            y = lg.argMax(axis: -1)
            if s == 0 { eval(y); headTokens = y.asArray(Int32.self).map { Int($0) } }
            asyncEval(y)
        }
        eval(y)
        return (prefillSec, Date().timeIntervalSince(d0), headTokens)
    }
    _ = once()                                              // warm the shapes; never reported
    GPU.resetPeakMemory()
    var best = (prefill: 0.0, decode: 0.0)
    for i in 0 ..< max(1, repeats) {
        let r = once()
        let pTps = Double(B * promptTokens) / r.prefill
        let dTps = Double(B * decodeSteps) / r.decode
        best.prefill = max(best.prefill, pTps); best.decode = max(best.decode, dTps)
        FileHandle.standardError.write(String(format: "  run %d: prefill %.0f tok/s, decode %.2f tok/s aggregate (%.2f per sequence), first tokens %@\n",
            i + 1, pTps, dTps, dTps / Double(B), String(describing: r.head.prefix(4))).data(using: .utf8)!)
    }
    print(String(format: "{\"batch\": %d, \"prompt_tokens\": %d, \"decode_steps\": %d, \"prefill_tps\": %.1f, \"decode_tps_total\": %.3f, \"decode_tps_per_seq\": %.3f, \"peak_gpu_gb\": %.2f}",
        B, promptTokens, decodeSteps, best.prefill, best.decode, best.decode / Double(B), Double(GPU.peakMemory) / 1e9))
}

/// engine ngram-ids: host-side n-gram row ids for a token list, from the model's config.json.
func ngramIds(_ o: Options) throws {
    guard let cfgDir = o.values["--config"], let idsArg = o.values["--ids"] else {
        throw EngineError.invalid("ngram-ids requires --config DIR --ids a,b,c [--prev p,q]")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: cfgDir).appendingPathComponent("config.json"))
    let cfg = try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data)
    let hash = Qwen4ExpNgramHash(cfg.text, pleLayerIndex: 0)
    let ids = idsArg.split(separator: ",").map { Int($0)! }
    let prev = (o.values["--prev"] ?? "").split(separator: ",").map { Int($0)! }
    let p = prev.isEmpty ? Array(repeating: cfg.text.eosTokenId, count: cfg.text.ngramSize - 1) : prev
    let rows = hash.rowIds(prev: p, ids: ids)
    let out: [String: Any] = ["multipliers": hash.multipliers, "vocab": hash.vocab, "offsets": hash.offsets,
                              "rows_per_shard": hash.rowsPerShard, "total_rows": hash.totalRows, "row_ids": rows]
    print(String(data: try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
}

// ENGINE default: let the encoder run a whole decode step ahead of the GPU (vendored MLX patch,
// OBS-ENG-020: 10 -> 64 in-flight command buffers = +24% serial decode). Must precede any MLX use.
if ProcessInfo.processInfo.environment["MLX_MAX_ACTIVE_TASKS"] == nil { setenv("MLX_MAX_ACTIVE_TASKS", "64", 1) }
if let gb = ProcessInfo.processInfo.environment["ENGINE_GPU_CACHE_LIMIT_GB"].flatMap(Double.init) {
    let before = GPU.cacheLimit
    GPU.set(cacheLimit: Int(gb * 1e9))
    FileHandle.standardError.write("engine: GPU cache limit \(before / 1_000_000) MB -> \(GPU.cacheLimit / 1_000_000) MB (memory limit \(GPU.memoryLimit / 1_000_000) MB)\n".data(using: .utf8)!)
}
if let gb = ProcessInfo.processInfo.environment["ENGINE_GPU_MEMORY_LIMIT_GB"].flatMap(Double.init) {
    GPU.set(memoryLimit: Int(gb * 1e9))
    FileHandle.standardError.write("engine: GPU memory limit -> \(GPU.memoryLimit / 1_000_000) MB\n".data(using: .utf8)!)
}

do {
    let o = try Options(CommandLine.arguments)
    _ = try StateCache.validatedMetallibIdentity(
        executable: URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath(),
        requested: ProcessInfo.processInfo.environment["MLXFAST_MLX_METALLIB"])
    switch o.verb {
    case "ngram-ids":
        try ngramIds(o)
    case "generate":
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await generateText(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    case "nll":
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await nllDump(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    case "logits":
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await logitsDump(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    case "batchprobe":
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await batchProbe(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    case "tokbench":
        try tokBench(o)
    case "kernelbench":
        try KernelBench.run(Array(CommandLine.arguments.dropFirst(2)))
    case "bench":
        // Top-level code is main-actor isolated; a plain Task {} would inherit
        // that and deadlock against the semaphore. Detach explicitly.
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await bench(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    case "serve":
        let sem = DispatchSemaphore(value: 0)
        let box = FailureBox()
        Task.detached { do { try await serve(o) } catch { box.error = error }; sem.signal() }
        sem.wait()
        if let e = box.error { throw e }
    default:
        throw EngineError.usage
    }
} catch {
    FileHandle.standardError.write("engine: \(error)\n".data(using: .utf8)!)
    exit(2)
}
