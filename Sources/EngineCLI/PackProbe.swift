// T-0180: `engine packprobe --model DIR [--prompt FILE] [--ns 2,4,6,8] [--lens 600,1200,2200] [--reps 4]`
// Lab bound for varlen multi-prompt prefill: N independent prompts of L tokens (distinct corpus windows, fresh caches)
// prefilled S = one forwardHidden per prompt (today's server: one prompt per prefill forward) against P = ONE
// forwardPacked over all N (Qwen4ExpMixed.swift). G1 per segment: last-row logits, pre-mixer stream and every cache
// array max |d| (0 = bit-identical). Timing: ABBA over --reps pairs, each arm on fresh caches, eval-synchronised; median
// ms of the whole arm -> ms per prompt token and the packed / solo ratio. The packed forward's wall time is also the
// first prompt's prefill latency under packing.
import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import Tokenizers
import Qwen4Exp

func packProbe(_ o: Options) async throws {
    guard let modelPath = o.values["--model"] else { throw EngineError.invalid("packprobe needs --model DIR") }
    let promptFile = o.values["--prompt"] ?? "bench/prompts/lalka_ctx256k.txt"
    let ns = (o.values["--ns"] ?? "2,4,6,8").split(separator: ",").compactMap { Int($0) }
    let lens = (o.values["--lens"] ?? "600,1200,2200").split(separator: ",").compactMap { Int($0) }
    let reps = try o.int("--reps", 4)
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    await registerPlainQwen35()
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    guard let qm = context.model as? Qwen4ExpModel else { throw EngineError.invalid("packprobe needs the qwen4_exp model") }
    let all = context.tokenizer.encode(text: try String(contentsOfFile: promptFile, encoding: .utf8), addSpecialTokens: false)
    setvbuf(stdout, nil, _IOLBF, 0)
    print("engine packprobe: corpus \(all.count) tokens, ENGINE_PREFILL_WIDTH_CANONICAL=\(ProcessInfo.processInfo.environment["ENGINE_PREFILL_WIDTH_CANONICAL"] ?? "-")")
    func maxd(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return -1 }
        return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    func cacheMaxd(_ a: [KVCache], _ b: [KVCache]) -> Float {
        var m: Float = 0
        for (x, y) in zip(a, b) {
            let sx = x.state, sy = y.state
            guard sx.count == sy.count else { return -1 }
            for (p, q) in zip(sx, sy) { let d = maxd(p, q); if d < 0 { return -1 }; m = max(m, d) }
        }
        return m
    }
    var window = 0
    for L in lens {
        for N in ns {
            guard all.count > (window + 2 * N) * L + L else { print("  N=\(N) L=\(L): corpus too short"); continue }
            let inputs: [MLXArray] = (0 ..< N).map { k in
                let c0 = (window + k) * L
                return MLXArray(all[c0 ..< (c0 + L)].map { Int32($0) }).reshaped([1, L])
            }
            window += N
            // ---- G1
            let cS = (0 ..< N).map { _ in qm.newCache(parameters: nil) }
            let solo = zip(inputs, cS).map { x, c in qm.forwardHidden(x, cache: c) }
            eval(solo.flatMap { [$0.0, $0.1] }, cS.flatMap { $0.flatMap { $0.state } })
            let cP = (0 ..< N).map { _ in qm.newCache(parameters: nil) }
            let packed = qm.forwardPacked(inputs, caches: cP)
            eval(packed.flatMap { [$0.logits, $0.hidden] }, cP.flatMap { $0.flatMap { $0.state } })
            var dl: Float = 0, dh: Float = 0, dc: Float = 0, agree = 0
            for k in 0 ..< N {
                dl = max(dl, maxd(solo[k].0, packed[k].logits)); dh = max(dh, maxd(solo[k].1, packed[k].hidden))
                dc = max(dc, cacheMaxd(cS[k], cP[k]))
                if solo[k].0.argMax(axis: -1).item(Int.self) == packed[k].logits.argMax(axis: -1).item(Int.self) { agree += 1 }
            }
            // ---- timing, ABBA
            func armS() -> Double {
                let cs = (0 ..< N).map { _ in qm.newCache(parameters: nil) }
                let t0 = Date()
                for (x, c) in zip(inputs, cs) { let (l, h) = qm.forwardHidden(x, cache: c); eval(l, h) }
                return Date().timeIntervalSince(t0) * 1e3
            }
            func armP() -> Double {
                let cs = (0 ..< N).map { _ in qm.newCache(parameters: nil) }
                let t0 = Date()
                let r = qm.forwardPacked(inputs, caches: cs); eval(r.flatMap { [$0.logits, $0.hidden] })
                return Date().timeIntervalSince(t0) * 1e3
            }
            _ = armS(); _ = armP()
            var ts: [Double] = [], tp: [Double] = []
            for r in 0 ..< reps { if r % 2 == 0 { ts.append(armS()); tp.append(armP()) } else { tp.append(armP()); ts.append(armS()) } }
            ts.sort(); tp.sort()
            let ms = ts[ts.count / 2], mp = tp[tp.count / 2], toks = Double(N * L)
            print(String(format: "  N=%d L=%4d | G1 logits max|d| %.3e hidden %.3e cache %.3e argmax %d/%d | solo %.1f ms (%.3f ms/tok, %.0f tok/s) packed %.1f ms (%.3f ms/tok, %.0f tok/s) | packed/solo %.3f",
                         N, L, dl, dh, dc, agree, N, ms, ms / toks, toks / ms * 1e3, mp, mp / toks, toks / mp * 1e3, mp / ms))
        }
    }
}
