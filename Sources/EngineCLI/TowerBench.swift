// T-0181: `engine towerbench --model DIR --images a.png,b.png,... [--groups 1,2,4,8] [--reps 5]`
// Does running several pages through E9's vision tower in ONE call cost less per page than one call per page? The pages
// are preprocessed exactly as the server does it (one preprocess call per image, the server's pixel cap), their patch
// rows concatenated with one THW per image (the tower attends per image segment; its matmuls run over all rows). For each
// group size g the first g pages run as one visionFeatures call: median ms of --reps after one warm call, ms per page.
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import Tokenizers
import MLXVLM
import Qwen4Exp

func towerBench(_ o: Options) async throws {
    guard let modelPath = o.values["--model"], let list = o.values["--images"] else {
        throw EngineError.invalid("towerbench needs --model DIR --images a.png,b.png,... [--groups 1,2,4,8] [--reps 5]")
    }
    let paths = list.split(separator: ",").map(String.init)
    let groups = (o.values["--groups"] ?? "1,2,4,8").split(separator: ",").compactMap { Int($0) }.filter { $0 <= paths.count }
    let reps = try o.int("--reps", 5)
    let dir = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
    await LLMTypeRegistry.shared.registerModelType("qwen4_exp") { data in
        Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }
    await registerPlainQwen35()
    applyWiredLimit()
    let context = try await LLMModelFactory.shared.load(from: dir, using: #huggingFaceTokenizerLoader())
    eval(context.model)
    guard let m = context.model as? Qwen4ExpModel, m.hasVision else { throw EngineError.invalid("towerbench: no vision tower") }
    let pcfg = try enginePreprocessorConfig(dir.path, maxPixelsCap: serveVisionMaxPixels)
    let proc = Qwen3VLProcessor(pcfg, tokenizer: context.tokenizer)
    var pix: [MLXArray] = [], thws: [THW] = []
    for p in paths {
        guard let ci = CIImage(contentsOf: URL(fileURLWithPath: p)) else { throw EngineError.invalid("towerbench: cannot read \(p)") }
        let (px, th) = try proc.preprocess(images: [ci], processing: nil)
        eval(px); pix.append(px); thws.append(th)
    }
    let merge = pcfg.mergeSize
    print("engine towerbench: \(paths.count) pages, grid \(thws[0].t)x\(thws[0].h)x\(thws[0].w), \((thws[0].t * thws[0].h * thws[0].w) / (merge * merge)) visual tokens per page")
    // T-0189: --dtype-arms 1 -- the first page through the tower with float32 pixels (as served) and with bf16 pixels,
    // interleaved: ms per page of each, the features' dtype, and how far the bf16 features are from the fp32 ones
    if o.values["--dtype-arms"] == "1" {
        let saved = Qwen4ExpModel.visionInputBF16
        defer { Qwen4ExpModel.visionInputBF16 = saved }
        func run(_ bf: Bool) -> (Double, MLXArray) {
            Qwen4ExpModel.visionInputBF16 = bf
            let t0 = Date()
            let f = m.visionFeatures(pixels: pix[0], gridTHW: [thws[0]])!
            eval(f)
            return (Date().timeIntervalSince(t0) * 1e3, f)
        }
        _ = run(false); _ = run(true)
        var tf: [Double] = [], tb: [Double] = []
        var ff: MLXArray? = nil, fb: MLXArray? = nil
        for _ in 0 ..< reps { let a = run(false); tf.append(a.0); ff = a.1; let b = run(true); tb.append(b.0); fb = b.1 }
        tf.sort(); tb.sort()
        let x = ff!.asType(.float32), y = fb!.asType(.float32)
        let cosRows = (x * y).sum(axis: -1) / (sqrt((x * x).sum(axis: -1)) * sqrt((y * y).sum(axis: -1)) + 1e-12)
        let maxd = abs(x - y).max().item(Float.self), maxRef = abs(x).max().item(Float.self)
        print(String(format: "  dtype arms: fp32 pixels %.1f ms (features %@) | bf16 pixels %.1f ms (features %@) | bf16/fp32 %.3f",
                     tf[tf.count / 2], "\(ff!.dtype)" as NSString, tb[tb.count / 2], "\(fb!.dtype)" as NSString, tb[tb.count / 2] / tf[tf.count / 2]))
        print(String(format: "  dtype arms: feature cosine per row min %.6f mean %.6f, max|d| %.4f (max|ref| %.3f)",
                     cosRows.min().item(Float.self), cosRows.mean().item(Float.self), maxd, maxRef))
    }
    // T-0191: --rotary-arms 1 -- the first page with the vision rotary as legacy / hoist / compiled (Qwen3VLVision.rotaryMode),
    // interleaved, bf16 pixels (production): ms per page and each arm's max |d| from legacy (0 = bitwise the same)
    if o.values["--rotary-arms"] == "1" {
        let savedMode = Qwen3VLVision.rotaryMode, savedBF16 = Qwen4ExpModel.visionInputBF16
        defer { Qwen3VLVision.rotaryMode = savedMode; Qwen4ExpModel.visionInputBF16 = savedBF16 }
        Qwen4ExpModel.visionInputBF16 = true
        let arms = ["legacy", "hoist", "compiled"]
        func run(_ mode: String) -> (Double, MLXArray) {
            Qwen3VLVision.rotaryMode = mode
            let t0 = Date()
            let f = m.visionFeatures(pixels: pix[0], gridTHW: [thws[0]])!
            eval(f)
            return (Date().timeIntervalSince(t0) * 1e3, f)
        }
        for a in arms { _ = run(a) }
        var times: [String: [Double]] = [:], feats: [String: MLXArray] = [:]
        for _ in 0 ..< reps { for a in arms { let r = run(a); times[a, default: []].append(r.0); feats[a] = r.1 } }
        let ref = feats["legacy"]!.asType(.float32)
        for a in arms {
            let t = times[a]!.sorted(), med = t[t.count / 2]
            let d = abs(feats[a]!.asType(.float32) - ref).max().item(Float.self)
            print(String(format: "  rotary %@: %.1f ms per page (min %.1f), max|d| from legacy %.3e", a as NSString, med, t[0], d))
        }
    }
    // T-0191: --sdpa72-arms 1 (run with MLX_SDPA_FULL_D72=1): the tower's attention padded to head_dim 80 vs the native 72
    // kernel, interleaved, production tower settings (bf16 pixels, compiled rotary): ms per page and max|d| (expected 0)
    if o.values["--sdpa72-arms"] == "1" {
        // the Swift side may pick 72 only when the C++ side has the fused 72 kernel on: otherwise MLX falls to the unfused
        // attention that materialises [heads, N, N] (P039 took the host down that way)
        guard ProcessInfo.processInfo.environment["MLX_SDPA_FULL_D72"] == "1" else {
            throw EngineError.invalid("towerbench --sdpa72-arms needs MLX_SDPA_FULL_D72=1")
        }
        let saved72 = Qwen3VLVision.sdpa72, savedMode = Qwen3VLVision.rotaryMode, savedBF16 = Qwen4ExpModel.visionInputBF16
        defer { Qwen3VLVision.sdpa72 = saved72; Qwen3VLVision.rotaryMode = savedMode; Qwen4ExpModel.visionInputBF16 = savedBF16 }
        Qwen4ExpModel.visionInputBF16 = true; Qwen3VLVision.rotaryMode = "compiled"
        func run(_ on: Bool) -> (Double, MLXArray) {
            Qwen3VLVision.sdpa72 = on
            let t0 = Date()
            let f = m.visionFeatures(pixels: pix[0], gridTHW: [thws[0]])!
            eval(f)
            return (Date().timeIntervalSince(t0) * 1e3, f)
        }
        _ = run(false); _ = run(true)
        var t80: [Double] = [], t72: [Double] = []
        var f80: MLXArray? = nil, f72: MLXArray? = nil
        for _ in 0 ..< reps { let a = run(false); t80.append(a.0); f80 = a.1; let b = run(true); t72.append(b.0); f72 = b.1 }
        t80.sort(); t72.sort()
        let d = abs(f72!.asType(.float32) - f80!.asType(.float32)).max().item(Float.self)
        print(String(format: "  sdpa72 arms: padded-80 %.1f ms | native-72 %.1f ms | 72/80 %.3f | max|d| %.3e",
                     t80[t80.count / 2], t72[t72.count / 2], t72[t72.count / 2] / t80[t80.count / 2], d))
    }
    var perPage1 = 0.0
    for g in groups {
        let px = g == 1 ? pix[0] : concatenated(Array(pix[0 ..< g]), axis: 0)
        let th = Array(thws[0 ..< g])
        eval(px)
        func once() -> Double {
            let t0 = Date()
            guard let f = m.visionFeatures(pixels: px, gridTHW: th) else { return -1 }
            eval(f)
            return Date().timeIntervalSince(t0) * 1e3
        }
        _ = once()
        let ts = (0 ..< reps).map { _ in once() }.sorted()
        let med = ts[ts.count / 2], per = med / Double(g)
        if g == 1 { perPage1 = per }
        print(String(format: "  group %d: %.1f ms per call (min %.1f) = %.1f ms per page%@", g, med, ts[0], per,
                     perPage1 > 0 && g > 1 ? String(format: " (%.2fx of one page alone)", per / perPage1) as NSString : "" as NSString))
    }
}
