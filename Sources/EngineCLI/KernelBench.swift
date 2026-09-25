import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Qwen4Exp

/// `engine kernelbench` -- kernel benches that run against the VENDORED MLX, not homebrew python mlx.
///
/// WHY THIS EXISTS (OBS-ENG-056 (1)): `lab/*.py` links stock libmlx, which contains neither
/// MLX_GATHER_QMM_WT nor gather_qmm_work. For the quantized GATHER family the lab and the engine are
/// DIFFERENT PROGRAMS, and a whole claim was framed on an env switch that is not even a symbol in the
/// library it measured. `lab/_engine_parity.py` now refuses those runs; this is where they move TO.
/// Everything here links exactly what CURRENT_CHAMPION links, so MLX_GATHER_QMM_WT / _BM are live.
enum KernelBench {
    static func timed(_ iters: Int = 9, _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        var ts: [Double] = []
        for _ in 0 ..< iters {
            StreamOrDevice.default.stream.synchronize()
            let t0 = Date()
            eval(body())
            StreamOrDevice.default.stream.synchronize()
            ts.append(Date().timeIntervalSince(t0))
        }
        ts.sort()
        return ts[ts.count / 2] * 1e3
    }

    /// counts -> a sorted expert-index vector, plus the tile and straddle statistics of that layout
    static func layout(_ counts: [Int], bm: Int) -> (idx: MLXArray, tiles: Int, straddles: Int, ne: Int) {
        var ids: [Int32] = []
        ids.reserveCapacity(counts.reduce(0, +))
        var tiles = 0, straddles = 0, ne = 0, running = 0
        for (e, c) in counts.enumerated() where c > 0 {
            ne += 1
            tiles += (c + bm - 1) / bm
            ids.append(contentsOf: repeatElement(Int32(e), count: c))
            running += c
            if running % bm != 0 { straddles += 1 }
        }
        if counts.last(where: { $0 > 0 }) != nil, running % bm != 0 { straddles -= 1 }   // last boundary is the end
        return (MLXArray(ids), tiles, straddles, ne)
    }

    static func run(_ args: [String]) throws {
        if args.contains("moe-decode") { return try moeDecode(args) }
        if args.contains("head") { return try headBench(args) }
        if args.contains("gather") { return gatherBench(args) }
        if args.contains("proj") { return try projBench(args) }
        if args.contains("gdn") { return try gdnBench(args) }
        if args.contains("topk") { return try topkBench(args) }
        if args.contains("idxscore") { return try idxScoreBench(args) }
        if args.contains("moe-rows") { return try moeRowsBench(args) }
        let bm = Int(ProcessInfo.processInfo.environment["MLX_GATHER_QMM_BM"] ?? "16") ?? 16
        let path = args.first(where: { $0.hasSuffix(".safetensors") })
            ?? "runs/flashnext/layer0_mlp.safetensors"
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        var W: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("switch_mlp.") {
            W[String(k.dropFirst("switch_mlp.".count))] = v
        }
        guard let w = W["gate_proj.weight"], let sc = W["gate_proj.scales"], let bi = W["gate_proj.biases"] else {
            throw EngineError.usage
        }
        let D = 2560, E = 512, K = 10, T = 4096, BM = bm
        let M = T * K
        MLXRandom.seed(1)
        let x = (MLXRandom.normal([M + E * BM, 1, D]) * 0.5).asType(.bfloat16)   // headroom for the PADDED arms
        eval(x)

        func bench(_ counts: [Int], _ name: String) {
            let L = layout(counts, bm: BM)
            let xx = x[0 ..< counts.reduce(0, +)]
            let ms = timed { gatherQuantizedMatmul(xx, w, scales: sc, biases: bi, rhsIndices: L.idx,
                                                   transpose: true, groupSize: 32, bits: 4, sortedIndices: true) }
            let rows = counts.reduce(0, +)
            setvbuf(stdout, nil, _IOLBF, 0)
            print(String(format: "  %-30s rows %6d  ne %4d  tiles %6d  strad %5d   %8.3f ms",
                         (name as NSString).utf8String!, rows, L.ne, L.tiles, L.straddles, ms))
        }

        print("engine kernelbench: VENDORED mlx  BM=\(BM)  MLX_GATHER_QMM_WT=\(ProcessInfo.processInfo.environment["MLX_GATHER_QMM_WT"] ?? "default(on)")")
        print("  \("case".padding(toLength: 30, withPad: " ", startingAt: 0))   rows / non-empty / tiles / straddles / time")

        // aligned reference: every count a multiple of BM -> tiles == M/BM and straddles == 0, for ANY skew
        var flat = [Int](repeating: 0, count: E)
        let unitsPer = (M / BM) / E
        for e in 0 ..< E { flat[e] = unitsPer * BM }
        flat[0] += M - flat.reduce(0, +)
        bench(flat, "aligned flat 512")

        // the real router, raw and padded -- the pair that isolates the straddle term
        for tag in ["0", "45"] {
            let p = "runs/rt.i32.\(tag)"
            guard let d = FileManager.default.contents(atPath: p) else { continue }
            var counts = [Int](repeating: 0, count: E)
            d.withUnsafeBytes { (b: UnsafeRawBufferPointer) in
                for v in b.bindMemory(to: Int32.self) where v >= 0 && Int(v) < E { counts[Int(v)] += 1 }
            }
            bench(counts, "REAL layer \(tag) raw")
            bench(counts.map { $0 == 0 ? 0 : ($0 + BM - 1) / BM * BM }, "REAL layer \(tag) PADDED")
        }
    }
    // MARK: P094 -- `engine kernelbench moe-decode [layer_mlp_e9.safetensors]`
    //
    // The DECODE expert path (rows <= 8 never reaches gather_qmm, OBS-ENG-130): the project's own
    // fused bodies, timed exactly as the engine dispatches them, at T in {1, 4, 8} on REAL routing.
    // Routing comes from runs/rt.i32.<layer> (the router's top-10 over a 4096-row prefill chunk,
    // written by ENGINE_DUMP_ROUTER): consecutive rows are one sequence's consecutive tokens (a verify
    // block), rows 512 apart stand in for a batch of different conversations. 32 index sets rotate
    // per call so the weight bytes are not served from the SLC. Reports, per case: distinct experts,
    // bytes that MUST be read (distinct x bank) and bytes the per-pair down lane reads (pairs x bank),
    // the median time of gate/up alone and of the full pair, and GB/s against the size-dependent
    // dependent roof (OBS-ENG-063, INST-ENG-014), which is a CEILING and never an attribution.
    static func moeDecode(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        func mark(_ m: String) { if ProcessInfo.processInfo.environment["ENGINE_KB_DEBUG"] != nil { FileHandle.standardError.write("kb: \(m)\n".data(using: .utf8)!) } }
        let path = args.first(where: { $0.hasSuffix(".safetensors") }) ?? "runs/flashnext/layer0_mlp_e9.safetensors"
        let layerTag = path.contains("layer45") ? "45" : "0"
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        var W: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("switch_mlp.") { W[String(k.dropFirst("switch_mlp.".count))] = v }
        func bank(_ n: String) throws -> (MLXArray, MLXArray, MLXArray) {
            guard let w = W[n + ".weight"], let s = W[n + ".scales"], let b = W[n + ".biases"] else { throw EngineError.usage }
            return (w, s, b)
        }
        let gate = try bank("gate_proj"), up = try bank("up_proj"), down = try bank("down_proj")
        mark("banks loaded \(gate.0.shape) \(gate.1.shape)")
        let words = gate.0.dim(2)                      // uint32 words per input row: D/4 at 8-bit, D/8 at 4-bit
        let bits = words == 640 ? 8 : 4
        let D = words * (bits == 8 ? 4 : 8)
        let I = gate.0.dim(1), E = gate.0.dim(0), K = 10
        precondition(D == 2560, "unexpected expert geometry")
        // bytes per expert per bank: weight words x4 + scales + biases (bf16)
        func bankBytes(_ b: (MLXArray, MLXArray, MLXArray)) -> Double { Double(b.0.dim(1) * b.0.dim(2) * 4 + 2 * b.1.dim(1) * b.1.dim(2) * 2) }
        let gateUpBytes = bankBytes(gate) + bankBytes(up), downBytes = bankBytes(down)
        guard let dump = FileManager.default.contents(atPath: "runs/rt.i32.\(layerTag)") else { throw EngineError.invalid("no runs/rt.i32.\(layerTag)") }
        let rows: [[Int32]] = dump.withUnsafeBytes { (b: UnsafeRawBufferPointer) -> [[Int32]] in
            let v = Array(b.bindMemory(to: Int32.self)); return stride(from: 0, to: v.count - K + 1, by: K).map { Array(v[$0 ..< $0 + K]) }
        }
        // the dependent-chain roof by read size (OBS-ENG-063 (1)), log-interpolated
        let roofMB: [(Double, Double)] = [(0.01, 2), (0.1, 18), (0.5, 72), (1, 119), (2, 201), (3.5, 283), (7, 405), (15, 539), (30, 593), (60, 602), (120, 593), (1000, 593)]
        func roof(_ mb: Double) -> Double {
            for i in 1 ..< roofMB.count where mb <= roofMB[i].0 {
                let (a, b) = (roofMB[i - 1], roofMB[i]); let t = (log(mb) - log(a.0)) / (log(b.0) - log(a.0)); return a.1 + t * (b.1 - a.1)
            }
            return 593
        }
        mark("dump rows \(rows.count)")
        print("engine kernelbench moe-decode: VENDORED mlx, layer \(layerTag), \(bits)-bit banks, E=\(E) K=\(K) D=\(D) I=\(I)  ENGINE_FUSE_MASK=\(Q4MoEDecodeBench.fuseMask)")
        print(String(format: "  bytes per expert: gate+up %.2f MB, down %.2f MB", gateUpBytes / 1e6, downBytes / 1e6))
        MLXRandom.seed(1)
        let NROT = 32
        for (T, pattern) in [(1, "seq"), (4, "seq"), (8, "seq"), (4, "far"), (8, "far")] {
            let x = (MLXRandom.normal([T, D]) * 0.5).asType(.bfloat16)
            var sets: [MLXArray] = []; var distinctSum = 0
            for r in 0 ..< NROT {
                let base = 100 + r * 97
                let picked: [[Int32]] = (0 ..< T).map { t in rows[(base + (pattern == "seq" ? t : t * 512)) % rows.count] }
                distinctSum += Set(picked.flatMap { $0 }).count
                sets.append(MLXArray(picked.flatMap { $0 }).reshaped([T, K]))
            }
            eval(x, sets)
            mark("T=\(T) \(pattern) sets built")
            let distinct = Double(distinctSum) / Double(NROT), pairs = Double(T * K)
            // A DEPENDENT chain of NROT calls in one eval, each call's input carrying a zero-weighted
            // scalar of the previous output, so the kernels serialise as they do across layers and
            // the sync/eval floor (~0.3 ms per eval here) is paid once, not per call. The chain adds
            // one tiny elementwise hop per call (~5 us, OBS-ENG-063 (2)); it is reported, not hidden.
            func chain(_ full: Bool) -> MLXArray {
                var xi = x
                var last = x
                for i in 0 ..< NROT {
                    let r = Q4MoEDecodeBench.run(x: xi, idx: sets[i], gate: gate, up: up, down: down, bits: bits, K: K, D: D, I: I)
                    last = full ? r.yk : r.h
                    xi = x + (last[0, 0, 0 ..< 1] * 0).asType(x.dtype)
                }
                return last
            }
            let msGU = timed(5) { chain(false) } / Double(NROT)
            mark("T=\(T) gate/up timed \(msGU)")
            let msAll = timed(5) { chain(true) } / Double(NROT)
            let msDown = max(0, msAll - msGU)
            let guMB = distinct * gateUpBytes / 1e6, dnMustMB = distinct * downBytes / 1e6, dnReadMB = pairs * downBytes / 1e6
            print(String(format: "  T=%d %s  distinct %.2f of %d pairs | gate/up %.1f us = %.0f GB/s over %.0f MB (roof %.0f, %.0f%%) | down %.1f us: reads %.0f MB = %.0f GB/s, must-read %.0f MB (roof %.0f, %.0f%%) | pair %.1f us",
                         T, (pattern as NSString).utf8String!, distinct, Int(pairs),
                         msGU * 1e3, guMB / msGU, guMB, roof(guMB), 100 * (guMB / msGU) / roof(guMB),
                         msDown * 1e3, dnReadMB, dnReadMB / max(msDown, 1e-6), dnMustMB, roof(dnReadMB), 100 * (dnMustMB / max(msDown, 1e-6)) / roof(dnReadMB),
                         msAll * 1e3))
            FileHandle.standardError.write("    dispatch: \(Q4MoEDecodeBench.witness(T: T, bits: bits))\n".data(using: .utf8)!)
        }
    }

    // MARK: P095 -- `engine kernelbench --case head --file runs/flashnext/lm_head_e9.safetensors`
    // The output head (248320 x 2560, 8-bit g64, 675 MB) at M rows as MLX dispatches it: the vector
    // kernel (one grid column per row, the weight streamed per row) below 12 rows, the tiled qmm from
    // 12; plus the padded-to-12 form the U3-B arm uses. Dependent chain of 16 calls per timing.
    static func headBench(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let path = args.first(where: { $0.hasSuffix(".safetensors") }) ?? "runs/flashnext/lm_head_e9.safetensors"
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        guard let w = raw["lm_head.weight"], let sc = raw["lm_head.scales"], let bi = raw["lm_head.biases"] else { throw EngineError.usage }
        let N = w.dim(0), K = w.dim(1) * 4
        let bytes = Double(w.size * 4 + sc.size * 2 + bi.size * 2)
        print(String(format: "engine kernelbench head: VENDORED mlx, N=%d K=%d, %.0f MB", N, K, bytes / 1e6))
        MLXRandom.seed(1)
        let NCH = 16
        for M in [1, 2, 4, 8, 12, 16, 32] {
            let x = (MLXRandom.normal([M, K]) * 0.5).asType(.bfloat16); eval(x)
            func chain(pad: Int) -> MLXArray {
                var xi = x; var last = x
                for _ in 0 ..< NCH {
                    let xin = pad > M ? concatenated([xi, MLXArray.zeros([pad - M, K], dtype: .bfloat16)], axis: 0) : xi
                    let y = quantizedMatmul(xin, w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8)[0 ..< M]
                    last = y
                    xi = x + (y[0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16)
                }
                return last
            }
            // equality: the M-row call against M single-row calls (M = 1 is always the stock kernel)
            let yM = quantizedMatmul(x, w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8)
            let yRows = concatenated((0 ..< M).map { quantizedMatmul(x[$0 ..< ($0 + 1)], w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8) }, axis: 0)
            let maxd = abs(yM.asType(.float32) - yRows.asType(.float32)).max().item(Float.self)
            let ms = timed(5) { chain(pad: 0) } / Double(NCH)
            let msPad = M < 12 ? timed(5) { chain(pad: 12) } / Double(NCH) : ms
            print(String(format: "  M=%2d  as dispatched %7.3f ms = %4.0f GB/s (weight once) | padded to 12 rows %7.3f ms  | per-row-stream floor %6.3f ms at 602 GB/s x M | max|d| vs single-row calls %.3e",
                         M, ms, bytes / ms / 1e6, msPad, bytes * Double(M) / 602e9 * 1e3, maxd))
        }
    }

    // MARK: P095/P097 -- `engine kernelbench --case gather`: the ragged QSA gather priced alone. Synthetic bf16
    // K/V at the engine's geometry (2 kv heads x 256, 24 query heads), rows at base + 512 i, 512 selected blocks
    // of 4 drawn uniformly from each row's own blocks; dependent chain of 16 calls. Arms: the single kernel
    // (one 1024-thread group per (row, head)) and the partitioned GQA-shared kernel + merge at P x HPG, each
    // with max|d| against the single kernel (the partition changes the online-softmax order: sub-ulp, never 0).
    static func gatherBench(_ args: [String]) {
        setvbuf(stdout, nil, _IOLBF, 0)
        let H = 24, KVH = 2, D = 256, R = 4, K = 512, NCH = 16
        let arms: [(Int, Int)] = [(4, 1), (4, 2), (8, 1), (8, 2), (16, 1), (16, 2), (32, 1), (32, 2)]
        print("engine kernelbench gather: VENDORED mlx, H=\(H) KVH=\(KVH) D=\(D) R=\(R) K=\(K), knobs \(Q4GatherBench.raggedKnobs); arms P x HPG \(arms)")
        MLXRandom.seed(2)
        // P099: the serial S = 1 kernel (the lone request's decode step) with the pipelined chain, at 8192 / 32768 / 131072
        for kvLen in [8192, 32768, 131072, 262144] {
            let offset = kvLen - 1, nBlocks = (offset + 1) / R, kcapS = ((kvLen + 511) / 512) * 512
            let k = (MLXRandom.normal([1, KVH, kcapS, D]) * 0.1).asType(.bfloat16)
            let v = (MLXRandom.normal([1, KVH, kcapS, D]) * 0.1).asType(.bfloat16)
            let q = (MLXRandom.normal([1, H, 1, D]) * 0.1).asType(.bfloat16)
            let topsS = (0 ..< NCH).map { _ in MLXArray(Array(0 ..< nBlocks).shuffled().prefix(K).sorted().map { Int32($0) }).reshaped([1, 1, K]) }
            let top = topsS[0]
            eval(k, v, q, topsS)
            func chainS(_ body: (MLXArray, MLXArray) -> MLXArray) -> MLXArray {
                var qi = q; var last = q
                for c in 0 ..< NCH { last = body(qi, topsS[c]); qi = q + (last[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16) }
                return last
            }
            let shipped: (MLXArray, MLXArray) -> MLXArray = { Q4GatherBench.serialPipe(q: $0, keys: k, values: v, top: $1, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: R, scale: 1.0 / 16, pf: 0) }
            let refS = shipped(q, top).asType(.float32)
            var line = fmt("  serial S=1 kvLen=%d (rotating sets): shipped %6.1f us", kvLen, timed(5) { chainS(shipped) } / Double(NCH) * 1e3)
            for pf in [2, 4, 8] {
                let body: (MLXArray, MLXArray) -> MLXArray = { Q4GatherBench.serialPipe(q: $0, keys: k, values: v, top: $1, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: R, scale: 1.0 / 16, pf: pf) }
                let d = abs(body(q, top).asType(.float32) - refS).max().item(Float.self)
                line += fmt(" | pipe PF%d %6.1f (%.1e)", pf, timed(5) { chainS(body) } / Double(NCH) * 1e3, d)
            }
            print(line)
        }
        // P098: the MTP verify block (S = K+1 = 4 rows of ONE sequence, serial): the position-loop part kernel at PARTS = 16
        // with HPG heads per simdgroup (bit-identical: max|d| must read 0) against the shipped single-head form
        for kvLen in [8192, 32768] {
            let S = 4, offset = kvLen - S, nBlocks = (offset + 1) / R, kcapV = ((kvLen + 511) / 512) * 512
            let k = (MLXRandom.normal([1, KVH, kcapV, D]) * 0.1).asType(.bfloat16)
            let v = (MLXRandom.normal([1, KVH, kcapV, D]) * 0.1).asType(.bfloat16)
            let q = (MLXRandom.normal([1, H, S, D]) * 0.1).asType(.bfloat16)
            var tops: [Int32] = []
            for _ in 0 ..< S { tops += Array(0 ..< nBlocks).shuffled().prefix(K).sorted().map { Int32($0) } }
            let top = MLXArray(tops).reshaped([1, S, K])
            eval(k, v, q, top)
            func chainV(_ body: (MLXArray) -> MLXArray) -> MLXArray {
                var qi = q; var last = q
                for _ in 0 ..< NCH { last = body(qi); qi = q + (last[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16) }
                return last
            }
            let shipped: (MLXArray) -> MLXArray = { Q4GatherBench.serialVerify(q: $0, keys: k, values: v, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: R, scale: 1.0 / 16, hpg: 0) }
            let refV = shipped(q).asType(.float32)
            var line = fmt("  verify S=%d kvLen=%d: shipped part P16/H1 %6.1f us", S, kvLen, timed(5) { chainV(shipped) } / Double(NCH) * 1e3)
            for hpg in [2, 3, 4, 6] {
                let body: (MLXArray) -> MLXArray = { Q4GatherBench.serialVerify(q: $0, keys: k, values: v, top: top, nBlocks: nBlocks, kvLen: kvLen, offset: offset, ratio: R, scale: 1.0 / 16, hpg: hpg) }
                let d = abs(body(q).asType(.float32) - refV).max().item(Float.self)
                line += fmt(" | hpgpos H%d %6.1f (%.1e)", hpg, timed(5) { chainV(body) } / Double(NCH) * 1e3, d)
            }
            print(line)
        }
        // DRAM conditions (P098): the rows sit at the END of a 65536-row cache (1 GB of K/V at B = 8, beyond the SLC) and every
        // chain link gathers a DIFFERENT selected set, so the rows are not served from the SLC across the chain. The P095/P097
        // numbers (kcap ~8k, one set) were SLC-served and overstated what a kernel change buys in situ.
        for base in [4096, 8192] {
            for B in [1, 4, 8] {
                let kcap = Int(argValue(args, "--kcap") ?? "65536") ?? 65536   // P106 H43: 262144 prices the gather at full context
                let rows = (0 ..< B).map { kcap - 1 - 512 * (B - 1 - $0) + (base - 4096) * 0 }   // the longest row ends at kcap - 1
                let k = (MLXRandom.normal([B, KVH, kcap, D]) * 0.1).asType(.bfloat16)
                let v = (MLXRandom.normal([B, KVH, kcap, D]) * 0.1).asType(.bfloat16)
                let q = (MLXRandom.normal([B, H, 1, D]) * 0.1).asType(.bfloat16)
                let nb = rows.map { ($0 + 1) / R }
                var topsets: [MLXArray] = []
                for _ in 0 ..< NCH {
                    var tops: [Int32] = []
                    for b in 0 ..< B { let ids = Array(0 ..< nb[b]).shuffled().prefix(K).sorted(); tops += ids.map { Int32($0) } }
                    topsets.append(MLXArray(tops).reshaped([B, 1, K]))
                }
                let top = topsets[0]
                eval(k, v, q, topsets)
                func chain(_ body: (MLXArray, MLXArray) -> MLXArray) -> MLXArray {
                    var qi = q; var last = q
                    for c in 0 ..< NCH { last = body(qi, topsets[c]); qi = q + (last[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16) }
                    return last
                }
                let single: (MLXArray, MLXArray) -> MLXArray = { Q4GatherBench.raggedPart(q: $0, keys: k, values: v, top: $1, rowOffsets: rows, rowNBlocks: nb, ratio: R, scale: 1.0 / 16, parts: 0, hpg: 1) }
                let ref = single(q, top).asType(.float32)
                if base != 4096 { continue }   // one geometry under DRAM conditions (rows near 64k); the second base is the same set
                let msS = timed(5) { chain(single) } / Double(NCH)
                // distinct bytes: each row's K and V for its K*R + 16 gathered positions, once per kv head (the 12 query heads share them)
                let mb = Double(B * KVH * (K * R + 16) * D * 2 * 2) / 1e6
                var line = fmt("  DRAM rows %s B=%d: single %6.1f us (%4.0f GB/s distinct, floor %5.1f us at 602)", cstr(rows.description), B, msS * 1e3, mb / msS, mb / 602e3 * 1e6)
                for (P, hpg) in arms {
                    let body: (MLXArray, MLXArray) -> MLXArray = { Q4GatherBench.raggedPart(q: $0, keys: k, values: v, top: $1, rowOffsets: rows, rowNBlocks: nb, ratio: R, scale: 1.0 / 16, parts: P, hpg: hpg) }
                    let d = abs(body(q, top).asType(.float32) - ref).max().item(Float.self)
                    let ms = timed(5) { chain(body) } / Double(NCH)
                    line += fmt(" | P%d/H%d %6.1f (%.1e)", P, hpg, ms * 1e3, d)
                }
                for pf in [2, 4, 8] {      // P099: the pipelined single kernel (must read max|d| 0)
                    let body: (MLXArray, MLXArray) -> MLXArray = { Q4GatherBench.raggedPipe(q: $0, keys: k, values: v, top: $1, rowOffsets: rows, rowNBlocks: nb, ratio: R, scale: 1.0 / 16, pf: pf) }
                    let d = abs(body(q, top).asType(.float32) - ref).max().item(Float.self)
                    let ms = timed(5) { chain(body) } / Double(NCH)
                    line += fmt(" | pipe PF%d %6.1f (%.1e)", pf, ms * 1e3, d)
                }
                print(line)
            }
        }
    }

    // MARK: P096 (INST-ENG-042) -- the mixer's per-row bodies priced alone at B rows.
    // Every case is a dependent chain of NCH calls on real E9 tensors, timed as `moeDecode` times its
    // chain (the eval/sync floor paid once); arms are compared against the SHIPPED body with a max|d|.

    static func fmt(_ f: String, _ a: CVarArg...) -> String { String(format: f, arguments: a) }
    static func cstr(_ s: String) -> UnsafePointer<CChar> { (s as NSString).utf8String! }
    static func quant(_ raw: [String: MLXArray], _ n: String) throws -> (MLXArray, MLXArray, MLXArray) {
        guard let w = raw[n + ".weight"], let s = raw[n + ".scales"], let b = raw[n + ".biases"] else { throw EngineError.invalid("no \(n) in fixture") }
        return (w, s, b)
    }
    static func argValue(_ args: [String], _ key: String) -> String? {
        guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// `--case proj`: the dense projections of one attention layer (qkvi, o_proj) and one GDN layer
    /// (in_proj_qkvz, out_proj) at M rows exactly as MLX dispatches them (qmv_fast below 12 rows, M on the
    /// grid, the weight streamed per row). Floor: the weight once. The M=1 multiple is the per-row cost.
    static func projBench(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let attnPath = argValue(args, "--file") ?? "runs/flashnext/attn_l3_e9.safetensors"
        let gdnPath = argValue(args, "--file2") ?? "runs/flashnext/gdn_l0_e9.safetensors"
        let attn = try loadArrays(url: URL(fileURLWithPath: attnPath)), gdn = try loadArrays(url: URL(fileURLWithPath: gdnPath))
        let banks: [(String, (MLXArray, MLXArray, MLXArray))] = [
            ("attn.qkvi", try quant(attn, "qkvi")), ("attn.o_proj", try quant(attn, "o_proj")),
            ("gdn.in_proj_qkvz", try quant(gdn, "in_proj_qkvz")), ("gdn.out_proj", try quant(gdn, "out_proj"))]
        let env = ProcessInfo.processInfo.environment
        print("engine kernelbench proj: VENDORED mlx, 8-bit g64, ENGINE_QMV_CROSSROW8=\(env["ENGINE_QMV_CROSSROW8"] ?? "default(1)") MIN_N=\(env["ENGINE_QMV_CROSSROW8_MIN_N"] ?? "default(65536)") ENGINE_DENSE_QMM_MIN=\(env["ENGINE_DENSE_QMM_MIN"] ?? "off")")
        MLXRandom.seed(1)
        let NCH = 16
        for (name, (w, sc, bi)) in banks {
            let N = w.dim(0), K = w.dim(1) * 4
            let bytes = Double(w.size * 4 + sc.size * 2 + bi.size * 2)
            var ms1 = 0.0
            for M in [1, 2, 4, 8] {
                let x = (MLXRandom.normal([M, K]) * 0.5).asType(.bfloat16); eval(x)
                func chain(pad: Int) -> MLXArray {
                    var xi = x; var last = x
                    for _ in 0 ..< NCH {
                        let xin = pad > M ? concatenated([xi, MLXArray.zeros([pad - M, K], dtype: .bfloat16)], axis: 0) : xi
                        let y = quantizedMatmul(xin, w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8)[0 ..< M]
                        last = y
                        xi = x + (y[0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16)
                    }
                    return last
                }
                let yM = quantizedMatmul(x, w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8)
                let yRows = concatenated((0 ..< M).map { quantizedMatmul(x[$0 ..< ($0 + 1)], w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8) }, axis: 0)
                let maxd = abs(yM.asType(.float32) - yRows.asType(.float32)).max().item(Float.self)
                let yPad = quantizedMatmul(concatenated([x, MLXArray.zeros([12 - M, K], dtype: .bfloat16)], axis: 0), w, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 8)[0 ..< M]
                let maxdPad = abs(yPad.asType(.float32) - yM.asType(.float32)).max().item(Float.self)
                let ms = timed(7) { chain(pad: 0) } / Double(NCH)
                let msPad = timed(7) { chain(pad: 12) } / Double(NCH)
                if M == 1 { ms1 = ms }
                print(fmt("  %-18s N=%5d K=%4d %5.1f MB  M=%d  %7.1f us = %4.0f GB/s (weight once) | x%.2f of M=1 | floor %6.1f us (weight once at 602) | padded-to-12 qmm %7.1f us (max|d| vs qmv %.3e) | max|d| vs single-row %.3e",
                          cstr(name), N, K, bytes / 1e6, M, ms * 1e3, bytes / ms / 1e6, ms / ms1, bytes / 602e9 * 1e6, msPad * 1e3, maxdPad, maxd))
            }
        }
    }

    /// `--case gdn`: the GDN layer's per-row bodies at B rows -- the fused front end (conv + silu + head
    /// norms + conv-state shift) against the unfused chain it replaces, the same chain at S=4 (the MTP
    /// verify block's path, which runs UNFUSED today), the recurrence with its fp32 state read and
    /// written (the byte floor at 602 GB/s r+w), the gate op chain against the fused gate kernel, and
    /// the gated output norm.
    static func gdnBench(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let path = argValue(args, "--file") ?? "runs/flashnext/gdn_l0_e9.safetensors"
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        guard let convW = raw["conv1d.weight"], let normW = raw["norm.weight"], let aLog = raw["A_log"], let dtBias = raw["dt_bias"] else { throw EngineError.invalid("gdn fixture keys") }
        let nK = 16, dK = 128, nV = 48, dV = 128, keyDim = nK * dK, valueDim = nV * dV, convDim = 2 * keyDim + valueDim, CK = 4, QKVZ = convDim + valueDim
        let inv = pow(Float(dK), -0.5)
        let NCH = 16
        let stateMB = Double(nV * dV * dK * 4) / 1e6
        print("engine kernelbench gdn: VENDORED mlx, nK=\(nK) dK=\(dK) nV=\(nV) dV=\(dV) convDim=\(convDim) CK=\(CK) qkvz=\(QKVZ)  state \(fmt("%.2f", stateMB)) MB per row fp32")
        MLXRandom.seed(3)
        // the unfused front end exactly as the S>1 branch runs it: concat + Conv1d(groups=convDim) + silu + split + head norms
        func unfused(_ mixed: MLXArray, _ st: MLXArray, B: Int, S: Int) -> (MLXArray, MLXArray) {
            let convInput = concatenated([st, mixed.reshaped(B, S, convDim)], axis: 1)
            let newConv = convInput[0..., (convInput.dim(1) - (CK - 1))..., 0...]
            let convOut = Q4MixerBench.silu(conv1d(convInput, convW, groups: convDim))
            let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = Q4MixerBench.headNormScale(parts[0].reshaped(B, S, nK, dK), headDim: dK, scale: inv * inv, eps: 1e-6)
            let k = Q4MixerBench.headNormScale(parts[1].reshaped(B, S, nK, dK), headDim: dK, scale: inv, eps: 1e-6)
            let v = parts[2].reshaped(B, S, nV, dV)
            return (concatenated([q.reshaped(B * S, keyDim), k.reshaped(B * S, keyDim), v.reshaped(B * S, valueDim)], axis: -1), newConv)
        }
        for B in [1, 4, 8] {
            let qkvz = (MLXRandom.normal([B, QKVZ]) * 0.5).asType(.bfloat16)
            let st0 = (MLXRandom.normal([B, CK - 1, convDim]) * 0.5).asType(.bfloat16)
            eval(qkvz, st0)
            func chainFused() -> MLXArray {
                var st = st0; var last = qkvz
                for _ in 0 ..< NCH {
                    let (qkv, nc) = Q4MixerBench.gdnConvNorm(qkvz: qkvz, convState: st, convW: convW, convDim: convDim, kernel: CK, headDim: dK, keyDim: keyDim,
                                                             scaleQ: inv * inv, scaleK: inv, eps: 1e-6)
                    st = nc; last = qkv
                }
                return last
            }
            func chainUnfused() -> MLXArray {
                var st = st0; var last = qkvz
                for _ in 0 ..< NCH { let r = unfused(qkvz[0..., 0 ..< convDim], st, B: B, S: 1); st = r.1; last = r.0 }
                return last
            }
            let f = Q4MixerBench.gdnConvNorm(qkvz: qkvz, convState: st0, convW: convW, convDim: convDim, kernel: CK, headDim: dK, keyDim: keyDim, scaleQ: inv * inv, scaleK: inv, eps: 1e-6)
            let u = unfused(qkvz[0..., 0 ..< convDim], st0, B: B, S: 1)
            let dQKV = abs(f.0.asType(.float32) - u.0.asType(.float32)).max().item(Float.self)
            let dConv = abs(f.1.asType(.float32) - u.1.asType(.float32)).max().item(Float.self)
            let msF = timed(7) { chainFused() } / Double(NCH), msU = timed(7) { chainUnfused() } / Double(NCH)
            print(fmt("  B=%d S=1 front end: fused conv_norm %6.1f us | unfused chain %6.1f us | max|d| qkv %.3e conv-state %.3e", B, msF * 1e3, msU * 1e3, dQKV, dConv))
        }
        // the verify block: S=4 at B=1 runs the UNFUSED chain today; a fused S=4 body would cost what the fused kernel costs at 4 rows
        do {
            let S = 4
            let qkvz = (MLXRandom.normal([1, S, QKVZ]) * 0.5).asType(.bfloat16)
            let st0 = (MLXRandom.normal([1, CK - 1, convDim]) * 0.5).asType(.bfloat16)
            eval(qkvz, st0)
            func chainU() -> MLXArray {
                var st = st0; var last = qkvz
                for _ in 0 ..< NCH { let r = unfused(qkvz[0..., 0..., 0 ..< convDim], st, B: 1, S: S); st = r.1; last = r.0 }
                return last
            }
            func chainF4() -> MLXArray {      // the same 4 tokens as four dependent S=1 fused calls (the state threads through)
                var st = st0; var last = qkvz
                for _ in 0 ..< NCH / 4 {
                    for t in 0 ..< S {
                        let (qkv, nc) = Q4MixerBench.gdnConvNorm(qkvz: qkvz[0..., t, 0...], convState: st, convW: convW, convDim: convDim, kernel: CK, headDim: dK, keyDim: keyDim,
                                                                 scaleQ: inv * inv, scaleK: inv, eps: 1e-6)
                        st = nc; last = qkv
                    }
                }
                return last
            }
            // equality of the S=4 unfused rows against four sequential fused S=1 calls (the class a fused S=4 body must hit)
            var st = st0; var rows: [MLXArray] = []
            for t in 0 ..< S {
                let (qkv, nc) = Q4MixerBench.gdnConvNorm(qkvz: qkvz[0..., t, 0...], convState: st, convW: convW, convDim: convDim, kernel: CK, headDim: dK, keyDim: keyDim, scaleQ: inv * inv, scaleK: inv, eps: 1e-6)
                st = nc; rows.append(qkv)
            }
            let u = unfused(qkvz[0..., 0..., 0 ..< convDim], st0, B: 1, S: S)
            let d = abs(concatenated(rows, axis: 0).asType(.float32) - u.0.asType(.float32)).max().item(Float.self)
            let msU = timed(7) { chainU() } / Double(NCH), msF4 = timed(7) { chainF4() } / Double(NCH / 4)
            print(fmt("  B=1 S=4 (verify block) front end: unfused chain %6.1f us per block | 4 x fused S=1 sequential %6.1f us | max|d| %.3e  -> a fused S=4 body is bounded by the B=4 S=1 fused row above", msU * 1e3, msF4 * 1e3, d))
        }
        // the recurrence: state fp32 [B, nV, dV, dK] read + written per layer
        for B in [1, 4, 8] {
            let q = (MLXRandom.normal([B, 1, nK, dK]) * 0.1).asType(.bfloat16), k = (MLXRandom.normal([B, 1, nK, dK]) * 0.1).asType(.bfloat16)
            let v = (MLXRandom.normal([B, 1, nV, dV]) * 0.5).asType(.bfloat16)
            let b = MLXRandom.normal([B, 1, nV]).asType(.bfloat16), a = MLXRandom.normal([B, 1, nV]).asType(.bfloat16)
            let ba = concatenated([b, a], axis: -1)
            let s0 = (MLXRandom.normal([B, nV, dV, dK]) * 0.1).asType(.float32)
            let z = MLXRandom.normal([B, 1, nV, dV]).asType(.bfloat16)
            eval(q, k, v, b, a, ba, s0, z)
            let (gF, betaF) = Q4MixerBench.gdnGate(ba: ba, aLog: aLog.asType(.bfloat16), dtBias: dtBias.asType(.bfloat16), nV: nV); eval(gF, betaF)
            func chainRec() -> MLXArray {          // shipped: gate ops + recurrence
                var st = s0; var last = v
                for _ in 0 ..< NCH { let r = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: st, mask: nil); st = r.1; last = r.0 }
                return last
            }
            func chainPre() -> MLXArray {          // recurrence alone (g, beta precomputed outside the chain)
                var st = s0; var last = v
                for _ in 0 ..< NCH { let r = gatedDeltaUpdatePrecomputed(q: q, k: k, v: v, g: gF, beta: betaF, state: st, mask: nil); st = r.1; last = r.0 }
                return last
            }
            func chainGateK() -> MLXArray {        // the fused gate kernel (bit 2048) + recurrence
                var st = s0; var last = v
                for _ in 0 ..< NCH {
                    let (g, be) = Q4MixerBench.gdnGate(ba: ba, aLog: aLog.asType(.bfloat16), dtBias: dtBias.asType(.bfloat16), nV: nV)
                    let r = gatedDeltaUpdatePrecomputed(q: q, k: k, v: v, g: g, beta: be, state: st, mask: nil); st = r.1; last = r.0
                }
                return last
            }
            let r1 = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: s0, mask: nil)
            let r3 = gatedDeltaUpdatePrecomputed(q: q, k: k, v: v, g: gF, beta: betaF, state: s0, mask: nil)
            let dOut = abs(r1.0.asType(.float32) - r3.0.asType(.float32)).max().item(Float.self)
            let dSt = abs(r1.1 - r3.1).max().item(Float.self)
            let msR = timed(7) { chainRec() } / Double(NCH), msP = timed(7) { chainPre() } / Double(NCH), msG = timed(7) { chainGateK() } / Double(NCH)
            let rw = 2 * Double(B) * stateMB
            print(fmt("  B=%d recurrence: shipped (gate ops + kernel) %6.1f us | kernel alone %6.1f us = %4.0f GB/s over %5.1f MB state r+w (floor %5.1f us at 602) | fused gate kernel + kernel %6.1f us | max|d| out %.3e state %.3e",
                      B, msR * 1e3, msP * 1e3, rw / msP, rw, rw / 602e3 * 1e3, msG * 1e3, dOut, dSt))
            let out = r1.0
            func chainNorm() -> MLXArray {
                var oi = out; var last = out
                for _ in 0 ..< NCH {
                    last = Q4MixerBench.gatedNormSG(oi, weight: normW, gate: z, headDim: dV, eps: 1e-6, sigmoidGate: true, outDType: .bfloat16)
                    oi = out + (last[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(out.dtype)
                }
                return last
            }
            let msN = timed(7) { chainNorm() } / Double(NCH)
            print(fmt("  B=%d gated output norm (q4_gated_norm_sg, %d groups): %6.1f us", B, B * nV, msN * 1e3))
        }
    }

    /// `--case topk`: the indexer's top-K selection at rows x n. Scores come from `ENGINE_RAGGED_WITNESS`
    /// dumps (`<dir>/call<i>.safetensors`, key `scores` [1,1,n], row 0 of a ragged step) when the
    /// directory exists (`--witness <dir>`, default runs/p096/witness), else a synthetic relu-sum.
    /// Arms: as dispatched, one group per row, two-phase P=2 / P=4 (three dispatches), 1D P=2 / P=4.
    static func topkBench(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let K = 512
        let dir = argValue(args, "--witness") ?? "runs/p096/witness"
        var byN: [Int: [MLXArray]] = [:]
        if let files = try? FileManager.default.contentsOfDirectory(atPath: dir) {
            for f in files.sorted() where f.hasPrefix("call") && f.hasSuffix(".safetensors") {
                if let a = try? loadArrays(url: URL(fileURLWithPath: dir + "/" + f)), let sc = a["scores"], sc.dtype == .float32, sc.ndim == 3 {
                    byN[sc.dim(2), default: []].append(sc.reshaped(1, sc.dim(2)))
                }
            }
        }
        print("engine kernelbench topk: VENDORED mlx, K=\(K), knobs \(Q4MixerBench.topKKnobs) parts-min-n=\(Q4MixerBench.topKPartsMinN), witness rows by n: \(byN.keys.sorted().map { "\($0):\(byN[$0]!.count)" }.joined(separator: " "))")
        MLXRandom.seed(4)
        let NCH = 16
        let targets = [1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536]   // witness where it exists (<= 4099), synthetic above
        for target in targets {
            let nReal = byN.keys.min { abs($0 - target) < abs($1 - target) }
            let n: Int, source: String, pool: [MLXArray]
            if let nr = nReal, abs(nr - target) <= target / 4 {
                n = nr; source = "witness"; pool = byN[nr]!
            } else {
                n = target; source = "synthetic"
                pool = (0 ..< 8).map { _ in (maximum(MLXRandom.normal([4, n]), MLXArray(Float(0))).sum(axis: 0) / sqrt(Float(128))).reshaped(1, n) }
            }
            for rows in [1, 4, 8] {
                let scores = concatenated((0 ..< rows).map { pool[$0 % pool.count] }, axis: 0).reshaped(rows, 1, n)
                eval(scores)
                let ref = Q4MixerBench.topKSingle(scores: scores, k: K)
                func check(_ out: MLXArray) -> String {
                    let setMiss = (sorted(out, axis: -1) .!= sorted(ref, axis: -1)).sum().item(Int32.self)
                    let orderMiss = (out .!= ref).sum().item(Int32.self)
                    return setMiss == 0 ? (orderMiss == 0 ? "identical" : "same set, order differs \(orderMiss)") : "SET DIFFERS \(setMiss)"
                }
                func chain(_ body: (MLXArray) -> MLXArray) -> MLXArray {
                    var si = scores; var last = ref
                    for _ in 0 ..< NCH { last = body(si); si = scores + (last[0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(.float32) }
                    return last
                }
                var arms: [(String, (MLXArray) -> MLXArray)] = [
                    ("dispatched", { Q4MixerBench.topK(scores: $0, k: K) }),
                    ("single-group", { Q4MixerBench.topKSingle(scores: $0, k: K) })]
                for P in [2, 4] where n % P == 0 && n / P >= K {
                    arms.append(("P=\(P) 3-dispatch", { Q4MixerBench.topKTwoPhase(scores: $0, k: K, parts: P, oneDispatch: false) }))
                    arms.append(("P=\(P) 1D", { Q4MixerBench.topKTwoPhase(scores: $0, k: K, parts: P, oneDispatch: true) }))
                }
                var line = fmt("  n=%4d (%s) rows=%d parts-as-dispatched=%s:", n, cstr(source), rows, cstr(Q4MixerBench.topKParts(rows: rows, n: n, k: K).map { "\($0)" } ?? "none"))
                for (name, body) in arms {
                    let ms = timed(7) { chain(body) } / Double(NCH)
                    line += fmt("  %s %6.1f us [%s]", cstr(name), ms * 1e3, cstr(check(body(scores))))
                }
                print(line)
            }
        }
    }

    /// `--case idxscore`: the indexer score at B rows -- the graph form (pooled cast to fp32, gemm, relu-sum
    /// kernel), the bf16 gemm form (ENGINE_IDX_BF16), and the fused q4_idx_score_fused2 with the query padded to
    /// an 8-row tile as the S=1 decode branch does (gated B == 1 in the engine today; the kernel carries a batch index).
    static func idxScoreBench(_ args: [String]) throws {
        if let path = argValue(args, "--file") { return try idxScoreReplay(args, path: path) }
        setvbuf(stdout, nil, _IOLBF, 0)
        let H = 4, D = 128, R = 4, NCH = 16
        let scale = sqrt(Float(D))
        print("engine kernelbench idxscore: VENDORED mlx, indexer heads \(H) x \(D), ratio \(R), NI=\(Q4MixerBench.idxScoreNI)")
        MLXRandom.seed(5)
        for n in [1024, 2048, 8192, 16384, 65536] {    // P101: 8192/16384 pooled blocks = 32k/64k rows; P106 H43: 65536 = 262144-row context
            for B in [1, 4, 8] {
                let q = (MLXRandom.normal([B, 1, H, D]) * 0.3).asType(.bfloat16)
                let pooled = (MLXRandom.normal([B, n, D]) * 0.3).asType(.bfloat16)
                eval(q, pooled)
                let vis = (kvLen: n * R, ratio: R)
                func graph(_ qi: MLXArray) -> MLXArray {
                    let q32 = qi.asType(.float32).reshaped(B, H, D)
                    let p32 = pooled.asType(.float32).transposed(0, 2, 1)
                    return Q4MixerBench.idxReluSum(matmul(q32, p32).reshaped(B, 1, H, n), scale: scale, visibility: vis)
                }
                func bf16(_ qi: MLXArray) -> MLXArray {
                    Q4MixerBench.idxReluSum(matmul(qi.reshaped(B, H, D), pooled.transposed(0, 2, 1)).reshaped(B, 1, H, n), scale: scale, visibility: vis)
                }
                func fused(_ qi: MLXArray) -> MLXArray {
                    let qpad = concatenated([tiled(qi[0..., 0 ..< 1, 0..., 0...], repetitions: [1, 7, 1, 1]), qi], axis: 1)
                    return Q4MixerBench.idxScoreFused(q: qpad, pooled: pooled, scale: scale, visibility: vis, mi: 1, ni: Q4MixerBench.idxScoreNI, sg: 1)[0..., 7 ..< 8, 0...]
                }
                func chain(_ body: (MLXArray) -> MLXArray) -> MLXArray {
                    var qi = q; var last = q
                    for _ in 0 ..< NCH { last = body(qi); qi = q + (last[0 ..< 1, 0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16) }
                    return last
                }
                let ref = graph(q)
                let dB = abs(bf16(q) - ref).max().item(Float.self), dF = abs(fused(q) - ref).max().item(Float.self)
                // P103: the bound on OBS-ENG-197's open lever. The ragged step calls the fused kernel ONCE PER ROW
                // (`raggedDecode`, each row with its own visibility); one call for all B rows is what a per-row-visibility
                // kernel would buy. `perRow` is today's cost, `fused` (one call, B rows) the floor -- the gap is the ceiling.
                func perRow(_ qi: MLXArray) -> MLXArray {
                    var outs: [MLXArray] = []
                    for b in 0 ..< B {
                        let qb = qi[b ..< (b + 1)]
                        let qpad = concatenated([tiled(qb[0..., 0 ..< 1, 0..., 0...], repetitions: [1, 7, 1, 1]), qb], axis: 1)
                        outs.append(Q4MixerBench.idxScoreFused(q: qpad, pooled: pooled[b ..< (b + 1)], scale: scale, visibility: vis,
                                                               mi: 1, ni: Q4MixerBench.idxScoreNI, sg: 1)[0..., 7 ..< 8, 0...])
                    }
                    return B == 1 ? outs[0] : concatenated(outs, axis: 0)
                }
                let msG = timed(7) { chain(graph) } / Double(NCH), msB = timed(7) { chain(bf16) } / Double(NCH), msF = timed(7) { chain(fused) } / Double(NCH)
                let msP = timed(7) { chain(perRow) } / Double(NCH)
                let dP = abs(perRow(q) - ref).max().item(Float.self)
                let castMB = Double(B * n * D * 4) / 1e6
                print(fmt("  n=%4d B=%d: graph fp32 %6.1f us (pooled cast %5.1f MB) | bf16 gemm %6.1f us (max|d| %.3e) | fused 1 call %6.1f us (max|d| %.3e) | fused %d per-row calls %6.1f us (max|d| %.3e) -> batching saves %5.1f us/layer",
                          n, B, msG * 1e3, castMB, msB * 1e3, dB, msF * 1e3, dF, B, msP * 1e3, dP, (msP - msF) * 1e3))
            }
        }
    }

    /// Real-tensor S1 replay, with no model load and no change to serving's selected score path.
    /// `--compare-only` runs only the reconstruction and ordered-ID gates. Otherwise score-only
    /// and score+selection each use ABBA blocks; every sample builds/evaluates a fresh graph.
    /// These are isolated host+GPU wall costs, not a GPU timer or an end-to-end speedup claim.
    private static func idxScoreReplay(_ args: [String], path: String) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        func emit(_ value: [String: Any]) throws {
            let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        }
        func invalid(_ why: String) -> EngineError { .invalid("INSTRUMENT_FAIL: idx replay: " + why) }
        let reuseFP32 = args.contains("--reuse-fp32")
        let batchedCandidate = args.contains("--fused-batch")
        guard !(reuseFP32 && batchedCandidate) else {
            throw invalid("--reuse-fp32 and --fused-batch are separate discriminators")
        }
        let (arrays, meta) = try loadArraysAndMetadata(url: URL(fileURLWithPath: path))
        guard meta["schema"] == "engine.idx-s1-replay.v1", meta["complete"] == "true",
              let q = arrays["q"], let storage = arrays["pooled_storage"],
              let referenceTop = arrays["reference_top"], let nbArray = arrays["nblocks"],
              let rowArray = arrays["row_offsets"], let route = meta["route"],
              ["equal", "masked", "unequal"].contains(route) else { throw invalid("missing fixture fields/schema") }
        let nb = nbArray.asArray(Int32.self).map(Int.init)
        let rows = rowArray.asArray(Int32.self).map(Int.init)
        guard q.ndim == 4, q.dim(0) == 8, q.dim(1) == 1,
              storage.ndim == 3, storage.dim(0) == 8, storage.dim(2) == q.dim(3),
              q.dtype == storage.dtype, [DType.bfloat16, .float16, .float32].contains(q.dtype),
              nb.count == 8, rows.count == 8, nb.min()! >= 64_000,
              nb.max()! <= storage.dim(1), referenceTop.ndim == 3,
              referenceTop.dim(0) == 8, referenceTop.dim(1) == 1, referenceTop.dtype == .int32,
              meta["ragged_batch_active"] == "true",
              let ratio = Int(meta["ratio"] ?? ""), ratio > 0,
              let budget = Int(meta["budget"] ?? ""), budget > 0 else { throw invalid("unsupported shape/dtype/context") }
        let B = q.dim(0), H = q.dim(2), D = q.dim(3), N = nb.max()!, K = referenceTop.dim(2)
        guard D % 8 == 0, K > 0, K <= nb.min()!,
              zip(rows, nb).allSatisfy({ ($0.0 + 1) / ratio == $0.1 }),
              meta["batch"] == String(B), meta["sequence"] == "1", meta["heads"] == String(H),
              meta["head_dim"] == String(D), meta["k"] == String(K), meta["nblocks_max"] == String(N),
              meta["pool_capacity"] == String(storage.dim(1)),
              meta["q_dtype"] == String(describing: q.dtype), meta["pool_dtype"] == String(describing: storage.dtype)
              else { throw invalid("invalid query/block counts or tensor metadata") }
        let equal = nb.allSatisfy { $0 == N }
        guard (route == "equal") == equal,
              route != "masked" || meta["ragged_select_masked"] == "true",
              route != "unequal" || meta["ragged_select_masked"] == "false" else { throw invalid("route/count mismatch") }
        for (key, applied) in Q4MixerBench.idxScoreReplayConfiguration {
            guard meta[key] == applied else { throw invalid("applied configuration differs: \(key), dump=\(meta[key] ?? "missing") replay=\(applied)") }
        }
        guard meta["topk_hist"] == "false", meta["topk_rounds"] == "4",
              Q4MixerBench.idxScoreNI > 0, nb.min()! >= 8 * Q4MixerBench.idxScoreNI else {
            throw invalid("timing witness/ablation or unsupported fused tile")
        }
        func view() -> MLXArray { storage[0..., 0..<N, 0...] }
        let pooled = view()
        eval(q, storage, pooled, referenceTop)
        func verifyLayout(_ array: MLXArray, key: String) throws {
            guard let value = meta[key] else { throw invalid("missing \(key)") }
            let expected = value.split(separator: ",").compactMap { Int($0) }
            let actual = array.asData(access: .noCopy).strides
            guard expected.count == array.ndim,
                  array.shape.indices.allSatisfy({ array.shape[$0] == 1 || expected[$0] == actual[$0] }) else {
                throw invalid("\(key) changed: dump=\(expected) replay=\(actual)")
            }
        }
        try verifyLayout(q, key: "q_strides")
        try verifyLayout(storage, key: "pool_storage_strides")
        try verifyLayout(pooled, key: "pool_view_strides")
        guard String(pooled.nbytes) == meta["pool_view_logical_bytes"],
              String(storage.nbytes) == meta["pool_storage_tensor_bytes"] else { throw invalid("logical/capacity bytes changed") }
        // D39: even TOPK_1D=1 must keep the production ragged dispatch guard. Without this,
        // an offline replay can silently select a kernel the captured server never executed.
        let previousRows = Qwen4ExpBatch.rowOffsets
        Qwen4ExpBatch.rowOffsets = rows
        defer { Qwen4ExpBatch.rowOffsets = previousRows }
        let perRow = route == "unequal", scale = sqrt(Float(D))
        let fusedReduction = meta["reduction_fused"] == "true"
        let radixSelection = meta["topk_select"] == "true"
        func reduce(_ raw: MLXArray) -> MLXArray {
            fusedReduction ? Q4MixerBench.idxReluSum(raw, scale: scale, visibility: nil)
                : maximum(raw.asType(.float32), MLXArray(Float(0))).sum(axis: 2) / scale
        }
        // The consumer caches int32Range; recreating its N-element host array per sample
        // would charge unrelated setup work to the masked control and candidate.
        let blockIndices = route == "masked" ? MLXArray((0..<N).map { Int32($0) }) : nil
        if let blockIndices { eval(blockIndices) }
        func mask(_ scores: MLXArray) -> MLXArray {
            guard let blockIndices else { return scores }
            let blocks = blockIndices.reshaped(1, 1, N)
            let counts = MLXArray(nb.map { Int32($0) }).reshaped(B, 1, 1)
            return which(blocks .< counts, scores, MLXArray(-Float.infinity))
        }
        func controlScores() -> [MLXArray] {
            let p = view(), q32 = q.asType(.float32).reshaped(B, H, D)
            if !perRow {
                let raw = matmul(q32, p.asType(.float32).transposed(0, 2, 1)).reshaped(B, 1, H, N)
                return [mask(reduce(raw))]
            }
            return (0..<B).map { b in
                let pb = p[b..<(b + 1), 0..<nb[b], 0...].asType(.float32).transposed(0, 2, 1)
                return reduce(matmul(q32[b..<(b + 1)], pb).reshaped(1, 1, H, nb[b]))
            }
        }
        func fusedOne(_ qi: MLXArray, _ p: MLXArray) -> MLXArray {
            // Padding, any input packing and the output slice remain INSIDE the timed body.
            let pad = concatenated([tiled(qi[0..., 0..<1, 0..., 0...], repetitions: [1, 7, 1, 1]), qi], axis: 1)
            return Q4MixerBench.idxScoreFused(q: pad, pooled: p, scale: scale, visibility: nil,
                                             mi: 1, ni: Q4MixerBench.idxScoreNI, sg: 1)[0..., 7..<8, 0...]
        }
        // Removal-only discriminator: materialize exactly the control's pooled casts once.
        // Query casts, matmul shapes, reduction, masking and selection stay inside each call.
        // This excludes maintenance, invalidation, allocation peaks and first-conversion cost
        // from candidate timing; it is NOT a serving cache implementation or speedup claim.
        var preparedFP32: [MLXArray] = []
        var firstConversionMS: Double? = nil
        var preparedMemory: [String: Int] = [:]
        if reuseFP32 {
            StreamOrDevice.default.stream.synchronize()
            let activeBefore = Memory.activeMemory
            let start = ProcessInfo.processInfo.systemUptime
            let p = view()
            preparedFP32 = perRow
                ? (0..<B).map { b in p[b..<(b + 1), 0..<nb[b], 0...].asType(.float32) }
                : [p.asType(.float32)]
            eval(preparedFP32)
            StreamOrDevice.default.stream.synchronize()
            firstConversionMS = (ProcessInfo.processInfo.systemUptime - start) * 1000
            preparedMemory = ["active_before_bytes": activeBefore, "active_after_bytes": Memory.activeMemory,
                              "cache_after_bytes": Memory.cacheMemory, "process_peak_after_bytes": Memory.peakMemory]
        }
        let preparedLayouts: [[String: Any]] = preparedFP32.enumerated().map { index, array in
            let strides = array.asData(access: .noCopy).strides
            let span = (1 + zip(array.shape, strides).reduce(0) { $0 + ($1.0 - 1) * abs($1.1) }) * array.itemSize
            return ["index": index, "shape": array.shape, "dtype": String(describing: array.dtype),
                    "strides": strides, "logical_bytes": array.nbytes,
                    "reachable_span_bytes_not_allocator": span]
        }
        let preparedBytes = preparedFP32.reduce(0) { $0 + $1.nbytes }
        // The fused alternatives retain their existing gates and shapes. They cannot be
        // combined with cast reuse: that would stop isolating the cast's removal cost.
        let candidateLabel = reuseFP32 ? (perRow ? "reuse_fp32_per_row" : "reuse_fp32_batched")
            : (perRow && !batchedCandidate ? "fused_per_row" : "fused_batched")
        func candidateScores() -> [MLXArray] {
            if reuseFP32 {
                let q32 = q.asType(.float32).reshaped(B, H, D)
                if !perRow {
                    let raw = matmul(q32, preparedFP32[0].transposed(0, 2, 1)).reshaped(B, 1, H, N)
                    return [mask(reduce(raw))]
                }
                return (0..<B).map { b in
                    let pb = preparedFP32[b].transposed(0, 2, 1)
                    return reduce(matmul(q32[b..<(b + 1)], pb).reshaped(1, 1, H, nb[b]))
                }
            }
            let p = view()
            if !perRow { return [mask(fusedOne(q, p))] }
            if batchedCandidate {
                let full = fusedOne(q, p)
                return (0..<B).map { b in full[b..<(b + 1), 0..., 0..<nb[b]] }
            }
            return (0..<B).map { b in fusedOne(q[b..<(b + 1)], p[b..<(b + 1), 0..<nb[b], 0...]) }
        }
        func selectOne(_ scores: MLXArray, _ count: Int) -> MLXArray {
            (radixSelection ? Q4MixerBench.topK(scores: scores, k: count)
             : argPartition(-scores, kth: count - 1, axis: -1)[.ellipsis, 0..<count]).asType(.int32)
        }
        func select(_ scores: [MLXArray]) -> MLXArray {
            if !perRow { return selectOne(scores[0], K) }
            let selected = scores.enumerated().map { b, score -> MLXArray in
                let kb = min(K, nb[b])
                let top = selectOne(score, kb)
                return kb == K ? top : concatenated([top, MLXArray.full([1, 1, K - kb], values: MLXArray(Int32(nb[b])))], axis: -1)
            }
            return concatenated(selected, axis: 0)
        }
        func idCheck(_ actual: MLXArray, expected: MLXArray) -> (report: [String: Any], mismatches: Int) {
            let a = actual.asArray(Int32.self), e = expected.asArray(Int32.self)
            var ordered = 0, sets = 0, symmetric = 0
            var first: [String: Int]? = nil
            for i in a.indices where a[i] != e[i] {
                ordered += 1
                if first == nil { first = ["row": i / K, "query": 0, "index": i % K, "actual": Int(a[i]), "expected": Int(e[i])] }
            }
            for b in 0..<B {
                let range = (b * K)..<((b + 1) * K)
                sets += zip(a[range].sorted(), e[range].sorted()).filter { $0.0 != $0.1 }.count
                symmetric += Set(a[range]).symmetricDifference(Set(e[range])).count
            }
            return (["ordered_id_mismatches": ordered, "sorted_id_mismatches": sets,
                     "set_symmetric_difference_count": symmetric,
                     "first_ordered_mismatch": first.map { $0 as Any } ?? NSNull()], ordered)
        }
        func scoreCheck(_ actual: [MLXArray], expected: [MLXArray]) -> [String: Any] {
            var bits = 0, nonfinite = 0
            var maxFinite: Double = 0
            var first: [String: Any]? = nil
            for (index, pair) in zip(actual, expected).enumerated() {
                let a = pair.0.asArray(Float.self), e = pair.1.asArray(Float.self)
                let width = pair.0.dim(2)
                for i in a.indices {
                    if a[i].isFinite && e[i].isFinite { maxFinite = max(maxFinite, abs(Double(a[i]) - Double(e[i]))) }
                    else if a[i].bitPattern != e[i].bitPattern { nonfinite += 1 }
                    if a[i].bitPattern != e[i].bitPattern {
                        bits += 1
                        if first == nil {
                            first = ["row": perRow ? index : i / width, "query": 0, "block": i % width,
                                     "actual_bits": String(a[i].bitPattern, radix: 16),
                                     "expected_bits": String(e[i].bitPattern, radix: 16)]
                        }
                    }
                }
            }
            return ["score_bit_mismatches": bits, "score_max_abs_finite": Double(maxFinite),
                    "score_nonfinite_bit_mismatches": nonfinite,
                    "first_score_mismatch": first.map { $0 as Any } ?? NSNull()]
        }
        let scoreCalls = perRow ? B : 1
        try emit(["event": "idx_replay_fixture", "file": path, "metadata": meta,
                  "nblocks": nb, "row_offsets": rows, "score_calls_per_invocation": scoreCalls,
                  "candidate": candidateLabel, "candidate_score_calls_per_invocation": reuseFP32 ? scoreCalls : (perRow && !batchedCandidate ? B : 1),
                  "reuse_fp32": reuseFP32, "prepared_cache_logical_bytes": preparedBytes,
                  "prepared_cache_layouts": preparedLayouts,
                  "prepared_first_conversion_ms": firstConversionMS.map { $0 as Any } ?? NSNull(),
                  "prepared_memory": preparedMemory,
                  "reuse_scope": reuseFP32 ? "optimistic cast-removal discriminator; excludes cache maintenance, invalidation, first conversion and production memory cost; prepared cache resident in both timing arms" : "not_applicable",
                  "pool_replay_strides": pooled.asData(access: .noCopy).strides,
                  "pool_storage_bytes": storage.nbytes, "pool_logical_bytes": pooled.nbytes,
                  "timing_metric": "isolated_host_plus_gpu_eval_wall_ms"])
        let control = controlScores()
        let controlTop = select(control)
        eval(control, controlTop)
        let reconstruction = idCheck(controlTop, expected: referenceTop)
        try emit(["event": "idx_replay_control_gate", "status": reconstruction.mismatches == 0 ? "PASS" : "INSTRUMENT_FAIL",
                  "checks": reconstruction.report])
        guard reconstruction.mismatches == 0 else { throw invalid("control does not reconstruct captured ordered IDs") }
        let candidate = candidateScores()
        let candidateTop = select(candidate)
        eval(candidate, candidateTop)
        let identity = idCheck(candidateTop, expected: controlTop)
        let scoreIdentity = scoreCheck(candidate, expected: control)
        let scoreBitsPass = !reuseFP32 || (scoreIdentity["score_bit_mismatches"] as? Int) == 0
        let gateStatus = identity.mismatches != 0 ? "CANDIDATE_ID_MISMATCH"
            : (scoreBitsPass ? "PASS" : "CANDIDATE_SCORE_BIT_MISMATCH")
        try emit(["event": "idx_replay_candidate_gate", "status": gateStatus,
                  "candidate": candidateLabel, "score_bits_required": reuseFP32,
                  "ids": identity.report, "scores": scoreIdentity])
        guard identity.mismatches == 0 else { throw EngineError.invalid("CANDIDATE_ID_MISMATCH: idx replay; no timings accepted") }
        guard scoreBitsPass else { throw EngineError.invalid("CANDIDATE_SCORE_BIT_MISMATCH: idx replay; no timings accepted") }
        if args.contains("--compare-only") { return }
        let cycles = Int(argValue(args, "--cycles") ?? "5") ?? 0
        let repeats = Int(argValue(args, "--repeats") ?? "7") ?? 0
        guard (1...100).contains(cycles), (1...1000).contains(repeats) else { throw invalid("invalid --cycles/--repeats") }
        var routeCalls: Int? = nil
        if let value = argValue(args, "--route-calls") {
            guard let count = Int(value), count >= 0, count % scoreCalls == 0 else { throw invalid("--route-calls must be a nonnegative multiple of score_calls_per_invocation") }
            routeCalls = count
        }
        func median(_ a: [Double]) -> Double {
            let sorted = a.sorted(), mid = a.count / 2
            return a.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
        }
        for includeSelection in [false, true] {
            let mode = includeSelection ? "score_and_selection" : "score_only"
            func body(_ arm: String) -> [MLXArray] {
                let scores = arm == "A" ? controlScores() : candidateScores()
                return includeSelection ? [select(scores)] : scores
            }
            for arm in ["A", "B"] { for _ in 0..<3 { eval(body(arm)) } }
            StreamOrDevice.default.stream.synchronize()
            var controls: [Double] = [], candidates: [Double] = [], pairedDeltas: [Double] = []
            for cycle in 0..<cycles {
                var cycleA: [Double] = [], cycleB: [Double] = []
                for (position, arm) in ["A", "B", "B", "A"].enumerated() {
                    var times: [Double] = []
                    for _ in 0..<repeats {
                        StreamOrDevice.default.stream.synchronize()
                        let start = ProcessInfo.processInfo.systemUptime
                        eval(body(arm))
                        StreamOrDevice.default.stream.synchronize()
                        times.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
                    }
                    let value = median(times)
                    if arm == "A" { controls.append(value); cycleA.append(value) }
                    else { candidates.append(value); cycleB.append(value) }
                    try emit(["event": "idx_replay_timing", "mode": mode, "cycle": cycle,
                              "position": position, "arm": arm, "median_ms": value, "samples_ms": times])
                }
                pairedDeltas.append(cycleA.reduce(0, +) / 2 - cycleB.reduce(0, +) / 2)
            }
            let a = median(controls), b = median(candidates), delta = median(pairedDeltas)
            var result: [String: Any] = ["event": "idx_replay_summary", "mode": mode,
                "control_median_ms_per_invocation": a, "candidate_median_ms_per_invocation": b,
                "paired_delta_median_ms_per_invocation": delta, "paired_deltas_ms": pairedDeltas,
                "score_calls_per_invocation": scoreCalls, "candidate": candidateLabel, "cycles": cycles, "repeats": repeats,
                "scope": reuseFP32
                    ? "one captured tensor; optimistic pooled-cast removal; excludes first conversion, maintenance, invalidation and production memory cost; no serving speedup claim"
                    : "one captured tensor; isolated host+GPU wall; no serving speedup claim",
                "prepared_cache_logical_bytes": preparedBytes,
                "prepared_first_conversion_ms": firstConversionMS.map { $0 as Any } ?? NSNull(),
                "removal_ceiling_assumptions": "representative tensors, all counted graphs executed, serial additive costs; not an unconditional runtime bound"]
            if let routeCalls {
                let invocations = Double(routeCalls / scoreCalls)
                result["supplied_graph_construction_score_calls"] = routeCalls
                result["derived_invocations"] = Int(invocations)
                result["derived_component_control_ms"] = invocations * a
                result["derived_candidate_saved_ms"] = invocations * delta
            }
            try emit(result)
        }
    }

    /// P099 `--case moe-rows`: the decode experts at T rows through MLX's gather path (`gather_qmm`, what the engine
    /// takes above 8 rows) on real routing, "seq" (consecutive rows = a verify block) and "far" (rows 512 apart = a
    /// batch of conversations), T in {4, 8, 12, 16, 24, 32}; the fused decode body from `--case moe-decode` is the
    /// comparison at T <= 8. Reports us per call, distinct experts and GB/s over the distinct bytes.
    static func moeRowsBench(_ args: [String]) throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        let path = argValue(args, "--file") ?? "runs/flashnext/layer0_mlp_e9.safetensors"
        let layerTag = path.contains("layer45") ? "45" : "0"
        let raw = try loadArrays(url: URL(fileURLWithPath: path))
        var W: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("switch_mlp.") { W[String(k.dropFirst("switch_mlp.".count))] = v }
        func bank(_ n: String) throws -> (MLXArray, MLXArray, MLXArray) {
            guard let w = W[n + ".weight"], let s = W[n + ".scales"], let b = W[n + ".biases"] else { throw EngineError.usage }
            return (w, s, b)
        }
        let gate = try bank("gate_proj"), up = try bank("up_proj"), down = try bank("down_proj")
        let words = gate.0.dim(2), bits = words == 640 ? 8 : 4, gs = bits == 8 ? 64 : 32
        let D = words * (bits == 8 ? 4 : 8), I = gate.0.dim(1), K = 10
        func bankBytes(_ b: (MLXArray, MLXArray, MLXArray)) -> Double { Double(b.0.dim(1) * b.0.dim(2) * 4 + 2 * b.1.dim(1) * b.1.dim(2) * 2) }
        let perExpert = bankBytes(gate) + bankBytes(up) + bankBytes(down)
        guard let dump = FileManager.default.contents(atPath: "runs/rt.i32.\(layerTag)") else { throw EngineError.invalid("no runs/rt.i32.\(layerTag)") }
        let rows: [[Int32]] = dump.withUnsafeBytes { (b: UnsafeRawBufferPointer) -> [[Int32]] in
            let v = Array(b.bindMemory(to: Int32.self)); return stride(from: 0, to: v.count - K + 1, by: K).map { Array(v[$0 ..< $0 + K]) }
        }
        print("engine kernelbench moe-rows: VENDORED mlx, layer \(layerTag), \(bits)-bit, gather_qmm path (the engine above 8 rows), \(String(format: "%.2f", perExpert / 1e6)) MB per expert (3 banks)")
        MLXRandom.seed(6)
        let NROT = 16
        let cases: [(Int, String)] = argValue(args, "--h51") != nil
            ? [(8, "seq4x2"), (12, "seq4x3"), (16, "seq4x4"), (20, "seq4x5"), (24, "seq4x6"), (32, "seq4x8"), (20, "far"), (32, "far")]
            : [(4, "seq"), (8, "seq"), (8, "far"), (12, "far"), (16, "far"), (16, "seq4x4"), (24, "far"), (32, "far"), (32, "seq4x8")]
        for (T, pattern) in cases {
            let x = (MLXRandom.normal([T, D]) * 0.5).asType(.bfloat16)
            var sets: [MLXArray] = []; var distinctSum = 0
            for r in 0 ..< NROT {
                let base = 100 + r * 97
                let picked: [[Int32]] = (0 ..< T).map { t in
                    let idx: Int
                    switch pattern {
                    case "seq": idx = base + t
                    case let p where p.hasPrefix("seq4x"): idx = base + (t / 4) * 128 + (t % 4)        // 4-row verify blocks of different conversations
                    default: idx = base + t * 128                      // 32 distinct conversations inside the 4096-row dump
                    }
                    return rows[idx % rows.count]
                }
                distinctSum += Set(picked.flatMap { $0 }).count
                sets.append(MLXArray(picked.flatMap { $0 }).reshaped([T, K]))
            }
            eval(x, sets)
            let distinct = Double(distinctSum) / Double(NROT)
            func chain() -> MLXArray {
                var xi = x; var last = x
                for i in 0 ..< NROT {
                    // the SwitchGLU shape: gather gate/up with the flat sorted index, silu*mul, gather down, weighted sum over K
                    let idx = sets[i]
                    let xe = expandedDimensions(xi, axes: [-2, -3])                                              // (T,1,1,D) as SwitchGLU
                    let g = gatherQuantizedMatmul(xe, gate.0, scales: gate.1, biases: gate.2, rhsIndices: idx, transpose: true, groupSize: gs, bits: bits)
                    let u = gatherQuantizedMatmul(xe, up.0, scales: up.1, biases: up.2, rhsIndices: idx, transpose: true, groupSize: gs, bits: bits)
                    let h = (g * sigmoid(g)) * u                                                                  // (T,K,1,I)
                    let d = gatherQuantizedMatmul(h, down.0, scales: down.1, biases: down.2, rhsIndices: idx, transpose: true, groupSize: gs, bits: bits)  // (T,K,1,D)
                    last = d.sum(axis: 1).squeezed(axis: 1)
                    xi = x + (last[0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16)
                }
                return last
            }
            let ms = timed(5) { chain() } / Double(NROT)
            let mb = distinct * perExpert / 1e6
            print(String(format: "  T=%2d %-7s distinct %5.1f experts | %7.1f us | %5.0f MB distinct = %4.0f GB/s (floor %5.1f us at 602)", T, (pattern as NSString).utf8String!, distinct, ms * 1e3, mb, mb / ms, mb / 602e3 * 1e6))
            // P106 H51b: what the engine ACTUALLY runs above 8 rows is SwitchGLU's SORTED path (indices.size >= 64:
            // gatherSort, sortedIndices gathers, scatterUnsort), not the unsorted gather above; and the question is
            // whether the fused per-pair expert kernels (the engine's path at <= 8 rows, which have no row limit)
            // beat it. Same routing sets, same dependent chain.
            if argValue(args, "--h51") != nil {
                func sortedChain() -> MLXArray {
                    var xi = x; var last = x
                    for i in 0 ..< NROT {
                        let idx = sets[i]
                        let (xs, ids, inv) = gatherSort(x: expandedDimensions(xi, axes: [-2, -3]), indices: idx)
                        let g = gatherQuantizedMatmul(xs, gate.0, scales: gate.1, biases: gate.2, rhsIndices: ids, transpose: true, groupSize: gs, bits: bits, sortedIndices: true)
                        let u = gatherQuantizedMatmul(xs, up.0, scales: up.1, biases: up.2, rhsIndices: ids, transpose: true, groupSize: gs, bits: bits, sortedIndices: true)
                        let h = (g * sigmoid(g)) * u
                        let d = gatherQuantizedMatmul(h, down.0, scales: down.1, biases: down.2, rhsIndices: ids, transpose: true, groupSize: gs, bits: bits, sortedIndices: true)
                        let yk = scatterUnsort(x: d, invOrder: inv, shape: idx.shape).squeezed(axis: -2)   // (T,K,D)
                        last = yk.sum(axis: 1)
                        xi = x + (last[0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16)
                    }
                    return last
                }
                func fusedChain() -> MLXArray {
                    var xi = x; var last = x
                    for i in 0 ..< NROT {
                        let r = Q4MoEDecodeBench.run(x: xi, idx: sets[i].asType(.int32), gate: gate, up: up, down: down, bits: bits, K: K, D: D, I: I)
                        last = r.yk.sum(axis: 1)
                        xi = x + (last[0 ..< 1, 0 ..< 1] * 0).asType(.bfloat16)
                    }
                    return last
                }
                let msS = timed(5) { sortedChain() } / Double(NROT)
                let msF = timed(5) { fusedChain() } / Double(NROT)
                let a = sortedChain(), b = fusedChain()
                let dmax = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
                print(String(format: "      sorted (engine >8 rows) %7.1f us = %4.0f GB/s | fused per-pair %7.1f us = %4.0f GB/s | fused/sorted %.3f | last-call max|d| %.4g  [%@]",
                             msS * 1e3, mb / msS, msF * 1e3, mb / msF, msF / msS, dmax, Q4MoEDecodeBench.witness(T: T, bits: bits)))
            }
        }
    }
}
