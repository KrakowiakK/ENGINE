import XCTest
import MLX
import MLXLLM
@testable import Qwen4Exp

/// T-0194: the R-rows GDN recurrence (`q4GDNRows`, ENGINE_GDN_ROWS=1) must give the shipped kernel's
/// (`gatedDeltaUpdatePrecomputed`) output and next state bit for bit, from a zero state and from a carried one, at
/// prefill lengths and E9's head layout (Hv a multiple of Hk). Runs the Metal kernels (GPU); opt in with
/// MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
final class GDNRowsTests: XCTestCase {
    func testRowsKernelIsBitIdenticalToTheShippedRecurrence() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        Device.setDefault(device: .gpu)
        let Hk = 4, Hv = 12, Dk = 128, Dv = 128
        for (T, carried) in [(64, false), (191, true), (1200, true)] {
            let q = (MLXRandom.normal([1, T, Hk, Dk]) * 0.01).asType(.bfloat16)
            let k = (MLXRandom.normal([1, T, Hk, Dk]) * 0.09).asType(.bfloat16)
            let v = MLXRandom.normal([1, T, Hv, Dv]).asType(.bfloat16)
            let g = MLXRandom.uniform(low: 0.85, high: 1.0, [1, T, Hv]).asType(.float32)
            let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [1, T, Hv]).asType(.float32)
            let st: MLXArray? = carried ? (MLXRandom.normal([1, Hv, Dv, Dk]) * 0.1).asType(.float32) : nil
            let (y0, s0) = gatedDeltaUpdatePrecomputed(q: q, k: k, v: v, g: g, beta: beta, state: st, mask: nil)
            let r = try XCTUnwrap(q4GDNRows(q: q, k: k, v: v, g: g, beta: beta, state: st), "T \(T): the shape fits")
            XCTAssertTrue(arrayEqual(r.0, y0).item(Bool.self), "T \(T): output differs from the shipped kernel")
            XCTAssertTrue(arrayEqual(r.1, s0).item(Bool.self), "T \(T): next state differs from the shipped kernel")
        }
    }

    /// production's branch (ENGINE_FUSE_MASK without the gdnGate bit) computes the gate with MLX ops inside
    /// gatedDeltaUpdate; q4GDNRowsFromGate must match that call bit for bit, gate math included (C-0629: the first wiring
    /// covered only the gdnGate branch, which production does not take)
    func testRowsFromGateIsBitIdenticalToGatedDeltaUpdate() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        Device.setDefault(device: .gpu)
        let Hk = 4, Hv = 12, Dk = 128, Dv = 128
        for (T, carried) in [(64, false), (1200, true)] {
            let q = (MLXRandom.normal([1, T, Hk, Dk]) * 0.01).asType(.bfloat16)
            let k = (MLXRandom.normal([1, T, Hk, Dk]) * 0.09).asType(.bfloat16)
            let v = MLXRandom.normal([1, T, Hv, Dv]).asType(.bfloat16)
            let a = MLXRandom.normal([1, T, Hv]).asType(.bfloat16)
            let b = MLXRandom.normal([1, T, Hv]).asType(.bfloat16)
            let aLog = (MLXRandom.normal([Hv]) * 0.5).asType(.float32)
            let dtBias = MLXRandom.normal([Hv]).asType(.bfloat16)
            let st: MLXArray? = carried ? (MLXRandom.normal([1, Hv, Dv, Dk]) * 0.1).asType(.float32) : nil
            let (y0, s0) = gatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: st, mask: nil)
            let r = try XCTUnwrap(q4GDNRowsFromGate(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: st))
            XCTAssertTrue(arrayEqual(r.0, y0).item(Bool.self), "T \(T): output differs from gatedDeltaUpdate")
            XCTAssertTrue(arrayEqual(r.1, s0).item(Bool.self), "T \(T): next state differs from gatedDeltaUpdate")
        }
    }

    /// a shape the kernel cannot take (Dv not a multiple of 16) is refused, so the caller keeps the shipped kernel
    func testUnfitShapeIsRefused() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        let q = MLXArray.zeros([1, 8, 2, 32], dtype: .bfloat16)
        let v = MLXArray.zeros([1, 8, 2, 24], dtype: .bfloat16)
        let g = MLXArray.ones([1, 8, 2], dtype: .float32)
        XCTAssertNil(q4GDNRows(q: q, k: q, v: v, g: g, beta: g, state: nil))
    }
}
