import XCTest
import MLX
@testable import Qwen4Exp

/// T-0198: the runtime-row variants (ENGINE_RUNTIME_ROWS=1) take the row count / sequence length from an int32 input
/// instead of the template, so a new prompt length compiles nothing. Their output must be the template kernels' bit for
/// bit, at odd and even lengths. Runs the Metal kernels (GPU); opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
final class RuntimeRowsTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        Device.setDefault(device: .gpu)
    }

    func testHCMixInjectRuntimeRowsIsBitIdentical() {
        let H = 4, D = 256
        for rows in [17, 191, 1247] {
            let normed = MLXRandom.normal([rows, H * D]).asType(.bfloat16)
            let u = MLXRandom.normal([rows, H * D]).asType(.bfloat16)
            let winj = (MLXRandom.normal([H, H * D]) * 0.02).asType(.bfloat16)
            for R in [1, 4] {
                let a = q4HCMixInject(normed: normed, u: u, winj: winj, hc: H, d: D, rows: R, runtime: false)
                let b = q4HCMixInject(normed: normed, u: u, winj: winj, hc: H, d: D, rows: R, runtime: true)
                XCTAssertTrue(arrayEqual(a.0, b.0).item(Bool.self), "rows \(rows) R \(R): mixed differs")
                XCTAssertTrue(arrayEqual(a.1, b.1).item(Bool.self), "rows \(rows) R \(R): inject differs")
            }
        }
    }

    func testRopePartialRuntimeSIsBitIdentical() {
        let HD = 64, ROT = 32
        for S in [17, 191] {
            let x = MLXRandom.normal([1, 4, S, HD]).asType(.bfloat16)
            let c = MLXRandom.normal([S, ROT]).asType(.float32), s = MLXRandom.normal([S, ROT]).asType(.float32)
            let a = q4RopePartial(x, cos: c, sin: s, rotDim: ROT, runtime: false)
            let b = q4RopePartial(x, cos: c, sin: s, rotDim: ROT, runtime: true)
            XCTAssertTrue(arrayEqual(a, b).item(Bool.self), "S \(S): rope differs")
        }
    }

    func testConvPrefillTailRuntimeRowsIsBitIdentical() {
        let convDim = 64, qkvzDim = 96, ck = 4
        for rows in [65, 191, 1247] {
            let qkvz = MLXRandom.normal([rows, qkvzDim]).asType(.bfloat16)
            let cs = MLXRandom.normal([1, ck - 1, convDim]).asType(.bfloat16)
            let w = MLXRandom.normal([convDim, 1, ck]).asType(.bfloat16)
            let a = q4GDNConvPrefillWithTail(qkvz: qkvz, convState: cs, convW: w, convDim: convDim, kernel: ck, runtime: false)
            let b = q4GDNConvPrefillWithTail(qkvz: qkvz, convState: cs, convW: w, convDim: convDim, kernel: ck, runtime: true)
            XCTAssertTrue(arrayEqual(a.0, b.0).item(Bool.self), "rows \(rows): conv output differs")
            XCTAssertTrue(arrayEqual(a.1, b.1).item(Bool.self), "rows \(rows): tail differs")
        }
    }
}
