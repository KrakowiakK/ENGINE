import XCTest
import MLX
@testable import Qwen4Exp

/// T-0156: the fused prefill conv with its next conv state written by the same kernel (`q4GDNConvPrefillWithTail`) must
/// give the old kernel's output bit for bit and, as the tail, exactly the last CK-1 rows of the mixed q|k|v columns of
/// qkvz -- the value the graph path stores (`mixedQKV[S-3...]`). Odd row counts (a chunk resumed at 191) included.
/// Runs the Metal kernels (GPU); opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
final class GDNConvPrefillTailTests: XCTestCase {
    func testTailIsTheLastMixedRowsAndTheOutputIsTheOldKernels() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        Device.setDefault(device: .gpu)
        let convDim = 64, qkvzDim = 96, ck = 4
        for rows in [65, 191, 1247, 1248] {
            let qkvz = MLXRandom.normal([rows, qkvzDim]).asType(.bfloat16)
            let cs = MLXRandom.normal([1, ck - 1, convDim]).asType(.bfloat16)
            let w = MLXRandom.normal([convDim, 1, ck]).asType(.bfloat16)
            let old = q4GDNConvPrefill(qkvz: qkvz, convState: cs, convW: w, convDim: convDim, kernel: ck)
            let (y, tail) = q4GDNConvPrefillWithTail(qkvz: qkvz, convState: cs, convW: w, convDim: convDim, kernel: ck)
            let want = qkvz[(rows - (ck - 1)) ..< rows, 0 ..< convDim]
            XCTAssertTrue(arrayEqual(y, old).item(Bool.self), "rows \(rows): conv output differs from the old kernel")
            XCTAssertTrue(arrayEqual(tail, want).item(Bool.self), "rows \(rows): tail is not the last mixed rows")
            XCTAssertEqual(tail.shape, [ck - 1, convDim])
        }
    }
}
