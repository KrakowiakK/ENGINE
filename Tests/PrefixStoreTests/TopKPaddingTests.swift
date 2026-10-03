import XCTest
import MLX
@testable import Qwen4Exp

/// The QSA block selection pads a row whose block count is not a multiple of the part count with -inf so the two-phase
/// selection runs on every decode step. The padded selection must be the reference selection exactly: every key above
/// the K-th, then the ties at it in index order -- including rows with -inf (masked) blocks and repeated keys.
/// Runs the Metal kernels (GPU); opt in with MLXFAST_RUN_MLX_RUNTIME_TESTS=1.
final class TopKPaddingTests: XCTestCase {
    private func reference(_ row: [Float], k: Int) -> [Int32] {
        row.indices.sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }.prefix(k).map { Int32($0) }.sorted()
    }

    func testPaddedSelectionEqualsTheReferenceOnRowsThatAreNotAMultipleOfTheParts() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1", "GPU kernel test")
        Device.setDefault(device: .gpu)
        var rng = SystemRandomNumberGenerator()
        for n in [32_769, 40_001, 65_538, 70_003] {
            let rows = 4
            var flat: [Float] = []
            for r in 0 ..< rows {
                for i in 0 ..< n {
                    // coarse keys force many ties; the last rows mask their tail to -inf like a verify block
                    var v = Float(Int.random(in: 0 ..< 2000, using: &rng)) / 7
                    if r >= 2 && i >= n - 3 * r { v = -Float.infinity }
                    flat.append(v)
                }
            }
            let scores = MLXArray(flat).reshaped(1, rows, n)
            let top = q4TopK(scores: scores, k: 512)
            eval(top)
            for r in 0 ..< rows {
                let got = top[0, r].asArray(Int32.self).sorted()
                XCTAssertEqual(got, reference(Array(flat[(r * n) ..< ((r + 1) * n)]), k: 512), "n=\(n) row \(r)")
            }
        }
    }
}
