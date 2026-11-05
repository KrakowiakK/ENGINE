import Foundation
import XCTest
@testable import EngineServeSupport

/// T-0092: ENGINE_ADMISSION_PER_ROW -- every row priced at its own length (its forward block bounded by its own prompt)
/// and the rows summed, instead of every row priced at the longest one.
final class AdmissionPerRowTests: XCTestCase {
    private func toy(perRow: Bool, capacity: Double) -> AdmissionBudget {
        AdmissionBudget(maxContext: 100_000, capacityBytes: capacity, bytesPerToken: 1, fixedBytesPerSequence: 0,
                        historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: capacity, historyFactor: 1,
                        perRow: perRow, historyCapacityBytesForRow: { length, _ in Double(length) })
    }

    func testPerRowSumsOwnLengthsInsteadOfPaddingToTheLongest() {
        // one 10_000 row, then 1_000-token rows: padded = n x 10_512; per row = 10_512 + n x 1_512
        let padded = toy(perRow: false, capacity: 40_000), own = toy(perRow: true, capacity: 40_000)
        XCTAssertNotNil(padded.reserve(length: 10_000, current: 5_000) { _, _ in true })
        XCTAssertNotNil(own.reserve(length: 10_000, current: 5_000) { _, _ in true })
        var a = 1, b = 1
        while padded.reserve(length: 1_000, current: 500, prepareCache: { _, _ in true }) != nil { a += 1 }
        while own.reserve(length: 1_000, current: 500, prepareCache: { _, _ in true }) != nil { b += 1 }
        XCTAssertEqual(a, 3, "3 x 10_512 = 31_536 fits 40_000, 4 x does not")
        XCTAssertEqual(b, 20, "10_512 + 19 x 1_512 = 39_240 fits, 20 x does not")
        XCTAssertEqual(own.snapshot()["admission_per_row"], 1)
        XCTAssertEqual(padded.snapshot()["admission_per_row"], 0)
    }

    func testEstimateFitsAgreesWithReserveAndReleaseFreesTheRow() {
        let b = toy(perRow: true, capacity: 5_000)
        let r = b.reserve(length: 3_000, current: 100) { _, _ in true }!
        XCTAssertFalse(b.estimateFits(length: 1_500, prompt: 100), "3_512 + 2_012 > 5_000")
        XCTAssertTrue(b.estimateFits(length: 900, prompt: 100), "3_512 + 1_412 <= 5_000")
        b.release(r)
        XCTAssertTrue(b.estimateFits(length: 1_500, prompt: 100))
    }

    func testRowPriceUsesItsOwnPrompt() {
        let seen = Seen()
        let b = AdmissionBudget(maxContext: 100_000, capacityBytes: 1e12, bytesPerToken: 1, fixedBytesPerSequence: 0,
                                historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 1e12, historyFactor: 1, perRow: true,
                                historyCapacityBytesForRow: { l, p in seen.add(p); return Double(l) })
        _ = b.reserve(length: 9_000, current: 2_000) { _, _ in true }
        XCTAssertTrue(seen.values.contains(2_000), "the prompt passed as `current` prices the row")
        XCTAssertFalse(seen.values.contains(9_000))
    }

    func testHotTargetStaysConsistentWithGrow() {
        let b = AdmissionBudget(maxContext: 100_000, capacityBytes: 50_000, bytesPerToken: 1, fixedBytesPerSequence: 0,
                                historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 50_000, growthWindow: 1_000,
                                historyFactor: 1, perRow: true, historyCapacityBytesForRow: { l, _ in Double(l) })
        let id = b.reserve(length: 10_000, current: 2_000) { _, _ in true }!
        XCTAssertEqual(b.committedLength(id), 3_000)
        XCTAssertEqual(b.snapshot()["active_committed_bytes"], 3_512)
        XCTAssertEqual(b.snapshot()["active_reservation_bytes"], 10_512)
        XCTAssertEqual(b.grow(id, current: 2_600) { _, _ in true }, true)
        XCTAssertEqual(b.snapshot()["active_committed_bytes"], 4_112)
    }

    /// The live mix of 2026-10-16 09:12 (the dashboard's nine rows, max_tokens 7000) on the 397B geometry: 16 attention
    /// layers (15 + MTP), 2 KV heads x 256, a 32.5 GB budget. The shipped formula (padded to the longest row, history x2,
    /// the 8192-row block, the defaulted indexer keys, 1.755 GB fixed) admits exactly the 9 the server ran; per-row with
    /// history x1, the block bounded by the prompt, no indexer keys and keep-rungs 1 (1.56 GB fixed) admits at least 14.
    func testLiveMixAdmitsNineShippedAndAtLeastFourteenPerRow() {
        let bpt = AdmissionBudget.perTokenBytes(attentionLayers: 16, kvHeads: 2, headDim: 256, indexerHeadDim: 128, indexerCompressRatio: 4)
        XCTAssertEqual(bpt, 37_888, "the server's kv_bytes_per_token")
        @Sendable func hist(_ l: Int, block: Int, indexer: Int) -> Double {
            AdmissionBudget.historyCapacityBytes(length: l, attentionLayers: 16, kvHeads: 2, headDim: 256,
                indexerHeadDim: indexer, indexerCompressRatio: 4, kvStep: 256, indexerStep: 1024,
                maxForwardRows: block, capacityGrowthLimit: 262_144)
        }
        let capacity = 32_518_770_647.2
        let prompts = [6_600, 3_400, 2_900, 4_200, 2_900, 3_100, 2_400, 4_000, 2_400]
        let mix = prompts + prompts                                   // keep arriving: the same mix again
        func admitted(_ b: AdmissionBudget) -> Int {
            var n = 0
            for p in mix where b.reserve(length: p + 7_000, current: p, prepareCache: { _, _ in true }) != nil { n += 1 }
            return n
        }
        let shipped = AdmissionBudget(maxContext: 262_144, capacityBytes: capacity, bytesPerToken: bpt,
            fixedBytesPerSequence: 1_755_119_616, historyCapacityBytes: { hist($0, block: 8_192, indexer: 128) },
            hotCacheCeilingBytes: 24e9, growthWindow: 32_768, historyFactor: 2)
        XCTAssertEqual(admitted(shipped), 9, "what the live server admitted")
        let perRow = AdmissionBudget(maxContext: 262_144, capacityBytes: capacity, bytesPerToken: bpt,
            fixedBytesPerSequence: 1_755_119_616 - 196_485_120, historyCapacityBytes: { hist($0, block: 8_192, indexer: 128) },
            hotCacheCeilingBytes: 24e9, growthWindow: 32_768, historyFactor: 1, perRow: true,
            historyCapacityBytesForRow: { l, p in hist(l, block: max(5, min(8_192, p)), indexer: 0) })
        let n = admitted(perRow); print("T-0092 live mix: shipped 9, per-row \(n)")
        XCTAssertGreaterThanOrEqual(n, 14)
    }
}

private final class Seen: @unchecked Sendable {
    private let lock = NSLock()
    private var all: [Int] = []
    func add(_ v: Int) { lock.lock(); all.append(v); lock.unlock() }
    var values: [Int] { lock.lock(); defer { lock.unlock() }; return all }
}
