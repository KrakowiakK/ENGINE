import Foundation
import XCTest
import MLX
@testable import EngineServeSupport
@testable import Qwen4Exp

final class SharedRAMBudgetTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    private final class Results: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int] = []
        func add(_ value: Int) { lock.lock(); values.append(value); lock.unlock() }
        func snapshot() -> [Int] { lock.lock(); defer { lock.unlock() }; return values }
    }

    func testConcurrentReservationsDoNotSpendTheSameCacheCreditAndReleaseRestoresTarget() {
        let one = 3.0 * (1000 + 512)
        let budget = AdmissionBudget(maxContext: 1000, capacityBytes: one * 8, bytesPerToken: 1,
            fixedBytesPerSequence: 0, historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: one * 8)
        let granted = Results()
        DispatchQueue.concurrentPerform(iterations: 32) { _ in
            if let id = budget.reserve(length: 1000, prepareCache: { target, growth in
                XCTAssertGreaterThanOrEqual(target, 0)
                XCTAssertEqual(growth, one)
                return true
            }) { granted.add(id) }
        }
        let ids = granted.snapshot()
        XCTAssertEqual(ids.count, 8)
        XCTAssertEqual(budget.snapshot()["active_reservation_bytes"], one * 8)
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"], 0)
        DispatchQueue.concurrentPerform(iterations: ids.count) { budget.release(ids[$0]) }
        XCTAssertEqual(budget.active, 0)
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"], one * 8)
    }

    func testInterleavedConcurrentAdmitAndReleaseNeverOverspends() {
        let budget = AdmissionBudget(maxContext: 1000, capacityBytes: 30_000, bytesPerToken: 1,
            fixedBytesPerSequence: 0, historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 20_000)
        let granted = Results()
        DispatchQueue.concurrentPerform(iterations: 32) { worker in
            for turn in 0..<16 {
                if let id = budget.reserve(length: (worker + turn).isMultiple(of: 2) ? 1000 : 100,
                                            prepareCache: { _, _ in true }) {
                    granted.add(id)
                    XCTAssertLessThanOrEqual(budget.snapshot()["active_reservation_bytes"]!, 30_000)
                    budget.release(id)
                }
            }
        }
        XCTAssertFalse(granted.snapshot().isEmpty)
        XCTAssertEqual(budget.active, 0)
        XCTAssertEqual(budget.snapshot()["active_reservation_bytes"], 0)
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"], 20_000)
    }

    func testLongestReservationRepricesEveryRowAndRejectedPreparationDoesNotCommit() {
        let budget = AdmissionBudget(maxContext: 1000, capacityBytes: 20_000, bytesPerToken: 1,
            fixedBytesPerSequence: 0, historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 18_000)
        XCTAssertNil(budget.reserve(length: 100), "shared mode may not bypass owner preparation")
        let first = budget.reserve(length: 100, prepareCache: { target, growth in
            XCTAssertEqual(target, 18_000); XCTAssertEqual(growth, 1836); return true
        })!
        let before = budget.snapshot()
        XCTAssertNil(budget.reserve(length: 1000, prepareCache: { target, growth in
            XCTAssertEqual(target, 10_928); XCTAssertEqual(growth, 7236); return false
        }))
        XCTAssertEqual(budget.snapshot(), before)
        let second = budget.reserve(length: 1000, prepareCache: { _, _ in true })!
        XCTAssertEqual(budget.snapshot()["active_reservation_bytes"], 9072)
        XCTAssertEqual(budget.snapshot()["max_length"], 1000)
        budget.release(second)
        XCTAssertEqual(budget.snapshot()["active_reservation_bytes"], 1836)
        XCTAssertEqual(budget.snapshot()["max_length"], 100)
        budget.release(first)
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"], 18_000)
    }

    func testFullB8BoundStaysUnchangedAndCacheTargetShrinksThenRecovers() {
        let history: @Sendable (Int) -> Double = { length in
            AdmissionBudget.historyCapacityBytes(length: length, attentionLayers: 13,
                kvHeads: 2, headDim: 256, indexerHeadDim: 128, indexerCompressRatio: 4,
                capacityGrowthLimit: 262_144)
        }
        let envelope = 219_542_702_835.2 // conditional arithmetic fixture, not a runtime memory measurement
        let budget = AdmissionBudget(maxContext: 262_144, capacityBytes: envelope,
            bytesPerToken: 30_784, historyCapacityBytes: history, hotCacheCeilingBytes: 80e9)
        var ids: [Int] = []
        for _ in 0..<8 { ids.append(budget.reserve(length: 262_144, prepareCache: { _, _ in true })!) }
        XCTAssertEqual(budget.snapshot()["active_reservation_bytes"], 197_126_455_296)
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"]!, envelope - 197_126_455_296, accuracy: 0.001)
        budget.release(ids.removeLast())
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"]!, envelope - 172_485_648_384, accuracy: 0.001)
        ids.forEach { budget.release($0) }
        XCTAssertEqual(budget.snapshot()["hot_target_bytes"], 80e9)
    }

    func testPrepareAndReleaseOrderingOnModelOwnerReturnsOnlyPlainReadback() {
        let mt = ModelThread(minBatch: 1, gatherWindow: 0, maxBatch: 8) { $0.map { _ in 0 } }
        defer { mt.shutdown() }
        let budget = AdmissionBudget(maxContext: 1000, capacityBytes: 20_000, bytesPerToken: 1,
            fixedBytesPerSequence: 0, historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 18_000)
        let id: Int = mt.exclusive {
            budget.reserve(length: 1000, prepareCache: { _, _ in
                XCTAssertTrue(mt.isOwner)
                return true
            })!
        }
        let retained: [String: Double] = mt.exclusive {
            XCTAssertTrue(mt.isOwner)
            // The real handler releases after its SeqState teardown. Delaying the
            // release preserves credit while teardown is still outstanding.
            XCTAssertEqual(budget.active, 1)
            return budget.snapshot()
        }
        mt.exclusive { budget.release(id) }
        XCTAssertEqual(retained["active_reservations"], 1)
        XCTAssertEqual(budget.snapshot()["active_reservations"], 0)
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: retained))
    }

    private func row(_ length: Int, salt: Int = 0) -> MLXArray {
        MLXArray((0..<(length * 16)).map { Float($0 + salt) }).reshaped([1, 1, length, 16])
    }
    private func head(_ length: Int) -> HotPrefixStore.Rung {
        HotPrefixStore.rung(fromExported: ["trunk.L1.a1": MLXArray([Float(length)])], length: length, ratio: 4)
    }
    private func store(_ cache: HotPrefixStore, _ tokens: [Int], _ array: MLXArray,
                       inFlight: Bool = false, owner: Int = -1) {
        cache.store(tokens: tokens, rows: ["trunk.L0.k": array], rowsValidTo: tokens.count,
                    mtpValidTo: 0, rungs: [head(tokens.count)], inFlight: inFlight, owner: owner)
    }
    private func address(_ array: MLXArray) -> UInt {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    }

    func testStrictCacheRejectsOversizeBeforeCopyAndKeepsExistingPredecessor() {
        let cache = HotPrefixStore(capBytes: 300_000, strictBudget: true)
        let tokens = Array(0..<64)
        store(cache, tokens, row(64))
        let copies = cache.copyAttempts
        let charged = cache.chargedBytes
        store(cache, Array(0..<4096), row(4096))
        XCTAssertEqual(cache.rejectedStores, 1)
        XCTAssertEqual(cache.copyAttempts, copies)
        XCTAssertEqual(cache.chargedBytes, charged)
        XCTAssertEqual(cache.lookup(tokens + [99])?.length, 64)
        XCTAssertTrue(cache.setBudgetBytes(0))
        XCTAssertEqual(cache.count, 0)
        XCTAssertEqual(cache.chargedBytes, 0)
    }

    func testStrictReplacementAndInflightEvictBeforeAllocationAndReadbackIsPlain() {
        // B41: the continuation inherits its predecessor's rung inside their common prefix (one more head array,
        // one more backing slack), so the ceiling that fits exactly two entries grew by that one rung.
        let cache = HotPrefixStore(capBytes: 220_000 + max(16_384, 3 * Int(getpagesize())) + 4, strictBudget: true)
        store(cache, Array(0..<64), row(64))
        store(cache, Array(0..<80), row(80))
        XCTAssertEqual(cache.count, 1, "a finished continuation replaces its predecessor")
        XCTAssertEqual(cache.entries[0].rungs.map(\.length), [64, 80], "and inherits its rung inside the common prefix")
        store(cache, Array(200..<264), row(64, salt: 10), inFlight: true, owner: 9)
        XCTAssertEqual(cache.count, 2, "in-flight storage consumes the same ceiling")
        let target = cache.chargedBytes
        XCTAssertTrue(cache.setBudgetBytes(target))
        cache.permitAllocation = { incoming in
            XCTAssertLessThanOrEqual(cache.chargedBytes + incoming, cache.budgetBytes)
            XCTAssertGreaterThan(cache.preCopyEvictions, 0, "reclaim happens before allocating this copy")
            return true
        }
        store(cache, Array(400..<464), row(64, salt: 20))
        XCTAssertLessThanOrEqual(cache.chargedBytes, target)
        let snapshot = cache.snapshot()
        XCTAssertEqual(snapshot["charged_bytes"], cache.chargedBytes)
        XCTAssertEqual(snapshot["logical_bytes"], cache.totalBytes)
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: snapshot))
        cache.permitAllocation = nil // avoid retaining the cache from the test observer
        cache.reapInFlight(owner: 9)
        XCTAssertEqual(cache.snapshot()["inflight_entries"], 0)
    }

    /// A long prefill stores its state at every chunk end for coalescing: each store replaces the same request's
    /// previous partial entry, so one prompt never holds more than one partial copy (nor evicts others for its own).
    func testInflightStoreReplacesTheSameRequestsPreviousPartialEntry() {
        let cache = HotPrefixStore(capBytes: 10_000_000, strictBudget: true)
        store(cache, Array(500..<564), row(64, salt: 5))
        for end in stride(from: 16, through: 64, by: 16) {
            store(cache, Array(0..<end), row(end), inFlight: true, owner: 4)
        }
        store(cache, Array(0..<32), row(32, salt: 7), inFlight: true, owner: 8)
        XCTAssertEqual(cache.entries.filter { $0.inFlight && $0.owner == 4 }.map(\.tokens.count), [64])
        XCTAssertEqual(cache.snapshot()["inflight_entries"], 2, "another request's partial entry stays")
        XCTAssertEqual(cache.snapshot()["inflight_replaced"], 3)
        XCTAssertEqual(cache.count, 3, "the finished entry is untouched")
    }

    func testPhysicalReclaimRefusalSkipsCacheCopy() {
        let cache = HotPrefixStore(capBytes: 300_000, strictBudget: true)
        cache.permitAllocation = { _ in false }
        store(cache, Array(0..<64), row(64))
        XCTAssertEqual(cache.count, 0)
        XCTAssertEqual(cache.rejectedStores, 1)
        XCTAssertEqual(cache.copyAttempts, 0)
    }

    func testBorrowedShortPrefixDetachesBeforeDonorEvictionWithExactValues() throws {
        let full = row(4096)
        eval(full)
        let tokens = Array(0..<4096)
        let rungs = [head(64), head(4096)]
        let legacy = HotPrefixStore(capBytes: 1 << 20)
        legacy.store(tokens: tokens, rows: ["trunk.L0.k": full], rowsValidTo: 4096,
                     mtpValidTo: 0, rungs: rungs)
        let prefix = Array(tokens.prefix(64)) + [-1]
        let legacyHit = try XCTUnwrap(legacy.lookup(prefix))
        let alias = try XCTUnwrap(legacy.exported(legacyHit, ratio: 4)?["trunk.L0.k"])
        eval(alias)
        XCTAssertEqual(address(alias), address(full), "control pins full backing through a small prefix")
        let cache = HotPrefixStore(capBytes: 1 << 20, strictBudget: true)
        cache.store(tokens: tokens, rows: ["trunk.L0.k": full], rowsValidTo: 4096,
                    mtpValidTo: 0, rungs: rungs)
        let hit = try XCTUnwrap(cache.lookup(prefix))
        let detached = try XCTUnwrap(cache.exported(hit, ratio: 4)?["trunk.L0.k"])
        XCTAssertNotEqual(address(detached), address(full))
        XCTAssertEqual(detached.asData().data, alias.asData().data)
        XCTAssertTrue(cache.setBudgetBytes(0))
        XCTAssertEqual(cache.count, 0)
        XCTAssertEqual(detached.asData().data, alias.asData().data)
    }
}
