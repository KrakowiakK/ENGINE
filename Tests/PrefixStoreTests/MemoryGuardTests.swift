import Foundation
import XCTest
@testable import EngineServeSupport

/// P125: the memory guard's policy (pure) and its hold on the admission budget.
final class MemoryGuardTests: XCTestCase {
    private let GB: Int64 = 1_000_000_000

    func testAvailableMemoryIsReadable() {
        let a = SystemMemory.availableBytes()
        XCTAssertNotNil(a)
        XCTAssertGreaterThan(a ?? 0, 0)
        XCTAssertLessThanOrEqual(a ?? 0, Int64(ProcessInfo.processInfo.physicalMemory))
    }

    func testDefaultsScaleWithTheMachine() {
        let big = MemoryGuardPolicy.defaults(physical: 550 * GB)
        XCTAssertEqual(big.lowBytes, 16_500_000_000)
        XCTAssertEqual(big.criticalBytes, 6_000_000_000)
        let small = MemoryGuardPolicy.defaults(physical: 64 * GB)
        XCTAssertEqual(small.lowBytes, 16 * GB)
        XCTAssertEqual(small.criticalBytes, 6 * GB)
    }

    func testRisesAtOnceAndFallsOnlyAfterTheRecoveryTime() {
        var p = MemoryGuardPolicy(lowBytes: 20 * GB, criticalBytes: 5 * GB, recoverSeconds: 30)
        XCTAssertFalse(p.observe(available: 100 * GB, now: 0))
        XCTAssertEqual(p.level, .normal)
        XCTAssertEqual(p.hotCeilingFactor, 1.0); XCTAssertTrue(p.admitNew)
        XCTAssertTrue(p.observe(available: 10 * GB, now: 1))
        XCTAssertEqual(p.level, .warning); XCTAssertEqual(p.hotCeilingFactor, 0.5); XCTAssertTrue(p.admitNew)
        XCTAssertTrue(p.observe(available: 2 * GB, now: 2))
        XCTAssertEqual(p.level, .critical); XCTAssertEqual(p.hotCeilingFactor, 0.0); XCTAssertFalse(p.admitNew)
        // better readings: no change before 30 s, then straight to what the reading says
        XCTAssertFalse(p.observe(available: 100 * GB, now: 3))
        XCTAssertFalse(p.observe(available: 100 * GB, now: 32))
        XCTAssertEqual(p.level, .critical)
        XCTAssertTrue(p.observe(available: 100 * GB, now: 33))
        XCTAssertEqual(p.level, .normal)
        XCTAssertEqual(p.warnings, 1); XCTAssertEqual(p.criticals, 1); XCTAssertEqual(p.transitions, 3)
    }

    func testABadReadingDuringRecoveryRestartsTheClock() {
        var p = MemoryGuardPolicy(lowBytes: 20 * GB, criticalBytes: 5 * GB, recoverSeconds: 30)
        _ = p.observe(available: 10 * GB, now: 0)
        _ = p.observe(available: 100 * GB, now: 1)
        _ = p.observe(available: 10 * GB, now: 20)          // warning again: the calm period is broken
        XCTAssertFalse(p.observe(available: 100 * GB, now: 40))
        XCTAssertEqual(p.level, .warning)
        XCTAssertTrue(p.observe(available: 100 * GB, now: 70))
        XCTAssertEqual(p.level, .normal)
    }

    func testTheKernelEventAndTheSampleCombineToTheWorse() {
        var p = MemoryGuardPolicy(lowBytes: 20 * GB, criticalBytes: 5 * GB, recoverSeconds: 0)
        XCTAssertTrue(p.observe(event: .critical, available: 100 * GB, now: 0))
        XCTAssertEqual(p.level, .critical, "the kernel says critical although free memory looks fine")
        XCTAssertTrue(p.observe(event: .normal, now: 1))
        XCTAssertEqual(p.level, .normal)
        XCTAssertTrue(p.observe(event: .normal, available: 10 * GB, now: 2))
        XCTAssertEqual(p.level, .warning, "the sample says warning although the kernel is calm")
    }

    func testPressureShrinksTheHotTargetAndPausesNewAdmissions() {
        let b = AdmissionBudget(maxContext: 1000, capacityBytes: 20_000, bytesPerToken: 1, fixedBytesPerSequence: 0,
                                historyCapacityBytes: { Double($0) }, hotCacheCeilingBytes: 10_000)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 10_000)
        b.setPressure(ceilingFactor: 0.5, admissionsPaused: false)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 5_000)
        var seen: Double = -1
        let id = b.reserve(length: 100) { t, _ in seen = t; return true }
        XCTAssertNotNil(id)
        XCTAssertEqual(seen, 5_000, "a reservation under pressure prepares the halved target")
        b.setPressure(ceilingFactor: 0, admissionsPaused: true)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 0)
        XCTAssertNil(b.reserve(length: 100) { _, _ in true }, "critical: no new admission")
        XCTAssertTrue(b.admissionsPaused)
        XCTAssertEqual(b.snapshot()["pressure_refusals"], 1)
        b.release(id!)
        b.setPressure(ceilingFactor: 1, admissionsPaused: false)
        XCTAssertEqual(b.snapshot()["hot_target_bytes"], 10_000)
        XCTAssertNotNil(b.reserve(length: 100) { _, _ in true })
    }
}
