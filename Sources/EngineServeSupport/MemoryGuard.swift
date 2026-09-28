import Foundation
import Darwin

/// P125: the engine sized itself as if it owned the machine (0.9 x physical memory - 80 GB) and never looked at what other
/// processes use or at the kernel's memory-pressure signal. On a shared Mac (a second model server, a benchmark harness) that
/// ends in swap or in the kernel killing the largest process -- this one -- with no message (a tester's report, 2026-09-27).
/// With `iogpu.disable_wired_collector=1` the kernel cannot take GPU-wired pages back, so the engine has to give memory back
/// itself. This file holds the measurement and the policy; Serve.swift applies it on the model owner.
public enum SystemMemory {
    /// Bytes the kernel can hand out without paging or compressing: free + inactive + purgeable + speculative pages.
    public static func availableBytes() -> Int64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        let page = Int64(getpagesize())          // the kernel page size (16 KiB on Apple silicon)
        return (Int64(stats.free_count) + Int64(stats.inactive_count) + Int64(stats.purgeable_count)
                + Int64(stats.speculative_count)) * page
    }
}

public enum MemoryPressureLevel: Int, Comparable, Sendable {
    case normal = 0, warning = 1, critical = 2
    public static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
    public var name: String { ["normal", "warning", "critical"][rawValue] }
}

/// Pure policy. The level is the worse of the kernel's last pressure event and the available-memory sample (below `lowBytes`
/// -> warning, below `criticalBytes` -> critical). It rises at once and falls only after `recoverSeconds` of a better
/// reading, so a pressure spike cannot make the hot cache refill and drain in a loop.
public struct MemoryGuardPolicy: Sendable {
    public let lowBytes: Int64
    public let criticalBytes: Int64
    public let recoverSeconds: Double
    public private(set) var level: MemoryPressureLevel = .normal
    public private(set) var eventLevel: MemoryPressureLevel = .normal
    public private(set) var lastAvailable: Int64? = nil
    public private(set) var transitions = 0, warnings = 0, criticals = 0
    private var betterSince: Double? = nil

    public init(lowBytes: Int64, criticalBytes: Int64, recoverSeconds: Double = 30) {
        precondition(lowBytes >= criticalBytes && criticalBytes >= 0 && recoverSeconds >= 0)
        self.lowBytes = lowBytes; self.criticalBytes = criticalBytes; self.recoverSeconds = recoverSeconds
    }

    /// Feed a kernel event (nil = none this time) and/or an available-memory sample; returns true when the level changed.
    public mutating func observe(event: MemoryPressureLevel? = nil, available: Int64? = nil, now: Double) -> Bool {
        if let event { eventLevel = event }
        if let available { lastAvailable = available }
        var raw = eventLevel
        if let a = lastAvailable {
            raw = max(raw, a < criticalBytes ? .critical : a < lowBytes ? .warning : .normal)
        }
        if raw > level {
            set(raw); betterSince = nil; return true
        }
        if raw < level {
            if let since = betterSince {
                if now - since >= recoverSeconds { set(raw); betterSince = nil; return true }
            } else {
                betterSince = now
                if recoverSeconds == 0 { set(raw); betterSince = nil; return true }
            }
        } else {
            betterSince = nil
        }
        return false
    }

    private mutating func set(_ l: MemoryPressureLevel) {
        if l == .warning { warnings += 1 }
        if l == .critical { criticals += 1 }
        level = l; transitions += 1
    }

    /// The share of the hot cache's ceiling the level allows: all, half, none.
    public var hotCeilingFactor: Double { [1.0, 0.5, 0.0][level.rawValue] }
    /// New requests are admitted unless the machine is critically short (running ones always finish).
    public var admitNew: Bool { level != .critical }

    /// Defaults for a machine of `physical` bytes: warning below max(16 GB, 3 %), critical below max(6 GB, 1 %).
    public static func defaults(physical: Int64) -> MemoryGuardPolicy {
        MemoryGuardPolicy(lowBytes: max(16_000_000_000, physical * 3 / 100), criticalBytes: max(6_000_000_000, physical / 100))
    }
}
