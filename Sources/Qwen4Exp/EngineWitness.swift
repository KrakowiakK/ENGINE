// T-0190 (QC-3, C-0606): activation witnesses. A production feature switched on but never actually taken is invisible
// unless something counts it -- packed prefill ran "on" for a whole window with zero packed forwards (C-0595), the vision
// tower ran in fp32 for every build (C-0604). Every feature switch counts the times its path is really taken here; the
// server reports the table (GET /v1/engine/sessions -> witness) and tools/qc/witness_check.py fails a deploy when a switch
// that production turns on has a zero count after a workload that should exercise it.
import Foundation

public enum EngineWitness {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var counts: [String: Int] = [:]
    /// one more use of the path `key` (cheap: one lock, one dictionary add; called per kernel dispatch at most)
    @inline(__always) public static func hit(_ key: String, _ n: Int = 1) {
        lock.lock(); counts[key, default: 0] += n; lock.unlock()
    }
    public static var snapshot: [String: Int] { lock.lock(); defer { lock.unlock() }; return counts }
}
