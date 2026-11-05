import Foundation

/// T-0095: how much of each prompt an earlier prompt already began with -- the reuse a prefix cache could have had at
/// best, whatever the store kept. Hash-only: no token ids are kept, only FNV-1a hashes of each prompt's prefixes at every
/// `step` tokens (up to `maxLength`) and of the whole prompt, for the last `capacity` prompts. `record` returns the longest
/// such prefix an earlier prompt in the window had (a multiple of `step`, or the whole prompt for an exact repeat).
public final class PrefixShareLedger: @unchecked Sendable {
    public let step: Int, capacity: Int, maxLength: Int
    /// The shared-prefix rung length whose savings `snapshot` reports (P119: 1024), and the prompt margin it needs.
    public let rung: Int, rungMargin: Int
    private let lock = NSLock()
    private var seen: [UInt64: Int] = [:]           // prefix key -> id of the newest prompt that had it
    private var window: [(id: Int, keys: [UInt64])] = []
    private var nextId = 0
    private var prompts = 0, promptTokens = 0, sharedTokens = 0, exactRepeats = 0, rungPrompts = 0
    /// Prompts by their longest shared prefix: 0, < 256, < 512, < 1024, < 2048, >= 2048 tokens (exact repeats included).
    private var buckets = [Int](repeating: 0, count: 6)
    private var recent: [[Int]] = []                // [prompt tokens, longest shared prefix] of the last 64 prompts

    public init(step: Int = 64, capacity: Int = 2048, maxLength: Int = 16384, rung: Int = 1024, rungMargin: Int = 256) {
        precondition(step > 0 && capacity > 0 && maxLength >= step)
        self.step = step; self.capacity = capacity; self.maxLength = maxLength; self.rung = rung; self.rungMargin = rungMargin
    }

    private static func key(_ h: UInt64, length: Int, whole: Bool) -> UInt64 {
        var k = h ^ (UInt64(length) &* 0x9E37_79B9_7F4A_7C15)
        if whole { k = ~k }
        return k
    }

    @discardableResult
    public func record(_ ids: [Int]) -> Int {
        guard !ids.isEmpty else { return 0 }
        // prefix keys at step, 2 * step, ... (up to maxLength), then the whole prompt's key when it is within maxLength
        let hashed = min(ids.count, maxLength)
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        var keys: [UInt64] = []
        keys.reserveCapacity(hashed / step + 1)
        for i in 0 ..< hashed {
            var v = UInt64(bitPattern: Int64(ids[i]))
            for _ in 0 ..< 8 { h ^= v & 0xff; h = h &* 0x0000_0100_0000_01b3; v >>= 8 }
            if (i + 1) % step == 0 { keys.append(Self.key(h, length: i + 1, whole: false)) }
        }
        let prefixKeys = keys.count
        let whole = ids.count <= maxLength ? Self.key(h, length: ids.count, whole: true) : nil
        if let whole { keys.append(whole) }
        lock.lock(); defer { lock.unlock() }
        var shared = 0
        if let whole, seen[whole] != nil {
            shared = ids.count
        } else {
            for j in stride(from: prefixKeys - 1, through: 0, by: -1) where seen[keys[j]] != nil { shared = (j + 1) * step; break }
        }
        nextId += 1
        for k in keys { seen[k] = nextId }
        window.append((nextId, keys))
        if window.count > capacity {
            let old = window.removeFirst()
            for k in old.keys where seen[k] == old.id { seen[k] = nil }
        }
        prompts += 1; promptTokens += ids.count; sharedTokens += min(shared, ids.count - 1)
        if shared == ids.count { exactRepeats += 1 }
        if ids.count >= rung + rungMargin, shared >= rung { rungPrompts += 1 }
        let b = shared == 0 ? 0 : shared < 256 ? 1 : shared < 512 ? 2 : shared < 1024 ? 3 : shared < 2048 ? 4 : 5
        buckets[b] += 1
        recent.append([ids.count, shared]); if recent.count > 64 { recent.removeFirst() }
        return shared
    }

    public func snapshot() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return ["prompts": prompts, "prompt_tokens": promptTokens, "shared_prefix_tokens": sharedTokens,
                "exact_repeats": exactRepeats, "step": step, "window": capacity, "max_length": maxLength,
                "rung": rung, "rung_eligible_prompts": rungPrompts, "rung_saveable_tokens": rungPrompts * rung,
                "shared_prefix_histogram": ["0": buckets[0], "1-255": buckets[1], "256-511": buckets[2],
                                            "512-1023": buckets[3], "1024-2047": buckets[4], "2048+": buckets[5]],
                "recent_prompt_shared": recent]
    }
}
