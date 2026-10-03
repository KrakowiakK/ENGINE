import Foundation

/// Least-recently-used order over string keys with O(1) touch, insert, remove and pop-oldest (a doubly linked list kept
/// in two dictionaries). The template token cache used a plain `[String]`: every HIT did `removeAll { $0 == k }` over all
/// keys under the lock every tokenizing thread shares -- 110 us per hit at 10k keys, 930 us at 100k -- and evicted in
/// insertion order, not use order.
public struct KeyLRU {
    private var prev: [String: String] = [:]
    private var next: [String: String] = [:]
    private var members = Set<String>()
    public private(set) var oldest: String?
    public private(set) var newest: String?
    public init() {}
    public var count: Int { members.count }
    public func contains(_ k: String) -> Bool { members.contains(k) }
    /// Make `k` the most recently used (inserting it when absent).
    public mutating func touch(_ k: String) {
        if members.contains(k) { remove(k) }
        members.insert(k)
        if let t = newest { next[t] = k; prev[k] = t } else { oldest = k }
        newest = k
    }
    public mutating func remove(_ k: String) {
        guard members.remove(k) != nil else { return }
        let p = prev.removeValue(forKey: k), n = next.removeValue(forKey: k)
        if let p { next[p] = n } else { oldest = n }
        if let n { prev[n] = p } else { newest = p }
        if let p, n == nil { next.removeValue(forKey: p) }
    }
    /// Remove and return the least recently used key.
    public mutating func popOldest() -> String? {
        guard let o = oldest else { return nil }
        remove(o)
        return o
    }
    /// Oldest first (tests and diagnostics; O(n)).
    public var ordered: [String] {
        var out: [String] = []; var k = oldest
        while let c = k { out.append(c); k = next[c] }
        return out
    }
}
