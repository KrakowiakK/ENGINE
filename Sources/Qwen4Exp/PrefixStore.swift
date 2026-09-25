// P085/P086 -- the shared, content-addressed prefix store.
//
// It lives in the model library rather than in the CLI for one reason: it is the piece most likely to
// be wrong in a way no benchmark notices (a wrong HIT returns a plausible answer computed from the
// wrong state), so it has to be unit-testable, and a Swift executable target cannot be imported by a
// test target.
import Foundation
import Darwin
import Crypto
import MLX
import MLXLMCommon

/// File metadata is indexed once, then maintained on this process's writes/reads.
/// A periodic refresh observes external removals/writers; missing blocks remain clean misses.
fileprivate final class PrefixDirectoryIndex: @unchecked Sendable {
    private struct Entry { var size: Double; var used: Date }
    private let lock = NSLock()
    private var entries: [URL: Entry] = [:]
    private var total = 0.0
    private var scannedAt = -Double.infinity
    private func recordLocked(_ url: URL) {
        guard let a = try? FileManager.default.attributesOfItem(atPath: url.path) else { return }
        let size = Double((a[.size] as? NSNumber)?.int64Value ?? 0)
        total += size - (entries[url]?.size ?? 0)
        entries[url] = Entry(size: size, used: a[.modificationDate] as? Date ?? .distantPast)
    }
    func record(_ url: URL) { lock.lock(); defer { lock.unlock() }; recordLocked(url) }
    func touch(_ url: URL) { lock.lock(); defer { lock.unlock() }; if entries[url] != nil { entries[url]?.used = Date() } }
    func evict(dir: URL, cap: Double, grace: Double) -> (before: Double, freed: Double, dropped: Int) {
        lock.lock(); defer { lock.unlock() }
        let now = ProcessInfo.processInfo.systemUptime
        if now - scannedAt >= 30 {
            guard let urls = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return (total, 0, 0) }
            entries.removeAll(keepingCapacity: true); total = 0
            for url in urls where url.lastPathComponent.hasPrefix("st_") || url.lastPathComponent.hasPrefix("blk_") { recordLocked(url) }
            scannedAt = now
        }
        let before = total
        guard total > cap else { return (before, 0, 0) }
        let cutoff = Date().addingTimeInterval(-grace)
        var freed = 0.0, dropped = 0
        for (url, entry) in entries.sorted(by: { $0.value.used < $1.value.used }) {
            if total <= cap * 0.9 || entry.used >= cutoff { break }
            if (try? FileManager.default.removeItem(at: url)) != nil || !FileManager.default.fileExists(atPath: url.path) {
                entries[url] = nil; total -= entry.size; freed += entry.size; dropped += 1
            }
        }
        return (before, freed, dropped)
    }
}

public struct StateCacheConfig: Sendable {
    public let dir: URL; public let identity: String; public let step: Int; public let writeExact: Bool
    /// P086: 0 = unbounded (and then the store is a disk leak); default from ENGINE_STATE_CACHE_MAX_GB
    public var maxBytes: Double = StateCache.maxBytes
    fileprivate var prefixKeys: [Int: String] = [:]
    fileprivate let directoryIndex = PrefixDirectoryIndex()
    public init(dir: URL, identity: String, step: Int, writeExact: Bool, maxBytes: Double = StateCache.maxBytes) {
        precondition(step > 0)
        self.dir = dir; self.identity = identity; self.step = step; self.writeExact = writeExact; self.maxBytes = maxBytes
    }
    /// One linear pass over the prompt; each rung/block reuses a SHA state snapshot.
    /// The indexed configuration is local to this request, never shared mutable state.
    public func indexed(for tokens: [Int]) -> StateCacheConfig {
        var copy = self
        copy.prefixKeys = [:]
        var counts = Set(stride(from: step, through: tokens.count, by: step))
        counts.formUnion(stride(from: max(1, StateCache.blockRows), through: tokens.count, by: max(1, StateCache.blockRows)))
        counts.insert(tokens.count)
        var h = SHA256(); h.update(data: Data(("prefix-v3:" + identity).utf8))
        var previous = 0
        for count in counts.sorted() {
            let part = tokens[previous..<count].map { Int32($0).littleEndian }
            part.withUnsafeBytes { h.update(data: Data($0)) }
            var at = h
            var length = Int64(count).littleEndian
            withUnsafeBytes(of: &length) { at.update(data: Data($0)) }
            copy.prefixKeys[count] = String(at.finalize().map { String(format: "%02x", $0) }.joined().prefix(32))
            previous = count
        }
        return copy
    }

}

public enum StateCache {
    /// What the state depends on besides the tokens: the weights, the config, and the kernel
    /// configuration that produced it. `fusionWitness()` is in the hash because two builds with
    /// different fused paths are different programs (D71) and their states must not collide.
    /// Anything this fails to capture must err toward a MISS, never toward a wrong hit.
    /// Full content digest, once at process startup (never per request). Reads fail
    /// closed. The checkpoint must be immutable while the loaded model is served.
    public static func fileDigest(_ url: URL) throws -> String {
        let f = try FileHandle(forReadingFrom: url)
        defer { try? f.close() }
        _ = fcntl(f.fileDescriptor, F_NOCACHE, 1)
        var h = SHA256()
        // Foundation's read buffers are autoreleased. A server's outer pool may
        // never drain: without this scope, hashing E9 retains the entire 182 GB.
        while try autoreleasepool(invoking: {
            guard let bytes = try f.read(upToCount: 4 << 20), !bytes.isEmpty else { return false }
            h.update(data: bytes)
            return true
        }) {}
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }
    public static func validatedMetallibIdentity(executable: URL, requested: String?) throws -> String {
        let actual = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
        let digest = try fileDigest(actual)
        if let requested, try fileDigest(URL(fileURLWithPath: requested)) != digest {
            throw NSError(domain: "ENGINE", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "MLXFAST_MLX_METALLIB differs from the colocated mlx.metallib that generic MLX actually loads"])
        }
        return digest
    }
    public static func programIdentity() throws -> String {
        var parts = [try fileDigest(URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()),
                     ProcessInfo.processInfo.operatingSystemVersionString, CommandLine.arguments.dropFirst().joined(separator: "\0")]
        // Generic Cmlx loads the colocated library first; MLXFAST_MLX_METALLIB
        // was only a harness hint, not an applied override in this executable.
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        parts.append(try validatedMetallibIdentity(executable: executable,
                     requested: ProcessInfo.processInfo.environment["MLXFAST_MLX_METALLIB"]))
        for (key, value) in ProcessInfo.processInfo.environment.sorted(by: { $0.key < $1.key })
            where key.hasPrefix("ENGINE_") || key.hasPrefix("MLX_") || key.hasPrefix("MLXFAST_") {
            parts.append(key + "=" + value)
        }
        return SHA256.hash(data: Data(parts.joined(separator: "\0").utf8)).map { String(format: "%02x", $0) }.joined()
    }
    public static func identity(dir: URL, witness: String, program: String? = nil) throws -> String {
        var h = SHA256()
        let runtime = try program ?? programIdentity()
        h.update(data: Data("state-v3:block=\(blockRows)\0\(runtime)\0\(witness)\0".utf8))
        h.update(data: try Data(contentsOf: dir.appendingPathComponent("config.json")))
        let fm = FileManager.default
        let names = try fm.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !names.isEmpty else { throw CocoaError(.fileReadNoSuchFile) }
        for name in names {
            let url = dir.appendingPathComponent(name)
            h.update(data: Data((name + "\0" + (try fileDigest(url)) + "\0").utf8))
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// v3 puts length AFTER the token bytes, allowing incremental SHA snapshots.
    public static func key(identity: String, tokens: [Int], count M: Int) -> String {
        precondition(M >= 0 && M <= tokens.count)
        var h = SHA256()
        h.update(data: Data(("prefix-v3:" + identity).utf8))
        let buf = tokens.prefix(M).map { Int32($0).littleEndian }
        buf.withUnsafeBytes { h.update(data: Data($0)) }
        var m = Int64(M).littleEndian
        withUnsafeBytes(of: &m) { h.update(data: Data($0)) }
        return String(h.finalize().map { String(format: "%02x", $0) }.joined().prefix(32))
    }

    public static func url(_ c: StateCacheConfig, _ tokens: [Int], _ M: Int) -> URL {
        c.dir.appendingPathComponent("st_\(c.prefixKeys[M] ?? key(identity: c.identity, tokens: tokens, count: M)).safetensors")
    }

    /// The longest STORED prefix of `tokens`: the exact length first (that state answers with no
    /// prefill at all), then the rungs downward. nil is a clean miss.
    public static func lookup(_ c: StateCacheConfig, _ tokens: [Int], minCount: Int = 1, maxCount: Int? = nil) -> (URL, Int)? {
        let fm = FileManager.default
        var candidates: [Int] = []
        let limit = min(tokens.count, maxCount ?? tokens.count)
        guard limit >= minCount else { return nil }
        if limit == tokens.count, !tokens.isEmpty { candidates.append(tokens.count) }
        var m = (limit / c.step) * c.step
        while m >= max(c.step, minCount) { if m != tokens.count || candidates.isEmpty { candidates.append(m) }; m -= c.step }
        for M in candidates {
            let u = url(c, tokens, M)
            if fm.fileExists(atPath: u.path) { return (u, M) }
        }
        return nil
    }

    // ---- P085: BLOCKS. A rung's state splits in two along the only line this model allows.
    //
    // `L*.a*` is the GDN recurrent state and the PLE/n-gram context: FIXED SIZE, independent of how
    // many tokens preceded it, and NOT reconstructible from pieces -- a recurrence has no prefix
    // decomposition, so it can only be snapshotted AT a position. Measured at 8192 tokens it is
    // 158 MB, 40% of the snapshot, and it stays 158 MB at 128k.
    //
    // `L*.k`, `L*.v`, `L*.i0` are POSITION-INDEXED ROWS: row r of the rung at M and row r of the rung
    // at M + step are the same bytes. Storing each rung whole therefore rewrites the whole prefix
    // every step -- which is why a 128k conversation cost ~119 GB. Those rows are written ONCE, in
    // blocks keyed by the hash of the token prefix that ENDS the block, so the store deduplicates
    // within a session and SHARES between sessions: two agents with the same system prompt reuse the
    // same block files without knowing about each other.
    /// var, not let, so a test can pin it; production reads it once from the environment.
    nonisolated(unsafe) public static var blockRows: Int = Int(ProcessInfo.processInfo.environment["ENGINE_STATE_CACHE_BLOCK"] ?? "2048") ?? 2048
    public static func blockURL(_ c: StateCacheConfig, _ tokens: [Int], _ end: Int) -> URL {
        c.dir.appendingPathComponent("blk_\(c.prefixKeys[end] ?? key(identity: c.identity, tokens: tokens, count: end)).safetensors")
    }
    /// axis the row index runs along, per array suffix (nil = not a row array)
    public static func rowAxis(_ name: String) -> Int? {
        if name.hasSuffix(".k") || name.hasSuffix(".v") { return 2 }
        if name.hasSuffix(".i0") { return 1 }
        return nil
    }

    private static func saveAtomically(_ arrays: [String: MLXArray], to url: URL) throws {
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".writing-" + UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try save(arrays: arrays, url: temporary)
        try FileManager.default.moveItem(at: temporary, to: url)
    }

    public static func write(_ c: StateCacheConfig, _ tokens: [Int], _ M: Int, _ d: [String: MLXArray]) {
        let u = url(c, tokens, M)
        guard !FileManager.default.fileExists(atPath: u.path) else { return }
        let t0 = Date()
        let B = max(1, blockRows)
        var head: [String: MLXArray] = [:]              // everything that is not rows
        var rows: [String: MLXArray] = [:]
        for (k, v) in d { if rowAxis(k) != nil { rows[k] = v } else { head[k] = v } }
        // the rows that fall short of a whole block ride in the head: they are at most B rows
        let whole = (M / B) * B
        for (k, v) in rows {
            let ax = rowAxis(k)!
            let have = v.dim(ax)
            let lo = min(whole, have), hi = min(M, have)
            if hi > lo { head[k + ".tail"] = ax == 2 ? v[0..., 0..., lo ..< hi, 0...] : v[0..., lo ..< hi, 0...] }
        }
        head["blocked"] = MLXArray(Int32(B))
        var wroteBlocks = 0, blockBytes = 0.0
        for b in 0 ..< (whole / B) {
            let end = (b + 1) * B
            let bu = blockURL(c, tokens, end)
            if FileManager.default.fileExists(atPath: bu.path) { continue }
            var bd: [String: MLXArray] = [:]
            for (k, v) in rows {
                let ax = rowAxis(k)!
                guard v.dim(ax) >= end else { continue }
                bd[k] = ax == 2 ? v[0..., 0..., (end - B) ..< end, 0...] : v[0..., (end - B) ..< end, 0...]
            }
            guard !bd.isEmpty else { continue }
            do { try saveAtomically(bd, to: bu); c.directoryIndex.record(bu); wroteBlocks += 1
                 blockBytes += Double((try? FileManager.default.attributesOfItem(atPath: bu.path)[.size] as? Int) as? Int ?? 0)
            } catch { FileHandle.standardError.write("engine: prefix block write failed at \(end): \(error)\n".data(using: .utf8)!) }
        }
        do { try saveAtomically(head, to: u); c.directoryIndex.record(u) } catch {
            FileHandle.standardError.write("engine: state cache write failed at \(M): \(error)\n".data(using: .utf8)!); return
        }
        let sz = Double((try? FileManager.default.attributesOfItem(atPath: u.path)[.size] as? Int) as? Int ?? 0) / 1e9
        FileHandle.standardError.write(String(format: "engine: state cache rung %d: head %.2f GB + %d new block(s) %.2f GB in %.1f s\n",
                                              M, sz, wroteBlocks, blockBytes / 1e9, Date().timeIntervalSince(t0)).data(using: .utf8)!)
        evictIfOver(c, cap: c.maxBytes, indexed: true)
    }

    // ---- P086: LRU EVICTION. The store was an unbounded disk leak (a single 128k conversation is
    // 16 GB), so it carries a cap and drops the least recently USED file when it is over.
    //
    // Two properties of this store make plain LRU the right policy rather than something cleverer.
    // A block is shared by every rung beyond it and by every session with the same prefix, so the
    // early blocks -- the ones a system prompt lives in -- are touched by every single restore and
    // are therefore the last things LRU will ever drop. And dropping a block a rung needs is SAFE:
    // `load` returns nil, the request takes a clean miss and re-prefills. So eviction can never
    // corrupt a restore, only cost one.
    public static let maxBytes: Double = (Double(ProcessInfo.processInfo.environment["ENGINE_STATE_CACHE_MAX_GB"] ?? "") ?? 64) * 1e9
    /// LRU needs a read to count as a use, and the filesystem is the index: touch on every hit.
    public static func touch(_ u: URL) {
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: u.path)
    }
    /// Drop least-recently-used entries until the store is under 90% of the cap (the slack keeps a
    /// store that sits exactly at the line from evicting on every single write).
    public static func evictIfOver(_ c: StateCacheConfig, cap: Double, graceSeconds: Double? = nil, indexed: Bool = false) {
        guard cap > 0 else { return }
        let grace = graceSeconds ?? (Double(ProcessInfo.processInfo.environment["ENGINE_STATE_CACHE_GRACE_S"] ?? "") ?? 300)
        let index = indexed ? c.directoryIndex : PrefixDirectoryIndex()
        let result = index.evict(dir: c.dir, cap: cap, grace: grace)
        if result.dropped > 0 {
            FileHandle.standardError.write(Data(String(format: "engine: prefix eviction %d files, %.2f GB (before %.2f / cap %.2f GB)\n",
                                                       result.dropped, result.freed / 1e9, result.before / 1e9, cap / 1e9).utf8))
        }
    }

    /// Reassemble a rung: the head, plus its row arrays rebuilt from the shared blocks. Returns nil
    /// when a block the rung needs has been evicted, which is a clean MISS, never a partial restore.
    public static func load(_ c: StateCacheConfig, _ tokens: [Int], _ M: Int, _ u: URL) -> [String: MLXArray]? {
        guard let head = try? loadArrays(url: u) else { return nil }
        touch(u); c.directoryIndex.touch(u)
        guard let bm = head["blocked"] else { return head }          // pre-P085 whole-rung entry
        let B = Int(bm.item(Int32.self))
        guard B > 0, B == blockRows, M > 0, M <= tokens.count else { return nil }
        let whole = (M / B) * B
        var blocks: [[String: MLXArray]] = []
        blocks.reserveCapacity(whole / B)
        for b in 0 ..< (whole / B) {
            let bu = blockURL(c, tokens, (b + 1) * B)
            guard let bd = try? loadArrays(url: bu) else { return nil }
            touch(bu); c.directoryIndex.touch(bu)
            blocks.append(bd)
        }
        var out: [String: MLXArray] = [:]
        for (k, v) in head where k != "blocked" && !k.hasSuffix(".tail") { out[k] = v }
        var names = Set<String>()
        for b in blocks { for k in b.keys { names.insert(k) } }
        for k in head.keys where k.hasSuffix(".tail") { names.insert(String(k.dropLast(5))) }
        for name in names {
            guard let ax = rowAxis(name) else { continue }
            var parts = blocks.compactMap { $0[name] }
            if let t = head[name + ".tail"] { parts.append(t) }
            guard !parts.isEmpty else { continue }
            out[name] = parts.count == 1 ? parts[0] : concatenated(parts, axis: ax)
        }
        return out
    }
}
