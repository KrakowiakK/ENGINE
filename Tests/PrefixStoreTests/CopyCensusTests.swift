import XCTest
@testable import EngineServeSupport

/// T-0217: the content-free copy census (what share of a request's output a prompt/output n-gram lookup would have drafted)
final class CopyCensusTests: XCTestCase {
    /// an output that re-types a span of the prompt: after the first MIN tokens, lookups draft the rest
    func testOutputCopiedFromThePromptIsDraftedByLookups() {
        let prompt = Array(100 ..< 400)
        let output = Array(150 ..< 250)            // a 100-token span of the prompt, written back
        let c = copyCensusWalk(prompt: prompt, output: output, minMatch: 4, startLen: 2, maxLen: 8)
        XCTAssertGreaterThan(c.lookupRounds, 0)
        XCTAssertGreaterThanOrEqual(c.lookupTokens, 90)                   // all but the first few tokens
        XCTAssertGreaterThanOrEqual(c.accepted, c.drafted - 8)            // every lookup draft is right but the last round's,
                                                                          // which may run past the output's end
        XCTAssertLessThanOrEqual(c.lookupTokens, output.count - 1)        // output[0] comes from the prefill, never a lookup
    }

    /// an output with no repeated n-gram anywhere in the context: no lookup ever fires
    func testOutputWithNoRepeatsHasNoLookups() {
        let prompt = Array(0 ..< 300)
        let output = Array(1000 ..< 1100)
        let c = copyCensusWalk(prompt: prompt, output: output, minMatch: 4, startLen: 2, maxLen: 8)
        XCTAssertEqual(c.lookupRounds, 0)
        XCTAssertEqual(c.lookupTokens, 0)
        XCTAssertEqual(c.drafted, 0)
    }

    /// a lookup round commits its accepted prefix + 1 and stops at the first mismatch; the block doubles only after a
    /// fully accepted round (L0 2 -> 4 -> 8), as the serial path's copy mode does
    func testLookupStopsAtTheFirstMismatchAndDoublesOnlyWhenFullyAccepted() {
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        // writes back 1..8, then diverges (99) where the prompt has 9
        let output = [1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97]
        let c = copyCensusWalk(prompt: prompt, output: output, minMatch: 4, startLen: 2, maxLen: 8)
        // at j=3 (context ends 1,2,3,4): lookup L=2 drafts [5,6] -> both right, commits 3 (5,6,7)
        // at j=6: full, L=4 drafts [8,9,10,11] -> 8 right, 9 wrong: accepted 1, commits 2 (8, 99)
        XCTAssertEqual(c.lookupRounds, 2)
        XCTAssertEqual(c.drafted, 6)
        XCTAssertEqual(c.accepted, 3)
        XCTAssertEqual(c.lookupTokens, 5)
    }

    /// the accumulator sums requests and exposes only counts
    func testTotalsSumRequestsAndHoldOnlyCounts() {
        let t = CopyCensusTotals(configs: [CopyCensusConfig(minMatch: 4, startLen: 2, maxLen: 8)])
        t.record(prompt: Array(100 ..< 400), output: Array(150 ..< 250))
        t.record(prompt: Array(0 ..< 300), output: Array(1000 ..< 1100))
        let s = t.snapshot()
        XCTAssertEqual(s["requests"] as? Int, 2)
        XCTAssertEqual(s["output_tokens"] as? Int, 200)
        let cfgs = s["configs"] as? [[String: Any]]
        XCTAssertEqual(cfgs?.count, 1)
        XCTAssertEqual(cfgs?.first?["name"] as? String, "min4_l2_max8")
        XCTAssertGreaterThanOrEqual(cfgs?.first?["lookup_tokens"] as? Int ?? 0, 90)
    }

    // MARK: T-0221 -- totals per client label, persisted across restarts (counts only)

    private func tempFile() -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("copy-census-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("census.json")
    }
    private let cfgs = [CopyCensusConfig(minMatch: 4, startLen: 2, maxLen: 8), CopyCensusConfig(minMatch: 10, startLen: 12, maxLen: 12)]

    /// each client label gets its own totals, and the labels sum to the overall totals
    func testTotalsAreKeptPerClientLabel() {
        let t = CopyCensusTotals(configs: cfgs)
        t.record(prompt: Array(100 ..< 400), output: Array(150 ..< 250), client: "closed-loop")
        t.record(prompt: Array(0 ..< 300), output: Array(1000 ..< 1100), client: "Python-urllib/3.12")
        t.record(prompt: Array(0 ..< 50), output: Array(2000 ..< 2010), client: "Python-urllib/3.12")
        t.record(prompt: Array(0 ..< 5), output: [1, 2], client: nil)
        let s = t.snapshot()
        XCTAssertEqual(s["requests"] as? Int, 4)
        let by = s["by_client"] as? [String: [String: Any]] ?? [:]
        XCTAssertEqual(Set(by.keys), ["closed-loop", "Python-urllib/3.12", "unknown"])
        XCTAssertEqual(by["closed-loop"]?["requests"] as? Int, 1)
        XCTAssertEqual(by["Python-urllib/3.12"]?["requests"] as? Int, 2)
        XCTAssertEqual(by["Python-urllib/3.12"]?["output_tokens"] as? Int, 110)
        XCTAssertEqual(by.values.map { $0["output_tokens"] as? Int ?? 0 }.reduce(0, +), s["output_tokens"] as? Int)
        XCTAssertEqual(by.values.map { $0["prompt_tokens"] as? Int ?? 0 }.reduce(0, +), s["prompt_tokens"] as? Int)
        // the copy-heavy request's lookups are counted under its own label only
        let loopCfg = (by["closed-loop"]?["configs"] as? [[String: Any]])?.first
        let urllibCfg = (by["Python-urllib/3.12"]?["configs"] as? [[String: Any]])?.first
        XCTAssertGreaterThanOrEqual(loopCfg?["lookup_tokens"] as? Int ?? 0, 90)
        XCTAssertEqual(urllibCfg?["lookup_tokens"] as? Int, 0)
        XCTAssertNil(s["file"])                       // no file: nothing about persistence is reported
    }

    /// a new process given the same file starts from the old totals, overall and per label
    func testTotalsSurviveARestartThroughTheFile() throws {
        let f = tempFile()
        let a = CopyCensusTotals(configs: cfgs, file: f, now: { 1000 })
        a.record(prompt: Array(100 ..< 400), output: Array(150 ..< 250), client: "client-x")
        a.record(prompt: Array(0 ..< 300), output: Array(1000 ..< 1100), client: "serve-suite")
        let before = a.snapshot()
        XCTAssertEqual(before["restored_from_file"] as? Bool, false)
        XCTAssertEqual(before["file_errors"] as? Int, 0)

        let b = CopyCensusTotals(configs: cfgs, file: f, now: { 2000 })     // the "restart"
        let restored = b.snapshot()
        XCTAssertEqual(restored["restored_from_file"] as? Bool, true)
        XCTAssertEqual(restored["process_starts"] as? Int, 2)
        XCTAssertEqual(restored["since"] as? Double, 1000)                  // the totals' start survives too
        XCTAssertEqual(restored["requests"] as? Int, 2)
        XCTAssertEqual(restored["output_tokens"] as? Int, before["output_tokens"] as? Int)
        let cb = (before["configs"] as? [[String: Any]])?.map { $0["lookup_tokens"] as? Int }
        let cr = (restored["configs"] as? [[String: Any]])?.map { $0["lookup_tokens"] as? Int }
        XCTAssertEqual(cb, cr)
        let byB = before["by_client"] as? [String: [String: Any]] ?? [:]
        let byR = restored["by_client"] as? [String: [String: Any]] ?? [:]
        XCTAssertEqual(Set(byR.keys), ["client-x", "serve-suite"])
        for k in byB.keys {
            XCTAssertEqual(byR[k]?["requests"] as? Int, byB[k]?["requests"] as? Int)
            XCTAssertEqual(byR[k]?["output_tokens"] as? Int, byB[k]?["output_tokens"] as? Int)
            XCTAssertEqual((byR[k]?["configs"] as? [[String: Any]])?.map { $0["accepted"] as? Int },
                           (byB[k]?["configs"] as? [[String: Any]])?.map { $0["accepted"] as? Int })
        }
        // counting continues on top of the restored totals
        b.record(prompt: Array(0 ..< 10), output: [7, 8, 9], client: "client-x")
        let after = b.snapshot()
        XCTAssertEqual(after["requests"] as? Int, 3)
        XCTAssertEqual((after["by_client"] as? [String: [String: Any]])?["client-x"]?["requests"] as? Int, 2)
        // and a third start sees all three requests
        XCTAssertEqual(CopyCensusTotals(configs: cfgs, file: f).snapshot()["requests"] as? Int, 3)
    }

    /// the file holds counts and labels only: none of the request's token ids appear in it
    func testFileHoldsNoTokenIds() throws {
        let f = tempFile()
        let t = CopyCensusTotals(configs: cfgs, file: f, now: { 1 })
        let prompt = [987_654_301, 987_654_302, 987_654_303, 987_654_304, 987_654_305, 987_654_306]
        let output = [987_654_303, 987_654_304, 987_654_305, 987_654_306, 987_654_399]
        t.record(prompt: prompt, output: output, client: "c")
        let text = try String(contentsOf: f, encoding: .utf8)
        for id in prompt + output { XCTAssertFalse(text.contains(String(id)), "token id \(id) leaked into the census file") }
        let d = try JSONSerialization.jsonObject(with: Data(contentsOf: f)) as? [String: Any]
        XCTAssertEqual(Set(d?.keys.map { $0 } ?? []), ["version", "since", "written", "process_starts", "total", "by_client"])
        // every leaf is a number, except config names
        func leaves(_ v: Any, _ path: String) {
            if let o = v as? [String: Any] { for (k, x) in o { leaves(x, path + "." + k) } }
            else if let a = v as? [Any] { for x in a { leaves(x, path + "[]") } }
            else if path.hasSuffix(".name") { XCTAssertTrue(cfgs.map(\.name).contains(v as? String ?? ""), path) }
            else { XCTAssertTrue(v is NSNumber, "non-count leaf at \(path)") }
        }
        leaves(d as Any, "")
    }

    /// labels are trimmed and capped in length, and past maxLabels distinct labels the rest count as "other"
    func testLabelsAreBoundedInLengthAndNumber() {
        XCTAssertEqual(CopyCensusTotals.label(nil), "unknown")
        XCTAssertEqual(CopyCensusTotals.label("  "), "unknown")
        XCTAssertEqual(CopyCensusTotals.label(" serve-suite "), "serve-suite")
        // per-request lab labels collapse to their run (burst_replay, rung_quality, mixed_smoke)
        XCTAssertEqual(CopyCensusTotals.label("burst-w7-31"), "burst")          // T-0260: per-run lab tags -> one label
        XCTAssertEqual(CopyCensusTotals.label("rq-a-0"), "rq")
        XCTAssertEqual(CopyCensusTotals.label("smoke-big-12"), "smoke-big")
        XCTAssertEqual(CopyCensusTotals.label("Python-urllib/3.12"), "Python-urllib/3.12")   // a version stays
        XCTAssertEqual(CopyCensusTotals.label("deploy-smoke"), "deploy-smoke")
        XCTAssertEqual(CopyCensusTotals.label("-7"), "-7")                                   // nothing before the dash
        XCTAssertEqual(CopyCensusTotals.label("client-"), "client-")
        XCTAssertEqual(CopyCensusTotals.label(String(repeating: "a", count: 500)).count, CopyCensusTotals.maxLabelLength)
        let t = CopyCensusTotals(configs: cfgs)
        for i in 0 ..< (CopyCensusTotals.maxLabels + 10) { t.record(prompt: [1, 2, 3], output: [4], client: "ua\(i)x") }
        t.record(prompt: [1, 2, 3], output: [4], client: "ua0x")          // a known label keeps counting under its name
        let by = t.snapshot()["by_client"] as? [String: [String: Any]] ?? [:]
        XCTAssertEqual(by.count, CopyCensusTotals.maxLabels + 1)
        XCTAssertEqual(by[CopyCensusTotals.otherLabel]?["requests"] as? Int, 10)
        XCTAssertEqual(by["ua0x"]?["requests"] as? Int, 2)
    }

    /// an unreadable file is moved aside (never overwritten or trusted) and the census starts afresh
    func testUnreadableFileIsMovedAsideAndCountingStartsAfresh() throws {
        let f = tempFile()
        try Data("not json".utf8).write(to: f)
        let t = CopyCensusTotals(configs: cfgs, file: f, now: { 42 })
        let s = t.snapshot()
        XCTAssertEqual(s["restored_from_file"] as? Bool, false)
        XCTAssertEqual(s["file_errors"] as? Int, 1)
        XCTAssertEqual(s["requests"] as? Int, 0)
        let aside = f.appendingPathExtension("unreadable-42")
        XCTAssertEqual(try String(contentsOf: aside, encoding: .utf8), "not json")
        t.record(prompt: [1, 2, 3], output: [4, 5], client: "c")
        XCTAssertEqual(CopyCensusTotals(configs: cfgs, file: f).snapshot()["requests"] as? Int, 1)
    }

    /// a file written under other configs keeps its request totals; counts of configs no longer measured are dropped
    func testRestoreMatchesConfigsByName() {
        let f = tempFile()
        let a = CopyCensusTotals(configs: cfgs, file: f)
        a.record(prompt: Array(100 ..< 400), output: Array(150 ..< 250), client: "c")
        let only10 = [CopyCensusConfig(minMatch: 10, startLen: 12, maxLen: 12), CopyCensusConfig(minMatch: 6, startLen: 2, maxLen: 8)]
        let b = CopyCensusTotals(configs: only10, file: f).snapshot()
        XCTAssertEqual(b["requests"] as? Int, 1)
        let c = b["configs"] as? [[String: Any]] ?? []
        let old10 = (a.snapshot()["configs"] as? [[String: Any]])?[1]["lookup_tokens"] as? Int
        XCTAssertEqual(c[0]["lookup_tokens"] as? Int, old10)
        XCTAssertEqual(c[1]["lookup_tokens"] as? Int, 0)                    // min6 was never measured
    }

    /// T-0229: the census's CPU cost per request at client shapes (W126: 1107 prompt / 488 output tokens on average; the
    /// T-0220 extract task 3000 chars / up to 1300 tokens), 4 production configs, with the file and 64 labels in it. Prints
    /// ms per record; the bound is generous (a debug build, ~10x the measured cost) and only guards a regression.
    func testRecordCostAtClientShapes() {
        var rng = SystemRandomNumberGenerator()
        func tokens(_ n: Int) -> [Int] { (0 ..< n).map { _ in Int.random(in: 0 ..< 150_000, using: &rng) } }
        let prod = [CopyCensusConfig(minMatch: 4, startLen: 2, maxLen: 8), CopyCensusConfig(minMatch: 6, startLen: 2, maxLen: 8),
                    CopyCensusConfig(minMatch: 10, startLen: 2, maxLen: 8), CopyCensusConfig(minMatch: 10, startLen: 12, maxLen: 12)]
        func cost(prompt n: Int, output m: Int, copied: Bool, file: URL?, label: (Int) -> String) -> Double {
            let t = CopyCensusTotals(configs: prod, file: file)
            for i in 0 ..< 64 { t.record(prompt: [1, 2, 3], output: [4], client: "warm\(i)x") }   // a full label table
            let prompt = tokens(n)
            let output = copied ? Array(prompt[(n / 4) ..< min(n, n / 4 + m)]) + tokens(max(0, m - (n - n / 4))) : tokens(m)
            let reps = 20
            let t0 = Date()
            for i in 0 ..< reps { t.record(prompt: prompt, output: output, client: label(i)) }
            return Date().timeIntervalSince(t0) * 1000 / Double(reps)
        }
        var lines: [String] = []
        for (n, m) in [(1107, 488), (3000, 1300)] {
            for copied in [false, true] {
                let mem = cost(prompt: n, output: m, copied: copied, file: nil) { _ in "client" }
                let disk = cost(prompt: n, output: m, copied: copied, file: tempFile()) { _ in "client" }
                lines.append(String(format: "census cost prompt %d output %d %@: %.2f ms in memory, %.2f ms with the file",
                                    n, m, copied ? "copied" : "fresh", mem, disk))
                XCTAssertLessThan(disk, 250, "census record() got expensive: \(disk) ms")
            }
        }
        lines.forEach { print($0) }
    }

    /// T-0244: each label sums its requests' MTP rounds / drafted / accepted, the file keeps them, and a file written before
    /// T-0244 (no mtp_* keys) restores them as 0
    func testMTPCountsPerLabelSurviveARestartAndOldFilesReadAsZero() throws {
        let f = tempFile()
        let a = CopyCensusTotals(configs: cfgs, file: f)
        a.record(prompt: [1, 2, 3], output: [4, 5], client: "client", mtp: CopyCensusMTP(rounds: 10, drafted: 30, accepted: 21))
        a.record(prompt: [1, 2, 3], output: [4, 5], client: "client", mtp: CopyCensusMTP(rounds: 5, drafted: 10, accepted: 6))
        a.record(prompt: [1, 2, 3], output: [4, 5], client: "lab:e9/closed_loop", mtp: CopyCensusMTP(rounds: 7, drafted: 14, accepted: 8))
        a.record(prompt: [1, 2, 3], output: [4], client: "curl/8.7.1")                       // no MTP: zeros
        func mtp(_ s: [String: Any], _ k: String?) -> [Int?] {
            let b = k.map { (s["by_client"] as? [String: [String: Any]])?[$0] ?? [:] } ?? s
            return ["mtp_rounds", "mtp_drafted", "mtp_accepted"].map { b[$0] as? Int }
        }
        let s1 = a.snapshot()
        XCTAssertEqual(mtp(s1, "client"), [15, 40, 27])
        XCTAssertEqual(mtp(s1, "lab:e9/closed_loop"), [7, 14, 8])
        XCTAssertEqual(mtp(s1, "curl/8.7.1"), [0, 0, 0])
        XCTAssertEqual(mtp(s1, nil), [22, 54, 35])
        let s2 = CopyCensusTotals(configs: cfgs, file: f).snapshot()             // the restart
        XCTAssertEqual(mtp(s2, "client"), [15, 40, 27])
        XCTAssertEqual(mtp(s2, nil), [22, 54, 35])
        // a pre-T-0244 file: strip the mtp_* keys everywhere, restore, then count on
        var d = try JSONSerialization.jsonObject(with: Data(contentsOf: f)) as! [String: Any]
        func strip(_ b: [String: Any]) -> [String: Any] { b.filter { !$0.key.hasPrefix("mtp_") } }
        d["total"] = strip(d["total"] as! [String: Any])
        d["by_client"] = (d["by_client"] as! [String: [String: Any]]).mapValues(strip)
        try JSONSerialization.data(withJSONObject: d).write(to: f)
        let c = CopyCensusTotals(configs: cfgs, file: f)
        XCTAssertEqual(mtp(c.snapshot(), "client"), [0, 0, 0])
        XCTAssertEqual(c.snapshot()["requests"] as? Int, 4)                       // the other totals survive
        c.record(prompt: [1], output: [2], client: "client", mtp: CopyCensusMTP(rounds: 1, drafted: 2, accepted: 2))
        XCTAssertEqual(mtp(c.snapshot(), "client"), [1, 2, 2])
    }

    /// T-0260 review: a file that cannot be read AND cannot be moved aside is never overwritten (persistence stops)
    func testUnreadableFileThatCannotBeMovedAsideIsNotOverwritten() throws {
        let f = tempFile()
        try Data("not json".utf8).write(to: f)
        let aside = f.appendingPathExtension("unreadable-42")
        try FileManager.default.createDirectory(at: aside, withIntermediateDirectories: true)   // the move target is taken
        try "x".write(to: aside.appendingPathComponent("keep"), atomically: true, encoding: .utf8)
        let t = CopyCensusTotals(configs: cfgs, file: f, now: { 42 })
        t.record(prompt: [1, 2, 3], output: [4, 5], client: "c")
        XCTAssertEqual(try String(contentsOf: f, encoding: .utf8), "not json")
        XCTAssertEqual(t.snapshot()["requests"] as? Int, 1)                 // counting goes on in memory
        XCTAssertNotNil(t.snapshot()["last_file_error"])
    }

    /// T-0271: each request's batched MTP and copy row-rounds are counted per label, persisted, and read as 0 from older files
    func testBatchedCountsPerLabelSurviveARestartAndOldFilesReadAsZero() throws {
        let f = tempFile()
        let a = CopyCensusTotals(configs: cfgs, file: f)
        a.record(prompt: [1, 2, 3], output: [4, 5], client: "client",
                 mtp: CopyCensusMTP(batchRounds: 100, batchDrafted: 200, batchAccepted: 150, copyRows: 20, copyDrafted: 90, copyAccepted: 80))
        a.record(prompt: [1, 2, 3], output: [4, 5], client: "client",
                 mtp: CopyCensusMTP(rounds: 3, drafted: 6, accepted: 4, batchRounds: 10, batchDrafted: 20, batchAccepted: 12))
        a.record(prompt: [1], output: [2], client: "lab:e9/closed_loop", mtp: CopyCensusMTP(copyRows: 1, copyDrafted: 7, copyAccepted: 7))
        let keys = ["batch_rounds", "batch_drafted", "batch_accepted", "copy_rows", "copy_drafted", "copy_accepted"]
        func get(_ s: [String: Any], _ k: String?) -> [Int?] {
            let b = k.map { (s["by_client"] as? [String: [String: Any]])?[$0] ?? [:] } ?? s
            return keys.map { b[$0] as? Int }
        }
        XCTAssertEqual(get(a.snapshot(), "client"), [110, 220, 162, 20, 90, 80])
        XCTAssertEqual(get(a.snapshot(), "lab:e9/closed_loop"), [0, 0, 0, 1, 7, 7])
        XCTAssertEqual(get(a.snapshot(), nil), [110, 220, 162, 21, 97, 87])
        XCTAssertEqual((a.snapshot()["by_client"] as? [String: [String: Any]])?["client"]?["mtp_rounds"] as? Int, 3)   // serial kept apart
        XCTAssertEqual(get(CopyCensusTotals(configs: cfgs, file: f).snapshot(), "client"), [110, 220, 162, 20, 90, 80])   // restart
        var d = try JSONSerialization.jsonObject(with: Data(contentsOf: f)) as! [String: Any]
        func strip(_ b: [String: Any]) -> [String: Any] { b.filter { !$0.key.hasPrefix("batch_") && !$0.key.hasPrefix("copy_") } }
        d["total"] = strip(d["total"] as! [String: Any])
        d["by_client"] = (d["by_client"] as! [String: [String: Any]]).mapValues(strip)
        try JSONSerialization.data(withJSONObject: d).write(to: f)
        let c = CopyCensusTotals(configs: cfgs, file: f)
        XCTAssertEqual(get(c.snapshot(), "client"), [0, 0, 0, 0, 0, 0])
        XCTAssertEqual((c.snapshot()["by_client"] as? [String: [String: Any]])?["client"]?["mtp_rounds"] as? Int, 3)
    }
}
