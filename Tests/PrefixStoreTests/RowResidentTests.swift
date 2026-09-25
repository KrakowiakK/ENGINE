import XCTest
import Foundation
import MLX
import MLXLMCommon
@testable import Qwen4Exp

/// P106 H48 -- CPU-only tests of the row-resident pool primitives; no model or checkpoint needed.
/// The GPU value gates (H20 fixed schedule, H48 churn arms) are the exactness evidence; these pin
/// the host-side contracts those gates rely on.
final class RowResidentTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }
    private func block(_ start: Int, _ count: Int, batch: Int, heads: Int = 2, dim: Int = 3) -> MLXArray {
        MLXArray((0..<(batch * heads * count * dim)).map { Float(start * 1000 + $0) }).reshaped([batch, heads, count, dim])
    }

    /// Per-row `updateRow` over the rows' own buffers holds, in every row's valid prefix, exactly what the
    /// stacked `updateRagged` holds in that row's slice -- including verify-width blocks, rejected rows
    /// overwritten on the next write, and growth past capacity (bounded and unbounded).
    func testUpdateRowMatchesUpdateRaggedPerRow() {
        for limit in [nil, 29] as [Int?] {
            let stacked = KVCacheSimple(); stacked.step = 8; stacked.capacityGrowthLimit = limit
            let own = (0..<3).map { _ -> KVCacheSimple in let c = KVCacheSimple(); c.step = 8; c.capacityGrowthLimit = limit; return c }
            var lengths = [5, 11, 17]
            // seed each row's history
            for (b, n) in lengths.enumerated() {
                let k = block(b, 17, batch: 1)[0..., 0..., 0 ..< n, 0...], v = k + 0.5
                own[b].updateRow(keys: k, values: v, at: 0)
            }
            stacked.updateRagged(keys: concatenated((0..<3).map { block($0, 17, batch: 1)[0..., 0..., 0 ..< 17, 0...] }, axis: 0),
                                 values: concatenated((0..<3).map { block($0, 17, batch: 1)[0..., 0..., 0 ..< 17, 0...] + 0.5 }, axis: 0),
                                 at: [0, 0, 0])
            var step = 0
            for (S, keep) in [(4, [1, 4, 2]), (1, [1, 1, 1]), (4, [4, 0, 3]), (4, [2, 2, 2]), (1, [1, 1, 1]), (4, [4, 4, 4])] {
                step += 1
                let k = block(100 + step, S, batch: 3), v = k * 2
                stacked.updateRagged(keys: k, values: v, at: lengths)
                for b in 0..<3 { own[b].updateRow(keys: k[b ..< (b + 1)], values: v[b ..< (b + 1)], at: lengths[b]) }
                lengths = (0..<3).map { lengths[$0] + keep[$0] }      // rejected verify rows stay behind as stale
                eval([stacked.rawKeys!, stacked.rawValues!] + own.flatMap { [$0.rawKeys!, $0.rawValues!] })
                for b in 0..<3 {
                    let n = lengths[b]
                    XCTAssertGreaterThanOrEqual(own[b].rawKeys!.dim(2), n)
                    XCTAssertEqual(stacked.rawKeys![b ..< (b + 1), 0..., 0 ..< n, 0...].asArray(Float.self),
                                   own[b].rawKeys![0..., 0..., 0 ..< n, 0...].asArray(Float.self), "row \(b) step \(step)")
                    XCTAssertEqual(stacked.rawValues![b ..< (b + 1), 0..., 0 ..< n, 0...].asArray(Float.self),
                                   own[b].rawValues![0..., 0..., 0 ..< n, 0...].asArray(Float.self))
                }
            }
        }
    }

    private func rowCaches(length: Int, fixedSeed: Float) -> [KVCache] {
        let kv = KVCacheSimple(); kv.step = 8
        kv.updateRow(keys: block(Int(fixedSeed), length, batch: 1), values: block(Int(fixedSeed), length, batch: 1), at: 0)
        let idx = ArraysCache(size: 3)
        idx[0] = MLXArray((0..<(length * 2)).map { Float($0) + fixedSeed }).reshaped([1, length, 2])
        idx[1] = MLXArray((0..<(length / 4 * 2)).map { Float($0) - fixedSeed }).reshaped([1, length / 4, 2])
        let fixed = ArraysCache(size: 6)
        fixed[0] = MLXArray([fixedSeed, fixedSeed + 1]).reshaped([1, 2])
        fixed[1] = MLXArray([fixedSeed * 2]).reshaped([1, 1])
        fixed[4] = MLXArray([Float(9)]).reshaped([1, 1])           // verify-block transient: dropped by the stack
        return [fixed, CacheList(kv, idx)]
    }

    /// The pool registers the members' own attention CacheLists behind a marker, stacks only fixed slots 0-3,
    /// and the unstack hands back the SAME objects at the committed length with fresh fixed-state slices.
    func testStackUnstackRoundTripKeepsObjectsAndReleasesMarkers() {
        let rows = [rowCaches(length: 12, fixedSeed: 1), rowCaches(length: 20, fixedSeed: 5)]
        let pool = Qwen4ExpModel.stackRowResident(rows)
        XCTAssertEqual(Qwen4ExpBatch.rowResidentRegistered, 1)
        let marker = (pool[1] as! CacheList)[0] as! KVCacheSimple
        XCTAssertNil(marker.rawKeys)
        XCTAssertEqual(marker.offset, 20)
        let lists = Qwen4ExpBatch.rowResident(marker)!
        XCTAssertTrue(lists[0] === (rows[0][1] as AnyObject) && lists[1] === (rows[1][1] as AnyObject))
        let fixed = pool[0] as! ArraysCache
        XCTAssertEqual(fixed[0]!.shape, [2, 2]); XCTAssertNil(fixed[4])
        // the ragged step would advance offsets past the committed length (a verify block's rejected rows)
        ((rows[1][1] as! CacheList)[0] as! KVCacheSimple).offset = 23
        let (back, aliased) = Qwen4ExpModel.unstackRowResidentCore(pool, slot: 1, length: 21, own: rows[1])
        XCTAssertTrue(aliased)
        XCTAssertTrue(back[1] as AnyObject === rows[1][1] as AnyObject)
        XCTAssertEqual(((back[1] as! CacheList)[0] as! KVCacheSimple).offset, 21)
        XCTAssertEqual((back[0] as! ArraysCache)[0]!.asArray(Float.self), [5, 6])
        XCTAssertEqual((back[0] as! ArraysCache)[1]!.asArray(Float.self), [10])
        XCTAssertNil((back[0] as! ArraysCache)[4])
        let (_, aliased0) = Qwen4ExpModel.unstackRowResidentCore(pool, slot: 0, length: 12, own: rows[1])
        XCTAssertFalse(aliased0)                                     // wrong owner list is reported, registry wins
        Qwen4ExpModel.releaseRowResident(pool)
        XCTAssertEqual(Qwen4ExpBatch.rowResidentRegistered, 0)
        XCTAssertNil(Qwen4ExpBatch.rowResident(marker))
    }

    /// B41: the in-place rung view of a pool member is the member's own attention objects with their offsets
    /// UNTOUCHED (a live pool must not be rewound by a capture) and slot `slot` of the stacked fixed state --
    /// the same fixed values a dissolve would hand back -- and it leaves the pool registered.
    func testRowResidentViewReadsTheMemberWithoutMutatingThePool() {
        let rows = [rowCaches(length: 12, fixedSeed: 1), rowCaches(length: 20, fixedSeed: 5)]
        let pool = Qwen4ExpModel.stackRowResident(rows)
        ((rows[1][1] as! CacheList)[0] as! KVCacheSimple).offset = 23      // a verify block past the committed length
        let view = Qwen4ExpModel.rowResidentView(pool, slot: 1)
        XCTAssertTrue(view[1] as AnyObject === rows[1][1] as AnyObject)
        XCTAssertEqual(((view[1] as! CacheList)[0] as! KVCacheSimple).offset, 23)
        XCTAssertEqual((view[0] as! ArraysCache)[0]!.asArray(Float.self), [5, 6])
        XCTAssertEqual((view[0] as! ArraysCache)[1]!.asArray(Float.self), [10])
        XCTAssertNil((view[0] as! ArraysCache)[4])
        XCTAssertEqual((Qwen4ExpModel.rowResidentView(pool, slot: 0)[0] as! ArraysCache)[0]!.asArray(Float.self), [1, 2])
        XCTAssertEqual(Qwen4ExpBatch.rowResidentRegistered, 1)
        let (back, _) = Qwen4ExpModel.unstackRowResidentCore(pool, slot: 1, length: 21, own: rows[1])
        XCTAssertEqual((back[0] as! ArraysCache)[0]!.asArray(Float.self), (view[0] as! ArraysCache)[0]!.asArray(Float.self))
        Qwen4ExpModel.releaseRowResident(pool)
        XCTAssertEqual(Qwen4ExpBatch.rowResidentRegistered, 0)
    }

    /// The retired-head copy is bit-exact, independent of its source (writes to one never reach the other),
    /// and keeps offsets.
    func testOwnedAttentionCopyIsIndependentAndExact() {
        let src = rowCaches(length: 16, fixedSeed: 3)[1] as! CacheList
        let copy = Qwen4ExpModel.ownedAttentionCopy(src)
        let skv = src[0] as! KVCacheSimple, ckv = copy[0] as! KVCacheSimple
        XCTAssertEqual(ckv.offset, skv.offset)
        XCTAssertEqual(ckv.state[0].asArray(Float.self), skv.state[0].asArray(Float.self))
        XCTAssertEqual((copy[1] as! ArraysCache)[1]!.asArray(Float.self)[0 ..< 8],
                       (src[1] as! ArraysCache)[1]!.asArray(Float.self)[0 ..< 8])
        let before = skv.state[0].asArray(Float.self)
        ckv.updateRow(keys: block(77, 2, batch: 1), values: block(77, 2, batch: 1), at: 3)
        eval([ckv.rawKeys!])
        XCTAssertEqual(skv.state[0].asArray(Float.self), before)
        XCTAssertNotEqual(ckv.state[0].asArray(Float.self), before)
    }
}
