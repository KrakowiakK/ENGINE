import XCTest
@testable import EngineServeSupport

final class KeyLRUTests: XCTestCase {
    func testTouchMovesAKeyToTheNewestEndAndPopTakesTheOldest() {
        var l = KeyLRU()
        for k in ["a", "b", "c", "d"] { l.touch(k) }
        XCTAssertEqual(l.ordered, ["a", "b", "c", "d"])
        l.touch("b")                                   // a hit
        XCTAssertEqual(l.ordered, ["a", "c", "d", "b"])
        XCTAssertEqual(l.popOldest(), "a")
        l.remove("d")
        XCTAssertEqual(l.ordered, ["c", "b"])
        l.touch("b"); l.touch("c")
        XCTAssertEqual(l.ordered, ["b", "c"])
        XCTAssertEqual(l.popOldest(), "b"); XCTAssertEqual(l.popOldest(), "c"); XCTAssertNil(l.popOldest())
        XCTAssertEqual(l.count, 0); XCTAssertNil(l.newest)
    }
    /// A hit costs the same at 100k keys as at 1k (the [String] version cost ~930 us per hit at 100k).
    func testTouchCostDoesNotGrowWithTheKeyCount() {
        func perTouch(_ n: Int) -> Double {
            var l = KeyLRU(); for i in 0 ..< n { l.touch(String(repeating: String(format: "%016x", i), count: 4)) }
            let keys = (0 ..< 2000).map { String(repeating: String(format: "%016x", ($0 * 7919) % n), count: 4) }
            let t0 = Date(); for k in keys { l.touch(k) }
            return Date().timeIntervalSince(t0) * 1e6 / 2000
        }
        let small = perTouch(1_000), large = perTouch(100_000)
        print(String(format: "KeyLRU touch: %.2f us at 1k keys, %.2f us at 100k keys", small, large))
        XCTAssertLessThan(large, max(20, small * 5), "touch must stay O(1)")
    }
}
