import XCTest
@testable import EngineServeSupport

/// P106 B41 (D49): a replayed tool call must reach the chat template with its arguments in the order the model
/// generated them. H50d measured the cost of losing it: a `write` call generated path-then-content came back
/// content-first (Jinja sorts dictionary keys) and the next turn re-prefilled ~19k tokens.
final class OrderedJSONTests: XCTestCase {
    func testMembersKeepSourceOrderAndRawValues() throws {
        let s = #"{"path":"harness\/st.js","content":"line 1\nline 2\n","n":1,"flag":false,"z":null,"o":{"b":1,"a":[1,{"y":2,"x":"}"}]}}"#
        let m = try XCTUnwrap(jsonObjectMembersInOrder(s))
        XCTAssertEqual(m.map(\.key), ["path", "content", "n", "flag", "z", "o"])
        XCTAssertEqual(m[0].raw, #""harness\/st.js""#)
        XCTAssertEqual(m[1].raw, #""line 1\nline 2\n""#)
        XCTAssertEqual(m[2].raw, "1")
        XCTAssertEqual(m[3].raw, "false")
        XCTAssertEqual(m[4].raw, "null")
        XCTAssertEqual(m[5].raw, #"{"b":1,"a":[1,{"y":2,"x":"}"}]}"#)
        XCTAssertEqual(try XCTUnwrap(jsonObjectMembersInOrder(m[5].raw)).map(\.key), ["b", "a"])
    }

    func testWhitespaceEscapesAndUnicodeKeys() throws {
        let s = "{ \n \"b\\\"q\" : \"x\\\\\" ,\t\"zażółć\":\"🖥️\" }\n"
        let m = try XCTUnwrap(jsonObjectMembersInOrder(s))
        XCTAssertEqual(m.map(\.key), ["b\"q", "zażółć"])
        XCTAssertEqual(m[0].raw, "\"x\\\\\"")
        XCTAssertEqual(try XCTUnwrap(jsonObjectMembersInOrder("{}")).count, 0)
    }

    func testMalformedIsRefusedNotGuessed() {
        for bad in ["", "[1,2]", "{\"a\":1,}", "{\"a\" 1}", "{\"a\":1} x", "{\"a\":\"unterminated}", "{1:2}"] {
            XCTAssertNil(jsonObjectMembersInOrder(bad), bad)
        }
    }
}
