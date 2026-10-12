import XCTest
import EngineServeSupport

/// T-0053: the plain-data half of image input on the serve path.
final class ServeVisionContentTests: XCTestCase {
    let ph = ServeVisionContent.imagePlaceholder

    func testTextOnlyPartsAreTheOldJoinedText() throws {
        let parts: [[String: Any]] = [["type": "text", "text": "a"], ["type": "input_audio", "x": 1], ["text": "b"]]
        let r = try ServeVisionContent.flatten(parts, role: "user")
        XCTAssertEqual(r.text, parts.compactMap { $0["text"] as? String }.joined())
        XCTAssertEqual(r.images, [])
    }

    func testImageFormsKeepTheirPlaceAndOrder() throws {
        let parts: [[String: Any]] = [
            ["type": "text", "text": "first "],
            ["type": "image_url", "image_url": ["url": "data:image/png;base64,AAAA"]],
            ["type": "text", "text": " then "],
            ["type": "image_url", "image_url": "https://example.com/b.png"],
            ["type": "input_image", "image_url": "data:image/jpeg;base64,BBBB"],
            ["type": "image", "source": ["type": "base64", "media_type": "image/webp", "data": "CCCC"]],
            ["type": "image", "image": "data:image/png;base64,DDDD"],
            ["type": "text", "text": "?"],
        ]
        let r = try ServeVisionContent.flatten(parts, role: "user")
        XCTAssertEqual(r.text, "first \(ph) then \(ph)\(ph)\(ph)\(ph)?")
        XCTAssertEqual(r.images, ["data:image/png;base64,AAAA", "https://example.com/b.png", "data:image/jpeg;base64,BBBB",
                                  "data:image/webp;base64,CCCC", "data:image/png;base64,DDDD"])
    }

    func testRefusals() {
        XCTAssertThrowsError(try ServeVisionContent.flatten([["type": "video_url", "video_url": ["url": "x"]]], role: "user")) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .video)
        }
        XCTAssertThrowsError(try ServeVisionContent.flatten([["type": "image_url", "image_url": ["url": "data:,"]]], role: "system")) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .imageInSystemMessage)
        }
        XCTAssertThrowsError(try ServeVisionContent.flatten([["type": "text", "text": "x"], ["type": "image_url"]], role: "user")) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .missingImageSource(part: 1))
        }
    }

    func testDataURLs() throws {
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3])
        let url = "data:image/png;base64," + bytes.base64EncodedString()
        XCTAssertEqual(try ServeVisionContent.load(url, limit: 1 << 20, allowFetch: false), bytes)
        XCTAssertThrowsError(try ServeVisionContent.decodeDataURL("data:image/png;base64,%%%%", limit: 1 << 20)) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .badBase64)
        }
        XCTAssertThrowsError(try ServeVisionContent.decodeDataURL("data:image/png,abc", limit: 1 << 20)) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .badBase64)
        }
        XCTAssertThrowsError(try ServeVisionContent.decodeDataURL(url, limit: 4)) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .tooLarge(bytes: bytes.count, limit: 4))
        }
        XCTAssertThrowsError(try ServeVisionContent.load("/etc/passwd", limit: 1 << 20, allowFetch: true)) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .unsupportedImageSource("/etc/passwd"))
        }
        XCTAssertThrowsError(try ServeVisionContent.load("https://example.com/a.png", limit: 1 << 20, allowFetch: false)) {
            XCTAssertEqual($0 as? ServeVisionContent.Failure, .fetchDisabled)
        }
    }

    func testExpandAndRopeDelta() throws {
        // [s, pad, e, x, s, pad, e, y]: two images of 2x3 and 1x2 post-merge patches
        let pad = 9
        let ids = [1, pad, 2, 7, 1, pad, 2, 8]
        let r = try ServeVisionContent.expand(ids: ids, padId: pad, grids: [(1, 2, 3), (1, 1, 2)])
        XCTAssertEqual(r.ids, [1] + Array(repeating: pad, count: 6) + [2, 7, 1] + Array(repeating: pad, count: 2) + [2, 8])
        XCTAssertEqual(r.spans, [.init(start: 1, t: 1, h: 2, w: 3), .init(start: 10, t: 1, h: 1, w: 2)])
        // HF get_rope_index by hand: text 0; span 1..6 at base 1 -> next 1+3 = 4; "2,7,1" -> 4,5,6; span at base 7 -> next 9;
        // "2,8" -> 9,10; so the next token (index 14) takes 11, a delta of 11 - 14 = -3 = (3-6) + (2-2)
        XCTAssertEqual(ServeVisionContent.ropeDelta(spans: r.spans), -3)
        XCTAssertThrowsError(try ServeVisionContent.expand(ids: ids, padId: pad, grids: [(1, 2, 3)])) {
            XCTAssertEqual($0 as? ServeVisionContent.LayoutFailure, .placeholderCount(found: 2, images: 1))
        }
    }

    func testPrefixKeysFollowTheImageContent() throws {
        let pad = 248056
        let ids = [10, pad, 11, 12, pad, 13]
        let r = try ServeVisionContent.expand(ids: ids, padId: pad, grids: [(1, 2, 2), (1, 1, 3)])
        let a = ServeVisionContent.keyed(ids: r.ids, spans: r.spans, hashes: [7, 9])
        let b = ServeVisionContent.keyed(ids: r.ids, spans: r.spans, hashes: [7, 9])
        let c = ServeVisionContent.keyed(ids: r.ids, spans: r.spans, hashes: [7, 8])    // the second image differs
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, r.ids.count)
        // text rows keep their ids; image rows are above every token id and distinct within a span
        for (i, id) in r.ids.enumerated() where id != pad { XCTAssertEqual(a[i], id) }
        for sp in r.spans {
            let rows = Array(a[sp.start ..< sp.start + sp.tokens])
            XCTAssertTrue(rows.allSatisfy { $0 >= 1 << 60 && $0 < 1 << 61 })
            XCTAssertEqual(Set(rows).count, rows.count)
        }
        // the common key prefix of a and c ends exactly where the second image starts
        let common = zip(a, c).prefix { $0 == $1 }.count
        XCTAssertEqual(common, r.spans[1].start)
        XCTAssertEqual(ServeVisionContent.hash64(digest: [0x01, 0x02, 0, 0, 0, 0, 0, 0x03, 0xFF]), 0x0102_0000_0000_0003)
    }
}
