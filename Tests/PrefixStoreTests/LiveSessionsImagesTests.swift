// T-0141: the sessions view says how many images a request carried, so a burst's prompts can be told apart as page
// images or text without reading them.
import XCTest

@testable import EngineServeSupport

final class LiveSessionsImagesTests: XCTestCase {
    func testImagesAreReportedForActiveAndRecentRowsAndDefaultToZero() {
        let s = LiveSessions()
        let text = s.begin(client: "t", promptTokens: 10, maxTokens: 5, stream: false, reasoningEffort: "off")
        let image = s.begin(client: "t", promptTokens: 900, maxTokens: 5, stream: false, reasoningEffort: "off")
        s.update(image) { $0.images = 2 }
        let active = s.snapshot()["active"] as! [[String: Any]]
        XCTAssertEqual(active.map { $0["images"] as? Int }, [0, 2])
        s.end(image)
        let recent = s.snapshot()["recent"] as! [[String: Any]]
        XCTAssertEqual(recent.first?["images"] as? Int, 2)
        XCTAssertEqual(recent.first?["prompt_tokens"] as? Int, 900)
        _ = text
    }
}
