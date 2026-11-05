import XCTest
@testable import EngineServeSupport

final class RequestDefaultsTests: XCTestCase {
    func testNoMaxTokensRunsToTheEndOfTheContext() {
        XCTAssertEqual(MaxTokensPolicy.effective(requested: nil, defaultTokens: 0, room: 250_000, clamp: true).tokens, 250_000)
    }

    func testAFixedDefaultIsKeptWhenItFits() {
        let d = MaxTokensPolicy.effective(requested: nil, defaultTokens: 32000, room: 250_000, clamp: true)
        XCTAssertEqual(d.reason, "default_fixed"); XCTAssertEqual(d.tokens, 32000)
    }

    func testAnOversizedRequestIsClampedToTheRoom() {
        let d = MaxTokensPolicy.effective(requested: 262_144, defaultTokens: 0, room: 261_000, clamp: true)
        XCTAssertEqual(d.reason, "clamped"); XCTAssertEqual(d.tokens, 261_000)
    }

    /// Negative control: with the clamp off an oversized request passes through, and admission refuses it (B50).
    func testWithoutTheClampAnOversizedRequestIsUnchanged() {
        let d = MaxTokensPolicy.effective(requested: 262_144, defaultTokens: 0, room: 261_000, clamp: false)
        XCTAssertEqual(d.reason, "requested"); XCTAssertEqual(d.tokens, 262_144)
    }

    func testARequestThatFitsIsNeverChanged() {
        XCTAssertEqual(MaxTokensPolicy.effective(requested: 4096, defaultTokens: 0, room: 261_000, clamp: true).tokens, 4096)
    }

    /// A prompt that fills the context leaves no room: the value is not invented, admission refuses the request.
    func testNoRoomPassesThroughForAdmissionToRefuse() {
        XCTAssertEqual(MaxTokensPolicy.effective(requested: nil, defaultTokens: 0, room: 0, clamp: true).reason, "no_room")
        XCTAssertEqual(MaxTokensPolicy.effective(requested: 100, defaultTokens: 0, room: -5, clamp: true).tokens, 100)
    }

    func testEffortLevelsAndSpellings() {
        XCTAssertEqual(ReasoningEffortPolicy.resolve(nil, serverDefault: "xhigh"), "xhigh")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("low", serverDefault: "xhigh"), "low")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("medium", serverDefault: "xhigh"), "medium")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("High", serverDefault: "medium"), "xhigh")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("max", serverDefault: "medium"), "xhigh")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("minimal", serverDefault: "xhigh"), "low")
        XCTAssertEqual(ReasoningEffortPolicy.resolve("banana", serverDefault: "xhigh"), "xhigh")
    }

    // T-0294 (Q-0001): programming requests are recognised conservatively
    func testProgrammingDetectorCodeToolsAndFences() {
        XCTAssertTrue(ProgrammingRequestDetector.isProgramming(texts: ["Fix this:\n```swift\nlet x = 1\n```"], hasTools: false))
        XCTAssertTrue(ProgrammingRequestDetector.isProgramming(texts: ["any text"], hasTools: true))
        let py = "def load(path):\n    with open(path) as f:\n        return json.load(f)\nclass Store:\n    pass\nimport os\n"
        XCTAssertTrue(ProgrammingRequestDetector.isProgramming(texts: ["Why does this fail?\n" + py], hasTools: false))
        let swift = "func run() {\n    let a = compute()\n    if (a > 0) {\n        print(a);\n    }\n}"
        XCTAssertTrue(ProgrammingRequestDetector.isProgramming(texts: [swift], hasTools: false))
    }
    func testProgrammingDetectorLeavesProseAndOCRTextAlone() {
        let ocr = "Import zboża z Rosji wzrósł w tym roku.\nWokulski wrócił do sklepu (jak zwykle) i zastał tam Rzeckiego.\n" +
                  "Pieniędzmi tymi obracają inni, a zyski ciągną dla siebie.\nKlasa robotnicza i klasa średnia; nikt nie wiedział.\n" +
                  "Jeżeli (co nieprawdopodobne) przyjdzie jutro, to dobrze.\nDla każdego coś miłego."
        XCTAssertFalse(ProgrammingRequestDetector.isProgramming(texts: [ocr], hasTools: false))
        XCTAssertFalse(ProgrammingRequestDetector.isProgramming(texts: ["Jesteś korektorem tekstów po OCR. Przepisz poniższy fragment.", ocr], hasTools: false))
        XCTAssertFalse(ProgrammingRequestDetector.isProgramming(texts: ["What is the capital of France?"], hasTools: false))
        XCTAssertFalse(ProgrammingRequestDetector.isProgramming(texts: [], hasTools: false))
        // two code-like lines are not enough
        XCTAssertFalse(ProgrammingRequestDetector.isProgramming(texts: ["Run import os and then\nfunction foo() {\nok"], hasTools: false))
    }
}
