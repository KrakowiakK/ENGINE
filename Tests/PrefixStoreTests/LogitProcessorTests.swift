import Foundation
import XCTest
import MLX
@testable import EngineServeSupport

/// P124: min_p, presence / frequency / repetition penalties and logit_bias -- parsed, validated, applied.
final class LogitProcessorTests: XCTestCase {
    override func setUp() { super.setUp(); Device.setDefault(device: .cpu) }

    private func json(_ s: String) -> [String: Any] {
        try! JSONSerialization.jsonObject(with: Data(s.utf8)) as! [String: Any]
    }
    private func values(_ a: MLXArray) -> [Float] { a.reshaped(-1).asArray(Float.self) }

    func testDefaultsAndNoOpValuesAreIdentity() {
        XCTAssertTrue(ServeSamplingParams.parse(json("{}")).params.isIdentity)
        let (p, e) = ServeSamplingParams.parse(json(#"{"min_p": 0, "presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1, "logit_bias": {}}"#))
        XCTAssertNil(e)
        XCTAssertTrue(p.isIdentity, "integer 1 / 0 JSON numbers are numbers, not booleans")
        XCTAssertTrue(ServeSamplingParams.parse(json(#"{"logit_bias": {"5": 0}}"#)).params.isIdentity)
    }

    func testParseValuesAliasesAndRangeErrors() {
        let (p, e) = ServeSamplingParams.parse(json(#"{"min_p": 0.05, "presence_penalty": 1.5, "frequency_penalty": -0.5, "repeat_penalty": 1.1, "logit_bias": {"7": -100, "9": 2}}"#), vocabularySize: 10)
        XCTAssertNil(e)
        XCTAssertEqual(p.minP, 0.05, accuracy: 1e-6)
        XCTAssertEqual(p.presencePenalty, 1.5)
        XCTAssertEqual(p.frequencyPenalty, -0.5)
        XCTAssertEqual(p.repetitionPenalty, 1.1, accuracy: 1e-6)
        XCTAssertEqual(p.logitBias, [7: -100, 9: 2])
        XCTAssertFalse(p.isIdentity)
        for bad in [#"{"min_p": 1.5}"#, #"{"presence_penalty": 3}"#, #"{"frequency_penalty": -2.5}"#, #"{"repetition_penalty": 0}"#,
                    #"{"repetition_penalty": true}"#, #"{"min_p": "0.1"}"#, #"{"logit_bias": [1, 2]}"#, #"{"logit_bias": {"x": 1}}"#,
                    #"{"logit_bias": {"3": 101}}"#, #"{"logit_bias": {"3": true}}"#, #"{"logit_bias": {"10": 1}}"#] {
            XCTAssertNotNil(ServeSamplingParams.parse(json(bad), vocabularySize: 10).error, bad)
        }
        // repetition_penalty wins over the alias when both are sent
        XCTAssertEqual(ServeSamplingParams.parse(json(#"{"repetition_penalty": 1.2, "repeat_penalty": 1.5}"#)).params.repetitionPenalty, 1.2, accuracy: 1e-6)
    }

    func testLogitBiasPresenceFrequencyAndRepetitionMath() {
        let params = ServeSamplingParams(presencePenalty: 0.5, frequencyPenalty: 0.25, repetitionPenalty: 2, logitBias: [4: 3])
        var proc = ServeLogitProcessor(params: params, promptIds: [1])
        let logits = MLXArray([1, 2, -2, 4, 0] as [Float]).reshaped(1, 5)
        // before any output: bias on 4; repetition on the prompt token 1 (2 / 2 = 1); nothing else moves
        XCTAssertEqual(values(proc.apply(logits)), [1, 1, -2, 4, 3])
        proc.observe(2); proc.observe(2); proc.observe(3)
        // token 2: seen -> -2 * 2 = -4, then -(0.5 + 0.25 * 2) = -1 -> -5; token 3: 4 / 2 = 2, then -(0.5 + 0.25) -> 1.25
        XCTAssertEqual(values(proc.apply(logits)), [1, 1, -5, 1.25, 3])
        XCTAssertEqual(proc.observed, 3)
    }

    func testIdentityProcessorLeavesLogitsUnchangedAndPenaltyFlipsAGreedyArgmax() {
        var id = ServeLogitProcessor(params: ServeSamplingParams(), promptIds: [0, 1, 2])
        let l = MLXArray([0.5, 3, 1] as [Float]).reshaped(1, 3)
        XCTAssertEqual(values(id.apply(l)), [0.5, 3, 1])
        var proc = ServeLogitProcessor(params: ServeSamplingParams(presencePenalty: 2), promptIds: [])
        XCTAssertEqual(proc.apply(l).argMax(axis: -1).item(Int.self), 1)
        proc.observe(1)                                   // 3 - 2 = 1 -> ties lose to the first index: token 2 (1) vs 1 (1)
        proc.observe(1)
        XCTAssertEqual(values(proc.apply(l)), [0.5, 1, 1])
        var freq = ServeLogitProcessor(params: ServeSamplingParams(frequencyPenalty: 1), promptIds: [])
        freq.observe(1); freq.observe(1); freq.observe(1)
        XCTAssertEqual(freq.apply(l).argMax(axis: -1).item(Int.self), 2, "3 - 3 = 0 < 1: the greedy token moves")
    }

    func testObserveBeforeFirstApplyAndOutOfRangeIdsAreSafe() {
        var proc = ServeLogitProcessor(params: ServeSamplingParams(frequencyPenalty: 1, repetitionPenalty: 1, logitBias: [99: 5]),
                                  promptIds: [-1, 42])
        proc.observe(0); proc.observe(77)                 // before the vocabulary is known; 77 is outside it
        XCTAssertEqual(values(proc.apply(MLXArray([1, 1, 1] as [Float]).reshaped(1, 3))), [0, 1, 1])
    }

    func testMinPKeepsOnlyTokensNearTheTop() {
        // probabilities ~ [0.665, 0.245, 0.090]: min_p 0.3 keeps p >= 0.2 -> tokens 0 and 1
        let l = MLXArray([2, 1, 0] as [Float]).reshaped(1, 3)
        let v = values(ServeLogitProcessor.applyMinP(l, minP: 0.3))
        XCTAssertEqual(Array(v[0 ..< 2]), [2, 1])
        XCTAssertEqual(v[2], -Float.infinity)
        XCTAssertEqual(values(ServeLogitProcessor.applyMinP(l, minP: 0)), [2, 1, 0])
        XCTAssertEqual(values(ServeLogitProcessor.applyMinP(l, minP: 1)).filter { $0.isFinite }.count, 1, "min_p 1 keeps only the top")
    }
}
