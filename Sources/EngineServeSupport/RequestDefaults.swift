import Foundation

/// P114: the output length a request is admitted with. The context the model can hold (262144) is the only real
/// limit; `max_tokens` is how much of it the reply may use. `--tokens 0` lets a request that names no max_tokens run
/// to the end of the context. With `clamp`, a max_tokens larger than the room left after the prompt is cut to that
/// room instead of refused with HTTP 400, so a client configured for "the whole context" works. With no room at all
/// the value passes through unchanged and admission refuses it, as before.
public enum MaxTokensPolicy {
    public static func effective(requested: Int?, defaultTokens: Int, room: Int,
                                 clamp: Bool) -> (reason: String, tokens: Int) {
        guard room > 0 else { return ("no_room", requested ?? max(defaultTokens, 1)) }
        guard let r = requested else {
            if defaultTokens == 0 { return ("default_rest_of_context", room) }
            if clamp && defaultTokens > room { return ("default_clamped", room) }
            return ("default_fixed", defaultTokens)
        }
        if clamp && r > room { return ("clamped", room) }
        return ("requested", r)
    }
}

/// P114: the checkpoint knows three reasoning levels (chat_template.jinja: xhigh, medium, low). A client's other
/// spellings map to the nearest one instead of silently falling to medium: "high"/"max" ask for the most thinking,
/// "minimal"/"none"/"off" for the least; anything else gets the server's default.
public enum ReasoningEffortPolicy {
    public static let levels = ["xhigh", "medium", "low"]
    public static func resolve(_ requested: String?, serverDefault: String) -> String {
        guard let raw = requested else { return serverDefault }
        let r = raw.lowercased()
        if levels.contains(r) { return r }
        switch r {
        case "high", "max", "maximum": return "xhigh"
        case "minimal", "none", "off": return "low"
        default: return serverDefault
        }
    }
}

/// Counts of every admission decision, published in /v1/engine/sessions as the witness that the policy ran.
public final class DecisionWitness: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    public init() {}
    public func record(_ key: String) { lock.lock(); counts[key, default: 0] += 1; lock.unlock() }
    public func snapshot() -> [String: Int] { lock.lock(); defer { lock.unlock() }; return counts }
}

/// T-0294 (Q-0001, operator 2026-11-05 C-0794): programming requests get reasoning effort `low` by default. A request is
/// "programming" when it carries tool definitions (agentic coding sessions), contains a fenced code block, or has at least
/// three lines that read as source code. Deliberately conservative: prose -- including OCR-corrected Polish text, where
/// "import" is an ordinary word -- must not trip it; a request's own reasoning_effort always wins over any default.
public enum ProgrammingRequestDetector {
    /// each an anchored pattern for one kind of source line (a list, not one alternation: ICU rejected the combined form)
    static let keywordLines: [NSRegularExpression] = [
        #"^\s*def\s+\w+\s*\("#, #"^\s*class\s+\w+[\s:\(\{<]"#, #"^\s*import\s+[\w.]+\s*$"#, #"^\s*from\s+[\w.]+\s+import\s"#,
        #"^\s*func\s+\w+\s*[\(<]"#, #"^\s*fn\s+\w+\s*[\(<]"#, #"^\s*(let|var|const)\s+\w+\s*[:=]"#, #"^\s*function\s+\w*\s*\("#,
        #"^\s*return\s+\S.*;\s*$"#, #"^\s*#include\s*[<\"]"#, #"^\s*package\s+[\w.]+\s*;?\s*$"#,
        #"^\s*(public|private|protected|static)\s+[\w<>\[\]]+\s+\w+\s*[\(=;]"#, #"^\s*(struct|enum|interface)\s+\w+\s*[\{<:]"#,
        #"^\s*(if|for|while|switch)\s*\(.*\)\s*\{\s*$"#,
    ].map { try! NSRegularExpression(pattern: $0) }
    static let endsLikeCode = try! NSRegularExpression(pattern: #"(\{|\};?|\);|=>\s*\{?)\s*$"#)
    public static func isProgramming(texts: [String], hasTools: Bool) -> Bool {
        if hasTools { return true }
        var codeLines = 0
        for t in texts {
            if t.contains("```") { return true }
            for line in t.split(separator: "\n", omittingEmptySubsequences: true) {
                let s = String(line); let r = NSRange(s.startIndex..., in: s)
                if keywordLines.contains(where: { $0.firstMatch(in: s, range: r) != nil }) || endsLikeCode.firstMatch(in: s, range: r) != nil {
                    codeLines += 1
                    if codeLines >= 3 { return true }
                }
            }
        }
        return false
    }
}
