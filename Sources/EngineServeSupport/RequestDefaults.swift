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
