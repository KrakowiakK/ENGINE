// P106 B41 (D49): JSON member order for replayed tool calls.
import Foundation

/// P106 B41 (D49): the members of a JSON object in SOURCE order with each value's raw text; nil unless `s` is one
/// well-formed object. JSONSerialization and `Jinja.Value(any:)` both drop member order (the latter SORTS keys),
/// so a replayed tool call rendered its parameters alphabetically -- not in the order the model wrote them.
public func jsonObjectMembersInOrder(_ s: String) -> [(key: String, raw: String)]? {
    let b = Array(s.utf8)
    var i = 0
    func ws() { while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 } }
    func stringEnd() -> Int? {                       // b[i] == '"'; index just past the closing quote
        var j = i + 1
        while j < b.count { if b[j] == 0x5C { j += 2; continue }; if b[j] == 0x22 { return j + 1 }; j += 1 }
        return nil
    }
    func valueEnd() -> Int? {
        guard i < b.count else { return nil }
        if b[i] == 0x22 { return stringEnd() }
        if b[i] == 0x7B || b[i] == 0x5B {
            var depth = 0, j = i
            while j < b.count {
                switch b[j] {
                case 0x22:
                    let save = i; i = j; guard let e = stringEnd() else { i = save; return nil }; i = save; j = e; continue
                case 0x7B, 0x5B: depth += 1
                case 0x7D, 0x5D: depth -= 1; if depth == 0 { return j + 1 }
                default: break
                }
                j += 1
            }
            return nil
        }
        var j = i
        while j < b.count, b[j] != 0x2C, b[j] != 0x7D, b[j] != 0x5D, b[j] != 0x20, b[j] != 0x0A, b[j] != 0x0D, b[j] != 0x09 { j += 1 }
        return j > i ? j : nil
    }
    func text(_ from: Int, _ to: Int) -> String { String(decoding: b[from ..< to], as: UTF8.self) }
    ws(); guard i < b.count, b[i] == 0x7B else { return nil }
    i += 1; ws()
    var out: [(String, String)] = []
    if i < b.count, b[i] == 0x7D { i += 1; ws(); return i == b.count ? [] : nil }
    while true {
        guard i < b.count, b[i] == 0x22, let ke = stringEnd(),
              let key = (try? JSONSerialization.jsonObject(with: Data(b[i ..< ke]), options: .fragmentsAllowed)) as? String else { return nil }
        i = ke; ws()
        guard i < b.count, b[i] == 0x3A else { return nil }
        i += 1; ws()
        guard let ve = valueEnd() else { return nil }
        out.append((key, text(i, ve))); i = ve; ws()
        guard i < b.count else { return nil }
        if b[i] == 0x2C { i += 1; ws(); continue }
        guard b[i] == 0x7D else { return nil }
        i += 1; ws()
        return i == b.count ? out.map { (key: $0.0, raw: $0.1) } : nil
    }
}
