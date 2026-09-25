import Foundation

/// Holds only a possible delimiter suffix. Completed stop strings never escape to SSE.
public struct DelimiterStream {
    private let markers: [[UInt8]]
    private var pending: [UInt8] = []
    public private(set) var matched: Int? = nil
    public init(_ markers: [String]) { self.markers = markers.filter { !$0.isEmpty }.map { Array($0.utf8) } }
    public mutating func append(_ text: String) -> (text: String, remainder: String) {
        guard matched == nil else { return ("", text) }
        pending.append(contentsOf: text.utf8)
        for i in pending.indices {
            for (j, marker) in markers.enumerated() where i + marker.count <= pending.count {
                if pending[i..<(i + marker.count)].elementsEqual(marker) {
                    matched = j
                    let result = (String(decoding: pending[..<i], as: UTF8.self),
                                  String(decoding: pending[(i + marker.count)...], as: UTF8.self))
                    pending.removeAll(keepingCapacity: true)
                    return result
                }
            }
        }
        var held = 0
        for marker in markers {
            for n in stride(from: min(pending.count, marker.count - 1), through: 1, by: -1) {
                if pending.suffix(n).elementsEqual(marker.prefix(n)) { held = max(held, n); break }
            }
        }
        let count = pending.count - held
        let out = String(decoding: pending.prefix(count), as: UTF8.self)
        pending.removeFirst(count)
        return (out, "")
    }
    public mutating func finish() -> String {
        defer { pending.removeAll() }
        return String(decoding: pending, as: UTF8.self)
    }
}

/// Linear incremental parser for the checkpoint's XML tool stream. Each parameter is
/// retained until complete, but only newly appended bytes plus delimiter overlap are scanned.
public struct XMLToolStream {
    public enum Event: Equatable { case open(String), parameter(String, String), close }
    private enum State { case call, name, body, parameterName, value(String) }
    private var state = State.call
    private var buffer = Data()
    private var searched = 0
    public init() {}
    private mutating func take(_ marker: String) -> String? {
        let bytes = Data(marker.utf8)
        let start = buffer.index(buffer.startIndex, offsetBy: min(searched, buffer.count))
        guard let range = buffer.range(of: bytes, in: start..<buffer.endIndex) else {
            searched = max(0, buffer.count - bytes.count + 1); return nil
        }
        let text = String(decoding: buffer[..<range.lowerBound], as: UTF8.self)
        buffer = Data(buffer[range.upperBound...]); searched = 0
        return text
    }
    public mutating func append(_ text: String) -> [Event] {
        buffer.append(contentsOf: text.utf8)
        var events: [Event] = []
        while true {
            switch state {
            case .call:
                guard take("<function=") != nil else { return events }
                state = .name
            case .name:
                guard let name = take(">") else { return events }
                events.append(.open(name.trimmingCharacters(in: .whitespaces))); state = .body
            case .body:
                let start = buffer.index(buffer.startIndex, offsetBy: min(searched, buffer.count))
                let range = start..<buffer.endIndex
                let param = buffer.range(of: Data("<parameter=".utf8), in: range)
                let close = buffer.range(of: Data("</function>".utf8), in: range)
                if let p = param, close == nil || p.lowerBound < close!.lowerBound {
                    buffer = Data(buffer[p.upperBound...]); searched = 0; state = .parameterName
                } else if let c = close {
                    buffer = Data(buffer[c.upperBound...]); searched = 0
                    events.append(.close); state = .call
                } else { searched = max(0, buffer.count - 11); return events }
            case .parameterName:
                guard let name = take(">") else { return events }
                state = .value(name.trimmingCharacters(in: .whitespaces))
            case .value(let name):
                guard var value = take("</parameter>") else { return events }
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }
                events.append(.parameter(name, value)); state = .body
            }
        }
    }
}
