import Foundation
import Darwin

/// Bounded HTTP/1.1 transport for one request per connection. No MLX state lives here.
public enum HTTPTransport {
    public struct Request {
        public let method: String
        public let path: String
        public let headers: [String: String]
        public let body: Data
    }
    public enum Failure: Error { case malformed, tooLarge, timeout, disconnected }

    public static func readRequest(_ fd: Int32, maxHeaders: Int = 65_536,
                                   maxBody: Int = 32 << 20, timeout: Double = 30) throws -> Request {
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        var buf = Data()
        var tmp = [UInt8](repeating: 0, count: 16_384)
        let terminator = Data([13, 10, 13, 10])
        func receive(_ count: Int) throws -> Int {
            while true {
                let remaining = deadline - ProcessInfo.processInfo.systemUptime
                guard remaining > 0 else { throw Failure.timeout }
                var p = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
                let ready = poll(&p, 1, Int32(min(remaining * 1000 + 1, Double(Int32.max))))
                if ready < 0 && errno == EINTR { continue }
                guard ready > 0 else { throw ready == 0 ? Failure.timeout : Failure.disconnected }
                let n = Darwin.read(fd, &tmp, count)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw Failure.disconnected }
                return n
            }
        }
        while buf.range(of: terminator) == nil {
            guard buf.count < maxHeaders else { throw Failure.tooLarge }
            let n = try receive(min(tmp.count, maxHeaders - buf.count))
            buf.append(contentsOf: tmp[..<n])
        }
        guard let end = buf.range(of: terminator), end.upperBound <= maxHeaders,
              let text = String(data: buf[..<end.lowerBound], encoding: .utf8) else { throw Failure.malformed }
        let lines = text.components(separatedBy: "\r\n")
        let parts = (lines.first ?? "").split(separator: " ")
        guard parts.count == 3, ["HTTP/1.0", "HTTP/1.1"].contains(String(parts[2])) else { throw Failure.malformed }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":"), colon != line.startIndex,
                  line.first != " ", line.first != "\t" else { throw Failure.malformed }
            let name = line[..<colon].lowercased()
            guard !name.contains(where: { $0.isWhitespace }), headers[name] == nil else { throw Failure.malformed }
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        // No ambiguous lengths, chunked upload, or unsupported Expect handshake.
        guard headers["transfer-encoding"] == nil, headers["expect"] == nil else { throw Failure.malformed }
        let lengthText = headers["content-length"] ?? "0"
        guard !lengthText.isEmpty, lengthText.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
              let length = Int(lengthText) else { throw Failure.malformed }
        guard length <= maxBody else { throw Failure.tooLarge }
        var body = Data(buf[end.upperBound...].prefix(length))
        while body.count < length {
            let n = try receive(min(tmp.count, length - body.count))
            body.append(contentsOf: tmp[..<n])
        }
        return Request(method: String(parts[0]), path: String(parts[1]), headers: headers, body: body)
    }

    @discardableResult
    public static func writeAll(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw in
            guard var p = raw.baseAddress else { return true }
            var remaining = raw.count
            while remaining > 0 {
                let n = Darwin.write(fd, p, remaining)
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { return false }
                p += n; remaining -= n
            }
            return true
        }
    }

    /// A read EOF is a valid HTTP half-close, not proof that the client stopped reading.
    /// Confirm loss through an error or a failed write instead.
    public static func peerClosed(_ fd: Int32) -> Bool {
        var byte: UInt8 = 0
        let n = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT)
        return n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR
    }

    /// One legal whitespace byte for JSON or SSE, without blocking the model owner.
    /// This elicits a transport error from a fully closed TCP peer during long prefills.
    public static func heartbeat(_ fd: Int32) -> Bool {
        var byte: UInt8 = 10
        while true {
            let n = send(fd, &byte, 1, MSG_DONTWAIT)
            if n < 0 && errno == EINTR { continue }
            return n == 1 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        }
    }

    public static func configure(_ fd: Int32, writeTimeout: Int = 30) {
        var value: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &value, socklen_t(MemoryLayout<Int32>.size))
        var timeout = timeval(tv_sec: writeTimeout, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    }
}
