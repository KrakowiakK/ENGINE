import Foundation
import Darwin

/// P100: a fatal signal prints the signal and a native backtrace to stderr (the server log) before the default
/// action -- the OMP fan-out server died with no crash report and no log line, and this is the cheapest trace.
/// Not async-signal-safe in the strict sense (backtrace_symbols_fd allocates on some paths), acceptable for a last word.
func installCrashTrace() {
    let sigs: [Int32] = [SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGTRAP, SIGFPE]
    for s in sigs {
        signal(s) { sig in
            let msg = "engine: FATAL SIGNAL \(sig) -- backtrace follows\n"
            msg.withCString { _ = write(2, $0, strlen($0)) }
            var frames = [UnsafeMutableRawPointer?](repeating: nil, count: 128)
            let n = backtrace(&frames, 128)
            backtrace_symbols_fd(&frames, n, 2)
            signal(sig, SIG_DFL)
            raise(sig)
        }
    }
}
