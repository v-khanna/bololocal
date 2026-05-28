// Bolo/BoloDebug.swift
// DEV-ONLY: file-based logger. NSLog gets eaten by the unified log system in
// signed app contexts; this writes line-buffered to /tmp/bolo-test.log so we
// have an unambiguous trace of what fired during user testing.
// Remove (or gate behind #if DEBUG) before ship.
import Foundation

enum BoloDebug {
    private static let path = "/tmp/bolo-test.log"
    private static let lock = NSLock()
    private static let df: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static func log(_ msg: String, file: String = #fileID, line: Int = #line) {
        let stamp = df.string(from: Date())
        let entry = "[\(stamp)] \(file):\(line) — \(msg)\n"
        lock.lock(); defer { lock.unlock() }
        if let data = entry.data(using: .utf8) {
            if let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: path)) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            } else {
                try? data.write(to: URL(fileURLWithPath: path))
            }
        }
        // Also write to stderr in case someone's tailing the launching shell.
        FileHandle.standardError.write(entry.data(using: .utf8) ?? Data())
    }
}
