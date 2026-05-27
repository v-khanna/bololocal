import Foundation

/// Opaque identifier for a voice in any engine.
struct VoiceID: Hashable, Codable, RawRepresentable {
    let rawValue: String
    init(rawValue: String) { self.rawValue = rawValue }

    static let systemDefault = VoiceID(rawValue: "system-default")
}

/// Clamped speech speed multiplier, 0.5x..2.0x.
struct Speed: Equatable, Codable {
    let value: Double
    init(_ raw: Double) {
        self.value = max(0.5, min(2.0, raw))
    }
}

/// Engine-agnostic TTS contract. Implementations: MockTTSEngine, Qwen3TTSEngine (Task 7).
protocol TTSEngine: Sendable {
    /// Synthesize the text and play it through the system audio output.
    /// Returns when playback finishes (or `stop()` is called).
    func synthesize(text: String, voice: VoiceID, speed: Speed) async throws

    /// Halt any in-progress playback immediately.
    func stop()
}

enum TTSError: Error, Equatable {
    case modelNotLoaded
    case synthesisFailed(String)
    case playbackFailed(String)
    case emptyText
}
