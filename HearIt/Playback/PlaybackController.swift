import Foundation

@MainActor
final class PlaybackController {
    private let engine: TTSEngine
    private var currentTask: Task<Void, Never>?

    var isPlaying: Bool { currentTask != nil }

    init(engine: TTSEngine) {
        self.engine = engine
    }

    func play(text: String, voice: VoiceID, speed: Speed) {
        stop()
        currentTask = Task { [engine] in
            do {
                try await engine.synthesize(text: text, voice: voice, speed: speed)
            } catch {
                NSLog("HearIt playback error: \(error)")
            }
            await MainActor.run { [weak self] in
                self?.currentTask = nil
            }
        }
    }

    func stop() {
        engine.stop()
        currentTask?.cancel()
        currentTask = nil
    }
}
