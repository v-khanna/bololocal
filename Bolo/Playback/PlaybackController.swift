import Foundation

@MainActor
final class PlaybackController {
    private let engine: any TTSEngine
    private var currentTask: Task<Void, Never>?

    var isPlaying: Bool { currentTask != nil }

    init(engine: any TTSEngine) {
        self.engine = engine
    }

    func play(text: String, voice: VoiceID, speed: Speed, onComplete: (@Sendable () -> Void)? = nil) {
        stop()
        currentTask = Task { [engine, onComplete] in
            do {
                try await engine.synthesize(text: text, voice: voice, speed: speed)
            } catch {
                NSLog("Bolo playback error: \(error)")
            }
            onComplete?()
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
