import AVFoundation
import Foundation

/// Built-in macOS voice via AVSpeechSynthesizer. Placeholder until Qwen3TTSEngine (Task 7).
/// No model download required.
final class MockTTSEngine: NSObject, TTSEngine, @unchecked Sendable {
    private let synth = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Error>?

    override init() {
        super.init()
        synth.delegate = self
    }

    func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            self.continuation = cont
            let utt = AVSpeechUtterance(string: text)
            utt.voice = AVSpeechSynthesisVoice(language: "en-US")
            utt.rate = AVSpeechUtteranceDefaultSpeechRate * Float(speed.value)
            synth.speak(utt)
        }
    }

    func stop() {
        synth.stopSpeaking(at: .immediate)
        continuation?.resume()
        continuation = nil
    }
}

extension MockTTSEngine: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        continuation?.resume()
        continuation = nil
    }
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        continuation?.resume()
        continuation = nil
    }
}
