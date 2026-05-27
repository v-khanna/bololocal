import AVFoundation
import Foundation
import Qwen3TTS

/// Real TTS engine backed by Qwen3-TTS via speech-swift (https://github.com/soniqo/speech-swift).
///
/// Discovered public API (from SPM checkout, package `Qwen3Speech` 0.0.18, module `Qwen3TTS`):
///
///   public class Qwen3TTSModel {
///       public static func fromPretrained(
///           modelId: String = "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit",
///           tokenizerModelId: String = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
///           cacheDir: URL? = nil,
///           offlineMode: Bool = false,
///           progressHandler: ((Double, String) -> Void)? = nil
///       ) async throws -> Qwen3TTSModel
///
///       public func synthesize(
///           text: String,
///           language: String = "english",
///           speaker: String? = nil,
///           instruct: String? = nil,
///           sampling: SamplingConfig = .default,
///           languageExplicit: Bool = false
///       ) -> [Float]                                  // 24kHz mono float samples, synchronous
///
///       public var sampleRate: Int { 24000 }
///   }
///
/// Notes:
/// - `synthesize` returns plain `[Float]` samples (NOT async, NOT throws). The model can
///   `fatalError` if the tokenizer isn't loaded — `fromPretrained` loads it for us.
/// - No `speed` parameter in the underlying API. We apply speed post-hoc via
///   `AVAudioUnitVarispeed` in the playback graph.
/// - First call to `fromPretrained()` downloads ~1.83GB of weights from Hugging Face into
///   `~/Documents/huggingface/...`. Subsequent calls reuse the cache.
///
/// Why an actor:
/// - `Qwen3TTSModel` isn't `Sendable`. Wrapping it in an actor serializes access (and the
///   model is single-threaded anyway — concurrent synthesize calls would corrupt MLX state).
/// - `ModelManager` (Task 9) owns the model lifecycle externally. The engine receives a
///   `modelProvider` closure and calls it on each synthesize — ModelManager handles caching,
///   lazy-loading, and idle-unload.
actor Qwen3TTSEngine: TTSEngine {
    private let modelProvider: @Sendable () async throws -> Qwen3TTSModel

    // Audio playback graph — created on first synthesize, reused across calls.
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var varispeedNode: AVAudioUnitVarispeed?

    init(modelProvider: @escaping @Sendable () async throws -> Qwen3TTSModel) {
        self.modelProvider = modelProvider
    }

    nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await _synthesize(text: text, voice: voice, speed: speed)
    }

    private func _synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        let model = try await modelProvider()
        let language = Self.languageString(for: voice)
        let sampleRate = Double(model.sampleRate)

        // Synchronous CPU-heavy work. We're already on the actor's executor — this
        // serializes calls naturally. Could move to a detached task if we ever want
        // to keep the actor responsive during long generations, but that requires
        // boxing the non-Sendable model.
        let samples: [Float] = model.synthesize(text: text, language: language)

        guard !samples.isEmpty else {
            throw TTSError.synthesisFailed("Qwen3 returned 0 samples")
        }

        try await play(samples: samples, sampleRate: sampleRate, speed: speed)
    }

    nonisolated func stop() {
        Task { await self._stop() }
    }

    private func _stop() {
        playerNode?.stop()
        engine?.stop()
    }

    // MARK: - Playback

    private func play(samples: [Float], sampleRate: Double, speed: Speed) async throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw TTSError.playbackFailed("Could not construct AVAudioFormat at \(sampleRate)Hz mono")
        }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw TTSError.playbackFailed("Could not allocate PCM buffer for \(samples.count) frames")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let dst = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: samples.count)
            }
        }

        let (engine, player, varispeed) = setupGraph(format: format)
        varispeed.rate = Float(speed.value)

        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                throw TTSError.playbackFailed("AVAudioEngine.start failed: \(error)")
            }
        }
        player.play()

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
                cont.resume()
            }
        }
    }

    private func setupGraph(format: AVAudioFormat) -> (AVAudioEngine, AVAudioPlayerNode, AVAudioUnitVarispeed) {
        if let engine, let player = playerNode, let varispeed = varispeedNode {
            return (engine, player, varispeed)
        }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let varispeed = AVAudioUnitVarispeed()
        engine.attach(player)
        engine.attach(varispeed)
        engine.connect(player, to: varispeed, format: format)
        engine.connect(varispeed, to: engine.mainMixerNode, format: nil)
        self.engine = engine
        self.playerNode = player
        self.varispeedNode = varispeed
        return (engine, player, varispeed)
    }

    // MARK: - Voice → language mapping

    /// VoiceID is repurposed as a language code under the hood for Qwen3-TTS.
    /// v1 ships English only; Task 11's language picker fills in the rest.
    nonisolated static func languageString(for voice: VoiceID) -> String {
        switch voice.rawValue {
        case VoiceID.systemDefault.rawValue: return "english"
        case "english", "chinese", "japanese", "korean",
             "russian", "german", "french", "italian",
             "spanish", "portuguese":
            return voice.rawValue
        default:
            return "english"
        }
    }
}
