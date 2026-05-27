// Bolo/Engine/Chatterbox/ChatterboxTTSEngine.swift
import Foundation
import AVFoundation

/// Chatterbox-Turbo TTS engine. Native MLX-Swift port of Resemble AI's
/// Chatterbox-Turbo model (MIT, https://github.com/resemble-ai/chatterbox).
///
/// Conforms to TTSEngine — drop-in replacement for Qwen3TTSEngine when ready.
/// Model lifecycle is owned externally by ModelManager; this actor receives a
/// `modelProvider` closure and calls it on each synthesize.
///
/// v1 Notes:
/// - `voice: VoiceID` is currently unused. Chatterbox v1 ships a single preset
///   voice baked into `conds.safetensors`. Future phases will add voice cloning.
/// - `speed: Speed` is applied post-hoc via `AVAudioUnitVarispeed.rate`, same
///   pattern as `Qwen3TTSEngine`.
/// - Sample rate is 24 kHz mono (matching Chatterbox-Turbo's vocoder output).
actor ChatterboxTTSEngine: TTSEngine {
    private let modelProvider: @Sendable () async throws -> ChatterboxPipeline

    // Audio playback graph — created on first synthesize, reused across calls.
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var varispeedNode: AVAudioUnitVarispeed?

    /// Sample rate produced by ChatterboxPipeline's vocoder.
    private static let sampleRate: Double = 24_000

    init(modelProvider: @escaping @Sendable () async throws -> ChatterboxPipeline) {
        self.modelProvider = modelProvider
    }

    nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await _synthesize(text: text, voice: voice, speed: speed)
    }

    private func _synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        let pipeline = try await modelProvider()

        // voice is unused in v1 — Chatterbox has one preset voice.
        // TODO(Phase 7+): pass voice clone reference audio when voice cloning lands.
        let samples: [Float] = try await pipeline.generate(text: text)

        guard !samples.isEmpty else {
            throw TTSError.synthesisFailed("ChatterboxPipeline returned 0 samples")
        }

        try await play(samples: samples, sampleRate: Self.sampleRate, speed: speed)
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

    private func setupGraph(
        format: AVAudioFormat
    ) -> (AVAudioEngine, AVAudioPlayerNode, AVAudioUnitVarispeed) {
        if let engine, let player = playerNode, let varispeed = varispeedNode {
            return (engine, player, varispeed)
        }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let varispeed = AVAudioUnitVarispeed()
        engine.attach(player)
        engine.attach(varispeed)
        // Varispeed is a time-stretcher, not a format converter. Its input and output
        // must be the same format (24 kHz mono float32 from Chatterbox). The mainMixerNode
        // handles the sample-rate + channel-count conversion to the system output format
        // (typically 48 kHz stereo). Passing format: nil on the second connect makes the
        // engine try to make Varispeed produce 48 kHz stereo, which fails with -10868
        // (kAudioUnitErr_FormatNotSupported).
        engine.connect(player, to: varispeed, format: format)
        engine.connect(varispeed, to: engine.mainMixerNode, format: format)
        self.engine = engine
        self.playerNode = player
        self.varispeedNode = varispeed
        return (engine, player, varispeed)
    }
}
