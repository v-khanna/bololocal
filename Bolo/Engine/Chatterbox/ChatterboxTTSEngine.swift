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
        // Source format: what Chatterbox produces (24 kHz mono float32 non-interleaved)
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw TTSError.playbackFailed("Could not construct source AVAudioFormat at \(sampleRate)Hz mono")
        }

        // Wrap the raw samples into a source-format buffer
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw TTSError.playbackFailed("Could not allocate source PCM buffer for \(samples.count) frames")
        }
        sourceBuffer.frameLength = AVAudioFrameCount(samples.count)
        if let dst = sourceBuffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { src in
                dst.update(from: src.baseAddress!, count: samples.count)
            }
        }

        // Set up the graph. We deliberately run the ENTIRE graph at the system output
        // format (typically 48 kHz stereo) — sample-rate and channel conversion happens
        // ONCE here via AVAudioConverter, before any node sees the buffer. Relying on
        // mainMixerNode for implicit resampling causes 2× pitch + truncation in practice;
        // pre-conversion via AVAudioConverter is the reliable path.
        let (engine, player, varispeed) = setupGraph(forSourceFormat: sourceFormat)
        varispeed.rate = Float(speed.value)

        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                throw TTSError.playbackFailed("AVAudioEngine.start failed: \(error)")
            }
        }

        // Convert source buffer → output format (e.g. 48 kHz stereo)
        let outputFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: sourceFormat, to: outputFormat) else {
            throw TTSError.playbackFailed("Could not create AVAudioConverter \(sourceFormat) → \(outputFormat)")
        }
        let ratio = outputFormat.sampleRate / sourceFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio + 1024)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outputCapacity
        ) else {
            throw TTSError.playbackFailed("Could not allocate output PCM buffer")
        }
        var convertError: NSError?
        var consumed = false
        converter.convert(to: outputBuffer, error: &convertError) { _, statusPtr in
            if consumed {
                statusPtr.pointee = .endOfStream
                return nil
            }
            consumed = true
            statusPtr.pointee = .haveData
            return sourceBuffer
        }
        if let err = convertError {
            throw TTSError.playbackFailed("AVAudioConverter failed: \(err)")
        }

        player.play()

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(outputBuffer, completionCallbackType: .dataPlayedBack) { _ in
                cont.resume()
            }
        }
    }

    private func setupGraph(
        forSourceFormat sourceFormat: AVAudioFormat
    ) -> (AVAudioEngine, AVAudioPlayerNode, AVAudioUnitVarispeed) {
        if let engine, let player = playerNode, let varispeed = varispeedNode {
            return (engine, player, varispeed)
        }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let varispeed = AVAudioUnitVarispeed()
        engine.attach(player)
        engine.attach(varispeed)
        // Connect the whole graph at the engine's output format (which itself was
        // negotiated to match the system output device — typically 48 kHz stereo).
        // We pre-convert source 24 kHz mono → output format via AVAudioConverter
        // before scheduling, so every node sees the same format. No implicit
        // resampling anywhere.
        let outputFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        engine.connect(player, to: varispeed, format: outputFormat)
        engine.connect(varispeed, to: engine.mainMixerNode, format: outputFormat)
        self.engine = engine
        self.playerNode = player
        self.varispeedNode = varispeed
        return (engine, player, varispeed)
    }
}
