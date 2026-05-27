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
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegate?

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
        audioPlayer?.stop()
        audioPlayer = nil
        audioPlayerDelegate = nil
        playerNode?.stop()
        engine?.stop()
    }

    // MARK: - Playback

    private func play(samples: [Float], sampleRate: Double, speed: Speed) async throws {
        NSLog("Bolo play(): \(samples.count) samples at \(sampleRate) Hz, speed=\(speed.value)x")
        guard !samples.isEmpty else {
            throw TTSError.playbackFailed("Chatterbox returned 0 audio samples")
        }

        // Build an in-memory WAV file. AVAudioPlayer handles SR/channel conversion
        // internally — much more reliable than the AVAudioEngine render-graph dance
        // that gave us 2× pitch + truncation + silent renders. This approach: build
        // a WAV blob, hand it to AVAudioPlayer, set rate for speed, play, await done.
        let wavData = makeWavFile(monoFloats: samples, sampleRate: Int(sampleRate))
        NSLog("Bolo play(): WAV blob is \(wavData.count) bytes")

        let player: AVAudioPlayer
        do {
            player = try AVAudioPlayer(data: wavData)
        } catch {
            throw TTSError.playbackFailed("AVAudioPlayer init failed: \(error)")
        }

        // AVAudioPlayer.rate is in [0.5, 2.0] when enableRate is true — matches our Speed clamping.
        player.enableRate = true
        player.rate = Float(speed.value)
        player.volume = 1.0
        player.prepareToPlay()

        // Keep a strong reference so player + delegate aren't GC'd mid-playback.
        self.audioPlayer = player

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            let delegate = AudioPlayerDelegate(onFinish: {
                NSLog("Bolo play(): playback finished")
                cont.resume()
            })
            self.audioPlayerDelegate = delegate
            player.delegate = delegate

            let started = player.play()
            NSLog("Bolo play(): AVAudioPlayer.play() returned \(started)")
            if !started {
                NSLog("Bolo play(): play() returned false; resuming continuation early")
                cont.resume()
            }
        }
    }

    /// Build a WAV file in memory from mono float samples.
    /// Format: WAVE, 32-bit float PCM, mono.
    private func makeWavFile(monoFloats samples: [Float], sampleRate: Int) -> Data {
        let bitsPerSample: UInt16 = 32
        let channels: UInt16 = 1
        let byteRate = UInt32(sampleRate) * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign: UInt16 = channels * (bitsPerSample / 8)
        let audioBytes = samples.count * MemoryLayout<Float>.size
        let dataChunkSize = UInt32(audioBytes)
        let fmtChunkSize: UInt32 = 16
        let riffChunkSize = 4 + (8 + fmtChunkSize) + (8 + dataChunkSize)

        var data = Data()
        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: riffChunkSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fmtChunkSize.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) })  // format = 3 (IEEE float)
        data.append(contentsOf: withUnsafeBytes(of: channels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataChunkSize.littleEndian) { Array($0) })
        samples.withUnsafeBufferPointer { ptr in
            data.append(UnsafeBufferPointer(start: ptr.baseAddress, count: samples.count))
        }
        return data
    }

    // Legacy graph setup — kept for reference but no longer used. Audio path is now
    // AVAudioPlayer-based for reliability.
    private func setupGraph_legacy(
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
        let outputFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        engine.connect(player, to: varispeed, format: outputFormat)
        engine.connect(varispeed, to: engine.mainMixerNode, format: outputFormat)
        self.engine = engine
        self.playerNode = player
        self.varispeedNode = varispeed
        return (engine, player, varispeed)
    }
}

/// Bridges AVAudioPlayer's delegate callback to a closure so the actor can await
/// playback completion via withCheckedContinuation.
final class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate, @unchecked Sendable {
    let onFinish: @Sendable () -> Void
    init(onFinish: @escaping @Sendable () -> Void) {
        self.onFinish = onFinish
    }
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish()
    }
    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        NSLog("Bolo AudioPlayerDelegate decode error: \(error?.localizedDescription ?? "nil")")
        onFinish()
    }
}
