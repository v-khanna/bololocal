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
/// Streaming architecture (sentence-chunked):
/// - `synthesize(text:)` splits text into sentences via `SentenceSplitter`.
/// - For each sentence, we call `pipeline.generate(text:)` sequentially.
/// - As each sentence's float samples arrive, we schedule them on an
///   `AVAudioPlayerNode` via `scheduleBuffer(...)`. The player keeps draining
///   buffers back-to-back with no gap, so synthesis of sentence N+1 overlaps
///   playback of sentence N.
/// - First-audio latency = synth time for the FIRST sentence only, not the
///   whole paragraph. We log this as the perceived-latency win.
///
/// v1 Notes:
/// - `voice: VoiceID` is currently unused. Chatterbox v1 ships a single preset
///   voice baked into `conds.safetensors`. Future phases will add voice cloning.
/// - `speed: Speed` is applied via `AVAudioUnitVarispeed` in the audio graph.
/// - Sample rate is 24 kHz mono (matching Chatterbox-Turbo's vocoder output).
actor ChatterboxTTSEngine: TTSEngine {
    private let modelProvider: @Sendable () async throws -> ChatterboxPipeline

    // Audio playback graph — created on first synthesize, reused across calls.
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    /// Number of buffers currently scheduled but not yet finished playing.
    /// We use this to know when playback has fully drained for a given synth.
    private var pendingBuffers: Int = 0

    /// Continuation that resolves when `pendingBuffers` reaches zero AND the
    /// synthesis loop has finished producing buffers. Used by `_synthesize`
    /// to keep its async frame alive until all audio has drained.
    private var drainContinuation: CheckedContinuation<Void, Never>?
    /// Set true once the sentence-synth loop is done. Once this is true AND
    /// `pendingBuffers == 0`, we resume `drainContinuation`.
    private var synthLoopFinished: Bool = false

    /// Hook for observers (e.g. UI) to react when the first sentence's audio
    /// is queued for playback. For now used only to log perceived-latency.
    var onFirstAudioReady: (@Sendable () -> Void)?

    /// Sample rate produced by ChatterboxPipeline's vocoder.
    static let sampleRate: Double = 24_000

    init(modelProvider: @escaping @Sendable () async throws -> ChatterboxPipeline) {
        self.modelProvider = modelProvider
    }

    nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await _synthesize(text: text, voice: voice, speed: speed)
    }

    private func _synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        // Tear down any in-flight playback BEFORE starting a new synthesis.
        _stop()

        // Adaptive chunking. Sentence-streaming earns its keep only on long
        // text, where starting audio after sentence 1 hides synthesis latency
        // and the inter-chunk gaps disappear among lots of content. On SHORT
        // text the gaps are glaring (sentence 1 plays in ~2s, then silence
        // while sentence 2 synthesizes for ~3s — synthesis is slower than
        // real-time) AND chunking needlessly breaks prosody across sentence
        // boundaries. So: below the threshold, synthesize the whole selection
        // as ONE continuous unit (no gap, better prosody); above it, stream.
        let streamingThreshold = 280   // chars; ~2-3 sentences
        let sentences: [String]
        if text.count <= streamingThreshold {
            sentences = [text]
        } else {
            sentences = SentenceSplitter.split(text)
        }
        BoloDebug.log("ChatterboxTTSEngine._synthesize start: \(sentences.count) chunk(s) (\(text.count) chars, streaming=\(text.count > streamingThreshold)), text=\(text.prefix(60))…")
        guard !sentences.isEmpty else { throw TTSError.emptyText }

        let pipeline: ChatterboxPipeline
        do {
            pipeline = try await modelProvider()
            BoloDebug.log("modelProvider() returned pipeline OK")
        } catch {
            BoloDebug.log("modelProvider() threw: \(error)")
            throw error
        }

        // Build / configure the audio graph for this synthesis.
        // NOTE: `speed` is currently ignored. AVAudioUnitVarispeed in the graph
        // made AVAudioEngine.start() fail with -10868 (kAudioUnitErr_FormatNotSupported)
        // because its rate-shifted format couldn't propagate to the output chain
        // (especially on Bluetooth output in call mode). Speed will be re-added
        // via a more robust path (AVAudioUnitTimePitch or pre-resampling). See v2.17.
        _ = speed

        // ── Phase 1: synthesize EVERYTHING first, into one continuous buffer ──
        // We accept the upfront wait in exchange for gapless playback. Long text
        // is split into sentences only so each model call stays bounded and
        // high-quality; the resulting samples are concatenated so playback never
        // stutters between sentences. (Short text is already a single chunk.)
        let synthStart = Date()
        var allSamples: [Float] = []
        for (idx, sentence) in sentences.enumerated() {
            if Task.isCancelled {
                BoloDebug.log("synthesis cancelled at chunk \(idx) (nothing scheduled yet)")
                return
            }
            let samples: [Float]
            do {
                samples = try await pipeline.generate(text: sentence)
            } catch {
                // First chunk failing is a hard error; later chunks we tolerate.
                if idx == 0 { throw error }
                BoloDebug.log("pipeline.generate(chunk #\(idx)) threw: \(error); skipping")
                continue
            }
            guard !samples.isEmpty else {
                BoloDebug.log("chunk #\(idx) produced 0 samples; skipping")
                continue
            }
            if idx == 0 { onFirstAudioReady?() }
            allSamples.append(contentsOf: samples)
            BoloDebug.log("chunk #\(idx): +\(samples.count) samples (running total \(allSamples.count))")
        }
        BoloDebug.log(String(format: "all chunks synthesized in %.2fs — %d samples (%.1fs audio)",
                             Date().timeIntervalSince(synthStart),
                             allSamples.count, Double(allSamples.count) / Self.sampleRate))

        guard !allSamples.isEmpty else {
            throw TTSError.synthesisFailed("ChatterboxPipeline produced no audio")
        }
        if Task.isCancelled { return }

        // ── Phase 2: play the whole thing as ONE gapless buffer ──
        let (audioEngine, player) = try setupGraph()
        if !audioEngine.isRunning {
            do {
                try audioEngine.start()
                BoloDebug.log("AVAudioEngine started")
            } catch {
                BoloDebug.log("AVAudioEngine.start() threw: \(error)")
                // Tear down the wedged engine so the next synth rebuilds fresh.
                engine?.stop()
                engine = nil
                playerNode = nil
                throw TTSError.playbackFailed("AVAudioEngine.start() failed: \(error)")
            }
        }
        player.play()

        synthLoopFinished = false
        pendingBuffers = 0
        scheduleSamples(allSamples, on: player)
        synthLoopFinished = true
        maybeResumeDrain()

        // Block until all queued buffers have drained (or stop() fires).
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            if synthLoopFinished && pendingBuffers == 0 {
                cont.resume()
            } else {
                drainContinuation = cont
            }
        }
        BoloDebug.log("playback fully drained")
    }

    nonisolated func stop() {
        Task { await self._stop() }
    }

    private func _stop() {
        // Cancel any in-flight synthesis would happen at the PlaybackController
        // task level — here we tear down audio playback specifically.
        playerNode?.stop()
        engine?.stop()
        // Drop any queued buffers — scheduleBuffer completions will still fire
        // (with .dataPlayedBack OR cancelled status) but we treat them as no-ops
        // via the pending counter being reset.
        pendingBuffers = 0
        synthLoopFinished = true
        // If a synth was awaiting drain, resume it so the task can exit.
        if let cont = drainContinuation {
            drainContinuation = nil
            cont.resume()
        }
    }

    // MARK: - Audio graph

    /// Build the streaming audio graph (engine → playerNode → mainMixer).
    /// Reuses the existing instances on subsequent synths to avoid churn.
    ///
    /// We deliberately connect the player straight to `mainMixerNode` with our
    /// 24 kHz mono source format and let the mixer resample to the output
    /// device's native format. We do NOT insert AVAudioUnitVarispeed: with it
    /// in the chain, `engine.start()` fails with -10868
    /// (kAudioUnitErr_FormatNotSupported) because its rate-shifted output
    /// format can't propagate down the output chain on some devices (notably
    /// Bluetooth headphones in call/SCO mode). That's the regression that made
    /// every playback silently fail. Speed control is reapplied separately.
    private func setupGraph() throws -> (AVAudioEngine, AVAudioPlayerNode) {
        // If a prior synth left a usable engine + player, reuse them — BUT only
        // if the engine isn't in a failed state. If a previous start() threw
        // -10868 the cached engine can be wedged; rebuild from scratch in that
        // case rather than handing back broken state.
        if let engine, let player = playerNode {
            return (engine, player)
        }
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)

        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw TTSError.playbackFailed("Failed to construct 24kHz mono AVAudioFormat")
        }
        // Touch mainMixerNode first so the engine instantiates its output chain
        // (mixer → hardware) at the device's native format, then connect the
        // player with our source format. The mixer handles 24 kHz mono → device
        // resampling internally.
        let mixer = engine.mainMixerNode
        engine.connect(player, to: mixer, format: sourceFormat)
        engine.prepare()

        self.engine = engine
        self.playerNode = player
        return (engine, player)
    }

    /// Convert raw mono float samples into an `AVAudioPCMBuffer` at 24kHz mono
    /// and schedule it on the player node. We bypass the WAV-blob detour here
    /// because `scheduleBuffer` takes PCM buffers directly.
    private func scheduleSamples(_ samples: [Float], on player: AVAudioPlayerNode) {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            BoloDebug.log("scheduleSamples: failed to construct format")
            return
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            BoloDebug.log("scheduleSamples: AVAudioPCMBuffer alloc failed")
            return
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            BoloDebug.log("scheduleSamples: floatChannelData nil")
            return
        }
        samples.withUnsafeBufferPointer { src in
            channel.update(from: src.baseAddress!, count: samples.count)
        }

        pendingBuffers += 1
        // scheduleBuffer's completion handler runs on an arbitrary thread —
        // hop back into the actor to mutate counters.
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { [weak self] in
                await self?.bufferFinished()
            }
        }
    }

    private func bufferFinished() {
        if pendingBuffers > 0 { pendingBuffers -= 1 }
        maybeResumeDrain()
    }

    private func maybeResumeDrain() {
        if synthLoopFinished && pendingBuffers == 0, let cont = drainContinuation {
            drainContinuation = nil
            cont.resume()
        }
    }

    // MARK: - WAV (legacy debug helper)

    /// Build a WAV file in memory from mono float samples. No longer on the
    /// hot path (we go straight to AVAudioPCMBuffer for streaming) but kept
    /// here so debug snapshots of generated audio can still be written to
    /// /tmp/bolo-last.wav by callers if needed.
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
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: riffChunkSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fmtChunkSize.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: channels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataChunkSize.littleEndian) { Array($0) })
        samples.withUnsafeBufferPointer { ptr in
            data.append(UnsafeBufferPointer(start: ptr.baseAddress, count: samples.count))
        }
        return data
    }
}
