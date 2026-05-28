// Bolo/Engine/GroqTTSEngine.swift
// Cloud TTS via Groq's OpenAI-compatible /audio/speech endpoint (Orpheus
// models). This is the DEFAULT engine — fast, no model download, no MLX at
// runtime. The on-device ChatterboxTTSEngine remains available as an opt-in.
//
// Flow: text → POST to Groq → WAV bytes → AVAudioPlayer(data:) → speakers.
// We deliberately use AVAudioPlayer (not AVAudioEngine + varispeed) because
// Groq returns a complete WAV and AVAudioPlayer is the simplest reliable
// player — it also sidesteps the -10868 format-negotiation failure that the
// engine+varispeed graph hit on Bluetooth output.

import Foundation
import AVFoundation

actor GroqTTSEngine: TTSEngine {
    /// Provides the current API key + voice at call time (reads live Settings).
    private let configProvider: @Sendable () async -> (apiKey: String, voice: String)

    private var audioPlayer: AVAudioPlayer?
    private var delegate: AudioPlayerDelegate?
    private var cancelled = false

    /// Groq enforces a per-request input limit; chunk conservatively by
    /// sentence so big selections still work, then play chunks back-to-back.
    private static let maxCharsPerRequest = 9_000
    private static let endpoint = URL(string: "https://api.groq.com/openai/v1/audio/speech")!
    private static let model = "canopylabs/orpheus-v1-english"

    init(configProvider: @escaping @Sendable () async -> (apiKey: String, voice: String)) {
        self.configProvider = configProvider
    }

    nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await _synthesize(text: text)
    }

    private func _synthesize(text: String) async throws {
        cancelled = false
        let (apiKey, voiceName) = await configProvider()
        guard !apiKey.isEmpty else {
            throw TTSError.synthesisFailed("No Groq API key set. Add one in Settings → General.")
        }

        // Chunk long text at sentence boundaries so each request stays under
        // Groq's limit. Most selections are a single chunk.
        let chunks = Self.chunk(text, maxChars: Self.maxCharsPerRequest)
        BoloDebug.log("GroqTTSEngine: \(chunks.count) chunk(s), \(text.count) chars, voice=\(voiceName)")

        let t0 = Date()
        for (i, chunk) in chunks.enumerated() {
            if cancelled { BoloDebug.log("GroqTTS cancelled at chunk \(i)"); return }
            let wav = try await fetch(text: chunk, apiKey: apiKey, voice: voiceName)
            BoloDebug.log("GroqTTS chunk \(i): \(wav.count) bytes in \(String(format: "%.2fs", Date().timeIntervalSince(t0)))")
            if cancelled { return }
            try await play(wav: wav)   // awaits until this chunk finishes
        }
    }

    // MARK: - Network

    private func fetch(text: String, apiKey: String, voice: String) async throws -> Data {
        var req = URLRequest(url: Self.endpoint)
        req.httpMethod = "POST"
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": Self.model,
            "input": text,
            "voice": voice,
            "response_format": "wav",
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        } catch {
            throw TTSError.synthesisFailed("Network error: \(error.localizedDescription)")
        }
        guard let http = response as? HTTPURLResponse else {
            throw TTSError.synthesisFailed("No HTTP response from Groq")
        }
        guard (200..<300).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "status \(http.statusCode)"
            throw TTSError.synthesisFailed("Groq \(http.statusCode): \(msg.prefix(200))")
        }
        return data
    }

    // MARK: - Playback (AVAudioPlayer — complete WAV, no varispeed)

    private func play(wav: Data) async throws {
        let player: AVAudioPlayer
        do {
            player = try AVAudioPlayer(data: wav)
        } catch {
            throw TTSError.playbackFailed("AVAudioPlayer init failed: \(error)")
        }
        player.volume = 1.0
        player.prepareToPlay()
        self.audioPlayer = player

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            let delegate = AudioPlayerDelegate(onFinish: { cont.resume() })
            self.delegate = delegate
            player.delegate = delegate
            if !player.play() {
                BoloDebug.log("GroqTTS: AVAudioPlayer.play() returned false")
                cont.resume()
            }
        }
    }

    nonisolated func stop() {
        Task { await self._stop() }
    }

    private func _stop() {
        cancelled = true
        audioPlayer?.stop()
        audioPlayer = nil
        delegate = nil
    }

    // MARK: - Chunking

    /// Split text into <= maxChars pieces at sentence boundaries (falling back
    /// to hard splits for pathologically long sentences).
    static func chunk(_ text: String, maxChars: Int) -> [String] {
        if text.count <= maxChars { return [text] }
        var chunks: [String] = []
        var current = ""
        for sentence in SentenceSplitter.split(text) {
            if current.count + sentence.count > maxChars, !current.isEmpty {
                chunks.append(current)
                current = ""
            }
            if sentence.count > maxChars {
                // Hard-split an oversized sentence.
                var s = Substring(sentence)
                while !s.isEmpty {
                    let end = s.index(s.startIndex, offsetBy: min(maxChars, s.count))
                    chunks.append(String(s[s.startIndex..<end]))
                    s = s[end...]
                }
            } else {
                current += sentence
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}

/// Bridges AVAudioPlayer's delegate callback to a closure so the actor can
/// await playback completion via withCheckedContinuation.
final class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate, @unchecked Sendable {
    let onFinish: @Sendable () -> Void
    init(onFinish: @escaping @Sendable () -> Void) { self.onFinish = onFinish }
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) { onFinish() }
    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        BoloDebug.log("GroqTTS decode error: \(error?.localizedDescription ?? "nil")")
        onFinish()
    }
}
