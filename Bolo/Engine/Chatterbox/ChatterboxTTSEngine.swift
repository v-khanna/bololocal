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
/// Phases 4-6 of the plan fill in the real pipeline. This file currently stubs
/// synthesize to throw — verifying the protocol conformance and integration
/// surface compile cleanly before any inference code lands.
actor ChatterboxTTSEngine: TTSEngine {
    private let modelProvider: @Sendable () async throws -> ChatterboxModel

    init(modelProvider: @escaping @Sendable () async throws -> ChatterboxModel) {
        self.modelProvider = modelProvider
    }

    nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        guard !text.isEmpty else { throw TTSError.emptyText }
        try await _synthesize(text: text, voice: voice, speed: speed)
    }

    private func _synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
        // Phases 4-6 (Tasks 8-20) implement the real pipeline. Stub for now.
        throw TTSError.synthesisFailed("ChatterboxTTSEngine: not implemented yet")
    }

    nonisolated func stop() {
        // Phase 6 (Task 20): stop AVAudioEngine playback.
    }
}

/// Top-level model container — Tokenizer + SpeakerEmbeddings + T3 + S3Gen + Vocoder.
/// Populated in Phase 6 (Task 19). Empty struct for the stub so ChatterboxTTSEngine compiles.
struct ChatterboxModel: Sendable {
    // Populated in later tasks.
}
