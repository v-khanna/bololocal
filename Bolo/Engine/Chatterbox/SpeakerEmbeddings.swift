// Bolo/Engine/Chatterbox/SpeakerEmbeddings.swift
import Foundation
import MLX

/// Pre-computed conditioning tensors loaded from the bundled conds.safetensors.
///
/// Chatterbox-Turbo's conditioning comes from two namespaces:
///
/// **T3 (text-to-token) conditioning:**
///   - `speakerEmbedding` (256-d) — CAMPPlus x-vector identifying the voice (t3.speaker_emb)
///   - `condPromptSpeechTokens` — speech token prompt for T3 conditioning (t3.cond_prompt_speech_tokens)
///   - `emotionAdv` — emotion adversarial control scalar (t3.emotion_adv)
///
/// **S3Gen (token-to-speech) conditioning — gen.* namespace:**
///   - `gen.embedding` (192-d) — S3Gen speaker embedding
///   - `gen.prompt_token` — reference speech tokens
///   - `gen.prompt_feat` — reference mel-spectrogram features (500 frames × 80 bins)
///   - `gen.prompt_token_len` — valid length of prompt_token
///
/// In v1 we ship a single preset voice (the default bundled with the model at
/// mlx-community/chatterbox-turbo-fp16). CAMPPlus Voice Encoder is bypassed;
/// user voice cloning lands in v1.1 once VE is ported.
struct SpeakerEmbeddings: Sendable {
    // MARK: - T3 conditioning (t3.* namespace)

    /// 256-d speaker embedding for the default preset voice (CAMPPlus x-vector).
    let speakerEmbedding: [Float]

    /// Speech token prompt used for T3 conditioning (t3.cond_prompt_speech_tokens).
    /// Shape: (375,) int32, stored as Int32.
    let condPromptSpeechTokens: [Int32]

    /// Emotion adversarial control scalar (t3.emotion_adv). Shape: (1, 1, 1).
    let emotionAdv: [Float]

    // MARK: - S3Gen conditioning (gen.* namespace)

    /// All gen.* tensors for S3Gen flow_inference, keyed by their original names.
    /// Float tensors are stored as [Float]; integer tensors (prompt_token, prompt_token_len)
    /// are stored as [Int32] via the genIntTensors companion dictionary.
    let s3GenConditioning: [String: [Float]]

    /// Integer tensors from the gen.* namespace (prompt_token, prompt_token_len).
    let s3GenIntTensors: [String: [Int32]]

    // MARK: - Bundle loader

    /// Load from the bundled conds.safetensors.
    static func loadFromBundle() throws -> SpeakerEmbeddings {
        let bundle = Bundle.main
        guard let url = bundle.url(forResource: "conds", withExtension: "safetensors")
            ?? bundle.url(forResource: "conds", withExtension: "safetensors", subdirectory: "Resources")
        else {
            throw TTSError.synthesisFailed(
                "SpeakerEmbeddings: conds.safetensors not bundled. Verify project.yml resources block."
            )
        }
        return try loadFromURL(url)
    }

    static func loadFromURL(_ url: URL) throws -> SpeakerEmbeddings {
        let arrays = try MLX.loadArrays(url: url)

        // Helper — extract 1D float array from an MLXArray, casting to float32 if needed
        func asFloats(_ arr: MLXArray) -> [Float] {
            arr.asType(.float32).asArray(Float.self)
        }

        // Helper — extract 1D int32 array from an MLXArray
        func asInt32s(_ arr: MLXArray) -> [Int32] {
            arr.asArray(Int32.self)
        }

        // Helper — require a key, with helpful error if missing
        func require(_ key: String) throws -> MLXArray {
            guard let arr = arrays[key] else {
                throw TTSError.synthesisFailed(
                    "SpeakerEmbeddings: key '\(key)' not found in conds.safetensors. " +
                    "Available keys: \(arrays.keys.sorted())"
                )
            }
            return arr
        }

        // T3 conditioning
        let speakerEmb = try require("t3.speaker_emb")   // (1, 256) float32
        let condTokens = try require("t3.cond_prompt_speech_tokens")  // (1, 375) int32
        let emotionAdvArr = try require("t3.emotion_adv")  // (1, 1, 1) float32

        // S3Gen conditioning — split float and int tensors
        var s3genFloats: [String: [Float]] = [:]
        var s3genInts: [String: [Int32]] = [:]
        for (key, value) in arrays where key.hasPrefix("gen.") {
            if value.dtype == .int32 || value.dtype == .int64 || value.dtype == .uint32 {
                s3genInts[key] = asInt32s(value.asType(.int32))
            } else {
                s3genFloats[key] = asFloats(value)
            }
        }

        return SpeakerEmbeddings(
            speakerEmbedding: asFloats(speakerEmb),
            condPromptSpeechTokens: asInt32s(condTokens.asType(.int32)),
            emotionAdv: asFloats(emotionAdvArr),
            s3GenConditioning: s3genFloats,
            s3GenIntTensors: s3genInts
        )
    }
}
