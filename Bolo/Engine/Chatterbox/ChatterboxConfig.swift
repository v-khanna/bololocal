// Bolo/Engine/Chatterbox/ChatterboxConfig.swift
import Foundation

/// Hyperparameters for the Chatterbox-Turbo model. Single source of truth.
/// Values pinned from the official config.json at
/// https://huggingface.co/mlx-community/chatterbox-turbo-fp16/blob/main/config.json
///
/// Architecture: GPT-2 style backbone (T3) feeding a Conformer-based flow-matching
/// decoder (S3Gen). Despite some early research describing the backbone as
/// "Llama-3 variant," the live config explicitly identifies it as GPT-2:
/// learned absolute positional embeddings, LayerNorm (not RMSNorm), standard
/// multi-head attention (not GQA), and gelu_new activation (not SwiGLU).
struct ChatterboxConfig: Sendable, Equatable {

    struct T3: Sendable, Equatable {
        let numLayers: Int
        let hiddenDim: Int
        let numHeads: Int
        let headDim: Int
        let vocabSize: Int        // text BPE vocab
        let maxContextLength: Int
        let layerNormEps: Double
        let activation: String    // "gelu_new"
    }

    struct S3Gen: Sendable, Equatable {
        let tokenEmbeddingDim: Int
        let encoderNumBlocks: Int
        let encoderAttentionHeads: Int
        let encoderLinearUnits: Int
        let decoderNumBlocks: Int
        let decoderNumMidBlocks: Int
        let decoderNumHeads: Int
        let decoderAttentionHeadDim: Int
        let speechVocabSize: Int  // 6,561 = 3^8
    }

    let t3: T3
    let s3gen: S3Gen

    /// Output audio sample rate produced by the vocoder.
    static let audioSampleRate: Double = 24000

    /// Speaker embedding dimensionality (from CAMPPlus voice encoder; we bypass
    /// the encoder in v1 and use precomputed embeddings from conds.safetensors).
    static let speakerEmbeddingDim: Int = 192

    /// Chatterbox-Turbo: 350M T3 backbone + distilled 1-step S3Gen decoder.
    /// English-only. The configuration we ship in v1.
    static let turbo = ChatterboxConfig(
        t3: T3(
            numLayers: 24,
            hiddenDim: 1024,
            numHeads: 16,
            headDim: 64,
            vocabSize: 50276,
            maxContextLength: 8196,
            layerNormEps: 1e-5,
            activation: "gelu_new"
        ),
        s3gen: S3Gen(
            tokenEmbeddingDim: 512,
            encoderNumBlocks: 6,
            encoderAttentionHeads: 8,
            encoderLinearUnits: 2048,
            decoderNumBlocks: 4,
            decoderNumMidBlocks: 12,
            decoderNumHeads: 8,
            decoderAttentionHeadDim: 64,
            speechVocabSize: 6561
        )
    )
}
