// Bolo/Engine/Chatterbox/S3Gen/S3Gen.swift
import Foundation
import MLX
import MLXNN

/// Swift port of the S3Gen flow-matching decoder.
///
/// Status (Phase 5 in-progress): the *deterministic encoder prefix* is fully
/// implemented and matched against the Python reference, but the CFM diffusion
/// decoder (`decoder`) and HiFiGAN vocoder (`mel2wav`) are not yet ported.
/// Use `encodeForDecoder` to exercise the encoder + projection + speaker affine
/// stages, which is the parity gate this phase ships.
///
/// Architecture (matches `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.S3Token2Wav`):
///
///   input_embedding     — Embedding(speech_vocab_size=6561, 512)
///   speaker_encoder     — CAMPPlus x-vector net    [NOT ported; use precomputed embedding]
///   spk_embed_affine    — Linear(192, 80)
///   encoder             — UpsampleConformerEncoder (this file's main payload)
///   encoder_proj        — Linear(512, 80)
///   decoder             — CausalConditionalCFM      [NOT yet ported]
///   mel2wav             — HiFTGenerator             [NOT yet ported]
///
/// Notes:
/// - The Python `S3Token2Mel.__call__` first concatenates the reference voice's
///   `prompt_token` with the speech tokens we want to synthesize, then runs the
///   encoder over the concatenation. The reference prompt comes from
///   `gen.prompt_token` in `conds.safetensors`.
/// - Token embeddings are zero-masked outside the valid length before the encoder
///   runs (see `encodeForDecoder`).
final class S3Gen: Module {

    let config: ChatterboxConfig.S3Gen

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear

    init(config: ChatterboxConfig.S3Gen) {
        self.config = config

        self._inputEmbedding.wrappedValue = Embedding(
            embeddingCount: config.speechVocabSize,
            dimensions: config.tokenEmbeddingDim
        )
        // Speaker affine: norm(x-vector 192d) → 80d
        self._spkEmbedAffineLayer.wrappedValue = Linear(
            ChatterboxConfig.speakerEmbeddingDim, 80, bias: true)
        self._encoder.wrappedValue = UpsampleConformerEncoder(config: config)
        // Project encoder hidden (512) to mel channels (80).
        self._encoderProj.wrappedValue = Linear(config.tokenEmbeddingDim, 80, bias: true)
        super.init()
    }

    /// Output of `encodeForDecoder`. Everything the CFM decoder needs from the
    /// encoder side of the pipeline.
    struct EncoderOutputs {
        /// Encoder hidden state, projected to mel channels.
        /// Shape `(B, 2(T_prompt + T_speech), 80)`.
        let encoderProjOut: MLXArray
        /// Raw encoder hidden state, BEFORE the projection.
        /// Shape `(B, 2(T_prompt + T_speech), 512)`.
        let encoderOut: MLXArray
        /// Output mask `(B, 1, 2(T_prompt + T_speech))`.
        let encoderMask: MLXArray
        /// Projected speaker embedding `(B, 80)`.
        let speakerEmbedding: MLXArray
    }

    /// Run the deterministic encoder prefix of S3Token2Mel.
    ///
    /// - Parameters:
    ///   - speechTokens: `(B, T_speech)` int32 generated speech tokens.
    ///   - promptToken: `(1, T_prompt)` reference voice prompt tokens.
    ///   - promptTokenLen: `(1,)` int32 — number of valid prompt tokens.
    ///   - speakerXVector: `(1, 192)` precomputed x-vector for the reference voice.
    /// - Returns: see `EncoderOutputs`.
    func encodeForDecoder(
        speechTokens: MLXArray,
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        speakerXVector: MLXArray
    ) -> EncoderOutputs {
        let B = speechTokens.shape[0]
        let speechLen = speechTokens.shape[1]

        // Concatenate [prompt | speech] along time, then build a length-aware mask.
        let token = concatenated([promptToken, speechTokens], axis: 1).asType(.int32)
        let speechLens = MLXArray((0..<B).map { _ in Int32(speechLen) })
        let tokenLen = (promptTokenLen + speechLens).asType(.int32)   // (B,)

        let maxLen = token.shape[1]
        let arange = MLXArray(Int32(0)..<Int32(maxLen))
        let mask = (arange.expandedDimensions(axis: 0) .< tokenLen.expandedDimensions(axis: 1))
            .asType(.float32)                                          // (B, T)
        let mask3d = mask.expandedDimensions(axis: -1)                 // (B, T, 1)

        // Embed and zero-out invalid positions.
        let tokenEmb = inputEmbedding(token) * mask3d                  // (B, T, 512)

        // Encoder.
        let (encOut, encMask) = encoder(tokenEmb, xsLens: tokenLen)    // (B, 2T, 512), (B, 1, 2T)

        // Project to mel space.
        let encProj = encoderProj(encOut)                              // (B, 2T, 80)

        // Speaker affine: normalize and project.
        let xv = speakerXVector
        let xvNorm = xv / (MLX.sqrt((xv * xv).sum(axis: -1, keepDims: true)) + 1e-8)
        let spkProjected = spkEmbedAffineLayer(xvNorm)                 // (B, 80)

        return EncoderOutputs(
            encoderProjOut: encProj,
            encoderOut: encOut,
            encoderMask: encMask,
            speakerEmbedding: spkProjected
        )
    }
}
