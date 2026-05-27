// Bolo/Engine/Chatterbox/S3Gen/UpsampleConformerEncoder.swift
import Foundation
import MLX
import MLXNN

/// The S3Gen "UpsampleConformer" encoder.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.encoder.UpsampleConformerEncoder`.
///
/// Composition (in order):
///   1. `embed`              — LinearInput(D, D): Linear + LayerNorm + relative pos enc.
///   2. `pre_lookahead_layer` — PreLookaheadLayer(D): 2-conv residual block.
///   3. `encoders[0..6)`     — six pre-norm ConformerEncoderLayer blocks.
///   4. `up_layer`           — Upsample1DEncoder(D, stride=2): doubles sequence length.
///   5. `up_embed`           — LinearInput(D, D): rebuilds pos enc for the longer seq.
///   6. `up_encoders[0..4)`  — four more ConformerEncoderLayer blocks.
///   7. `after_norm`         — LayerNorm(D) (note: eps defaults to 1e-5).
///
/// Input  : tokens already embedded as `(B, T, D=512)`, with mask zeroing.
/// Output : `(B, 2T, D=512)`, plus the broadcast mask `(B, 1, 2T)`.
final class UpsampleConformerEncoder: Module {

    let outputSize: Int
    let numBlocks: Int
    let numUpBlocks: Int

    @ModuleInfo(key: "embed") var embed: LinearInput
    @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: PreLookaheadLayer
    @ModuleInfo(key: "encoders") var encoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "up_layer") var upLayer: Upsample1DEncoder
    @ModuleInfo(key: "up_embed") var upEmbed: LinearInput
    @ModuleInfo(key: "up_encoders") var upEncoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    init(config: ChatterboxConfig.S3Gen) {
        self.outputSize = config.tokenEmbeddingDim
        self.numBlocks = config.encoderNumBlocks
        self.numUpBlocks = 4   // hard-coded in the Python reference

        let D = config.tokenEmbeddingDim
        self._embed.wrappedValue = LinearInput(inputSize: D, outputSize: D)
        self._preLookaheadLayer.wrappedValue = PreLookaheadLayer(channels: D, preLookaheadLen: 3)
        self._encoders.wrappedValue = (0..<config.encoderNumBlocks).map { _ in
            ConformerEncoderLayer(
                size: D,
                numHeads: config.encoderAttentionHeads,
                dInner: config.encoderLinearUnits
            )
        }
        self._upLayer.wrappedValue = Upsample1DEncoder(channels: D, stride: 2)
        self._upEmbed.wrappedValue = LinearInput(inputSize: D, outputSize: D)
        self._upEncoders.wrappedValue = (0..<numUpBlocks).map { _ in
            ConformerEncoderLayer(
                size: D,
                numHeads: config.encoderAttentionHeads,
                dInner: config.encoderLinearUnits
            )
        }
        self._afterNorm.wrappedValue = LayerNorm(dimensions: D, eps: 1e-5)
        super.init()
    }

    /// Forward.
    ///
    /// - Parameters:
    ///   - xs: `(B, T, D)` input embeddings (already mask-zeroed by the caller).
    ///   - xsLens: `(B,)` int32 lengths.
    /// - Returns:
    ///   - encoded features `(B, 2T, D)`.
    ///   - output mask `(B, 1, 2T)` as float.
    func callAsFunction(_ xs: MLXArray, xsLens: MLXArray) -> (MLXArray, MLXArray) {
        let B = xs.shape[0]
        let T = xs.shape[1]

        // Initial mask: (B, T)
        var mask = makeLengthMask(B: B, T: T, lens: xsLens)
            .expandedDimensions(axis: 1)                 // (B, 1, T)

        // Step 1: input projection + pos enc.
        var (h, posEmb, _) = embed(xs, mask: mask)

        // Step 2: pre-lookahead conv stack.
        h = preLookaheadLayer(h)

        // Step 3: 6 conformer blocks.
        let mask1d = mask[0..., 0, 0...]                  // (B, T)
        for layer in encoders {
            h = layer(h, mask: mask1d, posEmb: posEmb)
        }

        // Step 4: 2× upsample.
        var newLens: MLXArray
        (h, newLens) = upLayer(h, xLens: xsLens)
        let T2 = h.shape[1]

        // Step 5: rebuild mask for the doubled length and re-project.
        mask = makeLengthMask(B: B, T: T2, lens: newLens)
            .expandedDimensions(axis: 1)                 // (B, 1, 2T)
        let upOut = upEmbed(h, mask: mask)
        h = upOut.x
        posEmb = upOut.posEmb

        // Step 6: 4 more conformer blocks.
        let upMask1d = mask[0..., 0, 0...]
        for layer in upEncoders {
            h = layer(h, mask: upMask1d, posEmb: posEmb)
        }

        // Step 7: final norm.
        h = afterNorm(h)
        return (h, mask)
    }

    /// Build a `(B, T)` float mask from per-row lengths.
    /// `mask[i, t] = 1` if `t < lens[i]`, else 0.
    private func makeLengthMask(B: Int, T: Int, lens: MLXArray) -> MLXArray {
        let arange = MLXArray(Int32(0)..<Int32(T))                // (T,)
        let comparison = arange.expandedDimensions(axis: 0) .< lens.asType(.int32).expandedDimensions(axis: 1)
        return comparison.asType(.float32)
    }
}
