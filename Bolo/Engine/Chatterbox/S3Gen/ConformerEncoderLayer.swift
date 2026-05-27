// Bolo/Engine/Chatterbox/S3Gen/ConformerEncoderLayer.swift
import Foundation
import MLX
import MLXNN

/// Position-wise feed-forward used inside the S3Gen encoder layer.
///
/// Mirrors `PositionwiseFeedForward` in the Python reference:
///
///   w_1: Linear(D, D_inner)
///   w_2: Linear(D_inner, D)
///
/// Activation is Swish/SiLU (NOT GeLU — the Python reference comments out the
/// GeLU activation that's standard in vanilla Transformers).
final class PositionwiseFeedForward: Module {

    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(dModel: Int, dInner: Int) {
        self._w1.wrappedValue = Linear(dModel, dInner, bias: true)
        self._w2.wrappedValue = Linear(dInner, dModel, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return w2(silu(w1(x)))
    }
}

/// Single pre-norm encoder layer for the S3Gen UpsampleConformer.
///
/// Despite the "Conformer" name, the Python implementation in mlx-audio
/// uses a STRIPPED-DOWN block — only the multi-head self-attention and the
/// position-wise feed-forward survive. The classic Conformer convolution
/// module and macaron FFNs are absent. Pre-norm with residuals:
///
///   x = x + self_attn(norm_mha(x))
///   x = x + feed_forward(norm_ff(x))
///
/// LayerNorm eps is 1e-12 (matches the Python reference).
final class ConformerEncoderLayer: Module {

    @ModuleInfo(key: "norm_mha") var normMha: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadedAttention
    @ModuleInfo(key: "norm_ff") var normFf: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward

    init(size: Int, numHeads: Int, dInner: Int) {
        self._normMha.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self._selfAttn.wrappedValue = RelPositionMultiHeadedAttention(
            numHeads: numHeads, dModel: size)
        self._normFf.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self._feedForward.wrappedValue = PositionwiseFeedForward(dModel: size, dInner: dInner)
        super.init()
    }

    /// Forward.
    ///
    /// - Parameters:
    ///   - x: `(B, T, D)` input.
    ///   - mask: `(B, T)` key mask.
    ///   - posEmb: `(1, 2T-1, D)` ESPnet relative positional embedding.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, posEmb: MLXArray?
    ) -> MLXArray {
        var h = x + selfAttn(normMha(x), mask: mask, posEmb: posEmb)
        h = h + feedForward(normFf(h))
        return h
    }
}
