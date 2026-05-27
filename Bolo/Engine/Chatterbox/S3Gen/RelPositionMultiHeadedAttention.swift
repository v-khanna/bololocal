// Bolo/Engine/Chatterbox/S3Gen/RelPositionMultiHeadedAttention.swift
import Foundation
import MLX
import MLXNN

/// Multi-head self-attention with ESPnet-style relative positional encoding.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.encoder.RelPositionMultiHeadedAttention`.
///
/// The attention score is computed as
///
///     scores = ((Q + u) · Kᵀ + (Q + v) · Pᵀ) / √dₖ
///
/// where `u`, `v` are learned per-head bias vectors (`pos_bias_u`, `pos_bias_v`)
/// and `P` is the linear projection of the ESPnet relative positional embedding
/// (`linear_pos`). The `_rel_shift` step converts a (T, 2T-1) score matrix into
/// the (T, T) form by carving out the diagonal band of valid relative offsets.
///
/// Weight keys (relative to the layer prefix):
///   linear_q.{weight,bias}    Linear(D, D, bias=True)
///   linear_k.{weight,bias}    Linear(D, D, bias=True)   ← key_bias defaults to True
///   linear_v.{weight,bias}    Linear(D, D, bias=True)
///   linear_out.{weight,bias}  Linear(D, D, bias=True)
///   linear_pos.weight         Linear(D, D, bias=False)
///   pos_bias_u                (numHeads, headDim)
///   pos_bias_v                (numHeads, headDim)
final class RelPositionMultiHeadedAttention: Module {

    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    /// Per-head learnable position bias `u` (content–position).
    /// Shape `(numHeads, headDim)`. Reshaped to `(1, numHeads, 1, headDim)` for broadcasting.
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    /// Per-head learnable position bias `v` (position–content). Same shape as `u`.
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(numHeads: Int, dModel: Int) {
        precondition(dModel % numHeads == 0, "dModel must be divisible by numHeads")
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self._linearQ.wrappedValue = Linear(dModel, dModel, bias: true)
        self._linearK.wrappedValue = Linear(dModel, dModel, bias: true)
        self._linearV.wrappedValue = Linear(dModel, dModel, bias: true)
        self._linearOut.wrappedValue = Linear(dModel, dModel, bias: true)
        self._linearPos.wrappedValue = Linear(dModel, dModel, bias: false)
        self._posBiasU.wrappedValue = MLX.zeros([numHeads, headDim])
        self._posBiasV.wrappedValue = MLX.zeros([numHeads, headDim])
        super.init()
    }

    /// Convert attention scores indexed by (i, relative-offset j) into the standard
    /// (i, k) layout. See ESPnet / Transformer-XL Appendix B.
    ///
    /// Input shape:  `(B, H, T, 2T-1)`
    /// Output shape: `(B, H, T, T)`
    private func relShift(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let H = x.shape[1]
        let T1 = x.shape[2]
        let T2 = x.shape[3]               // = 2T1 - 1

        // Pad with zeros on the left along the last axis: (B, H, T1, T2+1)
        let zeroPad = MLX.zeros([B, H, T1, 1], dtype: x.dtype)
        var y = concatenated([zeroPad, x], axis: -1)

        // Reshape: (B, H, T2+1, T1)
        y = y.reshaped([B, H, T2 + 1, T1])

        // Drop first row of the reshaped (T2+1, T1) → (T2, T1), then back to (T1, T2)
        y = y[0..., 0..., 1..., 0...]
        y = y.reshaped([B, H, T1, T2])

        // Keep only the first T1 columns (positions 0..T1-1 = valid relative offsets)
        return y[0..., 0..., 0..., 0..<(T2 / 2 + 1)]
    }

    /// Forward.
    ///
    /// - Parameters:
    ///   - x: `(B, T, D)` input.
    ///   - mask: `(B, T)` boolean/float mask. Positions where mask > 0 are kept;
    ///     other positions get -inf scores. Pass `nil` to disable masking.
    ///   - posEmb: `(1, 2T-1, D)` relative positional embedding from
    ///     `EspnetRelPositionalEncoding`. Pass `nil` to fall back to absolute attention.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, posEmb: MLXArray?
    ) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let D = x.shape[2]

        // Q, K, V projections, reshape to (B, T, H, dK)
        let q = linearQ(x).reshaped([B, T, numHeads, headDim])
        let k = linearK(x).reshaped([B, T, numHeads, headDim])
            .transposed(0, 2, 1, 3)                      // (B, H, T, dK)
        let v = linearV(x).reshaped([B, T, numHeads, headDim])
            .transposed(0, 2, 1, 3)                      // (B, H, T, dK)

        // Content–content: (q + u) · kᵀ
        let qWithU = (q + posBiasU).transposed(0, 2, 1, 3)   // (B, H, T, dK)
        let matrixAC = matmul(qWithU, k.transposed(0, 1, 3, 2))  // (B, H, T, T)

        // Position score
        var scores: MLXArray
        if let posEmb {
            let Tpos = posEmb.shape[1]                       // 2T-1
            let p = linearPos(posEmb)                        // (1, 2T-1, D)
                .reshaped([1, Tpos, numHeads, headDim])
                .transposed(0, 2, 1, 3)                      // (1, H, 2T-1, dK)
            let qWithV = (q + posBiasV).transposed(0, 2, 1, 3)   // (B, H, T, dK)
            var matrixBD = matmul(qWithV, p.transposed(0, 1, 3, 2))  // (B, H, T, 2T-1)
            if matrixAC.shape != matrixBD.shape {
                matrixBD = relShift(matrixBD)               // (B, H, T, T)
            }
            scores = (matrixAC + matrixBD) * scale
        } else {
            scores = matrixAC * scale
        }

        // Apply key mask. We accept (B, T) and broadcast to (B, 1, 1, T).
        if let mask {
            let m: MLXArray
            if mask.ndim == 2 {
                m = mask.expandedDimensions(axes: [1, 2])    // (B, 1, 1, T)
            } else {
                // (B, T, T) → (B, 1, T, T)
                m = mask.expandedDimensions(axis: 1)
            }
            // mask > 0 ⇒ keep, else fill with -inf
            scores = MLX.which(
                m .> MLXArray(Float(0)),
                scores,
                MLXArray(-Float.infinity).asType(scores.dtype)
            )
        }

        var attn = softmax(scores, axis: -1)
        // softmax over an all--inf row produces NaNs; zero them out.
        let isNaN = attn .!= attn
        attn = MLX.which(isNaN, MLXArray(Float(0)).asType(attn.dtype), attn)

        let out = matmul(attn, v)                            // (B, H, T, dK)
        // (B, H, T, dK) → (B, T, H, dK) → (B, T, D)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, D])
        return linearOut(merged)
    }
}
