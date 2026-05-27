// Bolo/Engine/Chatterbox/S3Gen/Vocoder/SnakeActivation.swift
import Foundation
import MLX
import MLXNN

/// Snake activation: `x + sin^2(αx) / α`, clamped to avoid division by zero.
///
/// Mirrors `Snake` in `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan`.
/// `alpha` is a learnable per-channel parameter, stored bare (no nested
/// `wrappedValue` indirection) so the on-disk safetensors key
/// `…activationsN.M.alpha` lines up with the Swift parameter path
/// `…activationsN.M.alpha` directly.
///
/// Implementation notes:
/// - `alpha` shape is `(in_features,)`. At forward time we reshape to
///   `(1, C, 1)` so it broadcasts across `(B, C, T)` PyTorch-layout activations.
/// - We do NOT support `alpha_logscale` — the chatterbox-turbo weights are
///   trained with `alpha_logscale=False` (alpha initialized to ones).
/// - The Python reference clamps `alpha_clamped = sign(α) * max(|α|, 1e-4)`,
///   then overrides positions where `|α| < 1e-9` with `min_alpha = 1e-4`.
///   We replicate that exactly to match numerics under fp16 storage.
final class Snake: Module {

    /// Learnable per-channel α. Shape `(in_features,)`.
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(inFeatures: Int, alphaInit: Float = 1.0) {
        self._alpha.wrappedValue = MLXArray.ones([inFeatures]) * alphaInit
        super.init()
    }

    /// - Parameter x: `(B, C, T)` activations.
    /// - Returns: same shape.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Reshape (C,) -> (1, C, 1) so broadcast over (B, C, T) works.
        let a = alpha.reshaped([1, alpha.shape[0], 1])

        let noDivByZero: Float = 1e-9
        let minAlpha: Float = 1e-4

        let aAbs = MLX.abs(a)
        let aSign = MLX.sign(a)
        let aClampedRaw = aSign * MLX.maximum(aAbs, minAlpha)
        let aClamped = MLX.which(aAbs .< noDivByZero, minAlpha, aClampedRaw)

        // x + (1/α) * sin(αx)^2
        let s = MLX.sin(x * a)
        return x + (s * s) / aClamped
    }
}
