// Bolo/Engine/Chatterbox/S3Gen/Vocoder/F0Predictor.swift
import Foundation
import MLX
import MLXNN

/// ELU activation: `x if x > 0, alpha * (exp(x) - 1) otherwise`.
/// Used inline between F0Predictor condnet convolutions.
@inline(__always)
func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    return MLX.which(x .> 0, x, alpha * (MLX.exp(x) - 1))
}

/// F0 predictor from mel-spectrogram.
///
/// Mirrors `F0Predictor` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan`. Five
/// Conv1d-with-ELU layers (`condnet`) followed by a Linear classifier.
/// Output is post-`abs()` ensuring non-negative F0 in Hz.
///
/// Weight key layout:
///   condnet.N.conv.{weight,bias}    ← Conv1dPT, N ∈ [0, 4]
///   classifier.{weight,bias}         ← Linear
final class F0Predictor: Module {

    @ModuleInfo(key: "condnet") var condnet: [Conv1dPT]
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(inChannels: Int = 80, hiddenChannels: Int = 512, numLayers: Int = 5) {
        var layers: [Conv1dPT] = []
        for i in 0..<numLayers {
            let inCh = (i == 0) ? inChannels : hiddenChannels
            layers.append(
                Conv1dPT(
                    inputChannels: inCh,
                    outputChannels: hiddenChannels,
                    kernelSize: 3,
                    padding: 1
                )
            )
        }
        self._condnet.wrappedValue = layers
        self._classifier.wrappedValue = Linear(hiddenChannels, 1, bias: true)
        super.init()
    }

    /// - Parameter mel: `(B, 80, T)` mel features.
    /// - Returns: `(B, T)` non-negative predicted F0 (Hz).
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel
        for conv in condnet {
            x = conv(x)
            x = elu(x)
        }
        // (B, C, T) -> (B, T, C) for linear
        x = x.transposed(0, 2, 1)
        var f0 = classifier(x)                    // (B, T, 1)
        f0 = f0[0..., 0..., 0]                    // (B, T)
        return MLX.abs(f0)
    }
}
