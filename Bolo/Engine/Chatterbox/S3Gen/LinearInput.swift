// Bolo/Engine/Chatterbox/S3Gen/LinearInput.swift
import Foundation
import MLX
import MLXNN

/// Linear projection + LayerNorm + relative positional encoding.
///
/// Mirrors the Python S3Gen `LinearInput` (which wraps the ESPnet
/// `LinearNoSubsampling`):
///
///   x = linear(x)
///   x = norm(x)
///   x, pos_emb = pos_enc(x)        ← pos_enc is a stateless buffer, not a module
///
/// The safetensors keys are `linear.{weight,bias}` and `norm.{weight,bias}`.
/// `pos_enc` carries no parameters.
final class LinearInput: Module {

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    /// Non-parameter buffer. Not exposed through `@ModuleInfo` so MLX-Swift
    /// will not try to find safetensors entries for it.
    let posEnc: EspnetRelPositionalEncoding

    init(inputSize: Int, outputSize: Int) {
        self._linear.wrappedValue = Linear(inputSize, outputSize, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)
        self.posEnc = EspnetRelPositionalEncoding(dModel: outputSize)
        super.init()
    }

    /// Forward.
    ///
    /// - Parameters:
    ///   - x: `(B, T, D_in)` input.
    ///   - mask: `(B, 1, T)` mask (passed through unchanged for the caller).
    /// - Returns:
    ///   - projected and pos-scaled input `(B, T, D_out)`.
    ///   - positional embedding `(1, 2T-1, D_out)`.
    ///   - `mask` unchanged.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray
    ) -> (x: MLXArray, posEmb: MLXArray, mask: MLXArray) {
        var y = linear(x)
        y = norm(y)
        let (yScaled, posEmb) = posEnc(y)
        return (yScaled, posEmb, mask)
    }
}
