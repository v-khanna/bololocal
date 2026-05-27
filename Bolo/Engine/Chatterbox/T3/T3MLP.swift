// Bolo/Engine/Chatterbox/T3/T3MLP.swift
import Foundation
import MLX
import MLXNN

/// GPT-2 feed-forward block: Linear → gelu_new → Linear.
///
/// Intermediate dim = 4 × hidden (GPT-2 convention).
/// activation = gelu_new (the tanh-approximated GELU variant GPT-2 uses).
/// MLXNN exposes this as `geluApproximate(_:)`.
///
/// Forward: (B, S, H) → (B, S, H)
final class T3MLP: Module, UnaryLayer {

    @ModuleInfo(key: "c_fc") var fc: Linear      // (H) → (4H)
    @ModuleInfo(key: "c_proj") var proj: Linear  // (4H) → (H)

    init(config: ChatterboxConfig.T3) {
        let intermediate = 4 * config.hiddenDim
        self._fc.wrappedValue = Linear(config.hiddenDim, intermediate, bias: true)
        self._proj.wrappedValue = Linear(intermediate, config.hiddenDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = fc(x)
        // gelu_new = tanh-approximated GELU (the GPT-2 variant).
        let activated = geluApproximate(h)
        return proj(activated)
    }
}
