// Bolo/Engine/Chatterbox/T3/T3Block.swift
import Foundation
import MLX
import MLXNN

/// One GPT-2 transformer block with pre-norm layout:
///   ln_1(x) → attn → +x  →  ln_2(.) → mlp → +.
///
/// Forward: (B, S, H) → (B, S, H)
final class T3Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attn: T3Attention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: T3MLP

    init(config: ChatterboxConfig.T3) {
        self._ln1.wrappedValue = LayerNorm(dimensions: config.hiddenDim, eps: Float(config.layerNormEps))
        self._attn.wrappedValue = T3Attention(config: config)
        self._ln2.wrappedValue = LayerNorm(dimensions: config.hiddenDim, eps: Float(config.layerNormEps))
        self._mlp.wrappedValue = T3MLP(config: config)
        super.init()
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: Input tensor of shape `(B, S, H)`.
    ///   - mask: Optional additive causal mask. Use `T3Attention.causalMask(seqLen:)`.
    ///   - cache: Reserved for KV cache (Task 11). Pass `nil` for now.
    /// - Returns: Output tensor of shape `(B, S, H)`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: Any? = nil) -> MLXArray {
        var h = x + attn(ln1(x), mask: mask, cache: cache)
        h = h + mlp(ln2(h))
        return h
    }
}
