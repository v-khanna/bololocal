// Bolo/Engine/Chatterbox/T3/T3Attention.swift
import Foundation
import MLX
import MLXNN
import MLXFast

/// GPT-2 style multi-head self-attention for the T3 backbone.
///
/// Architecture mirrors the Swift project's ChatterboxConfig.turbo.t3:
///   - 16 heads, 64 dim each, 1024 hidden
///   - Combined QKV projection (c_attn: H → 3H), matching GPT-2 weight naming
///   - Output projection (c_proj: H → H)
///   - No GQA, no RoPE — standard MHA with learned positional embeddings
///     handled one layer up in T3.swift
///
/// Forward: (B, S, H) → (B, S, H)
///
/// Note on Python reference: the mlx-audio chatterbox implementation uses a
/// LLaMA backbone (LlamaModel from mlx_lm) rather than raw GPT-2. However
/// the Swift project's ChatterboxConfig was pinned from the live config.json
/// which identifies GPT-2 architecture. This Swift implementation follows
/// the config (standard MHA with combined c_attn projection), which is
/// equivalent to LLaMA's MHA when num_kv_heads == num_heads (no GQA).
final class T3Attention: Module {

    // MARK: - Hyperparameters

    let numHeads: Int
    let headDim: Int
    let hiddenDim: Int

    // MARK: - Layers

    /// Combined QKV projection: (H) → (3H). Key "c_attn" matches GPT-2 weight naming.
    @ModuleInfo(key: "c_attn") var qkvProj: Linear

    /// Output projection: (H) → (H). Key "c_proj" matches GPT-2 weight naming.
    @ModuleInfo(key: "c_proj") var outProj: Linear

    // MARK: - Init

    init(config: ChatterboxConfig.T3) {
        self.numHeads = config.numHeads
        self.headDim = config.headDim
        self.hiddenDim = config.hiddenDim
        self._qkvProj.wrappedValue = Linear(config.hiddenDim, 3 * config.hiddenDim, bias: true)
        self._outProj.wrappedValue = Linear(config.hiddenDim, config.hiddenDim, bias: true)
        super.init()
    }

    // MARK: - Forward

    /// Forward pass through multi-head self-attention.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape `(B, S, H)`.
    ///   - mask: Optional additive attention mask. Use `causalMask(seqLen:)` for
    ///     autoregressive generation. Shape should broadcast to `(B, h, S, S)`.
    ///   - cache: Optional KV cache (typed as `Any?` for forward-compat with the
    ///     pre-cache call sites; downcast to `T3Cache` inside). When provided,
    ///     the new K/V are appended to it and the full cached K/V are used as
    ///     the attention keys/values.
    /// - Returns: Output tensor of shape `(B, S, H)`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: Any? = nil) -> MLXArray {
        let B = x.shape[0]
        let S = x.shape[1]

        // Combined QKV projection: (B, S, H) → (B, S, 3H)
        let qkv = qkvProj(x)

        // Split into Q, K, V each of shape (B, S, H)
        let splits = split(qkv, parts: 3, axis: -1)
        let qRaw = splits[0]  // (B, S, H)
        let kRaw = splits[1]  // (B, S, H)
        let vRaw = splits[2]  // (B, S, H)

        // Reshape to (B, S, numHeads, headDim) then transpose to (B, numHeads, S, headDim)
        // This is the per-head layout expected by scaledDotProductAttention.
        let q = unflatten(qRaw, axis: -1, shape: [numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = unflatten(kRaw, axis: -1, shape: [numHeads, headDim]).transposed(0, 2, 1, 3)
        var v = unflatten(vRaw, axis: -1, shape: [numHeads, headDim]).transposed(0, 2, 1, 3)

        // If a KV cache is provided, append the new K/V and use the full history
        // for attention. The query stays single-step (or whatever length the
        // caller passed in); only K/V grow.
        if let t3Cache = cache as? T3Cache {
            (k, v) = t3Cache.update(keys: k, values: v)
        }

        // Scale: 1 / sqrt(headDim)
        let scale = 1.0 / sqrt(Float(headDim))

        // Scaled dot-product attention via MLXFast kernel.
        // Input shapes: q: (B, h, S_q, d), k/v: (B, h, S_k, d).
        // Output shape: (B, h, S_q, d).
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .none
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        // attended: (B, numHeads, S, headDim)

        // Transpose back to (B, S, numHeads, headDim) and flatten heads → (B, S, H)
        let merged = attended.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)
        // merged: (B, S, H)

        return outProj(merged)
    }

    // MARK: - Helpers

    /// Build an additive causal mask of shape `(S, S)`.
    ///
    /// Positions `[i, j]` where `j > i` get `-1e9` (effectively −∞ after softmax).
    /// Positions `[i, j]` where `j ≤ i` get `0` (no change to scores).
    ///
    /// This broadcasts naturally to `(B, h, S, S)` when added to attention scores.
    ///
    /// - Parameter seqLen: Sequence length `S`.
    /// - Returns: Additive causal mask of shape `(S, S)`.
    static func causalMask(seqLen: Int) -> MLXArray {
        return MultiHeadAttention.createAdditiveCausalMask(seqLen)
    }
}
