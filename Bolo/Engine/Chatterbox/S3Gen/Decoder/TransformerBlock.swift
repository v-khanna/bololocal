// Bolo/Engine/Chatterbox/S3Gen/Decoder/TransformerBlock.swift
import Foundation
import MLX
import MLXNN

// MARK: - SelfAttention1D (attn1)

/// Self-attention over a 1D sequence, matching PyTorch diffusers `attn1`
/// naming: `to_q`, `to_k`, `to_v`, `to_out.0`.
///
/// Mirrors `SelfAttention1D` in the Python `chatterbox_turbo` decoder. The
/// attention is bidirectional (no causal mask); the model achieves causality
/// via the `CausalConv1d` layers inside the resnet blocks. A `(B, T)`
/// padding mask is supported.
///
/// Weight keys (relative to the `attn1` prefix):
///   to_q.weight              (inner_dim, dim)   bias=False
///   to_k.weight              (inner_dim, dim)   bias=False
///   to_v.weight              (inner_dim, dim)   bias=False
///   to_out.0.{weight,bias}   (dim, inner_dim)
final class SelfAttention1D: Module {

    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    /// `to_out` is a PyTorch `Sequential([Linear, Dropout])`; the saved
    /// state-dict only contains `to_out.0.{w,b}`, so we model it as a single
    /// linear stored at index 0 of a `[Module]` array.
    @ModuleInfo(key: "to_out") var toOut: [Module]

    var outProj: Linear { toOut[0] as! Linear }

    init(dim: Int, numHeads: Int = 8, headDim: Int = 64) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        let innerDim = numHeads * headDim

        self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toOut.wrappedValue = [Linear(innerDim, dim, bias: true)]
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, T, C)` input.
    ///   - mask: optional `(B, T)` float mask (1 = keep, 0 = mask out).
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]

        let q = toQ(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = toK(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = toV(x).reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)

        var attn = matmul(q, k.transposed(0, 1, 3, 2)) * scale  // (B, H, T, T)
        if let mask {
            // (B, T) -> (B, 1, 1, T) and apply as additive bias (-inf where masked).
            let m = mask.expandedDimensions(axes: [1, 2])  // (B, 1, 1, T)
            attn = MLX.which(
                m .> MLXArray(Float(0)),
                attn,
                MLXArray(Float(-1e9)).asType(attn.dtype)
            )
        }
        attn = softmax(attn, axis: -1)
        let out = matmul(attn, v)  // (B, H, T, dK)
        // (B, H, T, dK) -> (B, T, H*dK)
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, T, numHeads * headDim])
        return outProj(merged)
    }
}

// MARK: - FeedForward (ff)

/// Feed-forward network for the transformer block.
///
/// Python structure: `ff.net = Sequential([GELU, Dropout, Linear])`. Sanitized
/// state-dict drops Dropout, so the on-disk keys are:
///
///   ff.net.0.proj.{weight,bias}   ← inner `GELU(Linear(dim, dim*4))`
///   ff.net.1.{weight,bias}        ← output `Linear(dim*4, dim)`
///
/// Note: index `1` not `2` — Dropout was already dropped in the safetensors.
///
/// Activation order: `Linear → GELU → Linear`. The diffusers `GELU` class
/// stores the projection as `proj` and applies `gelu` *after* the linear.
final class FeedForward: Module {

    @ModuleInfo(key: "net") var net: [Module]

    var gelu: GELUProj { net[0] as! GELUProj }
    var outProj: Linear { net[1] as! Linear }

    init(dim: Int, mult: Int = 4) {
        let innerDim = dim * mult
        self._net.wrappedValue = [
            GELUProj(dimIn: dim, dimOut: innerDim),
            Linear(innerDim, dim, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return outProj(gelu(x))
    }
}

/// `GELU(Linear)` block exposing a `proj` child for weight loading.
///
/// Mirrors the diffusers `GELU` activation that lives at `ff.net.0`. The
/// projection key on disk is `ff.net.0.proj.{w,b}`.
final class GELUProj: Module {

    @ModuleInfo(key: "proj") var proj: Linear

    init(dimIn: Int, dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXNN.gelu(proj(x))
    }
}

// MARK: - TransformerBlock

/// Bidirectional self-attention transformer block — `norm1 → attn1 → +
/// residual → norm3 → ff → + residual`. Mirrors `TransformerBlock` in the
/// Python `chatterbox_turbo` decoder.
///
/// Weight keys (relative to the block prefix):
///   norm1.{weight,bias}      LayerNorm(dim)
///   norm3.{weight,bias}      LayerNorm(dim)   ← named norm3, not norm2 (to match the
///                                              original diffusers BasicTransformerBlock,
///                                              which skipped cross-attention's norm2)
///   attn1.…                  SelfAttention1D
///   ff.…                     FeedForward
final class TransformerBlock: Module {

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: SelfAttention1D
    @ModuleInfo(key: "ff") var ff: FeedForward

    init(dim: Int, numHeads: Int = 8, headDim: Int = 64, ffMult: Int = 4) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim)
        self._attn1.wrappedValue = SelfAttention1D(dim: dim, numHeads: numHeads, headDim: headDim)
        self._ff.wrappedValue = FeedForward(dim: dim, mult: ffMult)
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, T, C)` input.
    ///   - mask: optional `(B, T)` padding mask.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + attn1(norm1(x), mask: mask)
        h = h + ff(norm3(h))
        return h
    }
}
