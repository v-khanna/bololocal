// Bolo/Engine/Chatterbox/S3Gen/Decoder/ResnetBlock1D.swift
import Foundation
import MLX
import MLXNN

// MARK: - CausalConv1d

/// 1D causal convolution. Left-pads the input by `(kernel_size - 1) * dilation`
/// zeros so that output position t depends only on input positions ≤ t.
///
/// Mirrors `CausalConv1d` in the Python `chatterbox_turbo` decoder. Operates
/// on `(B, C, T)` tensors.
///
/// Weight key (relative to the layer prefix):
///   conv.conv.weight    shape `(O, K, I)`     ← already in MLX layout in the
///                                              pre-converted safetensors.
final class CausalConv1d: Module {

    let kernelSize: Int
    let dilation: Int
    let causalPadding: Int

    @ModuleInfo(key: "conv") var conv: Conv1dPT

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        precondition(stride == 1, "CausalConv1d only supports stride=1")
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.causalPadding = (kernelSize - 1) * dilation
        self._conv.wrappedValue = Conv1dPT(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            bias: bias
        )
        super.init()
    }

    /// - Parameter x: `(B, C, T)` input.
    /// - Returns: `(B, C_out, T)` output (length unchanged).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pad on left of time axis only.
        let padded = MLX.padded(
            x,
            widths: [.init((0, 0)), .init((0, 0)), .init((causalPadding, 0))]
        )
        return conv(padded)
    }
}

// MARK: - Block1D / CausalBlock1D

/// Non-causal block: Conv1d (k=3, pad=1) + GroupNorm + Mish, all with a mask.
///
/// Used when `causal=False`. The Chatterbox-Turbo decoder is always causal,
/// so this is here for completeness only.
final class Block1D: Module {

    @ModuleInfo(key: "conv") var conv: Conv1dPT
    @ModuleInfo(key: "norm") var norm: GroupNorm

    init(dim: Int, dimOut: Int, groups: Int = 8) {
        self._conv.wrappedValue = Conv1dPT(
            inputChannels: dim, outputChannels: dimOut,
            kernelSize: 3, padding: 1
        )
        // PyTorch's nn.GroupNorm normalizes within (B, group, group_size * spatial).
        // MLX's `pytorchCompatible: true` matches that behaviour.
        self._norm.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: dimOut, pytorchCompatible: true
        )
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, C, T)` input.
    ///   - mask: `(B, 1, T)` float mask.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var y = conv(x * mask)              // (B, C_out, T)
        // GroupNorm expects channels-last in MLX: (B, T, C_out)
        y = y.transposed(0, 2, 1)
        y = norm(y)
        y = y.transposed(0, 2, 1)           // back to (B, C_out, T)
        y = MLXNN.mish(y)
        return y * mask
    }
}

/// Causal version of `Block1D` — `CausalConv1d` + `LayerNorm` + `Mish`, all
/// with a mask multiplier.
///
/// In PyTorch this is a `Sequential(CausalConv, LayerNorm, Mish)` whose saved
/// state-dict drops the activation, leaving `block.0.{...}` (the
/// `CausalConv1d`) and `block.1.{...}` (the `LayerNorm`). MLX-Swift's
/// parameter walker turns `[Module]` properties into `0`/`1` numeric segments,
/// so we expose `block` as a heterogeneous `[Module]` array — the two element
/// types differ but both are `Module`.
final class CausalBlock1D: Module {

    @ModuleInfo(key: "block") var block: [Module]

    /// Indexed accessor for typed forward dispatch.
    var causalConv: CausalConv1d { block[0] as! CausalConv1d }
    var layerNorm: LayerNorm { block[1] as! LayerNorm }

    init(dim: Int, dimOut: Int) {
        let conv = CausalConv1d(
            inputChannels: dim, outputChannels: dimOut, kernelSize: 3
        )
        let norm = LayerNorm(dimensions: dimOut)
        self._block.wrappedValue = [conv, norm]
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, C, T)` input.
    ///   - mask: `(B, 1, T)` mask.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var y = causalConv(x * mask)                  // (B, C_out, T)
        // LayerNorm normalizes over the LAST dim (channels). Transpose to
        // channels-last, apply, transpose back.
        y = y.transposed(0, 2, 1)
        y = layerNorm(y)
        y = y.transposed(0, 2, 1)
        y = MLXNN.mish(y)
        return y * mask
    }
}

// MARK: - ResnetBlock1D

/// 1D ResNet block with timestep conditioning.
///
/// Mirrors `ResnetBlock1D` in `chatterbox_turbo.models.s3gen.decoder`.
/// Composition:
///
///   h = block1(x, mask)
///   h = h + linear(mish(time_emb))[..., None]   ← time conditioning
///   h = block2(h, mask)
///   out = h + res_conv(x * mask)
///
/// The "mlp" container in PyTorch is `Sequential(Mish(), Linear(...))`, so the
/// linear lives at index 1. We expose it as `mlp.0` in Swift (sanitization
/// drops the Mish module before saving, and weight keys become `mlp.0.{w,b}`).
///
/// Wait — the safetensors keys we observed are `mlp.0.{w,b}` not `mlp.1.{w,b}`,
/// so the saved PyTorch state already strips the Mish module. Hence `mlp.0`
/// in our Swift module.
final class ResnetBlock1D: Module {

    let causal: Bool

    /// Linear that projects timestep embedding to channel dimension.
    /// Weight key `mlp.0.{weight,bias}`.
    @ModuleInfo(key: "mlp.0") var mlp0: Linear
    @ModuleInfo(key: "block1") var block1: Module
    @ModuleInfo(key: "block2") var block2: Module
    @ModuleInfo(key: "res_conv") var resConv: Conv1dPT

    init(dim: Int, dimOut: Int, timeEmbDim: Int, causal: Bool = true, groups: Int = 8) {
        self.causal = causal
        self._mlp0.wrappedValue = Linear(timeEmbDim, dimOut, bias: true)
        if causal {
            self._block1.wrappedValue = CausalBlock1D(dim: dim, dimOut: dimOut)
            self._block2.wrappedValue = CausalBlock1D(dim: dimOut, dimOut: dimOut)
        } else {
            self._block1.wrappedValue = Block1D(dim: dim, dimOut: dimOut, groups: groups)
            self._block2.wrappedValue = Block1D(dim: dimOut, dimOut: dimOut, groups: groups)
        }
        self._resConv.wrappedValue = Conv1dPT(
            inputChannels: dim, outputChannels: dimOut, kernelSize: 1
        )
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, C, T)`.
    ///   - mask: `(B, 1, T)`.
    ///   - timeEmb: `(B, time_embed_dim)`.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, timeEmb: MLXArray
    ) -> MLXArray {
        var h = applyBlock(block1, x, mask)
        // Mish then Linear (PyTorch order), then broadcast over time.
        let t = mlp0(MLXNN.mish(timeEmb))            // (B, C_out)
        h = h + t.expandedDimensions(axis: -1)       // (B, C_out, 1) broadcast over T
        h = applyBlock(block2, h, mask)
        return h + resConv(x * mask)
    }

    private func applyBlock(_ block: Module, _ x: MLXArray, _ mask: MLXArray) -> MLXArray {
        if let causal = block as? CausalBlock1D {
            return causal(x, mask: mask)
        }
        if let regular = block as? Block1D {
            return regular(x, mask: mask)
        }
        fatalError("ResnetBlock1D: unexpected block type \(type(of: block))")
    }
}

// MARK: - Downsample / Upsample

/// 1D downsampling: Conv1d with stride 2. Halves the time dimension.
final class Downsample1D: Module {

    @ModuleInfo(key: "conv") var conv: Conv1dPT

    init(channels: Int) {
        self._conv.wrappedValue = Conv1dPT(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3, stride: 2, padding: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

/// 1D upsampling via transposed convolution. Doubles the time dimension.
final class Upsample1D: Module {

    @ModuleInfo(key: "conv") var conv: ConvTranspose1dPT

    init(channels: Int) {
        self._conv.wrappedValue = ConvTranspose1dPT(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 4, stride: 2, padding: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}
