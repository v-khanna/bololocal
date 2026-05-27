// Bolo/Engine/Chatterbox/S3Gen/Decoder/DecoderPrimitives.swift
import Foundation
import MLX
import MLXNN

// MARK: - Conv1dPT

/// Conv1d wrapper that accepts inputs in PyTorch `(B, C, T)` layout.
///
/// Mirrors `Conv1dPT` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.decoder`. Internally
/// transposes to MLX's `(B, T, C)`, applies the underlying `Conv1d`, and
/// transposes back. The safetensors weight key for the inner conv is
/// `<prefix>.conv.weight`, hence the nested `@ModuleInfo(key: "conv")`.
final class Conv1dPT: Module {

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            bias: bias
        )
        super.init()
    }

    /// - Parameter x: `(B, C, T)` input.
    /// - Returns: `(B, C_out, T_out)` output.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // (B, C, T) -> (B, T, C)
        var y = x.transposed(0, 2, 1)
        y = conv(y)
        // (B, T_out, C_out) -> (B, C_out, T_out)
        return y.transposed(0, 2, 1)
    }
}

// MARK: - ConvTranspose1dPT

/// ConvTranspose1d wrapper that accepts inputs in PyTorch `(B, C, T)` layout.
///
/// Mirrors `ConvTranspose1dPT` in the Python reference.
final class ConvTranspose1dPT: Module {

    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        y = conv(y)
        return y.transposed(0, 2, 1)   // (B, T, C) -> (B, C, T)
    }
}

// MARK: - Sinusoidal timestep embedding

/// Sinusoidal positional embedding used for the diffusion timestep.
///
/// Mirrors `sinusoidal_pos_emb` in the Python reference (decoder.py). Pure
/// function: no trainable parameters.
///
/// - Parameters:
///   - timesteps: `(B,)` float timestep values.
///   - dim: embedding dimension (must be even).
///   - scale: multiplicative scale before applying sin/cos. Default 1000 to
///     match the Python reference.
/// - Returns: `(B, dim)` embeddings — `[sin(...), cos(...)]` concatenated.
func sinusoidalPosEmb(_ timesteps: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    precondition(dim % 2 == 0, "sinusoidalPosEmb requires even dim")
    var t = timesteps
    if t.ndim == 0 {
        t = t.expandedDimensions(axis: 0)
    }
    let halfDim = dim / 2
    let logTerm = Float(log(10000.0) / Double(halfDim - 1))
    let freqs = exp(MLXArray(0..<Int32(halfDim)).asType(.float32) * (-logTerm))
    // (B, 1) * (1, halfDim) -> (B, halfDim)
    let phase = scale * t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return concatenated([sin(phase), cos(phase)], axis: -1)
}

// MARK: - TimestepEmbedding MLP

/// Two-layer MLP for the diffusion timestep embedding.
///
/// Mirrors `TimestepEmbedding` in `chatterbox_turbo` decoder.py. Activation
/// is SiLU between the two linears (the original meanflow-disabled decoder).
///
/// Weight keys:
///   linear_1.{weight,bias}
///   linear_2.{weight,bias}
final class TimestepEmbedding: Module {

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        super.init()
    }

    /// - Parameter x: `(B, in_channels)` raw sinusoidal timestep embedding.
    /// - Returns: `(B, time_embed_dim)` timestep features.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(silu(linear1(x)))
    }
}
