// Bolo/Engine/Chatterbox/S3Gen/Upsample1DEncoder.swift
import Foundation
import MLX
import MLXNN

/// 1D nearest-neighbor upsampling + causal convolution used by the S3Gen encoder
/// between its two conformer stacks.
///
/// Mirrors `Upsample1DEncoder` in the Python reference. The operation:
///
///   x = repeat(x, stride, axis=1)          # nearest-neighbor upsample along time
///   x = pad_left(x, stride*2)              # causal padding
///   x = conv(x)                            # kernel = stride*2 + 1
///
/// For the S3Gen UpsampleConformer this is instantiated with `stride=2`, doubling
/// the sequence length and yielding a 5-tap convolution.
final class Upsample1DEncoder: Module {

    let stride: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, stride: Int = 2) {
        self.stride = stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: stride * 2 + 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    /// - Parameters:
    ///   - x: `(B, T, C)` input.
    ///   - xLens: `(B,)` int32 lengths.
    /// - Returns:
    ///   - upsampled `(B, T * stride, C)` output.
    ///   - updated lengths `xLens * stride`.
    func callAsFunction(_ x: MLXArray, xLens: MLXArray) -> (MLXArray, MLXArray) {
        // Nearest-neighbor upsample along time. MLX `repeated` repeats elements,
        // which matches PyTorch `torch.repeat_interleave`.
        var y = repeated(x, count: stride, axis: 1)
        // Causal left-padding then convolve.
        y = padded(y, widths: [.init((0, 0)), .init((stride * 2, 0)), .init((0, 0))])
        y = conv(y)
        return (y, xLens * Int32(stride))
    }
}
