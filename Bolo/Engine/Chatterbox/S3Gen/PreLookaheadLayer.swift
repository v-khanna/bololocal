// Bolo/Engine/Chatterbox/S3Gen/PreLookaheadLayer.swift
import Foundation
import MLX
import MLXNN

/// Pre-lookahead convolution block used by the S3Gen encoder before its
/// Conformer stack.
///
/// Mirrors `PreLookaheadLayer` in the Python reference. Composition:
///
///   y = leaky_relu(conv1(pad_right(x, look)))       conv1 kernel=look+1
///   y = conv2(pad_left(y, 2))                       conv2 kernel=3, causal pad
///   out = y + x                                      residual
///
/// Both convs are `Conv1d(D, D, …)`. The first uses right-padding (look-ahead),
/// the second uses left-padding (causal). MLX-Swift `Conv1d` operates on
/// `(B, T, C)` directly — no transposes required.
///
/// Weight keys (relative to the layer prefix):
///   conv1.{weight,bias}    shape (D, look+1, D)   default look = 3
///   conv2.{weight,bias}    shape (D, 3, D)
final class PreLookaheadLayer: Module {

    let preLookaheadLen: Int

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(channels: Int, preLookaheadLen: Int = 3) {
        self.preLookaheadLen = preLookaheadLen
        // Conv1d with NO internal padding; we apply the asymmetric padding manually.
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: preLookaheadLen + 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    /// - Parameter x: `(B, T, C)` input.
    /// - Returns: `(B, T, C)` output.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Right-pad with `preLookaheadLen` zeros along time, then conv1.
        // padded(_, widths:) pads each axis using a list of (before, after) pairs.
        var y = padded(x, widths: [.init((0, 0)), .init((0, preLookaheadLen)), .init((0, 0))])
        y = conv1(y)
        y = leakyRelu(y)

        // Left-pad with 2 zeros (causal) along time, then conv2.
        y = padded(y, widths: [.init((0, 0)), .init((2, 0)), .init((0, 0))])
        y = conv2(y)

        return y + x
    }
}
