// Bolo/Engine/Chatterbox/S3Gen/Vocoder/ResBlock1D.swift
import Foundation
import MLX
import MLXNN

/// Residual block with dilated 1-D convolutions and Snake activations.
///
/// Mirrors `ResBlock` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.hifigan`.
///
/// Structure per dilation step (len(dilations) == 3 in production):
///     x = x + conv2(act2(conv1(act1(x))))
///
/// Where `conv1` has the given dilation, `conv2` has dilation 1, and both
/// activations are independent learnable Snake layers.
///
/// Weight key layout (per child slot N ∈ [0, 1, 2]):
///   convs1.N.conv.{weight,bias}     ← Conv1dPT with dilation = dilations[N]
///   convs2.N.conv.{weight,bias}     ← Conv1dPT with dilation = 1
///   activations1.N.alpha            ← Snake α
///   activations2.N.alpha            ← Snake α
final class ResBlockVocoder: Module {

    @ModuleInfo(key: "convs1") var convs1: [Conv1dPT]
    @ModuleInfo(key: "convs2") var convs2: [Conv1dPT]
    @ModuleInfo(key: "activations1") var activations1: [Snake]
    @ModuleInfo(key: "activations2") var activations2: [Snake]

    init(channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        var c1: [Conv1dPT] = []
        var c2: [Conv1dPT] = []
        var a1: [Snake] = []
        var a2: [Snake] = []
        for d in dilations {
            let pad1 = ResBlockVocoder.getPadding(kernelSize: kernelSize, dilation: d)
            let pad2 = ResBlockVocoder.getPadding(kernelSize: kernelSize, dilation: 1)
            c1.append(
                Conv1dPT(
                    inputChannels: channels,
                    outputChannels: channels,
                    kernelSize: kernelSize,
                    padding: pad1,
                    dilation: d
                )
            )
            c2.append(
                Conv1dPT(
                    inputChannels: channels,
                    outputChannels: channels,
                    kernelSize: kernelSize,
                    padding: pad2,
                    dilation: 1
                )
            )
            a1.append(Snake(inFeatures: channels))
            a2.append(Snake(inFeatures: channels))
        }
        self._convs1.wrappedValue = c1
        self._convs2.wrappedValue = c2
        self._activations1.wrappedValue = a1
        self._activations2.wrappedValue = a2
        super.init()
    }

    /// Same-size padding for a dilated convolution.
    static func getPadding(kernelSize: Int, dilation: Int) -> Int {
        return Int((kernelSize * dilation - dilation) / 2)
    }

    /// - Parameter x: `(B, C, T)` PyTorch-layout activations.
    /// - Returns: same shape.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for i in 0..<convs1.count {
            var xt = activations1[i](y)
            xt = convs1[i](xt)
            xt = activations2[i](xt)
            xt = convs2[i](xt)
            y = xt + y
        }
        return y
    }
}
