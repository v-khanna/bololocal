// Bolo/Engine/Chatterbox/S3Gen/Decoder/UNetBlocks.swift
import Foundation
import MLX
import MLXNN

// MARK: - DownBlock

/// A "down" block in the U-Net: ResnetBlock1D + N TransformerBlocks +
/// downsample. In Chatterbox-Turbo, `channels = [256]`, so there is exactly
/// one DownBlock with `is_last = true`, meaning `downsample` is a length-
/// preserving `CausalConv1d`, not a stride-2 `Downsample1D`.
///
/// Weight keys (relative to the block prefix):
///   resnet.…
///   transformer_blocks.{i}.…
///   downsample.…
final class DownBlock: Module {

    let isLast: Bool

    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    @ModuleInfo(key: "downsample") var downsample: Module

    init(
        inputChannel: Int,
        outputChannel: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int,
        isLast: Bool
    ) {
        self.isLast = isLast
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: inputChannel, dimOut: outputChannel,
            timeEmbDim: timeEmbedDim, causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            TransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
        }
        if isLast {
            if causal {
                self._downsample.wrappedValue = CausalConv1d(
                    inputChannels: outputChannel, outputChannels: outputChannel,
                    kernelSize: 3
                )
            } else {
                self._downsample.wrappedValue = Conv1dPT(
                    inputChannels: outputChannel, outputChannels: outputChannel,
                    kernelSize: 3, padding: 1
                )
            }
        } else {
            self._downsample.wrappedValue = Downsample1D(channels: outputChannel)
        }
        super.init()
    }

    func applyDownsample(_ x: MLXArray) -> MLXArray {
        if let c = downsample as? CausalConv1d { return c(x) }
        if let c = downsample as? Conv1dPT { return c(x) }
        if let d = downsample as? Downsample1D { return d(x) }
        fatalError("DownBlock: unexpected downsample type \(type(of: downsample))")
    }
}

// MARK: - MidBlock

/// A "mid" block: ResnetBlock1D + N TransformerBlocks. No spatial sub/upsample.
final class MidBlock: Module {

    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]

    init(
        channels: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int
    ) {
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: channels, dimOut: channels,
            timeEmbDim: timeEmbedDim, causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            TransformerBlock(dim: channels, numHeads: numHeads, headDim: attentionHeadDim)
        }
        super.init()
    }
}

// MARK: - UpBlock

/// An "up" block: ResnetBlock1D + N TransformerBlocks + upsample. Skip
/// connection (the matching down-path hidden state) is concatenated to `x`
/// along the channel axis before this block's resnet runs.
final class UpBlock: Module {

    let isLast: Bool

    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [TransformerBlock]
    @ModuleInfo(key: "upsample") var upsample: Module

    init(
        inputChannel: Int,
        outputChannel: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int,
        isLast: Bool
    ) {
        self.isLast = isLast
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: inputChannel, dimOut: outputChannel,
            timeEmbDim: timeEmbedDim, causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            TransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
        }
        if isLast {
            if causal {
                self._upsample.wrappedValue = CausalConv1d(
                    inputChannels: outputChannel, outputChannels: outputChannel,
                    kernelSize: 3
                )
            } else {
                self._upsample.wrappedValue = Conv1dPT(
                    inputChannels: outputChannel, outputChannels: outputChannel,
                    kernelSize: 3, padding: 1
                )
            }
        } else {
            self._upsample.wrappedValue = Upsample1D(channels: outputChannel)
        }
        super.init()
    }

    func applyUpsample(_ x: MLXArray) -> MLXArray {
        if let c = upsample as? CausalConv1d { return c(x) }
        if let c = upsample as? Conv1dPT { return c(x) }
        if let u = upsample as? Upsample1D { return u(x) }
        fatalError("UpBlock: unexpected upsample type \(type(of: upsample))")
    }
}
