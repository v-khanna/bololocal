// Bolo/Engine/Chatterbox/S3Gen/Decoder/ConditionalDecoder.swift
import Foundation
import MLX
import MLXNN

/// 1D U-Net velocity-field estimator for the Chatterbox-Turbo flow-matching
/// decoder.
///
/// Mirrors `ConditionalDecoder` in
/// `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.decoder`.
///
/// Architecture (matching the on-disk safetensors layout
/// `s3gen.decoder.estimator.*`, 911 weight keys total):
///
///   time_mlp                 — `Sin(t) → Linear(320, 1024) → SiLU → Linear(1024, 1024)`
///   down_blocks[0]           — ResnetBlock1D(in=320, out=256) + 4 × TransformerBlock(256)
///                             + CausalConv1d(256, 256, 3) downsample (length-preserving)
///   mid_blocks[0..12]        — each: ResnetBlock1D(256, 256) + 4 × TransformerBlock(256)
///   up_blocks[0]             — ResnetBlock1D(in=512, out=256) (input doubled by skip)
///                             + 4 × TransformerBlock(256)
///                             + CausalConv1d(256, 256, 3) upsample
///   final_block              — CausalBlock1D(256, 256)
///   final_proj               — Conv1d(256, 80, 1)
///
/// Inputs:
///   - x:    (B, 80, T) noisy latent at the current ODE step
///   - mask: (B, 1, T)  float mask
///   - mu:   (B, 80, T) encoder mel conditioning
///   - t:    (B,)       timestep scalars in [0, 1]
///   - spks: (B, 80)    speaker embedding (concatenated along channel axis)
///   - cond: (B, 80, T) extra conditioning
///
/// Output: (B, 80, T) velocity field.
final class ConditionalDecoder: Module {

    let inChannels: Int
    let outChannels: Int
    let causal: Bool

    @ModuleInfo(key: "time_mlp") var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock]
    @ModuleInfo(key: "mid_blocks") var midBlocks: [MidBlock]
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
    @ModuleInfo(key: "final_block") var finalBlock: Module      // CausalBlock1D or Block1D
    @ModuleInfo(key: "final_proj") var finalProj: Conv1dPT

    init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        causal: Bool = true,
        channels: [Int] = [256],
        attentionHeadDim: Int = 64,
        nBlocks: Int = 4,
        numMidBlocks: Int = 12,
        numHeads: Int = 8
    ) {
        precondition(!channels.isEmpty, "channels must be non-empty")
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.causal = causal

        let timeEmbedDim = channels[0] * 4

        self._timeMlp.wrappedValue = TimestepEmbedding(
            inChannels: inChannels, timeEmbedDim: timeEmbedDim
        )

        // Down blocks.
        var downs: [DownBlock] = []
        var outputChannel = inChannels
        for (i, ch) in channels.enumerated() {
            let inputChannel = outputChannel
            outputChannel = ch
            let isLast = i == channels.count - 1
            downs.append(DownBlock(
                inputChannel: inputChannel,
                outputChannel: outputChannel,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim,
                isLast: isLast
            ))
        }
        self._downBlocks.wrappedValue = downs

        // Mid blocks.
        self._midBlocks.wrappedValue = (0..<numMidBlocks).map { _ in
            MidBlock(
                channels: channels.last!,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim
            )
        }

        // Up blocks. channels_up = reversed(channels) + [channels[0]]
        var channelsUp = Array(channels.reversed())
        channelsUp.append(channels[0])
        var ups: [UpBlock] = []
        for i in 0..<(channelsUp.count - 1) {
            let inputChannel = channelsUp[i] * 2  // skip connection doubles channels
            let outputCh = channelsUp[i + 1]
            let isLast = i == channelsUp.count - 2
            ups.append(UpBlock(
                inputChannel: inputChannel,
                outputChannel: outputCh,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim,
                isLast: isLast
            ))
        }
        self._upBlocks.wrappedValue = ups

        // Final layers.
        let finalCh = channelsUp.last!
        if causal {
            self._finalBlock.wrappedValue = CausalBlock1D(dim: finalCh, dimOut: finalCh)
        } else {
            self._finalBlock.wrappedValue = Block1D(dim: finalCh, dimOut: finalCh)
        }
        self._finalProj.wrappedValue = Conv1dPT(
            inputChannels: finalCh, outputChannels: outChannels, kernelSize: 1
        )
        super.init()
    }

    /// Forward pass producing the velocity field.
    ///
    /// - Parameters:
    ///   - x: `(B, 80, T)` noisy latent.
    ///   - mask: `(B, 1, T)` float mask.
    ///   - mu: `(B, 80, T)` encoder mel.
    ///   - t: `(B,)` timesteps.
    ///   - spks: optional `(B, 80)` speaker embedding.
    ///   - cond: optional `(B, 80, T)` conditioning.
    /// - Returns: `(B, 80, T)` velocity.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        // Time embedding.
        var tEmb = sinusoidalPosEmb(t, dim: inChannels)
        tEmb = timeMlp(tEmb)                                          // (B, time_embed_dim)

        // Concatenate inputs along channel axis: [x, mu, spks_exp, cond] = (B, 320, T).
        var hx = concatenated([x, mu], axis: 1)
        if let spks {
            let T = x.shape[2]
            let spksExp = MLX.broadcast(
                spks.expandedDimensions(axis: -1),
                to: [spks.shape[0], spks.shape[1], T]
            )
            hx = concatenated([hx, spksExp], axis: 1)
        }
        if let cond {
            hx = concatenated([hx, cond], axis: 1)
        }

        // Down path.
        var hiddens: [MLXArray] = []
        var masks: [MLXArray] = [mask]
        for downBlock in downBlocks {
            let maskDown = masks.last!
            hx = downBlock.resnet(hx, mask: maskDown, timeEmb: tEmb)

            // Transformer: (B, C, T) -> (B, T, C)
            hx = hx.transposed(0, 2, 1)
            let maskT = maskDown[0..., 0, 0...]      // (B, T)
            for block in downBlock.transformerBlocks {
                hx = block(hx, mask: maskT)
            }
            hx = hx.transposed(0, 2, 1)              // (B, T, C) -> (B, C, T)

            hiddens.append(hx)
            hx = downBlock.applyDownsample(hx * maskDown)
            // Downsample mask along time axis by stride 2.
            // For the last block, downsample is length-preserving — but the
            // mask stride below is unconditional in the Python reference, so
            // the mask becomes shorter than `x`. The up path uses the saved
            // skip's length, and the mid blocks use this downsampled mask.
            masks.append(maskDown[0..., 0..., .stride(by: 2)])
        }
        // Pop the final appended mask — it has no corresponding block.
        masks.removeLast()
        var maskMid = masks.last!

        // Mid path.
        for midBlock in midBlocks {
            hx = midBlock.resnet(hx, mask: maskMid, timeEmb: tEmb)
            hx = hx.transposed(0, 2, 1)
            let maskT = maskMid[0..., 0, 0...]
            for block in midBlock.transformerBlocks {
                hx = block(hx, mask: maskT)
            }
            hx = hx.transposed(0, 2, 1)
        }

        // Up path. `mask_up` is the saved mask from `masks`, popped LIFO.
        var maskUp: MLXArray = maskMid
        for upBlock in upBlocks {
            maskUp = masks.removeLast()
            let skip = hiddens.removeLast()

            // Truncate x to the saved skip length before concatenation.
            hx = hx[0..., 0..., 0..<skip.shape[2]]
            hx = concatenated([hx, skip], axis: 1)
            hx = upBlock.resnet(hx, mask: maskUp, timeEmb: tEmb)

            hx = hx.transposed(0, 2, 1)
            let maskT = maskUp[0..., 0, 0...]
            for block in upBlock.transformerBlocks {
                hx = block(hx, mask: maskT)
            }
            hx = hx.transposed(0, 2, 1)

            hx = upBlock.applyUpsample(hx * maskUp)
        }

        // Final block + projection.
        if let fb = finalBlock as? CausalBlock1D {
            hx = fb(hx, mask: maskUp)
        } else if let fb = finalBlock as? Block1D {
            hx = fb(hx, mask: maskUp)
        } else {
            fatalError("ConditionalDecoder: unexpected final_block type \(type(of: finalBlock))")
        }
        hx = finalProj(hx * maskUp)
        return hx * mask
    }
}
