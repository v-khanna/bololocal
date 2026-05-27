// Bolo/Engine/Chatterbox/S3Gen/Vocoder/VocoderWeightMapper.swift
import Foundation
import MLX
import MLXNN

/// Maps Chatterbox-Turbo safetensors weights for the `s3gen.mel2wav.*`
/// sub-tree into a Swift `HiFTGenerator` instance.
///
/// All Conv1d weights are already stored in MLX layout `(O, K, I)` because
/// the source repo `mlx-community/chatterbox-turbo-fp16` is pre-converted —
/// no transposition needed.
///
/// Key prefix:  `s3gen.mel2wav.<path>` ↔ Swift parameter path `<path>`
/// on the `HiFTGenerator`.
///
/// 246 source keys. All map directly to Swift parameter paths:
///   f0_predictor.classifier.{weight,bias}
///   f0_predictor.condnet.{0..4}.conv.{weight,bias}
///   m_source.l_linear.{weight,bias}
///   conv_pre.conv.{weight,bias}
///   conv_post.conv.{weight,bias}
///   ups.{0..2}.conv.{weight,bias}
///   source_downs.{0..2}.conv.{weight,bias}
///   source_resblocks.{0..2}.{convs1,convs2}.{0..2}.conv.{weight,bias}
///   source_resblocks.{0..2}.{activations1,activations2}.{0..2}.alpha
///   resblocks.{0..8}.{convs1,convs2}.{0..2}.conv.{weight,bias}
///   resblocks.{0..8}.{activations1,activations2}.{0..2}.alpha
enum VocoderWeightMapper {

    struct Report {
        /// Number of `s3gen.mel2wav.*` keys observed in the source.
        let sourceKeyCount: Int
        /// Number of source keys actually applied to the Swift vocoder.
        let appliedKeyCount: Int
        /// Source keys (after stripping prefix) that have no destination —
        /// indicates a mapping bug.
        let unmappedSourceKeys: [String]
        /// Swift parameter paths that did not receive a weight — still
        /// randomly initialized; indicates a mapping bug.
        let unfilledSwiftKeys: [String]
    }

    /// Source-key prefix in the safetensors.
    static let sourcePrefix = "s3gen.mel2wav."

    /// Apply weights from a full safetensors dictionary onto a Swift vocoder.
    @discardableResult
    static func apply(
        weights: [String: MLXArray],
        to vocoder: HiFTGenerator
    ) -> Report {
        var renamed: [String: MLXArray] = [:]
        var sourceKeyCount = 0

        for (key, value) in weights where key.hasPrefix(sourcePrefix) {
            sourceKeyCount += 1
            let stripped = String(key.dropFirst(sourcePrefix.count))
            if stripped.hasSuffix(".num_batches_tracked") { continue }
            renamed[stripped] = value
        }

        let expected = Set(vocoder.parameters().flattened().map { $0.0 })
        let provided = Set(renamed.keys)

        let unmappedSourceKeys = Array(provided.subtracting(expected)).sorted()
        let unfilledSwiftKeys = Array(expected.subtracting(provided)).sorted()

        let params = ModuleParameters.unflattened(renamed)
        vocoder.update(parameters: params)
        eval(vocoder)

        return Report(
            sourceKeyCount: sourceKeyCount,
            appliedKeyCount: renamed.count,
            unmappedSourceKeys: unmappedSourceKeys,
            unfilledSwiftKeys: unfilledSwiftKeys
        )
    }
}
