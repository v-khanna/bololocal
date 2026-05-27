// Bolo/Engine/Chatterbox/S3Gen/Decoder/DecoderWeightMapper.swift
import Foundation
import MLX
import MLXNN

/// Maps Chatterbox-Turbo safetensors weights for the `s3gen.decoder.estimator.*`
/// sub-tree into a Swift `ConditionalDecoder` instance.
///
/// The on-disk safetensors come from the pre-converted MLX repo
/// `mlx-community/chatterbox-turbo-fp16`, so Conv1d weights are already stored
/// in MLX layout `(O, K, I)` — no transposition needed.
///
/// Key prefix: `s3gen.decoder.estimator.<path>` ↔ Swift parameter path
/// `<path>` on the `ConditionalDecoder` instance.
///
/// Keys explicitly skipped:
///   `time_embed_mixer.weight`  — only used when `meanflow=True`. The
///     production decoder configures `meanflow=False`, so this key has no
///     destination on the Swift side and is dropped.
enum DecoderWeightMapper {

    struct Report {
        /// Number of `s3gen.decoder.estimator.*` keys observed in the source.
        let sourceKeyCount: Int
        /// Number of source keys actually applied to the Swift decoder.
        let appliedKeyCount: Int
        /// Source keys we intentionally skipped (not present in the ported
        /// decoder variant).
        let skippedSourceKeys: [String]
        /// Source keys (after stripping the prefix) that have no destination
        /// in the Swift module — indicates a mapping bug.
        let unmappedSourceKeys: [String]
        /// Swift parameter paths that did not receive a weight — still
        /// randomly initialized.
        let unfilledSwiftKeys: [String]
    }

    /// Source-key prefix in the safetensors. Everything below this prefix is
    /// the decoder estimator.
    static let sourcePrefix = "s3gen.decoder.estimator."

    /// Source keys we intentionally skip (no destination in `meanflow=False`).
    private static let skipKeys: Set<String> = [
        "time_embed_mixer.weight",
    ]

    /// Apply weights from a full safetensors dictionary onto a Swift decoder.
    @discardableResult
    static func apply(
        weights: [String: MLXArray],
        to decoder: ConditionalDecoder
    ) -> Report {
        var renamed: [String: MLXArray] = [:]
        var skipped: [String] = []
        var sourceKeyCount = 0

        for (key, value) in weights where key.hasPrefix(sourcePrefix) {
            sourceKeyCount += 1
            let stripped = String(key.dropFirst(sourcePrefix.count))
            if stripped.hasSuffix(".num_batches_tracked") { continue }
            if skipKeys.contains(stripped) {
                skipped.append(key)
                continue
            }
            renamed[stripped] = value
        }

        let expected = Set(decoder.parameters().flattened().map { $0.0 })
        let provided = Set(renamed.keys)

        let unmappedSourceKeys = Array(provided.subtracting(expected)).sorted()
        let unfilledSwiftKeys = Array(expected.subtracting(provided)).sorted()

        let params = ModuleParameters.unflattened(renamed)
        decoder.update(parameters: params)
        eval(decoder)

        return Report(
            sourceKeyCount: sourceKeyCount,
            appliedKeyCount: renamed.count,
            skippedSourceKeys: skipped.sorted(),
            unmappedSourceKeys: unmappedSourceKeys,
            unfilledSwiftKeys: unfilledSwiftKeys
        )
    }
}
