// Bolo/Engine/Chatterbox/S3Gen/S3GenWeightMapper.swift
import Foundation
import MLX
import MLXNN

/// Maps Chatterbox-Turbo safetensors weights into the Swift `S3Gen` instance.
///
/// The Python `S3Token2Wav` module structure matches the Swift `S3Gen` we expose
/// here for the modules we've ported so far:
///
///   safetensors key                      ↔ Swift parameter path
///   ─────────────────────────────────────────────────────────────────────
///   s3gen.input_embedding.weight         ↔ input_embedding.weight
///   s3gen.spk_embed_affine_layer.{w,b}   ↔ spk_embed_affine_layer.{weight,bias}
///   s3gen.encoder_proj.{weight,bias}     ↔ encoder_proj.{weight,bias}
///   s3gen.encoder.embed.linear.{w,b}     ↔ encoder.embed.linear.{w,b}
///   s3gen.encoder.embed.norm.{w,b}       ↔ encoder.embed.norm.{w,b}
///   s3gen.encoder.pre_lookahead_layer.*  ↔ encoder.pre_lookahead_layer.*
///   s3gen.encoder.encoders.{N}.{path}    ↔ encoder.encoders.{N}.{path}
///   s3gen.encoder.up_layer.conv.{w,b}    ↔ encoder.up_layer.conv.{w,b}
///   s3gen.encoder.up_embed.*             ↔ encoder.up_embed.*
///   s3gen.encoder.up_encoders.{N}.{path} ↔ encoder.up_encoders.{N}.{path}
///   s3gen.encoder.after_norm.{w,b}       ↔ encoder.after_norm.{w,b}
///
/// Conv1d weights in the safetensors are already stored in MLX layout `(O, K, I)`
/// (the `mlx-community/chatterbox-turbo-fp16` repo is a pre-converted MLX model)
/// so no transposition is needed.
///
/// Keys NOT mapped (intentionally, until those Swift modules are ported):
///   s3gen.speaker_encoder.*  CAMPPlus x-vector net
///   s3gen.decoder.*          ConditionalDecoder + CFM
///   s3gen.mel2wav.*          HiFTGenerator
///   s3gen.tokenizer.*        S3 tokenizer (CNN-RNN audio → discrete tokens)
enum S3GenWeightMapper {

    /// Diagnostic struct returned by `apply` to help debug mapping issues.
    struct Report {
        /// Number of `s3gen.*` weights observed in the source dictionary.
        let s3genKeyCount: Int
        /// Number of weights actually applied to the Swift modules ported so far.
        let appliedKeyCount: Int
        /// Source keys with the `s3gen.` prefix that were intentionally skipped
        /// (because the corresponding Swift module is not yet ported).
        let skippedSourceKeys: [String]
        /// Source keys (after stripping `s3gen.`) that had no Swift parameter
        /// destination — unexpected; indicates a mapping bug.
        let unmappedSourceKeys: [String]
        /// Swift parameter paths that did not receive a weight — still randomly
        /// initialized; indicates a mapping bug.
        let unfilledSwiftKeys: [String]
    }

    /// Sub-modules handled directly here by stripping the `s3gen.` prefix and
    /// applying the inner key to the Swift parameter tree. Source keys that
    /// don't start with one of these are routed elsewhere (decoder/vocoder
    /// sub-mappers) or reported as `skipped`.
    private static let directPrefixes: [String] = [
        "input_embedding.",
        "spk_embed_affine_layer.",
        "encoder.",
        "encoder_proj.",
    ]

    /// Sub-trees routed to dedicated mappers inside `apply`. Source keys with
    /// these prefixes are NOT counted as `skipped` even though they bypass
    /// the direct path.
    private static let routedPrefixes: [String] = [
        "decoder.",
        "mel2wav.",
    ]

    /// Apply weights to an `S3Gen` instance.
    ///
    /// Routes:
    ///   - `s3gen.input_embedding.*`, `s3gen.spk_embed_affine_layer.*`,
    ///     `s3gen.encoder.*`, `s3gen.encoder_proj.*` → directly here
    ///   - `s3gen.decoder.estimator.*` → `DecoderWeightMapper.apply(... to: s3gen.decoder)`
    ///   - `s3gen.mel2wav.*`           → `VocoderWeightMapper.apply(... to: s3gen.mel2wav)`
    ///   - other `s3gen.*` keys (e.g. `s3gen.speaker_encoder.*`, `s3gen.tokenizer.*`) → skipped
    ///
    /// - Parameters:
    ///   - weights: full safetensors dictionary (containing `t3.*`, `s3gen.*`, `ve.*` keys).
    ///   - s3gen: the Swift `S3Gen` module to populate.
    /// - Returns: a `Report` describing the mapping.
    @discardableResult
    static func apply(weights: [String: MLXArray], to s3gen: S3Gen) -> Report {
        var renamed: [String: MLXArray] = [:]
        var skipped: [String] = []
        var s3genKeyCount = 0

        for (key, value) in weights where key.hasPrefix("s3gen.") {
            s3genKeyCount += 1
            let stripped = String(key.dropFirst("s3gen.".count))
            // Skip num_batches_tracked buffers that PyTorch emits and are
            // irrelevant to inference (also: not parameters).
            if stripped.hasSuffix(".num_batches_tracked") { continue }
            if directPrefixes.contains(where: { stripped.hasPrefix($0) }) {
                renamed[stripped] = value
            } else if routedPrefixes.contains(where: { stripped.hasPrefix($0) }) {
                // Routed to sub-mappers — don't count as skipped, don't apply here.
                continue
            } else {
                skipped.append(key)
            }
        }

        // Apply direct keys.
        // For the `expected` check we only look at the encoder side of the
        // tree (the decoder/mel2wav sub-trees are populated by sub-mappers
        // below; including them here would generate spurious unfilled diags).
        let allExpected = Set(s3gen.parameters().flattened().map { $0.0 })
        let expectedDirect = allExpected.filter { path in
            directPrefixes.contains(where: { path.hasPrefix($0) })
        }
        let provided = Set(renamed.keys)
        let unmappedSourceKeys = Array(provided.subtracting(expectedDirect)).sorted()
        let unfilledSwiftKeys = Array(expectedDirect.subtracting(provided)).sorted()

        let params = ModuleParameters.unflattened(renamed)
        s3gen.update(parameters: params)
        eval(s3gen)

        // Route decoder + vocoder via sub-mappers. They each report their own
        // unmapped/unfilled diagnostics — aggregate notes into our report.
        let decReport = DecoderWeightMapper.apply(weights: weights, to: s3gen.decoder)
        let vocReport = VocoderWeightMapper.apply(weights: weights, to: s3gen.mel2wav)

        // Aggregate. The CFM solver references s3gen.decoder by reference, so
        // populating s3gen.decoder is sufficient.
        var aggUnmapped = unmappedSourceKeys
        aggUnmapped.append(contentsOf: decReport.unmappedSourceKeys.map { "decoder.estimator." + $0 })
        aggUnmapped.append(contentsOf: vocReport.unmappedSourceKeys.map { "mel2wav." + $0 })
        var aggUnfilled = unfilledSwiftKeys
        aggUnfilled.append(contentsOf: decReport.unfilledSwiftKeys.map { "decoder." + $0 })
        aggUnfilled.append(contentsOf: vocReport.unfilledSwiftKeys.map { "mel2wav." + $0 })

        return Report(
            s3genKeyCount: s3genKeyCount,
            appliedKeyCount: renamed.count + decReport.appliedKeyCount + vocReport.appliedKeyCount,
            skippedSourceKeys: skipped.sorted(),
            unmappedSourceKeys: aggUnmapped.sorted(),
            unfilledSwiftKeys: aggUnfilled.sorted()
        )
    }
}
