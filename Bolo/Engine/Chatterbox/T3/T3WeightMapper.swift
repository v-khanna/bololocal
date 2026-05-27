// Bolo/Engine/Chatterbox/T3/T3WeightMapper.swift
import Foundation
import MLX
import MLXNN

/// Maps Chatterbox-Turbo safetensors weights into a Swift `T3` instance.
///
/// The Python `chatterbox_turbo.T3` module structure already aligns perfectly
/// with the Swift T3 (we matched names via `@ModuleInfo(key: ...)`), so the
/// translation is essentially "strip the `t3.` prefix":
///
///   safetensors key                ↔  Swift parameter path
///   ───────────────────────────────────────────────────────────────────────
///   t3.text_emb.weight             ↔  text_emb.weight
///   t3.speech_emb.weight           ↔  speech_emb.weight
///   t3.cond_enc.spkr_enc.weight    ↔  cond_enc.spkr_enc.weight
///   t3.cond_enc.spkr_enc.bias      ↔  cond_enc.spkr_enc.bias
///   t3.tfmr.wte.weight             ↔  tfmr.wte.weight
///   t3.tfmr.wpe.weight             ↔  tfmr.wpe.weight
///   t3.tfmr.h.N.ln_1.{weight,bias} ↔  tfmr.h.N.ln_1.{weight,bias}
///   t3.tfmr.h.N.attn.c_attn.*      ↔  tfmr.h.N.attn.c_attn.*
///   t3.tfmr.h.N.attn.c_proj.*      ↔  tfmr.h.N.attn.c_proj.*
///   t3.tfmr.h.N.ln_2.{weight,bias} ↔  tfmr.h.N.ln_2.{weight,bias}
///   t3.tfmr.h.N.mlp.c_fc.*         ↔  tfmr.h.N.mlp.c_fc.*
///   t3.tfmr.h.N.mlp.c_proj.*       ↔  tfmr.h.N.mlp.c_proj.*
///   t3.tfmr.ln_f.{weight,bias}     ↔  tfmr.ln_f.{weight,bias}
///   t3.text_head.weight            ↔  text_head.weight
///   t3.speech_head.{weight,bias}   ↔  speech_head.{weight,bias}
///
/// The Python safetensors stores Linear weight matrices as `(out_features, in_features)`,
/// which is exactly what MLX-Swift `Linear` expects — no transposition needed.
enum T3WeightMapper {

    /// Diagnostic struct returned by `apply` to help debug mapping issues.
    struct Report {
        /// Number of T3 weights found in the source dictionary.
        let t3KeyCount: Int
        /// Keys in the source that start with `t3.` but had no Swift parameter to receive them.
        let unmappedSourceKeys: [String]
        /// Swift parameter paths that did not receive a weight (still randomly initialized).
        let unfilledSwiftKeys: [String]
    }

    /// Apply weights to a T3 instance.
    ///
    /// - Parameters:
    ///   - weights: full safetensors dictionary (containing `t3.*`, `s3gen.*`, `ve.*` keys).
    ///     Non-`t3.*` keys are ignored.
    ///   - t3: the Swift T3 module to populate.
    /// - Returns: a `Report` describing any unmapped/unfilled keys.
    @discardableResult
    static func apply(weights: [String: MLXArray], to t3: T3) -> Report {
        // Filter to T3 keys and strip the t3. prefix.
        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix("t3.") {
            let stripped = String(key.dropFirst("t3.".count))
            renamed[stripped] = value
        }

        // Build the set of expected Swift parameter paths from the model's parameters tree.
        let expected = Set(t3.parameters().flattened().map { $0.0 })
        let provided = Set(renamed.keys)

        let unmappedSourceKeys = Array(provided.subtracting(expected)).sorted()
        let unfilledSwiftKeys = Array(expected.subtracting(provided)).sorted()

        // Apply via the standard MLX-Swift Module.update path.
        let params = ModuleParameters.unflattened(renamed)
        t3.update(parameters: params)
        eval(t3)

        return Report(
            t3KeyCount: renamed.count,
            unmappedSourceKeys: unmappedSourceKeys,
            unfilledSwiftKeys: unfilledSwiftKeys
        )
    }
}
