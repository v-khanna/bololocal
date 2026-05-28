// Bolo/Engine/Chatterbox/T3/T3Tfmr.swift
import Foundation
import MLX
import MLXNN

/// GPT-2 backbone used by the Chatterbox-Turbo T3 model.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.t3.gpt2.GPT2Model`:
///
///   wte:  Embedding(vocab=50276, dim=1024) — unused when `inputs_embeds` is provided
///         but still has weights in the safetensors file
///   wpe:  Embedding(n_positions=8196, dim=1024) — learned absolute positional embedding
///   h:    [GPT2Block × 24]
///   ln_f: LayerNorm(dim=1024)
///
/// Naming mirrors the safetensors keys exactly so weights drop in without renaming
/// (modulo the outer `t3.tfmr.` prefix that the loader strips):
///   tfmr.wte.weight, tfmr.wpe.weight, tfmr.h.N.*, tfmr.ln_f.{weight,bias}
///
/// Forward takes pre-built `inputsEmbeds: (B, L, H)` (the concatenation of
/// `[condEmb | textEmb | speechEmb]` built one level up in T3) — *not* raw
/// token IDs. Position embeddings start at `cacheOffset` (0 for prefill,
/// `pastLength` when generating subsequent tokens with KV cache).
final class T3Tfmr: Module {

    @ModuleInfo(key: "wte") var wte: Embedding
    @ModuleInfo(key: "wpe") var wpe: Embedding
    @ModuleInfo(key: "h") var h: [T3Block]
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    let config: ChatterboxConfig.T3

    init(config: ChatterboxConfig.T3) {
        self.config = config
        self._wte.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenDim
        )
        self._wpe.wrappedValue = Embedding(
            embeddingCount: config.maxContextLength,
            dimensions: config.hiddenDim
        )
        self._h.wrappedValue = (0..<config.numLayers).map { _ in T3Block(config: config) }
        self._lnF.wrappedValue = LayerNorm(
            dimensions: config.hiddenDim,
            eps: Float(config.layerNormEps)
        )
        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `(B, L, H)` pre-built embeddings.
    ///   - cacheOffset: position offset for `wpe` lookups (0 for prefill).
    ///   - caches: optional per-layer KV caches (one per block) for incremental
    ///     decoding. Pass `nil` for full-sequence prefill.
    /// - Returns: `(B, L, H)` hidden states after final LayerNorm.
    func callAsFunction(
        inputsEmbeds: MLXArray,
        cacheOffset: Int = 0,
        caches: [T3Cache]? = nil
    ) -> MLXArray {
        let L = inputsEmbeds.shape[1]

        // wpe: positions [cacheOffset .. cacheOffset + L)
        let positions = MLXArray(Int32(cacheOffset)..<Int32(cacheOffset + L))
        let posEmb = wpe(positions)  // (L, H)
        var hidden = inputsEmbeds + posEmb  // broadcasts (B, L, H) + (L, H)

        // Causal mask is needed for any multi-token forward pass (prefill, with
        // or without cache). For a single-token incremental step (L == 1) with
        // a cache, causality is implicit — every cached key is in the past.
        let mask: MLXArray? = (L > 1) ? T3Attention.causalMask(seqLen: L) : nil

        for (i, block) in h.enumerated() {
            let c: Any? = caches?[i]
            hidden = block(hidden, mask: mask, cache: c)
        }

        return lnF(hidden)
    }
}
