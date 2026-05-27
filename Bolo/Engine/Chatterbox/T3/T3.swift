// Bolo/Engine/Chatterbox/T3/T3.swift
import Foundation
import MLX
import MLXNN

/// Full T3 backbone — text-and-speaker conditioning to speech-token logits.
///
/// Architecture (GPT-2 style, 24 layers, 1024 hidden, 16 heads):
///   wte: token embedding (vocabSize → H)
///   wpe: position embedding (maxContextLength → H)
///   speaker_proj: Linear(256 → H)
///   24 × T3Block
///   ln_f: final LayerNorm
///   lm_head: Linear(H → 6561, no bias)
///
/// Speaker conditioning: the 256-d speaker_emb is projected to hidden dim
/// and added to every position (broadcast across S). Future iterations may
/// also inject t3.emotion_adv and t3.cond_prompt_speech_tokens — currently
/// excluded; ChatterboxModel (Task 19) handles those.
final class T3: Module {
    let config: ChatterboxConfig.T3

    @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
    @ModuleInfo(key: "wpe") var positionEmbedding: Embedding
    @ModuleInfo(key: "speaker_proj") var speakerProj: Linear
    @ModuleInfo(key: "h") var blocks: [T3Block]
    @ModuleInfo(key: "ln_f") var lnFinal: LayerNorm
    @ModuleInfo(key: "lm_head") var speechHead: Linear

    init(config: ChatterboxConfig.T3) {
        self.config = config
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenDim
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxContextLength,
            dimensions: config.hiddenDim
        )
        self._speakerProj.wrappedValue = Linear(256, config.hiddenDim, bias: true)
        self._blocks.wrappedValue = (0..<config.numLayers).map { _ in T3Block(config: config) }
        self._lnFinal.wrappedValue = LayerNorm(
            dimensions: config.hiddenDim,
            eps: Float(config.layerNormEps)
        )
        self._speechHead.wrappedValue = Linear(config.hiddenDim, 6561, bias: false)
        super.init()
    }

    /// Forward pass.
    /// - inputIDs: (B, S) integer text/speech token IDs
    /// - speakerEmbedding: (B, 256) speaker conditioning vector (from conds.safetensors t3.speaker_emb)
    /// - cache: optional array of one T3Cache per layer (nil for prefill / non-incremental)
    /// Returns: (B, S, 6561) logits over the speech codebook.
    func callAsFunction(
        inputIDs: MLXArray,
        speakerEmbedding: MLXArray,
        cache: [T3Cache]?
    ) -> MLXArray {
        let B = inputIDs.shape[0]
        let S = inputIDs.shape[1]

        let tokenEmb = tokenEmbedding(inputIDs)  // (B, S, H)

        // Build position IDs [0, 1, ..., S-1] broadcast across batch
        let posIDs = MLXArray(0..<S).reshaped([1, S])
        let broadcastPos = broadcast(posIDs, to: [B, S])
        let posEmb = positionEmbedding(broadcastPos)  // (B, S, H)

        // Project speaker embedding (B, 256) → (B, H), broadcast to all positions
        let spk = speakerProj(speakerEmbedding)  // (B, H)
        let spkBroadcast = spk.reshaped([B, 1, config.hiddenDim])  // (B, 1, H)

        var h = tokenEmb + posEmb + spkBroadcast  // (B, S, H) via broadcasting

        // Causal mask only for non-cached forward (full-sequence prefill)
        let mask: MLXArray? = (cache == nil) ? T3Attention.causalMask(seqLen: S) : nil

        for (i, block) in blocks.enumerated() {
            let c: Any? = cache?[i]
            h = block(h, mask: mask, cache: c)
        }

        h = lnFinal(h)
        return speechHead(h)  // (B, S, 6561)
    }
}
