// Bolo/Engine/Chatterbox/T3/T3.swift
import Foundation
import MLX
import MLXNN

/// Full T3 backbone — text-and-speaker conditioning to speech-token logits.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.t3.T3` (the GPT-2-based
/// Chatterbox-Turbo backbone, NOT the LLaMA-based regular Chatterbox). Module
/// structure matches the safetensors layout exactly:
///
///   t3.text_emb           — Embedding(50276, 1024)
///   t3.speech_emb         — Embedding(6563, 1024)
///   t3.cond_enc.spkr_enc  — Linear(256, 1024, bias=True)
///   t3.tfmr               — GPT-2 backbone (24 blocks, wte/wpe, ln_f)
///   t3.text_head          — Linear(1024, 50276, bias=False)  [unused in v1]
///   t3.speech_head        — Linear(1024, 6563, bias=True)
///
/// Forward composition (replicating `T3.prepare_input_embeds` + `tfmr` forward):
///   1. cond_emb = cond_enc(speaker_emb, speech_emb(cond_prompt_speech_tokens))
///   2. inputs = concat([cond_emb, text_emb(text_ids), speech_emb(speech_ids)], axis=1)
///   3. hidden = tfmr(inputs)
///   4. speech_logits = speech_head(hidden)  # full sequence
///
/// Note: `tfmr.wte` is a learned embedding present in the safetensors but unused
/// at inference — `inputs_embeds` is built from `text_emb`/`speech_emb` instead.
/// The wte weights are still loaded for parity with the reference.
final class T3: Module {

    // MARK: - Config

    let config: ChatterboxConfig.T3

    // MARK: - Sub-modules (names match Python keys exactly)

    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
    @ModuleInfo(key: "cond_enc") var condEnc: T3CondEnc
    @ModuleInfo(key: "tfmr") var tfmr: T3Tfmr
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear

    // MARK: - Init

    init(config: ChatterboxConfig.T3) {
        self.config = config

        self._textEmb.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenDim
        )
        // Speech vocab: 6563 = 6561 audio tokens + start(6561) + stop(6562) per Turbo config.json
        self._speechEmb.wrappedValue = Embedding(
            embeddingCount: 6563,
            dimensions: config.hiddenDim
        )
        self._condEnc.wrappedValue = T3CondEnc(
            speakerEmbedSize: 256,
            hiddenDim: config.hiddenDim
        )
        self._tfmr.wrappedValue = T3Tfmr(config: config)
        self._textHead.wrappedValue = Linear(config.hiddenDim, config.vocabSize, bias: false)
        self._speechHead.wrappedValue = Linear(config.hiddenDim, 6563, bias: true)

        super.init()
    }

    // MARK: - Forward helpers

    /// Build the prefill input embeddings:
    /// `[cond_spkr | (optional) cond_prompt_speech_emb | text_emb | speech_emb]`.
    ///
    /// - Parameters:
    ///   - textTokens: `(B, T_text)` int32 token IDs.
    ///   - speechTokens: `(B, T_speech)` int32 speech token IDs (typically a single
    ///     `[start_speech_token]` for the prefill forward pass).
    ///   - speakerEmbedding: `(B, 256)` speaker x-vector.
    ///   - condPromptSpeechTokens: optional `(B, T_prompt)` speech-token prompt
    ///     for voice cloning (375 tokens in v1).
    /// - Returns: tuple of `(inputsEmbeds: (B, L, H), condLen: Int)` where
    ///   `L = condLen + textTokens.shape[1] + speechTokens.shape[1]`.
    func prepareInputEmbeds(
        textTokens: MLXArray,
        speechTokens: MLXArray,
        speakerEmbedding: MLXArray,
        condPromptSpeechTokens: MLXArray? = nil
    ) -> (inputsEmbeds: MLXArray, condLen: Int) {
        // Conditioning prefix
        let promptEmb: MLXArray? = condPromptSpeechTokens.map { speechEmb($0) }
        let condEmb = condEnc(speakerEmb: speakerEmbedding, condPromptSpeechEmb: promptEmb)

        // Token embeddings
        let textE = textEmb(textTokens)
        let speechE = speechEmb(speechTokens)

        let inputs = concatenated([condEmb, textE, speechE], axis: 1)
        return (inputs, condEmb.shape[1])
    }

    // MARK: - Forward

    /// Single forward pass over a pre-built input-embedding sequence.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `(B, L, H)` (built via `prepareInputEmbeds`).
    ///   - cacheOffset: position offset for the GPT-2 wpe lookup.
    ///   - caches: optional KV caches, one per block, for incremental decoding.
    /// - Returns: `(B, L, 6563)` speech logits over the entire input sequence.
    ///   The caller selects the last position for next-token prediction.
    func forwardEmbeds(
        inputsEmbeds: MLXArray,
        cacheOffset: Int = 0,
        caches: [T3Cache]? = nil
    ) -> MLXArray {
        let hidden = tfmr(inputsEmbeds: inputsEmbeds, cacheOffset: cacheOffset, caches: caches)
        return speechHead(hidden)
    }

    /// Convenience: build inputs and run the backbone in one call.
    func callAsFunction(
        textTokens: MLXArray,
        speechTokens: MLXArray,
        speakerEmbedding: MLXArray,
        condPromptSpeechTokens: MLXArray? = nil,
        caches: [T3Cache]? = nil
    ) -> MLXArray {
        let (inputs, _) = prepareInputEmbeds(
            textTokens: textTokens,
            speechTokens: speechTokens,
            speakerEmbedding: speakerEmbedding,
            condPromptSpeechTokens: condPromptSpeechTokens
        )
        return forwardEmbeds(inputsEmbeds: inputs, caches: caches)
    }
}
