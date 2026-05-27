// Bolo/Engine/Chatterbox/T3/T3CondEnc.swift
import Foundation
import MLX
import MLXNN

/// Speaker / prompt conditioning encoder for the T3 backbone.
///
/// Mirrors `mlx_audio.tts.models.chatterbox_turbo.models.t3.cond_enc.T3CondEnc`:
/// projects the 256-d CAMPPlus speaker embedding to the backbone hidden size,
/// then concatenates with any prompt-speech embeddings to form the conditioning
/// prefix that precedes text/speech in the GPT-2 input.
///
/// In Turbo, perceiver/CLAP/emotion-adv are all disabled — the only weights are
/// `spkr_enc.{weight,bias}` (Linear 256 → 1024).
///
/// Forward expects pre-embedded prompt speech tokens (call `speechEmb` outside
/// to embed `cond_prompt_speech_tokens` first), and returns
/// `(B, 1 + promptLen, H)` — one speaker slot plus the prompt embeddings.
final class T3CondEnc: Module {

    @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear

    let speakerEmbedSize: Int
    let hiddenDim: Int

    init(speakerEmbedSize: Int, hiddenDim: Int) {
        self.speakerEmbedSize = speakerEmbedSize
        self.hiddenDim = hiddenDim
        self._spkrEnc.wrappedValue = Linear(speakerEmbedSize, hiddenDim, bias: true)
        super.init()
    }

    /// Build the conditioning prefix.
    ///
    /// - Parameters:
    ///   - speakerEmb: shape `(B, 256)` — CAMPPlus x-vector.
    ///   - condPromptSpeechEmb: optional `(B, promptLen, H)` already-embedded prompt
    ///     speech tokens (caller should embed `cond_prompt_speech_tokens` with
    ///     `speechEmb` *outside* of this module — matches the Python flow where
    ///     embedding happens in `T3.prepare_conditioning`).
    /// - Returns: `(B, 1 + promptLen, H)` conditioning embeddings.
    func callAsFunction(
        speakerEmb: MLXArray,
        condPromptSpeechEmb: MLXArray?
    ) -> MLXArray {
        // (B, 256) → (B, H) → (B, 1, H)
        let spk = speakerEmb.reshaped([-1, speakerEmbedSize])
        let condSpkr = spkrEnc(spk).reshaped([spk.shape[0], 1, hiddenDim])

        if let promptEmb = condPromptSpeechEmb {
            return concatenated([condSpkr, promptEmb], axis: 1)
        }
        return condSpkr
    }
}
