// Bolo/Engine/Chatterbox/S3Gen/S3Gen.swift
import Foundation
import MLX
import MLXNN

/// Swift port of the S3Gen flow-matching token-to-audio model.
///
/// As of Phase 5e (end-to-end composition gate), S3Gen composes the full
/// `S3Token2Wav` pipeline:
///
///   input_embedding     — Embedding(speech_vocab_size=6561, 512)
///   spk_embed_affine    — Linear(192, 80)        (normalises x-vector → 80-d)
///   encoder             — UpsampleConformerEncoder
///   encoder_proj        — Linear(512, 80)
///   decoder             — ConditionalDecoder (meanflow=true; held by CFM)
///   cfm                 — CausalConditionalCFM (wraps decoder; not a Module
///                                                child — owns no params)
///   mel2wav             — HiFTGenerator
///
/// NOT ported: `speaker_encoder` (CAMPPlus) — Bolo uses the pre-computed
/// x-vector from `conds.safetensors` (`gen.embedding`).
///
/// The pipeline mirrors `mlx_audio.tts.models.chatterbox_turbo.models.s3gen.
/// S3Token2Wav.inference`:
///
///     synthesize(speech_tokens, conditioning)
///       1. encode_for_decoder(...)                           # encoder + projection
///       2. build cond  = concat(prompt_feat, zeros)          # (B, T_mel, 80) → (B, 80, T_mel)
///       3. cfm(noise, mu = h_proj.T, mask, spks, cond)       # 2-step meanflow Euler
///       4. feat = cfm_out[:, :, mel_len1:]                   # drop prompt prefix
///       5. mel2wav.inference(feat.T)                         # (B, T, 80) → (B, T_audio)
///
/// The pinned-randomness entry points (`pinnedCFMNoise`, `pinnedSineGenPhases`,
/// `pinnedSineGenNoise`) let tests replay Python's RNG bit-exactly.
final class S3Gen: Module {

    let config: ChatterboxConfig.S3Gen

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "encoder") var encoder: UpsampleConformerEncoder
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear

    /// Velocity-field estimator (lives under `s3gen.decoder.estimator.*` in
    /// the safetensors). Stored as a plain reference and NOT mounted via
    /// `@ModuleInfo` — MLX-Swift can't unflatten dotted keys, so the decoder's
    /// weights are loaded directly by `DecoderWeightMapper.apply(... to: s3gen.decoder)`.
    let decoder: ConditionalDecoder

    /// HiFTGenerator vocoder. Stored as plain ref for the same reason —
    /// `VocoderWeightMapper.apply(... to: s3gen.mel2wav)` populates it directly.
    let mel2wav: HiFTGenerator

    /// CFM solver wrapping `decoder`. NOT a Module — owns no parameters.
    let cfm: CausalConditionalCFM

    init(config: ChatterboxConfig.S3Gen) {
        self.config = config

        self._inputEmbedding.wrappedValue = Embedding(
            embeddingCount: config.speechVocabSize,
            dimensions: config.tokenEmbeddingDim
        )
        // Speaker affine: norm(x-vector 192d) → 80d
        self._spkEmbedAffineLayer.wrappedValue = Linear(
            ChatterboxConfig.speakerEmbeddingDim, 80, bias: true)
        self._encoder.wrappedValue = UpsampleConformerEncoder(config: config)
        // Project encoder hidden (512) to mel channels (80).
        self._encoderProj.wrappedValue = Linear(config.tokenEmbeddingDim, 80, bias: true)
        // Decoder + CFM. Chatterbox-Turbo uses meanflow=true (2-step Euler).
        // These are *not* @ModuleInfo children — their weights are loaded
        // via the dedicated DecoderWeightMapper / VocoderWeightMapper which
        // call .update(parameters:) directly on the inner module.
        let decoder = ConditionalDecoder(meanflow: true)
        self.decoder = decoder
        self.cfm = CausalConditionalCFM(estimator: decoder)
        self.mel2wav = HiFTGenerator()
        super.init()
    }

    /// Output of `encodeForDecoder`. Everything the CFM decoder needs from the
    /// encoder side of the pipeline.
    struct EncoderOutputs {
        /// Encoder hidden state, projected to mel channels.
        /// Shape `(B, 2(T_prompt + T_speech), 80)`.
        let encoderProjOut: MLXArray
        /// Raw encoder hidden state, BEFORE the projection.
        /// Shape `(B, 2(T_prompt + T_speech), 512)`.
        let encoderOut: MLXArray
        /// Output mask `(B, 1, 2(T_prompt + T_speech))`.
        let encoderMask: MLXArray
        /// Projected speaker embedding `(B, 80)`.
        let speakerEmbedding: MLXArray
    }

    /// Run the deterministic encoder prefix of S3Token2Mel.
    ///
    /// - Parameters:
    ///   - speechTokens: `(B, T_speech)` int32 generated speech tokens.
    ///   - promptToken: `(1, T_prompt)` reference voice prompt tokens.
    ///   - promptTokenLen: `(1,)` int32 — number of valid prompt tokens.
    ///   - speakerXVector: `(1, 192)` precomputed x-vector for the reference voice.
    /// - Returns: see `EncoderOutputs`.
    func encodeForDecoder(
        speechTokens: MLXArray,
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        speakerXVector: MLXArray
    ) -> EncoderOutputs {
        let B = speechTokens.shape[0]
        let speechLen = speechTokens.shape[1]

        // Concatenate [prompt | speech] along time, then build a length-aware mask.
        let token = concatenated([promptToken, speechTokens], axis: 1).asType(.int32)
        let speechLens = MLXArray((0..<B).map { _ in Int32(speechLen) })
        let tokenLen = (promptTokenLen + speechLens).asType(.int32)   // (B,)

        let maxLen = token.shape[1]
        let arange = MLXArray(Int32(0)..<Int32(maxLen))
        let mask = (arange.expandedDimensions(axis: 0) .< tokenLen.expandedDimensions(axis: 1))
            .asType(.float32)                                          // (B, T)
        let mask3d = mask.expandedDimensions(axis: -1)                 // (B, T, 1)

        // Embed and zero-out invalid positions.
        let tokenEmb = inputEmbedding(token) * mask3d                  // (B, T, 512)

        // Encoder.
        let (encOut, encMask) = encoder(tokenEmb, xsLens: tokenLen)    // (B, 2T, 512), (B, 1, 2T)

        // Project to mel space.
        let encProj = encoderProj(encOut)                              // (B, 2T, 80)

        // Speaker affine: normalize and project.
        let xv = speakerXVector
        let xvNorm = xv / (MLX.sqrt((xv * xv).sum(axis: -1, keepDims: true)) + 1e-8)
        let spkProjected = spkEmbedAffineLayer(xvNorm)                 // (B, 80)

        return EncoderOutputs(
            encoderProjOut: encProj,
            encoderOut: encOut,
            encoderMask: encMask,
            speakerEmbedding: spkProjected
        )
    }

    /// End-to-end S3Gen forward: speech tokens + reference conditioning → audio.
    ///
    /// Mirrors `S3Token2Wav.inference` in Python.
    ///
    /// - Parameters:
    ///   - speechTokens: `(B, T_speech)` int32 generated speech tokens (from T3).
    ///   - promptToken: `(1, T_prompt)` reference voice prompt tokens (from `gen.prompt_token`).
    ///   - promptTokenLen: `(1,)` int32 valid length of prompt tokens.
    ///   - promptFeat: `(1, T_prompt_mel, 80)` reference mel features (`gen.prompt_feat`).
    ///   - speakerXVector: `(1, 192)` x-vector (`gen.embedding`).
    ///   - nCFMTimesteps: number of CFM Euler steps (default 2 for meanflow Turbo).
    ///   - pinnedCFMNoise: optional pre-sampled `(B, 80, T_total)` noise for
    ///     the CFM solver — caller supplies for deterministic parity. If nil,
    ///     a fresh sample is drawn (production path).
    ///   - pinnedNoisedMels: optional pre-sampled `(B, 80, T_speech*2)` mel
    ///     noise that gets spliced onto the trailing part of `pinnedCFMNoise`
    ///     to match the Python meanflow=True path
    ///     (`noised_mels = mx.random.normal((B, 80, speech_tokens.shape[1] * 2))`).
    ///     If nil, a fresh sample is drawn.
    ///   - pinnedSineGenPhases: optional `(B, nbHarmonics, 1)` uniform phases
    ///     for the vocoder's SineGen.
    ///   - pinnedSineGenNoise: optional `(B, nbHarmonics+1, T_audio)` standard-
    ///     normal samples for SineGen.
    ///   - applyTrimFade: whether to apply the 20ms cosine trim fade to the
    ///     audio start (matches Python default).
    /// - Returns: `(audio: (B, T_audio), speechFeat: (B, 80, T_speech_mel))`.
    func synthesize(
        speechTokens: MLXArray,
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        speakerXVector: MLXArray,
        nCFMTimesteps: Int = 2,
        pinnedCFMNoise: MLXArray? = nil,
        pinnedNoisedMels: MLXArray? = nil,
        pinnedSineGenPhases: MLXArray? = nil,
        pinnedSineGenNoise: MLXArray? = nil,
        applyTrimFade: Bool = true
    ) -> (audio: MLXArray, speechFeat: MLXArray) {
        let B = speechTokens.shape[0]

        // ── 1. Encoder pipeline ───────────────────────────────────────────
        let enc = encodeForDecoder(
            speechTokens: speechTokens,
            promptToken: promptToken,
            promptTokenLen: promptTokenLen,
            speakerXVector: speakerXVector
        )
        // encoder_proj output: (B, T_total, 80) where T_total = 2*(T_prompt + T_speech)
        // h_masks: (B, 1, T_total)
        // We compute h_lengths and rebuild mask in the (B, 1, T) convention.
        let h = enc.encoderProjOut                                     // (B, T_total, 80)
        let hMasks = enc.encoderMask                                   // (B, 1, T_total)
        let melLen1 = promptFeat.shape[1]                              // T_prompt_mel (= 500 in v1)
        let tTotal = h.shape[1]
        let melLen2 = tTotal - melLen1

        // ── 2. Build conds: concat(prompt_feat, zeros) → (B, 80, T_total) ─
        // Python:
        //   zeros_padding = mx.zeros((B, mel_len2, 80))
        //   conds = mx.concatenate([prompt_feat, zeros_padding], axis=1)
        //   conds = conds.transpose(0, 2, 1)
        precondition(melLen2 >= 0,
            "S3Gen.synthesize: prompt_feat length \(melLen1) exceeds encoder output length \(tTotal). " +
            "speech_tokens is too short relative to the prompt.")
        let zerosPad = MLXArray.zeros([B, melLen2, 80])
        let conds = concatenated([promptFeat, zerosPad], axis: 1).transposed(0, 2, 1)   // (B, 80, T_total)

        // Decoder mask: (B, 1, T_total) of float32 (encoder mask already in this form
        // and width matches T_total).
        let mask = hMasks.asType(.float32)

        // ── 3. CFM Euler solve ───────────────────────────────────────────
        // mu = h_proj.T = (B, 80, T_total)
        let mu = h.transposed(0, 2, 1)                                  // (B, 80, T_total)
        // CFM noise. The CFM solver expects the FULL (B, 80, T_total) noise
        // AND optional `noisedMels` (B, 80, T_speech*2) for meanflow that
        // gets spliced onto the trailing part. If neither is pinned the
        // production code path would sample both (not yet wired).
        precondition(pinnedCFMNoise != nil,
            "S3Gen.synthesize: pinnedCFMNoise is required for now (production " +
            "random sampling path not yet implemented).")
        let noise = pinnedCFMNoise!
        let feat = cfm(
            noise: noise,
            mu: mu,
            mask: mask,
            nTimesteps: nCFMTimesteps,
            spks: enc.speakerEmbedding,
            cond: conds,
            noisedMels: pinnedNoisedMels,
            meanflow: true
        )                                                               // (B, 80, T_total)

        // ── 4. Drop the prompt prefix ────────────────────────────────────
        // Python: feat = feat[:, :, mel_len1:]
        let feat2 = feat[0..., 0..., melLen1..<tTotal]                  // (B, 80, mel_len2)

        // ── 5. Vocoder ───────────────────────────────────────────────────
        // mel2wav.inference expects (B, T, 80). Our feat2 is (B, 80, T) — transpose.
        let feat2T = feat2.transposed(0, 2, 1)                          // (B, mel_len2, 80)
        let (audioRaw, _) = mel2wav.inference(
            speechFeat: feat2T,
            sineGenRandomPhases: pinnedSineGenPhases,
            sineGenNoise: pinnedSineGenNoise
        )                                                               // (B, T_audio)

        // ── 6. Trim fade (optional) ──────────────────────────────────────
        // Python applies a 20ms cosine ramp at the start of the audio to reduce
        // artifacts from the prompt-suffix boundary.
        var audio = audioRaw
        if applyTrimFade {
            let sr = HiFTGenerator.samplingRateDefault
            let nTrim = sr / 50                                         // 20 ms at 24kHz = 480 samples
            let fadeLen = 2 * nTrim
            if audio.shape[1] >= fadeLen {
                // Trim fade: zeros for first nTrim samples, then cosine ramp.
                //   trim_fade[:nTrim] = 0
                //   trim_fade[nTrim:] = (cos(linspace(pi, 0, nTrim)) + 1) / 2
                var fade = [Float](repeating: 0, count: fadeLen)
                for i in 0..<nTrim {
                    let t = Float.pi - Float.pi * Float(i) / Float(nTrim - 1)
                    fade[nTrim + i] = (cos(t) + 1) / 2
                }
                let fadeArr = MLXArray(fade).reshaped([1, fadeLen])     // (1, fadeLen)
                let head = audio[0..., 0..<fadeLen] * fadeArr
                let tail = audio[0..., fadeLen..<audio.shape[1]]
                audio = concatenated([head, tail], axis: 1)
            }
        }

        return (audio, feat2)
    }
}

extension HiFTGenerator {
    /// Default sampling rate for the Chatterbox-Turbo HiFTGenerator.
    /// Hard-coded mirror of the constructor default so callers can compute
    /// e.g. trim-fade lengths without instantiating a vocoder.
    static let samplingRateDefault: Int = 24000
}
