// Bolo/Engine/Chatterbox/ChatterboxPipeline.swift
import Foundation
import MLX
import MLXRandom

/// Top-level Chatterbox-Turbo pipeline.
///
/// Composes every sub-model into a single text → audio pipeline:
///
///   text  ─► EnTokenizer ─► text_tokens
///   text_tokens + speaker_emb + cond_prompt_speech_tokens
///         ─► T3 ─► speech_tokens
///   speech_tokens + gen.* conditioning
///         ─► S3Gen ─► audio
///
/// Phase 5e (end-to-end composition gate) — this class is the integration
/// point. The actual integration into `ChatterboxTTSEngine` (driving AVAudio
/// playback, voice cloning, etc.) is Phase 6.
///
/// Notes:
/// - `synthesizeFromSpeechTokens` is the parity-test entry point: caller
///   supplies the speech tokens (typically from a saved Python reference)
///   plus any pinned randomness, and gets back the final audio.
/// - `generate(text:)` runs the full pipeline including T3 token generation.
///   T3 currently has no KV-cache hooked up to attention, so generation does
///   N full forward passes (O(N²) wall time). This is acceptable for a
///   correctness gate; performance work lands in a later phase.
final class ChatterboxPipeline {

    // MARK: - Sub-modules and conditioning

    let tokenizer: EnTokenizer
    let speakerEmbeddings: SpeakerEmbeddings
    let t3: T3
    let s3gen: S3Gen

    /// T3 hyperparams — start/stop speech tokens, etc. Pinned from the live
    /// config.json (mlx-community/chatterbox-turbo-fp16):
    ///   start_speech_token = 6561, stop_speech_token = 6562
    struct T3Hyperparams: Sendable {
        let startSpeechToken: Int32 = 6561
        let stopSpeechToken: Int32 = 6562
        let speechVocabSize: Int = 6563
    }
    let hp = T3Hyperparams()

    // MARK: - Init

    init(
        tokenizer: EnTokenizer,
        speakerEmbeddings: SpeakerEmbeddings,
        t3: T3,
        s3gen: S3Gen
    ) {
        self.tokenizer = tokenizer
        self.speakerEmbeddings = speakerEmbeddings
        self.t3 = t3
        self.s3gen = s3gen
    }

    // MARK: - Factory

    /// Construct the full pipeline: load tokenizer + speaker embeddings from the
    /// app bundle, download (if needed) and load model weights, and apply every
    /// weight mapper.
    static func load(
        progressHandler: @escaping @Sendable (Double, String) -> Void = { _, _ in }
    ) async throws -> ChatterboxPipeline {
        progressHandler(0.0, "Loading tokenizer…")
        let tokenizer = try EnTokenizer.loadFromBundle()

        progressHandler(0.05, "Loading speaker conditioning…")
        let speakerEmbeddings = try SpeakerEmbeddings.loadFromBundle()

        progressHandler(0.1, "Loading model weights…")
        let weights = try await WeightLoader.downloadAndLoad { fraction, message in
            // Map weight-loading progress into a 0.1 → 0.85 band.
            progressHandler(0.1 + fraction * 0.75, message)
        }

        progressHandler(0.85, "Mapping T3 weights…")
        let t3 = T3(config: ChatterboxConfig.turbo.t3)
        let t3Report = T3WeightMapper.apply(weights: weights, to: t3)
        if !t3Report.unmappedSourceKeys.isEmpty || !t3Report.unfilledSwiftKeys.isEmpty {
            print("[ChatterboxPipeline] T3 mapping warnings — " +
                  "unmapped: \(t3Report.unmappedSourceKeys.prefix(5)), " +
                  "unfilled: \(t3Report.unfilledSwiftKeys.prefix(5))")
        }

        progressHandler(0.92, "Mapping S3Gen weights…")
        let s3gen = S3Gen(config: ChatterboxConfig.turbo.s3gen)
        // The S3GenWeightMapper now routes decoder/vocoder via their dedicated
        // mappers internally — one call populates the whole tree.
        let s3genReport = S3GenWeightMapper.apply(weights: weights, to: s3gen)
        if !s3genReport.unmappedSourceKeys.isEmpty || !s3genReport.unfilledSwiftKeys.isEmpty {
            print("[ChatterboxPipeline] S3Gen mapping warnings — " +
                  "unmapped: \(s3genReport.unmappedSourceKeys.prefix(5)), " +
                  "unfilled: \(s3genReport.unfilledSwiftKeys.prefix(5))")
        }

        progressHandler(1.0, "Pipeline ready.")
        return ChatterboxPipeline(
            tokenizer: tokenizer,
            speakerEmbeddings: speakerEmbeddings,
            t3: t3,
            s3gen: s3gen
        )
    }

    // MARK: - S3Gen entry point (used by parity test)

    /// Run the S3Gen half of the pipeline with the supplied speech tokens.
    /// Used by the end-to-end parity gate (speech tokens come from Python's
    /// saved reference) and by `generate(text:)` (speech tokens come from T3).
    ///
    /// Conditioning comes from `speakerEmbeddings` (loaded from
    /// `conds.safetensors`). The pinned-randomness parameters let parity
    /// tests bit-exactly replay Python's RNG.
    func synthesizeFromSpeechTokens(
        speechTokens: MLXArray,
        pinnedCFMNoise: MLXArray? = nil,
        pinnedNoisedMels: MLXArray? = nil,
        pinnedSineGenPhases: MLXArray? = nil,
        pinnedSineGenNoise: MLXArray? = nil,
        applyTrimFade: Bool = true
    ) -> (audio: MLXArray, speechFeat: MLXArray) {
        let (promptToken, promptTokenLen, promptFeat, xvector) = buildS3GenConditioning()
        return s3gen.synthesize(
            speechTokens: speechTokens,
            promptToken: promptToken,
            promptTokenLen: promptTokenLen,
            promptFeat: promptFeat,
            speakerXVector: xvector,
            pinnedCFMNoise: pinnedCFMNoise,
            pinnedNoisedMels: pinnedNoisedMels,
            pinnedSineGenPhases: pinnedSineGenPhases,
            pinnedSineGenNoise: pinnedSineGenNoise,
            applyTrimFade: applyTrimFade
        )
    }

    // MARK: - Full text → audio path

    /// End-to-end synthesize: text → audio waveform `[Float]` at 24 kHz.
    ///
    /// Performance: T3 currently runs without a KV cache, so generation is
    /// O(N²). At max_gen_len ~50 speech tokens this is workable; production
    /// performance is Phase 6+ work.
    ///
    /// Sampling: deterministic argmax (no temperature/top-k/top-p). The Python
    /// reference uses temperature=0.8, top_k=1000, top_p=0.95, but those rely
    /// on `mx.random.categorical` which isn't replayed in Swift yet. Argmax is
    /// fine for a smoke-level sanity pass.
    func generate(text: String, maxGenLen: Int = 50) async throws -> [Float] {
        // 1. Tokenize text.
        let textTokenIDs = tokenizer.encode(text)
        let textTokens = MLXArray(textTokenIDs.map { Int32($0) })
            .reshaped([1, textTokenIDs.count])
            .asType(.int32)

        // 2. T3 conditioning.
        let speakerEmb = MLXArray(speakerEmbeddings.speakerEmbedding)
            .reshaped([1, 256])
        let condPromptSpeechTokens = MLXArray(speakerEmbeddings.condPromptSpeechTokens)
            .reshaped([1, speakerEmbeddings.condPromptSpeechTokens.count])
            .asType(.int32)

        // 3. T3 generation loop (no KV cache).
        let speechTokens = generateSpeechTokens(
            textTokens: textTokens,
            speakerEmbedding: speakerEmb,
            condPromptSpeechTokens: condPromptSpeechTokens,
            maxGenLen: maxGenLen
        )

        // 4. S3Gen → audio. Sample fresh CFM noise + noised_mels.
        // T_total is determined by the encoder; here we compute the same
        // formula it uses: 2 * (prompt_token_len + speech_tokens.length).
        // Note: prompt_token_len from conditioning may be < prompt_token.count
        // (some prompt tokens may be padding); but encoder uses the actual
        // count, so 2 * (prompt.count + speech.count) is safe.
        let nSpeech = speechTokens.shape[1]
        let nPrompt = speakerEmbeddings.s3GenIntTensors["gen.prompt_token"]?.count ?? 250
        let mu_T = 2 * (nPrompt + nSpeech)
        let cfmNoise = MLXRandom.normal([1, 80, mu_T])
        let noisedMels = MLXRandom.normal([1, 80, nSpeech * 2])
        // Vocoder randomness: SineGen will fall back to internal sampling if
        // we pass nil. That's fine for production.

        let (audio, _) = synthesizeFromSpeechTokens(
            speechTokens: speechTokens,
            pinnedCFMNoise: cfmNoise,
            pinnedNoisedMels: noisedMels,
            pinnedSineGenPhases: nil,
            pinnedSineGenNoise: nil,
            applyTrimFade: true
        )
        MLX.eval(audio)
        return audio[0].asArray(Float.self)
    }

    // MARK: - Conditioning helpers

    /// Build the 4 S3Gen conditioning tensors from `speakerEmbeddings`.
    /// Centralised so the parity test and generate(text:) use the SAME wiring.
    private func buildS3GenConditioning() -> (
        promptToken: MLXArray,
        promptTokenLen: MLXArray,
        promptFeat: MLXArray,
        xvector: MLXArray
    ) {
        guard let promptTokenFloats = speakerEmbeddings.s3GenIntTensors["gen.prompt_token"] else {
            fatalError("ChatterboxPipeline: speakerEmbeddings missing gen.prompt_token")
        }
        guard let promptTokenLenInts = speakerEmbeddings.s3GenIntTensors["gen.prompt_token_len"] else {
            fatalError("ChatterboxPipeline: speakerEmbeddings missing gen.prompt_token_len")
        }
        guard let promptFeatFloats = speakerEmbeddings.s3GenConditioning["gen.prompt_feat"] else {
            fatalError("ChatterboxPipeline: speakerEmbeddings missing gen.prompt_feat")
        }
        guard let xvectorFloats = speakerEmbeddings.s3GenConditioning["gen.embedding"] else {
            fatalError("ChatterboxPipeline: speakerEmbeddings missing gen.embedding")
        }
        let nPromptTokens = promptTokenFloats.count          // 250
        let promptToken = MLXArray(promptTokenFloats).reshaped([1, nPromptTokens]).asType(.int32)
        let promptTokenLen = MLXArray(promptTokenLenInts).reshaped([promptTokenLenInts.count]).asType(.int32)
        // prompt_feat is (1, 500, 80) flattened.
        let nFeat = promptFeatFloats.count
        precondition(nFeat % 80 == 0, "gen.prompt_feat length not divisible by 80")
        let tMel = nFeat / 80
        let promptFeat = MLXArray(promptFeatFloats).reshaped([1, tMel, 80])
        let xvector = MLXArray(xvectorFloats).reshaped([1, ChatterboxConfig.speakerEmbeddingDim])
        return (promptToken, promptTokenLen, promptFeat, xvector)
    }

    // MARK: - T3 generation (no KV cache — slow but correct)

    /// Generate speech tokens autoregressively by running full-sequence T3
    /// forward passes. O(N²) wall time but bit-correct vs the cached path.
    ///
    /// Sampling strategy: argmax (deterministic). Skips repetition penalty,
    /// temperature, top-k, top-p — those are productionisation tasks.
    private func generateSpeechTokens(
        textTokens: MLXArray,
        speakerEmbedding: MLXArray,
        condPromptSpeechTokens: MLXArray,
        maxGenLen: Int
    ) -> MLXArray {
        var speechTokens = MLXArray([hp.startSpeechToken]).reshaped([1, 1]).asType(.int32)
        var generated: [Int32] = []

        for _ in 0..<maxGenLen {
            let logits = t3(
                textTokens: textTokens,
                speechTokens: speechTokens,
                speakerEmbedding: speakerEmbedding,
                condPromptSpeechTokens: condPromptSpeechTokens,
                caches: nil
            )
            // logits: (1, L, 6563). Take the last position.
            let last = logits[0..., (logits.shape[1] - 1)..<logits.shape[1], 0...]
                .reshaped([1, hp.speechVocabSize])
            let nextID = last.argMax(axis: -1).item(Int32.self)
            if nextID == hp.stopSpeechToken { break }
            generated.append(nextID)
            // Append nextID to the running speech tokens.
            let newTok = MLXArray([nextID]).reshaped([1, 1]).asType(.int32)
            speechTokens = concatenated([speechTokens, newTok], axis: 1)
        }
        // Drop the start_speech_token from the front before returning.
        // Python's inference_turbo returns only the generated tokens.
        if generated.isEmpty {
            return MLXArray([Int32]()).reshaped([1, 0]).asType(.int32)
        }
        return MLXArray(generated).reshaped([1, generated.count]).asType(.int32)
    }
}
