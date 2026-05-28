// Bolo/Engine/Chatterbox/ChatterboxPipeline.swift
import Foundation
import MLX
import MLXNN
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
// ChatterboxPipeline is used exclusively through ChatterboxTTSEngine (an actor),
// which serialises all access. The sub-modules (T3, S3Gen) inherit from MLXNN.Module
// and are not Sendable; @unchecked Sendable is safe here because the actor guarantee
// prevents concurrent access.
final class ChatterboxPipeline: @unchecked Sendable {

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
    ///
    /// - Parameters:
    ///   - progressHandler: progress callback (0…1, label).
    ///   - quantizeBits: if non-nil, quantize Linear layers in T3 and S3Gen to this bit
    ///     width (4 or 8 are sensible choices) using `MLXNN.quantize` with `groupSize: 64`.
    ///     Embeddings, layer norms, output heads, and small projection layers stay fp16
    ///     because quantizing them hurts quality more than it speeds up inference.
    ///     Pass `nil` for full fp16 (the parity-test baseline). Default is 4-bit for
    ///     production speed.
    static func load(
        progressHandler: @escaping @Sendable (Double, String) -> Void = { _, _ in },
        quantizeBits: Int? = 4
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

        // Optional in-memory quantization of T3 and S3Gen Linear layers.
        if let bits = quantizeBits {
            progressHandler(0.96, "Quantizing T3 & S3Gen to \(bits)-bit…")
            applyQuantization(t3: t3, s3gen: s3gen, bits: bits, groupSize: 64)
        }

        progressHandler(1.0, "Pipeline ready.")
        return ChatterboxPipeline(
            tokenizer: tokenizer,
            speakerEmbeddings: speakerEmbeddings,
            t3: t3,
            s3gen: s3gen
        )
    }

    // MARK: - Quantization

    /// Quantize T3 and S3Gen Linear layers in-place using a hybrid two-pass
    /// strategy.
    ///
    /// **Why two passes?** `MLXNN.quantize()` walks `leafModules()`, computes
    /// quantized replacements, and applies them via `update(modules:)`. That
    /// path can't reconstruct a `[Module]` array wrapper when the array's
    /// elements have different child shapes — e.g. `FeedForward.net =
    /// [GELUProj, Linear]`. The unflattened update produces a mix of
    /// `.dictionary` (for `net.0.proj`) and `.value` (for `net.1`) entries
    /// under `net`, and `Module.update(modules:verify:)` throws
    /// `mismatchedContainers` when it tries to recurse those side-by-side.
    ///
    /// To work around this, we:
    /// 1. **Pass 1** — call `MLXNN.quantize` with a filter that excludes
    ///    every Linear whose path lives under one of the four known
    ///    heterogeneous-array wrappers (`.net.`, `.to_out.`, `.block.`,
    ///    `.mlp.`). This safely quantizes ~80% of the parameters.
    /// 2. **Pass 2** — walk the model with `namedModules()`, find each
    ///    `FeedForward` / `SelfAttention1D` / `ResnetBlock1D` instance, and
    ///    rebuild its array wrapper by hand: each element is either swapped
    ///    for its `QuantizedLinear` equivalent or its `proj` child is.
    ///    The rebuilt array is then re-installed via `updateModule(key:_:)`,
    ///    which avoids the partial-update reconciliation that breaks
    ///    `MLXNN.quantize`.
    ///
    /// Filter rules (skip = keep fp16):
    /// - All `Embedding` layers (text_emb, speech_emb, tfmr.wte, input_embedding).
    ///   Embeddings are looked up by index; quantizing them hurts quality and
    ///   the speedup is negligible.
    /// - `t3.speech_head` and `t3.text_head` — final output projections; quantizing
    ///   output heads tends to degrade quality.
    /// - `t3.cond_enc.spkr_enc` — small speaker projection (256→1024), runs once.
    /// - `s3gen.encoder_proj` (512→80) and `s3gen.spk_embed_affine_layer` (192→80)
    ///   — small projections with output dim not divisible by groupSize anyway.
    /// - Any Linear whose input dim is not divisible by `groupSize` (MLX
    ///   requirement; would crash otherwise).
    /// - Pass 1 only: any Linear nested under a known `[Module]` array
    ///   wrapper. These are picked up by Pass 2 instead.
    private static func applyQuantization(t3: T3, s3gen: S3Gen, bits: Int, groupSize: Int) {
        // Paths that should NEVER be quantized. Matched as suffixes on the
        // dotted module path supplied by MLX (e.g. "speech_head", "cond_enc.spkr_enc").
        let t3Skip: Set<String> = [
            "speech_head",
            "text_head",
            "cond_enc.spkr_enc",
        ]
        let s3genSkip: Set<String> = [
            "encoder_proj",
            "spk_embed_affine_layer",
        ]

        // Substring markers identifying paths that live inside one of the
        // four heterogeneous `[Module]` wrappers. Pass 1 skips these; Pass 2
        // handles them manually.
        let arrayWrapperMarkers: [String] = [
            ".net.",
            ".to_out.",
            ".block.",
            ".mlp.",
        ]

        func filter(skip: Set<String>) -> (String, Module) -> Bool {
            return { path, module in
                // Skip non-quantizable types up-front (Embeddings, LayerNorms, etc.
                // The default `apply` already returns nil for these, but skipping
                // here is cleaner.)
                if module is Embedding { return false }
                if !(module is Linear) { return false }
                if skip.contains(path) { return false }
                // Group-size divisibility check on input dim (Linear.weight shape
                // is [out, in]).
                if let linear = module as? Linear {
                    let inDim = linear.weight.shape.last ?? 0
                    if inDim % groupSize != 0 { return false }
                }
                // Defer Linears inside heterogeneous `[Module]` array wrappers
                // (FeedForward.net, SelfAttention1D.to_out, CausalBlock1D.block,
                // ResnetBlock1D.mlp) to the manual post-pass. Matching the
                // surrounding dots avoids accidental hits on top-level paths.
                let dottedPath = ".\(path)."
                for marker in arrayWrapperMarkers {
                    if dottedPath.contains(marker) { return false }
                }
                return true
            }
        }

        // ----------------------------------------------------------------
        // Pass 1: stock MLXNN.quantize() for everything outside the four
        // heterogeneous `[Module]` array wrappers.
        // ----------------------------------------------------------------

        // Pre-count Linears we expect Pass 1 to quantize, for reporting.
        func countQuantizable(_ root: Module, skip: Set<String>) -> Int {
            let f = filter(skip: skip)
            return root.namedModules().reduce(0) { acc, named in
                f(named.0, named.1) ? acc + 1 : acc
            }
        }
        let t3Pass1Expected = countQuantizable(t3, skip: t3Skip)
        let s3genPass1Expected = countQuantizable(s3gen, skip: s3genSkip)

        quantize(model: t3, groupSize: groupSize, bits: bits, filter: filter(skip: t3Skip))
        quantize(model: s3gen, groupSize: groupSize, bits: bits, filter: filter(skip: s3genSkip))

        // ----------------------------------------------------------------
        // Pass 2: manual rebuild of heterogeneous `[Module]` array wrappers.
        // ----------------------------------------------------------------

        var pass2Quantized = 0
        var pass2Skipped = 0

        // Helper: quantize a Linear via QuantizedLinear, respecting group-size
        // divisibility. Returns nil for already-quantized layers.
        func quantizeLinear(_ linear: Linear) -> Module? {
            if linear is QuantizedLinear { return nil }
            let inDim = linear.weight.shape.last ?? 0
            if inDim % groupSize != 0 { return nil }
            return quantizeSingle(layer: linear, groupSize: groupSize, bits: bits, mode: .affine)
        }

        // Visit every Module in the tree. We dispatch on the concrete types
        // that own one of the four heterogeneous arrays.
        for root in [t3 as Module, s3gen as Module] {
            for (_, module) in root.namedModules() {

                if let ff = module as? FeedForward {
                    // net = [GELUProj, Linear]
                    var newNet: [Module] = ff.net
                    for (i, element) in ff.net.enumerated() {
                        if let gelu = element as? GELUProj {
                            if let q = quantizeLinear(gelu.proj) {
                                // Swap the inner proj on the existing GELUProj.
                                // proj is a normal @ModuleInfo<Linear>, so
                                // updateModule(key: "proj", _:) works.
                                do {
                                    try gelu.updateModule(key: "proj", q)
                                    pass2Quantized += 1
                                } catch {
                                    pass2Skipped += 1
                                }
                            } else {
                                pass2Skipped += 1
                            }
                            newNet[i] = gelu
                        } else if let lin = element as? Linear {
                            if let q = quantizeLinear(lin) {
                                newNet[i] = q
                                pass2Quantized += 1
                            } else {
                                pass2Skipped += 1
                            }
                        }
                    }
                    try? ff.updateModule(key: "net", newNet)
                }

                if let attn = module as? SelfAttention1D {
                    // to_out = [Linear]
                    var newOut: [Module] = attn.toOut
                    for (i, element) in attn.toOut.enumerated() {
                        if let lin = element as? Linear {
                            if let q = quantizeLinear(lin) {
                                newOut[i] = q
                                pass2Quantized += 1
                            } else {
                                pass2Skipped += 1
                            }
                        }
                    }
                    try? attn.updateModule(key: "to_out", newOut)
                }

                if let res = module as? ResnetBlock1D {
                    // mlp = [Linear]
                    var newMlp: [Module] = res.mlp
                    for (i, element) in res.mlp.enumerated() {
                        if let lin = element as? Linear {
                            if let q = quantizeLinear(lin) {
                                newMlp[i] = q
                                pass2Quantized += 1
                            } else {
                                pass2Skipped += 1
                            }
                        }
                    }
                    try? res.updateModule(key: "mlp", newMlp)
                    // `block` (CausalBlock1D.block = [CausalConv1d, LayerNorm])
                    // contains no Linears, so nothing to do there.
                }
            }
        }

        print("[ChatterboxPipeline] Quantization complete — " +
              "Pass 1: T3 \(t3Pass1Expected) Linears, S3Gen \(s3genPass1Expected) Linears; " +
              "Pass 2: \(pass2Quantized) Linears quantized, \(pass2Skipped) skipped.")
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
    /// `maxGenLen` is a hard cap on speech tokens (25 Hz, so 1000 ≈ 40 s of audio).
    /// Generation stops earlier if T3 emits `stopSpeechToken`. Until KV cache is
    /// wired into T3Attention this loop is O(N²) — wall time scales quadratically
    /// with this value. 1000 is the production-safe ceiling: covers ~40 s sentences
    /// and is what Chatterbox-Turbo Python uses by default.
    func generate(text: String, maxGenLen: Int = 1000) async throws -> [Float] {
        let t0 = Date()
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
        let tPrep = Date()

        // 3. T3 generation loop — KV cached now (see v2.13).
        let speechTokens = generateSpeechTokens(
            textTokens: textTokens,
            speakerEmbedding: speakerEmb,
            condPromptSpeechTokens: condPromptSpeechTokens,
            maxGenLen: maxGenLen
        )
        // Force evaluation here so the next timing measurement isolates S3Gen
        // (vocoder) cost from T3 cost. Without this, MLX's lazy graph would
        // smear T3 work into the S3Gen timing because S3Gen depends on these
        // tokens and the actual compute happens at the next eval barrier.
        MLX.eval(speechTokens)
        let tT3 = Date()
        let nSpeech = speechTokens.shape[1]
        BoloDebug.log("T3 gen → \(nSpeech) speech tokens in \(String(format: "%.2fs", tT3.timeIntervalSince(tPrep)))")

        // 4. S3Gen → audio. Sample fresh CFM noise + noised_mels.
        let nPrompt = speakerEmbeddings.s3GenIntTensors["gen.prompt_token"]?.count ?? 250
        let mu_T = 2 * (nPrompt + nSpeech)
        let cfmNoise = MLXRandom.normal([1, 80, mu_T])
        let noisedMels = MLXRandom.normal([1, 80, nSpeech * 2])

        let (audio, _) = synthesizeFromSpeechTokens(
            speechTokens: speechTokens,
            pinnedCFMNoise: cfmNoise,
            pinnedNoisedMels: noisedMels,
            pinnedSineGenPhases: nil,
            pinnedSineGenNoise: nil,
            applyTrimFade: true
        )
        MLX.eval(audio)
        let tS3 = Date()
        BoloDebug.log("S3Gen (mel decoder + CFM + vocoder) in \(String(format: "%.2fs", tS3.timeIntervalSince(tT3)))")
        BoloDebug.log("ChatterboxPipeline.generate total: \(String(format: "%.2fs", tS3.timeIntervalSince(t0))) — text=\(textTokenIDs.count) tok, speech=\(nSpeech) tok")
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

    // MARK: - T3 generation (KV-cached — O(N) per token)

    /// Generate speech tokens autoregressively, reusing a per-layer KV cache so
    /// each step costs O(1) (in sequence length) instead of O(N). The first
    /// iteration is a full prefill that fills the caches; subsequent
    /// iterations feed a single new token and grow the cache by 1.
    ///
    /// Sampling strategy (production default): temperature + repetition penalty
    /// + categorical sampling, matching the Python reference defaults
    /// (temperature=0.8, repetition_penalty=1.2, repetition_window=64). This
    /// is critical for TTS: argmax causes the model to collapse into token
    /// loops (e.g. `[4672, 4672, 4672, ...]`) that never emit `stopSpeechToken`,
    /// so generation runs to `maxGenLen` and produces tens of seconds of
    /// repetitive garbled audio. Stochastic sampling breaks the loop and lets
    /// the natural stop token fire.
    ///
    /// - Parameters:
    ///   - useCache: when false, falls back to the O(N²) no-cache loop. Used
    ///     by parity tests to verify the cached path produces identical tokens
    ///     (which requires `useSampling: false`, since two stochastic runs
    ///     won't match unless the RNG state is reset).
    ///   - useSampling: when false, uses deterministic argmax. Used by parity
    ///     tests; production should leave this `true`.
    ///   - temperature: scales logits before sampling. 0.8 matches Python.
    ///   - repetitionPenalty: divisor applied to logits of recently emitted
    ///     tokens. 1.2 matches Python. Set to 1.0 to disable.
    ///   - repetitionWindow: how many recent tokens to penalise. 64 ≈ 2.5 s
    ///     of speech context at the 25 Hz speech-token rate.
    func generateSpeechTokens(
        textTokens: MLXArray,
        speakerEmbedding: MLXArray,
        condPromptSpeechTokens: MLXArray,
        maxGenLen: Int,
        useCache: Bool = true,
        useSampling: Bool = true,
        temperature: Float = 0.8,
        repetitionPenalty: Float = 1.2,
        repetitionWindow: Int = 64
    ) -> MLXArray {
        if !useCache {
            return generateSpeechTokensNoCache(
                textTokens: textTokens,
                speakerEmbedding: speakerEmbedding,
                condPromptSpeechTokens: condPromptSpeechTokens,
                maxGenLen: maxGenLen
            )
        }

        // One KV cache per transformer block.
        let cfg = t3.config
        let caches: [T3Cache] = (0..<cfg.numLayers).map { _ in
            T3Cache(numHeads: cfg.numHeads, headDim: cfg.headDim)
        }

        var generated: [Int32] = []
        let tPrefStart = Date()

        // --- Prefill: feed the full conditioning + text + start_speech_token. ---
        let speechStart = MLXArray([hp.startSpeechToken]).reshaped([1, 1]).asType(.int32)
        let (prefillEmbeds, _) = t3.prepareInputEmbeds(
            textTokens: textTokens,
            speechTokens: speechStart,
            speakerEmbedding: speakerEmbedding,
            condPromptSpeechTokens: condPromptSpeechTokens
        )
        let prefillLogits = t3.forwardEmbeds(
            inputsEmbeds: prefillEmbeds,
            cacheOffset: 0,
            caches: caches
        )
        // Take logits for the last (start_speech_token) position to predict the
        // first generated speech token.
        var L = prefillEmbeds.shape[1]
        var last = prefillLogits[0..., (prefillLogits.shape[1] - 1)..<prefillLogits.shape[1], 0...]
            .reshaped([1, hp.speechVocabSize])
        var nextID = sampleNext(
            logits: last,
            recent: generated,
            useSampling: useSampling,
            temperature: temperature,
            repetitionPenalty: repetitionPenalty,
            repetitionWindow: repetitionWindow
        )
        if nextID == hp.stopSpeechToken {
            return MLXArray([Int32]()).reshaped([1, 0]).asType(.int32)
        }
        generated.append(nextID)
        let tPrefDone = Date()
        BoloDebug.log("  T3 prefill (L=\(L)) in \(String(format: "%.2fs", tPrefDone.timeIntervalSince(tPrefStart)))")

        // --- Incremental steps: single new token per forward pass. ---
        let tLoopStart = Date()
        for _ in 1..<maxGenLen {
            let newTok = MLXArray([nextID]).reshaped([1, 1]).asType(.int32)
            // Embed just the single new speech token. The cache holds everything before it.
            let stepEmbed = t3.speechEmb(newTok)  // (1, 1, H)
            let stepLogits = t3.forwardEmbeds(
                inputsEmbeds: stepEmbed,
                cacheOffset: L,
                caches: caches
            )
            L += 1
            last = stepLogits[0..., (stepLogits.shape[1] - 1)..<stepLogits.shape[1], 0...]
                .reshaped([1, hp.speechVocabSize])
            nextID = sampleNext(
                logits: last,
                recent: generated,
                useSampling: useSampling,
                temperature: temperature,
                repetitionPenalty: repetitionPenalty,
                repetitionWindow: repetitionWindow
            )
            if nextID == hp.stopSpeechToken { break }
            generated.append(nextID)
        }
        let tLoopDone = Date()
        let loopSec = tLoopDone.timeIntervalSince(tLoopStart)
        let perTokenMs = generated.count > 1 ? loopSec * 1000 / Double(generated.count - 1) : 0
        BoloDebug.log("  T3 decode loop: \(generated.count) tokens in \(String(format: "%.2fs", loopSec)) (\(String(format: "%.1f", perTokenMs)) ms/tok)")

        if generated.isEmpty {
            return MLXArray([Int32]()).reshaped([1, 0]).asType(.int32)
        }
        return MLXArray(generated).reshaped([1, generated.count]).asType(.int32)
    }

    /// Pick the next speech token from a logits row.
    ///
    /// Pipeline (when sampling): apply repetition penalty to the recent
    /// window → divide by temperature → sample via `MLXRandom.categorical`.
    /// When `useSampling == false` falls back to argmax (used by parity tests).
    ///
    /// Repetition penalty implementation: for each unique token in the recent
    /// window, divide its logit if positive, multiply if negative — this is
    /// the standard HuggingFace `RepetitionPenaltyLogitsProcessor` formula and
    /// matches the Python Chatterbox-Turbo reference.
    private func sampleNext(
        logits: MLXArray,                 // shape [1, vocabSize]
        recent: [Int32],
        useSampling: Bool,
        temperature: Float,
        repetitionPenalty: Float,
        repetitionWindow: Int
    ) -> Int32 {
        // 1. Materialise to CPU so we can apply the penalty per-index cheaply.
        //    Vocab is 6563 → 26 KB / step. Negligible at the 25 Hz token rate
        //    and avoids the awkwardness of MLX scatter-update for a tiny edit.
        MLX.eval(logits)
        var l: [Float] = logits.reshaped([hp.speechVocabSize]).asArray(Float.self)

        // 2. Repetition penalty.
        if repetitionPenalty != 1.0 && !recent.isEmpty {
            let window = recent.suffix(repetitionWindow)
            for tok in Set(window) {
                let idx = Int(tok)
                if idx < l.count {
                    let v = l[idx]
                    l[idx] = v >= 0 ? v / repetitionPenalty : v * repetitionPenalty
                }
            }
        }

        if !useSampling {
            // Argmax path — used by parity tests.
            var bestIdx = 0
            var bestVal = -Float.infinity
            for (i, v) in l.enumerated() where v > bestVal {
                bestVal = v
                bestIdx = i
            }
            return Int32(bestIdx)
        }

        // 3. Temperature + categorical sample.
        if temperature != 1.0 && temperature > 0 {
            let invT = 1.0 / temperature
            for i in 0..<l.count { l[i] *= invT }
        }
        let processed = MLXArray(l).reshaped([1, hp.speechVocabSize])
        return MLXRandom.categorical(processed, axis: -1).item(Int32.self)
    }

    /// Reference O(N²) generation loop. Kept for parity testing against the
    /// cached path — they should produce bit-exact identical token sequences.
    private func generateSpeechTokensNoCache(
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
        if generated.isEmpty {
            return MLXArray([Int32]()).reshaped([1, 0]).asType(.int32)
        }
        return MLXArray(generated).reshaped([1, generated.count]).asType(.int32)
    }
}
