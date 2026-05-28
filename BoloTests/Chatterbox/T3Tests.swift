// BoloTests/Chatterbox/T3Tests.swift
import XCTest
import MLX
import MLXRandom
@testable import Bolo

final class T3Tests: XCTestCase {

    // MARK: - Shape test

    func test_attention_outputShape_matchesInput() {
        let cfg = ChatterboxConfig.turbo.t3
        let attn = T3Attention(config: cfg)
        // Input: (B=1, S=5, H=1024)
        let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
        let output = attn(input, mask: nil, cache: nil)
        XCTAssertEqual(output.shape, [1, 5, cfg.hiddenDim],
                       "T3Attention must preserve (B, S, H) shape through the forward pass")
    }

    // MARK: - Causality test

    func test_attention_causalMask_blocksFutureTokens() {
        let cfg = ChatterboxConfig.turbo.t3
        let attn = T3Attention(config: cfg)

        // Build two inputs that differ only at position 4 (the last token).
        // input1: all random
        let input1 = MLXRandom.normal([1, 5, cfg.hiddenDim])

        // Build input2: keep positions 0-3 identical to input1, replace position 4
        let prefix = input1[0..., 0..<4, 0...]           // (1, 4, H)
        let replacement = MLXRandom.normal([1, 1, cfg.hiddenDim])  // (1, 1, H)
        let input2 = concatenated([prefix, replacement], axis: 1)  // (1, 5, H)
        XCTAssertEqual(input2.shape, [1, 5, cfg.hiddenDim])

        // Evaluate both with a causal mask
        let causal = T3Attention.causalMask(seqLen: 5)
        let output1 = attn(input1, mask: causal, cache: nil)
        let output2 = attn(input2, mask: causal, cache: nil)

        // Force evaluation before comparison
        MLX.eval(output1, output2)

        // With a causal mask, positions 0..<4 of the output should be identical
        // because their attention windows do not reach position 4.
        let prefix1 = output1[0..., 0..<4, 0...]
        let prefix2 = output2[0..., 0..<4, 0...]
        let maxDiff = (prefix1 - prefix2).abs().max()
        let diffValue = maxDiff.item(Float.self)

        XCTAssertLessThan(diffValue, 1e-3,
            "Causal mask must prevent position 4's input from affecting outputs at positions 0-3. " +
            "Max diff was \(diffValue)")
    }

    func test_mlp_outputShape_matchesInput() {
        let cfg = ChatterboxConfig.turbo.t3
        let mlp = T3MLP(config: cfg)
        let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
        let output = mlp(input)
        XCTAssertEqual(output.shape, [1, 5, cfg.hiddenDim])
    }

    func test_block_outputShape_matchesInput() {
        let cfg = ChatterboxConfig.turbo.t3
        let block = T3Block(config: cfg)
        let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
        let mask = T3Attention.causalMask(seqLen: 5)
        let output = block(input, mask: mask, cache: nil)
        XCTAssertEqual(output.shape, [1, 5, cfg.hiddenDim])
    }

    func test_t3_forwardPass_outputShape() {
        let cfg = ChatterboxConfig.turbo.t3
        let t3 = T3(config: cfg)
        // Mirror the Python prefill flow: build (cond | text | speech_start) embeddings.
        let textTokens = MLXArray([42, 17, 8, 99, 3] as [Int32]).reshaped([1, 5])
        let speechStart = MLXArray([6561] as [Int32]).reshaped([1, 1])  // start_speech_token
        let speakerEmb = MLXRandom.normal([1, 256])
        let output = t3(
            textTokens: textTokens,
            speechTokens: speechStart,
            speakerEmbedding: speakerEmb,
            condPromptSpeechTokens: nil,
            caches: nil
        )
        // Output: (B, L, speech_vocab_size = 6563) where L = 1 (cond_spkr) + 5 (text) + 1 (speech_start) = 7
        XCTAssertEqual(output.shape, [1, 7, 6563])
    }

    // MARK: - Reference parity gate (heavy)

    /// THE high-risk-mitigation test from the spec. Loads real Chatterbox-Turbo
    /// weights into the Swift T3 and runs the same prefill forward pass the
    /// Python reference ran, then compares activations layer-by-layer and the
    /// final speech_logits argmax.
    ///
    /// Gated behind BOLO_RUN_HEAVY_TESTS=1 because it needs the 2.99 GB
    /// model.safetensors plus the pre-generated reference outputs at
    /// scripts/chatterbox-reference/reference-outputs/.
    func test_t3_referenceParity_matchesPythonForwardPass() async throws {
        // Heavy test — gated on whether reference outputs have been generated and
        // model weights are reachable. To run it:
        //   1. cd scripts/chatterbox-reference && source venv/bin/activate
        //   2. python generate-reference.py     (produces reference-outputs/*.bin)
        //   3. Ensure ~/Library/Application Support/Bolo/models/chatterbox-turbo-fp16/
        //      model.safetensors exists (WeightLoader will download otherwise — slow).
        //   4. Run the test. If reference outputs are missing the test skips with a
        //      descriptive message rather than failing.

        // 1. Locate reference outputs (skip with diagnostic if not generated).
        let refDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()    // Chatterbox/
            .deletingLastPathComponent()    // BoloTests/
            .deletingLastPathComponent()    // bolo/
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
        let textTokensURL = refDir.appendingPathComponent("text_tokens.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: textTokensURL.path),
            "Reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-reference.py first."
        )

        // 2. Load reference tensors.
        let textTokens = try ReferenceBin.loadInt32(refDir.appendingPathComponent("text_tokens.bin"))
        let speakerEmb = try ReferenceBin.loadFloat32(refDir.appendingPathComponent("t3_cond_spk_emb.bin"))
        let condPromptTokens = try ReferenceBin.loadInt32(
            refDir.appendingPathComponent("t3_cond_prompt_tokens.bin"))
        let pythonInputsEmbeds = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("tfmr_inputs_embeds.bin"))
        let pythonBlock0Out = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("tfmr_block_0_out.bin"))
        let pythonBlock23Out = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("tfmr_block_23_out.bin"))
        let pythonLnFOut = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("tfmr_ln_f_out.bin"))
        let pythonSpeechLogitsFirst = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("speech_logits_first.bin"))

        // 3. Download (no-op if cached) and load model weights.
        let weights = try await WeightLoader.downloadAndLoad { _, _ in }
        print("[parity] loaded \(weights.count) safetensors keys")

        // 4. Build Swift T3 and apply weights.
        let cfg = ChatterboxConfig.turbo.t3
        let t3 = T3(config: cfg)
        let report = T3WeightMapper.apply(weights: weights, to: t3)
        print("[parity] T3WeightMapper: \(report.t3KeyCount) keys applied")
        print("[parity]   unmapped source keys: \(report.unmappedSourceKeys)")
        print("[parity]   unfilled swift keys: \(report.unfilledSwiftKeys)")
        XCTAssertTrue(report.unmappedSourceKeys.isEmpty,
            "All t3.* keys should map to a Swift parameter; got unmapped: \(report.unmappedSourceKeys)")
        XCTAssertTrue(report.unfilledSwiftKeys.isEmpty,
            "All Swift T3 parameters should receive a weight; unfilled: \(report.unfilledSwiftKeys)")

        // 5. Build inputs that mirror Python's prefill.
        let textTokensArr = MLXArray(textTokens.values)
            .reshaped(textTokens.shape).asType(.int32)
        let speakerEmbArr = MLXArray(speakerEmb.values)
            .reshaped(speakerEmb.shape)
        let condPromptArr = MLXArray(condPromptTokens.values)
            .reshaped(condPromptTokens.shape).asType(.int32)
        // Python uses start_speech_token (6561) as the single speech token for prefill.
        let speechStart = MLXArray([Int32(6561)]).reshaped([1, 1])

        // 6. Build inputs_embeds and compare against Python's tfmr_inputs_embeds.
        let (swiftInputsEmbeds, condLen) = t3.prepareInputEmbeds(
            textTokens: textTokensArr,
            speechTokens: speechStart,
            speakerEmbedding: speakerEmbArr,
            condPromptSpeechTokens: condPromptArr
        )
        MLX.eval(swiftInputsEmbeds)
        print("[parity] inputs_embeds: swift=\(swiftInputsEmbeds.shape), python=\(pythonInputsEmbeds.shape), condLen=\(condLen)")
        XCTAssertEqual(swiftInputsEmbeds.shape, pythonInputsEmbeds.shape,
                       "inputs_embeds shapes must match")
        let inputsEmbedsMSE = mseAgainstReference(
            swift: swiftInputsEmbeds, reference: pythonInputsEmbeds)
        print("[parity] inputs_embeds MSE: \(inputsEmbedsMSE)")
        XCTAssertLessThan(inputsEmbedsMSE, 1e-4,
            "inputs_embeds MSE too high (\(inputsEmbedsMSE)); cond_enc / text_emb / speech_emb diverged")

        // 7. Full forward pass via tfmr; capture intermediates by running blocks manually
        //    so we can compare against Python's block 0, 23, ln_f.
        let positions = MLXArray(Int32(0)..<Int32(swiftInputsEmbeds.shape[1]))
        let posEmb = t3.tfmr.wpe(positions)
        var hidden = swiftInputsEmbeds + posEmb
        let mask = T3Attention.causalMask(seqLen: swiftInputsEmbeds.shape[1])
        for (i, block) in t3.tfmr.h.enumerated() {
            hidden = block(hidden, mask: mask, cache: nil)
            if i == 0 || i == t3.tfmr.h.count - 1 {
                MLX.eval(hidden)
                let pyRef = (i == 0) ? pythonBlock0Out : pythonBlock23Out
                let mse = mseAgainstReference(swift: hidden, reference: pyRef)
                let maxAbs = (hidden - mlxFromReference(pyRef)).abs().max().item(Float.self)
                print("[parity] block \(i) — MSE=\(mse), max|diff|=\(maxAbs)")
                // Tolerance grows with depth (numerical compounding); be lenient at depth 23.
                let tol: Float = (i == 0) ? 1e-2 : 5.0
                XCTAssertLessThan(mse, tol,
                    "Block \(i) MSE \(mse) exceeds tolerance \(tol)")
            }
        }
        hidden = t3.tfmr.lnF(hidden)
        MLX.eval(hidden)
        let lnFMSE = mseAgainstReference(swift: hidden, reference: pythonLnFOut)
        print("[parity] ln_f MSE=\(lnFMSE)")
        XCTAssertLessThan(lnFMSE, 1.0,
            "ln_f MSE \(lnFMSE) exceeds tolerance 1.0 (final-norm output should be close)")

        // 8. Apply speech_head to the LAST hidden position and compare logits.
        let lastHidden = hidden[0..., (hidden.shape[1] - 1)..<hidden.shape[1], 0...]
        let swiftLogits = t3.speechHead(lastHidden).reshaped([1, 6563])
        MLX.eval(swiftLogits)
        let logitsMSE = mseAgainstReference(swift: swiftLogits, reference: pythonSpeechLogitsFirst)
        let swiftArgmax = swiftLogits.argMax(axis: -1).item(Int32.self)
        let pythonArgmax = mlxFromReference(pythonSpeechLogitsFirst).argMax(axis: -1).item(Int32.self)
        print("[parity] speech_logits_first MSE=\(logitsMSE)")
        print("[parity]   swift argmax = \(swiftArgmax)")
        print("[parity]   python argmax = \(pythonArgmax)")
        XCTAssertEqual(swiftArgmax, pythonArgmax,
            "Swift T3 must predict the same first speech token as Python.")
    }

    // MARK: - Parity helpers

    private struct ReferenceTensor<T> {
        let values: [T]
        let shape: [Int]
    }

    private enum ReferenceBin {
        static func loadFloat32(_ url: URL) throws -> ReferenceTensor<Float> {
            let shape = try loadShape(url)
            let data = try Data(contentsOf: url)
            let count = data.count / MemoryLayout<Float>.size
            let values = data.withUnsafeBytes { raw -> [Float] in
                let buf = raw.bindMemory(to: Float.self)
                return Array(UnsafeBufferPointer(start: buf.baseAddress, count: count))
            }
            return ReferenceTensor(values: values, shape: shape)
        }
        static func loadInt32(_ url: URL) throws -> ReferenceTensor<Int32> {
            let shape = try loadShape(url)
            let data = try Data(contentsOf: url)
            let count = data.count / MemoryLayout<Int32>.size
            let values = data.withUnsafeBytes { raw -> [Int32] in
                let buf = raw.bindMemory(to: Int32.self)
                return Array(UnsafeBufferPointer(start: buf.baseAddress, count: count))
            }
            return ReferenceTensor(values: values, shape: shape)
        }
        private static func loadShape(_ url: URL) throws -> [Int] {
            let sidecar = url.appendingPathExtension("shape.json")
            let data = try Data(contentsOf: sidecar)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            return json?["shape"] as? [Int] ?? []
        }
    }

    private func mlxFromReference(_ ref: ReferenceTensor<Float>) -> MLXArray {
        MLXArray(ref.values).reshaped(ref.shape)
    }

    private func mseAgainstReference(swift: MLXArray, reference: ReferenceTensor<Float>) -> Float {
        let ref = mlxFromReference(reference)
        let diff = swift - ref
        return (diff * diff).mean().item(Float.self)
    }

    // MARK: - KV cache parity (no model weights)
    //
    // Drives a tiny T3 with random initial weights through both the cached
    // incremental loop and the no-cache full-resequence loop, then asserts the
    // resulting token sequences match exactly. This is integer comparison,
    // so it should be bit-exact, not approximate.
    //
    // Also measures wall time as a sanity signal — cached should be faster,
    // though with random weights and N=20 the difference is modest.
    func test_cache_vs_noCache_producesIdenticalTokens() {
        // Pin RNG so module-init weights are deterministic across the two paths.
        // T3 module init uses MLXRandom for parameter init via MLXNN, so seeding
        // first makes the random "weights" reproducible inside one test run.
        MLXRandom.seed(0xB0_10_DE_AD)

        let cfg = ChatterboxConfig.turbo.t3
        let t3 = T3(config: cfg)

        // Build a tiny pipeline shell with the random-weight T3. The S3Gen
        // half is unused here — wrap a no-op stub by reaching into the
        // private generateSpeechTokens via internal access.
        //
        // We replicate the generation loops inline rather than constructing
        // a ChatterboxPipeline (which needs S3Gen, tokenizer, etc.) so the
        // test stays light.
        let textTokens = MLXArray([42, 17, 8, 99, 3, 11, 200, 88] as [Int32]).reshaped([1, 8])
        let speakerEmb = MLXRandom.normal([1, 256])
        // No cond-prompt-speech-tokens for this synthetic test.
        let startTok: Int32 = 6561
        let stopTok: Int32 = 6562
        let N = 20

        // --- Path A: no cache, O(N²) ---
        let tStart1 = Date()
        var tokensNoCache: [Int32] = []
        do {
            var speechTokens = MLXArray([startTok]).reshaped([1, 1]).asType(.int32)
            for _ in 0..<N {
                let logits = t3(
                    textTokens: textTokens,
                    speechTokens: speechTokens,
                    speakerEmbedding: speakerEmb,
                    condPromptSpeechTokens: nil,
                    caches: nil
                )
                let last = logits[0..., (logits.shape[1] - 1)..<logits.shape[1], 0...]
                    .reshaped([1, 6563])
                let nextID = last.argMax(axis: -1).item(Int32.self)
                if nextID == stopTok { break }
                tokensNoCache.append(nextID)
                let newTok = MLXArray([nextID]).reshaped([1, 1]).asType(.int32)
                speechTokens = concatenated([speechTokens, newTok], axis: 1)
            }
        }
        let noCacheElapsed = Date().timeIntervalSince(tStart1)

        // --- Path B: with KV cache, O(N) ---
        let tStart2 = Date()
        var tokensCached: [Int32] = []
        do {
            let caches: [T3Cache] = (0..<cfg.numLayers).map { _ in
                T3Cache(numHeads: cfg.numHeads, headDim: cfg.headDim)
            }
            // Prefill.
            let speechStart = MLXArray([startTok]).reshaped([1, 1]).asType(.int32)
            let (prefillEmbeds, _) = t3.prepareInputEmbeds(
                textTokens: textTokens,
                speechTokens: speechStart,
                speakerEmbedding: speakerEmb,
                condPromptSpeechTokens: nil
            )
            let prefillLogits = t3.forwardEmbeds(
                inputsEmbeds: prefillEmbeds, cacheOffset: 0, caches: caches)
            var L = prefillEmbeds.shape[1]
            var last = prefillLogits[0..., (prefillLogits.shape[1] - 1)..<prefillLogits.shape[1], 0...]
                .reshaped([1, 6563])
            var nextID = last.argMax(axis: -1).item(Int32.self)
            if nextID != stopTok {
                tokensCached.append(nextID)
                for _ in 1..<N {
                    let newTok = MLXArray([nextID]).reshaped([1, 1]).asType(.int32)
                    let stepEmbed = t3.speechEmb(newTok)
                    let stepLogits = t3.forwardEmbeds(
                        inputsEmbeds: stepEmbed, cacheOffset: L, caches: caches)
                    L += 1
                    last = stepLogits[0..., (stepLogits.shape[1] - 1)..<stepLogits.shape[1], 0...]
                        .reshaped([1, 6563])
                    nextID = last.argMax(axis: -1).item(Int32.self)
                    if nextID == stopTok { break }
                    tokensCached.append(nextID)
                }
            }
        }
        let cachedElapsed = Date().timeIntervalSince(tStart2)

        print("[kv-cache parity] no-cache=\(noCacheElapsed)s cached=\(cachedElapsed)s " +
              "speedup=\(noCacheElapsed / max(cachedElapsed, 1e-9))x " +
              "tokens_no_cache=\(tokensNoCache.count) tokens_cached=\(tokensCached.count)")
        print("[kv-cache parity] no-cache tokens: \(tokensNoCache)")
        print("[kv-cache parity] cached   tokens: \(tokensCached)")

        XCTAssertEqual(tokensCached, tokensNoCache,
            "Cached generation must produce bit-identical token sequence to no-cache reference.")
        XCTAssertFalse(tokensCached.isEmpty, "Test should produce at least one token.")
        // Soft speedup check — should be meaningfully faster, but don't gate the
        // run on a strict ratio (CI noise, cold MLX kernels, etc.).
        XCTAssertLessThan(cachedElapsed, noCacheElapsed,
            "Cached path (\(cachedElapsed)s) should beat no-cache path (\(noCacheElapsed)s).")
    }

    func test_cache_appendingTokens_growsCorrectly() {
        let cfg = ChatterboxConfig.turbo.t3
        let cache = T3Cache(numHeads: cfg.numHeads, headDim: cfg.headDim)

        XCTAssertEqual(cache.sequenceLength, 0)

        // First update: shape (B=1, h=16, S=1, d=64)
        let k1 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
        let v1 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
        let (k1Out, v1Out) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(k1Out.shape, [1, cfg.numHeads, 1, cfg.headDim])
        XCTAssertEqual(v1Out.shape, [1, cfg.numHeads, 1, cfg.headDim])
        XCTAssertEqual(cache.sequenceLength, 1)

        // Second update: cache should now hold both, full shape S=2
        let k2 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
        let v2 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
        let (k2Out, v2Out) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(k2Out.shape, [1, cfg.numHeads, 2, cfg.headDim])
        XCTAssertEqual(v2Out.shape, [1, cfg.numHeads, 2, cfg.headDim])
        XCTAssertEqual(cache.sequenceLength, 2)

        // Reset clears the cache
        cache.reset()
        XCTAssertEqual(cache.sequenceLength, 0)
    }
}
