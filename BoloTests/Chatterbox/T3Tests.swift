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
}
