// BoloTests/Chatterbox/ConditionalDecoderTests.swift
import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Bolo

final class ConditionalDecoderTests: XCTestCase {

    // MARK: - Sinusoidal positional embedding

    func test_sinusoidalPosEmb_outputShape() {
        let t = MLXArray([Float(0.0), 0.5, 1.0])
        let emb = sinusoidalPosEmb(t, dim: 320)
        MLX.eval(emb)
        XCTAssertEqual(emb.shape, [3, 320])
    }

    func test_sinusoidalPosEmb_zeroTimestepIsCanonical() {
        // sin(0) = 0, cos(0) = 1. The concatenation should produce
        // [0, 0, …, 0, 1, 1, …, 1].
        let t = MLXArray([Float(0.0)])
        let emb = sinusoidalPosEmb(t, dim: 8)
        MLX.eval(emb)
        let v = emb[0].asArray(Float.self)
        XCTAssertEqual(v.count, 8)
        for i in 0..<4 { XCTAssertEqual(v[i], 0.0, accuracy: 1e-6) }
        for i in 4..<8 { XCTAssertEqual(v[i], 1.0, accuracy: 1e-6) }
    }

    // MARK: - TimestepEmbedding

    func test_timestepEmbedding_outputShape() {
        let mlp = TimestepEmbedding(inChannels: 320, timeEmbedDim: 1024)
        let x = MLXRandom.normal([2, 320])
        let y = mlp(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [2, 1024])
    }

    // MARK: - Conv1dPT

    func test_conv1dPT_acceptsBCT() {
        let c = Conv1dPT(inputChannels: 4, outputChannels: 8, kernelSize: 3, padding: 1)
        let x = MLXRandom.normal([1, 4, 10])  // (B, C, T)
        let y = c(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 8, 10])
    }

    // MARK: - CausalConv1d

    func test_causalConv1d_preservesLength() {
        let c = CausalConv1d(inputChannels: 4, outputChannels: 8, kernelSize: 3)
        let x = MLXRandom.normal([1, 4, 12])
        let y = c(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 8, 12])
    }

    // MARK: - CausalBlock1D

    func test_causalBlock1D_preservesLengthAndAppliesMask() {
        let b = CausalBlock1D(dim: 16, dimOut: 16)
        let x = MLXRandom.normal([1, 16, 8])
        let mask = MLXArray.ones([1, 1, 8])
        let y = b(x, mask: mask)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 16, 8])
    }

    // MARK: - ResnetBlock1D

    func test_resnetBlock1D_outputShape() {
        let r = ResnetBlock1D(dim: 320, dimOut: 256, timeEmbDim: 1024)
        let x = MLXRandom.normal([1, 320, 10])
        let mask = MLXArray.ones([1, 1, 10])
        let tEmb = MLXRandom.normal([1, 1024])
        let y = r(x, mask: mask, timeEmb: tEmb)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 256, 10])
    }

    // MARK: - SelfAttention1D

    func test_selfAttention1D_outputShape() {
        let attn = SelfAttention1D(dim: 256, numHeads: 8, headDim: 64)
        let x = MLXRandom.normal([1, 16, 256])
        let mask = MLXArray.ones([1, 16])
        let y = attn(x, mask: mask)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 16, 256])
    }

    // MARK: - FeedForward

    func test_feedForward_outputShape() {
        let ff = FeedForward(dim: 256, mult: 4)
        let x = MLXRandom.normal([1, 16, 256])
        let y = ff(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 16, 256])
    }

    // MARK: - TransformerBlock

    func test_transformerBlock_outputShape() {
        let block = TransformerBlock(dim: 256)
        let x = MLXRandom.normal([1, 16, 256])
        let mask = MLXArray.ones([1, 16])
        let y = block(x, mask: mask)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 16, 256])
    }

    // MARK: - DownBlock / MidBlock / UpBlock

    func test_downBlock_lastIsLengthPreserving() {
        let block = DownBlock(
            inputChannel: 320,
            outputChannel: 256,
            timeEmbedDim: 1024,
            causal: true,
            nBlocks: 4,
            numHeads: 8,
            attentionHeadDim: 64,
            isLast: true
        )
        // Just exercise the resnet + downsample paths through the helper.
        let x = MLXRandom.normal([1, 320, 8])
        let mask = MLXArray.ones([1, 1, 8])
        let tEmb = MLXRandom.normal([1, 1024])
        var y = block.resnet(x, mask: mask, timeEmb: tEmb)
        XCTAssertEqual(y.shape, [1, 256, 8])
        // downsample (CausalConv1d on last) preserves length
        y = block.applyDownsample(y)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 256, 8])
    }

    func test_midBlock_outputShape() {
        let block = MidBlock(
            channels: 256,
            timeEmbedDim: 1024,
            causal: true,
            nBlocks: 4,
            numHeads: 8,
            attentionHeadDim: 64
        )
        let x = MLXRandom.normal([1, 256, 8])
        let mask = MLXArray.ones([1, 1, 8])
        let tEmb = MLXRandom.normal([1, 1024])
        let y = block.resnet(x, mask: mask, timeEmb: tEmb)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 256, 8])
    }

    // MARK: - ConditionalDecoder shape test

    func test_conditionalDecoder_outputShape() {
        let dec = ConditionalDecoder()
        let B = 1, T = 16
        let x = MLXRandom.normal([B, 80, T])
        let mask = MLXArray.ones([B, 1, T])
        let mu = MLXRandom.normal([B, 80, T])
        let t = MLXArray([Float(0.0)])
        let spks = MLXRandom.normal([B, 80])
        let cond = MLXRandom.normal([B, 80, T])
        let y = dec(x, mask: mask, mu: mu, t: t, spks: spks, cond: cond)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [B, 80, T])
    }

    // MARK: - Reference parity gate

    /// Loads the real Chatterbox-Turbo weights into the Swift
    /// `ConditionalDecoder` and runs the forward pass on the same
    /// `(x_t, mask, mu, t, spks, cond)` triple captured by the Python
    /// reference script, then compares against the saved velocity field.
    func test_conditionalDecoder_parity_matchesPythonForwardPass() async throws {
        let refDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
        let sentinel = refDir.appendingPathComponent("s3gen_decoder_velocity_out.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: sentinel.path),
            "Decoder reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-reference.py first."
        )

        // 1. Load reference tensors.
        let xt = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_x_t.bin"))
        let mu = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_mu.bin"))
        let mask = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_mask.bin"))
        let spks = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_spks.bin"))
        let cond = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_cond.bin"))
        let tval = try loadFloat32(refDir.appendingPathComponent("s3gen_decoder_t.bin"))
        let velocityRef = try loadFloat32(
            refDir.appendingPathComponent("s3gen_decoder_velocity_out.bin"))

        // 2. Build decoder and apply weights.
        let weights = try await WeightLoader.downloadAndLoad { _, _ in }
        print("[decoder-parity] loaded \(weights.count) safetensors keys")
        let decoder = ConditionalDecoder()
        let report = DecoderWeightMapper.apply(weights: weights, to: decoder)
        print("[decoder-parity] mapper: total=\(report.sourceKeyCount), " +
              "applied=\(report.appliedKeyCount), skipped=\(report.skippedSourceKeys.count)")
        if !report.unmappedSourceKeys.isEmpty {
            print("[decoder-parity] UNMAPPED source keys (\(report.unmappedSourceKeys.count)): " +
                  "\(report.unmappedSourceKeys.prefix(10)) …")
        }
        if !report.unfilledSwiftKeys.isEmpty {
            print("[decoder-parity] UNFILLED Swift keys (\(report.unfilledSwiftKeys.count)): " +
                  "\(report.unfilledSwiftKeys.prefix(10)) …")
        }
        XCTAssertTrue(report.unmappedSourceKeys.isEmpty,
            "All decoder source keys must map: \(report.unmappedSourceKeys.prefix(20))")
        XCTAssertTrue(report.unfilledSwiftKeys.isEmpty,
            "All Swift decoder parameters must receive a weight: \(report.unfilledSwiftKeys.prefix(20))")

        // 3. Build input arrays.
        let xtArr = mlxFromBin(xt)
        let muArr = mlxFromBin(mu)
        let maskArr = mlxFromBin(mask)
        let spksArr = mlxFromBin(spks)
        let condArr = mlxFromBin(cond)
        let tArr = mlxFromBin(tval)
        let velocityExpected = mlxFromBin(velocityRef)

        // 4. Run Swift forward.
        let velocitySwift = decoder(
            xtArr, mask: maskArr, mu: muArr, t: tArr, spks: spksArr, cond: condArr
        )
        MLX.eval(velocitySwift)

        // 5. Compare.
        XCTAssertEqual(velocitySwift.shape, velocityExpected.shape,
            "velocity shape mismatch: swift=\(velocitySwift.shape), expected=\(velocityExpected.shape)")
        let diff = velocitySwift - velocityExpected
        let mseVal = (diff * diff).mean().item(Float.self)
        let maxAbs = diff.abs().max().item(Float.self)
        print("[decoder-parity] velocity MSE = \(mseVal), max|diff| = \(maxAbs)")
        // The decoder is very deep (1 down + 12 mid + 1 up, each with a
        // resnet + 4 transformer blocks). Accumulation of fp16 storage error
        // limits how tight we can be — 1e-3 is the announced target.
        XCTAssertLessThan(mseVal, 1e-3,
            "ConditionalDecoder velocity MSE \(mseVal) exceeds 1e-3 tolerance")
    }

    // MARK: - Reference loader

    private struct BinTensor {
        let values: [Float]
        let shape: [Int]
    }

    private func loadFloat32(_ url: URL) throws -> BinTensor {
        let sidecar = url.appendingPathExtension("shape.json")
        let metaData = try Data(contentsOf: sidecar)
        let json = try JSONSerialization.jsonObject(with: metaData) as? [String: Any]
        let shape = json?["shape"] as? [Int] ?? []
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        let values = data.withUnsafeBytes { raw -> [Float] in
            let buf = raw.bindMemory(to: Float.self)
            return Array(UnsafeBufferPointer(start: buf.baseAddress, count: count))
        }
        return BinTensor(values: values, shape: shape)
    }

    private func mlxFromBin(_ t: BinTensor) -> MLXArray {
        return MLXArray(t.values).reshaped(t.shape)
    }
}
