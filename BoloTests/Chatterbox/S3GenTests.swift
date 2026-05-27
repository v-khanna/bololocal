// BoloTests/Chatterbox/S3GenTests.swift
import XCTest
import MLX
import MLXRandom
@testable import Bolo

final class S3GenTests: XCTestCase {

    // MARK: - Shape tests (no weights needed)

    func test_relPositionalEncoding_returns2TminusOneRows() {
        let pe = EspnetRelPositionalEncoding(dModel: 64, maxLen: 32)
        let x = MLXRandom.normal([1, 10, 64])
        let (scaled, posEmb) = pe(x)
        MLX.eval(scaled, posEmb)
        XCTAssertEqual(scaled.shape, [1, 10, 64])
        // 2*T - 1 = 19 rows
        XCTAssertEqual(posEmb.shape, [1, 19, 64])
    }

    func test_relPositionalEncoding_growsWhenSequenceExceedsCache() {
        let pe = EspnetRelPositionalEncoding(dModel: 32, maxLen: 4)
        // Below cached size — fast path
        let small = MLXRandom.normal([1, 3, 32])
        let (_, smallPos) = pe(small)
        XCTAssertEqual(smallPos.shape, [1, 5, 32])
        // Exceed cached size — must extend transparently
        let big = MLXRandom.normal([1, 16, 32])
        let (_, bigPos) = pe(big)
        XCTAssertEqual(bigPos.shape, [1, 31, 32])
    }

    func test_linearInput_outputShape() {
        let m = LinearInput(inputSize: 512, outputSize: 512)
        let x = MLXRandom.normal([1, 20, 512])
        let mask = MLXArray.ones([1, 1, 20])
        let (y, posEmb, m2) = m(x, mask: mask)
        MLX.eval(y, posEmb)
        XCTAssertEqual(y.shape, [1, 20, 512])
        XCTAssertEqual(posEmb.shape, [1, 39, 512])
        XCTAssertEqual(m2.shape, [1, 1, 20])
    }

    func test_relPosMHA_outputShape() {
        let mha = RelPositionMultiHeadedAttention(numHeads: 8, dModel: 512)
        let x = MLXRandom.normal([1, 20, 512])
        // Build a valid pos_emb by reusing the encoding
        let pe = EspnetRelPositionalEncoding(dModel: 512)
        let (_, posEmb) = pe(x)
        let mask = MLXArray.ones([1, 20])
        let y = mha(x, mask: mask, posEmb: posEmb)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 20, 512])
    }

    func test_conformerLayer_preservesShape() {
        let layer = ConformerEncoderLayer(size: 512, numHeads: 8, dInner: 2048)
        let x = MLXRandom.normal([1, 20, 512])
        let pe = EspnetRelPositionalEncoding(dModel: 512)
        let (_, posEmb) = pe(x)
        let mask = MLXArray.ones([1, 20])
        let y = layer(x, mask: mask, posEmb: posEmb)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 20, 512])
    }

    func test_preLookaheadLayer_preservesShape() {
        let pl = PreLookaheadLayer(channels: 512, preLookaheadLen: 3)
        let x = MLXRandom.normal([1, 20, 512])
        let y = pl(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 20, 512])
    }

    func test_upsample1D_doublesTimeAxis() {
        let up = Upsample1DEncoder(channels: 512, stride: 2)
        let x = MLXRandom.normal([1, 20, 512])
        let lens = MLXArray([Int32(20)])
        let (y, newLens) = up(x, xLens: lens)
        MLX.eval(y, newLens)
        XCTAssertEqual(y.shape, [1, 40, 512])
        XCTAssertEqual(newLens.item(Int32.self), 40)
    }

    func test_upsampleConformerEncoder_outputShape() {
        let cfg = ChatterboxConfig.turbo.s3gen
        let enc = UpsampleConformerEncoder(config: cfg)
        let x = MLXRandom.normal([1, 10, cfg.tokenEmbeddingDim])
        let lens = MLXArray([Int32(10)])
        let (y, mask) = enc(x, xsLens: lens)
        MLX.eval(y, mask)
        XCTAssertEqual(y.shape, [1, 20, cfg.tokenEmbeddingDim])
        XCTAssertEqual(mask.shape, [1, 1, 20])
    }

    func test_s3gen_encodeForDecoder_outputShape() {
        let cfg = ChatterboxConfig.turbo.s3gen
        let s3gen = S3Gen(config: cfg)
        let speechTokens = MLXArray((0..<5).map { Int32($0 % 1000) }).reshaped([1, 5])
        let promptToken = MLXArray((0..<7).map { Int32($0 % 1000) }).reshaped([1, 7])
        let promptTokenLen = MLXArray([Int32(7)])
        let speakerXVector = MLXRandom.normal([1, ChatterboxConfig.speakerEmbeddingDim])
        let out = s3gen.encodeForDecoder(
            speechTokens: speechTokens,
            promptToken: promptToken,
            promptTokenLen: promptTokenLen,
            speakerXVector: speakerXVector
        )
        MLX.eval(out.encoderOut, out.encoderProjOut, out.encoderMask, out.speakerEmbedding)
        // Combined token length 12 → encoder upsamples to 24
        XCTAssertEqual(out.encoderOut.shape, [1, 24, 512])
        XCTAssertEqual(out.encoderProjOut.shape, [1, 24, 80])
        XCTAssertEqual(out.encoderMask.shape, [1, 1, 24])
        XCTAssertEqual(out.speakerEmbedding.shape, [1, 80])
    }

    // MARK: - Reference parity gate (heavy)

    /// Phase 5 parity gate. Loads real Chatterbox-Turbo weights into the Swift
    /// S3Gen and runs the deterministic encoder pipeline on the same speech
    /// tokens the Python reference used, then compares against the captured
    /// reference outputs.
    ///
    /// Gated behind reference outputs existing on disk:
    ///   1. cd scripts/chatterbox-reference && source venv/bin/activate
    ///   2. python generate-reference.py
    ///   3. Ensure model.safetensors is cached at
    ///      ~/Library/Application Support/Bolo/models/chatterbox-turbo-fp16/.
    ///
    /// The encoder is fully deterministic — no random sampling between the
    /// safetensors and `encoder_proj` — so we expect MSE on the order of 1e-5
    /// or tighter (limited only by the fp16 storage of the safetensors).
    func test_s3gen_encoderParity_matchesPythonForwardPass() async throws {
        let refDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
        let sentinel = refDir.appendingPathComponent("s3gen_encoder_out.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: sentinel.path),
            "S3Gen reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-reference.py first."
        )

        // 1. Load reference tensors.
        let speechTokensRef = try ReferenceBin.loadInt32(
            refDir.appendingPathComponent("speech_tokens.bin"))
        let promptTokenRef = try ReferenceBin.loadInt32(
            refDir.appendingPathComponent("s3gen_gen_prompt_token.bin"))
        let promptTokenLenRef = try ReferenceBin.loadInt32(
            refDir.appendingPathComponent("s3gen_gen_prompt_token_len.bin"))
        let xvectorRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("s3gen_gen_embedding.bin"))
        let tokenEmbRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("s3gen_encoder_token_emb.bin"))
        let encoderOutRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("s3gen_encoder_out.bin"))
        let encoderProjOutRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("s3gen_encoder_proj_out.bin"))
        let spkAffineRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("s3gen_spk_embed_affine_out.bin"))

        // 2. Load model weights and build Swift S3Gen.
        let weights = try await WeightLoader.downloadAndLoad { _, _ in }
        print("[s3gen-parity] loaded \(weights.count) safetensors keys")
        let cfg = ChatterboxConfig.turbo.s3gen
        let s3gen = S3Gen(config: cfg)
        let report = S3GenWeightMapper.apply(weights: weights, to: s3gen)
        print("[s3gen-parity] mapper: total=\(report.s3genKeyCount), applied=\(report.appliedKeyCount), skipped=\(report.skippedSourceKeys.count)")
        if !report.unmappedSourceKeys.isEmpty {
            print("[s3gen-parity] UNMAPPED source keys: \(report.unmappedSourceKeys)")
        }
        if !report.unfilledSwiftKeys.isEmpty {
            print("[s3gen-parity] UNFILLED Swift keys: \(report.unfilledSwiftKeys)")
        }
        XCTAssertTrue(report.unmappedSourceKeys.isEmpty,
            "All keys destined for ported modules must map: \(report.unmappedSourceKeys)")
        XCTAssertTrue(report.unfilledSwiftKeys.isEmpty,
            "All Swift parameters must receive a weight: \(report.unfilledSwiftKeys)")

        // 3. Build inputs.
        let speechTokens = MLXArray(speechTokensRef.values).reshaped(speechTokensRef.shape).asType(.int32)
        let promptToken = MLXArray(promptTokenRef.values).reshaped(promptTokenRef.shape).asType(.int32)
        let promptTokenLen = MLXArray(promptTokenLenRef.values).reshaped(promptTokenLenRef.shape).asType(.int32)
        let xvector = MLXArray(xvectorRef.values).reshaped(xvectorRef.shape)

        // 4. Run encoder pipeline.
        let out = s3gen.encodeForDecoder(
            speechTokens: speechTokens,
            promptToken: promptToken,
            promptTokenLen: promptTokenLen,
            speakerXVector: xvector
        )
        MLX.eval(out.encoderOut, out.encoderProjOut, out.speakerEmbedding)

        // 5. Compare against references.
        let tokenEmbExpected = mlxFromReference(tokenEmbRef)
        // Recompute Swift's token_emb internally for an intermediate check
        let token = concatenated([promptToken, speechTokens], axis: 1)
        let tokenLen = (promptTokenLen + MLXArray([Int32(speechTokens.shape[1])])).asType(.int32)
        let arange = MLXArray(Int32(0)..<Int32(token.shape[1]))
        let mask = (arange.expandedDimensions(axis: 0) .< tokenLen.expandedDimensions(axis: 1)).asType(.float32)
        let mask3d = mask.expandedDimensions(axis: -1)
        let tokenEmbSwift = s3gen.inputEmbedding(token) * mask3d
        MLX.eval(tokenEmbSwift)
        let tokenEmbMSE = mse(tokenEmbSwift, tokenEmbExpected)
        print("[s3gen-parity] token_emb MSE = \(tokenEmbMSE)")
        XCTAssertLessThan(tokenEmbMSE, 1e-5,
            "input_embedding diverged from Python: MSE=\(tokenEmbMSE)")

        // Speaker affine parity
        let spkAffineExpected = mlxFromReference(spkAffineRef)
        let spkAffineMSE = mse(out.speakerEmbedding, spkAffineExpected)
        print("[s3gen-parity] spk_embed_affine MSE = \(spkAffineMSE)")
        XCTAssertLessThan(spkAffineMSE, 1e-5,
            "spk_embed_affine_layer diverged: MSE=\(spkAffineMSE)")

        // Encoder output
        let encOutExpected = mlxFromReference(encoderOutRef)
        XCTAssertEqual(out.encoderOut.shape, encOutExpected.shape,
                       "encoder_out shape mismatch")
        let encOutMSE = mse(out.encoderOut, encOutExpected)
        let encOutMax = (out.encoderOut - encOutExpected).abs().max().item(Float.self)
        print("[s3gen-parity] encoder_out MSE = \(encOutMSE), max|diff| = \(encOutMax)")
        // Encoder is deterministic but compounds error across 10 conformer blocks.
        // Tolerance of 1e-3 is generous; tighter values may be achievable.
        XCTAssertLessThan(encOutMSE, 1e-3,
            "UpsampleConformerEncoder output MSE \(encOutMSE) exceeds 1e-3")

        // Encoder proj output
        let encProjExpected = mlxFromReference(encoderProjOutRef)
        let encProjMSE = mse(out.encoderProjOut, encProjExpected)
        print("[s3gen-parity] encoder_proj_out MSE = \(encProjMSE)")
        XCTAssertLessThan(encProjMSE, 1e-3,
            "encoder_proj output MSE \(encProjMSE) exceeds 1e-3")
    }

    // MARK: - Reference loader

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

    private func mse(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = a - b
        return (diff * diff).mean().item(Float.self)
    }
}
