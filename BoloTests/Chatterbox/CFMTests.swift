// BoloTests/Chatterbox/CFMTests.swift
import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Bolo

/// Tests for the `CausalConditionalCFM` Euler ODE solver that wraps the
/// `ConditionalDecoder` to produce mel-like `speech_feat` from noise.
///
/// For Chatterbox-Turbo the solver runs the meanflow path:
///   - n_timesteps = 2, t_span = [0, 0.5, 1.0] (no cosine reshape)
///   - basicEuler: no CFG; decoder takes (t, r) and applies time-mixing
final class CFMTests: XCTestCase {

    // MARK: - Shape / smoke

    /// Pure shape preservation with random (untrained) weights.
    func test_cfm_outputShape() {
        let dec = ConditionalDecoder(meanflow: true)
        let cfm = CausalConditionalCFM(estimator: dec)

        let B = 1, T = 16
        let noise = MLXRandom.normal([B, 80, T])
        let mu = MLXRandom.normal([B, 80, T])
        let mask = MLXArray.ones([B, 1, T])
        let spks = MLXRandom.normal([B, 80])
        let cond = MLXRandom.normal([B, 80, T])

        let out = cfm(
            noise: noise,
            mu: mu,
            mask: mask,
            nTimesteps: 2,
            spks: spks,
            cond: cond,
            meanflow: true
        )
        MLX.eval(out)
        XCTAssertEqual(out.shape, [B, 80, T])
    }

    /// A 1-step Euler solve is equivalent to one decoder call scaled by dt.
    /// With t_span = [0, 1] and meanflow=true:
    ///   x' = noise + (1 - 0) * decoder(noise, mu, t=0, r=1)
    func test_cfm_singleStep_equalsOneDecoderCall() {
        let dec = ConditionalDecoder(meanflow: true)
        let cfm = CausalConditionalCFM(estimator: dec)

        // The decoder is configured for inChannels=320 (= x:80 + mu:80 + spks:80
        // + cond:80). All four inputs are required.
        let B = 1, T = 16
        let noise = MLXRandom.normal([B, 80, T])
        let mu = MLXRandom.normal([B, 80, T])
        let mask = MLXArray.ones([B, 1, T])
        let spks = MLXRandom.normal([B, 80])
        let cond = MLXRandom.normal([B, 80, T])

        let out = cfm(
            noise: noise, mu: mu, mask: mask, nTimesteps: 1,
            spks: spks, cond: cond, meanflow: true
        )
        let t0 = MLXArray([Float(0.0)])
        let r0 = MLXArray([Float(1.0)])
        let v = dec(noise, mask: mask, mu: mu, t: t0, spks: spks, cond: cond, r: r0)
        let expected = noise + (r0 - t0) * v
        MLX.eval(out, expected)

        let diff = (out - expected).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-5,
            "single-step Euler must equal one decoder call: max|diff|=\(diff)")
    }

    // MARK: - Reference parity gate

    /// Loads real Chatterbox-Turbo weights into a meanflow `ConditionalDecoder`,
    /// wraps it in `CausalConditionalCFM`, runs the 2-step Euler solve with
    /// the pinned noise from the Python reference, and compares the resulting
    /// `speech_feat` against the saved reference tensor.
    ///
    /// Gated on the reference outputs existing locally (run
    /// `scripts/chatterbox-reference/generate-reference.py` to produce them).
    func test_cfm_parity_matchesPythonForwardPass() async throws {
        let refDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
        let sentinel = refDir.appendingPathComponent("s3gen_cfm_speech_feat.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: sentinel.path),
            "CFM reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-reference.py first."
        )

        // 1. Load reference tensors.
        let noiseBin = try loadFloat32(refDir.appendingPathComponent("s3gen_cfm_noise.bin"))
        let muBin = try loadFloat32(refDir.appendingPathComponent("s3gen_cfm_mu.bin"))
        let maskBin = try loadFloat32(refDir.appendingPathComponent("s3gen_cfm_mask.bin"))
        let spksBin = try loadFloat32(refDir.appendingPathComponent("s3gen_cfm_spks.bin"))
        let condBin = try loadFloat32(refDir.appendingPathComponent("s3gen_cfm_cond.bin"))
        let speechFeatBin = try loadFloat32(
            refDir.appendingPathComponent("s3gen_cfm_speech_feat.bin"))

        let noise = mlxFromBin(noiseBin)
        let mu = mlxFromBin(muBin)
        let mask = mlxFromBin(maskBin)
        let spks = mlxFromBin(spksBin)
        let cond = mlxFromBin(condBin)
        let speechFeatExpected = mlxFromBin(speechFeatBin)

        // 2. Build meanflow decoder + apply weights (including time_embed_mixer).
        let weights = try await WeightLoader.downloadAndLoad { _, _ in }
        print("[cfm-parity] loaded \(weights.count) safetensors keys")
        let decoder = ConditionalDecoder(meanflow: true)
        let report = DecoderWeightMapper.apply(weights: weights, to: decoder)
        print("[cfm-parity] mapper: total=\(report.sourceKeyCount), " +
              "applied=\(report.appliedKeyCount), skipped=\(report.skippedSourceKeys.count)")
        if !report.unmappedSourceKeys.isEmpty {
            print("[cfm-parity] UNMAPPED source keys (\(report.unmappedSourceKeys.count)): " +
                  "\(report.unmappedSourceKeys.prefix(10)) …")
        }
        if !report.unfilledSwiftKeys.isEmpty {
            print("[cfm-parity] UNFILLED Swift keys (\(report.unfilledSwiftKeys.count)): " +
                  "\(report.unfilledSwiftKeys.prefix(10)) …")
        }
        XCTAssertTrue(report.unmappedSourceKeys.isEmpty,
            "All decoder source keys must map: \(report.unmappedSourceKeys.prefix(20))")
        XCTAssertTrue(report.unfilledSwiftKeys.isEmpty,
            "All Swift decoder parameters must receive a weight: \(report.unfilledSwiftKeys.prefix(20))")

        // 3. Build CFM and run forward.
        let cfm = CausalConditionalCFM(estimator: decoder)
        let speechFeatSwift = cfm(
            noise: noise,
            mu: mu,
            mask: mask,
            nTimesteps: 2,
            spks: spks,
            cond: cond,
            meanflow: true
        )
        MLX.eval(speechFeatSwift)

        // 4. Compare against Python reference.
        XCTAssertEqual(speechFeatSwift.shape, speechFeatExpected.shape,
            "speech_feat shape mismatch")
        let mseVal = mse(speechFeatSwift, speechFeatExpected)
        let maxDiff = (speechFeatSwift - speechFeatExpected).abs().max().item(Float.self)
        let swiftMean = speechFeatSwift.mean().item(Float.self)
        let swiftStd = (speechFeatSwift - swiftMean).square().mean().sqrt().item(Float.self)
        let expMean = speechFeatExpected.mean().item(Float.self)
        let expStd = (speechFeatExpected - expMean).square().mean().sqrt().item(Float.self)
        print("[cfm-parity] speech_feat MSE=\(mseVal), max|diff|=\(maxDiff)")
        print("[cfm-parity] Swift  mean=\(swiftMean) std=\(swiftStd)")
        print("[cfm-parity] Python mean=\(expMean) std=\(expStd)")

        // Tolerance: CFM accumulates Euler-step floating-point error. The
        // decoder alone parity sat at MSE ~6e-6, and we do 2 steps with
        // additional time-mixing; the bar is 1e-2 per the phase plan.
        XCTAssertLessThan(mseVal, 1e-2,
            "speech_feat parity exceeds tolerance: MSE=\(mseVal)")
    }

    // MARK: - Test helpers (binary-tensor loaders shared with other tests)

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

    private func mse(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = a - b
        return (diff * diff).mean().item(Float.self)
    }
}
