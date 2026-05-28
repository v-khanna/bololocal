// BoloTests/Chatterbox/ChatterboxPipelineTests.swift
import XCTest
import MLX
import MLXNN
@testable import Bolo

/// End-to-end Phase 5e composition gate.
///
/// Loads the FULL Chatterbox-Turbo pipeline (T3 + S3Gen encoder + CFM + vocoder),
/// feeds in the same speech tokens that Python's reference rerun produced, runs
/// the entire S3Gen pipeline with pinned randomness, and compares the audio
/// output against the Python reference (`e2e_audio.bin`).
///
/// Reference outputs come from `scripts/chatterbox-reference/generate-e2e-reference.py`.
/// The Swift test skips if those bins are absent on disk.
final class ChatterboxPipelineTests: XCTestCase {

    // MARK: - End-to-end parity (the gate)

    func test_pipeline_endToEnd_matchesPythonAudio() async throws {
        let refDir = referenceOutputsDir()
        let sentinel = refDir.appendingPathComponent("e2e_audio.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: sentinel.path),
            "End-to-end reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-e2e-reference.py first."
        )

        // 1. Load reference tensors.
        let speechTokensRef = try ReferenceBin.loadInt32(
            refDir.appendingPathComponent("speech_tokens.bin"))
        let cfmNoiseRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_cfm_noise.bin"))
        let noisedMelsRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_noised_mels.bin"))
        let sineGenPhasesRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_sinegen_phases.bin"))
        let sineGenNoiseRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_sinegen_noise.bin"))
        let speechFeatRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_speech_feat.bin"))
        let audioRef = try ReferenceBin.loadFloat32(
            refDir.appendingPathComponent("e2e_audio.bin"))

        // 2. Materialize tensors. CFM solver takes raw noise + noised_mels;
        //    splicing happens internally (matches `S3Token2Mel.__call__`).
        let cfmNoise = mlxFromFloat(cfmNoiseRef)           // (1, 80, 582)
        let noisedMels = mlxFromFloat(noisedMelsRef)       // (1, 80, 82)
        let speechTokens = MLXArray(speechTokensRef.values)
            .reshaped(speechTokensRef.shape)
            .asType(.int32)
        let sineGenPhases = mlxFromFloat(sineGenPhasesRef)
        let sineGenNoise = mlxFromFloat(sineGenNoiseRef)

        // 3. Load the full pipeline (downloads weights if needed).
        //    Force fp16 — the parity test bit-compares against the Python reference,
        //    so any quantization noise would break it. Production defaults to 4-bit.
        let pipeline = try await ChatterboxPipeline.load(quantizeBits: nil)

        // 4. Run S3Gen with pinned everything.
        let (audioSwift, speechFeatSwift) = pipeline.synthesizeFromSpeechTokens(
            speechTokens: speechTokens,
            pinnedCFMNoise: cfmNoise,
            pinnedNoisedMels: noisedMels,
            pinnedSineGenPhases: sineGenPhases,
            pinnedSineGenNoise: sineGenNoise,
            applyTrimFade: true
        )
        MLX.eval(audioSwift, speechFeatSwift)

        // 5. Stage A — speech_feat parity (verify CFM half landed correctly).
        let speechFeatExpected = mlxFromFloat(speechFeatRef)
        XCTAssertEqual(speechFeatSwift.shape, speechFeatExpected.shape,
            "speech_feat shape mismatch")
        let featMSE = mse(speechFeatSwift, speechFeatExpected)
        let featMax = (speechFeatSwift - speechFeatExpected).abs().max().item(Float.self)
        print("[e2e-parity] speech_feat MSE=\(featMSE), max|diff|=\(featMax)")
        print("[e2e-parity]   swift stats: mean=\(speechFeatSwift.mean().item(Float.self)), " +
              "std=\(stdOf(speechFeatSwift))")
        print("[e2e-parity]   python stats: mean=\(speechFeatExpected.mean().item(Float.self)), " +
              "std=\(stdOf(speechFeatExpected))")
        XCTAssertLessThan(featMSE, 1e-2,
            "speech_feat diverged from Python reference: MSE=\(featMSE)")

        // 6. Stage B — final audio parity.
        let audioExpected = mlxFromFloat(audioRef)
        XCTAssertEqual(audioSwift.shape, audioExpected.shape,
            "audio shape mismatch — Swift=\(audioSwift.shape), Python=\(audioExpected.shape)")
        let audioMSE = mse(audioSwift, audioExpected)
        let audioMax = (audioSwift - audioExpected).abs().max().item(Float.self)
        let swiftStd = stdOf(audioSwift)
        let expStd = stdOf(audioExpected)
        print("[e2e-parity] AUDIO MSE=\(audioMSE), max|diff|=\(audioMax)")
        print("[e2e-parity]   swift std=\(swiftStd), python std=\(expStd)")
        print("[e2e-parity]   swift shape=\(audioSwift.shape), python shape=\(audioExpected.shape)")

        // Save Swift audio for ear test.
        let swiftAudioPath = refDir.appendingPathComponent("e2e_audio_swift.bin")
        do {
            let arr = audioSwift.asArray(Float.self)
            let data = arr.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: swiftAudioPath)
            print("[e2e-parity] Swift audio saved at \(swiftAudioPath.path)")
        } catch {
            print("[e2e-parity] could not save swift audio: \(error)")
        }

        // Per the phase plan: MSE < 1e-2 = exceptional, MSE < 1.0 is acceptable
        // if envelope/std match. The vocoder alone passed at MSE < 1e-1, and
        // we add the CFM on top (which already passes at MSE < 1e-2 in
        // isolation). Compose tolerance: < 1e-1.
        XCTAssertLessThan(audioMSE, 1e-1,
            "End-to-end audio MSE \(audioMSE) exceeds 1e-1 tolerance. " +
            "Likely a plumbing bug between modules.")
    }

    // MARK: - Smoke: pipeline constructs

    func test_pipeline_constructsWithoutWeights() throws {
        // Build pieces in isolation to verify the wiring compiles and shapes
        // make sense. No weight loading, no parity.
        let tokenizer = try EnTokenizer.loadFromBundle()
        let speakerEmbeddings = try SpeakerEmbeddings.loadFromBundle()
        let t3 = T3(config: ChatterboxConfig.turbo.t3)
        let s3gen = S3Gen(config: ChatterboxConfig.turbo.s3gen)
        let pipeline = ChatterboxPipeline(
            tokenizer: tokenizer,
            speakerEmbeddings: speakerEmbeddings,
            t3: t3,
            s3gen: s3gen
        )
        // Just verify we can reach the sub-modules.
        XCTAssertNotNil(pipeline.t3)
        XCTAssertNotNil(pipeline.s3gen)
        XCTAssertNotNil(pipeline.s3gen.decoder)
        XCTAssertNotNil(pipeline.s3gen.mel2wav)
        XCTAssertEqual(pipeline.s3gen.config.tokenEmbeddingDim, 512)
    }

    // MARK: - Reference loaders

    private func referenceOutputsDir() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()    // Chatterbox/
            .deletingLastPathComponent()    // BoloTests/
            .deletingLastPathComponent()    // bolo/
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
    }

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

    private func mlxFromFloat(_ ref: ReferenceTensor<Float>) -> MLXArray {
        MLXArray(ref.values).reshaped(ref.shape)
    }

    private func mse(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = a - b
        return (diff * diff).mean().item(Float.self)
    }

    private func stdOf(_ a: MLXArray) -> Float {
        let m = a.mean()
        let d = a - m
        return MLX.sqrt((d * d).mean()).item(Float.self)
    }
}
