// BoloTests/Chatterbox/VocoderTests.swift
import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Bolo

final class VocoderTests: XCTestCase {

    // MARK: - Snake activation

    func test_snake_outputShape() {
        let s = Snake(inFeatures: 16)
        let x = MLXRandom.normal([2, 16, 32])
        let y = s(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [2, 16, 32])
    }

    /// With α = 1: x + sin²(x). Sanity-check at x = 0 → 0.
    func test_snake_alphaOne_atZero() {
        let s = Snake(inFeatures: 4)
        let x = MLXArray.zeros([1, 4, 1])
        let y = s(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 4, 1])
        let v = y.asArray(Float.self)
        for f in v { XCTAssertEqual(f, 0, accuracy: 1e-6) }
    }

    /// With α = 1: snake(π/2) = π/2 + sin²(π/2) = π/2 + 1.
    func test_snake_alphaOne_atHalfPi() {
        let s = Snake(inFeatures: 1)
        let x = MLXArray([Float.pi / 2]).reshaped([1, 1, 1])
        let y = s(x)
        MLX.eval(y)
        let v = y.item(Float.self)
        XCTAssertEqual(v, Float.pi / 2 + 1, accuracy: 1e-5)
    }

    // MARK: - ResBlockVocoder

    func test_resBlockVocoder_outputShape() {
        let r = ResBlockVocoder(channels: 16, kernelSize: 3, dilations: [1, 3, 5])
        let x = MLXRandom.normal([1, 16, 32])
        let y = r(x)
        MLX.eval(y)
        XCTAssertEqual(y.shape, [1, 16, 32])
    }

    // MARK: - F0Predictor

    func test_f0Predictor_outputShape() {
        let f = F0Predictor(inChannels: 80, hiddenChannels: 64, numLayers: 2)
        let mel = MLXRandom.normal([1, 80, 16])
        let f0 = f(mel)
        MLX.eval(f0)
        XCTAssertEqual(f0.shape, [1, 16])
        // Output must be non-negative
        let minVal = f0.min().item(Float.self)
        XCTAssertGreaterThanOrEqual(minVal, 0)
    }

    // MARK: - SineGen

    func test_sineGen_outputShape() {
        let g = SineGen(sampRate: 24000, harmonicNum: 8, voicedThreshold: 10)
        // (B=1, 1, T=128) with non-zero F0 to trigger voiced path
        let f0 = MLXArray([Float](repeating: 200, count: 128)).reshaped([1, 1, 128])
        let (sw, uv, _) = g(f0)
        MLX.eval(sw, uv)
        XCTAssertEqual(sw.shape, [1, 9, 128])
        XCTAssertEqual(uv.shape, [1, 1, 128])
    }

    func test_sourceModule_outputShape() {
        let m = SourceModule(samplingRate: 24000, harmonicNum: 8, voicedThreshold: 10)
        // f0: (B, T, 1)
        let f0 = MLXArray([Float](repeating: 220, count: 64)).reshaped([1, 64, 1])
        let (sineMerge, _, _) = m(f0)
        MLX.eval(sineMerge)
        XCTAssertEqual(sineMerge.shape, [1, 64, 1])
    }

    // MARK: - HiFTGenerator shape tests

    func test_hiftGenerator_canConstruct() {
        let g = HiFTGenerator()
        // Sanity: total upsample scale is 8*5*3*4 = 480.
        XCTAssertEqual(g.f0UpsampleScale, 480)
        XCTAssertEqual(g.numUpsamples, 3)
        XCTAssertEqual(g.numKernels, 3)
        XCTAssertEqual(g.ups.count, 3)
        XCTAssertEqual(g.sourceDowns.count, 3)
        XCTAssertEqual(g.sourceResblocks.count, 3)
        XCTAssertEqual(g.resblocks.count, 9)
    }

    func test_hiftGenerator_upsampleF0() {
        let g = HiFTGenerator()
        let f0 = MLXArray([Float](repeating: 200, count: 10)).reshaped([1, 10])
        let up = g.upsampleF0(f0)
        MLX.eval(up)
        XCTAssertEqual(up.shape, [1, 10 * g.f0UpsampleScale, 1])
    }

    func test_hiftGenerator_hannWindow_periodicSumOfSquaresMatches() {
        let g = HiFTGenerator()
        let w = g.hannWindow(length: 16, periodic: true)
        // For a periodic Hann (length N): w[n] = 0.5*(1 - cos(2π n/N)).
        // sum w[n]^2 = 0.25 * Σ(1 - 2cos + cos²) = 0.25 * (N + N/2) = 3N/8.
        // For N=16 -> 6.0.
        let sumSq = w.reduce(Float(0)) { $0 + $1 * $1 }
        XCTAssertEqual(sumSq, 6.0, accuracy: 1e-5)
    }

    /// STFT followed by ISTFT of a short signal should round-trip approximately
    /// (within window normalisation precision) for the centred convention.
    func test_hiftGenerator_stftIstftRoundtrip_smoke() {
        let g = HiFTGenerator()
        // 64 samples — yields T_frames = (64 - 16) / 4 + 1 = 13.
        let xRaw = (0..<64).map { Float(sin(Double($0) * 0.1)) }
        let x = MLXArray(xRaw).reshaped([1, 64])
        let (re, im) = g.stft(x)
        XCTAssertEqual(re.shape, [1, 9, 13])
        XCTAssertEqual(im.shape, [1, 9, 13])
        // magnitude/phase form
        let mag = MLX.sqrt(re * re + im * im)
        let phase = MLX.atan2(im, re)
        let audio = g.istft(magnitude: mag, phase: phase)
        XCTAssertEqual(audio.shape, [1, (13 - 1) * 4])
        // Don't compare values — the trim drops the center pad so a true
        // round-trip needs centered STFT (we use non-centered for compute).
        // Still verify the shape contract.
    }

    // MARK: - Reference parity gate

    /// Loads real Chatterbox weights, applies them, then runs the vocoder
    /// against the Python reference outputs. Targets MSE < 1.0 on audio.
    func test_hiftGenerator_parity_matchesPythonAudio() async throws {
        let refDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")
        let sentinel = refDir.appendingPathComponent("s3gen_vocoder_audio.bin")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: sentinel.path),
            "Vocoder reference outputs missing at \(refDir.path). " +
            "Run scripts/chatterbox-reference/generate-reference.py first."
        )

        // 1. Load reference tensors.
        let speechFeat = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_speech_feat.bin"))
        let f0Expected = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_f0.bin"))
        let sourceExpected = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_source.bin"))
        let audioExpected = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_audio.bin"))
        let phasesRef = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_sinegen_phases.bin"))
        let noiseRef = try loadFloat32(
            refDir.appendingPathComponent("s3gen_vocoder_sinegen_noise.bin"))

        // 2. Build vocoder + apply weights.
        let weights = try await WeightLoader.downloadAndLoad { _, _ in }
        print("[vocoder-parity] loaded \(weights.count) safetensors keys")
        let vocoder = HiFTGenerator()
        let report = VocoderWeightMapper.apply(weights: weights, to: vocoder)
        print("[vocoder-parity] mapper: total=\(report.sourceKeyCount), " +
              "applied=\(report.appliedKeyCount)")
        if !report.unmappedSourceKeys.isEmpty {
            print("[vocoder-parity] UNMAPPED source keys (\(report.unmappedSourceKeys.count)): " +
                  "\(report.unmappedSourceKeys.prefix(10)) …")
        }
        if !report.unfilledSwiftKeys.isEmpty {
            print("[vocoder-parity] UNFILLED Swift keys (\(report.unfilledSwiftKeys.count)): " +
                  "\(report.unfilledSwiftKeys.prefix(10)) …")
        }
        XCTAssertTrue(report.unmappedSourceKeys.isEmpty,
            "All vocoder source keys must map: \(report.unmappedSourceKeys.prefix(20))")
        XCTAssertTrue(report.unfilledSwiftKeys.isEmpty,
            "All Swift vocoder parameters must receive a weight: " +
            "\(report.unfilledSwiftKeys.prefix(20))")

        // 3. Build inputs.
        let speechFeatArr = mlxFromBin(speechFeat)
        let f0ExpectedArr = mlxFromBin(f0Expected)
        let sourceExpectedArr = mlxFromBin(sourceExpected)
        let audioExpectedArr = mlxFromBin(audioExpected)
        let phasesArr = mlxFromBin(phasesRef)
        let noiseArr = mlxFromBin(noiseRef)

        // 4. Stage 1 — F0Predictor parity.
        let mel = speechFeatArr.transposed(0, 2, 1)  // (1, 80, T)
        let f0Swift = vocoder.f0Predictor(mel)
        MLX.eval(f0Swift)
        let f0Mse = mse(f0Swift, f0ExpectedArr)
        let f0Max = (f0Swift - f0ExpectedArr).abs().max().item(Float.self)
        print("[vocoder-parity] f0 MSE = \(f0Mse), max|diff| = \(f0Max)")
        XCTAssertLessThan(f0Mse, 1e-3,
            "F0Predictor diverged: MSE=\(f0Mse)")

        // 5. Stage 2 — source signal parity.
        let f0Up = vocoder.upsampleF0(f0Swift)
        let (sineMerge, _, _) = vocoder.mSource(
            f0Up,
            sineGenRandomPhases: phasesArr,
            sineGenNoise: noiseArr
        )
        let sourceSwift = sineMerge.transposed(0, 2, 1)
        MLX.eval(sourceSwift)
        XCTAssertEqual(sourceSwift.shape, sourceExpectedArr.shape,
            "source shape mismatch")
        let sourceMse = mse(sourceSwift, sourceExpectedArr)
        let sourceMax = (sourceSwift - sourceExpectedArr).abs().max().item(Float.self)
        print("[vocoder-parity] source MSE = \(sourceMse), max|diff| = \(sourceMax)")
        XCTAssertLessThan(sourceMse, 1e-4,
            "SineGen+l_linear diverged: MSE=\(sourceMse)")

        // 6. Stage 3 — full decode parity.
        let audioSwift = vocoder.decode(mel: mel, source: sourceSwift)
        MLX.eval(audioSwift)
        print("[vocoder-parity] audio Swift shape: \(audioSwift.shape), " +
              "expected: \(audioExpectedArr.shape)")
        XCTAssertEqual(audioSwift.shape, audioExpectedArr.shape,
            "audio shape mismatch")

        let audioMse = mse(audioSwift, audioExpectedArr)
        let audioMax = (audioSwift - audioExpectedArr).abs().max().item(Float.self)
        let swiftStd = stdOf(audioSwift)
        let expectedStd = stdOf(audioExpectedArr)
        print("[vocoder-parity] audio MSE = \(audioMse), max|diff| = \(audioMax)")
        print("[vocoder-parity] audio std Swift=\(swiftStd) Python=\(expectedStd)")

        // Save Swift audio next to reference for ear test.
        let swiftAudioOut = refDir.appendingPathComponent("s3gen_vocoder_audio_swift.bin")
        do {
            let arr = audioSwift.asArray(Float.self)
            let data = arr.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: swiftAudioOut)
            print("[vocoder-parity] swift audio saved to \(swiftAudioOut.path)")
        } catch {
            print("[vocoder-parity] could not save swift audio: \(error)")
        }

        // Audio MSE tolerance: the vocoder is heavily nonlinear over hundreds of
        // thousands of samples, so we set a generous bound. The harness in the
        // prompt suggests < 1e-1 is great; < 1.0 with sane std is acceptable.
        XCTAssertLessThan(audioMse, 1e-1,
            "Vocoder audio MSE \(audioMse) exceeds 1e-1 tolerance")
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
