// BoloTests/Chatterbox/ChatterboxQuantizationTests.swift
import XCTest
import MLX
@testable import Bolo

/// Smoke + benchmark coverage for the in-memory 4-bit quantization path
/// (Phase 6 perf gate). The end-to-end fp16 parity test lives in
/// `ChatterboxPipelineTests` and is unaffected — it now passes
/// `quantizeBits: nil` explicitly.
///
/// These tests are GATED on the weights already being downloaded so CI
/// doesn't pull the ~3 GB safetensors. Run locally with:
///     xcodebuild test -only-testing:BoloTests/ChatterboxQuantizationTests …
final class ChatterboxQuantizationTests: XCTestCase {

    // MARK: - Smoke: quantized pipeline produces non-empty, reasonable audio

    func test_quantizedPipeline_4bit_producesReasonableAudio() async throws {
        guard WeightLoader.isAlreadyDownloaded() else {
            throw XCTSkip("Chatterbox model weights not found — skipping heavy test")
        }

        let pipeline = try await ChatterboxPipeline.load(quantizeBits: 4)
        // Short prompt — ~50 speech tokens is plenty to verify audio comes out.
        let samples = try await pipeline.generate(
            text: "This is a quantization smoke test.",
            maxGenLen: 200
        )

        XCTAssertFalse(samples.isEmpty, "Quantized pipeline returned 0 samples")
        // > 1 s of audio at 24 kHz
        XCTAssertGreaterThan(samples.count, 24_000,
                             "Expected at least 1 s of audio, got \(samples.count) samples")
        let peak = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 0.05,
                             "Peak amplitude too low (\(peak)) — quantization likely destroyed the signal")
        XCTAssertLessThan(peak, 2.0,
                          "Peak amplitude unexpectedly large (\(peak)) — possible quantization overflow")
        print("[quantize-smoke] 4-bit produced \(samples.count) samples, peak=\(peak)")
    }

    // MARK: - Benchmark: fp16 vs 4-bit timing

    /// One-shot benchmark — disabled by default because it loads the weights
    /// twice. Run it manually:
    ///     xcodebuild test -only-testing:BoloTests/ChatterboxQuantizationTests/test_benchmark_fp16_vs_4bit …
    func test_benchmark_fp16_vs_4bit() async throws {
        guard WeightLoader.isAlreadyDownloaded() else {
            throw XCTSkip("Chatterbox model weights not found — skipping benchmark")
        }
        // Belt-and-suspenders: skip unless an env var asks for it. Avoid
        // accidental runs (this is ~30 s).
        guard ProcessInfo.processInfo.environment["BOLO_RUN_BENCH"] == "1" else {
            throw XCTSkip("Set BOLO_RUN_BENCH=1 to run the quant benchmark")
        }

        let text = "This is a benchmark sentence for measuring inference time."

        // --- fp16 ---
        let fp16Pipeline = try await ChatterboxPipeline.load(quantizeBits: nil)
        let tFp0 = Date()
        let fp16Audio = try await fp16Pipeline.generate(text: text, maxGenLen: 200)
        let tFp1 = Date()
        let fp16Wall = tFp1.timeIntervalSince(tFp0)
        XCTAssertFalse(fp16Audio.isEmpty)

        // --- 4-bit ---
        let q4Pipeline = try await ChatterboxPipeline.load(quantizeBits: 4)
        let tQ0 = Date()
        let q4Audio = try await q4Pipeline.generate(text: text, maxGenLen: 200)
        let tQ1 = Date()
        let q4Wall = tQ1.timeIntervalSince(tQ0)
        XCTAssertFalse(q4Audio.isEmpty)

        let speedup = fp16Wall / q4Wall
        print("[quantize-bench] fp16 total: \(String(format: "%.2fs", fp16Wall))")
        print("[quantize-bench] 4-bit total: \(String(format: "%.2fs", q4Wall))")
        print("[quantize-bench] speedup: \(String(format: "%.2fx", speedup))")
    }
}
