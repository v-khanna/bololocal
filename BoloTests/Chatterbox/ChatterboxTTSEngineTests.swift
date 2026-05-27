// BoloTests/Chatterbox/ChatterboxTTSEngineTests.swift
import XCTest
@testable import Bolo

final class ChatterboxTTSEngineTests: XCTestCase {

    // MARK: - Cheap tests (always run)

    func test_initWithModelProvider_doesNotThrow() async throws {
        // Provider is a closure; init must not invoke it eagerly.
        let engine = ChatterboxTTSEngine { fatalError("loader should not run yet") }
        _ = engine
    }

    func test_synthesize_throwsOnEmptyText() async {
        let engine = ChatterboxTTSEngine { fatalError("loader should not run") }
        do {
            try await engine.synthesize(text: "", voice: .systemDefault, speed: Speed(1.0))
            XCTFail("Expected TTSError.emptyText to be thrown")
        } catch TTSError.emptyText {
            // Correct — empty text is rejected before the provider is called.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func test_stop_doesNotCrash_whenNotPlaying() async {
        // stop() on an idle engine must not crash. The nonisolated dispatch to
        // _stop() is fire-and-forget; we just verify the call completes without error.
        let engine = ChatterboxTTSEngine { fatalError("loader should not run") }
        engine.stop()
        // Allow the actor-hop Task to drain before the test exits.
        try? await Task.sleep(nanoseconds: 50_000_000)  // 50 ms
    }

    // MARK: - Heavy test (gated on model weights being present on disk)
    //
    // Verifies that ChatterboxPipeline.generate(text:) returns non-empty audio
    // samples at 24 kHz when the full model is loaded. Audio playback is
    // intentionally not tested here — AVAudioEngine cannot start in the
    // xcodebuild test runner sandbox (no audio hardware; error -10868). The
    // playback path is structurally identical to Qwen3TTSEngine and is exercised
    // in production (Phase 7 signed-build smoke test).
    //
    // Expected wall time: 15–120 s with cached weights.

    func test_synthesize_realModel_producesAudioSamples() async throws {
        guard WeightLoader.isAlreadyDownloaded() else {
            throw XCTSkip("Chatterbox model weights not found — skipping heavy test")
        }

        let pipeline = try await ChatterboxPipeline.load()
        let samples = try await pipeline.generate(text: "Hello world.")

        XCTAssertFalse(samples.isEmpty, "Pipeline returned 0 samples")
        // 24 kHz × ~0.5 s minimum for any meaningful speech
        XCTAssertGreaterThan(samples.count, 24_000 / 2, "Expected at least 0.5 s of audio")
        // Sanity: samples should not all be zero
        let maxAmplitude = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(maxAmplitude, 0.001, "All samples are near-zero — pipeline may have failed silently")
    }
}
