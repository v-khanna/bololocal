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

    // MARK: - Benchmark (real measurements for the writeup)
    //
    // Times model load + synthesis and samples peak process memory
    // (phys_footprint, the metric Activity Monitor shows). Prints BENCH lines.
    // Run explicitly:
    //   xcodebuild test -scheme Bolo -destination 'platform=macOS,arch=arm64' \
    //     -only-testing:BoloTests/ChatterboxTTSEngineTests/test_BENCHMARK_synthLatencyAndMemory

    func test_BENCHMARK_synthLatencyAndMemory() async throws {
        guard WeightLoader.isAlreadyDownloaded() else {
            throw XCTSkip("Chatterbox model weights not found — skipping benchmark")
        }

        let peak = PeakMemTracker()
        peak.update(currentPhysFootprintMB())
        let poller = Task.detached {
            while !Task.isCancelled {
                peak.update(currentPhysFootprintMB())
                try? await Task.sleep(nanoseconds: 100_000_000) // 100 ms
            }
        }

        let loadStart = Date()
        let pipeline = try await ChatterboxPipeline.load()
        let loadSeconds = Date().timeIntervalSince(loadStart)
        peak.update(currentPhysFootprintMB())

        let sentence = "The quick brown fox jumps over the lazy dog, and then it ran off into the night."
        let synthStart = Date()
        let samples = try await pipeline.generate(text: sentence)
        let synthSeconds = Date().timeIntervalSince(synthStart)
        peak.update(currentPhysFootprintMB())

        poller.cancel()

        let audioSeconds = Double(samples.count) / 24_000.0
        let rtf = audioSeconds > 0 ? synthSeconds / audioSeconds : -1

        let line = String(
            format: "BENCH chars=%d model_load_s=%.2f synth_s=%.2f audio_s=%.2f realtime_factor=%.2f peak_mem_mb=%.0f samples=%d",
            sentence.count, loadSeconds, synthSeconds, audioSeconds, rtf, peak.value, samples.count
        )
        print(line)
        NSLog("%@", line)

        XCTAssertFalse(samples.isEmpty)
    }
}

/// Thread-safe max-tracker for the memory poller.
final class PeakMemTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Double = 0
    func update(_ x: Double) { lock.lock(); _value = max(_value, x); lock.unlock() }
    var value: Double { lock.lock(); defer { lock.unlock() }; return _value }
}

/// Current process memory footprint in MB (phys_footprint — matches Activity Monitor).
func currentPhysFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / 1_048_576.0
}
