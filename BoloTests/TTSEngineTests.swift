import XCTest
@testable import Bolo
@preconcurrency import Qwen3TTS

final class TTSEngineTests: XCTestCase {
    func test_voiceID_isHashable() {
        let a = VoiceID(rawValue: "system-default")
        let b = VoiceID(rawValue: "system-default")
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    func test_speed_clampsToValidRange() {
        XCTAssertEqual(Speed(0.25).value, 0.5)
        XCTAssertEqual(Speed(3.0).value, 2.0)
        XCTAssertEqual(Speed(1.0).value, 1.0)
    }

    // MockTTSEngine smoke test — does NOT actually play audio (would block tests with
    // the AVSpeechSynthesizer delegate callback). Instead just exercises the
    // empty-text precondition.
    func test_mockEngine_emptyText_throws() async throws {
        let engine = MockTTSEngine()
        do {
            try await engine.synthesize(text: "", voice: .systemDefault, speed: Speed(1.0))
            XCTFail("Should have thrown")
        } catch Bolo.TTSError.emptyText {
            // expected
        }
    }

    // Cheap: just construction. Does NOT load the 1.83GB model.
    func test_qwen3Engine_initializes() {
        XCTAssertNotNil(Qwen3TTSEngine(modelProvider: { throw TTSError.modelNotLoaded }))
    }

    // VoiceID.systemDefault → "english" mapping (no model load).
    func test_qwen3Engine_languageMapping() {
        XCTAssertEqual(Qwen3TTSEngine.languageString(for: VoiceID.systemDefault), "english")
        XCTAssertEqual(Qwen3TTSEngine.languageString(for: VoiceID(rawValue: "chinese")), "chinese")
        XCTAssertEqual(Qwen3TTSEngine.languageString(for: VoiceID(rawValue: "klingon")), "english")
    }

    // Heavy: gated behind HEARIT_RUN_HEAVY_TESTS=1 — first run downloads ~1.83GB.
    func test_qwen3Engine_synthesize_realModel() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["HEARIT_RUN_HEAVY_TESTS"] != "1",
            "Skipping heavy Qwen3 real-model test. Set HEARIT_RUN_HEAVY_TESTS=1 to enable (first run downloads ~1.83GB)."
        )
        let engine = Qwen3TTSEngine(modelProvider: {
            try await Qwen3TTSModel.fromPretrained()
        })
        try await engine.synthesize(
            text: "Hello world.",
            voice: VoiceID.systemDefault,
            speed: Speed(1.0)
        )
    }
}
