import XCTest
@testable import HearIt

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
        } catch TTSError.emptyText {
            // expected
        }
    }
}
