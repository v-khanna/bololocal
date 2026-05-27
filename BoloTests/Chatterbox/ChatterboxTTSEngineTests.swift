// BoloTests/Chatterbox/ChatterboxTTSEngineTests.swift
import XCTest
@testable import Bolo

final class ChatterboxTTSEngineTests: XCTestCase {
    func test_initWithModelProvider_doesNotThrow() async throws {
        // Provider returns a placeholder; engine init shouldn't load anything yet.
        let engine = ChatterboxTTSEngine { fatalError("loader should not run yet") }
        _ = engine
    }

    func test_synthesize_throwsNotImplemented() async {
        let engine = ChatterboxTTSEngine { fatalError("loader should not run yet") }
        do {
            try await engine.synthesize(text: "hi", voice: .systemDefault, speed: Speed(1.0))
            XCTFail("Expected throw")
        } catch TTSError.synthesisFailed(let msg) {
            XCTAssertTrue(msg.contains("not implemented"), "got: \(msg)")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }
}
