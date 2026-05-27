// BoloTests/Chatterbox/SpeakerEmbeddingsTests.swift
import XCTest
@testable import Bolo

final class SpeakerEmbeddingsTests: XCTestCase {
    func test_loadFromBundle_succeeds() throws {
        _ = try SpeakerEmbeddings.loadFromBundle()
    }

    func test_speakerEmbedding_has256Dimensions() throws {
        let conds = try SpeakerEmbeddings.loadFromBundle()
        // t3.speaker_emb is (1, 256) — CAMPPlus x-vector from Chatterbox-Turbo
        XCTAssertEqual(conds.speakerEmbedding.count, 256,
                       "Chatterbox-Turbo t3.speaker_emb is 256-dimensional (CAMPPlus x-vector)")
    }

    func test_allConditioningTensors_areLoaded() throws {
        let conds = try SpeakerEmbeddings.loadFromBundle()
        // T3 conditioning components
        XCTAssertFalse(conds.speakerEmbedding.isEmpty,
                       "t3.speaker_emb should be non-empty")
        XCTAssertFalse(conds.condPromptSpeechTokens.isEmpty,
                       "t3.cond_prompt_speech_tokens should be non-empty (375 tokens)")
        XCTAssertFalse(conds.emotionAdv.isEmpty,
                       "t3.emotion_adv should be non-empty")
        // S3Gen-namespace tensors
        XCTAssertFalse(conds.s3GenConditioning.isEmpty,
                       "Should have at least one gen.* tensor for S3Gen conditioning")
    }
}
