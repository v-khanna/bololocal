// BoloTests/Chatterbox/ChatterboxConfigTests.swift
import XCTest
@testable import Bolo

final class ChatterboxConfigTests: XCTestCase {
    func test_t3Hyperparameters_matchOfficialConfig() {
        let cfg = ChatterboxConfig.turbo
        XCTAssertEqual(cfg.t3.numLayers, 24)
        XCTAssertEqual(cfg.t3.hiddenDim, 1024)
        XCTAssertEqual(cfg.t3.numHeads, 16)
        XCTAssertEqual(cfg.t3.headDim, 64)  // 1024 / 16
        XCTAssertEqual(cfg.t3.vocabSize, 50276)
        XCTAssertEqual(cfg.t3.maxContextLength, 8196)
        XCTAssertEqual(cfg.t3.layerNormEps, 1e-5, accuracy: 1e-9)
    }

    func test_s3genHyperparameters_matchOfficialConfig() {
        let cfg = ChatterboxConfig.turbo
        XCTAssertEqual(cfg.s3gen.tokenEmbeddingDim, 512)
        XCTAssertEqual(cfg.s3gen.encoderNumBlocks, 6)
        XCTAssertEqual(cfg.s3gen.encoderAttentionHeads, 8)
        XCTAssertEqual(cfg.s3gen.encoderLinearUnits, 2048)
        XCTAssertEqual(cfg.s3gen.decoderNumBlocks, 4)
        XCTAssertEqual(cfg.s3gen.decoderNumMidBlocks, 12)
        XCTAssertEqual(cfg.s3gen.decoderNumHeads, 8)
        XCTAssertEqual(cfg.s3gen.decoderAttentionHeadDim, 64)
        XCTAssertEqual(cfg.s3gen.speechVocabSize, 6561)
    }

    func test_audioConstants() {
        XCTAssertEqual(ChatterboxConfig.audioSampleRate, 24000)
        XCTAssertEqual(ChatterboxConfig.speakerEmbeddingDim, 192)
    }
}
