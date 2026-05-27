// BoloTests/Chatterbox/EnTokenizerTests.swift
import XCTest
@testable import Bolo

final class EnTokenizerTests: XCTestCase {
    func test_loadFromBundle_succeeds() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        XCTAssertEqual(tokenizer.vocabSize, 50276,
                       "Expected vocab size 50276 from official Chatterbox-Turbo config")
    }

    func test_specialTokenIDs_areExpected() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        // At least one of bos/eos/pad should be defined (most tokenizers define eos)
        let hasAnySpecial = tokenizer.bosTokenID != nil
            || tokenizer.eosTokenID != nil
            || tokenizer.padTokenID != nil
        XCTAssertTrue(hasAnySpecial, "Expected at least one special token ID to be defined")
    }
}
