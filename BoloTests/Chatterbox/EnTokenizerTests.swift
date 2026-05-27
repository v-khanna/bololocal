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

    // MARK: - Task 5: encode/decode tests

    func test_encode_helloWorld_matchesPythonReference() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        let tokens = tokenizer.encode("Hello world")
        // Reference IDs from Python tokenizers library with ByteLevel pre-tokenizer + BPE
        let expected: [Int] = [15496, 995]
        XCTAssertEqual(tokens, expected, "BPE token IDs must match Python reference exactly")
    }

    func test_encodeDecode_roundTrip_simpleAscii() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        let original = "The quick brown fox jumps over the lazy dog."
        let tokens = tokenizer.encode(original)
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(decoded, original)
    }

    func test_encodeDecode_roundTrip_punctuation() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        let original = "Hello, world! It's a test."
        let tokens = tokenizer.encode(original)
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(decoded, original)
    }

    func test_encodeDecode_roundTrip_paralinguisticTags() throws {
        let tokenizer = try EnTokenizer.loadFromBundle()
        // [laugh] is decomposed by BPE into "[", "laugh", "]" — NOT a single special token ID.
        // The round-trip still reconstructs the original string exactly.
        let original = "That is funny [laugh] really."
        let tokens = tokenizer.encode(original)
        let decoded = tokenizer.decode(tokens)
        XCTAssertEqual(decoded, original)
    }
}
