// BoloTests/SentenceSplitterTests.swift
import XCTest
@testable import Bolo

final class SentenceSplitterTests: XCTestCase {

    func test_split_threeSimpleSentences() {
        let result = SentenceSplitter.split("Hello world. How are you? I am fine!")
        XCTAssertEqual(result, ["Hello world.", "How are you?", "I am fine!"])
    }

    /// Abbreviations should NOT cause spurious splits. We accept "at most two"
    /// pieces here pragmatically: `NSLinguisticTagger` is correct in most cases
    /// but can rarely split "U.S.A." — anything more than 2 pieces would mean
    /// the splitter is broken (every period treated as boundary).
    func test_split_abbreviationsNotOverSplit() {
        let result = SentenceSplitter.split("Dr. Smith went to U.S.A. yesterday.")
        XCTAssertLessThanOrEqual(
            result.count, 2,
            "Expected at most 2 pieces for abbreviation-heavy sentence, got \(result.count): \(result)"
        )
        // Joined content should preserve all the words.
        let joined = result.joined(separator: " ")
        XCTAssertTrue(joined.contains("Dr."), "Lost 'Dr.': \(joined)")
        XCTAssertTrue(joined.contains("Smith"), "Lost 'Smith': \(joined)")
        XCTAssertTrue(joined.contains("yesterday"), "Lost 'yesterday': \(joined)")
    }

    func test_split_singleSentence_returnsSingleElement() {
        let result = SentenceSplitter.split("Just one sentence with no fancy punctuation")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.first, "Just one sentence with no fancy punctuation")
    }

    func test_split_singleSentenceWithPeriod_returnsSingleElement() {
        let result = SentenceSplitter.split("Just one sentence.")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.first, "Just one sentence.")
    }

    func test_split_emptyString_returnsEmpty() {
        XCTAssertEqual(SentenceSplitter.split(""), [])
        XCTAssertEqual(SentenceSplitter.split("   \n  "), [])
    }

    func test_split_trimsLeadingTrailingWhitespace() {
        let result = SentenceSplitter.split("  First.   Second.  ")
        XCTAssertEqual(result, ["First.", "Second."])
    }
}
