// Bolo/Engine/SentenceSplitter.swift
import Foundation

/// Splits a string into sentences for streaming TTS playback.
///
/// Uses `String.enumerateSubstrings(in:options: .bySentences)`, the Foundation
/// API backed by ICU's sentence-break iterator. It correctly handles common
/// edge cases like abbreviations ("Mr.", "Dr.", "U.S.A.") and ellipses out of
/// the box on macOS.
///
/// Design choices:
/// - Pragmatic: we don't try to be perfect. If ICU is wrong on a corner case,
///   the worst outcome is a slightly long or short chunk — playback still works.
/// - Trimmed whitespace: leading/trailing whitespace stripped per sentence,
///   but punctuation is preserved (so the TTS model gets natural prosody cues).
/// - Empty input → empty array. Single-sentence input (no terminal punctuation
///   or one period) → single-element array, preserving original text.
enum SentenceSplitter {
    /// Split `text` into sentences. Returns an array of trimmed, non-empty
    /// sentence strings, in order. If `text` has no detectable boundaries
    /// the entire trimmed string is returned as a single element.
    static func split(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var sentences: [String] = []
        trimmed.enumerateSubstrings(
            in: trimmed.startIndex..<trimmed.endIndex,
            options: [.bySentences, .localized]
        ) { substring, _, _, _ in
            guard let substring else { return }
            let s = substring.trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty {
                sentences.append(s)
            }
        }

        // Fallback: enumeration produced nothing (rare — e.g. very short or
        // unusual input). Return the whole trimmed string so callers always
        // get audio.
        if sentences.isEmpty {
            return [trimmed]
        }
        return sentences
    }
}
