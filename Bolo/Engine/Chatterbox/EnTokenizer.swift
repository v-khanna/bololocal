// Bolo/Engine/Chatterbox/EnTokenizer.swift
import Foundation

/// BPE text tokenizer for Chatterbox-Turbo.
/// Ports the Python tokenizer at mlx-community/chatterbox-turbo-fp16 to pure Swift.
///
/// Task 4: loading skeleton (this file).
/// Task 5: encode/decode methods extension.
struct EnTokenizer: Sendable {
    let bosTokenID: Int?
    let eosTokenID: Int?
    let padTokenID: Int?

    /// Combined vocab: base vocab.json + added_tokens.json (50257 + 19 = 50276 for Chatterbox-Turbo).
    let vocab: [String: Int]
    let inverseVocab: [Int: String]
    let merges: [(String, String)]

    var vocabSize: Int { vocab.count }

    /// Load tokenizer files bundled inside the .app's Resources.
    /// xcodegen flattens the Resources directory into Contents/Resources at the bundle root.
    static func loadFromBundle() throws -> EnTokenizer {
        let bundle = Bundle.main

        func url(forResource name: String, ext: String) -> URL? {
            bundle.url(forResource: name, withExtension: ext)
                ?? bundle.url(forResource: name, withExtension: ext, subdirectory: "Resources")
        }

        guard let vocabURL = url(forResource: "vocab", ext: "json"),
              let mergesURL = url(forResource: "merges", ext: "txt"),
              let configURL = url(forResource: "tokenizer_config", ext: "json") else {
            throw TTSError.synthesisFailed(
                "EnTokenizer: tokenizer resources not bundled. Verify project.yml resources block."
            )
        }

        let addedTokensURL = url(forResource: "added_tokens", ext: "json")

        return try loadFromURLs(
            vocab: vocabURL,
            merges: mergesURL,
            config: configURL,
            addedTokens: addedTokensURL
        )
    }

    /// Test hook + reusable loader given explicit URLs.
    static func loadFromURLs(
        vocab vocabURL: URL,
        merges mergesURL: URL,
        config configURL: URL,
        addedTokens addedTokensURL: URL? = nil
    ) throws -> EnTokenizer {
        // Load base vocab
        let vocabData = try Data(contentsOf: vocabURL)
        var vocabDict = try JSONDecoder().decode([String: Int].self, from: vocabData)

        // Merge added_tokens.json (emotion/style tokens) into the base vocab
        if let addedURL = addedTokensURL,
           let addedData = try? Data(contentsOf: addedURL),
           let addedDict = try? JSONDecoder().decode([String: Int].self, from: addedData) {
            for (token, id) in addedDict {
                vocabDict[token] = id
            }
        }

        let inverseVocab = Dictionary(uniqueKeysWithValues: vocabDict.map { ($1, $0) })

        // Load BPE merge rules
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges: [(String, String)] = mergesText
            .components(separatedBy: .newlines)
            .compactMap { line -> (String, String)? in
                guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
                let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
                guard parts.count == 2 else { return nil }
                return (parts[0], parts[1])
            }

        // Parse tokenizer_config.json for special token IDs
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode([String: AnyTokenizerConfigValue].self, from: configData)

        func tokenID(forKey key: String) -> Int? {
            guard let entry = config[key] else { return nil }
            switch entry.value {
            case .int(let i): return i
            case .string(let s): return vocabDict[s]
            case .dict(let d):
                if let id = d["id"] as? Int { return id }
                if let content = d["content"] as? String { return vocabDict[content] }
                return nil
            case .other: return nil
            }
        }

        return EnTokenizer(
            bosTokenID: tokenID(forKey: "bos_token") ?? vocabDict["<|endoftext|>"],
            eosTokenID: tokenID(forKey: "eos_token") ?? vocabDict["<|endoftext|>"],
            padTokenID: tokenID(forKey: "pad_token"),
            vocab: vocabDict,
            inverseVocab: inverseVocab,
            merges: merges
        )
    }
}

// MARK: - BPE encode / decode

extension EnTokenizer {

    /// Encode UTF-8 text to BPE token IDs.
    /// Reference: GPT-2's tokenizer (https://github.com/openai/gpt-2/blob/master/src/encoder.py)
    func encode(_ text: String) -> [Int] {
        let preTokenized = byteLevelPreTokenize(text)
        var result: [Int] = []
        for word in preTokenized {
            let bpeTokens = bpe(word)
            for token in bpeTokens {
                if let id = vocab[token] {
                    result.append(id)
                } else {
                    // Fallback: byte-by-byte using byte-to-unicode mapping
                    for byte in token.utf8 {
                        let byteString = String(Self.byteToUnicode[Int(byte)])
                        if let id = vocab[byteString] {
                            result.append(id)
                        }
                    }
                }
            }
        }
        return result
    }

    /// Decode token IDs back to text.
    func decode(_ tokens: [Int]) -> String {
        let pieces = tokens.compactMap { inverseVocab[$0] }
        let joined = pieces.joined()
        return byteLevelDecode(joined)
    }

    // MARK: - GPT-2 byte-level pre-tokenization

    /// Split text into pre-tokens via GPT-2's regex pattern, then byte-level encode each.
    private func byteLevelPreTokenize(_ text: String) -> [String] {
        // GPT-2 regex from openai/gpt-2/blob/master/src/encoder.py
        // Matches contractions, words, numbers, runs of punctuation, and whitespace runs
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return [text]
        }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        return matches.map { ns.substring(with: $0.range) }.map(byteLevelEncode)
    }

    /// Map a string's UTF-8 bytes through the byte-to-unicode table.
    private func byteLevelEncode(_ text: String) -> String {
        var result = ""
        for byte in text.utf8 {
            result.append(Self.byteToUnicode[Int(byte)])
        }
        return result
    }

    /// Inverse of byteLevelEncode for decode().
    private func byteLevelDecode(_ text: String) -> String {
        var bytes: [UInt8] = []
        for scalar in text.unicodeScalars {
            if let byte = Self.unicodeToByte[scalar] {
                bytes.append(byte)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - GPT-2 byte-to-unicode mapping

    /// Maps the 256 byte values to printable unicode characters.
    /// Printable bytes (!..~, ¡..¬, ®..ÿ) map to themselves;
    /// non-printable bytes map to chars 256+.
    /// Reproduces openai/gpt-2/blob/master/src/encoder.py bytes_to_unicode().
    static let byteToUnicode: [Character] = {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs: [Int] = bs
        var n = 0
        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        var table = [Character](repeating: " ", count: 256)
        for i in 0..<bs.count {
            guard let scalar = Unicode.Scalar(cs[i]) else { continue }
            table[bs[i]] = Character(scalar)
        }
        return table
    }()

    /// Inverse map for decode.
    static let unicodeToByte: [Unicode.Scalar: UInt8] = {
        var result: [Unicode.Scalar: UInt8] = [:]
        for byte in 0..<256 {
            let char = byteToUnicode[byte]
            if let scalar = char.unicodeScalars.first {
                result[scalar] = UInt8(byte)
            }
        }
        return result
    }()

    // MARK: - BPE merge algorithm

    /// Apply BPE merge rules greedily to a single pre-tokenized "word."
    /// Each merge step picks the lowest-rank adjacent pair and merges all its occurrences.
    private func bpe(_ word: String) -> [String] {
        if word.isEmpty { return [] }
        var symbols = word.map(String.init)
        if symbols.count < 2 { return symbols }

        let mergeRanks: [String: Int] = Dictionary(
            uniqueKeysWithValues: merges.enumerated().map { (i, pair) in
                ("\(pair.0) \(pair.1)", i)
            }
        )

        while symbols.count >= 2 {
            var best: (rank: Int, a: String, b: String)? = nil
            for i in 0..<(symbols.count - 1) {
                let key = "\(symbols[i]) \(symbols[i+1])"
                if let rank = mergeRanks[key] {
                    if best == nil || rank < best!.rank {
                        best = (rank, symbols[i], symbols[i+1])
                    }
                }
            }
            guard let (_, a, b) = best else { break }
            var merged: [String] = []
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1, symbols[i] == a, symbols[i+1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }
        return symbols
    }
}

// MARK: - Internal JSON helpers

/// Flexible decoder for JSON values of unknown shape in tokenizer_config.json.
private struct AnyTokenizerConfigValue: Decodable {
    enum Value {
        case int(Int)
        case string(String)
        case dict([String: Any])
        case other
    }
    let value: Value

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            value = .int(int)
        } else if let str = try? container.decode(String.self) {
            value = .string(str)
        } else if let dict = try? container.decode([String: NestedAny].self) {
            value = .dict(dict.mapValues { $0.raw })
        } else {
            value = .other
        }
    }
}

private struct NestedAny: Decodable {
    let raw: Any
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) { raw = int }
        else if let str = try? container.decode(String.self) { raw = str }
        else if let bool = try? container.decode(Bool.self) { raw = bool }
        else { raw = NSNull() }
    }
}
