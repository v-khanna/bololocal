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
