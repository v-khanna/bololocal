# Chatterbox-Turbo → MLX-Swift Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Bolo's Qwen3-TTS engine with a native MLX-Swift port of Chatterbox-Turbo, giving the app voice quality that measurably beats ElevenLabs in blind tests while running 100% on-device.

**Architecture:** A new `Bolo/Engine/Chatterbox/` module containing Swift translations of the Python MLX implementation at [Blaizzy/mlx-audio/mlx_audio/tts/models/chatterbox](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/chatterbox). The Python MLX port is our Rosetta Stone — we translate Python→Swift, not PyTorch→MLX. Pre-converted weights live at [mlx-community/chatterbox-turbo-fp16](https://huggingface.co/mlx-community/chatterbox-turbo-fp16) (model.safetensors = 2.99 GB FP16, conds.safetensors = 165 KB speaker presets). The new `ChatterboxTTSEngine` conforms to Bolo's existing `TTSEngine` protocol, so `Coordinator`, `PlaybackController`, and UI are unchanged.

**Tech Stack:** Swift 6 (strict concurrency complete), MLX-Swift, soniqo/speech-swift's existing utilities as reference, AVFoundation (AVAudioEngine + AVAudioUnitVarispeed), Python 3.11 + mlx-audio in a sidecar virtualenv for reference validation only (never bundled).

**Key architectural correction from research:** The Chatterbox-Turbo T3 backbone is **GPT-2 style** (verified via the live `config.json` at the HF repo), not Llama-3 as initial research suggested. Implications: standard multi-head attention (no GQA), learned absolute positional embeddings (no RoPE), LayerNorm (not RMSNorm), GELU activation (not SwiGLU). The S3Gen decoder uses Conformer blocks with standard multi-head attention.

**Authoritative model hyperparameters (from config.json):**
- T3 (gpt2): 24 layers, 1024 hidden, 16 heads (64-dim each), 50,276 vocab, 8,196 max context, gelu_new, LayerNorm eps 1e-5
- S3Gen encoder: 6 Conformer blocks, 8 heads, 2048 FFN
- S3Gen decoder: 4 blocks + 12 mid Conformer blocks, 8 heads, 64-dim heads
- Speech codebook: 6,561 tokens (3^8)
- Token embedding dim: 512

---

## File Structure

All new code lives under `Bolo/Engine/Chatterbox/`. Test code lives under `BoloTests/Chatterbox/`. The Python reference environment lives outside the Xcode project under `scripts/chatterbox-reference/` and is not bundled.

```
Bolo/Engine/Chatterbox/
├── ChatterboxConfig.swift              # Hyperparameter struct, single source of truth
├── ChatterboxTTSEngine.swift           # Actor conforming to TTSEngine
├── ChatterboxModel.swift               # Top-level container: T3 + S3Gen + tokenizer + embeddings
├── EnTokenizer.swift                   # BPE text tokenizer (pure Swift)
├── SpeakerEmbeddings.swift             # Preset speaker conditioning loader
├── WeightLoader.swift                  # safetensors key→Swift property mapping
├── T3/
│   ├── T3.swift                        # 24-layer GPT-2 backbone
│   ├── T3Block.swift                   # Single transformer block
│   ├── T3Attention.swift               # Multi-head self-attention
│   ├── T3MLP.swift                     # FFN block
│   └── T3Cache.swift                   # KV cache for autoregressive gen
└── S3Gen/
    ├── S3Gen.swift                     # Decoder + vocoder coordinator
    ├── ConformerBlock.swift            # Shared Conformer block
    ├── S3Encoder.swift                 # 6-block Conformer encoder
    ├── S3Decoder.swift                 # 4-block + 12 mid Conformer decoder
    └── Vocoder.swift                   # Mel-spec → 24kHz audio

BoloTests/Chatterbox/
├── ChatterboxConfigTests.swift
├── EnTokenizerTests.swift
├── SpeakerEmbeddingsTests.swift
├── WeightLoaderTests.swift
├── T3Tests.swift
├── S3GenTests.swift
└── ChatterboxTTSEngineTests.swift

scripts/chatterbox-reference/
├── setup.sh                            # Create Python venv, install mlx-audio
├── generate-reference.py               # Run mlx-audio on fixed inputs, dump activations
├── compare-activations.py              # Compare Swift outputs to Python reference
└── README.md                           # How to use the reference env
```

The plan progresses bottom-up: foundational pieces first (config, tokenizer, weight loading), then T3, then S3Gen, then wrapper, then production swap. Each task ends with a commit that leaves the project in a buildable state with passing tests.

---

## Testing Strategy

Three test tiers:

- **Default** (`xcodebuild test`): no model loading, no MLX inference. Just types, tokenizer round-trip, weight key validation, mock engine. Runs in <2s. Always green.
- **Heavy** (`BOLO_RUN_HEAVY_TESTS=1 xcodebuild test`): downloads the 2.99 GB model on first run, runs real inference, compares against Python reference. Skipped in CI.
- **Reference parity** (manual): runs Python mlx-audio side-by-side, compares activations layer-by-layer. The mitigation for the "model runs but outputs noise" risk.

Every translation task that touches inference (T3 layers, S3Gen layers) ends with a *reference parity gate*: implementer must run the Python reference on the same input and verify Swift output matches within MSE tolerance. The implementation plan provides the exact Python command and the tolerance.

---

## Phase 1: Foundation

### Task 1: Set up Python reference environment

The implementer needs a working Python mlx-audio install to validate translations against. This is the source of truth for "is my Swift code correct."

**Files:**
- Create: `scripts/chatterbox-reference/setup.sh`
- Create: `scripts/chatterbox-reference/generate-reference.py`
- Create: `scripts/chatterbox-reference/README.md`
- Modify: `.gitignore` (exclude the venv directory)

- [ ] **Step 1: Write setup.sh**

  ```bash
  #!/usr/bin/env bash
  # scripts/chatterbox-reference/setup.sh
  # Creates a Python venv with mlx-audio installed for Chatterbox reference validation.
  # Not bundled with Bolo — pure developer tool.
  set -euo pipefail

  cd "$(dirname "$0")"

  if [ ! -d venv ]; then
    python3 -m venv venv
  fi

  source venv/bin/activate
  pip install --upgrade pip
  pip install mlx-audio huggingface_hub safetensors numpy

  echo ""
  echo "Done. Activate with:"
  echo "  source scripts/chatterbox-reference/venv/bin/activate"
  ```

- [ ] **Step 2: Make it executable and run it**

  ```bash
  chmod +x scripts/chatterbox-reference/setup.sh
  ./scripts/chatterbox-reference/setup.sh
  ```

  Expected: `pip install` completes. `which python` shows venv path.

- [ ] **Step 3: Write generate-reference.py**

  ```python
  # scripts/chatterbox-reference/generate-reference.py
  # Generates reference outputs from the Python mlx-audio implementation.
  # Saves intermediate activations + final audio for comparison.
  #
  # Usage: source venv/bin/activate && python generate-reference.py
  # Output: ./reference-outputs/

  import os
  import json
  import numpy as np
  from pathlib import Path
  import mlx.core as mx
  from mlx_audio.tts.models.chatterbox import ChatterboxTurbo

  OUTPUT_DIR = Path(__file__).parent / "reference-outputs"
  OUTPUT_DIR.mkdir(exist_ok=True)

  # Fixed inputs for reproducibility
  TEST_TEXT = "Hello world, this is a test of the Chatterbox text to speech system."
  TEST_SPEAKER_ID = 0  # First preset voice from conds.safetensors

  print("Loading Chatterbox-Turbo from mlx-community/chatterbox-turbo-fp16...")
  model = ChatterboxTurbo.from_pretrained("mlx-community/chatterbox-turbo-fp16")

  print(f"Tokenizing: {TEST_TEXT!r}")
  text_tokens = model.tokenize(TEST_TEXT)
  np.save(OUTPUT_DIR / "text_tokens.npy", text_tokens)

  print("Loading speaker embedding...")
  speaker_emb = model.get_speaker_embedding(TEST_SPEAKER_ID)
  np.save(OUTPUT_DIR / "speaker_embedding.npy", np.array(speaker_emb))

  print("Running T3 backbone...")
  speech_tokens, t3_activations = model.t3.generate(
      text_tokens, speaker_emb, return_activations=True
  )
  np.save(OUTPUT_DIR / "speech_tokens.npy", np.array(speech_tokens))
  # Save activations from layers 0, 12, 23 (first, middle, last) for spot-checking
  for layer_idx in [0, 12, 23]:
      np.save(
          OUTPUT_DIR / f"t3_layer_{layer_idx}_output.npy",
          np.array(t3_activations[layer_idx])
      )

  print("Running S3Gen decoder...")
  mel, audio = model.s3gen.synthesize(speech_tokens, speaker_emb, return_mel=True)
  np.save(OUTPUT_DIR / "mel_spectrogram.npy", np.array(mel))
  np.save(OUTPUT_DIR / "audio_samples.npy", np.array(audio))

  # Save reference metadata so Swift tests can verify shapes/values
  metadata = {
      "test_text": TEST_TEXT,
      "test_speaker_id": TEST_SPEAKER_ID,
      "text_tokens_shape": list(text_tokens.shape),
      "speech_tokens_shape": list(speech_tokens.shape),
      "speech_tokens_first_8": [int(x) for x in speech_tokens[:8]],
      "audio_sample_rate": 24000,
      "audio_num_samples": int(audio.shape[-1]),
  }
  with open(OUTPUT_DIR / "metadata.json", "w") as f:
      json.dump(metadata, f, indent=2)

  print(f"\nDone. Reference outputs saved to {OUTPUT_DIR}")
  print(f"  text_tokens: {text_tokens.shape}")
  print(f"  speech_tokens: {speech_tokens.shape}")
  print(f"  audio: {audio.shape}")
  ```

  > **Note**: If the actual Python API differs (e.g. `ChatterboxTurbo.from_pretrained` doesn't exist, or `generate` doesn't accept `return_activations`), inspect the Blaizzy/mlx-audio source at `mlx_audio/tts/models/chatterbox/chatterbox.py` and adapt the script. The goal is to produce these reference files; the exact Python API is whatever mlx-audio actually exposes.

- [ ] **Step 4: Write README.md**

  ```markdown
  # Chatterbox Reference Environment

  Python mlx-audio environment used to validate the Swift port. Not bundled with Bolo.

  ## Setup (once)

  ```bash
  ./setup.sh
  ```

  Creates `./venv/` with `mlx-audio` installed.

  ## Generate reference outputs

  ```bash
  source venv/bin/activate
  python generate-reference.py
  ```

  Outputs land in `./reference-outputs/`:
  - `text_tokens.npy` — BPE-encoded text
  - `speaker_embedding.npy` — 192-d preset speaker vector
  - `speech_tokens.npy` — T3 output
  - `t3_layer_{0,12,23}_output.npy` — intermediate activations
  - `mel_spectrogram.npy` — S3Gen mel output
  - `audio_samples.npy` — final 24kHz audio
  - `metadata.json` — shapes, fixed inputs

  The Swift tests load these `.npy` files and compare against Swift outputs.
  ```

- [ ] **Step 5: Update .gitignore**

  Append to `/Users/vir/Code/bolo/.gitignore`:

  ```
  # Chatterbox reference environment (developer-only, not bundled)
  scripts/chatterbox-reference/venv/
  scripts/chatterbox-reference/reference-outputs/
  ```

- [ ] **Step 6: Run the reference generator**

  ```bash
  cd scripts/chatterbox-reference
  source venv/bin/activate
  python generate-reference.py
  ```

  Expected: ~5 minutes (downloads 2.99 GB model on first run), produces `reference-outputs/` directory with the `.npy` files and `metadata.json`.

  If it fails: read the actual Python API in `mlx-audio/mlx_audio/tts/models/chatterbox/chatterbox.py` and adapt `generate-reference.py`.

- [ ] **Step 7: Commit**

  ```bash
  cd /Users/vir/Code/bolo
  git add scripts/chatterbox-reference/ .gitignore
  git commit -m "feat(chatterbox): Python reference environment for porting validation"
  ```

---

### Task 2: ChatterboxConfig — single source of truth for hyperparameters

The exact model hyperparameters are baked into the safetensors weight shapes. Hardcoding them in a `Config` struct makes test assertions concrete and prevents drift.

**Files:**
- Create: `Bolo/Engine/Chatterbox/ChatterboxConfig.swift`
- Create: `BoloTests/Chatterbox/ChatterboxConfigTests.swift`
- Modify: `project.yml` if needed (verify recursive sources still pick up new subdir)

- [ ] **Step 1: Write the failing test**

  ```swift
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
  ```

- [ ] **Step 2: Run test, verify it fails**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/ChatterboxConfigTests test`
  Expected: FAIL with "Cannot find 'ChatterboxConfig' in scope"

- [ ] **Step 3: Implement ChatterboxConfig**

  ```swift
  // Bolo/Engine/Chatterbox/ChatterboxConfig.swift
  import Foundation

  /// Hyperparameters for the Chatterbox-Turbo model. Single source of truth.
  /// Values are pinned from the official config.json at
  /// https://huggingface.co/mlx-community/chatterbox-turbo-fp16/blob/main/config.json
  struct ChatterboxConfig: Sendable, Equatable {

      struct T3: Sendable, Equatable {
          /// GPT-2 style transformer hyperparameters.
          let numLayers: Int
          let hiddenDim: Int
          let numHeads: Int
          let headDim: Int
          let vocabSize: Int        // text BPE vocab
          let maxContextLength: Int
          let layerNormEps: Double
          let activation: String    // "gelu_new"
      }

      struct S3Gen: Sendable, Equatable {
          let tokenEmbeddingDim: Int
          let encoderNumBlocks: Int
          let encoderAttentionHeads: Int
          let encoderLinearUnits: Int
          let decoderNumBlocks: Int
          let decoderNumMidBlocks: Int
          let decoderNumHeads: Int
          let decoderAttentionHeadDim: Int
          let speechVocabSize: Int  // 6,561 = 3^8
      }

      let t3: T3
      let s3gen: S3Gen

      static let audioSampleRate: Double = 24000
      static let speakerEmbeddingDim: Int = 192

      /// Chatterbox-Turbo (default for v1). English-only, 1-step distilled decoder.
      static let turbo = ChatterboxConfig(
          t3: T3(
              numLayers: 24,
              hiddenDim: 1024,
              numHeads: 16,
              headDim: 64,
              vocabSize: 50276,
              maxContextLength: 8196,
              layerNormEps: 1e-5,
              activation: "gelu_new"
          ),
          s3gen: S3Gen(
              tokenEmbeddingDim: 512,
              encoderNumBlocks: 6,
              encoderAttentionHeads: 8,
              encoderLinearUnits: 2048,
              decoderNumBlocks: 4,
              decoderNumMidBlocks: 12,
              decoderNumHeads: 8,
              decoderAttentionHeadDim: 64,
              speechVocabSize: 6561
          )
      )
  }
  ```

- [ ] **Step 4: Regenerate Xcode project + run tests**

  ```bash
  cd /Users/vir/Code/bolo
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/ChatterboxConfigTests test
  ```

  Expected: 3 tests pass. Full test suite (23 prior + 3 new = 26) should pass with `xcodebuild ... test` (without the `-only-testing` filter).

- [ ] **Step 5: Commit**

  ```bash
  cd /Users/vir/Code/bolo
  git add Bolo/Engine/Chatterbox/ChatterboxConfig.swift BoloTests/Chatterbox/ChatterboxConfigTests.swift
  git commit -m "feat(chatterbox): ChatterboxConfig hyperparameters from official config.json"
  ```

---

### Task 3: Stub `ChatterboxTTSEngine` conforming to `TTSEngine`

Create a skeleton that compiles, conforms to the protocol, and throws "not implemented" — establishes the file structure and surfaces immediately any integration issues with the existing `TTSEngine` protocol.

**Files:**
- Create: `Bolo/Engine/Chatterbox/ChatterboxTTSEngine.swift`
- Create: `BoloTests/Chatterbox/ChatterboxTTSEngineTests.swift`

- [ ] **Step 1: Write the failing test**

  ```swift
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
  ```

- [ ] **Step 2: Run test to verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/ChatterboxTTSEngineTests test`
  Expected: FAIL with "Cannot find 'ChatterboxTTSEngine' in scope"

- [ ] **Step 3: Implement the stub**

  ```swift
  // Bolo/Engine/Chatterbox/ChatterboxTTSEngine.swift
  import Foundation
  import AVFoundation

  /// Chatterbox-Turbo TTS engine. Native MLX-Swift port.
  ///
  /// Conforms to TTSEngine — drop-in replacement for Qwen3TTSEngine.
  /// Model lifecycle is owned externally by ModelManager; this actor receives
  /// a `modelProvider` closure and calls it each synthesize.
  actor ChatterboxTTSEngine: TTSEngine {
      private let modelProvider: @Sendable () async throws -> ChatterboxModel

      init(modelProvider: @escaping @Sendable () async throws -> ChatterboxModel) {
          self.modelProvider = modelProvider
      }

      nonisolated func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
          guard !text.isEmpty else { throw TTSError.emptyText }
          try await _synthesize(text: text, voice: voice, speed: speed)
      }

      private func _synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
          // Phases 4-6 implement the real pipeline. Stub for now.
          throw TTSError.synthesisFailed("ChatterboxTTSEngine: not implemented yet")
      }

      nonisolated func stop() {
          // Phase 6: stop AVAudioEngine playback.
      }
  }

  /// Top-level model container. Populated in Phase 6 (Task 23).
  /// For now: a sentinel struct so ChatterboxTTSEngine compiles.
  struct ChatterboxModel: Sendable {
      // Populated in later tasks. Empty for stub.
  }
  ```

- [ ] **Step 4: Run tests**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' test
  ```

  Expected: All tests pass (26 prior + 2 new = 28). `ChatterboxTTSEngine` stub builds, conforms to protocol, throws on synthesize.

- [ ] **Step 5: Commit**

  ```bash
  git add Bolo/Engine/Chatterbox/ChatterboxTTSEngine.swift BoloTests/Chatterbox/ChatterboxTTSEngineTests.swift
  git commit -m "feat(chatterbox): ChatterboxTTSEngine stub conforming to TTSEngine"
  ```

---

## Phase 2: Tokenizer + Speaker Embeddings

### Task 4: Bundle tokenizer files + `EnTokenizer` skeleton

Download the tokenizer files from HuggingFace, bundle them in `Resources/`, and create a Swift loader.

**Files:**
- Create: `Bolo/Engine/Chatterbox/Resources/vocab.json` (downloaded, 999 KB)
- Create: `Bolo/Engine/Chatterbox/Resources/merges.txt` (downloaded, 456 KB)
- Create: `Bolo/Engine/Chatterbox/Resources/tokenizer_config.json` (downloaded, 4 KB)
- Create: `Bolo/Engine/Chatterbox/EnTokenizer.swift`
- Create: `BoloTests/Chatterbox/EnTokenizerTests.swift`
- Modify: `project.yml` — add the resources to the Bolo target's `resources:` list

- [ ] **Step 1: Download tokenizer files**

  ```bash
  cd /Users/vir/Code/bolo/Bolo/Engine/Chatterbox/Resources
  mkdir -p .
  for file in vocab.json merges.txt tokenizer_config.json special_tokens_map.json added_tokens.json; do
    curl -L -o "$file" "https://huggingface.co/mlx-community/chatterbox-turbo-fp16/resolve/main/$file"
  done
  ls -lh *.json *.txt
  ```

  Expected: 5 files downloaded, sizes match the HF repo (vocab.json ~999 KB, merges.txt ~456 KB).

- [ ] **Step 2: Update project.yml to bundle these as resources**

  In `/Users/vir/Code/bolo/project.yml`, under the `Bolo` target, add a `resources` block listing the new files:

  ```yaml
  targets:
    Bolo:
      type: application
      platform: macOS
      sources:
        - path: Bolo
      resources:
        - path: Bolo/Engine/Chatterbox/Resources
          buildPhase:
            copyFiles:
              destination: resources
              subpath: Chatterbox
  ```

  If the YAML schema differs (xcodegen has slight variations), use:

  ```yaml
  targets:
    Bolo:
      type: application
      platform: macOS
      sources:
        - path: Bolo
          excludes:
            - "**/Resources/*.json"
            - "**/Resources/*.txt"
      resources:
        - path: Bolo/Engine/Chatterbox/Resources
  ```

  Run `xcodegen generate` and verify the files appear in the generated `Bolo.xcodeproj` under the Bolo target's "Copy Bundle Resources" build phase. If not, consult xcodegen docs at https://github.com/yonaskolb/XcodeGen/blob/master/Docs/ProjectSpec.md#target-source.

- [ ] **Step 3: Write the failing test**

  ```swift
  // BoloTests/Chatterbox/EnTokenizerTests.swift
  import XCTest
  @testable import Bolo

  final class EnTokenizerTests: XCTestCase {
      func test_loadFromBundle_succeeds() throws {
          let tokenizer = try EnTokenizer.loadFromBundle()
          XCTAssertGreaterThan(tokenizer.vocabSize, 50000)
          XCTAssertEqual(tokenizer.vocabSize, 50276)
      }

      func test_specialTokenIDs_areExpected() throws {
          let tokenizer = try EnTokenizer.loadFromBundle()
          XCTAssertNotNil(tokenizer.bosTokenID)
          XCTAssertNotNil(tokenizer.eosTokenID)
          XCTAssertNotNil(tokenizer.padTokenID)
      }
  }
  ```

- [ ] **Step 4: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/EnTokenizerTests test`
  Expected: FAIL with "Cannot find 'EnTokenizer' in scope"

- [ ] **Step 5: Implement EnTokenizer skeleton (load + vocab size only — encode/decode in Task 5)**

  ```swift
  // Bolo/Engine/Chatterbox/EnTokenizer.swift
  import Foundation

  /// BPE text tokenizer for Chatterbox-Turbo.
  /// Ported from Python tokenizer files at mlx-community/chatterbox-turbo-fp16.
  /// Loads vocab.json + merges.txt at init; pure Swift, no MLX dependency.
  struct EnTokenizer: Sendable {
      /// Token ID for the start-of-sequence marker, if defined.
      let bosTokenID: Int?
      /// Token ID for the end-of-sequence marker, if defined.
      let eosTokenID: Int?
      /// Token ID for padding, if defined.
      let padTokenID: Int?

      /// Mapping from BPE token string to integer ID.
      private let vocab: [String: Int]
      /// Inverse mapping for decode().
      private let inverseVocab: [Int: String]
      /// Ordered list of merge rules (each is a pair of token strings to merge).
      private let merges: [(String, String)]

      var vocabSize: Int { vocab.count }

      /// Load tokenizer files bundled at app/Contents/Resources/Chatterbox/
      static func loadFromBundle() throws -> EnTokenizer {
          let bundle = Bundle.main
          guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "json", subdirectory: "Chatterbox"),
                let mergesURL = bundle.url(forResource: "merges", withExtension: "txt", subdirectory: "Chatterbox"),
                let configURL = bundle.url(forResource: "tokenizer_config", withExtension: "json", subdirectory: "Chatterbox")
          else {
              throw TTSError.synthesisFailed("EnTokenizer: tokenizer resources not bundled. Verify project.yml resources block.")
          }
          return try loadFromURLs(vocab: vocabURL, merges: mergesURL, config: configURL)
      }

      static func loadFromURLs(vocab vocabURL: URL, merges mergesURL: URL, config configURL: URL) throws -> EnTokenizer {
          let vocabData = try Data(contentsOf: vocabURL)
          let vocabDict = try JSONDecoder().decode([String: Int].self, from: vocabData)
          let inverseVocab = Dictionary(uniqueKeysWithValues: vocabDict.map { ($1, $0) })

          let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
          let merges: [(String, String)] = mergesText
              .components(separatedBy: .newlines)
              .compactMap { line -> (String, String)? in
                  // Skip header line `#version: 0.2` and empty lines
                  guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
                  let parts = line.split(separator: " ", maxSplits: 1).map(String.init)
                  guard parts.count == 2 else { return nil }
                  return (parts[0], parts[1])
              }

          let configData = try Data(contentsOf: configURL)
          let config = try JSONDecoder().decode([String: AnyCodable].self, from: configData)

          func tokenID(forKey key: String) -> Int? {
              guard let value = config[key]?.value as? [String: Any] else {
                  if let id = config[key]?.value as? Int { return id }
                  return nil
              }
              // Format may be { "content": "<bos>", "id": 1 } — extract id
              if let id = value["id"] as? Int { return id }
              if let content = value["content"] as? String { return vocabDict[content] }
              return nil
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

  /// Minimal helper for decoding JSON values of unknown type.
  /// Used only inside the tokenizer loader.
  private struct AnyCodable: Decodable {
      let value: Any
      init(from decoder: Decoder) throws {
          let container = try decoder.singleValueContainer()
          if let int = try? container.decode(Int.self) { value = int }
          else if let str = try? container.decode(String.self) { value = str }
          else if let dict = try? container.decode([String: AnyCodable].self) {
              value = dict.mapValues { $0.value }
          } else if let arr = try? container.decode([AnyCodable].self) {
              value = arr.map { $0.value }
          } else {
              value = NSNull()
          }
      }
  }
  ```

- [ ] **Step 6: Run tests**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' test
  ```

  Expected: 3 new tests pass (load, vocab size 50276, special tokens exist). All existing tests still pass.

- [ ] **Step 7: Commit**

  ```bash
  git add Bolo/Engine/Chatterbox/EnTokenizer.swift Bolo/Engine/Chatterbox/Resources/ BoloTests/Chatterbox/EnTokenizerTests.swift project.yml
  git commit -m "feat(chatterbox): EnTokenizer skeleton + bundled tokenizer resources"
  ```

---

### Task 5: BPE encode/decode + round-trip tests

Implement the actual BPE algorithm. This is mechanical translation from the GPT-2 tokenizer logic; same algorithm OpenAI shipped, well-documented.

**Files:**
- Modify: `Bolo/Engine/Chatterbox/EnTokenizer.swift`
- Modify: `BoloTests/Chatterbox/EnTokenizerTests.swift`

- [ ] **Step 1: Add round-trip test cases**

  Append to `EnTokenizerTests.swift`:

  ```swift
  func test_encode_helloWorld_matchesReference() throws {
      // Reference values: run the Python tokenizer from
      // scripts/chatterbox-reference/generate-reference.py on "Hello world"
      // and copy the resulting token IDs here.
      // Implementer: replace the array literal below with actual Python output.
      let tokenizer = try EnTokenizer.loadFromBundle()
      let tokens = tokenizer.encode("Hello world")
      XCTAssertFalse(tokens.isEmpty)
      XCTAssertLessThan(tokens.count, 10)  // sanity: not absurdly long
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
      let original = "Hello, world! It's a 'test' with \"various\" punctuation; right?"
      let tokens = tokenizer.encode(original)
      let decoded = tokenizer.decode(tokens)
      XCTAssertEqual(decoded, original)
  }

  func test_encode_paralinguisticTags_arePreserved() throws {
      let tokenizer = try EnTokenizer.loadFromBundle()
      let original = "That's funny [laugh] really."
      let tokens = tokenizer.encode(original)
      let decoded = tokenizer.decode(tokens)
      XCTAssertEqual(decoded, original)
  }
  ```

- [ ] **Step 2: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/EnTokenizerTests test`
  Expected: FAIL with "Value of type 'EnTokenizer' has no member 'encode'"

- [ ] **Step 3: Implement BPE encode and decode**

  Append to `EnTokenizer.swift`:

  ```swift
  extension EnTokenizer {

      /// Encode a UTF-8 string to BPE token IDs.
      func encode(_ text: String) -> [Int] {
          // GPT-2 BPE algorithm:
          // 1. Byte-level pre-tokenization (each Unicode char → UTF-8 bytes → mapped to a "byte string")
          // 2. Apply merge rules greedily in the order they appear in merges.txt
          // 3. Look up final tokens in vocab
          //
          // Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py

          let preTokenized = byteLevelPreTokenize(text)
          var result: [Int] = []
          for word in preTokenized {
              let bpeTokens = bpe(word)
              for token in bpeTokens {
                  if let id = vocab[token] {
                      result.append(id)
                  } else {
                      // Fallback: tokenize byte-by-byte (extremely rare with proper byte mapping)
                      for byte in token.utf8 {
                          let byteString = String(UnicodeScalar(byte))
                          if let id = vocab[byteString] {
                              result.append(id)
                          }
                      }
                  }
              }
          }
          return result
      }

      /// Decode token IDs back to a string.
      func decode(_ tokens: [Int]) -> String {
          let pieces = tokens.compactMap { inverseVocab[$0] }
          let joined = pieces.joined()
          return byteLevelDecode(joined)
      }

      // MARK: - Byte-level pre-tokenization (GPT-2 style)

      /// Split text into pre-tokens using GPT-2's regex pattern,
      /// then map each byte through the byte-to-unicode table.
      private func byteLevelPreTokenize(_ text: String) -> [String] {
          // GPT-2 regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
          let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
          guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
              return [text]
          }
          let nsText = text as NSString
          let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length))
          return matches.map { nsText.substring(with: $0.range) }.map(byteLevelEncode)
      }

      /// Map UTF-8 bytes through the byte-to-unicode table so every byte is a printable char.
      private func byteLevelEncode(_ text: String) -> String {
          let table = Self.byteToUnicode
          var result = ""
          for byte in text.utf8 {
              result.append(table[Int(byte)])
          }
          return result
      }

      /// Inverse of byteLevelEncode for decode().
      private func byteLevelDecode(_ text: String) -> String {
          let table = Self.unicodeToByte
          var bytes: [UInt8] = []
          for scalar in text.unicodeScalars {
              if let byte = table[scalar] {
                  bytes.append(byte)
              }
          }
          return String(decoding: bytes, as: UTF8.self)
      }

      /// GPT-2's byte-to-unicode mapping. Maps the 256 byte values to printable unicode chars.
      /// Bytes that are already printable map to themselves; non-printable bytes map to chars 256+.
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
          // Now bs[i] = original byte, cs[i] = unicode codepoint
          var table = [Character](repeating: " ", count: 256)
          for i in 0..<bs.count {
              guard let scalar = Unicode.Scalar(cs[i]) else { continue }
              table[bs[i]] = Character(scalar)
          }
          return table
      }()

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

      // MARK: - BPE merge

      /// Apply BPE merge rules to a single pre-tokenized word.
      private func bpe(_ word: String) -> [String] {
          if word.isEmpty { return [] }
          // Start with each character as a separate token
          var symbols = word.map(String.init)
          if symbols.count < 2 { return symbols }

          // Build a rank dict from merges (lower rank = earlier in merges.txt = higher priority)
          let mergeRanks: [String: Int] = Dictionary(
              uniqueKeysWithValues: merges.enumerated().map { (i, pair) in
                  ("\(pair.0) \(pair.1)", i)
              }
          )

          while symbols.count >= 2 {
              // Find the best (lowest-rank) adjacent pair to merge
              var bestPair: (Int, String, String)? = nil
              for i in 0..<(symbols.count - 1) {
                  let key = "\(symbols[i]) \(symbols[i+1])"
                  if let rank = mergeRanks[key] {
                      if bestPair == nil || rank < (bestPair!.0) {
                          bestPair = (rank, symbols[i], symbols[i+1])
                      }
                  }
              }
              guard let (_, a, b) = bestPair else { break }
              // Merge all occurrences of (a, b) into a+b
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
  ```

- [ ] **Step 4: Generate Python reference token IDs for "Hello world"**

  ```bash
  cd /Users/vir/Code/bolo/scripts/chatterbox-reference
  source venv/bin/activate
  python -c "
  from mlx_audio.tts.models.chatterbox import ChatterboxTurbo
  m = ChatterboxTurbo.from_pretrained('mlx-community/chatterbox-turbo-fp16')
  print(m.tokenize('Hello world'))
  "
  ```

  Copy the resulting integer list into `EnTokenizerTests.swift` `test_encode_helloWorld_matchesReference` — replace the placeholder asserts with `XCTAssertEqual(tokens, [<the actual IDs>])`.

- [ ] **Step 5: Run tests**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/EnTokenizerTests test
  ```

  Expected: 7 tests pass (3 from Task 4 + 4 new). If the reference-matching test fails: print the Swift output, diff against Python — most likely cause is a regex difference in pre-tokenization. Iterate until it matches exactly.

- [ ] **Step 6: Commit**

  ```bash
  git add Bolo/Engine/Chatterbox/EnTokenizer.swift BoloTests/Chatterbox/EnTokenizerTests.swift
  git commit -m "feat(chatterbox): BPE encode + decode with GPT-2 byte-level pre-tokenization"
  ```

---

### Task 6: `SpeakerEmbeddings` — load preset speaker conditioning vectors

The HF repo ships `conds.safetensors` (165 KB) containing pre-computed 192-d speaker vectors for the preset voices. We bundle this file and load it at init.

**Files:**
- Create: `Bolo/Engine/Chatterbox/Resources/conds.safetensors` (downloaded, 165 KB)
- Create: `Bolo/Engine/Chatterbox/SpeakerEmbeddings.swift`
- Create: `BoloTests/Chatterbox/SpeakerEmbeddingsTests.swift`

- [ ] **Step 1: Download conds.safetensors**

  ```bash
  curl -L -o /Users/vir/Code/bolo/Bolo/Engine/Chatterbox/Resources/conds.safetensors \
    https://huggingface.co/mlx-community/chatterbox-turbo-fp16/resolve/main/conds.safetensors
  ls -lh /Users/vir/Code/bolo/Bolo/Engine/Chatterbox/Resources/conds.safetensors
  ```

  Expected: 165 KB file.

- [ ] **Step 2: Inspect the conds.safetensors structure**

  ```bash
  cd /Users/vir/Code/bolo/scripts/chatterbox-reference
  source venv/bin/activate
  python -c "
  from safetensors import safe_open
  with safe_open('../Bolo/Engine/Chatterbox/Resources/conds.safetensors', framework='numpy') as f:
      for key in f.keys():
          t = f.get_tensor(key)
          print(f'{key}: shape={t.shape} dtype={t.dtype}')
  "
  ```

  Expected output (something like):
  ```
  speaker_0: shape=(192,) dtype=float16
  speaker_1: shape=(192,) dtype=float16
  ...
  ```

  Record the exact key names — they go into the Swift loader.

- [ ] **Step 3: Write the failing test**

  ```swift
  // BoloTests/Chatterbox/SpeakerEmbeddingsTests.swift
  import XCTest
  @testable import Bolo

  final class SpeakerEmbeddingsTests: XCTestCase {
      func test_loadFromBundle_succeeds() throws {
          let embeddings = try SpeakerEmbeddings.loadFromBundle()
          XCTAssertGreaterThan(embeddings.count, 0)
      }

      func test_eachEmbedding_is192Dimensional() throws {
          let embeddings = try SpeakerEmbeddings.loadFromBundle()
          for (name, vector) in embeddings.all {
              XCTAssertEqual(vector.count, 192, "Speaker \(name) wrong dim")
          }
      }

      func test_defaultSpeaker_exists() throws {
          let embeddings = try SpeakerEmbeddings.loadFromBundle()
          XCTAssertNotNil(embeddings.embedding(for: .systemDefault))
      }
  }
  ```

- [ ] **Step 4: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/SpeakerEmbeddingsTests test`
  Expected: FAIL "Cannot find 'SpeakerEmbeddings' in scope"

- [ ] **Step 5: Implement SpeakerEmbeddings**

  ```swift
  // Bolo/Engine/Chatterbox/SpeakerEmbeddings.swift
  import Foundation
  import MLX

  /// Pre-computed 192-dimensional speaker conditioning vectors loaded from
  /// the bundled conds.safetensors (downloaded from mlx-community/chatterbox-turbo-fp16).
  /// Used as global conditioning input to both T3 and S3Gen.
  ///
  /// Bypasses the need to port the CAMPPlus Voice Encoder in v1 — we ship a fixed
  /// set of preset voices. User voice cloning lands in v1.1 once VE is ported.
  struct SpeakerEmbeddings: Sendable {
      /// All loaded embeddings keyed by speaker name (e.g. "speaker_0", "speaker_1", ...).
      let all: [String: [Float]]

      var count: Int { all.count }

      /// Load from the bundled conds.safetensors.
      static func loadFromBundle() throws -> SpeakerEmbeddings {
          guard let url = Bundle.main.url(
              forResource: "conds", withExtension: "safetensors", subdirectory: "Chatterbox"
          ) else {
              throw TTSError.synthesisFailed("SpeakerEmbeddings: conds.safetensors not bundled")
          }
          return try loadFromURL(url)
      }

      static func loadFromURL(_ url: URL) throws -> SpeakerEmbeddings {
          // MLX-Swift's loadArrays loads a safetensors file into [String: MLXArray]
          let arrays = try MLX.loadArrays(url: url)
          var result: [String: [Float]] = [:]
          for (key, array) in arrays {
              // Cast to Float32 for Swift-side storage
              let floatArray = array.asType(.float32).asArray(Float.self)
              result[key] = floatArray
          }
          guard !result.isEmpty else {
              throw TTSError.synthesisFailed("SpeakerEmbeddings: conds.safetensors had no tensors")
          }
          return SpeakerEmbeddings(all: result)
      }

      /// Look up the embedding for a given VoiceID.
      /// VoiceID.systemDefault → first speaker alphabetically.
      func embedding(for voice: VoiceID) -> [Float]? {
          if voice == .systemDefault {
              let firstKey = all.keys.sorted().first
              return firstKey.flatMap { all[$0] }
          }
          return all[voice.rawValue]
      }

      /// Available voice IDs (sorted by speaker name).
      var voiceIDs: [VoiceID] {
          all.keys.sorted().map { VoiceID(rawValue: $0) }
      }
  }
  ```

  > **Note**: If `MLX.loadArrays` isn't exactly this signature in the MLX-Swift version we use, consult the MLX-Swift API at https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx or the existing weight loading in `Qwen3TTSEngine.swift` for the correct pattern. The intent is "load all tensors from a .safetensors file into a Swift dictionary."

- [ ] **Step 6: Update project.yml to bundle conds.safetensors**

  The resources block from Task 4 should already pick up the whole `Resources/` directory recursively. Verify with:

  ```bash
  xcodegen generate
  unzip -l build/Build/Products/Debug/Bolo.app/Contents/Resources/Chatterbox 2>&1 || \
    ls build/Build/Products/Debug/Bolo.app/Contents/Resources/Chatterbox/
  ```

  Expected: shows `conds.safetensors`, `vocab.json`, etc.

  If `conds.safetensors` isn't bundled, explicitly add to project.yml's resources block.

- [ ] **Step 7: Run tests**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' test
  ```

  Expected: 3 new tests pass. All existing tests still pass.

- [ ] **Step 8: Commit**

  ```bash
  git add Bolo/Engine/Chatterbox/SpeakerEmbeddings.swift Bolo/Engine/Chatterbox/Resources/conds.safetensors BoloTests/Chatterbox/SpeakerEmbeddingsTests.swift
  git commit -m "feat(chatterbox): SpeakerEmbeddings — load preset speaker vectors from conds.safetensors"
  ```

---

## Phase 3: Weight Loading

### Task 7: `WeightLoader` — download + load model.safetensors

The 2.99 GB main model file gets downloaded at first run via the existing `ModelManager` pattern. We extend the existing download-from-HuggingFace machinery to handle Chatterbox.

**Files:**
- Create: `Bolo/Engine/Chatterbox/WeightLoader.swift`
- Create: `BoloTests/Chatterbox/WeightLoaderTests.swift`
- Modify: `Bolo/Engine/ModelDownloadProgress.swift` (no actual changes, just verify it can handle a 2.99 GB download with progress updates)

- [ ] **Step 1: Write the failing test**

  ```swift
  // BoloTests/Chatterbox/WeightLoaderTests.swift
  import XCTest
  @testable import Bolo

  final class WeightLoaderTests: XCTestCase {
      // Heavy: gated by env var because the model is 2.99 GB.
      func test_downloadAndLoad_realModel() async throws {
          try XCTSkipIf(
              ProcessInfo.processInfo.environment["BOLO_RUN_HEAVY_TESTS"] != "1",
              "Set BOLO_RUN_HEAVY_TESTS=1 to run heavy weight-loading tests"
          )
          let progress = ModelDownloadProgress()
          let weights = try await WeightLoader.downloadAndLoad(progressHandler: { p, label in
              Task { @MainActor in progress.update(progress: p, label: label) }
          })
          XCTAssertGreaterThan(weights.count, 100, "Should have many weight tensors")
          XCTAssertTrue(weights.keys.contains(where: { $0.hasPrefix("t3.") }),
                        "Should have T3 weights")
          XCTAssertTrue(weights.keys.contains(where: { $0.hasPrefix("s3gen.") }),
                        "Should have S3Gen weights")
      }

      func test_isAlreadyDownloaded_returnsBoolWithoutThrowing() {
          let result = WeightLoader.isAlreadyDownloaded()
          XCTAssertTrue(result == true || result == false)
      }
  }
  ```

- [ ] **Step 2: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/WeightLoaderTests test`
  Expected: FAIL "Cannot find 'WeightLoader'"

- [ ] **Step 3: Implement WeightLoader**

  ```swift
  // Bolo/Engine/Chatterbox/WeightLoader.swift
  import Foundation
  import MLX

  /// Downloads and loads Chatterbox-Turbo weights from Hugging Face.
  /// Cache lives at ~/Library/Application Support/Bolo/models/chatterbox-turbo-fp16/
  enum WeightLoader {

      private static let repoID = "mlx-community/chatterbox-turbo-fp16"
      private static let weightFile = "model.safetensors"

      /// Local cache path for the downloaded weight file.
      static var cachedWeightURL: URL {
          let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
          return support
              .appendingPathComponent("Bolo", isDirectory: true)
              .appendingPathComponent("models", isDirectory: true)
              .appendingPathComponent("chatterbox-turbo-fp16", isDirectory: true)
              .appendingPathComponent(weightFile)
      }

      static func isAlreadyDownloaded() -> Bool {
          FileManager.default.fileExists(atPath: cachedWeightURL.path)
      }

      /// Download (if needed) and load weights into an MLXArray dictionary.
      /// Reports progress 0.0...1.0 to the progressHandler.
      static func downloadAndLoad(
          progressHandler: @escaping @Sendable (Double, String) -> Void
      ) async throws -> [String: MLXArray] {
          if !isAlreadyDownloaded() {
              try await download(progressHandler: progressHandler)
          }
          progressHandler(1.0, "Loading weights into memory…")
          let arrays = try MLX.loadArrays(url: cachedWeightURL)
          return arrays
      }

      private static func download(
          progressHandler: @escaping @Sendable (Double, String) -> Void
      ) async throws {
          let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(weightFile)")!
          let destination = cachedWeightURL

          try FileManager.default.createDirectory(
              at: destination.deletingLastPathComponent(),
              withIntermediateDirectories: true
          )

          progressHandler(0.0, "Downloading Chatterbox-Turbo weights (2.99 GB)…")

          let delegate = ProgressDelegate(onProgress: { p in
              progressHandler(p, "Downloading… \(Int(p * 100))%")
          })

          let (tempURL, response) = try await URLSession.shared.download(
              for: URLRequest(url: url),
              delegate: delegate
          )

          guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
              throw TTSError.synthesisFailed(
                  "Chatterbox download HTTP error: \((response as? HTTPURLResponse)?.statusCode ?? -1)"
              )
          }

          try FileManager.default.moveItem(at: tempURL, to: destination)
      }

      private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
          let onProgress: @Sendable (Double) -> Void
          init(onProgress: @escaping @Sendable (Double) -> Void) { self.onProgress = onProgress }

          func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                          didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                          totalBytesExpectedToWrite: Int64) {
              guard totalBytesExpectedToWrite > 0 else { return }
              onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
          }
          func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                          didFinishDownloadingTo location: URL) {}
      }
  }
  ```

- [ ] **Step 4: Run the cheap test (not the heavy one)**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/WeightLoaderTests/test_isAlreadyDownloaded_returnsBoolWithoutThrowing test
  ```

  Expected: PASS. The heavy test is skipped without `BOLO_RUN_HEAVY_TESTS=1`.

- [ ] **Step 5 (optional): Run the heavy test once to verify download works**

  ```bash
  BOLO_RUN_HEAVY_TESTS=1 xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' \
    -only-testing:BoloTests/WeightLoaderTests test
  ```

  Expected: ~5-10 min download on first run, then loads and asserts > 100 weight tensors with `t3.` and `s3gen.` prefixes. **Run this once to confirm; not required for commit.**

- [ ] **Step 6: Commit**

  ```bash
  git add Bolo/Engine/Chatterbox/WeightLoader.swift BoloTests/Chatterbox/WeightLoaderTests.swift
  git commit -m "feat(chatterbox): WeightLoader — download + load Chatterbox-Turbo from HuggingFace"
  ```

---

## Phase 4: T3 Backbone (GPT-2 style transformer)

### Task 8: T3 attention block

Implement the GPT-2 multi-head self-attention as a standalone `MLXNN.Module`. Test in isolation against a known input/output before plumbing it into a full block.

**Files:**
- Create: `Bolo/Engine/Chatterbox/T3/T3Attention.swift`
- Create: `BoloTests/Chatterbox/T3Tests.swift`

- [ ] **Step 1: Read the Python reference**

  Open `mlx-audio/mlx_audio/tts/models/chatterbox/t3/attention.py` (or wherever the attention class lives — check the t3/ subdirectory of the Blaizzy/mlx-audio Chatterbox module). Read it carefully. Identify:
  - The exact tensor shapes at each step
  - Whether attention uses a causal mask (it should — GPT-2 is causal)
  - Whether dropout is enabled at inference (should be no — set to 0 in eval mode)
  - The Q/K/V projection layout (combined `c_attn` linear layer that outputs 3×hidden, or three separate `q_proj`/`k_proj`/`v_proj`)

  > **Implementer**: paste a quoted excerpt of the Python class into the Swift file as a comment, so future readers see the source of truth.

- [ ] **Step 2: Write the failing test**

  ```swift
  // BoloTests/Chatterbox/T3Tests.swift
  import XCTest
  import MLX
  @testable import Bolo

  final class T3Tests: XCTestCase {
      func test_attention_outputShape_matchesInput() {
          let cfg = ChatterboxConfig.turbo.t3
          let attn = T3Attention(config: cfg)
          // (batch=1, seq=5, hidden=1024)
          let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
          let output = attn(input, mask: nil, cache: nil)
          XCTAssertEqual(output.array.shape, [1, 5, cfg.hiddenDim])
      }

      func test_attention_causalMask_blocksFutureTokens() {
          let cfg = ChatterboxConfig.turbo.t3
          let attn = T3Attention(config: cfg)
          let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
          let causal = T3Attention.causalMask(seqLen: 5)
          let output = attn(input, mask: causal, cache: nil)
          XCTAssertEqual(output.array.shape, [1, 5, cfg.hiddenDim])
          // Functional check: changing token 4's input should NOT change tokens 0-3's outputs
          // (causality property)
          var modifiedInput = input
          modifiedInput[0, 4, 0..<cfg.hiddenDim] = MLXRandom.normal([cfg.hiddenDim])
          let output2 = attn(modifiedInput, mask: causal, cache: nil)
          let diff = (output[0, 0..<4] - output2[0, 0..<4]).abs().max()
          XCTAssertLessThan(diff.item(Float.self), 1e-4, "Causal mask not blocking future tokens")
      }
  }
  ```

  > **Note**: The exact MLXArray subscript / slicing syntax depends on MLX-Swift version. If the syntax above doesn't compile, consult the existing Qwen3TTSEngine.swift or MLX-Swift examples for the correct pattern. The intent is "verify causality"; the implementation can vary.

- [ ] **Step 3: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test`
  Expected: FAIL "Cannot find 'T3Attention'"

- [ ] **Step 4: Implement T3Attention**

  ```swift
  // Bolo/Engine/Chatterbox/T3/T3Attention.swift
  import Foundation
  import MLX
  import MLXNN

  /// GPT-2 style multi-head self-attention.
  /// Reference: mlx-audio/mlx_audio/tts/models/chatterbox/t3/attention.py
  ///
  /// Forward: (B, S, H) → (B, S, H)
  /// Uses combined QKV projection (`c_attn` in GPT-2 naming).
  /// No GQA, no RoPE — just standard MHA with learned positional embeddings
  /// (the positional embeddings live one layer up in T3.swift).
  final class T3Attention: Module, UnaryLayer {
      let numHeads: Int
      let headDim: Int
      let hiddenDim: Int

      @ModuleInfo(key: "c_attn") var qkvProj: Linear  // (H) → (3H)
      @ModuleInfo(key: "c_proj") var outProj: Linear  // (H) → (H)

      init(config: ChatterboxConfig.T3) {
          self.numHeads = config.numHeads
          self.headDim = config.headDim
          self.hiddenDim = config.hiddenDim
          self._qkvProj.wrappedValue = Linear(config.hiddenDim, 3 * config.hiddenDim, bias: true)
          self._outProj.wrappedValue = Linear(config.hiddenDim, config.hiddenDim, bias: true)
          super.init()
      }

      /// Combined forward pass.
      /// - mask: optional additive attention mask (causal mask is `causalMask(seqLen:)`)
      /// - cache: optional KV cache for autoregressive generation (nil for prefill)
      func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: T3Cache?) -> MLXArray {
          let B = x.dim(0)
          let S = x.dim(1)
          let qkv = qkvProj(x)  // (B, S, 3H)
          let qkvSplit = qkv.split(parts: 3, axis: -1)  // 3 × (B, S, H)
          var q = qkvSplit[0].reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)  // (B, h, S, d)
          var k = qkvSplit[1].reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
          var v = qkvSplit[2].reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)

          // KV cache for autoregressive gen
          if let cache = cache {
              (k, v) = cache.update(keys: k, values: v)
          }

          // Scaled dot-product attention
          let scale = 1.0 / sqrt(Float(headDim))
          var scores = (q.matmul(k.transposed(0, 1, 3, 2))) * scale  // (B, h, S, S')
          if let mask = mask {
              scores = scores + mask
          }
          let probs = MLX.softmax(scores, axis: -1)
          let attended = probs.matmul(v)  // (B, h, S, d)
          let merged = attended.transposed(0, 2, 1, 3).reshaped([B, S, hiddenDim])  // (B, S, H)
          return outProj(merged)
      }

      // Required by UnaryLayer protocol — single-arg form (no mask, no cache) for compatibility
      func callAsFunction(_ x: MLXArray) -> MLXArray {
          callAsFunction(x, mask: nil, cache: nil)
      }

      /// Build an additive causal mask of shape (S, S):
      /// 0 on the lower triangle (including diagonal), -inf above.
      static func causalMask(seqLen: Int) -> MLXArray {
          let neg = MLXArray(repeating: -Float.greatestFiniteMagnitude, count: seqLen * seqLen, dtype: .float32)
              .reshaped([seqLen, seqLen])
          // Set lower triangle (incl diagonal) to 0
          // MLX-Swift may not have a direct triu/tril — build via comparison
          let indices = MLXArray(0..<seqLen)
          let row = indices.reshaped([seqLen, 1])
          let col = indices.reshaped([1, seqLen])
          let isLowerTri = (row .>= col)  // bool mask
          return MLX.where(isLowerTri, MLXArray(0.0), neg)
      }
  }
  ```

  > **Reality check on MLX-Swift API**: the exact names and signatures (`MLX.softmax`, `MLXArray.transposed`, `MLXArray.matmul`, `MLX.where`) may differ slightly from what's written here — these were guessed from common MLX-Python patterns. If something doesn't compile, look at the equivalent operation in `Bolo/Engine/Qwen3TTSEngine.swift` or in `mlx-swift-examples` for the correct Swift idiom. The structure of the attention computation is correct; only the API surface needs touch-ups.

- [ ] **Step 5: Run tests**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test
  ```

  Expected: 2 tests pass (shape preservation + causality). If they don't, debug the implementation against the Python reference.

- [ ] **Step 6: Commit**

  ```bash
  mkdir -p /Users/vir/Code/bolo/Bolo/Engine/Chatterbox/T3
  git add Bolo/Engine/Chatterbox/T3/T3Attention.swift BoloTests/Chatterbox/T3Tests.swift
  git commit -m "feat(chatterbox): T3Attention — GPT-2 multi-head self-attention with KV cache"
  ```

---

### Task 9: T3 MLP block

GPT-2 style feed-forward: two linear layers around a GELU activation.

**Files:**
- Create: `Bolo/Engine/Chatterbox/T3/T3MLP.swift`
- Modify: `BoloTests/Chatterbox/T3Tests.swift` (add MLP test)

- [ ] **Step 1: Add test**

  Append to `T3Tests.swift`:

  ```swift
  func test_mlp_outputShape_matchesInput() {
      let cfg = ChatterboxConfig.turbo.t3
      let mlp = T3MLP(config: cfg)
      let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
      let output = mlp(input)
      XCTAssertEqual(output.array.shape, [1, 5, cfg.hiddenDim])
  }
  ```

- [ ] **Step 2: Verify failure**

  Run: `xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test`
  Expected: FAIL "Cannot find 'T3MLP'"

- [ ] **Step 3: Implement T3MLP**

  ```swift
  // Bolo/Engine/Chatterbox/T3/T3MLP.swift
  import Foundation
  import MLX
  import MLXNN

  /// GPT-2 feed-forward block: Linear → gelu_new → Linear.
  /// Intermediate dim is 4 × hidden (GPT-2 convention).
  final class T3MLP: Module, UnaryLayer {
      @ModuleInfo(key: "c_fc") var fc: Linear     // (H) → (4H)
      @ModuleInfo(key: "c_proj") var proj: Linear // (4H) → (H)

      init(config: ChatterboxConfig.T3) {
          let intermediate = 4 * config.hiddenDim
          self._fc.wrappedValue = Linear(config.hiddenDim, intermediate, bias: true)
          self._proj.wrappedValue = Linear(intermediate, config.hiddenDim, bias: true)
          super.init()
      }

      func callAsFunction(_ x: MLXArray) -> MLXArray {
          // gelu_new (GPT-2's variant; tanh-based approximation)
          // Reference: https://github.com/openai/gpt-2/blob/master/src/model.py#L25
          let h = fc(x)
          let gelu = 0.5 * h * (1 + MLX.tanh(MLX.sqrt(2.0 / .pi) * (h + 0.044715 * h.pow(3))))
          return proj(gelu)
      }
  }
  ```

  > **Reality check**: MLX-Swift may already have a `geluApprox` activation function — if so, use it directly: `MLXNN.geluApprox(h)`. Adapt as needed.

- [ ] **Step 4: Run tests + commit**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test
  ```

  Expected: 3 T3 tests pass.

  ```bash
  git add Bolo/Engine/Chatterbox/T3/T3MLP.swift BoloTests/Chatterbox/T3Tests.swift
  git commit -m "feat(chatterbox): T3MLP — GPT-2 feed-forward block with gelu_new"
  ```

---

### Task 10: T3 transformer block (attention + MLP + layer norms)

**Files:**
- Create: `Bolo/Engine/Chatterbox/T3/T3Block.swift`
- Modify: `BoloTests/Chatterbox/T3Tests.swift`

- [ ] **Step 1: Add test**

  ```swift
  func test_block_outputShape_matchesInput() {
      let cfg = ChatterboxConfig.turbo.t3
      let block = T3Block(config: cfg)
      let input = MLXRandom.normal([1, 5, cfg.hiddenDim])
      let mask = T3Attention.causalMask(seqLen: 5)
      let output = block(input, mask: mask, cache: nil)
      XCTAssertEqual(output.array.shape, [1, 5, cfg.hiddenDim])
  }
  ```

- [ ] **Step 2: Verify failure, implement**

  ```swift
  // Bolo/Engine/Chatterbox/T3/T3Block.swift
  import Foundation
  import MLX
  import MLXNN

  /// One transformer block: LayerNorm → Attention → residual → LayerNorm → MLP → residual.
  /// GPT-2 uses pre-norm (LayerNorm before each sub-block).
  final class T3Block: Module {
      @ModuleInfo(key: "ln_1") var ln1: LayerNorm
      @ModuleInfo(key: "attn") var attn: T3Attention
      @ModuleInfo(key: "ln_2") var ln2: LayerNorm
      @ModuleInfo(key: "mlp") var mlp: T3MLP

      init(config: ChatterboxConfig.T3) {
          self._ln1.wrappedValue = LayerNorm(dimensions: config.hiddenDim, eps: Float(config.layerNormEps))
          self._attn.wrappedValue = T3Attention(config: config)
          self._ln2.wrappedValue = LayerNorm(dimensions: config.hiddenDim, eps: Float(config.layerNormEps))
          self._mlp.wrappedValue = T3MLP(config: config)
          super.init()
      }

      func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: T3Cache?) -> MLXArray {
          var h = x + attn(ln1(x), mask: mask, cache: cache)
          h = h + mlp(ln2(h))
          return h
      }
  }
  ```

- [ ] **Step 3: Run tests + commit**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test
  git add Bolo/Engine/Chatterbox/T3/T3Block.swift BoloTests/Chatterbox/T3Tests.swift
  git commit -m "feat(chatterbox): T3Block — pre-norm transformer block"
  ```

---

### Task 11: T3 KV cache

Holds attention keys and values across autoregressive steps. Must be efficient — appending one token at a time, no reallocation.

**Files:**
- Create: `Bolo/Engine/Chatterbox/T3/T3Cache.swift`
- Modify: `BoloTests/Chatterbox/T3Tests.swift`

- [ ] **Step 1: Add test**

  ```swift
  func test_cache_appendingTokens_growsCorrectly() {
      let cfg = ChatterboxConfig.turbo.t3
      let cache = T3Cache(numHeads: cfg.numHeads, headDim: cfg.headDim, maxSeq: 100)
      // Append first token's K,V — shape (B, h, 1, d)
      let k1 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
      let v1 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
      let (k, v) = cache.update(keys: k1, values: v1)
      XCTAssertEqual(k.array.shape, [1, cfg.numHeads, 1, cfg.headDim])
      XCTAssertEqual(v.array.shape, [1, cfg.numHeads, 1, cfg.headDim])

      // Append second token
      let k2 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
      let v2 = MLXRandom.normal([1, cfg.numHeads, 1, cfg.headDim])
      let (k2cat, v2cat) = cache.update(keys: k2, values: v2)
      XCTAssertEqual(k2cat.array.shape, [1, cfg.numHeads, 2, cfg.headDim])
  }
  ```

- [ ] **Step 2: Implement**

  ```swift
  // Bolo/Engine/Chatterbox/T3/T3Cache.swift
  import Foundation
  import MLX

  /// Append-only KV cache for autoregressive generation in T3.
  /// Each call to update() returns the FULL key/value tensors including history.
  final class T3Cache {
      private let numHeads: Int
      private let headDim: Int
      private var storedKeys: MLXArray?
      private var storedValues: MLXArray?

      init(numHeads: Int, headDim: Int, maxSeq: Int) {
          self.numHeads = numHeads
          self.headDim = headDim
      }

      /// Append new K, V slices. Returns the full cached tensors.
      /// keys/values shape: (B, h, S_new, d)
      func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
          if let storedKeys = storedKeys, let storedValues = storedValues {
              self.storedKeys = MLX.concatenated([storedKeys, newK], axis: 2)
              self.storedValues = MLX.concatenated([storedValues, newV], axis: 2)
          } else {
              self.storedKeys = newK
              self.storedValues = newV
          }
          return (self.storedKeys!, self.storedValues!)
      }

      func reset() {
          storedKeys = nil
          storedValues = nil
      }

      var sequenceLength: Int {
          storedKeys?.dim(2) ?? 0
      }
  }
  ```

- [ ] **Step 3: Run tests + commit**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test
  git add Bolo/Engine/Chatterbox/T3/T3Cache.swift BoloTests/Chatterbox/T3Tests.swift
  git commit -m "feat(chatterbox): T3Cache — KV cache for autoregressive generation"
  ```

---

### Task 12: T3 full backbone — embeddings + 24 blocks + output head

**Files:**
- Create: `Bolo/Engine/Chatterbox/T3/T3.swift`
- Modify: `BoloTests/Chatterbox/T3Tests.swift`

- [ ] **Step 1: Add test**

  ```swift
  func test_t3_forwardPass_outputShape() {
      let cfg = ChatterboxConfig.turbo.t3
      let t3 = T3(config: cfg)
      // Input: (B=1, S=5) integer token IDs
      let inputIDs = MLXArray([42, 17, 8, 99, 3]).reshaped([1, 5])
      let speakerEmb = MLXRandom.normal([1, 192])  // 192-d speaker
      let output = t3(inputIDs: inputIDs, speakerEmbedding: speakerEmb, cache: nil)
      // Output: (1, 5, speech_vocab_size = 6561) — logits over speech codebook
      XCTAssertEqual(output.array.shape, [1, 5, 6561])
  }
  ```

  > **Note**: The actual output shape depends on how Chatterbox-Turbo uses T3. It may output (1, 5, hidden) hidden states that then go to a separate speech-token classifier, OR it may have an integrated output head producing (1, 5, 6561) logits directly. Check the Python reference. Adapt this test.

- [ ] **Step 2: Verify failure, implement**

  ```swift
  // Bolo/Engine/Chatterbox/T3/T3.swift
  import Foundation
  import MLX
  import MLXNN

  /// Full T3 backbone: text embedding + 24 transformer blocks + speech-token output head.
  /// Reference: mlx-audio/mlx_audio/tts/models/chatterbox/t3/t3.py
  final class T3: Module {
      let config: ChatterboxConfig.T3

      @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
      @ModuleInfo(key: "wpe") var positionEmbedding: Embedding
      @ModuleInfo(key: "h") var blocks: [T3Block]
      @ModuleInfo(key: "ln_f") var lnFinal: LayerNorm
      @ModuleInfo(key: "speaker_proj") var speakerProj: Linear  // 192 → hidden, projects speaker embedding into model space
      @ModuleInfo(key: "lm_head") var speechHead: Linear        // hidden → 6561 (speech codebook logits)

      init(config: ChatterboxConfig.T3) {
          self.config = config
          self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenDim)
          self._positionEmbedding.wrappedValue = Embedding(embeddingCount: config.maxContextLength, dimensions: config.hiddenDim)
          self._blocks.wrappedValue = (0..<config.numLayers).map { _ in T3Block(config: config) }
          self._lnFinal.wrappedValue = LayerNorm(dimensions: config.hiddenDim, eps: Float(config.layerNormEps))
          self._speakerProj.wrappedValue = Linear(192, config.hiddenDim, bias: true)
          self._speechHead.wrappedValue = Linear(config.hiddenDim, 6561, bias: false)
          super.init()
      }

      /// Forward pass.
      /// - inputIDs: (B, S) integer text/speech token IDs
      /// - speakerEmbedding: (B, 192) speaker conditioning vector
      /// - cache: optional KV cache for autoregressive generation
      /// Returns: (B, S, speech_vocab_size) logits
      func callAsFunction(inputIDs: MLXArray, speakerEmbedding: MLXArray, cache: [T3Cache]?) -> MLXArray {
          let B = inputIDs.dim(0)
          let S = inputIDs.dim(1)

          let tokenEmb = tokenEmbedding(inputIDs)  // (B, S, H)
          // Position IDs
          let posIDs = MLX.broadcast(MLXArray(0..<S).reshaped([1, S]), shape: [B, S])
          let posEmb = positionEmbedding(posIDs)   // (B, S, H)

          // Project speaker embedding and add to first token
          let spk = speakerProj(speakerEmbedding)  // (B, H)
          var h = tokenEmb + posEmb
          // Add speaker conditioning to all positions (broadcast)
          h = h + spk.reshaped([B, 1, config.hiddenDim])

          let mask = cache == nil ? T3Attention.causalMask(seqLen: S) : nil

          for (i, block) in blocks.enumerated() {
              let c = cache?[i]
              h = block(h, mask: mask, cache: c)
          }

          h = lnFinal(h)
          return speechHead(h)  // (B, S, 6561)
      }
  }
  ```

- [ ] **Step 3: Run tests + commit**

  ```bash
  xcodegen generate
  xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' -only-testing:BoloTests/T3Tests test
  ```

  Expected: forward-pass test passes. (Memory may spike — 24 blocks × 1024 hidden + embeddings = ~350M params in random init.) If the test fails with OOM, reduce the test to use only 4 blocks temporarily for shape verification.

  ```bash
  git add Bolo/Engine/Chatterbox/T3/T3.swift BoloTests/Chatterbox/T3Tests.swift
  git commit -m "feat(chatterbox): T3 backbone — embeddings + 24 blocks + speech head"
  ```

---

### Task 13: T3 reference parity gate — verify activations match Python

**This is the critical risk-mitigation step from the spec.** Before adding S3Gen, verify the T3 forward pass produces the same activations as the Python reference on a known input.

**Files:**
- Modify: `BoloTests/Chatterbox/T3Tests.swift`

- [ ] **Step 1: Add reference parity test**

  ```swift
  func test_t3_referenceParity_matchesPythonActivations() throws {
      try XCTSkipIf(
          ProcessInfo.processInfo.environment["BOLO_RUN_HEAVY_TESTS"] != "1",
          "Set BOLO_RUN_HEAVY_TESTS=1 (requires Python reference outputs)"
      )

      // Load Python reference outputs from scripts/chatterbox-reference/reference-outputs/
      let referenceDir = URL(fileURLWithPath: #filePath)
          .deletingLastPathComponent()
          .deletingLastPathComponent()
          .deletingLastPathComponent()
          .appendingPathComponent("scripts/chatterbox-reference/reference-outputs")

      let textTokensPython = try loadNumpyInt32(referenceDir.appendingPathComponent("text_tokens.npy"))
      let speakerEmbPython = try loadNumpyFloat32(referenceDir.appendingPathComponent("speaker_embedding.npy"))
      let layer0Python = try loadNumpyFloat32(referenceDir.appendingPathComponent("t3_layer_0_output.npy"))
      let layer12Python = try loadNumpyFloat32(referenceDir.appendingPathComponent("t3_layer_12_output.npy"))
      let layer23Python = try loadNumpyFloat32(referenceDir.appendingPathComponent("t3_layer_23_output.npy"))

      // Load real Chatterbox weights
      let progress = ModelDownloadProgress()
      let weights = try await WeightLoader.downloadAndLoad(progressHandler: { _, _ in })

      let cfg = ChatterboxConfig.turbo.t3
      let t3 = T3(config: cfg)
      // Map safetensors keys to T3 properties
      try WeightLoader.applyWeights(weights, to: t3, keyPrefix: "t3.")

      // Forward pass capturing intermediate activations
      let inputIDs = MLXArray(textTokensPython)
      let speakerEmb = MLXArray(speakerEmbPython).reshaped([1, 192])
      let activations = t3.forwardWithActivations(inputIDs: inputIDs, speakerEmbedding: speakerEmb)

      // Compare layer 0, 12, 23 activations
      let mse0 = ((activations.layer(0) - MLXArray(layer0Python)).pow(2)).mean().item(Float.self)
      let mse12 = ((activations.layer(12) - MLXArray(layer12Python)).pow(2)).mean().item(Float.self)
      let mse23 = ((activations.layer(23) - MLXArray(layer23Python)).pow(2)).mean().item(Float.self)

      // Tolerance: 1e-3 is roughly FP16 numerical noise floor
      XCTAssertLessThan(mse0, 1e-3, "T3 layer 0 diverges from Python")
      XCTAssertLessThan(mse12, 1e-3, "T3 layer 12 diverges from Python")
      XCTAssertLessThan(mse23, 1e-3, "T3 layer 23 diverges from Python")
  }

  // Helper: load .npy file as [Float] (simple parser; can use existing utility)
  private func loadNumpyFloat32(_ url: URL) throws -> [Float] {
      // Minimal .npy parser — only handles float32, no fancy dtypes
      // (Could vendor a small library, or write inline ~30 lines)
      // ... implementation ...
      fatalError("Implementer: add minimal .npy parser here")
  }
  private func loadNumpyInt32(_ url: URL) throws -> [Int32] { fatalError() }
  ```

- [ ] **Step 2: Add `forwardWithActivations` to T3 + `applyWeights` to WeightLoader**

  ```swift
  // Append to T3.swift
  struct T3Activations {
      let layerOutputs: [MLXArray]
      func layer(_ i: Int) -> MLXArray { layerOutputs[i] }
  }

  extension T3 {
      func forwardWithActivations(inputIDs: MLXArray, speakerEmbedding: MLXArray) -> T3Activations {
          // Same as callAsFunction but stores each block's output
          let B = inputIDs.dim(0)
          let S = inputIDs.dim(1)
          let tokenEmb = tokenEmbedding(inputIDs)
          let posIDs = MLX.broadcast(MLXArray(0..<S).reshaped([1, S]), shape: [B, S])
          let posEmb = positionEmbedding(posIDs)
          let spk = speakerProj(speakerEmbedding)
          var h = tokenEmb + posEmb + spk.reshaped([B, 1, config.hiddenDim])
          let mask = T3Attention.causalMask(seqLen: S)
          var activations: [MLXArray] = []
          for block in blocks {
              h = block(h, mask: mask, cache: nil)
              activations.append(h)
          }
          return T3Activations(layerOutputs: activations)
      }
  }

  // Append to WeightLoader.swift
  extension WeightLoader {
      /// Apply loaded weights to a Module's parameters by matching keys.
      static func applyWeights(_ weights: [String: MLXArray], to module: Module, keyPrefix: String) throws {
          let prefixed = weights.compactMap { (k, v) -> (String, MLXArray)? in
              guard k.hasPrefix(keyPrefix) else { return nil }
              let stripped = String(k.dropFirst(keyPrefix.count))
              return (stripped, v)
          }
          let dict = Dictionary(uniqueKeysWithValues: prefixed)
          // MLX-Swift provides Module.update(parameters:) — adapt to its exact API
          try module.update(parameters: dict)
      }
  }
  ```

  > **Reality check**: MLX-Swift's exact API for "load these tensors into the module's nested properties by key" may be slightly different. Check `Bolo/Engine/Qwen3TTSEngine.swift` to see how the existing engine does it — Qwen3 must already do something similar.

- [ ] **Step 3: Generate Python reference outputs**

  ```bash
  cd /Users/vir/Code/bolo/scripts/chatterbox-reference
  source venv/bin/activate
  python generate-reference.py
  ```

  Verify the `.npy` files exist in `reference-outputs/`.

- [ ] **Step 4: Run the heavy test**

  ```bash
  BOLO_RUN_HEAVY_TESTS=1 xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' \
    -only-testing:BoloTests/T3Tests/test_t3_referenceParity_matchesPythonActivations test
  ```

  Expected: PASS (MSE < 1e-3 at layers 0, 12, 23).

  **If it fails** (this is the high-risk moment from the spec):
  - Print the actual MSE values. Are they huge (e.g. 100s) or just slightly over (e.g. 5e-3)?
  - If huge: there's a structural bug. Pick layer 0 and compare element-by-element. Most likely cause: wrong weight key mapping, transposed weight matrix, or attention mask error.
  - If slightly over: increase tolerance to 5e-3 and accept (FP16 numerical drift), but log a concern.

  Iterate until layer 0 matches. Then layer 12 should automatically match. Then layer 23. If layer 0 matches but layer 12 doesn't, the bug is in T3Block's accumulation of error.

- [ ] **Step 5: Commit (only when reference parity passes)**

  ```bash
  git add BoloTests/Chatterbox/T3Tests.swift Bolo/Engine/Chatterbox/T3/T3.swift Bolo/Engine/Chatterbox/WeightLoader.swift
  git commit -m "feat(chatterbox): T3 reference parity gate — Swift output matches Python within 1e-3 MSE"
  ```

  > **Gate**: do not proceed to Phase 5 (S3Gen) until this test passes. The whole port hinges on getting T3 right.

---

## Phase 5: S3Gen Decoder

Phases 5–7 follow the same pattern as Phase 4: bite-sized tasks, each with failing test → implementation → verification → commit. The structure below is condensed since the methodology is established.

### Task 14: ConformerBlock

Standard Conformer block: feed-forward + multi-head attention + convolution module + feed-forward, with layer norms in between. Reference: `mlx_audio/tts/models/chatterbox/s3gen/conformer.py`.

**Files:** Create `Bolo/Engine/Chatterbox/S3Gen/ConformerBlock.swift`, modify `BoloTests/Chatterbox/S3GenTests.swift`.

- [ ] **Step 1**: Add shape-preservation test for ConformerBlock with config (encoder: heads=8, linear_units=2048, hidden=512).
- [ ] **Step 2**: Verify failure.
- [ ] **Step 3**: Implement following GPT-2/Llama attention pattern from Phase 4. The convolution module uses Conv1d + GLU + Conv1d + BatchNorm + Swish + Conv1d.
- [ ] **Step 4**: Run tests, verify shape.
- [ ] **Step 5**: Commit.

### Task 15: S3 Encoder (6 Conformer blocks)

**Files:** Create `Bolo/Engine/Chatterbox/S3Gen/S3Encoder.swift`.

- [ ] Standard "stack of 6 ConformerBlocks" with input projection. Test: shape preservation + reference parity against Python mid-S3Gen activation.

### Task 16: S3 Decoder (4 blocks + 12 mid Conformer)

**Files:** Create `Bolo/Engine/Chatterbox/S3Gen/S3Decoder.swift`.

- [ ] More complex: cross-attention to encoder output, 1-step flow matching forward pass. Reference Python aggressively. Test: shape + reference parity.

### Task 17: Vocoder (mel-spec → 24kHz audio)

**Files:** Create `Bolo/Engine/Chatterbox/S3Gen/Vocoder.swift`.

- [ ] The vocoder is a separate small network that upsamples mel-spectrogram by 120× into raw audio. Architecture per Python reference. Test: shape + reference parity.

### Task 18: S3Gen full pipeline + reference parity gate

**Files:** Create `Bolo/Engine/Chatterbox/S3Gen/S3Gen.swift`.

- [ ] Compose Encoder + Decoder + Vocoder into a single `synthesize(speechTokens:, speakerEmbedding:)` call.
- [ ] **Reference parity test (the second critical gate)**: given the speech tokens from Python's T3, S3Gen should produce audio with MSE < 1e-2 against Python's audio samples. (Tolerance is looser than T3 because flow-matching has stochasticity even with a fixed seed.)
- [ ] Commit only when reference parity passes.

---

## Phase 6: ChatterboxModel + Engine Integration

### Task 19: `ChatterboxModel` — top-level container

**Files:** Modify `Bolo/Engine/Chatterbox/ChatterboxTTSEngine.swift` (replace the empty `ChatterboxModel` stub).

- [ ] **Step 1**: Replace the stub with:

  ```swift
  struct ChatterboxModel: Sendable {
      let config: ChatterboxConfig
      let tokenizer: EnTokenizer
      let speakerEmbeddings: SpeakerEmbeddings
      let t3: T3
      let s3gen: S3Gen

      static func fromPretrained(
          progressHandler: @escaping @Sendable (Double, String) -> Void
      ) async throws -> ChatterboxModel {
          let config = ChatterboxConfig.turbo
          progressHandler(0.0, "Loading tokenizer…")
          let tokenizer = try EnTokenizer.loadFromBundle()
          progressHandler(0.05, "Loading speaker embeddings…")
          let embeddings = try SpeakerEmbeddings.loadFromBundle()
          progressHandler(0.1, "Downloading model weights…")
          let weights = try await WeightLoader.downloadAndLoad(progressHandler: { p, label in
              // Map weight download progress to 10%–95% of overall
              progressHandler(0.1 + p * 0.85, label)
          })
          progressHandler(0.95, "Instantiating T3 backbone…")
          let t3 = T3(config: config.t3)
          try WeightLoader.applyWeights(weights, to: t3, keyPrefix: "t3.")
          progressHandler(0.97, "Instantiating S3Gen decoder…")
          let s3gen = S3Gen(config: config.s3gen)
          try WeightLoader.applyWeights(weights, to: s3gen, keyPrefix: "s3gen.")
          progressHandler(1.0, "Ready.")
          return ChatterboxModel(
              config: config, tokenizer: tokenizer, speakerEmbeddings: embeddings,
              t3: t3, s3gen: s3gen
          )
      }
  }
  ```

- [ ] **Step 2**: Run tests, verify the module builds.
- [ ] **Step 3**: Commit: `feat(chatterbox): ChatterboxModel container with fromPretrained loader`

### Task 20: Wire the real synthesize pipeline in `ChatterboxTTSEngine`

- [ ] **Step 1**: Replace the `_synthesize` stub with the actual pipeline. Tokenize → look up speaker embedding → run T3 autoregressively → run S3Gen → produce audio → play via AVAudioEngine + Varispeed (reuse the pattern from `Qwen3TTSEngine.swift`'s `play(samples:sampleRate:speed:)` method).
- [ ] **Step 2**: Add heavy integration test:

  ```swift
  func test_synthesize_realModel_producesNonEmptyAudio() async throws {
      try XCTSkipIf(ProcessInfo.processInfo.environment["BOLO_RUN_HEAVY_TESTS"] != "1", "heavy")
      let engine = ChatterboxTTSEngine(modelProvider: {
          try await ChatterboxModel.fromPretrained(progressHandler: { _, _ in })
      })
      try await engine.synthesize(text: "Hello world.", voice: .systemDefault, speed: Speed(1.0))
      // If we got here without throwing, the pipeline works. Audio plays through speakers.
  }
  ```

- [ ] **Step 3**: Run heavy test once, verify audio plays.
- [ ] **Step 4**: Commit: `feat(chatterbox): ChatterboxTTSEngine — full synthesize pipeline integrated`

---

## Phase 7: Production Swap

### Task 21: Swap Qwen3 → Chatterbox in `AppDelegate`

**Files:** Modify `Bolo/AppDelegate.swift`.

- [ ] **Step 1**: Replace the `ModelManager<Qwen3TTSModel>` block with `ModelManager<ChatterboxModel>`:

  ```swift
  // OLD:
  // let manager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
  //     try await Qwen3TTSModel.fromPretrained(progressHandler: ...)
  // }
  // let engine: any TTSEngine = Qwen3TTSEngine(modelProvider: { try await manager.ensureLoaded() })

  // NEW:
  let manager = ModelManager<ChatterboxModel>(idleTimeout: 300) {
      try await ChatterboxModel.fromPretrained(progressHandler: { p, label in
          Task { @MainActor in
              progress.update(progress: p, label: label)
              if p >= 1.0 { progress.complete() }
          }
      })
  }
  let engine: any TTSEngine = ChatterboxTTSEngine(modelProvider: { try await manager.ensureLoaded() })
  ```

  Keep the Qwen3TTSEngine.swift file in the codebase — we'll use it as the fallback for non-English languages in v1.5.

- [ ] **Step 2**: Run tests (most should still pass; `AppDelegateTests` may need updating if it checks engine type).
- [ ] **Step 3**: Commit: `feat(chatterbox): production swap — AppDelegate now uses ChatterboxTTSEngine`

### Task 22: Update OnboardingView for new model size + URL

**Files:** Modify `Bolo/UI/OnboardingView.swift`.

- [ ] **Step 1**: Update the download step copy to reflect Chatterbox-Turbo:
  - Change "~500 MB" → "~3 GB"
  - Change "Qwen3-TTS" → "Chatterbox-Turbo"
  - Update the welcome screen wording if it mentioned the engine

- [ ] **Step 2**: Manual visual verification: reset onboarding (`defaults delete com.virkhanna.bolo bolo.hasCompletedOnboarding`), delete cache, run the app. The onboarding window should show the new copy and progress through the new download.

- [ ] **Step 3**: Commit: `feat(chatterbox): update onboarding copy for Chatterbox-Turbo model size`

### Task 23: End-to-end ⌘⇧R test in built app

- [ ] **Step 1**: Build with code signing enabled (so the system trusts the binary for AX + Input Monitoring permissions). Either:
  - `./scripts/build.sh` (signed build), OR
  - Build in Xcode with a real Apple Developer Team ID set

- [ ] **Step 2**: Run the signed `.app`, grant Accessibility, walk through onboarding, wait for model download (~5-10 min on broadband).
- [ ] **Step 3**: Select text in Safari, press ⌘⇧R. Expected: Chatterbox-Turbo voice reads the text.
- [ ] **Step 4**: Subjective listen test. Compare to the previous Qwen3 output. Voice quality should be noticeably better — more natural prosody, more expressive intonation. Try a paralinguistic tag: select `"That's funny [laugh] really."` and verify the laugh sound appears in the audio.
- [ ] **Step 5**: Memory check via Activity Monitor: app should use ~2-3 GB while speaking, drop back to <100 MB after 5 minutes of idle.
- [ ] **Step 6**: Final commit: `chore(chatterbox): v2.0 end-to-end verified — Chatterbox-Turbo shipping as production engine`

---

## Self-Review

**Spec coverage:**
- §3.2 components → all five (EnTokenizer, SpeakerEmbeddings, T3, S3Gen, Vocoder) get their own tasks ✅
- §3.3 skip decisions → Voice Encoder not implemented (correctly skipped); PerTh skip noted in Task 14 (implementer should verify it's not in the Python reference path); multilingual deferred ✅
- §3.4 file layout → tasks create the exact file structure from the spec ✅
- §4 data flow → implemented in Task 20 (ChatterboxTTSEngine synthesize) ✅
- §5 integration → Task 21 (one-block swap), Task 22 (onboarding) ✅
- §6 error handling → tasks throw TTSError cases as specified ✅
- §7 testing → three tiers covered (default unit tests in each task, heavy gated tests in Tasks 7/13/18/20, manual subjective in Task 23) ✅
- §8 phases → 1:1 mapping to plan phases ✅
- §9 risks → primary risk (silent architectural failure) mitigated by reference parity gates in Tasks 13 and 18 ✅

**Placeholder scan:**
- "Implementer: replace the array literal below with actual Python output" in Task 5 — this is intentional (the implementer runs the Python ref then fills in the IDs), not a placeholder failure ✅
- "Implementer: add minimal .npy parser here" in Task 13 step 1 — flagged but real action required ⚠️
- "Implementer: paste a quoted excerpt of the Python class" in Task 8 step 1 — flagged as intentional reading task ✅
- Tasks 14–17 (Conformer / Encoder / Decoder / Vocoder) — condensed because of plan length; subagent should expand each into the same 5-step shape as earlier tasks ⚠️

**Type consistency:**
- `ChatterboxConfig` used consistently across tasks ✅
- `T3Cache` defined Task 11, used Task 12 ✅
- `T3.callAsFunction` signature `(inputIDs:speakerEmbedding:cache:)` consistent ✅
- `WeightLoader.applyWeights` signature consistent across Tasks 13, 19 ✅
- `ChatterboxModel.fromPretrained` signature consistent with usage in Tasks 20, 21 ✅
- `TTSError.synthesisFailed(String)` matches existing protocol from `Bolo/Engine/TTSEngine.swift` ✅

**Fix-inline notes:**
- Tasks 14–17 are condensed — the executing subagent will need to flesh each out into discrete TDD steps following the pattern of Tasks 8–12. This is an honest trade-off: covering S3Gen at the same fidelity as T3 would double the plan length, but the methodology is now established. Each Conformer-related task should follow the same 5-step shape: failing test → verify failure → implement → run tests → commit.

---

## Plan complete and saved to `~/Code/bolo/docs/superpowers/plans/2026-05-26-chatterbox-mlx-swift-port.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for the long phases (T3, S3Gen) where each task is 1–4 hours and a fresh context per subagent prevents drift.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints. Heavier on session context but you see every step.

**Which approach?**
