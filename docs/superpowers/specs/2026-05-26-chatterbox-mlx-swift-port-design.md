# Chatterbox-Turbo → MLX-Swift Port: Design Spec

**Status:** Draft for review
**Date:** 2026-05-26
**Authors:** Vir + Claude
**Scope:** Bolo v2.0
**Companion docs:** [Research result](../../research/results/01-chatterbox-port-feasibility.md), [v1 architecture](../../ARCHITECTURE.md), [v1 plan](../plans/2026-05-26-bolo-menubar-tts.md)

---

## 1. Goal

Replace Bolo's current Qwen3-TTS engine with a native MLX-Swift port of **Chatterbox-Turbo** (350M params, 1-step distilled flow-matching decoder), giving the app voice quality that **measurably beats ElevenLabs in blind tests** while running 100% on-device on Apple Silicon.

One-sentence pitch: *Bolo reads selected text aloud in a voice indistinguishable from a human narrator, fully on your Mac, faster than real-time.*

## 2. Why this, why now

- The current Qwen3-TTS engine works but Vir explicitly rejects its voice quality. The base product can't ship to Setapp at this bar.
- Chatterbox-Turbo (Resemble AI, MIT license) is the highest-quality open-weight TTS model in 2026, validated by independent blind tests (65.3% vs ElevenLabs' 24.5%).
- The hardest piece of porting work — translating PyTorch math into MLX — has already been done by the open-source community in Python (`Blaizzy/mlx-audio`). Pre-converted MLX weights are on Hugging Face. We translate Python→Swift, not PyTorch→MLX.
- Every operator we need exists natively in MLX-Swift. Nothing to invent at the framework level. `soniqo/speech-swift`'s CosyVoice port proves flow-matching audio decoders work in Swift.
- The full effort is bounded at **50–66 hours of focused engineering** across 5 phases.

## 3. Architecture

### 3.1 What's being built

A new file tree under `Bolo/Engine/Chatterbox/` containing a complete Swift implementation of the Chatterbox-Turbo model, conforming to Bolo's existing `TTSEngine` protocol. The protocol-first design means **the rest of the app does not change** — Coordinator, PlaybackController, UI all keep working unchanged.

### 3.2 Five components (mapped from Python MLX reference)

```
┌────────────────────────────────────────────────────────────────────┐
│  Chatterbox-Turbo pipeline                                         │
│                                                                    │
│  English text                                                      │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────────────────┐                                          │
│  │ EnTokenizer (BPE)    │  ── ports Python EnTokenizer to Swift   │
│  └──────────┬───────────┘                                          │
│             │ [Int] text tokens                                    │
│             ▼                                                      │
│  ┌──────────────────────┐                                          │
│  │ SpeakerEmbedding     │  ── precomputed 192-d vector per voice  │
│  │ (preset, FP32)       │     bundled with app, no neural net     │
│  └──────────┬───────────┘                                          │
│             │ 192-d speaker vector + text tokens                   │
│             ▼                                                      │
│  ┌──────────────────────┐                                          │
│  │ T3 backbone          │  ── 350M Llama-3 variant                │
│  │ (autoregressive)     │     RoPE + SwiGLU + GQA + KV cache      │
│  └──────────┬───────────┘                                          │
│             │ [Int] speech tokens (6,561-codebook, 3^8)            │
│             ▼                                                      │
│  ┌──────────────────────┐                                          │
│  │ S3Gen decoder        │  ── 1D U-Net + 12 Conformer blocks      │
│  │ (1-step flow match)  │     1-step distilled, no iteration      │
│  └──────────┬───────────┘                                          │
│             │ mel-spectrogram                                      │
│             ▼                                                      │
│  ┌──────────────────────┐                                          │
│  │ Vocoder              │  ── built into S3Gen, 120× upsample     │
│  └──────────┬───────────┘                                          │
│             │                                                      │
│             ▼ [Float] @ 24kHz mono                                 │
│         🔊 AVAudioEngine playback                                  │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 What we deliberately skip in v1

| Component | Why we skip | When it returns |
|---|---|---|
| **Voice Encoder (CAMPPlus)** — neural net that produces 192-d embedding from a reference clip | Translating it is significant work. We bypass for v1 by **pre-computing embeddings** for 4–6 preset voices in Python, bundling the raw vectors with the app. | v1.1 if/when we add user voice cloning |
| **PerTh watermarking** | Optional under MIT. Adds inference overhead. We're not running a deepfake service. | Reconsider if Resemble's stance changes |
| **Multilingual support** | Turbo is English-only by design. Multilingual variant is a different model. | v1.5 with translation feature (see §10) |
| **Voice cloning from user audio** | Requires VE port + UX surface. | v1.1 once VE is ported |

### 3.4 File layout

```
Bolo/Engine/Chatterbox/
├── ChatterboxTTSEngine.swift          # actor, conforms to TTSEngine — the public face
├── EnTokenizer.swift                  # BPE text tokenizer; pure Swift, no MLX
├── SpeakerEmbeddings.swift            # bundled preset 192-d vectors loaded from .safetensors
├── T3.swift                           # 350M Llama-3 variant backbone (MLXNN.Module)
├── T3Cache.swift                      # KV cache for autoregressive generation
├── S3Gen.swift                        # 1D U-Net + Conformer flow-matching decoder
├── ConformerBlock.swift               # shared layer used inside S3Gen
├── Vocoder.swift                      # mel-spec → 24kHz audio (also inside S3Gen pipeline)
├── WeightLoader.swift                 # safetensors → Swift struct property mapping
└── Resources/
    ├── chatterbox-turbo-fp16.safetensors      # downloaded on first run (~700 MB)
    ├── chatterbox-tokenizer.json              # BPE vocab + merges (bundled, ~1 MB)
    └── speaker-embeddings.safetensors          # bundled preset voices (~30 KB)
```

`ChatterboxTTSEngine` slots in alongside `Qwen3TTSEngine` and `MockTTSEngine`. AppDelegate flips one line to make it the default. Qwen3 stays in the codebase as a fallback (and as the multilingual engine for v1.5).

### 3.5 Concurrency model

Same pattern as existing engines:

- `ChatterboxTTSEngine` is an `actor` — wraps the non-Sendable model state, serializes synthesize calls
- Engine conforms to `TTSEngine` (which is `Sendable`) via `nonisolated` methods that delegate to actor-internal `_synthesize`
- T3 and S3Gen are `MLXNN.Module` structs held inside the actor
- KV cache is reset per synthesize call
- Same `ModelManager<ChatterboxModel>` lazy-load + idle-unload semantics

## 4. Data flow (with concurrency annotations)

```
@MainActor Coordinator.handleHotkey()
  └─ playback.play(text, voice, speed, onComplete)
       │
       ▼
  @MainActor PlaybackController.play()
  └─ Task {  // detached
       │
       ▼
     await engine.synthesize(text, voice, speed)   // nonisolated, delegates to actor
       │
       ▼
     ┌─────────── on Chatterbox actor ────────────────┐
     │                                                │
     │ let model = try await modelProvider()           │
     │   └─ ModelManager.ensureLoaded()                │
     │       └─ load chatterbox-turbo-fp16.safetensors │
     │       └─ instantiate T3, S3Gen, EmbeddingStore  │
     │                                                │
     │ let textTokens = EnTokenizer.encode(text)       │
     │ let spkEmb = SpeakerEmbeddings[voice]           │
     │                                                │
     │ var speechTokens: [Int] = []                    │
     │ var cache = T3Cache()                           │
     │ while !isEOS(speechTokens.last) {               │
     │   let next = model.T3.forward(                  │
     │     textTokens, speechTokens, spkEmb, &cache    │
     │   )                                             │
     │   speechTokens.append(next)                     │
     │   if speechTokens.count % 32 == 0 { eval() }    │
     │ }                                               │
     │                                                │
     │ let mel = model.S3Gen.synthesize(               │
     │   speechTokens, spkEmb                          │
     │ )  // 1-step pass                               │
     │                                                │
     │ let audio = model.Vocoder.melToAudio(mel)       │
     │             // [Float] @ 24kHz                  │
     └─────────────────────────────────────────────────┘
       │
       ▼
     buffer = AVAudioPCMBuffer(samples: audio, sr: 24000)
     playerNode → varispeed (speed) → mainMixer
     await scheduleBuffer completion
     onComplete?()
  }
```

## 5. Integration with existing Bolo architecture

**No changes** to:
- `TTSEngine` protocol (Chatterbox conforms as-is)
- `PlaybackController` (works against any `TTSEngine`)
- `Coordinator`, `CoordinatorState`, `HotkeyManager`, `TextCaptureManager`
- `PopoverView`, `SettingsView`, `OnboardingView`
- `ModelManager<Model>` (generic — just swap type parameter)

**Minimal changes** to:
- `AppDelegate.applicationDidFinishLaunching` — one block swap:

  ```swift
  // BEFORE
  let manager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
      try await Qwen3TTSModel.fromPretrained(progressHandler: { ... })
  }
  let engine: any TTSEngine = Qwen3TTSEngine(modelProvider: { try await manager.ensureLoaded() })

  // AFTER
  let manager = ModelManager<ChatterboxModel>(idleTimeout: 300) {
      try await ChatterboxModel.fromPretrained(progressHandler: { ... })
  }
  let engine: any TTSEngine = ChatterboxTTSEngine(modelProvider: { try await manager.ensureLoaded() })
  ```

- `SettingsView` "Language" picker — replaced with "Voice" picker showing the 4–6 preset voices
- `OnboardingView` — same flow but new model URL and size string (~700 MB instead of ~500 MB)

That's it. The protocol-first v1 design pays off here.

## 6. Error handling

| Failure | Handling |
|---|---|
| First-run download fails (network, HF rate limit) | Retry with exponential backoff up to 3 times; surface to OnboardingView with retry button; `TTSError.synthesisFailed(.modelDownloadFailed)` |
| Weight loading fails (missing/extra keys, shape mismatch) | Fatal at init — log the specific key that failed, throw `TTSError.synthesisFailed` with detail. This is a developer error (key mapping wrong), not a user error. Tests should catch before ship. |
| Token vocab mismatch (text contains characters tokenizer doesn't know) | Tokenizer falls back to byte-level encoding; log a warning. No user-visible failure. |
| Speaker embedding for selected voice not found | Fall back to default voice; log warning |
| MLX OOM during T3 generation (rare; long input, low-memory device) | Catch, throw `TTSError.synthesisFailed(.outOfMemory)`. `ModelManager` unloads, retries once after GC. If still fails: surface to user. |
| S3Gen produces silent output (all-zeros sample buffer) | Throw `TTSError.synthesisFailed(.degeneratedOutput)`. Symptom of model corruption — log and ask user to delete cache and re-download. |
| Audio playback failure (mixer error) | Throw `TTSError.playbackFailed`; UI shows "audio device issue" |

## 7. Testing strategy

Three tiers, gated by environment:

**Tier 1 — Always run (default `xcodebuild test`)**

- Tokenizer round-trip: `encode(text) → decode(tokens) == text` for ~50 test sentences
- Speaker embedding loader: verify all bundled embeddings load to correct shape `(1, 192)`
- Weight key sanitization: load safetensors keys, verify against expected Swift property paths (no missing, no extra)
- TTSEngine protocol conformance: `ChatterboxTTSEngine()` instantiates, doesn't load model
- `Speed` clamping still works through Chatterbox (existing test should pass unchanged)

**Tier 2 — Gated by `BOLO_RUN_HEAVY_TESTS=1` (heavy integration)**

- Full synthesize against reference Python MLX output: same text, same seed, same speaker → audio samples within MSE tolerance (need to define exact bar — likely <1e-3 mean squared error of waveform)
- T3 intermediate activations vs Python reference, layer-by-layer
- S3Gen output mel-spectrogram vs Python reference

These tests download the model (~700 MB) on first run, cache locally. CI doesn't run them.

**Tier 3 — Manual verification (no automation)**

- Subjective listen-tests across a curated set:
  - Plain prose (NYT article)
  - Dialogue with quotation marks
  - Technical text with acronyms
  - Text with paralinguistic tags: "[laugh] that's funny" → expect a laugh sound followed by speech
  - Long passages (1000+ words) for prosody coherence
- Time-to-first-audio measurement: should be <500ms on M2+, <1s on M1
- Memory usage during synthesis: should stay under 2 GB peak with FP16
- 5-minute idle unload still works correctly

## 8. Implementation phases (preview for writing-plans)

This section is intentionally lightweight — the `writing-plans` skill will turn each phase into discrete TDD tasks.

| Phase | Scope | Effort | Confidence |
|---|---|---|---|
| **P1: Skeleton + scaffolding** | Bolo/Engine/Chatterbox/ created, empty Swift structs matching Python MLX hierarchy, project compiles cleanly, dummy ChatterboxTTSEngine throws "not implemented" | 4–6 h | High |
| **P2: Tokenizer + speaker embeddings** | EnTokenizer (BPE) ported and unit-tested round-trip. SpeakerEmbeddings loader works. Both can run with zero model load. | 6–8 h | High |
| **P3: Weight loading** | Download `mlx-community/chatterbox-turbo-fp16` safetensors. WeightLoader maps every key to a Swift property. Verify all tensor shapes match Python reference. Model object instantiates with weights populated but no inference yet. | 8–12 h | Medium |
| **P4: T3 backbone** | Llama-3 variant struct (reuse mlx-swift-examples). Autoregressive token generation with KV cache. Verify intermediate activations match Python reference at every layer on a fixed input. Generate full speech-token sequence for "hello world." | 14–18 h | Medium |
| **P5: S3Gen decoder** | 1D U-Net + Conformer blocks. 1-step flow matching. Vocoder upsample. Verify mel-spectrogram and final audio match Python output sample-by-sample (within MSE tolerance). | 10–14 h | Medium |
| **P6: ChatterboxTTSEngine + playback** | Wire actor wrapper, AVAudioPCMBuffer construction, AVAudioEngine + Varispeed playback chain (reuse from Qwen3TTSEngine). End-to-end synthesize call works. | 4–6 h | High |
| **P7: AppDelegate swap + onboarding update** | Replace Qwen3 with Chatterbox in production wiring. Update onboarding flow text, download size, model URL. End-to-end ⌘⇧R test passes. | 4–6 h | High |

**Total: 50–70 hours.** Matches research estimate (50–66).

**Optional follow-up (post-v2.0):**
- P8 (optional): 4-bit quantization via `QuantizedLinear` if memory pressure shows up on M1 8 GB devices
- P9 (optional): Voice Encoder port for user voice cloning (v1.1 scope)

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Silent architectural failure** — model runs but outputs noise / garbled audio. Caused by subtle Python↔Swift tensor indexing or broadcasting differences. | **High** (research flagged as primary risk) | High (eats days of debugging) | Layer-by-layer activation comparison against Python MLX reference during P4 and P5. Don't advance a module to "done" until its activations match Python within tolerance on a known input. |
| **Effort overrun** — 50–70 h grows to 100+ h | Medium | Medium | Time-box each phase. If P4 (T3) isn't done in 25 h, escalate. Worst case: fall back to Chatterbox base (10-step decoder, slower but architecturally simpler S3Gen). |
| **Memory pressure on 8 GB M1 Macs** — ~700 MB FP16 + app baseline could swap | Medium | Medium | Ship FP16 default. Detect `ProcessInfo.physicalMemory < 12 GB` at first run and offer 4-bit quantized variant (P8 follow-up). |
| **Hugging Face download fails** at high scale | Low | Low | Already handled — `ModelManager` retries with exponential backoff. Bundle download URL is configurable; could mirror to our own CDN if HF rate-limits. |
| **Tokenizer subtle bugs** — vocab edge cases produce mistokenized text → garbled speech | Medium | Medium | P2 includes a 50-sentence round-trip test against Python tokenizer reference. Catch before P4 starts. |
| **MLX-Swift framework breaking changes** during the port | Low | Low | Pin SPM dependency to specific version. Watch ml-explore/mlx-swift releases. |
| **Resemble AI revokes Chatterbox license** | Very low | High (legal scramble) | MIT license is irrevocable for existing weights. We have the weights cached. Even if they pulled it tomorrow, we keep shipping. |
| **PerTh watermarking skip is legally contested** | Very low | Medium | MIT license permits modification. We document the decision in PRIVACY.md. If Resemble objects, we add it back — it's a single forward pass, recoverable. |

## 10. Out of scope (deferred to v1.5+)

Capturing these explicitly so they don't sneak in:

- **Translation** (English → other languages) — v1.5. Will use a local LLM (Gemma 3 or similar via MLX-Swift) for translation, and either Qwen3-TTS or Chatterbox Multilingual for non-English output.
- **Multilingual TTS** — v1.5, paired with translation.
- **User voice cloning** — v1.1 once Voice Encoder is ported.
- **LLM-orchestrated context-aware reading** (Path B from earlier brainstorm) — v2.0 separate workstream.
- **Screenshot-aware "read what's on screen"** — v2.0.
- **HUD-style UI replacement** — v2.0 design refresh.
- **Audio export to .m4a / .wav** — no current demand, v2.x.
- **Reading queue / history** — v2.x.

## 11. Open questions / decisions still needed

- **How many preset voices ship in v1?** Research suggests 4–6. We need to listen to what the Python implementation outputs with various reference clips and pick. Suggestion: 6 voices spanning warm/neutral/crisp + male/female + younger/older. Decision deferred to P2.
- **Default speaker?** Likely the "warmest narrative voice" picked from the 6 presets. Decision deferred to P2.
- **Quantization at ship time?** Default FP16; auto-fall-back to 4-bit on <12 GB devices. Decision deferred to P8.
- **PerTh watermarking?** Skip in v1. Revisit if Resemble or external pressure makes it necessary.
- **What happens to Qwen3TTSEngine?** Keeps its place in the codebase for v1.5 multilingual support. Settings adds a hidden dev-only switch "use Qwen3 instead" for debugging.

## 12. Self-review (per brainstorming skill)

**Placeholder scan:** none — every section has substantive content.
**Internal consistency:** verified — architecture in §3 matches data flow in §4 matches phases in §8.
**Scope check:** focused on the Chatterbox port. Translation, HUD, screenshots all explicitly deferred to v1.5+ in §10. No subsystem decomposition needed.
**Ambiguity check:** the "verify activations match Python reference within tolerance" claim in P4/P5 doesn't define exact tolerance numbers — that's deferred to the implementation plan where we'll set MSE thresholds empirically once we have the Python reference running.

---

## 13. Approval gate

This spec describes a 50–70 hour engineering project that fundamentally changes Bolo's voice quality and sets the stage for the v1.5 translation feature. Before invoking `writing-plans` to generate the detailed task-by-task implementation plan, **the user (Vir) should review this document** and either approve or request changes.

After approval, the workflow is:
1. `writing-plans` skill produces a step-by-step implementation plan saved to `docs/superpowers/plans/`
2. `subagent-driven-development` skill executes the plan task-by-task with review checkpoints
3. v2.0 ships when all 7 phases pass verification
