# Deep Research Prompt: TTS Model State-of-the-Art 2026 — Local-Deployable Edition

You are doing deep technical research on the current state of the art in text-to-speech models, focused on what can actually run locally on consumer Apple Silicon hardware.

## Context

I'm building a macOS menu bar TTS app (Bolo) that runs fully on-device on Apple Silicon. I currently use Qwen3-TTS via MLX-Swift and want to know if there's a better option for my constraints. I'm planning to port Chatterbox to MLX-Swift as a v2 upgrade — but I want to make sure I'm targeting the right model before sinking a week into the port.

## What to investigate

### 1. The full TTS model landscape in late 2026

Closed / cloud-only models (for benchmark reference only — we can't use these):

- ElevenLabs v3
- Cartesia Sonic
- OpenAI tts-1-hd, gpt-4o-tts
- Google Cloud Text-to-Speech
- Microsoft Azure Neural TTS
- PlayHT
- Resemble.ai (Resemble Pro)
- Suno (audio generation more broadly)
- Anything new released in the last 6 months I might've missed

Open-weight models (the relevant ones for us):

- Chatterbox / Chatterbox-Turbo (Resemble AI)
- Sesame CSM-1B (Sesame AI Labs)
- Fish Speech S1, S2 Pro (Fish Audio / OpenAudio)
- Qwen3-TTS (Alibaba)
- Kokoro-82M
- F5-TTS
- MeloTTS
- VibeVoice
- Dia2
- StyleTTS2
- OpenVoice / OpenVoice v2
- XTTS-v2 (Coqui)
- Anything new released in the last 3 months

For every model, document:

- License (MIT, Apache 2.0, CC-BY-NC, custom restrictive — be exact)
- Parameter count
- Model file size on disk
- Languages supported
- Voice library size (presets) or voice-cloning support
- Expressive capabilities (emotion knobs, natural-language instructions, audio tags)
- Streaming support (token-by-token audio output)
- Quality bar — MOS scores, blind test results vs ElevenLabs, listener reports
- Inference framework support (PyTorch, MLX-Python, MLX-Swift, CoreML, llama.cpp, ONNX)
- Real-time factor on M2/M3/M4 if reported (1x = real-time, 2x = twice as fast as real-time)

### 2. The Apple Silicon deployability question

For each open-weight model, answer: **can this actually run on a 16GB M2 Mac?**

Specifically:
- Memory footprint at inference (peak)
- Generation speed on M-series (real-time factor)
- Quality with quantization (if 4-bit or 8-bit quantized — how much quality is lost?)
- Whether a Swift-native runtime exists or only Python/MPS

### 3. Three specific deep-dives

For these three models — the most relevant to Bolo — give a thorough writeup each:

**Chatterbox / Chatterbox-Turbo** (Resemble AI):
- Architecture (Llama backbone + decoder)
- Weight file location on Hugging Face
- Real-world quality (find demo audio samples and describe what they sound like)
- Inference speed reports
- Current framework support — PyTorch only? Anyone working on MLX/CoreML port?
- Voice cloning capability

**Sesame CSM-1B**:
- Same questions
- The "transcends uncanny valley" claim — is it real per blind testing?
- 8 GB VRAM requirement reported elsewhere — is this true for Apple Silicon?

**OpenAudio S1 / Fish Speech S2 Pro**:
- Same questions
- The TTS-Arena #1 ranking claim — does it hold up?
- The Fish Audio internal blind test (S2 Pro beat ElevenLabs V3 60/40) — independently verifiable?

### 4. The "expressiveness" feature matrix

Build a feature matrix:

|  Model | Emotion knob | Natural-language instructions | Voice cloning | Streaming | Local Swift runtime | License |
|---|---|---|---|---|---|---|
| Chatterbox | ✓ | ? | ✓ | ? | ✗ | MIT |
| Qwen3-TTS | ? | ✓ (instruct param) | ✓ (speaker param) | ✓ | ✓ (speech-swift) | Apache 2.0 |
| ...etc | | | | | | |

### 5. The honest rankings

Answer these specific questions with your best assessment:

**a) Best open-weight local-deployable model for a Mac app in 2026** (highest voice quality, runs on 16GB M2, permissive license):

**b) Best lightweight option** (smallest model that still sounds expressive):

**c) Best voice cloning quality** in open weights:

**d) Best expressive control** (emotion / tone / context-awareness):

**e) Most active community / future-proofing**:

For each, rank top 3 with reasoning.

### 6. Trajectory check — what's coming next

What's expected to drop in the next 3-6 months? Any major open releases anticipated? Major framework improvements (MLX-Swift, etc.)? Anything that would change a "port Chatterbox now" decision?

### 7. The "should we actually do the Chatterbox port" verdict

Given everything above, your honest take:

- Is Chatterbox the right model to port to MLX-Swift in May 2026?
- Or is there a better candidate I'm missing?
- Or should we wait 1-3 months and see what drops?

## Output format

Long-form markdown:

```
# TTS SOA 2026 — Local Deployable Edition

## TL;DR
(5 bullets)

## The 2026 landscape
(all models, brief overview)

## Expressiveness feature matrix
(the table)

## Apple Silicon deployability
(memory + speed analysis)

## Deep-dive: Chatterbox
## Deep-dive: Sesame CSM-1B
## Deep-dive: OpenAudio S1

## Rankings
(answer all 5 ranking questions)

## What's coming next
(near-term trajectory)

## Recommendation for Bolo
(do the port, port something else, wait)

## Sources
```

End with one sentence: "If you can only port one model to MLX-Swift in 2026, it should be: ____"
