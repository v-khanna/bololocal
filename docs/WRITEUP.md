# Porting Chatterbox-Turbo to MLX-Swift: a from-scratch on-device TTS engine

> Draft writeup / blog post. Fill the **`[BENCHMARK: …]`** blanks with real
> numbers before publishing, and add audio samples. This is the credential the
> work deserves — written for a technical reader (or a recruiter who can read code).

## TL;DR

I ported **Chatterbox-Turbo** — a state-of-the-art open TTS model — from its
Python/MLX reference to **native MLX-Swift**, from scratch, so it runs entirely
on-device on Apple Silicon with no Python runtime. The port reproduces the full
pipeline (a GPT-2-style T3 transformer → S3Gen token-to-mel decoder → HiFT
vocoder), is validated against the reference with **six numerical parity gates**,
and runs in 4-bit quantization with a KV cache. Weights:
`mlx-community/chatterbox-turbo-fp16`.

## Why

I wanted a Mac dictation/read-aloud tool whose *listen* half could run fully
offline. Cloud TTS is fast and great, but "your text never leaves the machine"
is a real feature for some people — and, honestly, porting a modern TTS stack to
Swift end-to-end is the kind of ML-systems work I wanted to prove I could do.
(The shipped product, [Bolo](https://github.com/v-khanna/bolo), ultimately uses
cloud Groq TTS by default for speed; this engine is the optional on-device path
and the real engineering story.)

## The model, in three stages

Chatterbox-Turbo isn't one network — it's a pipeline, and each stage was its own
port:

1. **T3 (text→speech-token transformer).** A **GPT-2-style** backbone (learned
   positional embeddings, LayerNorm, standard multi-head attention, `gelu_new`)
   — *not* the Llama-3 architecture some secondary sources claimed; I verified
   this against the live `config.json`, which simplified the port (no RoPE/GQA
   needed). Conditioning is richer than "just a speaker vector": it combines a
   speaker embedding, a CLAP-style embedding, and an emotion-advance term, with
   classifier-free guidance (CFG) at inference (text tokens duplicated into
   conditional/unconditional streams).
2. **S3Gen (speech-token→mel).** Decomposes into a flow stage and a vocoder
   stage. The flow stage is a **1-D U-Net conditional decoder** driving a
   **causal conditional flow-matching** module (Euler ODE solver + CFG) that
   turns discrete speech tokens into mel-like features.
3. **HiFTGenerator vocoder (mel→waveform).** Produces 24 kHz audio.

## Correctness: six parity gates

The scariest failure mode in an ML port is "it runs and produces *plausible
noise*." To prevent silent drift I built a Python reference harness (mlx-audio)
and gated each stage against it numerically:

- T3 backbone activations vs. reference (MSE thresholds).
- S3Gen flow features (`speech_feat` MSE) vs. reference.
- End-to-end audio-sample comparison.
- (Plus per-component gates for the conditional decoder, the CFM solver, and the
  vocoder.)

Every stage had to match the reference within tolerance before I trusted it.
`[BENCHMARK: cite the actual MSE thresholds / numbers from BoloTests/Chatterbox]`

## The hard parts (and the bugs)

The port wasn't "translate the Python line by line." The interesting failures:

- **Token-loop collapse → robotic 40s output.** Greedy `argmax` decoding made
  the T3 stage fall into repeating-token loops. Fixed with **stochastic
  sampling**: temperature ≈ 0.8 + a repetition penalty ≈ 1.2 + categorical
  sampling. This was the difference between "robotic" and "natural."
- **System-wide jetsam on 16GB Macs.** Running the model in fp16 exhausted unified
  memory and the OS started killing processes. Fix: **4-bit quantization** as the
  default — which then surfaced...
- **`MLXNN.quantize` crashing on the T3 FeedForward.** The stock quantize helper
  choked on the module array layout; I used a **hybrid two-pass quantize** to get
  around it.
- **O(N²) generation.** T3 attention recomputed the whole sequence each step;
  wiring a **KV cache** into `T3Attention` made generation O(N).
- **A `-10868` (kAudioUnitErr_FormatNotSupported) playback crash.** Came from an
  `AVAudioEngine` + varispeed graph; dropped the varispeed node and played the
  synthesized WAV directly.

## Results

Measured on an **Apple M5, 16 GB** (one synthesis run, 4-bit weights, model
already on disk). Reproduce with the `test_BENCHMARK_synthLatencyAndMemory`
benchmark in `BoloTests/Chatterbox/`.

| Metric | Value |
|---|---|
| Input | 80-character sentence |
| Audio produced | 4.56 s @ 24 kHz |
| Model load | 2.6 s |
| Synthesis | 10.3 s |
| **Real-time factor** | **~2.3×** (≈2.3 s of compute per 1 s of audio) |
| **Peak process memory** | **~9.1 GB** (phys_footprint, incl. test-runner overhead) |
| Model on disk | 2.8 GB (fp16); 4-bit at runtime |

`[AUDIO: add 2–3 sample clips — a neutral sentence, an expressive one]`

**Honest take:** on-device is **slower than real-time and memory-heavy** — which
is precisely why the shipped product ([Bolo](https://github.com/v-khanna/bolo))
defaults to cloud TTS and offers this engine as an opt-in. That tradeoff is the
point: I built the on-device path, *measured* it, and chose the default from the
data rather than guessing. The engine works, matches the reference within the
parity gates, and runs with no Python runtime — on a consumer 16 GB Mac.

## How it's structured (so it's reusable)

The engine is being extracted into a standalone, MIT-licensed Swift package,
**`ChatterboxMLX`**, depending only on `mlx-swift` — so it can be consumed by any
app, not just mine. `[TODO: link the package repo once published]`

## What I'd reuse for on-device dictation (the honest version)

People ask whether this TTS work "gives me" local *dictation* for free. It
doesn't — speech-to-text is a different model (Whisper, e.g. via WhisperKit), and
none of the Chatterbox model code transfers. What *would* carry over is the
**out-of-process, load-on-demand engine pattern** (run the heavy model in a
helper, keep the host app lean). Local dictation is a possible future, not a
shipped feature.

## Links

- Standalone engine package: `[TODO: ChatterboxMLX repo]`
- The product that uses it (cloud-default, on-device optional):
  [Bolo](https://github.com/v-khanna/bolo)
- Reference model weights: `mlx-community/chatterbox-turbo-fp16`
- Original model: Chatterbox-Turbo (Resemble AI, MIT)
