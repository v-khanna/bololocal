# Deep Research Prompt: Porting Chatterbox to MLX-Swift on macOS

You are doing deep technical research. Output should be a long-form markdown document with sources cited inline for every claim.

## Context

I am building a macOS menu bar app (Apple Silicon only, macOS 15+) that uses an open-source TTS model to read selected text aloud in a natural AI voice — fully on-device, no cloud. The app is written in Swift 6 + SwiftUI, uses xcodegen for project management, and integrates SPM dependencies.

I currently use Qwen3-TTS via the [soniqo/speech-swift](https://github.com/soniqo/speech-swift) package (MLX-Swift native). The voice quality is "good" but not great. I want to switch to [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI — MIT-licensed, beat ElevenLabs in blind tests (65.3% vs 24.5%), but only available as PyTorch/CUDA code today.

The question I need answered: **Should I port Chatterbox to MLX-Swift myself, or has someone already done it?**

## What to investigate

### 1. Existing ports — any framework

Search exhaustively for any existing port of Chatterbox to Apple Silicon. Try these specific terms:

- `chatterbox mlx`
- `chatterbox swift`
- `chatterbox apple silicon`
- `chatterbox coreml`
- `chatterbox ane` (Apple Neural Engine)
- `chatterbox m1 m2 m3`
- `Chatterbox-Turbo MLX`
- `resemble-ai chatterbox port`

Look in:

- **GitHub** — public repos, forks, work-in-progress branches
- **Hugging Face** — converted weights, model cards in the `mlx-community` org or elsewhere
- **PyPI** — any Chatterbox-related packages
- Swift Package Index
- Reddit r/LocalLLaMA, r/MachineLearning, r/MLX
- Hacker News
- Apple Developer forums
- X/Twitter discussions involving the Chatterbox or MLX maintainers

For each port you find, document:

- Repo / link
- Author and last commit date (is it abandoned or active?)
- Completeness — full port? Partial? Just weight conversion?
- Quality of code, license, any issues reported
- Does it actually work? Any user reports?

### 2. Specifically check these projects

Some near-neighbors that may already include Chatterbox support:

- **[Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)** — does this Python MLX library include Chatterbox alongside Kokoro/Qwen3?
- **[soniqo/speech-swift](https://github.com/soniqo/speech-swift)** — does this Swift package include a Chatterbox module beyond Qwen3TTS / KokoroTTS?
- **[Jimmi42/chatterbox-tts-apple-silicon](https://huggingface.co/Jimmi42/chatterbox-tts-apple-silicon)** — what does this actually do? Is it MPS (PyTorch backend) or true MLX? Is it usable from Swift?
- **[AtomGradient/swift-qwen3-tts](https://github.com/AtomGradient/swift-qwen3-tts)** — any related Chatterbox work by this author?

### 3. Chatterbox vs Chatterbox-Turbo

Resemble released both a base Chatterbox and a distilled "Chatterbox-Turbo" variant. Investigate the differences:

- Architecture differences (Turbo is 350M distilled, but how?)
- How many inference steps does base require vs Turbo? (Reports say ~10 steps base, ~1 step Turbo)
- Are Turbo weights publicly downloadable on Hugging Face?
- Quality comparison: blind tests, MOS scores, listener reports
- Realistic speed on M-series Macs (if anyone has benchmarked)
- Which one should we target for a real-time reader app?

### 4. Chatterbox architecture deep-dive

Read the actual code in `resemble-ai/chatterbox` on GitHub. Document:

- The Llama-3 backbone — exactly which Llama-3 variant (Llama-3.1, 3.2, custom)?
- The speech decoder — is it autoregressive, diffusion-based, flow-matching, or something else?
- The voice conditioner — how does the reference-clip voice cloning work mechanically?
- The full forward pass: text → tokens → embeddings → ??? → audio samples
- Total parameter count of each component
- Weight file size and format
- Any custom CUDA kernels or PyTorch operations that wouldn't map cleanly to MLX

### 5. MLX-Swift readiness for this architecture

MLX-Swift (the Swift binding for Apple's MLX framework):

- Does it have Llama-3 implementations we can extract for the backbone? Reference: any examples in [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) or similar.
- Does it have the operators needed for Chatterbox's speech decoder? Specifically: depending on what the decoder is, we'd need conv1d, attention variants, possibly flow-matching or diffusion sampling code.
- Has anyone done a similar port — Llama-style backbone + audio decoder — in pure MLX-Swift?

### 6. Conversion path: PyTorch weights → MLX

For a Llama-based model with custom decoder:

- What's the standard pattern for converting `.safetensors` from PyTorch to MLX format?
- Key naming conventions — what renames have to happen?
- Are there existing scripts in `mlx-community` we can adapt?
- Common pitfalls (rotational embedding layouts, attention head ordering, etc.)

### 7. Effort estimate from someone who knows

If you can find any blog posts, conference talks, or Twitter threads from developers who've ported Llama-based TTS or speech models to MLX-Swift, surface their effort estimates and lessons learned. Look for:

- The original soniqo/speech-swift maintainer's writeup on porting Qwen3-TTS
- Any "I ported X to MLX-Swift" blog posts on Medium, dev.to, personal blogs
- MLX maintainers' notes on similar work

## Output format

Long-form markdown document. Structure:

```
# Chatterbox → MLX-Swift Port Feasibility Research

## Executive summary
(3-5 bullets — TL;DR of the whole report)

## Existing ports
(any work already done, with links)

## Chatterbox vs Turbo
(architectural and practical comparison)

## Chatterbox architecture
(what we'd be porting — layer-by-layer)

## MLX-Swift readiness
(does the framework have what we need?)

## Realistic effort estimate
(hours/days, breakdown by job, with confidence)

## Recommendation
(do it, don't do it, or wait — with reasoning)

## Sources
(all URLs cited)
```

Cite every fact with the URL it came from. If a claim is your own inference, label it clearly as such. End with a single sentence: "Bottom line: PORT YOURSELF / FORK EXISTING / SKIP — because..."
