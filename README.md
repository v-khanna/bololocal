<h1 align="center">Bolo Local</h1>

<p align="center">
  <b>On-device read-aloud for macOS — a from-scratch <a href="https://github.com/resemble-ai/chatterbox">Chatterbox-Turbo</a> TTS engine ported to MLX-Swift.</b><br>
  Select text anywhere, press a hotkey, hear it spoken — entirely on your Mac. No cloud, no API key, no account.
</p>

<p align="center">
  <code>MIT</code> · <code>Apple Silicon</code> · <code>macOS 15+</code> · <code>MLX-Swift</code> · <code>zero network calls after install</code>
</p>

---

## What this is

**Bolo Local** is the on-device, offline sibling of **[Bolo](https://github.com/v-khanna/bolo)**
(a fast, cloud-default two-way voice app). Where Bolo uses cloud TTS for speed,
Bolo Local runs the **entire text-to-speech pipeline on your machine**.

It exists for two reasons:

1. **Privacy / offline** — your text never leaves the Mac. One network request,
   ever: the first-run model download.
2. **The engineering** — it's a ground-up port of a modern TTS model to native
   Swift. If you want the real story, read the **[writeup →](docs/WRITEUP.md)**.

> **Why "Bolo"?** *Bolo* (बोलो) is Hindi for **"speak."**

## The engine

Bolo Local is a from-scratch port of **Chatterbox-Turbo** (Resemble AI, MIT)
from its Python/MLX reference to **native MLX-Swift** — no Python runtime. It's
not one model but a three-stage pipeline, each ported and validated separately:

```
text ─▶ T3 (GPT-2-style transformer) ─▶ speech tokens
     ─▶ S3Gen (1-D U-Net + causal flow-matching) ─▶ mel features
     ─▶ HiFTGenerator vocoder ─▶ 24 kHz audio
```

- **GPT-2-style T3 backbone** (learned positional embeddings, LayerNorm, standard
  MHA, `gelu_new`) — verified against the live config, not the Llama-3 some
  sources assumed. Conditioning combines speaker + CLAP + emotion embeddings with
  classifier-free guidance.
- **S3Gen** — a 1-D U-Net conditional decoder driving a causal conditional
  flow-matching module (Euler ODE solver + CFG).
- **HiFTGenerator vocoder** — mel → 24 kHz waveform.
- **Runs in 4-bit** with a **KV cache** (O(N) generation).

### Correctness: six parity gates

The scary failure mode in an ML port is "it runs and produces *plausible noise*."
Every stage is gated numerically against the Python reference (MSE thresholds on
T3 activations, S3Gen `speech_feat`, and end-to-end audio) so drift can't slip
through. Details + the hard bugs (token-loop collapse, fp16 jetsam, the
`MLXNN.quantize` crash) are in the **[writeup](docs/WRITEUP.md)**.

## Quick start

```bash
brew install xcodegen
git clone https://github.com/v-khanna/bololocal ~/Code/bololocal
cd ~/Code/bololocal
xcodegen generate
xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' build
open build/Build/Products/Debug/Bolo.app
```

On first launch you'll grant Accessibility + download the model (~3 GB). Then
select text in any app and press your hotkey to hear it read aloud.

## Honest trade-offs

On-device is the harder, heavier path — and that's the point. Versus cloud:

| | Bolo (cloud) | Bolo Local (on-device) |
|---|---|---|
| Latency | ~1 s | slower (on-device synth) |
| RAM | tiny | multiple GB during synthesis |
| Download | none | ~3 GB model, once |
| Network | per request | **none after install** |
| API key | required | **none** |

If you want the fast everyday experience, use **[Bolo](https://github.com/v-khanna/bolo)**.
If you want it fully private/offline — or you want to see the engine — you're in
the right place.

## Roadmap

- **On-device dictation — coming soon.** Speech-to-text running fully locally too
  (via [WhisperKit](https://github.com/argmaxinc/WhisperKit) + a local LLM), so
  the *talk* half can be offline, not just *listen*. Note: dictation is a
  separate model from the TTS engine above — none of the Chatterbox code
  transfers; it reuses the same on-device / out-of-process pattern, not the model.

## Docs

| Doc | What's in it |
|---|---|
| [`docs/WRITEUP.md`](docs/WRITEUP.md) | **The engineering story** — how the port was built, the parity gates, the hard bugs. Start here. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, high and low level. |
| [`docs/DEVELOPING.md`](docs/DEVELOPING.md) | Build / test / debug guide. |

## Credits

- **[Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox)** — Resemble AI (MIT), the model this port reproduces.
- **[mlx-swift](https://github.com/ml-explore/mlx-swift)** — Apple's MLX for Swift, the runtime.
- Reference weights: [`mlx-community/chatterbox-turbo-fp16`](https://huggingface.co/mlx-community/chatterbox-turbo-fp16).
- The cloud-default product that links here: **[Bolo](https://github.com/v-khanna/bolo)**.

## License

MIT.
