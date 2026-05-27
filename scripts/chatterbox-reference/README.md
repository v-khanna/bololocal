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

**First run downloads ~2.99 GB of model weights from Hugging Face.** Subsequent runs use the cache at `~/.cache/huggingface/hub/`.

Outputs land in `./reference-outputs/`:
- `text_tokens.npy` — BPE-encoded text (shape: 1 × seq_len)
- `text_tokens_padded.npy` — tokens with SOT/EOT padding + CFG duplicate (shape: 2 × padded_len)
- `t3_cond_spk_emb.npy` — speaker embedding from built-in conds.safetensors
- `t3_cond_clap_emb.npy` — CLAP audio embedding
- `t3_cond_emotion_adv.npy` — emotion exaggeration scalar
- `speech_tokens_raw.npy` — T3 backbone output before filtering (shape: 2 × tokens, CFG pair)
- `speech_tokens.npy` — speech codebook tokens after filtering invalid tokens (shape: 1 × tokens)
- `speech_feat.npy` — S3Gen flow decoder output (mel-like feature; shape: 1 × frames × feat_dim)
- `audio_samples.npy` — final 24 kHz audio waveform (shape: num_samples)
- `metadata.json` — shapes, fixed inputs, generation params, sample rate

The Swift tests in `BoloTests/Chatterbox/` load these `.npy` files and compare against the
Swift port's outputs (gated by `BOLO_RUN_HEAVY_TESTS=1`).

## Architecture notes (for porting)

Chatterbox-Turbo has two main components:

**T3 backbone** — autoregressive transformer that converts text tokens to discrete speech tokens.
- Input: BPE text tokens + speaker/CLAP/emotion conditioning
- Output: discrete speech codebook tokens (vocab size ~8192)
- Uses classifier-free guidance (CFG): two forward passes (conditional + unconditional) are
  batched together; guidance is applied during sampling

**S3Gen decoder** — converts speech tokens to waveform in two stages:
- `flow_inference`: speech tokens → mel-like speech features (via flow matching)
- `hift_inference`: speech features → 24 kHz waveform (HiFi-GAN vocoder)

The default conditioning vector (speaker embedding + CLAP embedding + emotion scalar) is
loaded from `conds.safetensors` in the model directory and corresponds to a built-in neutral
voice. No reference audio clip is required for inference with the default voice.

## When to regenerate

- After updating the mlx-audio dependency
- After confirming any architectural change in the upstream Python implementation
- After the Swift port produces audio that sounds wrong (re-run to get fresh Python reference)
