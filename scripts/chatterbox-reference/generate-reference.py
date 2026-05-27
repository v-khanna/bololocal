#!/usr/bin/env python3
# scripts/chatterbox-reference/generate-reference.py
# Generates reference outputs from the Python mlx-audio implementation.
# Saves intermediate activations + final audio for later Swift port comparison.
#
# DO NOT RUN as part of Task 1 — downloads 2.99 GB on first run.
# Run manually once model weights are needed (Phase 4, Task 13).
#
# Usage:
#   source venv/bin/activate
#   python generate-reference.py

import os
import json
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_audio.tts.models.chatterbox.chatterbox import Model

OUTPUT_DIR = Path(__file__).parent / "reference-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed inputs for reproducibility — same text every time.
# Chatterbox-Turbo uses reference audio for voice conditioning, not a speaker_id.
# When loaded from the mlx-community repo, conds.safetensors provides a default
# built-in voice — no reference audio clip is required.
TEST_TEXT = "Hello world, this is a test of the Chatterbox text to speech system."

# Fixed generation parameters for reproducibility
GEN_PARAMS = dict(
    exaggeration=0.1,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0,
    max_new_tokens=1000,
)

MODEL_REPO = "mlx-community/chatterbox-turbo-fp16"

print(f"Loading Chatterbox-Turbo from {MODEL_REPO} ...")
print("(First run downloads ~2.99 GB; subsequent runs use HF cache at ~/.cache/huggingface/hub/)")
model = Model.from_pretrained(MODEL_REPO)
model.eval()

# ── Text tokenization ───────────────────────────────────────────────────────
print(f"\nTokenizing: {TEST_TEXT!r}")
from mlx_audio.tts.models.chatterbox.chatterbox import punc_norm
normalized_text = punc_norm(TEST_TEXT)
text_tokens = model.tokenizer.text_to_tokens(normalized_text)  # shape: (1, seq_len)
np.save(OUTPUT_DIR / "text_tokens.npy", np.array(text_tokens))
print(f"  text_tokens: {text_tokens.shape}")

# ── Conditionals (default built-in voice from conds.safetensors) ────────────
print("\nLoading built-in voice conditionals (from conds.safetensors in model dir)...")
if model._conds is None:
    raise RuntimeError(
        "model._conds is None — the model directory did not contain conds.safetensors. "
        "Supply a reference audio clip via model.prepare_conditionals() instead."
    )
conds = model._conds

# Save T3 conditioning tensors
t3_cond = conds.t3
np.save(OUTPUT_DIR / "t3_cond_spk_emb.npy", np.array(t3_cond.speaker_emb))
np.save(OUTPUT_DIR / "t3_cond_clap_emb.npy", np.array(t3_cond.clap_emb))
np.save(OUTPUT_DIR / "t3_cond_emotion_adv.npy", np.array(t3_cond.emotion_adv))
print(f"  t3_cond.speaker_emb: {t3_cond.speaker_emb.shape}")
print(f"  t3_cond.clap_emb: {t3_cond.clap_emb.shape}")
print(f"  t3_cond.emotion_adv: {t3_cond.emotion_adv.shape}")

# ── T3 backbone: text tokens → speech tokens ────────────────────────────────
print("\nRunning T3 backbone (text tokens → speech tokens)...")

# Replicate exactly what Model.generate does before calling t3.inference
sot = model.t3.hp.start_text_token
eot = model.t3.hp.stop_text_token
text_tokens_cfg = mx.concatenate([text_tokens, text_tokens], axis=0)  # CFG pair
sot_tokens = mx.full((text_tokens_cfg.shape[0], 1), sot, dtype=mx.int32)
eot_tokens = mx.full((text_tokens_cfg.shape[0], 1), eot, dtype=mx.int32)
text_tokens_padded = mx.concatenate([sot_tokens, text_tokens_cfg, eot_tokens], axis=1)

np.save(OUTPUT_DIR / "text_tokens_padded.npy", np.array(text_tokens_padded))

speech_tokens_raw = model.t3.inference(
    t3_cond=t3_cond,
    text_tokens=text_tokens_padded,
    max_new_tokens=GEN_PARAMS["max_new_tokens"],
    temperature=GEN_PARAMS["temperature"],
    cfg_weight=GEN_PARAMS["cfg_weight"],
    repetition_penalty=GEN_PARAMS["repetition_penalty"],
    min_p=GEN_PARAMS["min_p"],
    top_p=GEN_PARAMS["top_p"],
)
mx.eval(speech_tokens_raw)
np.save(OUTPUT_DIR / "speech_tokens_raw.npy", np.array(speech_tokens_raw))
print(f"  speech_tokens_raw: {speech_tokens_raw.shape}")

# Post-process: extract conditional batch, drop invalid tokens (mirrors Model.generate)
from mlx_audio.tts.models.chatterbox.chatterbox import drop_invalid_tokens, SPEECH_VOCAB_SIZE
speech_tokens = speech_tokens_raw[0:1]  # first of CFG pair
speech_tokens = drop_invalid_tokens(speech_tokens)
mask = speech_tokens < SPEECH_VOCAB_SIZE
valid_count = int(mx.sum(mask.astype(mx.int32)))
sorted_indices = mx.argsort(-mask.astype(mx.int32))
valid_indices = sorted_indices[:valid_count]
speech_tokens = mx.take(speech_tokens, valid_indices)
speech_tokens = mx.expand_dims(speech_tokens, 0)
mx.eval(speech_tokens)
np.save(OUTPUT_DIR / "speech_tokens.npy", np.array(speech_tokens))
print(f"  speech_tokens (post-filter): {speech_tokens.shape}")
print(f"  speech_tokens first 8: {speech_tokens[0, :8].tolist()}")

# ── S3Gen decoder: speech tokens → waveform ─────────────────────────────────
print("\nRunning S3Gen decoder (speech tokens → waveform)...")

# Run flow_inference separately so we can capture the mel/speech_feat
speech_feat = model.s3gen.flow_inference(
    speech_tokens=speech_tokens,
    ref_dict=conds.gen,
    finalize=True,
)
mx.eval(speech_feat)
np.save(OUTPUT_DIR / "speech_feat.npy", np.array(speech_feat))
print(f"  speech_feat (mel-like): {speech_feat.shape}")

# Run hift_inference to get the waveform from speech_feat
wav = model.s3gen.hift_inference(speech_feat=speech_feat)
mx.eval(wav)

# Flatten to 1D
if wav.ndim == 2:
    wav = wav.squeeze(0)

np.save(OUTPUT_DIR / "audio_samples.npy", np.array(wav))
print(f"  audio_samples: {wav.shape}  (sample_rate=24000)")

# ── Metadata ────────────────────────────────────────────────────────────────
metadata = {
    "test_text": TEST_TEXT,
    "model_repo": MODEL_REPO,
    "gen_params": GEN_PARAMS,
    "text_tokens_shape": list(text_tokens.shape),
    "speech_tokens_shape": list(speech_tokens.shape),
    "speech_tokens_first_8": [int(x) for x in np.array(speech_tokens[0, :8])],
    "speech_feat_shape": list(speech_feat.shape),
    "audio_sample_rate": 24000,
    "audio_num_samples": int(wav.shape[-1]),
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Reference outputs saved to {OUTPUT_DIR}/")
print(f"  text_tokens:          {text_tokens.shape}")
print(f"  speech_tokens:        {speech_tokens.shape}")
print(f"  speech_feat:          {speech_feat.shape}")
print(f"  audio_samples:        {wav.shape}  ({wav.shape[0] / 24000:.2f}s at 24 kHz)")
