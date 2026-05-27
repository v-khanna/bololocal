#!/usr/bin/env python3
"""
Generate end-to-end S3Gen reference outputs for Phase 5e (composition gate).

The existing generate-reference.py captures per-module references:
  - CFM uses pinned noise of shape (1, 80, 582) but uses prompt_feat *inside*
    conds (so the CFM output is the FULL T_total).
  - Vocoder uses prompt_feat (1, 500, 80) directly — NOT the CFM output.

This script wires them together: feed the CFM output (1, 80, 582) into the
post-CFM trimming (`feat[:, :, mel_len1:]`) → (1, 80, 82), then through the
vocoder. Produces:

  e2e_cfm_noise.bin                — same pinned noise as s3gen_cfm_noise (mirrored for clarity)
  e2e_sinegen_phases.bin           — sized for the SHORT post-trim audio
  e2e_sinegen_noise.bin            — sized for the SHORT post-trim audio
  e2e_speech_feat.bin              — the (1, 80, 82) trimmed mel handed to vocoder
  e2e_audio.bin                    — the FINAL (1, T_audio) waveform — parity target
  e2e_audio.wav                    — same, listenable
  e2e_metadata.json                — shapes and stats

The Swift test loads these and runs `ChatterboxPipeline.synthesizeFromSpeechTokens`
with the same pinned tensors, then compares the audio.
"""

import json
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

# ── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--mlx-community--chatterbox-turbo-fp16"
    / "snapshots"
)
# Find the snapshot directory.
snapshot_dirs = list(WEIGHTS_PATH.iterdir())
if not snapshot_dirs:
    raise FileNotFoundError(
        f"No HuggingFace snapshot found under {WEIGHTS_PATH}. "
        "Run generate-reference.py first to populate the cache."
    )
WEIGHTS_PATH = snapshot_dirs[0] / "model.safetensors"
CONDS_PATH = REPO_ROOT / "Bolo/Engine/Chatterbox/Resources/conds.safetensors"

OUTPUT_DIR = Path(__file__).resolve().parent / "reference-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_raw(arr, path: Path, dtype):
    """Save MLXArray as raw little-endian bytes + a sidecar JSON describing shape/dtype."""
    np_arr = np.array(arr).astype(dtype)
    np_arr.tofile(path)
    sidecar = path.with_suffix(path.suffix + ".shape.json")
    with open(sidecar, "w") as f:
        json.dump({"shape": list(np_arr.shape), "dtype": str(dtype.__name__)}, f)
    return np_arr


# ── Load existing reference inputs ─────────────────────────────────────────
print("Loading existing reference inputs ...")
# Speech tokens from previous T3 run
speech_tokens_np = np.fromfile(
    OUTPUT_DIR / "speech_tokens.bin", dtype=np.int32
).reshape(1, -1)
print(f"  speech_tokens: {speech_tokens_np.shape}")
speech_tokens = mx.array(speech_tokens_np)

# ── Load S3Gen weights and build full model ────────────────────────────────
print(f"\nLoading S3Gen weights from {WEIGHTS_PATH} ...")
s3gen_weights = {}
with safe_open(str(WEIGHTS_PATH), framework="mlx") as f:
    for k in f.keys():
        if k.startswith("s3gen."):
            s3gen_weights[k[len("s3gen."):]] = f.get_tensor(k)
print(f"  loaded {len(s3gen_weights)} s3gen.* parameters")

from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen as PyS3Gen

# Use meanflow=True for the Turbo configuration.
print("\nBuilding S3Token2Wav (meanflow=True) ...")
s3gen = PyS3Gen(meanflow=True)
s3gen_sanitized = s3gen.sanitize(s3gen_weights)
s3gen.load_weights(list(s3gen_sanitized.items()), strict=False)
s3gen.eval()
print("  built S3Gen (meanflow=True)")

# ── Conditioning ──────────────────────────────────────────────────────────
print(f"\nLoading conditioning from {CONDS_PATH} ...")
with safe_open(str(CONDS_PATH), framework="mlx") as f:
    gen_embedding = f.get_tensor("gen.embedding")              # (1, 192)
    gen_prompt_feat = f.get_tensor("gen.prompt_feat")          # (1, 500, 80)
    gen_prompt_token = f.get_tensor("gen.prompt_token")        # (1, 250)
    gen_prompt_token_len = f.get_tensor("gen.prompt_token_len")  # (1,)

ref_dict = {
    "prompt_token": gen_prompt_token,
    "prompt_token_len": gen_prompt_token_len,
    "prompt_feat": gen_prompt_feat,
    "embedding": gen_embedding,
}

# ── Replicate S3Token2Mel.__call__ with PINNED randomness ─────────────────
print("\nReplicating S3Token2Mel.__call__ deterministically ...")

B = speech_tokens.shape[0]
prompt_token = gen_prompt_token
prompt_token_len = gen_prompt_token_len
prompt_feat = gen_prompt_feat
embedding = gen_embedding

# Speaker affine projection.
embedding_norm = embedding / (mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)
embedding_proj = s3gen.spk_embed_affine_layer(embedding_norm)  # (B, 80)

# Token concatenation + mask.
token_len = mx.array([speech_tokens.shape[1]] * B)
token = mx.concatenate([prompt_token, speech_tokens], axis=1)
token_len = prompt_token_len + token_len

max_len = token.shape[1]
mask = mx.arange(max_len)[None, :] < token_len[:, None]
mask = mask[:, :, None].astype(mx.float32)

# Embed + encoder + projection.
token_emb = s3gen.input_embedding(token.astype(mx.int32)) * mask
h, h_masks = s3gen.encoder(token_emb, token_len)
h_lengths = mx.sum(h_masks[:, 0, :].astype(mx.int32), axis=-1)
mel_len1 = prompt_feat.shape[1]
mel_len2 = h.shape[1] - mel_len1
h_proj = s3gen.encoder_proj(h)

# Conds: prompt_feat in the prefix, zeros in the suffix.
zeros_padding = mx.zeros((B, mel_len2, 80))
conds = mx.concatenate([prompt_feat, zeros_padding], axis=1)
conds = conds.transpose(0, 2, 1)  # (B, 80, T_total)

# Decoder mask.
mask_dec = mx.arange(h.shape[1])[None, :] < h_lengths[:, None]
mask_dec = mask_dec[:, None, :].astype(mx.float32)

T_total = h.shape[1]
print(f"  T_total = {T_total} (= 2*(prompt={prompt_token.shape[1]} + speech={speech_tokens.shape[1]}))")
print(f"  mel_len1 (prompt_feat T) = {mel_len1}")
print(f"  mel_len2 (post-prompt) = {mel_len2}")

# ── Pin CFM noise ─────────────────────────────────────────────────────────
np.random.seed(7777)
cfm_noise_np = np.random.randn(B, 80, T_total).astype(np.float32)
cfm_noise = mx.array(cfm_noise_np)
mx.eval(cfm_noise)
print(f"\n  cfm_noise shape: {cfm_noise.shape}")
save_raw(cfm_noise, OUTPUT_DIR / "e2e_cfm_noise.bin", np.float32)

# meanflow=True also needs `noised_mels` per S3Token2Mel.__call__:
#   noised_mels = mx.random.normal((B, 80, speech_tokens.shape[1] * 2))
# This is spliced into the trailing part of `noise` inside the CFM solver.
# We pin this too with a separate seed to keep things bit-exactly replayable.
np.random.seed(8888)
noised_mels_np = np.random.randn(B, 80, speech_tokens.shape[1] * 2).astype(np.float32)
noised_mels = mx.array(noised_mels_np)
mx.eval(noised_mels)
print(f"  noised_mels shape: {noised_mels.shape}")
save_raw(noised_mels, OUTPUT_DIR / "e2e_noised_mels.bin", np.float32)

# Monkey-patch mx.random.normal to return our pinned noised_mels for the
# single call inside S3Token2Mel.__call__. We've already extracted the
# pre-existing call into the args of decoder() below, so just invoke
# decoder directly to get the feat.

# Use the CFM solver directly (bypass __call__).
print("\nRunning CFM (meanflow=True, n_timesteps=2) ...")
# The Python decoder has signature (mu, mask, n_timesteps, spks, cond, noised_mels, meanflow)
# but it samples noise INTERNALLY. We need to replicate _basic_euler with
# our pinned noise.

# Override the CFM's noise sample: do the splice manually.
# Python:
#   z = noise (sampled inside)
#   if noised_mels is not None:
#     prompt_len = mu.shape[2] - noised_mels.shape[2]
#     z = concat([z[:, :, :prompt_len], noised_mels], axis=2)
mu = h_proj.transpose(0, 2, 1)  # (B, 80, T_total)
spks = embedding_proj           # (B, 80)
cond = conds                    # (B, 80, T_total)
mask_for_cfm = mask_dec         # (B, 1, T_total)

z = cfm_noise
prompt_len = mu.shape[2] - noised_mels.shape[2]
if prompt_len > 0:
    z = mx.concatenate(
        [z[:, :, :prompt_len], noised_mels], axis=2
    )
else:
    z = noised_mels
print(f"  z (post-splice) shape: {z.shape}, prompt_len: {prompt_len}")

# Run the 2-step meanflow Euler solver.
mf_estimator = s3gen.decoder.estimator
n_cfm_timesteps = 2
t_span_cfm = mx.linspace(0, 1, n_cfm_timesteps + 1)  # [0, 0.5, 1.0]
print(f"  t_span: {t_span_cfm.tolist()}")

x = z
for i in range(n_cfm_timesteps):
    t = t_span_cfm[i : i + 1]
    r = t_span_cfm[i + 1 : i + 2]
    dxdt = mf_estimator(
        x=x, mask=mask_for_cfm, mu=mu, t=t, spks=spks, cond=cond, r=r
    )
    dt = r - t
    x = x + dt * dxdt
    mx.eval(x)
    print(f"    step {i}: t={float(t.item()):.4f} r={float(r.item()):.4f} "
          f"x_mean={float(x.mean()):+.5f} x_std={float(x.std()):.5f}")

speech_feat_full = x  # (B, 80, T_total)
mx.eval(speech_feat_full)
print(f"  speech_feat_full shape: {speech_feat_full.shape}")
save_raw(speech_feat_full, OUTPUT_DIR / "e2e_speech_feat_full.bin", np.float32)

# Drop the prompt prefix.
speech_feat = speech_feat_full[:, :, mel_len1:]  # (B, 80, mel_len2)
mx.eval(speech_feat)
print(f"  speech_feat (post-trim) shape: {speech_feat.shape}, "
      f"mean={float(speech_feat.mean()):+.5f}, std={float(speech_feat.std()):.5f}")
save_raw(speech_feat, OUTPUT_DIR / "e2e_speech_feat.bin", np.float32)

# ── Vocoder (with pinned SineGen randomness) ──────────────────────────────
print("\nRunning vocoder with pinned SineGen noise ...")
mel2wav = s3gen.mel2wav
speech_feat_T = speech_feat.transpose(0, 2, 1)  # (B, T_mel, 80)
mel_BCT = speech_feat  # (B, 80, T_mel) — already in this shape post-CFM
f0_pred = mel2wav.f0_predictor(mel_BCT)  # (B, T_mel)
mx.eval(f0_pred)
print(f"  f0_predictor out: {f0_pred.shape}")

f0_up = mel2wav._upsample_f0(f0_pred)
T_audio = f0_up.shape[1]
print(f"  T_audio: {T_audio}")

# Pin SineGen randomness.
NB_HARMONICS = 8
rng = np.random.RandomState(7777)
sinegen_random_phases_np = rng.uniform(
    -np.pi, np.pi, size=(B, NB_HARMONICS, 1)
).astype(np.float32)
sine_shape = (B, NB_HARMONICS + 1, T_audio)
sinegen_noise_np = rng.standard_normal(size=sine_shape).astype(np.float32)
sinegen_random_phases = mx.array(sinegen_random_phases_np)
sinegen_noise = mx.array(sinegen_noise_np)
save_raw(sinegen_random_phases, OUTPUT_DIR / "e2e_sinegen_phases.bin", np.float32)
save_raw(sinegen_noise, OUTPUT_DIR / "e2e_sinegen_noise.bin", np.float32)
print(f"  sinegen_random_phases shape: {sinegen_random_phases.shape}")
print(f"  sinegen_noise shape: {sinegen_noise.shape}")

# Replicate SineGen.__call__ deterministically.
f0_for_sinegen = f0_up.transpose(0, 2, 1)  # (B, 1, T_audio)
sinegen = mel2wav.m_source.l_sin_gen
B_sg, _, T_sg = f0_for_sinegen.shape
harmonics = mx.arange(1, sinegen.harmonic_num + 2)[None, :, None]
F_mat = f0_for_sinegen * harmonics / sinegen.sampling_rate
theta_mat = 2 * np.pi * mx.cumsum(F_mat, axis=-1)
theta_mat = theta_mat % (2 * np.pi)
zero_phase = mx.zeros((B_sg, 1, 1))
phase_vec = mx.concatenate([zero_phase, sinegen_random_phases], axis=1)
sine_waves = sinegen.sine_amp * mx.sin(theta_mat + phase_vec)
uv = (f0_for_sinegen > sinegen.voiced_threshold).astype(mx.float32)
noise_amp = uv * sinegen.noise_std + (1 - uv) * sinegen.sine_amp / 3
noise = noise_amp * sinegen_noise
sine_waves = sine_waves * uv + noise
sine_wavs_T = sine_waves.transpose(0, 2, 1)
sine_merge = mx.tanh(mel2wav.m_source.l_linear(sine_wavs_T))
source_signal = sine_merge.transpose(0, 2, 1)
mx.eval(source_signal)

# Final decode.
audio = mel2wav.decode(mel_BCT, source_signal)
mx.eval(audio)
print(f"  audio shape: {audio.shape}")

# Apply trim fade.
S3GEN_SR = 24000
n_trim = S3GEN_SR // 50  # 20ms = 480
fade = np.zeros(2 * n_trim, dtype=np.float32)
fade[n_trim:] = (np.cos(np.linspace(np.pi, 0, n_trim)) + 1) / 2
trim_fade = mx.array(fade)
fade_len = trim_fade.shape[0]
audio_np = np.array(audio)
if audio_np.shape[1] >= fade_len:
    audio_np[:, :fade_len] = audio_np[:, :fade_len] * np.array(trim_fade)
audio = mx.array(audio_np)
print(f"  audio after trim fade: shape={audio.shape}, "
      f"mean={float(audio.mean()):+.5f}, std={float(audio.std()):.5f}, "
      f"min={float(audio.min()):+.5f}, max={float(audio.max()):+.5f}")

save_raw(audio, OUTPUT_DIR / "e2e_audio.bin", np.float32)

# Listenable wav.
try:
    audio_clipped = np.clip(audio_np[0], -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    with wave.open(str(OUTPUT_DIR / "e2e_audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(S3GEN_SR)
        w.writeframes(audio_int16.tobytes())
    print(f"  wav written to {OUTPUT_DIR / 'e2e_audio.wav'}")
except Exception as e:
    print(f"  (skipped wav: {e})")

# ── Metadata ──────────────────────────────────────────────────────────────
metadata = {
    "purpose": "End-to-end S3Gen pipeline reference for Phase 5e composition gate",
    "speech_tokens_shape": list(speech_tokens.shape),
    "T_total": int(T_total),
    "mel_len1": int(mel_len1),
    "mel_len2": int(mel_len2),
    "cfm_noise_shape": list(cfm_noise.shape),
    "noised_mels_shape": list(noised_mels.shape),
    "speech_feat_shape": list(speech_feat.shape),
    "speech_feat_mean": float(speech_feat.mean()),
    "speech_feat_std": float(speech_feat.std()),
    "T_audio": int(T_audio),
    "audio_shape": list(audio.shape),
    "audio_mean": float(audio.mean()),
    "audio_std": float(audio.std()),
    "audio_min": float(audio.min()),
    "audio_max": float(audio.max()),
    "trim_fade_len": int(fade_len),
    "sampling_rate": S3GEN_SR,
    "cfm_noise_seed": 7777,
    "noised_mels_seed": 8888,
    "sinegen_noise_seed": 7777,
}
with open(OUTPUT_DIR / "e2e_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\nDone. End-to-end refs saved to {OUTPUT_DIR}/")
print(f"  audio length: {audio.shape[1]} samples = {audio.shape[1] / S3GEN_SR:.3f}s")
