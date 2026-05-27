#!/usr/bin/env python3
# scripts/chatterbox-reference/generate-reference.py
#
# Generates reference outputs from the Python mlx-audio chatterbox_turbo
# implementation for layer-by-layer parity verification of the Swift port.
#
# Outputs (saved to ./reference-outputs/):
#   text_tokens.npy              - tokenized text (int32, shape [1, T])
#   t3_cond_spk_emb.npy          - 256-d speaker embedding from conds.safetensors
#   t3_cond_prompt_tokens.npy    - 375-token speech prompt from conds
#   tfmr_inputs_embeds.npy       - [cond | text | speech_start] before wpe (B,L,1024)
#   tfmr_after_wpe.npy           - inputs_embeds + wpe (B,L,1024) prefill input to block 0
#   tfmr_block_0_out.npy         - prefill output of GPT-2 block 0 (B,L,1024)
#   tfmr_block_12_out.npy        - prefill output of GPT-2 block 12 (B,L,1024)
#   tfmr_block_23_out.npy        - prefill output of GPT-2 block 23 (B,L,1024)
#   tfmr_ln_f_out.npy            - final layer norm output, prefill (B,L,1024)
#   speech_logits_first.npy      - speech_head(last hidden) BEFORE sampling (B, 6563)
#   speech_tokens.npy            - generated speech tokens (int32, shape [1, N])
#   metadata.json
#
# This script uses chatterbox_turbo.models.t3.T3 directly (GPT-2 backbone),
# bypasses the high-level ChatterboxTurboTTS class (which requires gated repo
# access), and loads weights directly from mlx-community/chatterbox-turbo-fp16.
#
# Usage:
#   source venv/bin/activate
#   python generate-reference.py

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

from mlx_audio.tts.models.chatterbox_turbo.models.t3 import T3
from mlx_audio.tts.models.chatterbox_turbo.models.t3.t3_config import T3Config
from mlx_audio.tts.models.chatterbox_turbo.models.t3.cond_enc import T3Cond


def save_raw(arr, path: Path, dtype):
    """Save an MLX or NumPy array as raw little-endian bytes (no header).
    Also saves a `.shape.json` sidecar with shape + dtype for Swift to parse."""
    np_arr = np.array(arr).astype(dtype)
    np_arr.tofile(path)
    with open(path.with_suffix(path.suffix + ".shape.json"), "w") as f:
        json.dump({"shape": list(np_arr.shape), "dtype": str(np_arr.dtype)}, f)

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "reference-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = (
    Path.home() / ".cache/huggingface/hub"
    / "models--mlx-community--chatterbox-turbo-fp16"
    / "snapshots/b2d0a13aa7cfff0a06d9acb247ae91c8f19a6d75"
    / "model.safetensors"
)
CONDS_PATH = SCRIPT_DIR.parent.parent / "Bolo/Engine/Chatterbox/Resources/conds.safetensors"
TOKENIZER_DIR = SCRIPT_DIR.parent.parent / "Bolo/Engine/Chatterbox/Resources"

# ── Inputs (fixed for reproducibility) ───────────────────────────────────────
TEST_TEXT = "Hello world, this is a test of the Chatterbox text to speech system."
SEED = 42
GEN_PARAMS = dict(
    temperature=0.8,
    top_k=1000,
    top_p=0.95,
    repetition_penalty=1.2,
    max_gen_len=40,
)

# ── Load T3 weights ─────────────────────────────────────────────────────────
print(f"Loading T3 weights from {WEIGHTS_PATH} ...")
weights = {}
with safe_open(str(WEIGHTS_PATH), framework="mlx") as f:
    for k in f.keys():
        if k.startswith("t3."):
            weights[k[3:]] = f.get_tensor(k)
print(f"  loaded {len(weights)} T3 parameters")

# ── Build T3 model ──────────────────────────────────────────────────────────
print("\nBuilding T3 (chatterbox_turbo, GPT-2 backbone) ...")
hp = T3Config()
t3 = T3(hp=hp)
t3.load_weights(list(weights.items()), strict=True)
t3.eval()

# ── Load conditionals ───────────────────────────────────────────────────────
print(f"\nLoading conditionals from {CONDS_PATH} ...")
with safe_open(str(CONDS_PATH), framework="mlx") as f:
    spk_emb = f.get_tensor("t3.speaker_emb")  # (1, 256)
    cond_prompt_tokens = f.get_tensor("t3.cond_prompt_speech_tokens")  # (1, 375) int32
print(f"  speaker_emb: {spk_emb.shape}")
print(f"  cond_prompt_speech_tokens: {cond_prompt_tokens.shape}")

t3_cond = T3Cond(
    speaker_emb=spk_emb,
    clap_emb=None,
    cond_prompt_speech_tokens=cond_prompt_tokens,
    cond_prompt_speech_emb=None,
    emotion_adv=None,
)

np.save(OUTPUT_DIR / "t3_cond_spk_emb.npy", np.array(spk_emb))
np.save(OUTPUT_DIR / "t3_cond_prompt_tokens.npy", np.array(cond_prompt_tokens))
save_raw(spk_emb, OUTPUT_DIR / "t3_cond_spk_emb.bin", np.float32)
save_raw(cond_prompt_tokens, OUTPUT_DIR / "t3_cond_prompt_tokens.bin", np.int32)

# ── Tokenize text ───────────────────────────────────────────────────────────
print(f"\nTokenizing: {TEST_TEXT!r}")
tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
out = tok(TEST_TEXT, return_tensors="np")
text_tokens = mx.array(out.input_ids)
print(f"  text_tokens: {text_tokens.shape} = {text_tokens[0].tolist()}")
np.save(OUTPUT_DIR / "text_tokens.npy", np.array(text_tokens))
save_raw(text_tokens, OUTPUT_DIR / "text_tokens.bin", np.int32)

# ── Capture inputs_embeds for prefill (replicate inference_turbo step 1) ────
print("\nCapturing prefill inputs_embeds ...")
B = text_tokens.shape[0]
speech_start = mx.ones((B, 1), dtype=mx.int32) * hp.start_speech_token  # 6561

inputs_embeds, len_cond = t3.prepare_input_embeds(
    t3_cond=t3_cond,
    text_tokens=text_tokens,
    speech_tokens=speech_start,
)
mx.eval(inputs_embeds)
print(f"  inputs_embeds: {inputs_embeds.shape}  (cond_len={len_cond})")
np.save(OUTPUT_DIR / "tfmr_inputs_embeds.npy", np.array(inputs_embeds))
save_raw(inputs_embeds, OUTPUT_DIR / "tfmr_inputs_embeds.bin", np.float32)

# ── Replicate GPT-2 prefill forward, capturing intermediate activations ────
print("\nManual GPT-2 forward pass to capture intermediates ...")
tfmr = t3.tfmr
B_, T_, _ = inputs_embeds.shape

# wpe: position embedding for positions [0..T_-1] (prefill, no cache offset)
positions = mx.arange(0, T_)
position_embeds = tfmr.wpe(positions)
h = inputs_embeds + position_embeds
mx.eval(h)
np.save(OUTPUT_DIR / "tfmr_after_wpe.npy", np.array(h))
save_raw(h, OUTPUT_DIR / "tfmr_after_wpe.bin", np.float32)
print(f"  after wpe: {h.shape}")

# Manually iterate blocks, saving outputs at 0, 12, 23
from mlx_lm.models.cache import KVCache
caches = [KVCache() for _ in range(len(tfmr.h))]

for i, block in enumerate(tfmr.h):
    h, _ = block(h, attention_mask=None, cache=caches[i])
    if i in (0, 12, 23):
        mx.eval(h)
        np.save(OUTPUT_DIR / f"tfmr_block_{i}_out.npy", np.array(h))
        save_raw(h, OUTPUT_DIR / f"tfmr_block_{i}_out.bin", np.float32)
        print(f"  block {i:>2} out: {h.shape}, mean={h.mean().item():+.5f}, std={h.std().item():.5f}")

h = tfmr.ln_f(h)
mx.eval(h)
np.save(OUTPUT_DIR / "tfmr_ln_f_out.npy", np.array(h))
save_raw(h, OUTPUT_DIR / "tfmr_ln_f_out.bin", np.float32)
print(f"  after ln_f: {h.shape}, mean={h.mean().item():+.5f}, std={h.std().item():.5f}")

# ── First speech logits (the model's first-token prediction) ────────────────
speech_logits_first = t3.speech_head(h[:, -1:, :])  # (B, 1, 6563)
mx.eval(speech_logits_first)
speech_logits_first_2d = speech_logits_first.squeeze(1)  # (B, 6563)
np.save(OUTPUT_DIR / "speech_logits_first.npy", np.array(speech_logits_first_2d))
save_raw(speech_logits_first_2d, OUTPUT_DIR / "speech_logits_first.bin", np.float32)
argmax_first = int(mx.argmax(speech_logits_first_2d[0]).item())
print(f"  speech_logits_first: {speech_logits_first_2d.shape}")
print(f"    argmax = {argmax_first}, top-1 logit = {speech_logits_first_2d[0, argmax_first].item():.4f}")

# ── Full inference for end-to-end reference ─────────────────────────────────
print("\nRunning full t3.inference_turbo ...")
mx.random.seed(SEED)
speech_tokens = t3.inference_turbo(
    t3_cond=t3_cond,
    text_tokens=text_tokens,
    **GEN_PARAMS,
)
mx.eval(speech_tokens)
print(f"  speech_tokens: {speech_tokens.shape}")
print(f"  first 8: {speech_tokens[0, :8].tolist()}")
np.save(OUTPUT_DIR / "speech_tokens.npy", np.array(speech_tokens))
save_raw(speech_tokens, OUTPUT_DIR / "speech_tokens.bin", np.int32)

# ───────────────────────────────────────────────────────────────────────────
# S3Gen reference outputs (Phase 5)
# ───────────────────────────────────────────────────────────────────────────
# We exercise the S3Gen pipeline as far as we can deterministically. The full
# flow goes:
#     speech_tokens + ref_dict
#       → concat [prompt_token | speech_tokens]
#       → input_embedding              (Embedding)
#       → encoder (UpsampleConformer)  ← captured here as `s3gen_encoder_out`
#       → encoder_proj                 (Linear → 80)
#       → spk_embed_affine + concat prompt_feat
#       → decoder (CausalConditionalCFM with 10 Euler steps + CFG)
#       → output mel                   ← `speech_feat`
#       → mel2wav (HiFTGenerator)      ← `audio_samples`
#
# The encoder pipeline (everything up to encoder_proj) is fully deterministic
# given fixed weights+inputs and is the parity target for the Swift port of
# UpsampleConformerEncoder. Everything from `decoder()` onwards involves
# random noise (`mx.random.normal(mu.shape)`), so reproducible parity requires
# capturing the noise seed AND the noise itself. We do both: the noise tensor
# is saved as `s3gen_cfm_noise.bin` for the Swift test to feed back in, and a
# seed-aware Python rerun yields the same `speech_feat`.

print("\n" + "═" * 60)
print("S3Gen reference outputs (Phase 5)")
print("═" * 60)

# Load S3Gen weights (skip CAMPPlus speaker_encoder - we use the pre-computed
# x-vector from conds.safetensors `gen.embedding` instead).
print(f"\nLoading S3Gen weights from {WEIGHTS_PATH} ...")
s3gen_weights = {}
with safe_open(str(WEIGHTS_PATH), framework="mlx") as f:
    for k in f.keys():
        if k.startswith("s3gen."):
            s3gen_weights[k[len("s3gen."):]] = f.get_tensor(k)
print(f"  loaded {len(s3gen_weights)} s3gen.* parameters")

from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3Gen as PyS3Gen

print("\nBuilding S3Token2Wav (meanflow=False) ...")
s3gen = PyS3Gen(meanflow=False)
s3gen_weights_sanitized = s3gen.sanitize(s3gen_weights)
s3gen.load_weights(list(s3gen_weights_sanitized.items()), strict=False)
s3gen.eval()
print("  built and weights loaded")

# Load pre-computed conditioning from conds.safetensors (the preset reference voice).
print(f"\nLoading S3Gen conditioning from {CONDS_PATH} ...")
with safe_open(str(CONDS_PATH), framework="mlx") as f:
    gen_embedding = f.get_tensor("gen.embedding")              # (1, 192)
    gen_prompt_feat = f.get_tensor("gen.prompt_feat")          # (1, 500, 80)
    gen_prompt_token = f.get_tensor("gen.prompt_token")        # (1, 250)
    gen_prompt_token_len = f.get_tensor("gen.prompt_token_len")  # (1,)
print(f"  gen.embedding:        {gen_embedding.shape}")
print(f"  gen.prompt_feat:      {gen_prompt_feat.shape}")
print(f"  gen.prompt_token:     {gen_prompt_token.shape}")
print(f"  gen.prompt_token_len: {gen_prompt_token_len.shape} = {gen_prompt_token_len.tolist()}")

# Save conditioning as reference bins (Swift will load these).
for name, arr, dtype in [
    ("s3gen_gen_embedding", gen_embedding, np.float32),
    ("s3gen_gen_prompt_feat", gen_prompt_feat, np.float32),
    ("s3gen_gen_prompt_token", gen_prompt_token, np.int32),
    ("s3gen_gen_prompt_token_len", gen_prompt_token_len, np.int32),
]:
    np.save(OUTPUT_DIR / f"{name}.npy", np.array(arr))
    save_raw(arr, OUTPUT_DIR / f"{name}.bin", dtype)

# Replicate the deterministic part of S3Token2Mel.__call__ step by step.
print("\nReplicating S3Gen encoder forward pass ...")
B = speech_tokens.shape[0]
prompt_token = gen_prompt_token
prompt_token_len = gen_prompt_token_len
prompt_feat = gen_prompt_feat
embedding = gen_embedding

# Concatenate prompt tokens with generated speech tokens
token_len_only = mx.array([speech_tokens.shape[1]] * B)
token = mx.concatenate([prompt_token, speech_tokens], axis=1)
token_len = prompt_token_len + token_len_only
print(f"  concatenated tokens: {token.shape}, total length: {token_len.tolist()}")

# Build mask (B, T, 1) on the concatenated sequence
max_len = token.shape[1]
mask = mx.arange(max_len)[None, :] < token_len[:, None]
mask_3d = mask[:, :, None].astype(mx.float32)

# Embed tokens
token_emb = s3gen.input_embedding(token.astype(mx.int32)) * mask_3d
mx.eval(token_emb)
print(f"  token_emb (post-embedding, masked): {token_emb.shape}, mean={float(token_emb.mean()):+.5f}")
save_raw(token, OUTPUT_DIR / "s3gen_encoder_input_tokens.bin", np.int32)
save_raw(token_emb, OUTPUT_DIR / "s3gen_encoder_token_emb.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_encoder_input_tokens.npy", np.array(token))
np.save(OUTPUT_DIR / "s3gen_encoder_token_emb.npy", np.array(token_emb))

# Run encoder (the parity target for Swift UpsampleConformerEncoder)
h, h_masks = s3gen.encoder(token_emb, token_len)
mx.eval(h, h_masks)
print(f"  encoder out h:        {h.shape}, mean={float(h.mean()):+.5f}, std={float(h.std()):.5f}")
print(f"  encoder out h_masks:  {h_masks.shape}, sum={int(h_masks.sum())}")
save_raw(h, OUTPUT_DIR / "s3gen_encoder_out.bin", np.float32)
save_raw(h_masks, OUTPUT_DIR / "s3gen_encoder_out_mask.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_encoder_out.npy", np.array(h))
np.save(OUTPUT_DIR / "s3gen_encoder_out_mask.npy", np.array(h_masks))

# Apply encoder_proj (Linear 512 → 80)
h_proj = s3gen.encoder_proj(h)
mx.eval(h_proj)
print(f"  encoder_proj out:     {h_proj.shape}")
save_raw(h_proj, OUTPUT_DIR / "s3gen_encoder_proj_out.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_encoder_proj_out.npy", np.array(h_proj))

# spk_embed_affine: normalize + Linear(192 → 80)
embedding_norm = embedding / (mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)
spk_emb_projected = s3gen.spk_embed_affine_layer(embedding_norm)
mx.eval(spk_emb_projected)
print(f"  spk_embed_affine out: {spk_emb_projected.shape}")
save_raw(spk_emb_projected, OUTPUT_DIR / "s3gen_spk_embed_affine_out.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_spk_embed_affine_out.npy", np.array(spk_emb_projected))

# Add to metadata
s3gen_metadata = {
    "speech_tokens_shape": list(speech_tokens.shape),
    "concat_token_shape": list(token.shape),
    "concat_token_len": [int(x) for x in token_len.tolist()],
    "token_emb_shape": list(token_emb.shape),
    "encoder_out_shape": list(h.shape),
    "encoder_proj_out_shape": list(h_proj.shape),
    "spk_embed_affine_out_shape": list(spk_emb_projected.shape),
    "prompt_feat_shape": list(prompt_feat.shape),
    "embedding_shape": list(embedding.shape),
}

# ───────────────────────────────────────────────────────────────────────────
# Decoder reference (Phase 5b)
# ───────────────────────────────────────────────────────────────────────────
# The ConditionalDecoder is the velocity-field estimator inside the CFM
# diffusion. The decoder is fully deterministic given its inputs (no randomness
# inside the forward pass — randomness only enters via the noise z sampled by
# the CFM solver outside the decoder).
#
# Inputs:
#   x_t  : (B, 80, T) latent at the current ODE step
#   mask : (B, 1, T)  float mask
#   mu   : (B, 80, T) encoder-projected mel conditioning
#   t    : (B,)       timestep scalars in [0, 1]
#   spks : (B, 80)    speaker embedding
#   cond : (B, 80, T) extra conditioning (zeros except over prompt prefix)
#
# We construct deterministic inputs using the already-captured encoder outputs
# above, plus a pinned noise tensor seeded with numpy.RandomState(42) so the
# Swift port can replay them bit-exactly.
print("\n" + "═" * 60)
print("Decoder reference outputs (Phase 5b)")
print("═" * 60)

# Build mu (B, 80, T) by transposing the encoder projection output (B, T, 80).
mu_ref = h_proj.transpose(0, 2, 1)  # (B, 80, T)
T_mu = mu_ref.shape[2]

# Build encoder mask (B, 1, T). The encoder returned (B, 1, 2T) already.
# h_masks is (B, 1, 2T) float — same T as mu_ref.
mask_ref = h_masks.astype(mx.float32)
assert mask_ref.shape[2] == T_mu, f"mask T {mask_ref.shape[2]} != mu T {T_mu}"

# Speaker embedding: (B, 80) — broadcast across batch from gen.embedding.
spk_ref = spk_emb_projected  # (1, 80)

# Cond (B, 80, T): zeros except over the prompt prefix (matching how the CFM
# solver concatenates prompt_feat into cond in S3Token2Mel).
# In the real pipeline:
#   cond = mx.zeros_like(mu)
#   cond[:, :, : prompt_len_mel] = prompt_feat.T
# Here we replicate that:
prompt_feat_T = prompt_feat.transpose(0, 2, 1)  # (1, 80, 500)
prompt_mel_len = prompt_feat_T.shape[2]
cond_ref = mx.zeros_like(mu_ref)
cond_ref[:, :, :prompt_mel_len] = prompt_feat_T[:, :, :min(prompt_mel_len, T_mu)]

# Pinned noise — must be reproducible from Swift. Use numpy with a fixed seed.
np.random.seed(42)
noise_np = np.random.randn(*mu_ref.shape).astype(np.float32)
x_t_ref = mx.array(noise_np)  # (B, 80, T)

# Timestep — start of the ODE (closest to noise). Use t=0.0.
t_ref = mx.array([0.0], dtype=mx.float32)  # (B,) with B=1

print(f"  mu shape:    {mu_ref.shape}")
print(f"  mask shape:  {mask_ref.shape}")
print(f"  x_t shape:   {x_t_ref.shape}")
print(f"  spks shape:  {spk_ref.shape}")
print(f"  cond shape:  {cond_ref.shape}")
print(f"  t value:     {t_ref.tolist()}")

# Run the Python decoder. estimator is s3gen.decoder.estimator
estimator = s3gen.decoder.estimator
# Eval ensures weights loaded before any compute.
mx.eval(mu_ref, mask_ref, x_t_ref, spk_ref, cond_ref, t_ref)

# Capture a few intermediate values for layer-by-layer bisection.
print("  Capturing intermediates ...")
# Helper: re-import the sinusoidal embed to be sure.
from mlx_audio.tts.models.chatterbox_turbo.models.s3gen.decoder import sinusoidal_pos_emb

# 1. After sinusoidal + time_mlp.
t_emb_sin = sinusoidal_pos_emb(t_ref, estimator.in_channels)
t_emb = estimator.time_mlp(t_emb_sin)
mx.eval(t_emb)
print(f"    t_emb after time_mlp:    {t_emb.shape}, "
      f"mean={float(t_emb.mean()):+.5f}, std={float(t_emb.std()):.5f}")
save_raw(t_emb, OUTPUT_DIR / "s3gen_decoder_t_emb.bin", np.float32)

# 2. After input concat (x, mu, spks_expanded, cond).
spks_expanded = mx.broadcast_to(
    spk_ref[:, :, None], (spk_ref.shape[0], spk_ref.shape[1], x_t_ref.shape[2])
)
concat_in = mx.concatenate([x_t_ref, mu_ref, spks_expanded, cond_ref], axis=1)
mx.eval(concat_in)
print(f"    concat input:            {concat_in.shape}")
save_raw(concat_in, OUTPUT_DIR / "s3gen_decoder_concat_input.bin", np.float32)

# 3. After the only down_block (resnet + 4 transformer + downsample).
down_block = estimator.down_blocks[0]
resnet0 = down_block.resnet

# 3a. Inside resnet: just block1.
x_after_block1 = resnet0.block1(concat_in, mask_ref)
mx.eval(x_after_block1)
print(f"    after resnet.block1:     {x_after_block1.shape}, "
      f"mean={float(x_after_block1.mean()):+.5f}, std={float(x_after_block1.std()):.5f}")
save_raw(x_after_block1, OUTPUT_DIR / "s3gen_decoder_after_block1.bin", np.float32)

# 3b. After adding time embedding via mlp.0 (Linear) applied to mish(t_emb).
# In the meanflow=False model, mlp is [Linear] (mish is inline).
import mlx.nn as _nn_mod
t_proj = resnet0.mlp[0](_nn_mod.mish(t_emb))
mx.eval(t_proj)
print(f"    t_proj (mlp.0 of mish):  {t_proj.shape}, "
      f"mean={float(t_proj.mean()):+.5f}, std={float(t_proj.std()):.5f}")
save_raw(t_proj, OUTPUT_DIR / "s3gen_decoder_t_proj.bin", np.float32)

# 3c. After resnet (full).
x_after_resnet = down_block.resnet(concat_in, mask_ref, t_emb)
mx.eval(x_after_resnet)
print(f"    after down_block.resnet: {x_after_resnet.shape}, "
      f"mean={float(x_after_resnet.mean()):+.5f}, std={float(x_after_resnet.std()):.5f}")
save_raw(x_after_resnet, OUTPUT_DIR / "s3gen_decoder_after_down_resnet.bin", np.float32)

velocity = estimator(x_t_ref, mask_ref, mu_ref, t_ref, spk_ref, cond_ref)
mx.eval(velocity)
print(f"  velocity out shape: {velocity.shape}, "
      f"mean={float(velocity.mean()):+.5f}, std={float(velocity.std()):.5f}")

# Save inputs + output for Swift to consume.
save_raw(x_t_ref, OUTPUT_DIR / "s3gen_decoder_x_t.bin", np.float32)
save_raw(mu_ref, OUTPUT_DIR / "s3gen_decoder_mu.bin", np.float32)
save_raw(mask_ref, OUTPUT_DIR / "s3gen_decoder_mask.bin", np.float32)
save_raw(spk_ref, OUTPUT_DIR / "s3gen_decoder_spks.bin", np.float32)
save_raw(cond_ref, OUTPUT_DIR / "s3gen_decoder_cond.bin", np.float32)
save_raw(t_ref, OUTPUT_DIR / "s3gen_decoder_t.bin", np.float32)
save_raw(velocity, OUTPUT_DIR / "s3gen_decoder_velocity_out.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_decoder_x_t.npy", np.array(x_t_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_mu.npy", np.array(mu_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_mask.npy", np.array(mask_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_spks.npy", np.array(spk_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_cond.npy", np.array(cond_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_t.npy", np.array(t_ref))
np.save(OUTPUT_DIR / "s3gen_decoder_velocity_out.npy", np.array(velocity))

s3gen_metadata["decoder"] = {
    "x_t_shape": list(x_t_ref.shape),
    "mu_shape": list(mu_ref.shape),
    "mask_shape": list(mask_ref.shape),
    "spks_shape": list(spk_ref.shape),
    "cond_shape": list(cond_ref.shape),
    "t_value": [float(x) for x in t_ref.tolist()],
    "velocity_shape": list(velocity.shape),
    "velocity_mean": float(velocity.mean()),
    "velocity_std": float(velocity.std()),
    "noise_seed": 42,
    "prompt_mel_len": int(prompt_mel_len),
}

# ───────────────────────────────────────────────────────────────────────────
# CFM reference (Phase 5c) — CausalConditionalCFM Euler ODE solver
# ───────────────────────────────────────────────────────────────────────────
# The CFM module wraps the ConditionalDecoder and runs Euler integration to
# produce speech_feat (a mel-like tensor) from noise. Chatterbox-Turbo uses
# `meanflow=True` and `n_timesteps=2`, so the CFM:
#   - samples noise z ~ N(0, 1) of shape (B, 80, T)
#   - builds t_span = linspace(0, 1, 3) = [0, 0.5, 1.0]  (no cosine reshape)
#   - calls the decoder 2 times with (t=0, r=0.5) then (t=0.5, r=1.0)
#   - each call uses meanflow time-mixing: t_emb = mixer(concat(t_emb, r_emb))
#   - applies basic Euler: x <- x + (r-t) * decoder(x, mu, t, r, spks, cond)
# No CFG in the meanflow path; the decoder bakes in the conditioning directly.
#
# To make this reproducible across Python ↔ Swift we externalise the initial
# noise: numpy.random.seed(7777), generate z, then save the noise tensor as
# `s3gen_cfm_noise.bin` and the final `speech_feat` as `s3gen_cfm_speech_feat.bin`.
# Swift loads the pinned noise (rather than calling MLXRandom internally) so
# the entire path is bit-deterministic.
print("\n" + "═" * 60)
print("CFM reference outputs (Phase 5c)")
print("═" * 60)

# Build a second S3Gen instance with meanflow=True so we get the meanflow
# decoder variant with `time_embed_mixer`. Use the same weights dict.
print("\nBuilding S3Token2Wav (meanflow=True) ...")
s3gen_mf = PyS3Gen(meanflow=True)
s3gen_mf_weights_sanitized = s3gen_mf.sanitize(s3gen_weights)
s3gen_mf.load_weights(list(s3gen_mf_weights_sanitized.items()), strict=False)
s3gen_mf.eval()
print("  built meanflow s3gen and weights loaded")

# Reuse mu_ref, mask_ref, spk_ref, cond_ref from the decoder section above.
# Pin a fresh noise z (independent seed from the decoder x_t).
np.random.seed(7777)
cfm_noise_np = np.random.randn(*mu_ref.shape).astype(np.float32)
cfm_noise = mx.array(cfm_noise_np)  # (B, 80, T)
mx.eval(cfm_noise)
print(f"  cfm_noise shape: {cfm_noise.shape}, "
      f"mean={float(cfm_noise.mean()):+.5f}, std={float(cfm_noise.std()):.5f}")

# Replicate `_basic_euler` (meanflow path: no CFG, decoder takes r).
mf_estimator = s3gen_mf.decoder.estimator
n_cfm_timesteps = 2
t_span_cfm = mx.linspace(0, 1, n_cfm_timesteps + 1)
# meanflow path does NOT apply the cosine reshape.
print(f"  t_span: {t_span_cfm.tolist()}")

x = cfm_noise
for i in range(n_cfm_timesteps):
    t = t_span_cfm[i : i + 1]
    r = t_span_cfm[i + 1 : i + 2]
    dxdt = mf_estimator(
        x=x,
        mask=mask_ref,
        mu=mu_ref,
        t=t,
        spks=spk_ref,
        cond=cond_ref,
        r=r,
    )
    dt = r - t
    x = x + dt * dxdt
    mx.eval(x)
    print(f"    step {i}: t={float(t.item()):.4f} r={float(r.item()):.4f} "
          f"x_mean={float(x.mean()):+.5f} x_std={float(x.std()):.5f}")

speech_feat = x
mx.eval(speech_feat)
print(f"  speech_feat shape: {speech_feat.shape}, "
      f"mean={float(speech_feat.mean()):+.5f}, std={float(speech_feat.std()):.5f}")

# Save inputs + output for Swift.
save_raw(cfm_noise, OUTPUT_DIR / "s3gen_cfm_noise.bin", np.float32)
save_raw(mu_ref, OUTPUT_DIR / "s3gen_cfm_mu.bin", np.float32)
save_raw(mask_ref, OUTPUT_DIR / "s3gen_cfm_mask.bin", np.float32)
save_raw(spk_ref, OUTPUT_DIR / "s3gen_cfm_spks.bin", np.float32)
save_raw(cond_ref, OUTPUT_DIR / "s3gen_cfm_cond.bin", np.float32)
save_raw(speech_feat, OUTPUT_DIR / "s3gen_cfm_speech_feat.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_cfm_noise.npy", np.array(cfm_noise))
np.save(OUTPUT_DIR / "s3gen_cfm_speech_feat.npy", np.array(speech_feat))

s3gen_metadata["cfm"] = {
    "noise_seed": 7777,
    "n_timesteps": n_cfm_timesteps,
    "t_span": [float(x) for x in t_span_cfm.tolist()],
    "meanflow": True,
    "noise_shape": list(cfm_noise.shape),
    "speech_feat_shape": list(speech_feat.shape),
    "speech_feat_mean": float(speech_feat.mean()),
    "speech_feat_std": float(speech_feat.std()),
}

# ───────────────────────────────────────────────────────────────────────────
# Vocoder reference (Phase 5d) — HiFTGenerator (s3gen.mel2wav.*)
# ───────────────────────────────────────────────────────────────────────────
# The HiFTGenerator turns a mel-like feature `speech_feat` of shape (B, T, 80)
# into raw 24 kHz audio of shape (B, T_audio). It is mostly deterministic
# given fixed inputs *except* for the SineGen noise (random uniform phase per
# harmonic + Gaussian additive noise). To make this reproducible across the
# Python and Swift implementations we replace mx.random calls with a numpy
# RNG seeded to 7777, capture all randomness as bins, and let the Swift side
# replay them bit-exactly when it runs the parity gate.
#
# We capture:
#   s3gen_vocoder_speech_feat.bin   — input mel (1, T_mel, 80)
#   s3gen_vocoder_f0.bin            — F0Predictor output (1, T_mel)
#   s3gen_vocoder_source.bin        — m_source output (1, 1, T_audio)
#   s3gen_vocoder_audio.bin         — final audio (1, T_audio)
#
# The deterministic intermediate `f0` is the strongest single bisection point
# (it isolates F0Predictor from the rest), so we save it. The source signal
# is the next bisection point (isolates SineGen + l_linear), and the audio is
# the final parity gate.
print("\n" + "═" * 60)
print("Vocoder reference outputs (Phase 5d)")
print("═" * 60)

# Build a small, reproducible mel input. The "real" mel here would come from
# the CFM Euler solver (Phase 5c), but that's not yet ported on the Swift
# side; instead we use prompt_feat (a real reference mel from conds.safetensors)
# to provide a realistic input. Shape: (1, 500, 80).
speech_feat_vocoder = gen_prompt_feat  # (1, T_mel=500, 80)
print(f"  speech_feat shape: {speech_feat_vocoder.shape}")
save_raw(speech_feat_vocoder, OUTPUT_DIR / "s3gen_vocoder_speech_feat.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_vocoder_speech_feat.npy", np.array(speech_feat_vocoder))

# Get the mel2wav module.
mel2wav = s3gen.mel2wav

# Step 1: F0 prediction is deterministic.
mel_BCT = speech_feat_vocoder.transpose(0, 2, 1)  # (1, 80, T_mel)
f0_pred = mel2wav.f0_predictor(mel_BCT)            # (1, T_mel)
mx.eval(f0_pred)
print(f"  f0_predictor out: {f0_pred.shape}, "
      f"mean={float(f0_pred.mean()):+.5f}, std={float(f0_pred.std()):.5f}")
save_raw(f0_pred, OUTPUT_DIR / "s3gen_vocoder_f0.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_vocoder_f0.npy", np.array(f0_pred))

# Step 2: Upsample F0.
f0_up = mel2wav._upsample_f0(f0_pred)              # (1, T_mel*scale, 1)
T_audio = f0_up.shape[1]
print(f"  f0_upsampled shape: {f0_up.shape}, T_audio={T_audio}")

# Step 3: Source generation. We replace random noise with pinned numpy so
# Swift can replay it. The SineGen forward uses two random calls:
#   (a) random_phases for harmonics: (B, harmonic_num, 1) uniform[-pi, pi]
#   (b) noise: shape == sine_waves.shape == (B, harmonic_num+1, T_audio)
# Then in SourceModule:
#   (c) tanh-noise: shape == uv.shape == (B, T_audio, 1)
# But (c) is computed and immediately discarded (not used downstream — only
# sine_merge is consumed by .decode()). Likewise (b) is consumed inside the
# sine path. We capture (a) and (b) and rebuild the source ourselves to keep
# Swift+Python in lockstep.

NB_HARMONICS = 8  # default in s3gen config
B = speech_feat_vocoder.shape[0]
rng = np.random.RandomState(7777)
sinegen_random_phases_np = rng.uniform(-np.pi, np.pi, size=(B, NB_HARMONICS, 1)).astype(np.float32)
sine_shape = (B, NB_HARMONICS + 1, T_audio)
sinegen_noise_np = rng.standard_normal(size=sine_shape).astype(np.float32)
sinegen_random_phases = mx.array(sinegen_random_phases_np)
sinegen_noise = mx.array(sinegen_noise_np)
save_raw(sinegen_random_phases, OUTPUT_DIR / "s3gen_vocoder_sinegen_phases.bin", np.float32)
save_raw(sinegen_noise, OUTPUT_DIR / "s3gen_vocoder_sinegen_noise.bin", np.float32)

# Replicate SineGen.__call__ deterministically using pinned randoms.
f0_for_sinegen = f0_up.transpose(0, 2, 1)  # (B, 1, T_audio)
sinegen = mel2wav.m_source.l_sin_gen
B_sg, _, T_sg = f0_for_sinegen.shape
harmonics = mx.arange(1, sinegen.harmonic_num + 2)[None, :, None]  # (1, H, 1)
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
# Transpose to (B, T, H+1) and run l_linear
sine_wavs_T = sine_waves.transpose(0, 2, 1)
sine_merge = mx.tanh(mel2wav.m_source.l_linear(sine_wavs_T))  # (B, T, 1)
source_signal = sine_merge.transpose(0, 2, 1)  # (B, 1, T_audio)
mx.eval(source_signal)
print(f"  source signal shape: {source_signal.shape}, "
      f"mean={float(source_signal.mean()):+.5f}, std={float(source_signal.std()):.5f}")
save_raw(source_signal, OUTPUT_DIR / "s3gen_vocoder_source.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_vocoder_source.npy", np.array(source_signal))

# Step 4: Decode (deterministic given mel and source).
audio = mel2wav.decode(mel_BCT, source_signal)
mx.eval(audio)
print(f"  audio shape: {audio.shape}, "
      f"mean={float(audio.mean()):+.5f}, std={float(audio.std()):.5f}, "
      f"min={float(audio.min()):+.5f}, max={float(audio.max()):+.5f}")
save_raw(audio, OUTPUT_DIR / "s3gen_vocoder_audio.bin", np.float32)
np.save(OUTPUT_DIR / "s3gen_vocoder_audio.npy", np.array(audio))

# Optional: write a wav for ear-test purposes.
try:
    import wave
    audio_clipped = np.clip(np.array(audio[0]), -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    with wave.open(str(OUTPUT_DIR / "s3gen_vocoder_audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(mel2wav.sampling_rate)
        w.writeframes(audio_int16.tobytes())
    print(f"  wav written to {OUTPUT_DIR / 's3gen_vocoder_audio.wav'}")
except Exception as e:
    print(f"  (skipped wav: {e})")

s3gen_metadata["vocoder"] = {
    "speech_feat_shape": list(speech_feat_vocoder.shape),
    "f0_shape": list(f0_pred.shape),
    "f0_mean": float(f0_pred.mean()),
    "f0_std": float(f0_pred.std()),
    "source_shape": list(source_signal.shape),
    "audio_shape": list(audio.shape),
    "audio_mean": float(audio.mean()),
    "audio_std": float(audio.std()),
    "audio_min": float(audio.min()),
    "audio_max": float(audio.max()),
    "sinegen_phases_shape": list(sinegen_random_phases.shape),
    "sinegen_noise_shape": list(sinegen_noise.shape),
    "nb_harmonics": NB_HARMONICS,
    "sampling_rate": mel2wav.sampling_rate,
    "upsample_scale": mel2wav.f0_upsample_scale,
    "n_fft": mel2wav.istft_params["n_fft"],
    "hop_len": mel2wav.istft_params["hop_len"],
    "noise_seed": 7777,
}

# ── Metadata ────────────────────────────────────────────────────────────────
metadata = {
    "test_text": TEST_TEXT,
    "seed": SEED,
    "gen_params": GEN_PARAMS,
    "model_repo": "mlx-community/chatterbox-turbo-fp16",
    "weights_path": str(WEIGHTS_PATH),
    "text_tokens_shape": list(text_tokens.shape),
    "text_tokens": [int(x) for x in np.array(text_tokens[0])],
    "len_cond": int(len_cond),
    "prefill_seq_len": int(T_),
    "inputs_embeds_shape": list(inputs_embeds.shape),
    "speech_tokens_shape": list(speech_tokens.shape),
    "speech_tokens_first_8": [int(x) for x in np.array(speech_tokens[0, :8])],
    "speech_logits_first_argmax": argmax_first,
    "speech_logits_first_top1_logit": float(speech_logits_first_2d[0, argmax_first].item()),
    "t3_config": {
        "start_text_token": hp.start_text_token,
        "stop_text_token": hp.stop_text_token,
        "text_tokens_dict_size": hp.text_tokens_dict_size,
        "start_speech_token": hp.start_speech_token,
        "stop_speech_token": hp.stop_speech_token,
        "speech_tokens_dict_size": hp.speech_tokens_dict_size,
        "speaker_embed_size": hp.speaker_embed_size,
    },
    "s3gen": s3gen_metadata,
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Reference outputs saved to {OUTPUT_DIR}/")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p.name}")
