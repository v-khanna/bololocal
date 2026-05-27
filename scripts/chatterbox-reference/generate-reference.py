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
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Reference outputs saved to {OUTPUT_DIR}/")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p.name}")
