"""Inspect SmolLM2-135M-Instruct: tensor shapes, dtypes, op set, sizes.

Goal: produce the concrete list of primitives the safetensors2verilog tool
would have to support for faithful translation of a LLaMA-style transformer.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

ROOT = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)

print("=== config.json ===")
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
for k in [
    "model_type", "architectures",
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "vocab_size",
    "max_position_embeddings", "rope_theta", "rope_scaling",
    "rms_norm_eps", "hidden_act", "torch_dtype", "tie_word_embeddings",
]:
    if k in cfg:
        print(f"  {k}: {cfg[k]}")

print("\n=== tensor inventory ===")
sf = ROOT / "model.safetensors"
print(f"safetensors size: {sf.stat().st_size/1e6:.1f} MB")

with safe_open(str(sf), framework="pt") as f:
    keys = sorted(f.keys())
    by_pattern: dict[str, list[tuple[str, tuple, str]]] = defaultdict(list)
    total_params = 0
    dtypes = set()
    for k in keys:
        t = f.get_tensor(k)
        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        dtypes.add(dtype)
        total_params += t.numel()
        # Pattern: strip layer index
        parts = k.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            pattern = ".".join(parts[:idx] + ["layers", "N"] + parts[idx+2:])
        else:
            pattern = k
        by_pattern[pattern].append((k, shape, dtype))

    print(f"total params: {total_params/1e6:.2f} M")
    print(f"dtypes seen: {dtypes}")
    print(f"distinct tensor patterns: {len(by_pattern)}")
    print()
    for pat, instances in sorted(by_pattern.items()):
        rep = instances[0]
        n = len(instances)
        print(f"  {n:>3}x  {pat:<55} shape={rep[1]}  dtype={rep[2]}")

print("\n=== implied op set per layer ===")
print("""
  embed_tokens:  Embedding lookup       (Gather over ROM, vocab=49152, d=576)
  for each of 30 decoder layers:
    input_layernorm        RMSNorm     (square, sum, sqrt, div, mul)
    self_attn.q_proj       Linear       (576 -> 576)
    self_attn.k_proj       Linear       (576 -> 192)   GQA, 3 KV heads
    self_attn.v_proj       Linear       (576 -> 192)
    RoPE on Q, K           rot pos emb  (sin/cos LUT, mul, add)
    Q @ K^T                MatMul       (per head, with KV cache)
    /sqrt(d_k)             div by const (shift if d_k power of 2)
    causal mask + Softmax  Softmax      (max, sub, exp, sum, div)
    attn @ V               MatMul
    self_attn.o_proj       Linear       (576 -> 576)
    residual               Add
    post_attention_layernorm  RMSNorm
    mlp.gate_proj          Linear       (576 -> 1536)
    mlp.up_proj            Linear       (576 -> 1536)
    SiLU(gate) * up        SiLU + mul   (sigmoid: exp, add, div)
    mlp.down_proj          Linear       (1536 -> 576)
    residual               Add
  norm                     RMSNorm
  lm_head                  Linear       (576 -> 49152, tied with embed)
  argmax / sample          decoding logic
""")

# Check if the model has weight scales (for int8/AWQ/etc)
print("=== quantization scales present? ===")
scale_keys = [k for k in keys if "scale" in k.lower() or "zero_point" in k.lower()]
print(f"  found {len(scale_keys)} scale/zero_point tensors")
if scale_keys:
    for k in scale_keys[:5]:
        print(f"    {k}")

print("\n=== weight value range sample (one MLP gate_proj) ===")
with safe_open(str(sf), framework="pt") as f:
    candidate = next(
        (k for k in keys if k.endswith("mlp.gate_proj.weight")),
        None,
    )
    if candidate:
        t = f.get_tensor(candidate).to("cpu").float()
        print(f"  {candidate}: shape={tuple(t.shape)}, dtype-original=bf16/fp16")
        print(f"    min={t.min().item():.4f}  max={t.max().item():.4f}")
        print(f"    mean={t.mean().item():.4e}  std={t.std().item():.4e}")
        # Are these integer-valued? (the int8_linear front-end check)
        rounded = t.round()
        is_int = bool((t - rounded).abs().max().item() < 1e-3)
        is_ternary = is_int and bool((rounded.abs() <= 1).all().item())
        print(f"    integer-valued? {is_int}    ternary? {is_ternary}")
