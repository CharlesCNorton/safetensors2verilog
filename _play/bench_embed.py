"""Benchmark embedding sidecar emission for SmolLM2."""
import time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors2verilog.blocks.embedding import embedding_block

p = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac\model.safetensors"
)
t0 = time.time()
with safe_open(str(p), framework="pt") as f:
    W = f.get_tensor("model.embed_tokens.weight").to(torch.float32)
print(f"{time.time()-t0:.1f}s load weight {tuple(W.shape)}")

t0 = time.time()
qmax = 127
fp_max = float(W.abs().max().item())
scale = fp_max / qmax
q = (W / scale).round().clamp(-qmax, qmax).to(torch.int32)
print(f"{time.time()-t0:.1f}s quantize")

t0 = time.time()
ql = q.tolist()
print(f"{time.time()-t0:.1f}s tolist (Python list of lists)")

t0 = time.time()
sub = embedding_block(V=49152, H=576, abits=8, weights=ql, inline_init=False)
print(f"{time.time()-t0:.1f}s embedding_block emission")

total_mb = sum(len(v) for v in sub.sidecar_files.values()) / 1e6
print(f"sidecar size: {total_mb:.1f} MB")
