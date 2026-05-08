"""Build XOR3 (3-input parity) as a threshold network.

Two-layer construction: thermometer code t1/t2/t3 = (popcount(a,b,c) >= k),
output = (t1 - t2 + t3 >= 1). Truth table:

    popcnt   t1 t2 t3   t1-t2+t3   parity
    0         0  0  0      0          0
    1         1  0  0      1          1
    2         1  1  0      0          0
    3         1  1  1      1          1
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

OUT = Path(__file__).parent / "xor3.safetensors"

sr = {
    "0": "#0", "1": "#1",
    "2": "$a", "3": "$b", "4": "$c",
    "5": "t1", "6": "t2", "7": "t3",
    "8": "y",
}

tensors: dict[str, torch.Tensor] = {}


def gate(name: str, weights, bias: int, ids):
    tensors[f"{name}.weight"] = torch.tensor(weights, dtype=torch.int8)
    tensors[f"{name}.bias"] = torch.tensor([bias], dtype=torch.int8)
    tensors[f"{name}.inputs"] = torch.tensor(ids, dtype=torch.int64)


# Layer 1: thermometer code from popcount of (a,b,c)
gate("t1", [1, 1, 1], -1, [2, 3, 4])   # (a+b+c) >= 1
gate("t2", [1, 1, 1], -2, [2, 3, 4])   # (a+b+c) >= 2
gate("t3", [1, 1, 1], -3, [2, 3, 4])   # (a+b+c) >= 3

# Layer 2: XOR3(a,b,c) = (t1 - t2 + t3) >= 1
gate("y", [1, -1, 1], -1, [5, 6, 7])

save_file(
    tensors, str(OUT),
    metadata={"signal_registry": json.dumps(sr), "schema_version": "1"},
)
print(f"wrote {OUT} with {len(tensors)//3} gates")
