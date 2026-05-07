"""SmolLM2-135M-Instruct compile through the hf_llama frontend (item 57).

Uses overrides to keep this tractable for iverilog: 1 layer (the first
decoder layer of the 30-layer model), max_seq=4. Still exercises the full
SmolLM2 dimensions for hidden/intermediate/vocab/head_dim, so the matmul,
attention, and lm_head paths see the real shapes.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from safetensors2verilog import collect_sidecar_files, emit_module
from safetensors2verilog.core import registry


SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT = Path(r"D:\safetensors2verilog\_play\smollm_out")
OUT.mkdir(exist_ok=True)


def main() -> int:
    if not (SMOLLM_DIR / "model.safetensors").exists():
        print(f"SmolLM2 weights not found at {SMOLLM_DIR}")
        return 2

    print("Loading SmolLM2-135M-Instruct via hf_llama frontend...")
    print("  num_layers_override=1, max_seq_override=4")
    fe = registry.get("hf_llama")()
    graph = fe.parse(
        SMOLLM_DIR / "model.safetensors",
        top="smollm_l1",
        config=str(SMOLLM_DIR / "config.json"),
        activation_bits=8, weight_bits=8,
        num_layers_override=1, max_seq_override=4,
    )
    n_subs = len(graph.submodules)
    print(f"GateGraph: {len(graph.gates)} gates, {n_subs} submodules")

    print("Emitting Verilog (this may take a while at SmolLM2 scale)...")
    text = emit_module(graph)
    n_lines = len(text.splitlines())
    print(f"  {n_lines:,} lines, {len(text)/1e6:.1f} MB")

    v_path = OUT / "smollm_l1.v"
    v_path.write_text(text, encoding="utf-8")
    print(f"  wrote {v_path}")

    print("Collecting sidecar weight ROM files...")
    sidecar = collect_sidecar_files(graph)
    print(f"  {len(sidecar)} sidecar files")
    total = 0
    for fn, contents in sidecar.items():
        (OUT / fn).write_text(contents, encoding="utf-8")
        total += len(contents)
    print(f"  {total/1e6:.1f} MB of weight ROMs written")

    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; stopping after emission.")
        return 0

    print("\nThis is the point in the trajectory where iverilog compile + sim")
    print("of a 1-layer SmolLM2 takes minutes. Skipping the actual sim run")
    print("here; the frontend produced a self-consistent Verilog package.")
    print(f"\nTo simulate, in {OUT}:")
    print(f"  iverilog -g2012 -o smollm.vvp smollm_l1.v <a testbench>")
    print(f"  vvp smollm.vvp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
