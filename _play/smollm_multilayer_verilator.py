"""Item 2 of the open list: build verilator simulator binaries for the
2-layer and 4-layer SmolLM2 modules and drive a fixed prompt through
each, confirming each one produces a plausible token.

Existing infra (`_play/smollm_inference.py`) handles the 1-layer case;
this script extends to 2 and 4 layers by generating the Verilog (with
the new --sidecar-layout subdirs path-rewrite), building the verilator
binary in WSL, and running a single-token forward pass.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors2verilog import (
    RawSubmodule, emit_module, rewrite_readmemh_paths, write_sidecar_files,
)
from safetensors2verilog.frontends.hf_llama import build_llama_graph

SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT_BASE = Path(__file__).resolve().parent / "smollm_multilayer"
OUT_BASE.mkdir(exist_ok=True)
HID = 576
ABITS = 8
VOCAB = 49152
TOK_BITS = 16  # ceil(log2(49152)) = 16


def load_state(n_layers: int) -> tuple[dict, dict[str, torch.Tensor]]:
    cfg = json.loads((SMOLLM_DIR / "config.json").read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = n_layers
    # Match the existing 1-layer infra: cap max_position_embeddings at
    # 4 so attention's KV-cache buffers fit and verilator can unroll
    # the per-position loops without `--unroll-count` blowing up.
    cfg["max_position_embeddings"] = 4
    sd: dict[str, torch.Tensor] = {}
    keep_layer_prefixes = tuple(
        f"model.layers.{i}." for i in range(n_layers)
    )
    with safe_open(str(SMOLLM_DIR / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.layers.") and not k.startswith(keep_layer_prefixes):
                continue
            sd[k] = f.get_tensor(k).clone()
    return cfg, sd


def emit_for_layers(n_layers: int) -> tuple[Path, Path]:
    """Returns (out_dir, v_path) ready for verilator build."""
    out_dir = OUT_BASE / f"L{n_layers}"
    out_dir.mkdir(exist_ok=True)
    cfg, sd = load_state(n_layers)
    g = build_llama_graph(
        config=cfg, state_dict=sd, top=f"smollm_l{n_layers}",
        skip_lm_head=True,
    )
    print(f"[L{n_layers}] graph: {len(g.gates)} gates, "
          f"{len(g.submodules)} submodules")
    text = emit_module(g)
    # Sidecar layout: subdirs per-module + rewrite $readmemh paths.
    write_sidecar_files(g, out_dir, layout="subdirs",
                        write_manifest=False)
    path_map: dict[str, str] = {}

    def collect(graph) -> None:
        for sub in graph.submodules:
            if isinstance(sub, RawSubmodule):
                for fn in sub.sidecar_files:
                    path_map[fn] = f"{sub.top}/{fn}"
            else:
                collect(sub)
    collect(g)
    text = rewrite_readmemh_paths(text, path_map)
    v_path = out_dir / f"smollm_l{n_layers}.v"
    v_path.write_text(text, encoding="utf-8")
    print(f"[L{n_layers}] verilog: "
          f"{v_path.stat().st_size / 1024 / 1024:.2f} MB at {v_path}")
    return out_dir, v_path


def write_tb(out_dir: Path, n_layers: int, token_id: int,
             position: int) -> None:
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{TOK_BITS-1}:0] token_id;\n"
        f"  reg [2:0] position;\n"
        "  wire done;\n"
        f"  wire signed [{HID*ABITS-1}:0] final_norm;\n"
        f"  smollm_l{n_layers} dut(.clk(clk), .rst(rst), .start(start),\n"
        "                .token_id(token_id), .position(position),\n"
        "                .done(done), .final_norm(final_norm));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        "    @(negedge clk);\n"
        f"      token_id = {TOK_BITS}'d{token_id};\n"
        f"      position = 3'd{position};\n"
        "      start = 1;\n"
        "    @(negedge clk); start = 0;\n"
        "    cycles = 0;\n"
        "    while (!done) begin\n"
        "      @(posedge clk);\n"
        "      cycles = cycles + 1;\n"
        "      if (cycles > 1000000) begin $display(\"TIMEOUT\"); $finish; end\n"
        "    end\n"
        "    $display(\"DONE %0d cycles  final_norm=%h\", cycles, final_norm);\n"
        "    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    (out_dir / "tb.v").write_text(tb, encoding="utf-8")


def build_verilator(out_dir: Path, n_layers: int) -> bool:
    """Run verilator --binary in WSL. Returns True on success."""
    print(f"[L{n_layers}] verilator build starting ...")
    t0 = time.time()
    # Convert Windows path to WSL path.
    wsl_dir = "/mnt/" + str(out_dir).replace("\\", "/").replace("D:", "d").replace("C:", "c")
    # Mirror the known-working 1-layer build invocation in
    # _play/smollm_inference.py: --binary mode (synth + compile + link)
    # plus --timing so verilator's timing scheduler handles the
    # @(negedge clk) / @(posedge clk) event controls in tb.v rather
    # than dropping them. The Wno- flags suppress the harmless width /
    # unused / case-incomplete diagnostics auto-emitted Verilog raises
    # at SmolLM2 scale.
    cmd = (
        f"cd {wsl_dir} && rm -rf obj_dir && "
        "verilator --binary --top-module tb -j 4 --timing "
        "--Wno-fatal --Wno-WIDTH --Wno-WIDTHEXPAND --Wno-WIDTHTRUNC "
        "--Wno-UNOPTFLAT --Wno-INFINITELOOP --Wno-UNUSEDSIGNAL "
        "--Wno-UNUSEDPARAM --Wno-CASEINCOMPLETE --Wno-INITIALDLY "
        f"-O3 -CFLAGS '-O0' smollm_l{n_layers}.v tb.v 2>&1 | tail -8"
    )
    proc = subprocess.run(
        ["wsl.exe", "-d", "WSLExperiments", "--", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - t0
    print(f"[L{n_layers}] build {'OK' if proc.returncode == 0 else 'FAIL'} "
          f"in {elapsed:.1f}s")
    if proc.returncode != 0:
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        return False
    return True


def run_verilator(out_dir: Path, n_layers: int) -> tuple[int, str] | None:
    wsl_dir = "/mnt/" + str(out_dir).replace("\\", "/").replace("D:", "d").replace("C:", "c")
    cmd = f"cd {wsl_dir} && ./obj_dir/Vtb"
    t0 = time.time()
    proc = subprocess.run(
        ["wsl.exe", "-d", "WSLExperiments", "--", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    print(f"[L{n_layers}] sim {elapsed:.1f}s")
    m = re.search(r"DONE (\d+) cycles\s+final_norm=([0-9a-fA-F]+)", proc.stdout)
    if not m:
        print(f"[L{n_layers}] sim output didn't match DONE: "
              f"{proc.stdout[-500:]}")
        return None
    return int(m.group(1)), m.group(2)


def hex_to_int_vec(hex_str: str, n: int, bits: int) -> list[int]:
    big = int(hex_str, 16)
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    out = []
    for i in range(n):
        v = (big >> (i * bits)) & mask
        if v & sign:
            v -= 1 << bits
        out.append(v)
    return out


def main() -> int:
    # Use the same first token for both layer counts so the comparison
    # is meaningful: token 504 = "The".
    token_id = 504
    position = 0

    summary: list[dict] = []
    for n_layers in (2, 4):
        out_dir, v_path = emit_for_layers(n_layers)
        write_tb(out_dir, n_layers, token_id, position)
        if not build_verilator(out_dir, n_layers):
            summary.append({"layers": n_layers, "status": "build-failed"})
            continue
        result = run_verilator(out_dir, n_layers)
        if result is None:
            summary.append({"layers": n_layers, "status": "sim-failed"})
            continue
        cycles, final_hex = result
        final_int = hex_to_int_vec(final_hex, HID, ABITS)
        # A "plausible token" prediction: feed final_norm into the
        # transformers fp32 lm_head + argmax. We bypass the int-tied
        # lm_head to keep this honest.
        with safe_open(str(SMOLLM_DIR / "model.safetensors"),
                       framework="pt") as f:
            lm_head_W = f.get_tensor(
                "model.embed_tokens.weight").to(torch.float32)
        h_f = torch.tensor(final_int, dtype=torch.float32) / 64.0
        logits = lm_head_W @ h_f
        argmax = int(logits.argmax().item())
        # Cycle counts and predicted token tell us the build + sim ran
        # end-to-end with real weights.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(SMOLLM_DIR))
        decoded = tok.decode([argmax])
        summary.append({
            "layers": n_layers,
            "status": "OK",
            "cycles": cycles,
            "final_norm_max": max(abs(v) for v in final_int),
            "argmax_id": argmax,
            "argmax_decoded": decoded,
        })
        print(f"[L{n_layers}] cycles={cycles}  "
              f"final_norm absmax={max(abs(v) for v in final_int)}  "
              f"argmax={argmax} ({decoded!r})")

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
