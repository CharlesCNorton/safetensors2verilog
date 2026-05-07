"""End-to-end test of safetensors2verilog.frontends.hf_llama on a synthetic
1-layer LlamaForCausalLM-shaped model.

Builds a HF-layout safetensors + config.json with int weights stored as bf16
(crafted so per-channel symmetric quant returns scale=1 exactly), runs the
hf_llama frontend, emits Verilog, simulates, and confirms the lm_done pulse
fires for each token plus the argmax produces non-X token ids.

A bit-exact comparison against a Python reference is non-trivial here
because the frontend's quantization adds per-channel scales that the simple
Python ref doesn't replicate; we verify the structural pieces work and the
Verilog completes a full forward pass.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from safetensors2verilog import emit_module
from safetensors2verilog.core import registry


# Tiny shape (matches tiny_llama_e2e.py)
HID = 8
N_LAYERS = 1
H = 2
KV = 1
INTER = 16
VOCAB = 8
MAX_SEQ = 4
ABITS = 8
WBITS = 8


def _craft_int_weights(shape, *, lo=-32, hi=32, force_127=True):
    """Random int matrix [shape] in [lo, hi]. If force_127, force one
    element per row to be ±127 so per-channel quant scale = 1.
    """
    t = torch.randint(lo, hi + 1, shape, dtype=torch.int32)
    if force_127 and len(shape) >= 1:
        # for each output channel (axis=0), set one element to +127
        for j in range(shape[0]):
            if len(shape) == 2:
                t[j, j % shape[1]] = 127
            else:
                t[j] = 127
    return t.to(torch.float32).to(torch.bfloat16)


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2

    random.seed(0)
    torch.manual_seed(0)

    out_dir = Path(r"D:\safetensors2verilog\_play\hf_llama_out")
    out_dir.mkdir(exist_ok=True)

    # ---- Build the synthetic model ----
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HID,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": H,
        "num_key_value_heads": KV,
        "intermediate_size": INTER,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_SEQ,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2),
                                          encoding="utf-8")

    # All RMSNorm gammas = 1.0
    state_dict = {
        "model.embed_tokens.weight": _craft_int_weights(
            (VOCAB, HID), lo=-64, hi=64, force_127=True),
        "model.norm.weight": torch.ones(HID, dtype=torch.bfloat16),
        "lm_head.weight": _craft_int_weights(
            (VOCAB, HID), lo=-64, hi=64, force_127=True),
    }
    for li in range(N_LAYERS):
        state_dict[f"model.layers.{li}.input_layernorm.weight"] = \
            torch.ones(HID, dtype=torch.bfloat16)
        state_dict[f"model.layers.{li}.post_attention_layernorm.weight"] = \
            torch.ones(HID, dtype=torch.bfloat16)
        state_dict[f"model.layers.{li}.self_attn.q_proj.weight"] = \
            _craft_int_weights((HID, HID))
        state_dict[f"model.layers.{li}.self_attn.k_proj.weight"] = \
            _craft_int_weights((KV * (HID // H), HID))
        state_dict[f"model.layers.{li}.self_attn.v_proj.weight"] = \
            _craft_int_weights((KV * (HID // H), HID))
        state_dict[f"model.layers.{li}.self_attn.o_proj.weight"] = \
            _craft_int_weights((HID, HID))
        state_dict[f"model.layers.{li}.mlp.gate_proj.weight"] = \
            _craft_int_weights((INTER, HID))
        state_dict[f"model.layers.{li}.mlp.up_proj.weight"] = \
            _craft_int_weights((INTER, HID))
        state_dict[f"model.layers.{li}.mlp.down_proj.weight"] = \
            _craft_int_weights((HID, INTER))

    sf_path = out_dir / "model.safetensors"
    save_file(state_dict, str(sf_path))
    print(f"wrote {sf_path} ({sf_path.stat().st_size/1024:.1f} KB) + config.json")

    # ---- Frontend ----
    fe = registry.get("hf_llama")()
    graph = fe.parse(sf_path, top="tiny_llama",
                     activation_bits=ABITS, weight_bits=WBITS)
    text = emit_module(graph)
    v_path = out_dir / "tiny_llama.v"
    v_path.write_text(text, encoding="utf-8")
    print(f"frontend emitted {len(text.splitlines())} lines of Verilog "
          f"({v_path.stat().st_size/1024:.1f} KB), "
          f"{len(graph.submodules)} submodule slots")

    # ---- Build testbench ----
    pos_bits = max(1, (MAX_SEQ - 1).bit_length() + 1)
    token_bits = max(1, (VOCAB - 1).bit_length())
    lm_obits = ABITS + WBITS + max(1, (HID - 1).bit_length()) + 1

    tokens = [random.randint(0, VOCAB - 1) for _ in range(MAX_SEQ)]
    print(f"input tokens: {tokens}")

    drives = []
    for t, tid in enumerate(tokens):
        drives.append(f"""\
    @(negedge clk);
      token_id = {token_bits}'d{tid};
      position = {pos_bits}'d{t};
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 5000) begin $display("TIMEOUT t={t}"); $finish; end
    end
    $display("R {t} cyc=%0d nt=%0d logits=%h", cycles, next_token_id, logits_packed);""")

    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{token_bits-1}:0] token_id;\n"
        f"  reg [{pos_bits-1}:0] position;\n"
        "  wire done;\n"
        f"  wire signed [{VOCAB*lm_obits - 1}:0] logits_packed;\n"
        f"  wire [{token_bits-1}:0] next_token_id;\n"
        "  tiny_llama dut(.clk(clk), .rst(rst), .start(start),\n"
        "                 .token_id(token_id), .position(position),\n"
        "                 .done(done), .logits_packed(logits_packed),\n"
        "                 .next_token_id(next_token_id));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        + "\n".join(drives) +
        "\n    $finish;\n  end\nendmodule\n"
    )
    tb_path = out_dir / "tb.v"
    tb_path.write_text(tb, encoding="utf-8")

    print("Compiling with iverilog...")
    vvp = out_dir / "tb.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(v_path), str(tb_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("iverilog compile failed:")
        print(proc.stderr[:5000])
        return 1
    print("Running simulation...")
    proc = subprocess.run(
        ["vvp", str(vvp)], capture_output=True, text=True, timeout=600,
    )
    log = proc.stdout
    (out_dir / "log.txt").write_text(log, encoding="utf-8")
    print("\n--- simulation output ---")
    for line in log.splitlines():
        if line.startswith("R "):
            print(f"  {line}")

    # Check that all 4 tokens produced a clean done pulse and a non-X argmax
    by_t = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            t = int(toks[1])
            cyc = int(toks[2].split("=")[1])
            try:
                nt = int(toks[3].split("=")[1])
            except ValueError:
                nt = -1
            try:
                logits_int = int(toks[4].split("=")[1], 16)
            except ValueError:
                logits_int = None
            by_t[t] = (cyc, nt, logits_int)

    fails = 0
    for t in range(MAX_SEQ):
        if t not in by_t:
            print(f"  t={t} MISSING")
            fails += 1
            continue
        cyc, nt, logits_int = by_t[t]
        if nt < 0 or logits_int is None:
            print(f"  t={t} got X (nt={nt}, logits_int={logits_int})")
            fails += 1

    if fails == 0:
        print(f"\n{MAX_SEQ}/{MAX_SEQ} tokens completed cleanly through the "
              f"frontend-generated forward pass")
        return 0
    print(f"\n{MAX_SEQ - fails}/{MAX_SEQ} tokens completed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
