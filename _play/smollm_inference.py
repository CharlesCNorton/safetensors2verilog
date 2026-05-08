"""End-to-end inference: drive a token through the SmolLM2 verilator
binary (1-layer truncation), capture final_norm hidden state, finish the
forward pass on CPU (lm_head + argmax + tokenizer decode).

This is the literal "perform an inference" demo: real input text -> real
quantised SmolLM2 weights walked through Verilog -> a predicted next token.
The 1-layer truncation means the prediction is what a *severely truncated*
SmolLM2 produces, not what fp32 SmolLM2 produces. Quality is not the
target; demonstrating the data path is.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT = Path(r"D:\safetensors2verilog\_play\smollm_out")
TB_PATH = OUT / "tb.v"
VTB_BIN = "/mnt/d/safetensors2verilog/_play/smollm_out/obj_dir/Vtb"

HID = 576
ABITS = 8
VOCAB = 49152


def write_tb(token_id: int, position: int) -> None:
    """Rewrite the testbench to drive a specific token + position."""
    pos_bits = max(1, (4 - 1).bit_length() + 1)
    token_bits = max(1, (VOCAB - 1).bit_length())
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{token_bits-1}:0] token_id;\n"
        f"  reg [{pos_bits-1}:0] position;\n"
        "  wire done;\n"
        f"  wire signed [{HID*ABITS-1}:0] final_norm;\n"
        "  smollm_l1 dut(.clk(clk), .rst(rst), .start(start),\n"
        "                .token_id(token_id), .position(position),\n"
        "                .done(done), .final_norm(final_norm));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        "    @(negedge clk);\n"
        f"      token_id = {token_bits}'d{token_id};\n"
        f"      position = {pos_bits}'d{position};\n"
        "      start = 1;\n"
        "    @(negedge clk); start = 0;\n"
        "    cycles = 0;\n"
        "    while (!done) begin\n"
        "      @(posedge clk);\n"
        "      cycles = cycles + 1;\n"
        "      if (cycles > 200000) begin $display(\"TIMEOUT\"); $finish; end\n"
        "    end\n"
        "    $display(\"DONE %0d cycles  final_norm=%h\", cycles, final_norm);\n"
        "    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    TB_PATH.write_text(tb, encoding="utf-8")


def rebuild_vtb() -> None:
    """Rebuild Vtb on WSL after rewriting tb.v."""
    print("Re-running verilator (only tb.v changed; should be fast)...")
    t0 = time.time()
    proc = subprocess.run(
        ["wsl.exe", "-d", "WSLExperiments", "--", "bash", "-c",
         "cd /mnt/d/safetensors2verilog/_play/smollm_out && "
         "verilator --binary --top-module tb -j 4 "
         "--Wno-fatal --Wno-WIDTH --Wno-WIDTHEXPAND --Wno-WIDTHTRUNC "
         "--Wno-UNOPTFLAT --Wno-INFINITELOOP --Wno-UNUSEDSIGNAL "
         "--Wno-UNUSEDPARAM --Wno-CASEINCOMPLETE --Wno-INITIALDLY "
         "-O3 -CFLAGS '-O0' smollm_l1.v tb.v 2>&1 | tail -3"],
        capture_output=True, text=True, timeout=1800,
    )
    print(f"  build done in {time.time()-t0:.1f}s")
    if proc.returncode != 0:
        print("verilator/g++ failed:")
        print(proc.stderr[-2000:])
        raise SystemExit(1)


def run_vtb() -> tuple[int, str]:
    """Run the verilator binary, return (cycles, final_norm_hex)."""
    t0 = time.time()
    proc = subprocess.run(
        ["wsl.exe", "-d", "WSLExperiments", "--", "bash", "-c",
         f"cd /mnt/d/safetensors2verilog/_play/smollm_out && {VTB_BIN}"],
        capture_output=True, text=True, timeout=600,
    )
    print(f"  verilator sim ran in {time.time()-t0:.1f}s")
    out = proc.stdout
    m = re.search(r"DONE (\d+) cycles\s+final_norm=([0-9a-fA-F]+)", out)
    if not m:
        print("Sim output didn't match DONE pattern:")
        print(out[-2000:])
        raise SystemExit(1)
    return int(m.group(1)), m.group(2)


def hex_to_int_vec(hex_str: str, n: int, bits: int) -> list[int]:
    """Convert a Verilog %h dump (high-element-first) to a signed int list.

    Verilog $display(%h) emits the value as one big hex string with the most-
    significant bit on the left. Element i sits at bits [(i+1)*bits-1 : i*bits],
    so the LAST hex chunk (rightmost) is element 0.
    """
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
    # -- Tokenizer --
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(SMOLLM_DIR))

    # -- lm_head weights (fp32 from the safetensors; bypass the int8 quant
    # so the lm_head step is reference-quality) --
    print("Loading lm_head fp32 weights for CPU finish...")
    with safe_open(str(SMOLLM_DIR / "model.safetensors"), framework="pt") as f:
        # SmolLM2 ties embed and lm_head; both live at model.embed_tokens.weight
        lm_head_W = f.get_tensor("model.embed_tokens.weight").to(torch.float32)
    print(f"  lm_head weight shape: {tuple(lm_head_W.shape)}")

    # -- Choose a prompt token --
    prompt_text = "The"
    ids = tok.encode(prompt_text, add_special_tokens=False)
    print(f"prompt: {prompt_text!r}")
    print(f"token ids: {ids}")
    if not ids:
        ids = [tok.bos_token_id]
    token_id = ids[0]
    print(f"feeding first token to verilog: id={token_id} "
          f"({tok.decode([token_id])!r})")

    # -- Drive verilog --
    write_tb(token_id, 0)
    rebuild_vtb()
    cycles, final_hex = run_vtb()
    print(f"verilog forward pass: {cycles} cycles, "
          f"final_norm = {len(final_hex)} hex chars")

    # -- Unpack the int8 hidden vector --
    final_int = hex_to_int_vec(final_hex, HID, ABITS)
    print(f"final_norm int8: min={min(final_int)} max={max(final_int)} "
          f"first 8 = {final_int[:8]}")

    # -- CPU finish: dequantise the int8 hidden, apply lm_head, argmax --
    # The hardware emits final_norm in some Q-format; without proper PTQ
    # calibration we treat it as "small int" and let lm_head's fp32 scale
    # absorb the magnitude. Specifically, treat final_norm[i] as the
    # quantised representation of the true RMSNorm output, with the
    # post-RMSNorm magnitudes nominally in [-2, 2]. We map [-128, 127]
    # back to [-2, 2] via division by 64.
    h = torch.tensor(final_int, dtype=torch.float32) / 64.0
    logits = lm_head_W @ h         # [vocab]
    top_k = torch.topk(logits, k=5)
    print()
    print("Top-5 next-token predictions from the SmolLM2 1-layer Verilog model:")
    for rank, (val, idx) in enumerate(
        zip(top_k.values.tolist(), top_k.indices.tolist())
    ):
        decoded = tok.decode([idx])
        print(f"  {rank+1}. id={idx:>5}  logit={val:>8.3f}  -> {decoded!r}")

    print()
    print(f"Argmax next-token: id={top_k.indices[0].item()} "
          f"-> {tok.decode([top_k.indices[0].item()])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
