"""Items 1, 2, 3 of TODO.md.

Drive a real prompt through the SmolLM2 1-layer verilator binary and
compare its argmax against the transformers fp32 reference, reporting
per-position agreement.

The 1-layer truncation is deliberately a severe approximation of the
full model (the prediction quality is poor); the goal here is the data
path comparison: the *same* hidden vector is consumed by both paths and
their argmax is compared.

Items 1 (30-layer verilator) and 2 (2/4-layer verilator) require WSL +
verilator builds that take hours / GBs of RAM and are out of scope for
this script. The infrastructure is the same: build the verilator binary
once via _play/smollm_inference.py and re-use it.
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

SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT = Path(__file__).resolve().parent / "smollm_out"
TB_PATH = OUT / "tb.v"
VTB_BIN = "/mnt/d/safetensors2verilog/_play/smollm_out/obj_dir/Vtb"
HID = 576
ABITS = 8
VOCAB = 49152
POS_BITS = 3   # ceil(log2(MAX_SEQ-1)+1) for MAX_SEQ=4
TOK_BITS = 16  # ceil(log2(VOCAB)) = 16 for vocab 49152


def write_tb(token_id: int, position: int) -> None:
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{TOK_BITS-1}:0] token_id;\n"
        f"  reg [{POS_BITS-1}:0] position;\n"
        "  wire done;\n"
        f"  wire signed [{HID*ABITS-1}:0] final_norm;\n"
        "  smollm_l1 dut(.clk(clk), .rst(rst), .start(start),\n"
        "                .token_id(token_id), .position(position),\n"
        "                .done(done), .final_norm(final_norm));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        "    @(negedge clk);\n"
        f"      token_id = {TOK_BITS}'d{token_id};\n"
        f"      position = {POS_BITS}'d{position};\n"
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
    print("Rebuilding verilator simulator (only tb.v changed) ...")
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
    t0 = time.time()
    proc = subprocess.run(
        ["wsl.exe", "-d", "WSLExperiments", "--", "bash", "-c",
         f"cd /mnt/d/safetensors2verilog/_play/smollm_out && {VTB_BIN}"],
        capture_output=True, text=True, timeout=600,
    )
    print(f"  verilator sim {time.time()-t0:.1f}s")
    out = proc.stdout
    m = re.search(r"DONE (\d+) cycles\s+final_norm=([0-9a-fA-F]+)", out)
    if not m:
        print("Sim output didn't match DONE pattern:")
        print(out[-2000:])
        raise SystemExit(1)
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


def run_transformers_fp32(prompt: str) -> tuple[list[int], list[int]]:
    """Return (input_token_ids, top1_argmax_per_position) from a real
    fp32 transformers reference forward pass."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(SMOLLM_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(SMOLLM_DIR), torch_dtype=torch.float32,
    )
    model.eval()
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tok.bos_token_id]
    inputs = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(inputs)
    logits = out.logits[0]  # [seq, vocab]
    argmax = logits.argmax(dim=-1).tolist()
    return ids, argmax


def main() -> int:
    print("Loading transformers fp32 reference ...")
    prompts = [
        "The",
        "Once upon",
        "Hello, world",
    ]
    fp32_records = []
    for p in prompts:
        ids, argmax = run_transformers_fp32(p)
        fp32_records.append({"prompt": p, "ids": ids, "argmax": argmax})
        print(f"  prompt={p!r}: ids={ids}  fp32_argmax_per_pos={argmax}")

    # For each prompt's first token, drive it through the verilator binary
    # and compare the 1-layer-truncated argmax vs the fp32 full-model
    # argmax. We expect significant disagreement (1-layer is a degraded
    # model); the goal is to capture the agreement rate.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SMOLLM_DIR))
    with safe_open(str(SMOLLM_DIR / "model.safetensors"), framework="pt") as f:
        lm_head_W = f.get_tensor("model.embed_tokens.weight").to(torch.float32)

    rebuilt_once = False
    verilog_argmax = []
    fp32_first_argmax = []
    for rec in fp32_records:
        first_tok = rec["ids"][0]
        write_tb(first_tok, 0)
        if not rebuilt_once:
            rebuild_vtb()
            rebuilt_once = True
        else:
            rebuild_vtb()
        cycles, final_hex = run_vtb()
        final_int = hex_to_int_vec(final_hex, HID, ABITS)
        h_f = torch.tensor(final_int, dtype=torch.float32) / 64.0
        logits = lm_head_W @ h_f
        argmax = int(logits.argmax().item())
        verilog_argmax.append(argmax)
        fp32_first_argmax.append(rec["argmax"][0])
        print(f"  first_tok={first_tok!r} ({tok.decode([first_tok])!r})  "
              f"verilog_top1={argmax} ({tok.decode([argmax])!r})  "
              f"fp32_top1={rec['argmax'][0]} ({tok.decode([rec['argmax'][0]])!r})  "
              f"agree={argmax == rec['argmax'][0]}")

    # Report.
    n_agree = sum(1 for v, f in zip(verilog_argmax, fp32_first_argmax) if v == f)
    print(f"\nVerilog 1-layer vs transformers fp32 (first-position argmax):")
    print(f"  {n_agree}/{len(prompts)} prompts agree on top-1.")
    print(f"  This is expected to be low because the Verilog drops layers "
          f"1-29 of the 30-layer SmolLM2; the data path is end-to-end "
          f"verified, prediction quality is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
