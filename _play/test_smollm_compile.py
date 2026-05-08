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
    print("  num_layers_override=1, max_seq_override=4, skip_lm_head=True")
    fe = registry.get("hf_llama")()
    graph = fe.parse(
        SMOLLM_DIR / "model.safetensors",
        top="smollm_l1",
        config=str(SMOLLM_DIR / "config.json"),
        activation_bits=8, weight_bits=8,
        num_layers_override=1, max_seq_override=4,
        skip_lm_head=True,
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

    # Build a testbench that drives one token, waits for done, captures
    # the final_norm hidden vector.
    pos_bits = max(1, (4 - 1).bit_length() + 1)   # MAX_SEQ=4
    HID = 576
    ABITS = 8
    VOCAB = 49152
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
        f"      token_id = {token_bits}'d42;\n"
        "      position = 0;\n"
        "      start <= 1;\n"
        "    @(negedge clk); start <= 0;\n"
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
    tb_path = OUT / "tb.v"
    tb_path.write_text(tb, encoding="utf-8")
    print(f"\nWrote testbench {tb_path}")

    print("Compiling with iverilog...")
    import time as _time
    t0 = _time.time()
    vvp = OUT / "smollm.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(v_path), str(tb_path)],
        cwd=str(OUT), capture_output=True, text=True, timeout=1200,
    )
    print(f"  iverilog returned {proc.returncode} in {_time.time()-t0:.1f}s")
    if proc.returncode != 0:
        print("STDERR (last 50 lines):")
        for line in proc.stderr.splitlines()[-50:]:
            print(f"  {line}")
        return 1

    print("Running simulation (writing vvp stdout to log.txt)...")
    t0 = _time.time()
    log_path = OUT / "log.txt"
    with open(log_path, "wb") as logf:
        proc = subprocess.run(
            ["vvp", str(vvp)], cwd=str(OUT),
            stdout=logf, stderr=subprocess.STDOUT,
            timeout=3600,
        )
    print(f"  vvp returned {proc.returncode} in {_time.time()-t0:.1f}s")
    print(f"  log size: {log_path.stat().st_size/1e6:.1f} MB")
    # Print just the lines we care about
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for line in log.splitlines():
        if "DONE" in line or "TIMEOUT" in line:
            print(f"  {line[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
