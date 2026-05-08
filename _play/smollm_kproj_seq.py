"""SmolLM2 k_proj 16-output slice through the new matmul_seq hierarchical path.

Compares against pure Python int matmul (W_q @ x). Same slice and same
quantization as `smollm_one_layer.py` but generated via matmul_seq_block /
matmul_seq_invoke instead of the old inline `linear` lowering.

This is the demonstration that the hierarchical backend produces faithful
Verilog at SmolLM2 scale.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from safetensors2verilog import (
    Gate, GateGraph, Signal, emit_module,
)
from safetensors2verilog.blocks import matmul_seq_invoke

ROOT = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
HERE = Path(__file__).parent

LAYER_NAME = "model.layers.0.self_attn.k_proj.weight"
OUTPUTS_KEPT = 16
ACT_BITS = 8
WEIGHT_BITS = 8

with safe_open(str(ROOT / "model.safetensors"), framework="pt") as f:
    W_full = f.get_tensor(LAYER_NAME).to(torch.float32)

W = W_full[:OUTPUTS_KEPT]
M, K = W.shape
print(f"slice: {M} of {W_full.shape[0]} k_proj outputs, K={K} inputs")

# int8 symmetric per-tensor quant
w_scale = W.abs().max().item() / 127.0
W_q = W.div(w_scale).round().clamp(-128, 127).to(torch.int32)
print(f"weight scale: {w_scale:.6f}")
print(f"W_q range: [{W_q.min().item()}, {W_q.max().item()}]")

# Build GateGraph
parent_inputs = [Signal(f"x{i}", width=ACT_BITS, signed=True) for i in range(K)]

sub, mm_gates = matmul_seq_invoke(
    instance_name="kproj",
    parent_x_signals=[s.name for s in parent_inputs],
    parent_clk="clk", parent_rst="rst", parent_start="start",
    weights=W_q.tolist(),
    weight_bits=WEIGHT_BITS,
    act_bits=ACT_BITS,
    y_prefix="y",
    done_signal="done",
)
slice_gates = [g for g in mm_gates if g.kind == "slice"]
OBITS = slice_gates[0].output_width
print(f"OBITS: {OBITS}")

graph = GateGraph(
    inputs=[Signal("clk"), Signal("rst"), Signal("start"), *parent_inputs],
    outputs=[
        Signal("done", width=1),
        *[Signal(f"y{j}", width=OBITS, signed=True) for j in range(M)],
    ],
    gates=mm_gates,
    top="kproj_seq16",
    submodules=[sub],
)

text = emit_module(graph)
v_path = HERE / "kproj_seq16.v"
v_path.write_text(text, encoding="utf-8")
print(f"emitted {v_path}: {len(text.splitlines())} lines, "
      f"{v_path.stat().st_size/1024:.1f} KB")

if shutil.which("iverilog") is None:
    print("no iverilog; stopping")
    raise SystemExit(0)

# Test cases
torch.manual_seed(0)
N = 5
x_cases = torch.randint(-32, 32, (N, K), dtype=torch.int32)
expected = (x_cases.to(torch.int64) @ W_q.to(torch.int64).t()).tolist()

x_decl = "\n".join(f"  reg signed [{ACT_BITS-1}:0] x{i};" for i in range(K))
y_decl = "\n".join(f"  wire signed [{OBITS-1}:0] y{j};" for j in range(M))
x_ports = ", ".join(f".x{i}(x{i})" for i in range(K))
y_ports = ", ".join(f".y{j}(y{j})" for j in range(M))
y_args = ", ".join(f"y{j}" for j in range(M))
y_fmt = " ".join(f"y{j}=%0d" for j in range(M))

case_blocks = []
for ci, x in enumerate(x_cases):
    drives = "\n".join(f"      x{i} = {int(v)};" for i, v in enumerate(x))
    case_blocks.append(f"""\
    @(posedge clk);
{drives}
      start <= 1;
    @(posedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > {2*K}) begin $display("TIMEOUT case {ci}"); $finish; end
    end
    $display("CASE {ci} cycles=%0d  {y_fmt}", cycles, {y_args});""")

tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
{x_decl}
{y_decl}
  wire done;

  kproj_seq16 dut (
    .clk(clk), .rst(rst), .start(start),
    {x_ports},
    .done(done), {y_ports}
  );

  integer cycles;
  initial begin
    rst = 1; #20 rst = 0;
{chr(10).join(case_blocks)}
    $finish;
  end
endmodule
"""
tb_path = HERE / "kproj_seq16_tb.v"
tb_path.write_text(tb, encoding="utf-8")
print(f"testbench: {tb_path.stat().st_size/1024:.1f} KB")

vvp = HERE / "kproj_seq16.vvp"
print("compiling...")
subprocess.run(["iverilog", "-g2012", "-o", str(vvp), str(v_path),
                str(tb_path)], check=True)
print("running...")
proc = subprocess.run(["vvp", str(vvp)], check=True,
                      capture_output=True, text=True)

import re as _re
def _y_idx(name): m = _re.search(r"y(\d+)$", name); return int(m.group(1)) if m else -1
out_names = sorted([f"y{j}" for j in range(M)], key=_y_idx)

print("\n--- simulation ---")
result_lines = [l for l in proc.stdout.splitlines() if l.startswith("CASE")]
fails = 0
for ci, line in enumerate(result_lines):
    toks = dict(t.split("=") for t in line.split() if "=" in t)
    cycles = int(toks["cycles"])
    sim = [int(toks[n]) for n in out_names]
    exp = expected[ci]
    ok = sim == exp
    if not ok:
        fails += 1
        # show first 4 mismatches
        diffs = [(i, s, e) for i, (s, e) in enumerate(zip(sim, exp)) if s != e]
        print(f"  case {ci} cycles={cycles} FAIL  first 4 diffs: "
              f"{diffs[:4]}")
    else:
        print(f"  case {ci} cycles={cycles} OK  first 3: "
              f"{sim[0]}/{sim[1]}/{sim[2]}")

print(f"\n{N - fails}/{N} match PyTorch int matmul exactly via "
      f"hierarchical matmul_seq")
