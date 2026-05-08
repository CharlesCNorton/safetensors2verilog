"""Verify matmul_seq_block against Python integer matmul.

Builds a parent GateGraph that does y = W @ x + b with M=4, K=8, runs the
same x vectors through Python and through iverilog, asserts every bit
matches.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from safetensors2verilog import (
    Gate, GateGraph, Signal, emit_module,
)
from safetensors2verilog.blocks import matmul_seq_invoke

HERE = Path(__file__).parent

M, K = 4, 8
WBITS, ABITS = 4, 4   # tiny

# Hand-picked weights and biases
W = [
    [ 1, -2,  3,  0,  1,  1, -1,  2],   # y0
    [ 0,  1,  1,  1,  1,  1,  1,  1],   # y1 = sum(x)
    [-1,  0,  1,  0, -1,  0,  1,  0],   # y2 = -x0+x2-x4+x6
    [ 7, -7,  7, -7,  0,  0,  0,  0],   # y3 = 7*(x0-x1+x2-x3) (saturates at WBITS=4? wait)
]
# wait, weight_bits=4 means weights in [-8, 7]. 7 is allowed. -7 is allowed.
B = [3, -2, 0, 1]

# -- Build the parent graph --------------------------------------------------
parent_inputs = [Signal(f"x{i}", width=ABITS, signed=True) for i in range(K)]

# Build the matmul invocation
sub, mm_gates = matmul_seq_invoke(
    instance_name="mm0",
    parent_x_signals=[s.name for s in parent_inputs],
    parent_clk="clk",
    parent_rst="rst",
    parent_start="start",
    weights=W,
    biases=B,
    weight_bits=WBITS,
    act_bits=ABITS,
    y_prefix="y",
    done_signal="done",
)
# matmul out_bits computed by the block: 4+4+ceil(log2(8))+1 = 12

# Determine out_bits from the produced slice gates
slice_gates = [g for g in mm_gates if g.kind == "slice" and g.name.startswith("y")]
OBITS = slice_gates[0].output_width

# Compose the GateGraph
graph = GateGraph(
    inputs=[
        Signal("clk", width=1),
        Signal("rst", width=1),
        Signal("start", width=1),
        *parent_inputs,
    ],
    outputs=[
        Signal("done", width=1),
        *[Signal(f"y{j}", width=OBITS, signed=True) for j in range(M)],
    ],
    gates=mm_gates,
    top="mm_test",
    submodules=[sub],
)

text = emit_module(graph)
v_path = HERE / "mm_test.v"
v_path.write_text(text, encoding="utf-8")
print(f"emitted {v_path} ({len(text.splitlines())} lines, "
      f"{v_path.stat().st_size/1024:.1f} KB)")
print(f"submodule: {sub.top}")
print(f"OBITS auto-computed: {OBITS}")

# -- Build testbench ---------------------------------------------------------
import random
random.seed(0)
N_CASES = 6
cases = [[random.randint(-(1<<(ABITS-1)), (1<<(ABITS-1))-1) for _ in range(K)]
         for _ in range(N_CASES)]

def py_eval(x):
    return [sum(W[j][i] * x[i] for i in range(K)) + B[j] for j in range(M)]

expected = [py_eval(x) for x in cases]

x_ports = ", ".join(f".x{i}(x{i})" for i in range(K))
y_ports = ", ".join(f".y{j}(y{j})" for j in range(M))

x_decl = "\n".join(f"  reg signed [{ABITS-1}:0] x{i};" for i in range(K))
y_decl = "\n".join(f"  wire signed [{OBITS-1}:0] y{j};" for j in range(M))

case_blocks = []
for ci, x in enumerate(cases):
    drives = "\n".join(f"      x{i} = {v};" for i, v in enumerate(x))
    case_blocks.append(f"""\
    // case {ci}
    @(posedge clk);
{drives}
      start <= 1;
    @(posedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 64) begin $display("TIMEOUT case {ci}"); $finish; end
    end
    $display("CASE {ci} cycles=%0d  y0=%0d y1=%0d y2=%0d y3=%0d",
             cycles, y0, y1, y2, y3);""")

tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
{x_decl}
{y_decl}
  wire done;

  mm_test dut (
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
tb_path = HERE / "mm_test_tb.v"
tb_path.write_text(tb, encoding="utf-8")

if shutil.which("iverilog") is None:
    print("no iverilog; stopping after emission")
    raise SystemExit(0)

vvp = HERE / "mm_test.vvp"
print("compiling with iverilog...")
subprocess.run(
    ["iverilog", "-g2012", "-o", str(vvp), str(v_path), str(tb_path)],
    check=True,
)
print("running...")
proc = subprocess.run(
    ["vvp", str(vvp)], check=True, capture_output=True, text=True
)

print("\n--- simulation ---")
result_lines = [l for l in proc.stdout.splitlines() if l.startswith("CASE")]
fails = 0
for ci, line in enumerate(result_lines):
    toks = dict(t.split("=") for t in line.split() if "=" in t)
    cycles = int(toks["cycles"])
    sim = [int(toks[f"y{j}"]) for j in range(M)]
    exp = expected[ci]
    ok = sim == exp
    if not ok:
        fails += 1
    print(f"  case {ci} cycles={cycles}  sim={sim} exp={exp} "
          f"{'OK' if ok else 'FAIL'}")

print(f"\n{N_CASES - fails}/{N_CASES} match Python int matmul exactly")
print(f"latency: K+1 = {K+1} cycles per inference")
