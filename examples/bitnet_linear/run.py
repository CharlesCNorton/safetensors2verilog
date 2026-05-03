"""End-to-end example: a 3-input -> 2-output ternary linear layer.

Builds a small bitnet-style safetensors file (ternary weights, integer
bias), converts it to Verilog through the bitnet_linear frontend,
simulates the result with Icarus Verilog, and cross-checks the
simulator output against a Python evaluation of the same matmul.

Usage:
    python run.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetensors2verilog.core import registry  # noqa: E402
from safetensors2verilog.verilog import emit_module  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)


# y[0] =  x0 + x1 - x2
# y[1] = -x0 + x1 + x2
# bias: [+3, -2]
W = [[1, 1, -1], [-1, 1, 1]]
B = [3, -2]


def py_eval(x):
    return [
        sum(W[j][i] * x[i] for i in range(len(x))) + B[j]
        for j in range(len(W))
    ]


TB_TEMPLATE = """\
`timescale 1ns / 1ps

module tb;
  reg signed [3:0] x0, x1, x2;
  wire signed [6:0] L0_y0;
  wire signed [6:0] L0_y1;

  bn dut (
    .x0(x0), .x1(x1), .x2(x2),
    .L0_y0(L0_y0), .L0_y1(L0_y1)
  );

  integer i, j, k;
  initial begin
    for (i = -4; i < 4; i = i + 2) begin
      for (j = -4; j < 4; j = j + 2) begin
        for (k = -4; k < 4; k = k + 2) begin
          x0 = i; x1 = j; x2 = k;
          #1;
          $display("%0d %0d %0d %0d %0d", i, j, k, L0_y0, L0_y1);
        end
      end
    end
    $finish;
  end
endmodule
"""


def main() -> int:
    safetensors_path = OUT_DIR / "bn.safetensors"
    verilog_path = OUT_DIR / "bn.v"
    tb_path = OUT_DIR / "bn_tb.v"

    print("[1/4] Building bitnet-style ternary linear layer ...")
    save_file(
        {
            "layers.0.weight": torch.tensor(W, dtype=torch.int8),
            "layers.0.bias":   torch.tensor(B, dtype=torch.int32),
        },
        str(safetensors_path),
    )

    print(f"[2/4] Converting {safetensors_path.name} -> {verilog_path.name} ...")
    frontend = registry.get("bitnet_linear")()
    graph = frontend.parse(safetensors_path, top="bn", activation_bits=4)
    verilog_path.write_text(emit_module(graph), encoding="utf-8")
    print(
        f"      {len(graph.gates)} gates, "
        f"{len(graph.inputs)} inputs, {len(graph.outputs)} outputs"
    )

    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; skipping simulation step.")
        return 0

    print("[3/4] Simulating with iverilog ...")
    tb_path.write_text(TB_TEMPLATE, encoding="utf-8")
    vvp = OUT_DIR / "bn_tb.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(verilog_path), str(tb_path)],
        check=True,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True
    )
    sim_lines = [l for l in proc.stdout.splitlines() if l and l[0] in "-0123456789"]

    print("[4/4] Cross-checking simulation against Python ...")
    failures = 0
    for line in sim_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        x = [int(parts[0]), int(parts[1]), int(parts[2])]
        sim_y = [int(parts[3]), int(parts[4])]
        py_y = py_eval(x)
        ok = sim_y == py_y
        status = "OK" if ok else "FAIL"
        print(f"  x={tuple(x)}: sim={sim_y} py={py_y} [{status}]")
        if not ok:
            failures += 1

    if failures:
        print(f"\n{failures} mismatch(es).")
        return 1
    print(f"\nAll {len(sim_lines)} cases match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
