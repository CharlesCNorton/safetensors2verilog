"""End-to-end example: build a small threshold network, convert to Verilog,
simulate with iverilog, cross-check against Python.

The network is a half-adder built from boolean gates:

    sum   = A XOR B
    carry = A AND B

XOR is constructed in two layers from OR + NAND -> AND, since XOR is not
linearly separable. The whole thing uses ternary weights and small
integer biases.

Usage:
    python run.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file
from safetensors import safe_open

# Make the parent package importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetensors2verilog.core import registry  # noqa: E402
from safetensors2verilog.verilog import emit_module  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build a tiny threshold network as a safetensors file
# ---------------------------------------------------------------------------

def build_safetensors(path: Path) -> None:
    """Construct a half-adder threshold network and write it to `path`.

    Signal registry layout:
        0 -> "#0" (constant 0)
        1 -> "#1" (constant 1)
        2 -> "$a"
        3 -> "$b"
        4 -> "or_ab"
        5 -> "nand_ab"
        6 -> "sum_ab"   (XOR via or AND nand)
        7 -> "carry_ab" (AND via direct)
    """
    registry_map: Dict[str, str] = {
        "0": "#0", "1": "#1",
        "2": "$a", "3": "$b",
        "4": "or_ab", "5": "nand_ab",
        "6": "sum_ab", "7": "carry_ab",
    }

    tensors: Dict[str, torch.Tensor] = {}

    def add_gate(name: str, weights: List[int], bias: int, input_ids: List[int]) -> None:
        tensors[f"{name}.weight"] = torch.tensor(weights, dtype=torch.int8)
        tensors[f"{name}.bias"] = torch.tensor([bias], dtype=torch.int8)
        tensors[f"{name}.inputs"] = torch.tensor(input_ids, dtype=torch.int64)

    # Layer 1 of XOR: OR(a, b) and NAND(a, b)
    add_gate("or_ab",   [1, 1],  -1, [2, 3])   # H(a + b - 1)
    add_gate("nand_ab", [-1, -1], 1, [2, 3])   # H(-a - b + 1)
    # Layer 2 of XOR: AND(or, nand)
    add_gate("sum_ab", [1, 1], -2, [4, 5])     # H(or + nand - 2)
    # Carry: AND(a, b)
    add_gate("carry_ab", [1, 1], -2, [2, 3])   # H(a + b - 2)

    metadata = {"signal_registry": json.dumps(registry_map)}
    save_file(tensors, str(path), metadata=metadata)


# ---------------------------------------------------------------------------
# 2. Python reference evaluator (so we can cross-check Verilog simulation)
# ---------------------------------------------------------------------------

def evaluate(path: Path, a: int, b: int) -> Tuple[int, int]:
    """Compute (sum, carry) by walking the threshold network in Python."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        sr = json.loads(meta["signal_registry"])
        sr = {int(k): v for k, v in sr.items()}
        tensors = {k: f.get_tensor(k).clone() for k in f.keys()}

    name_to_id = {v: k for k, v in sr.items()}
    values: Dict[int, int] = {
        name_to_id["#0"]: 0,
        name_to_id["#1"]: 1,
        name_to_id["$a"]: a,
        name_to_id["$b"]: b,
    }

    # Topological order (small fixed pipeline; no need for general sort)
    order = ["or_ab", "nand_ab", "sum_ab", "carry_ab"]
    for gate in order:
        w = [int(x) for x in tensors[f"{gate}.weight"].tolist()]
        bias = int(tensors[f"{gate}.bias"].item())
        ids = [int(x) for x in tensors[f"{gate}.inputs"].tolist()]
        s = sum(weight * values[sid] for weight, sid in zip(w, ids)) + bias
        values[name_to_id[gate]] = 1 if s >= 0 else 0

    return values[name_to_id["sum_ab"]], values[name_to_id["carry_ab"]]


# ---------------------------------------------------------------------------
# 3. Verilog testbench + simulation
# ---------------------------------------------------------------------------

TESTBENCH_TEMPLATE = """\
`timescale 1ns / 1ps

module tb;
  reg a, b;
  wire sum_ab, carry_ab;

  half_adder dut (
    ._a(a),
    ._b(b),
    .sum_ab(sum_ab),
    .carry_ab(carry_ab)
  );

  integer i;
  initial begin
    for (i = 0; i < 4; i = i + 1) begin
      {a, b} = i[1:0];
      #1;
      $display("a=%b b=%b sum=%b carry=%b", a, b, sum_ab, carry_ab);
    end
    $finish;
  end
endmodule
"""


def run_iverilog(verilog_path: Path, tb_path: Path) -> List[str]:
    """Compile + simulate; return one line per stimulus from $display."""
    if shutil.which("iverilog") is None:
        raise RuntimeError("iverilog not found on PATH")
    vvp_path = OUT_DIR / "tb.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp_path), str(verilog_path), str(tb_path)],
        check=True,
    )
    proc = subprocess.run(
        ["vvp", str(vvp_path)], check=True, capture_output=True, text=True
    )
    return [
        line for line in proc.stdout.splitlines()
        if line.startswith("a=")
    ]


# ---------------------------------------------------------------------------
# 4. Driver
# ---------------------------------------------------------------------------

def main() -> int:
    safetensors_path = OUT_DIR / "half_adder.safetensors"
    verilog_path = OUT_DIR / "half_adder.v"
    tb_path = OUT_DIR / "half_adder_tb.v"

    print("[1/4] Building threshold-network safetensors ...")
    build_safetensors(safetensors_path)

    print(f"[2/4] Converting {safetensors_path.name} -> {verilog_path.name} ...")
    frontend = registry.get("threshold_logic")()
    graph = frontend.parse(safetensors_path, top="half_adder")
    verilog_path.write_text(emit_module(graph), encoding="utf-8")
    print(f"      {len(graph.gates)} gates, {len(graph.inputs)} inputs, "
          f"{len(graph.outputs)} outputs")

    print("[3/4] Simulating with iverilog ...")
    tb_path.write_text(TESTBENCH_TEMPLATE, encoding="utf-8")
    sim_lines = run_iverilog(verilog_path, tb_path)

    print("[4/4] Cross-checking against Python evaluation ...")
    failures = 0
    for line in sim_lines:
        # parse "a=X b=Y sum=Z carry=W"
        parts = dict(token.split("=") for token in line.split())
        a = int(parts["a"], 2)
        b = int(parts["b"], 2)
        sim_sum = int(parts["sum"], 2)
        sim_carry = int(parts["carry"], 2)
        py_sum, py_carry = evaluate(safetensors_path, a, b)
        truth_sum = a ^ b
        truth_carry = a & b
        ok = (sim_sum == py_sum == truth_sum) and (sim_carry == py_carry == truth_carry)
        status = "OK" if ok else "FAIL"
        print(f"  a={a} b={b}: sim=({sim_sum},{sim_carry}) "
              f"py=({py_sum},{py_carry}) truth=({truth_sum},{truth_carry}) [{status}]")
        if not ok:
            failures += 1

    if failures:
        print(f"\n{failures} mismatch(es).")
        return 1
    print("\nAll cases match across Verilog simulation, Python evaluator, "
          "and ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
