"""End-to-end iverilog tests for the combinational primitives:
  sigmoid, exp, requantize, argmax, rsqrt.

For each block: build it, wrap in a one-instance harness module, simulate
with iverilog over a sweep of inputs, compare to the Python reference.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import (
    Gate, GateGraph, Signal, emit_module,
)
from safetensors2verilog.blocks.argmax import argmax_block
from safetensors2verilog.blocks.exp import exp_block
from safetensors2verilog.blocks.requantize import requantize_block
from safetensors2verilog.blocks.rsqrt import rsqrt_block
from safetensors2verilog.blocks.sigmoid import sigmoid_block

PASS, FAIL = 0, 0


def _run_iverilog(td: Path, dut_text: str, tb_text: str) -> str:
    (td / "dut.v").write_text(dut_text, encoding="utf-8")
    (td / "tb.v").write_text(tb_text, encoding="utf-8")
    vvp = td / "out.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp),
         str(td / "dut.v"), str(td / "tb.v")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"iverilog failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True
    )
    return proc.stdout


def _check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if ok: PASS += 1
    else:  FAIL += 1


# ---------------------------------------------------------------------------
# 1. sigmoid
# ---------------------------------------------------------------------------
def test_sigmoid():
    print("\n== sigmoid ==")
    sub = sigmoid_block(in_bits=8, out_bits=8, in_q_frac_bits=4)
    # Wrap in a passthrough harness module so emit_module has a parent.
    parent = GateGraph(
        inputs=[Signal("x", width=8, signed=True)],
        outputs=[Signal("y", width=8, signed=False)],
        gates=[
            Gate(
                name="y", kind="instance",
                inputs=["x"],
                attrs={
                    "module_name": sub.top, "instance_name": "sig_inst",
                    "input_ports": ["x"], "output_port": "y",
                },
                output_width=8, output_signed=False,
            ),
        ],
        top="sig_test",
        submodules=[sub],
    )
    text = emit_module(parent)

    # Sweep all 256 input bit patterns.
    cases = list(range(-128, 128))
    drives = []
    for ci, x in enumerate(cases):
        # Verilog signed literal is awkward for negative numbers; pass as a
        # masked integer cast.
        masked = x & 0xff
        drives.append(f"    x = 8'h{masked:02x}; #1; "
                      f"$display(\"R %0d %0d\", {ci}, y);")
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg signed [7:0] x;\n"
        "  wire [7:0] y;\n"
        "  sig_test dut(.x(x), .y(y));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\n"
        "endmodule\n"
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)

    by_idx = {}
    for line in log.splitlines():
        if line.startswith("R "):
            _, ci, y = line.split()
            by_idx[int(ci)] = int(y)

    # Python reference
    fails_at = []
    for ci, x in enumerate(cases):
        x_real = x / 16.0
        x_clamped = max(-8.0, min(8.0, x_real))
        s = 1.0 / (1.0 + math.exp(-x_clamped))
        expected = max(0, min(255, round(s * 256)))
        sim = by_idx.get(ci, -1)
        if sim != expected:
            fails_at.append((ci, x, sim, expected))
    _check("sigmoid sweep over [-128, 127]",
           len(fails_at) == 0,
           "" if not fails_at else f"first fail: {fails_at[0]}")


# ---------------------------------------------------------------------------
# 2. exp
# ---------------------------------------------------------------------------
def test_exp():
    print("\n== exp ==")
    sub = exp_block(in_bits=8, out_bits=12, in_q_frac_bits=4,
                    in_clamp=(-16.0, 0.0))
    parent = GateGraph(
        inputs=[Signal("x", width=8, signed=True)],
        outputs=[Signal("y", width=12, signed=False)],
        gates=[
            Gate(
                name="y", kind="instance",
                inputs=["x"],
                attrs={
                    "module_name": sub.top, "instance_name": "exp_inst",
                    "input_ports": ["x"], "output_port": "y",
                },
                output_width=12, output_signed=False,
            ),
        ],
        top="exp_test",
        submodules=[sub],
    )
    text = emit_module(parent)

    cases = list(range(-128, 128))
    drives = []
    for ci, x in enumerate(cases):
        masked = x & 0xff
        drives.append(f"    x = 8'h{masked:02x}; #1; "
                      f"$display(\"R %0d %0d\", {ci}, y);")
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg signed [7:0] x;\n"
        "  wire [11:0] y;\n"
        "  exp_test dut(.x(x), .y(y));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    by_idx = {int(l.split()[1]): int(l.split()[2])
              for l in log.splitlines() if l.startswith("R ")}

    out_max = (1 << 12) - 1
    fails_at = []
    for ci, x in enumerate(cases):
        x_real = x / 16.0
        x_clamped = max(-16.0, min(0.0, x_real))
        v = math.exp(x_clamped)
        expected = max(0, min(out_max, round(v * out_max)))
        sim = by_idx.get(ci, -1)
        if sim != expected:
            fails_at.append((ci, x, sim, expected))
    _check("exp sweep over [-128, 127]",
           len(fails_at) == 0,
           "" if not fails_at else f"first fail: {fails_at[0]}")


# ---------------------------------------------------------------------------
# 3. requantize
# ---------------------------------------------------------------------------
def test_requantize():
    print("\n== requantize ==")
    K = 4
    IN_BITS = 16
    OUT_BITS = 8
    MUL_BITS = 8
    muls = [2, -3, 5, 1]
    shifts = [4, 3, 4, 0]

    sub = requantize_block(
        K=K, in_bits=IN_BITS, out_bits=OUT_BITS,
        muls=muls, shifts=shifts, mul_bits=MUL_BITS,
    )
    parent = GateGraph(
        inputs=[Signal("x_packed", width=K * IN_BITS, signed=True)],
        outputs=[Signal("y_packed", width=K * OUT_BITS, signed=True)],
        gates=[
            Gate(
                name="y_packed", kind="instance",
                inputs=["x_packed"],
                attrs={
                    "module_name": sub.top, "instance_name": "rq_inst",
                    "input_ports": ["x_packed"], "output_port": "y_packed",
                },
                output_width=K * OUT_BITS, output_signed=True,
            ),
        ],
        top="rq_test",
        submodules=[sub],
    )
    text = emit_module(parent)

    out_lo = -(1 << (OUT_BITS - 1)) + 1
    out_hi = (1 << (OUT_BITS - 1)) - 1
    cases = [
        [0, 0, 0, 0],
        [100, 200, -50, 75],
        [3000, -3000, 12000, -12000],
        [(1 << 15) - 1, -(1 << 15), 32, -32],
    ]
    expected = []
    for x in cases:
        row = []
        for j in range(K):
            v = (x[j] * muls[j])
            # Python arithmetic shift: floor division by 2**shift
            if shifts[j] >= 0:
                v = v >> shifts[j]
            v = max(out_lo, min(out_hi, v))
            row.append(v)
        expected.append(row)

    fmt = " ".join("%0d" for _ in range(K))
    args = ", ".join(
        f"$signed(y_packed[{(j+1)*OUT_BITS-1}:{j*OUT_BITS}])"
        for j in range(K)
    )
    drives = []
    for ci, x in enumerate(cases):
        packed = 0
        for j in range(K):
            packed |= (x[j] & ((1 << IN_BITS) - 1)) << (j * IN_BITS)
        drives.append(
            f'    x_packed = {K * IN_BITS}\'h{packed:x}; #1;\n'
            f'    $display("R {ci} {fmt}", {args});'
        )
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  reg signed [{K*IN_BITS-1}:0] x_packed;\n"
        f"  wire signed [{K*OUT_BITS-1}:0] y_packed;\n"
        "  rq_test dut(.x_packed(x_packed), .y_packed(y_packed));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)

    fails = []
    for line in log.splitlines():
        if not line.startswith("R "):
            continue
        toks = line.split()
        ci = int(toks[1])
        sim = [int(t) for t in toks[2:]]
        if sim != expected[ci]:
            fails.append((ci, sim, expected[ci]))
    _check("requantize 4 cases x 4 channels",
           len(fails) == 0,
           "" if not fails else f"first fail: {fails[0]}")


# ---------------------------------------------------------------------------
# 4. argmax
# ---------------------------------------------------------------------------
def test_argmax():
    print("\n== argmax ==")
    K, ABITS = 7, 8   # odd K stresses the tournament tree's odd-leftover path
    sub = argmax_block(K=K, abits=ABITS)
    idx_bits = (K - 1).bit_length()
    parent = GateGraph(
        inputs=[Signal("x_packed", width=K * ABITS, signed=True)],
        outputs=[Signal("argmax_idx", width=idx_bits, signed=False)],
        gates=[
            Gate(
                name="argmax_idx", kind="instance",
                inputs=["x_packed"],
                attrs={
                    "module_name": sub.top, "instance_name": "am_inst",
                    "input_ports": ["x_packed"], "output_port": "argmax_idx",
                },
                output_width=idx_bits, output_signed=False,
            ),
        ],
        top="am_test",
        submodules=[sub],
    )
    text = emit_module(parent)

    import random
    random.seed(0)
    cases = [[random.randint(-127, 127) for _ in range(K)] for _ in range(8)]
    # also include a tie at index 0 case (python max returns first; the
    # tournament "if a >= b" also keeps the lower index when tied).
    cases.append([5, 5, 5, 4, 3, 2, 1])

    expected = []
    for x in cases:
        # argmax with leftmost-tie-breaking
        m = max(x)
        expected.append(x.index(m))

    drives = []
    for ci, x in enumerate(cases):
        packed = 0
        for j in range(K):
            packed |= (x[j] & ((1 << ABITS) - 1)) << (j * ABITS)
        drives.append(
            f"    x_packed = {K*ABITS}'h{packed:x}; #1; "
            f"$display(\"R {ci} %0d\", argmax_idx);"
        )
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  reg signed [{K*ABITS-1}:0] x_packed;\n"
        f"  wire [{idx_bits-1}:0] argmax_idx;\n"
        "  am_test dut(.x_packed(x_packed), .argmax_idx(argmax_idx));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    by_idx = {int(l.split()[1]): int(l.split()[2])
              for l in log.splitlines() if l.startswith("R ")}
    fails = []
    for ci, x in enumerate(cases):
        sim = by_idx.get(ci, -1)
        if sim != expected[ci]:
            fails.append((ci, x, sim, expected[ci]))
    _check(f"argmax K={K}, {len(cases)} cases",
           len(fails) == 0,
           "" if not fails else f"first fail: {fails[0]}")


# ---------------------------------------------------------------------------
# 5. rsqrt
# ---------------------------------------------------------------------------
def test_rsqrt():
    print("\n== rsqrt ==")
    IN_BITS = 16   # smaller for sweep speed
    OUT_BITS = 16
    OUT_FRAC = 14
    sub = rsqrt_block(
        in_bits=IN_BITS, out_bits=OUT_BITS,
        out_frac_bits=OUT_FRAC, lut_idx_bits=8,
    )
    parent = GateGraph(
        inputs=[Signal("x", width=IN_BITS, signed=False)],
        outputs=[Signal("y", width=OUT_BITS, signed=False)],
        gates=[
            Gate(
                name="y", kind="instance",
                inputs=["x"],
                attrs={
                    "module_name": sub.top, "instance_name": "rs_inst",
                    "input_ports": ["x"], "output_port": "y",
                },
                output_width=OUT_BITS, output_signed=False,
            ),
        ],
        top="rs_test",
        submodules=[sub],
    )
    text = emit_module(parent)

    # Sample inputs spanning the dynamic range
    import random
    random.seed(1)
    cases = sorted({1, 2, 3, 4, 16, 100, 1000, 10000, 65535} |
                   {random.randint(1, 65535) for _ in range(40)})
    drives = []
    for ci, x in enumerate(cases):
        drives.append(f"    x = {IN_BITS}'h{x:x}; #1; "
                      f"$display(\"R {ci} %0d\", y);")
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  reg [{IN_BITS-1}:0] x;\n"
        f"  wire [{OUT_BITS-1}:0] y;\n"
        "  rs_test dut(.x(x), .y(y));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    by_idx = {int(l.split()[1]): int(l.split()[2])
              for l in log.splitlines() if l.startswith("R ")}

    # Approximate-correctness check: each rsqrt output should be within
    # ~2^-(out_frac - lut_idx_bits) ~ 2^-6 relative error of the true value.
    # That's ~1.5%. Allow 5% slack for the half-power-of-two rescale.
    fails = []
    for ci, x in enumerate(cases):
        true = 1.0 / math.sqrt(x)
        sim_int = by_idx.get(ci, -1)
        sim = sim_int / (1 << OUT_FRAC)
        rel_err = abs(sim - true) / true if true > 0 else 0.0
        if rel_err > 0.05:
            fails.append((ci, x, sim, true, rel_err))
    _check(f"rsqrt {len(cases)} cases within 5% relative error",
           len(fails) == 0,
           "" if not fails else f"first fail: {fails[0]}")


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2
    test_sigmoid()
    test_exp()
    test_requantize()
    test_argmax()
    test_rsqrt()
    print(f"\nTotal: {PASS} passed, {FAIL} failed.")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
