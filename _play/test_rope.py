"""Bit-exact iverilog test for RoPE.

Replicates the hardware's fixed-point math in Python and asserts identical
outputs across a sweep of (x, position) pairs.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import Gate, GateGraph, Signal, emit_module
from safetensors2verilog.blocks.rope import rope_block


def _pack_signed(values, bits):
    mask = (1 << bits) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (i * bits)
    return out


def _unpack_signed(packed, K, bits):
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    out = []
    for i in range(K):
        v = (packed >> (i * bits)) & mask
        if v & sign_bit:
            v -= 1 << bits
        out.append(v)
    return out


def _signed_arsh(x, n):
    """Arithmetic right shift on Python int matching Verilog >>>."""
    return x >> n


def _sat(x, bits):
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return max(lo, min(hi, x))


def py_rope(x_int, position, *, head_dim, max_seq, theta_base, abits,
            sincos_bits, sincos_frac_bits):
    """Bit-exact Python ref of the rope_apply Verilog block."""
    half = head_dim // 2
    sincos_max = (1 << (sincos_bits - 1)) - 1
    out = list(x_int)
    for i in range(half):
        theta = position / (theta_base ** (2 * i / head_dim))
        s_int = max(-sincos_max - 1,
                    min(sincos_max, round(math.sin(theta) * (1 << sincos_frac_bits))))
        c_int = max(-sincos_max - 1,
                    min(sincos_max, round(math.cos(theta) * (1 << sincos_frac_bits))))
        x_e = x_int[2 * i]
        x_o = x_int[2 * i + 1]
        e_cos = x_e * c_int
        e_sin = x_e * s_int
        o_cos = x_o * c_int
        o_sin = x_o * s_int
        y_e = _signed_arsh(e_cos - o_sin, sincos_frac_bits)
        y_o = _signed_arsh(e_sin + o_cos, sincos_frac_bits)
        out[2 * i] = _sat(y_e, abits)
        out[2 * i + 1] = _sat(y_o, abits)
    return out


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2

    HEAD_DIM = 8
    MAX_SEQ = 16
    THETA = 10000.0
    ABITS = 8
    SINCOS_BITS = 16
    SINCOS_FRAC = 14

    sub = rope_block(
        head_dim=HEAD_DIM, max_seq=MAX_SEQ, theta_base=THETA,
        abits=ABITS, sincos_bits=SINCOS_BITS, sincos_frac_bits=SINCOS_FRAC,
    )
    pos_bits = (MAX_SEQ - 1).bit_length()

    parent = GateGraph(
        inputs=[
            Signal("x_packed", width=HEAD_DIM * ABITS, signed=True),
            Signal("position", width=pos_bits, signed=False),
        ],
        outputs=[Signal("y_packed", width=HEAD_DIM * ABITS, signed=True)],
        gates=[
            Gate(
                name="y_packed", kind="instance",
                inputs=["x_packed", "position"],
                attrs={
                    "module_name": sub.top, "instance_name": "rope_inst",
                    "input_ports": ["x_packed", "position"],
                    "output_port": "y_packed",
                },
                output_width=HEAD_DIM * ABITS, output_signed=True,
            ),
        ],
        top="rope_test", submodules=[sub],
    )
    text = emit_module(parent)
    print(f"emitted {len(text.splitlines())} lines of Verilog "
          f"(rope module + harness)")

    import random
    random.seed(0)
    cases = []
    for _ in range(8):
        x = [random.randint(-100, 100) for _ in range(HEAD_DIM)]
        pos = random.randint(0, MAX_SEQ - 1)
        cases.append((x, pos))
    expected = [
        py_rope(x, pos, head_dim=HEAD_DIM, max_seq=MAX_SEQ,
                theta_base=THETA, abits=ABITS,
                sincos_bits=SINCOS_BITS, sincos_frac_bits=SINCOS_FRAC)
        for x, pos in cases
    ]

    drives = []
    for ci, (x, pos) in enumerate(cases):
        packed = _pack_signed(x, ABITS)
        drives.append(
            f"    x_packed = {HEAD_DIM*ABITS}'h{packed:x}; "
            f"position = {pos_bits}'d{pos}; #1; "
            f"$display(\"R {ci} %h\", y_packed);"
        )
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  reg signed [{HEAD_DIM*ABITS-1}:0] x_packed;\n"
        f"  reg [{pos_bits-1}:0] position;\n"
        f"  wire signed [{HEAD_DIM*ABITS-1}:0] y_packed;\n"
        "  rope_test dut(.x_packed(x_packed), .position(position), "
        ".y_packed(y_packed));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n  end\nendmodule\n"
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dut.v").write_text(text, encoding="utf-8")
        (td / "tb.v").write_text(tb, encoding="utf-8")
        vvp = td / "out.vvp"
        proc = subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp),
             str(td / "dut.v"), str(td / "tb.v")],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("iverilog failed:", proc.stderr)
            return 1
        proc = subprocess.run(
            ["vvp", str(vvp)], check=True, capture_output=True, text=True
        )

    by_idx = {}
    for line in proc.stdout.splitlines():
        if line.startswith("R "):
            toks = line.split()
            by_idx[int(toks[1])] = int(toks[2], 16)

    fails = []
    for ci, (x, pos) in enumerate(cases):
        sim = _unpack_signed(by_idx[ci], HEAD_DIM, ABITS)
        if sim != expected[ci]:
            fails.append((ci, x, pos, sim, expected[ci]))

    print(f"\n{len(cases) - len(fails)}/{len(cases)} cases bit-exact")
    if fails:
        print(f"first fail: case {fails[0][0]} x={fails[0][1]} pos={fails[0][2]}")
        print(f"  sim={fails[0][3]}")
        print(f"  exp={fails[0][4]}")
        return 1
    print(f"head_dim={HEAD_DIM}, max_seq={MAX_SEQ}, theta={THETA}, "
          f"abits={ABITS}, sincos=Q1.{SINCOS_FRAC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
