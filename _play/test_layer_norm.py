"""iverilog test for layer_norm_block.

Compares the hardware to a Python reference doing the same fixed-point
math. Pass criterion: bit-exact agreement on a small sweep.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import Gate, GateGraph, Signal, emit_module
from safetensors2verilog.blocks.layer_norm import layer_norm_block

K, ABITS, OBITS = 8, 8, 8
GBITS = BBITS = 16
EPS_Q = 16
EPS = 1e-5
RSQRT_OUT_FRAC = 14
OUTPUT_SHIFT = 14


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


def _python_rsqrt_lut(in_bits, out_bits, out_frac_bits, lut_idx_bits=8):
    n_entries = 1 << lut_idx_bits
    lut = [
        max(0, min((1 << out_bits) - 1,
                   round((1.0 / math.sqrt(1.0 + i / n_entries)) *
                         (1 << out_frac_bits))))
        for i in range(n_entries)
    ]
    sqrt_half_q = round(math.sqrt(0.5) * (1 << out_frac_bits))
    out_max = (1 << out_bits) - 1

    def rsqrt(x):
        if x == 0:
            return out_max
        p = x.bit_length() - 1
        nlz = in_bits - 1 - p
        x_norm = (x << nlz) & ((1 << in_bits) - 1)
        idx = (x_norm >> (in_bits - 1 - lut_idx_bits)) & (n_entries - 1)
        lut_val = lut[idx]
        half_p = p >> 1
        p_odd = p & 1
        shifted = lut_val >> half_p
        if p_odd:
            return (shifted * sqrt_half_q >> out_frac_bits) & out_max
        return shifted
    return rsqrt


def py_layer_norm(x, gamma_int, beta_int, *, eps_int, K_eps_int,
                  rsqrt_in_bits, py_rsqrt, output_shift,
                  abits=ABITS, obits=OBITS):
    K = len(x)
    sum_x = sum(x)
    is_pow2 = (K & (K - 1)) == 0
    if is_pow2:
        mean = sum_x >> (K.bit_length() - 1)
    else:
        recip = round((1 << 16) / K)
        mean = (sum_x * recip) >> 16
    centered = [v - mean for v in x]
    sum_sq = sum(c * c for c in centered)
    sum_sq_eps = sum_sq + K_eps_int
    rsqrt_val = py_rsqrt(sum_sq_eps)
    out = []
    out_lo, out_hi = -(1 << (obits - 1)), (1 << (obits - 1)) - 1
    for c, g, b in zip(centered, gamma_int, beta_int):
        xg = c * g
        xgr = xg * rsqrt_val
        shifted = xgr >> output_shift
        with_beta = shifted + b
        out.append(max(out_lo, min(out_hi, with_beta)))
    return out


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2

    gamma_int = [16384] * K   # ~1.0 in Q14
    beta_int = [0] * K
    rsqrt_in_bits = 2 * ABITS + max(1, K.bit_length()) + 4 + 4

    ln, rsq = layer_norm_block(
        K=K, gamma_int=gamma_int, beta_int=beta_int,
        gamma_bits=GBITS, beta_bits=BBITS,
        abits=ABITS, obits=OBITS, eps=EPS, eps_q=EPS_Q,
        rsqrt_in_bits=rsqrt_in_bits,
        rsqrt_out_bits=16, rsqrt_out_frac_bits=RSQRT_OUT_FRAC,
        output_shift=OUTPUT_SHIFT,
    )

    parent = GateGraph(
        inputs=[Signal("clk"), Signal("rst"), Signal("start"),
                Signal("x_packed", width=K * ABITS, signed=False)],
        outputs=[Signal("done", width=1),
                 Signal("y_packed", width=K * OBITS, signed=False)],
        gates=[
            Gate(name="done", kind="extern_wire", output_width=1),
            Gate(name="y_packed", kind="instance",
                 inputs=["clk", "rst", "start", "x_packed"],
                 attrs={
                     "module_name": ln.top, "instance_name": "ln",
                     "input_ports": ["clk", "rst", "start", "x_packed"],
                     "output_port": "y_packed",
                     "extra_output_ports": [("done", "done")],
                 },
                 output_width=K * OBITS, output_signed=False),
        ],
        top="ln_test",
        submodules=[ln, rsq],
    )
    text = emit_module(parent)

    py_rsqrt = _python_rsqrt_lut(rsqrt_in_bits, 16, RSQRT_OUT_FRAC)
    eps_int = round(EPS * (1 << EPS_Q))
    K_eps_int = K * eps_int

    import random
    random.seed(0)
    cases_int, expected = [], []
    for _ in range(5):
        x = [random.randint(-30, 30) for _ in range(K)]
        cases_int.append(_pack_signed(x, ABITS))
        expected.append(py_layer_norm(
            x, gamma_int, beta_int,
            eps_int=eps_int, K_eps_int=K_eps_int,
            rsqrt_in_bits=rsqrt_in_bits, py_rsqrt=py_rsqrt,
            output_shift=OUTPUT_SHIFT,
        ))

    drives = []
    for ci, x_packed in enumerate(cases_int):
        drives.append(f"""\
    @(negedge clk);
      x_packed = {K*ABITS}'h{x_packed:x};
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 200) begin $display("TIMEOUT"); $finish; end
    end
    $display("R {ci} cyc=%0d %h", cycles, y_packed);""")

    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{K*ABITS-1}:0] x_packed;\n"
        "  wire done;\n"
        f"  wire [{K*OBITS-1}:0] y_packed;\n"
        "  ln_test dut(.clk(clk), .rst(rst), .start(start),\n"
        "              .x_packed(x_packed),\n"
        "              .done(done), .y_packed(y_packed));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
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
            ["vvp", str(vvp)], capture_output=True, text=True, timeout=60,
        )

    fails = 0
    for line in proc.stdout.splitlines():
        if not line.startswith("R "):
            continue
        toks = line.split()
        ci = int(toks[1])
        sim = _unpack_signed(int(toks[3], 16), K, OBITS)
        ok = sim == expected[ci]
        print(f"  case {ci} {'OK' if ok else 'FAIL'}  sim={sim}  exp={expected[ci]}")
        if not ok:
            fails += 1
    print(f"\n{5 - fails}/5 cases bit-exact")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
