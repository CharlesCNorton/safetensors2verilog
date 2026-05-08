"""Bit-exact iverilog test for matmul_streaming_block + argmax_stream_block.

Drives a small matmul (M=8, K=4) through the streaming block, captures all
outputs and the final argmax, compares against Python int matmul.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import (
    Gate, GateGraph, Signal, emit_module, collect_sidecar_files,
)
from safetensors2verilog.blocks.matmul_stream import (
    matmul_streaming_block, argmax_stream_block,
)

M, K = 8, 4
ABITS, WBITS = 8, 8


def _pack_signed(values, bits):
    mask = (1 << bits) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (i * bits)
    return out


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2

    # Hand weights
    W = [
        [1, -1, 0, 2],
        [3, 0, -2, 1],
        [-1, 2, 1, -1],
        [0, 1, 1, 1],
        [4, -3, 0, 2],
        [-5, 1, 1, 1],
        [2, 2, 2, 2],
        [-1, -1, 1, 1],
    ]
    B = [10, -3, 0, 5, -1, 1, 0, 2]

    OUT_BITS = ABITS + WBITS + max(1, (K - 1).bit_length()) + 1   # 19
    mm = matmul_streaming_block(
        weights=W, weight_bits=WBITS, act_bits=ABITS, biases=B,
    )
    am = argmax_stream_block(M=M, abits=OUT_BITS)
    j_bits = max(1, (M - 1).bit_length() + 1)

    # Build GateGraph
    parent = GateGraph(
        inputs=[
            Signal("clk"), Signal("rst"), Signal("start"),
            Signal("x_packed", width=K * ABITS, signed=True),
        ],
        outputs=[
            Signal("done", width=1),
            Signal("argmax_idx", width=j_bits, signed=False),
            Signal("argmax_value", width=OUT_BITS, signed=True),
        ],
        gates=[
            # The instance gate's name BECOMES the parent wire that connects
            # to the submodule's primary output port. Extra outputs need
            # extern_wires for their parent-side signals.
            Gate(name="j_idx", kind="extern_wire", output_width=j_bits),
            Gate(name="y_value", kind="extern_wire", output_width=OUT_BITS,
                 output_signed=True),
            Gate(name="done", kind="extern_wire", output_width=1),
            Gate(
                name="y_valid", kind="instance",
                inputs=["clk", "rst", "start", "x_packed"],
                attrs={
                    "module_name": mm.top, "instance_name": "mm",
                    "input_ports": ["clk", "rst", "start", "x_packed"],
                    "output_port": "y_valid",
                    "extra_output_ports": [
                        ("j_idx", "j_idx"),
                        ("y_value", "y_value"),
                        ("done", "done"),
                    ],
                },
                output_width=1, output_signed=False,
            ),
            Gate(name="argmax_value", kind="extern_wire",
                 output_width=OUT_BITS, output_signed=True),
            Gate(
                name="argmax_idx", kind="instance",
                inputs=["clk", "rst", "start", "y_valid", "j_idx", "y_value"],
                attrs={
                    "module_name": am.top, "instance_name": "am",
                    "input_ports": ["clk", "rst", "start", "y_valid",
                                    "j_idx", "y_value"],
                    "output_port": "argmax_idx",
                    "extra_output_ports": [("argmax_value", "argmax_value")],
                },
                output_width=j_bits, output_signed=False,
            ),
        ],
        top="ms_test",
        submodules=[mm, am],
    )
    text = emit_module(parent)
    sidecar = collect_sidecar_files(parent)

    # Test cases
    cases = [
        [10, -5, 7, -2],
        [0, 0, 0, 0],
        [3, 3, 3, 3],
        [-10, 10, -10, 10],
    ]
    expected = []
    for x in cases:
        ys = [sum(W[j][i] * x[i] for i in range(K)) + B[j] for j in range(M)]
        argmax_idx = max(range(M), key=lambda j: ys[j])
        expected.append((ys, argmax_idx))

    drives = []
    for ci, x in enumerate(cases):
        x_h = _pack_signed(x, ABITS)
        drives.append(f"""\
    @(negedge clk);
      x_packed = {K*ABITS}'h{x_h:x};
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 200) begin $display("TIMEOUT case {ci}"); $finish; end
    end
    @(posedge clk);
    $display("R {ci} cyc=%0d argmax_idx=%0d argmax_value=%0d",
             cycles, argmax_idx, argmax_value);""")

    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg signed [{K*ABITS-1}:0] x_packed;\n"
        "  wire done;\n"
        f"  wire [{j_bits-1}:0] argmax_idx;\n"
        f"  wire signed [{OUT_BITS-1}:0] argmax_value;\n"
        "  ms_test dut(.clk(clk), .rst(rst), .start(start),\n"
        "              .x_packed(x_packed), .done(done),\n"
        "              .argmax_idx(argmax_idx), .argmax_value(argmax_value));\n"
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
        for fn, contents in sidecar.items():
            (td / fn).write_text(contents, encoding="utf-8")
        vvp = td / "out.vvp"
        proc = subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp),
             str(td / "dut.v"), str(td / "tb.v")],
            cwd=str(td), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("iverilog failed:")
            print(proc.stderr)
            return 1
        proc = subprocess.run(
            ["vvp", str(vvp)], cwd=str(td),
            capture_output=True, text=True, timeout=60,
        )

    by_ci = {}
    for line in proc.stdout.splitlines():
        if line.startswith("R "):
            toks = line.split()
            ci = int(toks[1])
            cyc = int(toks[2].split("=")[1])
            am_idx = int(toks[3].split("=")[1])
            am_val = int(toks[4].split("=")[1])
            by_ci[ci] = (cyc, am_idx, am_val)

    fails = 0
    for ci, x in enumerate(cases):
        ys, exp_idx = expected[ci]
        cyc, sim_idx, sim_val = by_ci.get(ci, (-1, -1, -1))
        ok = sim_idx == exp_idx and sim_val == ys[exp_idx]
        status = "OK" if ok else "FAIL"
        print(f"  case {ci} cyc={cyc} sim_argmax={sim_idx} value={sim_val}  "
              f"exp_argmax={exp_idx} value={ys[exp_idx]}  [{status}]")
        if not ok:
            fails += 1
    print(f"\n{len(cases) - fails}/{len(cases)} cases bit-exact")
    print(f"M={M}, K={K}, latency = M*K + small overhead = ~{M*K} cycles per token")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
