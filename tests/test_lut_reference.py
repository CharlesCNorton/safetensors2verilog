"""Bit-exact tests for the Python LUT mirrors against the iverilog-
simulated blocks. Item 6 of the open list."""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _has_iv() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


pytestmark = pytest.mark.skipif(
    not _has_iv(), reason="iverilog/vvp not on PATH",
)


def _compile_run(text: str, top: str, tb: str, td: Path) -> str:
    v = td / f"{top}.v"
    v.write_text(text, encoding="utf-8")
    tbf = td / f"{top}_tb.v"
    tbf.write_text(tb, encoding="utf-8")
    vvp = td / f"{top}.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(v), str(tbf)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True,
        timeout=300,
    )
    return proc.stdout


def test_rsqrt_lut_eval_bit_exact_against_iverilog():
    """For every input in a representative sweep, the Python rsqrt LUT
    eval should match the iverilog-simulated rsqrt_block bit-for-bit."""
    import random
    from safetensors2verilog.blocks.rsqrt import rsqrt_block
    from safetensors2verilog.lut_reference import rsqrt_lut_eval

    in_bits = 32
    out_bits = 16
    out_frac_bits = 14
    sub = rsqrt_block(in_bits=in_bits, out_bits=out_bits,
                      out_frac_bits=out_frac_bits)
    rng = random.Random(0)
    samples = [0, 1, 2, 3, 4, 7, 15, 255, 256, 1023, 1024, 65535,
               (1 << 16), (1 << 20), (1 << 28), (1 << 31) - 1,
               (1 << in_bits) - 1]
    for _ in range(48):
        samples.append(rng.randint(0, (1 << in_bits) - 1))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build a testbench that drives every sample through the rsqrt
        # block and prints the result. iverilog is fast enough to run
        # all ~64 cases in well under a second.
        drive = "\n".join(
            f"    x = {in_bits}'d{s}; #1; "
            f'$display("R %0d %0d", x, y);'
            for s in samples
        )
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            f"  reg [{in_bits-1}:0] x;\n"
            f"  wire [{out_bits-1}:0] y;\n"
            f"  {sub.top} dut(.x(x), .y(y));\n"
            "  initial begin\n"
            f"{drive}\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    sim_outputs: dict[int, int] = {}
    for line in out.splitlines():
        if line.startswith("R "):
            _, x_s, y_s = line.split()
            sim_outputs[int(x_s)] = int(y_s)
    for x in samples:
        py = rsqrt_lut_eval(
            x, in_bits=in_bits, out_bits=out_bits,
            out_frac_bits=out_frac_bits,
        )
        sim = sim_outputs[x]
        assert py == sim, (
            f"rsqrt({x}): py={py} sim={sim} (delta={py-sim})"
        )


def test_sigmoid_lut_eval_bit_exact_against_iverilog():
    """Every 8-bit input matches between Python and iverilog."""
    from safetensors2verilog.blocks.sigmoid import sigmoid_block
    from safetensors2verilog.lut_reference import sigmoid_lut_eval

    sub = sigmoid_block(in_bits=8, out_bits=8, in_q_frac_bits=4)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg signed [7:0] x;\n"
            "  wire [7:0] y;\n"
            f"  {sub.top} dut(.x(x), .y(y));\n"
            "  integer i;\n"
            "  initial begin\n"
            "    for (i = 0; i < 256; i = i + 1) begin\n"
            "      x = i[7:0]; #1;\n"
            "      $display(\"S %0d %0d\", i, y);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    for line in out.splitlines():
        if line.startswith("S "):
            _, raw_s, y_s = line.split()
            raw = int(raw_s); sim = int(y_s)
            py = sigmoid_lut_eval(raw, in_bits=8, out_bits=8,
                                  in_q_frac_bits=4)
            assert py == sim, (
                f"sigmoid raw={raw}: py={py} sim={sim}"
            )


def test_exp_lut_eval_bit_exact_against_iverilog():
    """Every 8-bit input matches Python."""
    from safetensors2verilog.blocks.exp import exp_block
    from safetensors2verilog.lut_reference import exp_lut_eval

    sub = exp_block(in_bits=8, out_bits=12, in_q_frac_bits=4)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg signed [7:0] x;\n"
            "  wire [11:0] y;\n"
            f"  {sub.top} dut(.x(x), .y(y));\n"
            "  integer i;\n"
            "  initial begin\n"
            "    for (i = 0; i < 256; i = i + 1) begin\n"
            "      x = i[7:0]; #1;\n"
            "      $display(\"E %0d %0d\", i, y);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    for line in out.splitlines():
        if line.startswith("E "):
            _, raw_s, y_s = line.split()
            raw = int(raw_s); sim = int(y_s)
            py = exp_lut_eval(raw, in_bits=8, out_bits=12,
                              in_q_frac_bits=4)
            assert py == sim, (
                f"exp raw={raw}: py={py} sim={sim}"
            )


def test_silu_lut_eval_bit_exact_against_iverilog():
    """SiLU sequential output matches iverilog element-by-element."""
    from safetensors2verilog.blocks.silu import silu_block
    from safetensors2verilog.lut_reference import silu_lut_eval

    K = 8
    sub_silu, sub_sig = silu_block(
        K=K, abits=8, obits=8,
        sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8,
        output_shift=8,
    )
    x_vals = [-32, -16, -4, -1, 0, 1, 8, 32]
    x_packed = 0
    for i, v in enumerate(x_vals):
        x_packed |= (v & 0xff) << (i * 8)

    py_out = silu_lut_eval(
        x_vals, abits=8, obits=8,
        sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8,
        output_shift=8,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        full = sub_sig.text + "\n" + sub_silu.text
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg clk = 0; always #5 clk = ~clk;\n"
            "  reg rst = 1, start = 0;\n"
            "  reg signed [63:0] x_packed;\n"
            "  wire signed [63:0] y_packed;\n"
            "  wire done;\n"
            f"  {sub_silu.top} dut(.clk(clk), .rst(rst), .start(start),\n"
            "                     .x_packed(x_packed),\n"
            "                     .y_packed(y_packed), .done(done));\n"
            "  integer cycles, i;\n"
            "  initial begin\n"
            f"    rst = 1; x_packed = 64'h{x_packed:x};\n"
            "    #20 rst = 0;\n"
            "    @(negedge clk); start = 1;\n"
            "    @(negedge clk); start = 0;\n"
            "    cycles = 0;\n"
            "    while (!done) begin\n"
            "      @(posedge clk); cycles = cycles + 1;\n"
            "      if (cycles > 1000) begin $display(\"TIMEOUT\"); $finish; end\n"
            "    end\n"
            f"    for (i = 0; i < {K}; i = i + 1)\n"
            "      $display(\"Y %0d %0d\", i, $signed(y_packed[i*8 +: 8]));\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(full, sub_silu.top, tb, td)
    sim_out: dict[int, int] = {}
    for line in out.splitlines():
        if line.startswith("Y "):
            _, i_s, v_s = line.split()
            sim_out[int(i_s)] = int(v_s)
    for i, exp in enumerate(py_out):
        assert sim_out[i] == exp, (
            f"silu element {i}: py={exp} sim={sim_out[i]}"
        )


def test_rms_norm_lut_eval_bit_exact_against_iverilog():
    """RMSNorm output matches iverilog element-by-element."""
    from safetensors2verilog.blocks.rms_norm import rms_norm_block
    from safetensors2verilog.lut_reference import rms_norm_lut_eval

    K = 8
    abits = 8
    gamma_bits = 16
    obits = 8
    rsqrt_out_bits = 16
    rsqrt_out_frac_bits = 14
    output_shift = 14
    eps = 1e-5
    eps_q = 16

    gamma_int = [
        round(1.0 * (1 << 14)) for _ in range(K)
    ]
    sub_rms, sub_rsqrt = rms_norm_block(
        K=K, gamma_int=gamma_int, gamma_bits=gamma_bits,
        abits=abits, obits=obits, eps=eps, eps_q=eps_q,
        rsqrt_out_bits=rsqrt_out_bits,
        rsqrt_out_frac_bits=rsqrt_out_frac_bits,
        output_shift=output_shift,
    )
    x_vals = [-30, -15, -5, -1, 0, 5, 15, 28]
    x_packed = 0
    for i, v in enumerate(x_vals):
        x_packed |= (v & 0xff) << (i * abits)

    py_out = rms_norm_lut_eval(
        x_vals, gamma_int, K=K, abits=abits, gamma_bits=gamma_bits,
        obits=obits, eps=eps, eps_q=eps_q,
        rsqrt_out_bits=rsqrt_out_bits,
        rsqrt_out_frac_bits=rsqrt_out_frac_bits,
        output_shift=output_shift,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        full = sub_rsqrt.text + "\n" + sub_rms.text
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg clk = 0; always #5 clk = ~clk;\n"
            "  reg rst = 1, start = 0;\n"
            f"  reg signed [{K*abits-1}:0] x_packed;\n"
            f"  wire signed [{K*obits-1}:0] y_packed;\n"
            "  wire done;\n"
            f"  {sub_rms.top} dut(.clk(clk), .rst(rst), .start(start),\n"
            "                    .x_packed(x_packed),\n"
            "                    .y_packed(y_packed), .done(done));\n"
            "  integer cycles, i;\n"
            "  initial begin\n"
            f"    rst = 1; x_packed = {K*abits}'h{x_packed:x};\n"
            "    #20 rst = 0;\n"
            "    @(negedge clk); start = 1;\n"
            "    @(negedge clk); start = 0;\n"
            "    cycles = 0;\n"
            "    while (!done) begin\n"
            "      @(posedge clk); cycles = cycles + 1;\n"
            "      if (cycles > 10000) begin $display(\"TIMEOUT\"); $finish; end\n"
            "    end\n"
            f"    for (i = 0; i < {K}; i = i + 1)\n"
            f"      $display(\"Y %0d %0d\", i, $signed(y_packed[i*{obits} +: {obits}]));\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(full, sub_rms.top, tb, td)
    sim_out: dict[int, int] = {}
    for line in out.splitlines():
        if line.startswith("Y "):
            _, i_s, v_s = line.split()
            sim_out[int(i_s)] = int(v_s)
    for i, exp in enumerate(py_out):
        assert sim_out[i] == exp, (
            f"rms_norm element {i}: py={exp} sim={sim_out[i]}"
        )
