"""Functional iverilog tests for the sequential bitnet variants.

For each of `--handshake`, `--streaming-input`, `--weight-bram`,
`--parallelism N`, and `--mac-sharing`, drive a stimulus through the
emitted Verilog and assert the outputs match a reference computed in
Python from the same ternary weights and inputs.

These complement the structural tests in `test_new_features.py` (which
only check that the right gates and ports exist).
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _has_iverilog() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


pytestmark = pytest.mark.skipif(
    not _has_iverilog(),
    reason="iverilog/vvp not on PATH",
)


def _bitnet_fixture(td: Path, weights: list[list[int]]) -> Path:
    """Save a 1-layer ternary linear with the given weights as a
    safetensors file the bitnet_linear frontend can parse."""
    sf = td / "m.safetensors"
    save_file({
        "layers.0.weight": torch.tensor(weights, dtype=torch.int8),
    }, str(sf))
    return sf


def _ref_outputs(weights: list[list[int]], x: list[int]) -> list[int]:
    """Pure-Python reference: y = W @ x for a 1-layer ternary linear."""
    return [
        sum(w * xi for w, xi in zip(row, x))
        for row in weights
    ]


def _compile_and_run(
    v_text: str, top: str, tb_text: str, td: Path,
) -> str:
    """Compile + run an iverilog testbench, return its stdout."""
    v = td / f"{top}.v"
    v.write_text(v_text, encoding="utf-8")
    tb = td / f"{top}_tb.v"
    tb.write_text(tb_text, encoding="utf-8")
    vvp = td / f"{top}.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(v), str(tb)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True,
        timeout=120,
    )
    return proc.stdout


def test_handshake_holds_done_until_ready_in():
    """The handshake variant should hold valid_out high after compute
    completes and only release on ready_in."""
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import registry
    weights = [[1, -1, 1, 0], [0, 1, -1, 1]]
    in_size = len(weights[0])
    out_size = len(weights)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = _bitnet_fixture(td, weights)
        g = registry.get("bitnet_linear")().parse(
            sf, top="bn_hs", sequential=True, handshake=True,
        )
        v_text = emit_module(g)
        # Output port widths from the graph itself.
        y_widths = [s.width for s in g.outputs if s.name.startswith("y")]
        x = [3, -2, 1, -4]
        ref = _ref_outputs(weights, x)

        x_drive = "\n".join(f"    x{i} = {v};" for i, v in enumerate(x))
        y_decls = "\n".join(
            f"  wire signed [{w-1}:0] y{j};"
            for j, w in enumerate(y_widths)
        )
        y_assoc = ", ".join(f".y{j}(y{j})" for j in range(out_size))
        # In handshake mode, the FSM holds DONE until ready_in. Drive
        # ready_in low for several cycles after done goes high to confirm
        # the hold; then assert ready_in and confirm valid_out drops.
        y_args = ", ".join(f"y{j}" for j in range(out_size))
        y_fmt = " ".join("%0d" for _ in range(out_size))
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0, ready_in = 0;
  reg signed [7:0] {", ".join(f"x{i}" for i in range(in_size))};
  wire done, valid_out;
{y_decls}
  bn_hs dut(.clk(clk), .rst(rst), .start(start), .ready_in(ready_in),
            {", ".join(f".x{i}(x{i})" for i in range(in_size))},
            .done(done), .valid_out(valid_out), {y_assoc});
  integer cycles, hold_cycles;
  initial begin
    rst = 1; #20 rst = 0;
{x_drive}
    @(negedge clk);
    start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!valid_out) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    $display("DONE cycles=%0d {y_fmt}", cycles, {y_args});
    // Hold valid_out for a few cycles without ready_in.
    hold_cycles = 0;
    repeat (5) begin
      @(posedge clk);
      hold_cycles = hold_cycles + 1;
      if (!valid_out) begin
        $display("FAIL valid_out dropped without ready_in at cycle %0d", hold_cycles);
        $finish;
      end
    end
    $display("HOLD ok=%0d", hold_cycles);
    // Now release.
    @(negedge clk); ready_in = 1;
    @(negedge clk); ready_in = 0;
    @(posedge clk);
    if (valid_out) $display("FAIL valid_out still high after ready_in");
    else $display("RELEASED");
    $finish;
  end
endmodule
"""
        out = _compile_and_run(v_text, "bn_hs", tb, td)
        # Parse "DONE cycles=N v0 v1 ..."
        done_line = next(l for l in out.splitlines() if l.startswith("DONE"))
        ys = [int(t) for t in done_line.split()[2:]]
        assert ys == ref, f"got {ys}, expected {ref}"
        assert "HOLD ok=5" in out
        assert "RELEASED" in out


def test_streaming_input_collects_x_via_valid_in():
    """--streaming-input: drive each input element via x + valid_in
    handshake and confirm the COMPUTE phase produces the expected
    outputs."""
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import registry
    weights = [[1, -1, 1, 0], [-1, 0, 1, 1]]
    in_size = len(weights[0])
    out_size = len(weights)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = _bitnet_fixture(td, weights)
        g = registry.get("bitnet_linear")().parse(
            sf, top="bn_si", sequential=True, streaming_input=True,
        )
        v_text = emit_module(g)
        y_widths = [s.width for s in g.outputs if s.name.startswith("y")]
        x = [2, -1, 3, 1]
        ref = _ref_outputs(weights, x)

        y_decls = "\n".join(
            f"  wire signed [{w-1}:0] y{j};"
            for j, w in enumerate(y_widths)
        )
        y_assoc = ", ".join(f".y{j}(y{j})" for j in range(out_size))
        y_args = ", ".join(f"y{j}" for j in range(out_size))
        y_fmt = " ".join("%0d" for _ in range(out_size))
        # Push each x element via x + valid_in on consecutive cycles.
        push_lines = []
        for v in x:
            push_lines.append(f"    x = {v}; valid_in = 1; @(negedge clk);")
        push_block = "\n".join(push_lines)
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0, valid_in = 0;
  reg signed [7:0] x;
  wire done, ready_out;
{y_decls}
  bn_si dut(.clk(clk), .rst(rst), .start(start), .x(x), .valid_in(valid_in),
            .done(done), .ready_out(ready_out), {y_assoc});
  integer cycles;
  initial begin
    rst = 1; #20 rst = 0;
    @(negedge clk);
    start = 1;
    @(negedge clk); start = 0;
    // Wait for ready_out before pushing.
    while (!ready_out) @(negedge clk);
{push_block}
    valid_in = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    $display("DONE cycles=%0d {y_fmt}", cycles, {y_args});
    $finish;
  end
endmodule
"""
        out = _compile_and_run(v_text, "bn_si", tb, td)
        done_line = next(l for l in out.splitlines() if l.startswith("DONE"))
        ys = [int(t) for t in done_line.split()[2:]]
        assert ys == ref, f"got {ys}, expected {ref}"


def dedent_pad(s: str) -> str:
    """Indent helper for the testbench string assembly."""
    lines = s.split("\n")
    return "\n".join("    " + l if l.strip() else l for l in lines)


def test_weight_bram_load_then_compute():
    """--weight-bram: load weights via the weight_addr/data/we ports,
    then assert start, then check outputs match the loaded weights."""
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import registry
    # Initial weights are all zeros (the safetensors fixture); we load
    # a non-zero set at runtime.
    init_weights = [[0, 0, 0], [0, 0, 0]]
    runtime_weights = [[1, -1, 1], [-1, 0, 1]]
    in_size = len(init_weights[0])
    out_size = len(init_weights)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = _bitnet_fixture(td, init_weights)
        g = registry.get("bitnet_linear")().parse(
            sf, top="bn_wb", sequential=True, weight_bram=True,
        )
        v_text = emit_module(g)
        y_widths = [s.width for s in g.outputs if s.name.startswith("y")]
        # Find the layer-idx / output-idx / position widths from the
        # graph's input ports.
        in_by_name = {s.name: s.width for s in g.inputs}
        layer_w = in_by_name["weight_addr_layer"]
        outidx_w = in_by_name["weight_addr_output"]
        pos_w = in_by_name["weight_addr_position"]
        x = [2, 3, -1]
        ref = _ref_outputs(runtime_weights, x)

        # Write each runtime weight via weight_addr_(layer, output, position).
        # Encode -1 as the 2-bit signed pattern 0b11.
        def encode(w: int) -> int:
            if w == 0:
                return 0
            if w == 1:
                return 1
            if w == -1:
                return 3
            raise ValueError(f"non-ternary weight {w}")
        load_block = ""
        for j, row in enumerate(runtime_weights):
            for p, w in enumerate(row):
                load_block += (
                    f"    weight_addr_layer = 0;\n"
                    f"    weight_addr_output = {j};\n"
                    f"    weight_addr_position = {p};\n"
                    f"    weight_data = {encode(w)};\n"
                    f"    weight_we = 1;\n"
                    f"    @(negedge clk);\n"
                )
        load_block += "    weight_we = 0;\n"

        x_drive = "\n".join(f"    x{i} = {v};" for i, v in enumerate(x))
        y_decls = "\n".join(
            f"  wire signed [{w-1}:0] y{j};"
            for j, w in enumerate(y_widths)
        )
        y_assoc = ", ".join(f".y{j}(y{j})" for j in range(out_size))
        y_args = ", ".join(f"y{j}" for j in range(out_size))
        y_fmt = " ".join("%0d" for _ in range(out_size))
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [7:0] {", ".join(f"x{i}" for i in range(in_size))};
  reg [{layer_w-1}:0] weight_addr_layer;
  reg [{outidx_w-1}:0] weight_addr_output;
  reg [{pos_w-1}:0] weight_addr_position;
  reg signed [1:0] weight_data;
  reg weight_we;
  wire done;
{y_decls}
  bn_wb dut(.clk(clk), .rst(rst), .start(start),
            {", ".join(f".x{i}(x{i})" for i in range(in_size))},
            .weight_addr_layer(weight_addr_layer),
            .weight_addr_output(weight_addr_output),
            .weight_addr_position(weight_addr_position),
            .weight_data(weight_data), .weight_we(weight_we),
            .done(done), {y_assoc});
  integer cycles;
  initial begin
    rst = 1; weight_we = 0;
    #20 rst = 0;
    @(negedge clk);
{load_block}
    @(negedge clk);
{x_drive}
    @(negedge clk);
    start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    $display("DONE cycles=%0d {y_fmt}", cycles, {y_args});
    $finish;
  end
endmodule
"""
        out = _compile_and_run(v_text, "bn_wb", tb, td)
        done_line = next(l for l in out.splitlines() if l.startswith("DONE"))
        ys = [int(t) for t in done_line.split()[2:]]
        assert ys == ref, (
            f"weight_bram: got {ys}, expected {ref} after weight load"
        )


def test_parallelism_outputs_match_baseline():
    """--parallelism N should produce identical outputs to the baseline
    (no parallelism), at the cost of more cycles."""
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import registry
    weights = [[1, -1, 1, 0],
               [0, 1, -1, 1],
               [1, 0, 1, -1],
               [-1, 1, 0, 1]]
    in_size = len(weights[0])
    out_size = len(weights)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = _bitnet_fixture(td, weights)
        x = [2, -3, 1, -1]
        ref = _ref_outputs(weights, x)

        results: dict[str, list[int]] = {}
        for label, kw in (
            ("baseline", {}),
            ("parallelism2", {"parallelism": 2}),
        ):
            g = registry.get("bitnet_linear")().parse(
                sf, top=f"bn_{label}", sequential=True, **kw,
            )
            v_text = emit_module(g)
            y_widths = [s.width for s in g.outputs if s.name.startswith("y")]
            x_drive = "\n".join(f"    x{i} = {v};" for i, v in enumerate(x))
            y_decls = "\n".join(
                f"  wire signed [{w-1}:0] y{j};"
                for j, w in enumerate(y_widths)
            )
            y_assoc = ", ".join(f".y{j}(y{j})" for j in range(out_size))
            y_args = ", ".join(f"y{j}" for j in range(out_size))
            y_fmt = " ".join("%0d" for _ in range(out_size))
            tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [7:0] {", ".join(f"x{i}" for i in range(in_size))};
  wire done;
{y_decls}
  bn_{label} dut(.clk(clk), .rst(rst), .start(start),
            {", ".join(f".x{i}(x{i})" for i in range(in_size))},
            .done(done), {y_assoc});
  integer cycles;
  initial begin
    rst = 1; #20 rst = 0;
{x_drive}
    @(negedge clk);
    start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    $display("DONE cycles=%0d {y_fmt}", cycles, {y_args});
    $finish;
  end
endmodule
"""
            out = _compile_and_run(v_text, f"bn_{label}", tb, td)
            done_line = next(l for l in out.splitlines() if l.startswith("DONE"))
            tokens = done_line.split()
            cycles = int(tokens[1].split("=")[1])
            ys = [int(t) for t in tokens[2:]]
            results[label] = ys
            results[f"{label}_cycles"] = cycles

        assert results["baseline"] == ref, (
            f"baseline: got {results['baseline']}, expected {ref}"
        )
        assert results["parallelism2"] == ref, (
            f"parallelism=2: got {results['parallelism2']}, expected {ref}"
        )
        # parallelism=2 with out_size=4 -> 2 groups, so cycle count grows
        # by 2x compared to baseline (in_size cycles per group).
        assert results["parallelism2_cycles"] >= results["baseline_cycles"], (
            f"parallelism=2 cycles {results['parallelism2_cycles']} < "
            f"baseline {results['baseline_cycles']} (should be >=)"
        )


def test_mac_sharing_storage_matches_accumulator_at_done():
    """--mac-sharing: at done, the storage register values should equal
    the (now-final) accumulator values."""
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import registry
    weights = [[1, -1, 1], [-1, 0, 1], [1, 1, -1]]
    in_size = len(weights[0])
    out_size = len(weights)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = _bitnet_fixture(td, weights)
        g = registry.get("bitnet_linear")().parse(
            sf, top="bn_ms", sequential=True, mac_sharing=True,
        )
        v_text = emit_module(g)
        y_widths = [s.width for s in g.outputs if s.name.startswith("y")]
        x = [2, -1, 3]
        ref = _ref_outputs(weights, x)

        x_drive = "\n".join(f"    x{i} = {v};" for i, v in enumerate(x))
        y_decls = "\n".join(
            f"  wire signed [{w-1}:0] y{j};"
            for j, w in enumerate(y_widths)
        )
        y_assoc = ", ".join(f".y{j}(y{j})" for j in range(out_size))
        y_args = ", ".join(f"y{j}" for j in range(out_size))
        y_fmt = " ".join("%0d" for _ in range(out_size))
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [7:0] {", ".join(f"x{i}" for i in range(in_size))};
  wire done;
{y_decls}
  bn_ms dut(.clk(clk), .rst(rst), .start(start),
            {", ".join(f".x{i}(x{i})" for i in range(in_size))},
            .done(done), {y_assoc});
  integer cycles;
  initial begin
    rst = 1; #20 rst = 0;
{x_drive}
    @(negedge clk);
    start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    $display("DONE cycles=%0d {y_fmt}", cycles, {y_args});
    $finish;
  end
endmodule
"""
        out = _compile_and_run(v_text, "bn_ms", tb, td)
        done_line = next(l for l in out.splitlines() if l.startswith("DONE"))
        ys = [int(t) for t in done_line.split()[2:]]
        # mac_sharing reads the y outputs from the storage registers, so
        # if they match the reference then storage successfully captured
        # the accumulator values.
        assert ys == ref, (
            f"mac_sharing: got {ys}, expected {ref} (storage register "
            f"capture didn't pick up the final accumulator values)"
        )
