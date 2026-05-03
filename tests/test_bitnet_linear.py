"""Tests for the bitnet_linear frontend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from safetensors2verilog.core import registry
from safetensors2verilog.verilog import emit_module


def _save_one_layer(path: Path, weight, bias=None, prefix="layers"):
    tensors = {f"{prefix}.0.weight": torch.tensor(weight, dtype=torch.float32)}
    if bias is not None:
        tensors[f"{prefix}.0.bias"] = torch.tensor(bias, dtype=torch.int32)
    save_file(tensors, str(path))


def test_single_layer_round_trip():
    """A 2-input -> 3-output ternary linear layer round-trips through the IR."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        # y[0] =  x0 + x1
        # y[1] =  x0 - x1
        # y[2] =  -x1
        _save_one_layer(p, weight=[[1, 1], [1, -1], [0, -1]])
        graph = registry.get("bitnet_linear")().parse(p, top="m")
        assert sorted(s.name for s in graph.inputs) == ["x0", "x1"]
        assert sorted(s.name for s in graph.outputs) == ["L0.y0", "L0.y1", "L0.y2"]
        text = emit_module(graph)
        assert "module m" in text
        assert "$signed(" in text
        assert "input wire signed [7:0] x0" in text
        assert "output wire signed [" in text


def test_multi_layer_chains():
    """Chained layers connect output of layer N to input of layer N+1."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        save_file(
            {
                "layers.0.weight": torch.tensor([[1, 1], [-1, 1]], dtype=torch.int8),
                "layers.1.weight": torch.tensor([[1, -1]], dtype=torch.int8),
            },
            str(p),
        )
        graph = registry.get("bitnet_linear")().parse(p, top="m")
        assert sorted(s.name for s in graph.outputs) == ["L1.y0"]
        # The L1 linear gate's inputs should be L0.y0 and L0.y1.
        l1_linear = next(
            g for g in graph.gates if g.name.startswith("L1.") and g.kind == "linear"
        )
        assert set(l1_linear.inputs) == {"L0.y0", "L0.y1"}


def test_bias_is_emitted():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 0]], bias=[5])
        graph = registry.get("bitnet_linear")().parse(p, top="m")
        # Bias rides on the linear gate's attrs.
        linear_gates = [g for g in graph.gates if g.kind == "linear"]
        assert linear_gates, "expected at least one linear gate"
        assert any(g.attrs.get("bias") == 5 for g in linear_gates), (
            "bias=5 should appear in some linear gate's attrs"
        )


def test_non_ternary_weight_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 2]])
        with pytest.raises(ValueError, match="not ternary"):
            registry.get("bitnet_linear")().parse(p)


def test_emits_synthesizable_verilog():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, -1, 0, 1]])
        graph = registry.get("bitnet_linear")().parse(p, top="bn1", activation_bits=4)
        text = emit_module(graph)
        for i in range(4):
            assert f"input wire signed [3:0] x{i}" in text
        assert "output wire signed [" in text


def test_options_surfaced():
    cls = registry.get("bitnet_linear")
    opt_names = {o.name for o in cls.options()}
    assert "activation-bits" in opt_names
    assert "layer-prefix" in opt_names
    assert "output-clamp" in opt_names
    assert "pipeline" in opt_names


# ---- New options: clamp + pipeline -----------------------------------------


def test_output_clamp_emits_clamp_gate():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 1], [1, -1]])
        graph = registry.get("bitnet_linear")().parse(
            p, top="m", output_clamp="-8,7"
        )
        clamp_gates = [g for g in graph.gates if g.kind == "clamp"]
        assert len(clamp_gates) == 2
        for g in clamp_gates:
            assert g.attrs == {"lo": -8, "hi": 7}
        text = emit_module(graph)
        # Verilog should reference the saturation bounds
        assert "> 7" in text and "< -8" in text


def test_output_clamp_argument_validation():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1]])
        with pytest.raises(ValueError, match="LO,HI"):
            registry.get("bitnet_linear")().parse(p, output_clamp="not-a-pair")
        with pytest.raises(ValueError, match="lo=10 > hi=-10"):
            registry.get("bitnet_linear")().parse(p, output_clamp="10,-10")


def test_pipeline_emits_register_gates_and_clk_port():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 1], [1, -1]])
        graph = registry.get("bitnet_linear")().parse(p, top="m", pipeline=True)
        register_gates = [g for g in graph.gates if g.kind == "register"]
        assert len(register_gates) == 2
        text = emit_module(graph)
        assert "input wire clk" in text
        assert "always @(posedge clk)" in text
        assert "output reg signed" in text


def test_pipeline_with_clamp_chains_clamp_then_register():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1]])
        graph = registry.get("bitnet_linear")().parse(
            p, top="m", pipeline=True, output_clamp="-4,3"
        )
        # Find the register; its input must be the clamp gate.
        register = next(g for g in graph.gates if g.kind == "register")
        clamp_name = register.inputs[0]
        assert clamp_name.endswith(".clamped")
        clamp_gate = next(g for g in graph.gates if g.name == clamp_name)
        assert clamp_gate.kind == "clamp"


# ---- Numeric round-trip (opt-in: needs iverilog) ---------------------------


def _have_iverilog() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def test_sequential_and_pipeline_mutually_exclusive():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, -1]])
        with pytest.raises(ValueError, match="mutually exclusive"):
            registry.get("bitnet_linear")().parse(
                p, sequential=True, pipeline=True
            )


def test_sequential_emits_fsm_and_rom_gates():
    """Sequential mode produces an FSM (state register, counter, layer_idx)
    and per-output ROM gates."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, -1, 0], [-1, 1, 1]], bias=[3, -2])
        graph = registry.get("bitnet_linear")().parse(
            p, top="seq", activation_bits=4, sequential=True
        )
        kinds = [g.kind for g in graph.gates]
        assert "register" in kinds, "sequential mode must use register kind"
        assert "rom" in kinds, "sequential mode must use rom kind for weights"
        assert "eq" in kinds, "FSM uses eq for state comparisons"
        names = {g.name for g in graph.gates}
        # State machine and counter are wired up
        assert "state.curr" in names
        assert "counter.curr" in names
        # Per-output ROMs and accumulators
        assert "L0.rom0" in names and "L0.rom1" in names
        assert "L0.acc0.curr" in names and "L0.acc1.curr" in names
        # Module ports include start/done
        assert any(s.name == "start" for s in graph.inputs)
        assert any(s.name == "done" for s in graph.outputs)


def test_sequential_emits_synthesizable_verilog():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, -1, 0], [-1, 1, 1]], bias=[3, -2])
        graph = registry.get("bitnet_linear")().parse(
            p, top="seq", activation_bits=4, sequential=True
        )
        text = emit_module(graph)
        assert "input wire clk" in text
        assert "input wire rst" in text
        assert "input wire start" in text
        assert "output wire done" in text
        assert "always @(posedge clk or posedge rst)" in text


def _run_sequential_simulation(td: Path, vpath: Path, in_size: int,
                               out_widths: dict, cases: list[tuple],
                               module_name: str = "seq"):
    """Drive a sequential bitnet module in iverilog. Returns simulator output."""
    out_names = sorted(out_widths)
    x_decl = ", ".join(f"x{i}" for i in range(in_size))
    y_decl = "\n".join(
        f"  wire signed [{out_widths[n]-1}:0] {n};" for n in out_names
    )
    x_inst = ", ".join(f".x{i}(x{i})" for i in range(in_size))
    y_inst = ", ".join(f".{n}({n})" for n in out_names)
    fmts = " ".join(f"{n}=%0d" for n in out_names)
    args = ", ".join(out_names)

    task_inputs = ", ".join(f"input signed [3:0] a{i}" for i in range(in_size))
    task_assigns = "; ".join(f"x{i} = a{i}" for i in range(in_size)) + ";"
    case_calls = []
    for case in cases:
        ass = ", ".join(str(v) for v in case)
        case_calls.append(f"    one_inference({ass});")

    tb = f"""\
`timescale 1ns / 1ps

module tb;
  reg clk = 0;
  reg rst = 1;
  reg start = 0;
  reg signed [3:0] {x_decl};
  wire done;
{y_decl}

  always #5 clk = ~clk;

  {module_name} dut (
    .clk(clk), .rst(rst), .start(start),
    {x_inst},
    .done(done), {y_inst}
  );

  integer cycles;
  task one_inference({task_inputs});
    begin
      cycles = 0;
      {task_assigns}
      @(posedge clk); start = 1;
      @(posedge clk); start = 0;
      while (!done) begin
        @(posedge clk);
        cycles = cycles + 1;
        if (cycles > 100) begin $display("TIMEOUT"); $finish; end
      end
      $display("RESULT in=({", ".join("%0d" for _ in range(in_size))}) {fmts}",
               {", ".join(f"a{i}" for i in range(in_size))}, {args});
    end
  endtask

  initial begin
    rst = 1;
    #20 rst = 0;
{chr(10).join(case_calls)}
    $finish;
  end
endmodule
"""
    tb_path = td / "tb.v"
    tb_path.write_text(tb)
    vvp = td / "tb.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(vpath), str(tb_path)],
        check=True,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True
    )
    return proc.stdout


@pytest.mark.skipif(not _have_iverilog(),
                    reason="iverilog/vvp not on PATH")
def test_sequential_single_layer_numeric_round_trip():
    """Single-layer ternary linear running through sequential FSM matches Python."""
    W = [[1, -1, 0], [-1, 1, 1]]
    B = [3, -2]
    cases = [(0, 0, 0), (1, 1, 1), (-1, 0, 1), (3, -2, 1), (-4, 4, -4)]

    def py_eval(x):
        return [
            sum(W[j][i] * x[i] for i in range(len(x))) + B[j]
            for j in range(len(W))
        ]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = td / "m.safetensors"
        save_file(
            {
                "layers.0.weight": torch.tensor(W, dtype=torch.int8),
                "layers.0.bias":   torch.tensor(B, dtype=torch.int32),
            },
            str(sf),
        )
        graph = registry.get("bitnet_linear")().parse(
            sf, top="seq", activation_bits=4, sequential=True
        )
        verilog = emit_module(graph)
        out_widths = {s.name: s.width for s in graph.outputs if s.name != "done"}
        vpath = td / "seq.v"
        vpath.write_text(verilog)
        output = _run_sequential_simulation(td, vpath, 3, out_widths, cases, "seq")

    for case in cases:
        line = next(l for l in output.splitlines() if l.startswith("RESULT")
                    and f"in=({', '.join(str(v) for v in case)})" in l)
        parts = dict(t.split("=") for t in line.split() if "=" in t)
        sim = [int(parts[f"y{i}"]) for i in range(2)]
        expected = py_eval(list(case))
        assert sim == expected, f"case {case}: sim={sim} expected={expected}"


@pytest.mark.skipif(not _have_iverilog(),
                    reason="iverilog/vvp not on PATH")
def test_sequential_multi_layer_numeric_round_trip():
    """Two-layer 3->2->1 ternary linear in sequential mode matches Python."""
    W1 = [[1, -1, 0], [-1, 1, 1]]
    B1 = [3, -2]
    W2 = [[1, -1]]
    B2 = [5]
    cases = [(0, 0, 0), (1, 1, 1), (-1, 0, 1), (3, -2, 1), (-4, 4, -4)]

    def py_eval(x):
        h = [
            sum(W1[j][i] * x[i] for i in range(len(x))) + B1[j]
            for j in range(len(W1))
        ]
        return [
            sum(W2[j][i] * h[i] for i in range(len(h))) + B2[j]
            for j in range(len(W2))
        ]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = td / "m.safetensors"
        save_file(
            {
                "layers.0.weight": torch.tensor(W1, dtype=torch.int8),
                "layers.0.bias":   torch.tensor(B1, dtype=torch.int32),
                "layers.1.weight": torch.tensor(W2, dtype=torch.int8),
                "layers.1.bias":   torch.tensor(B2, dtype=torch.int32),
            },
            str(sf),
        )
        graph = registry.get("bitnet_linear")().parse(
            sf, top="seq2", activation_bits=4, sequential=True
        )
        verilog = emit_module(graph)
        out_widths = {s.name: s.width for s in graph.outputs if s.name != "done"}
        vpath = td / "seq2.v"
        vpath.write_text(verilog)
        output = _run_sequential_simulation(td, vpath, 3, out_widths, cases, "seq2")

    for case in cases:
        line = next(l for l in output.splitlines() if l.startswith("RESULT")
                    and f"in=({', '.join(str(v) for v in case)})" in l)
        parts = dict(t.split("=") for t in line.split() if "=" in t)
        sim = [int(parts["y0"])]
        expected = py_eval(list(case))
        assert sim == expected, f"case {case}: sim={sim} expected={expected}"


@pytest.mark.skipif(not _have_iverilog(),
                    reason="iverilog/vvp not on PATH")
def test_numeric_round_trip_against_python():
    """Build a tiny ternary linear layer, simulate the emitted Verilog in
    iverilog, and verify outputs match Python's matmul on every test case."""
    W = [[1, 0, -1], [-1, 1, 1]]
    b = [3, -2]

    def py_eval(x):
        return [
            sum(W[j][i] * x[i] for i in range(len(x))) + b[j]
            for j in range(len(W))
        ]

    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "m.safetensors"
        save_file(
            {
                "layers.0.weight": torch.tensor(W, dtype=torch.int8),
                "layers.0.bias":   torch.tensor(b, dtype=torch.int32),
            },
            str(sf),
        )
        graph = registry.get("bitnet_linear")().parse(
            sf, top="bn", activation_bits=4
        )
        verilog = emit_module(graph)
        out_widths = {s.name: s.width for s in graph.outputs}

        vpath = Path(td) / "bn.v"
        tb_path = Path(td) / "tb.v"
        vpath.write_text(verilog)

        out_names = sorted(out_widths)
        tb_lines = ["`timescale 1ns/1ps", "module tb;",
                    "  reg signed [3:0] x0, x1, x2;"]
        for n in out_names:
            tb_lines.append(
                f"  wire signed [{out_widths[n]-1}:0] {n.replace('.', '_')};"
            )
        tb_lines.append("  bn dut (")
        port_lines = ["    .x0(x0), .x1(x1), .x2(x2)"]
        for n in out_names:
            v = n.replace(".", "_")
            port_lines.append(f"    , .{v}({v})")
        tb_lines.extend(port_lines)
        tb_lines.append("  );")

        cases = [(0, 0, 0), (1, 1, 1), (-1, 0, 1), (3, -2, 1), (-4, 4, -4)]
        tb_lines.append("  initial begin")
        for i, (a, bb, c) in enumerate(cases):
            tb_lines.append(f"    x0 = {a}; x1 = {bb}; x2 = {c}; #1;")
            fmts = " ".join("%0d" for _ in out_names)
            args = ", ".join(n.replace(".", "_") for n in out_names)
            tb_lines.append(
                f"    $display(\"{i} {fmts}\", {args});"
            )
        tb_lines.append("    $finish; end endmodule")
        tb_path.write_text("\n".join(tb_lines))

        vvp = Path(td) / "tb.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp), str(vpath), str(tb_path)],
            check=True,
        )
        proc = subprocess.run(
            ["vvp", str(vvp)], check=True, capture_output=True, text=True
        )

        sim_lines = [l for l in proc.stdout.splitlines() if l and l[0].isdigit()]
        for i, (a, bb, c) in enumerate(cases):
            line = next(l for l in sim_lines if l.startswith(f"{i} "))
            sim = [int(t) for t in line.split()[1:]]
            expected = py_eval([a, bb, c])
            assert sim == expected, (
                f"case {i} x={(a, bb, c)}: sim={sim} expected={expected}"
            )
