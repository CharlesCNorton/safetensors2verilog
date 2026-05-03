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
