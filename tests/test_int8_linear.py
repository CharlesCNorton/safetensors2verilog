"""Tests for the int8_linear frontend."""

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


def test_options_surfaced():
    cls = registry.get("int8_linear")
    opts = {o.name for o in cls.options()}
    assert {"activation-bits", "weight-bits", "layer-prefix",
            "output-clamp", "pipeline"}.issubset(opts)


def test_single_layer_int_weights():
    """Arbitrary integer weights round-trip through the IR as a 'linear' gate."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[3, -7, 12]])
        graph = registry.get("int8_linear")().parse(p, top="m")
        linear_gates = [g for g in graph.gates if g.kind == "linear"]
        assert len(linear_gates) == 1
        assert linear_gates[0].attrs["weights"] == [3, -7, 12]
        text = emit_module(graph)
        # Expression should contain the integer coefficients
        assert "3*" in text
        assert "12*" in text
        assert "-7*" in text or "+ -7*" in text or "+ -7 *" in text or "-$signed" in text


def test_bias_in_linear_attrs():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 2]], bias=[42])
        graph = registry.get("int8_linear")().parse(p, top="m")
        linear = next(g for g in graph.gates if g.kind == "linear")
        assert linear.attrs["bias"] == 42


def test_weight_bits_validation():
    """Weights outside [-2^(N-1), 2^(N-1)-1] raise."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[200]])  # outside int8 range
        with pytest.raises(ValueError, match="outside"):
            registry.get("int8_linear")().parse(p, weight_bits=8)


def test_weight_bits_widening_allows_larger_values():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[200]])
        # With weight_bits=16 the value 200 fits.
        graph = registry.get("int8_linear")().parse(p, weight_bits=16)
        assert any(g.kind == "linear" for g in graph.gates)


def test_non_integer_weight_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        save_file(
            {"layers.0.weight": torch.tensor([[1.5]], dtype=torch.float32)},
            str(p),
        )
        with pytest.raises(ValueError, match="not integer-valued"):
            registry.get("int8_linear")().parse(p)


def test_multi_layer_chains_through_int8():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        save_file(
            {
                "layers.0.weight": torch.tensor([[2, -3], [1, 4]], dtype=torch.int8),
                "layers.1.weight": torch.tensor([[5, -6]], dtype=torch.int8),
            },
            str(p),
        )
        graph = registry.get("int8_linear")().parse(p, top="m")
        assert sorted(s.name for s in graph.outputs) == ["L1.y0"]
        l1_linear = next(
            g for g in graph.gates if g.name.startswith("L1.") and g.kind == "linear"
        )
        assert set(l1_linear.inputs) == {"L0.y0", "L0.y1"}
        assert l1_linear.attrs["weights"] == [5, -6]


def test_pipeline_and_clamp_options():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[3, -2]])
        graph = registry.get("int8_linear")().parse(
            p, top="m", pipeline=True, output_clamp="-128,127"
        )
        kinds = [g.kind for g in graph.gates]
        assert "linear" in kinds
        assert "clamp" in kinds
        assert "register" in kinds
        text = emit_module(graph)
        assert "input wire clk" in text
        assert "always @(posedge clk)" in text


def _have_iverilog() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


@pytest.mark.skipif(not _have_iverilog(),
                    reason="iverilog/vvp not on PATH")
def test_numeric_round_trip_against_python():
    """Single int8 linear layer; sim output matches Python on every test case."""
    W = [[3, -2, 1], [-1, 5, 2]]
    b = [10, -4]

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
        graph = registry.get("int8_linear")().parse(
            sf, top="i8", activation_bits=4, weight_bits=4
        )
        verilog = emit_module(graph)
        out_widths = {s.name: s.width for s in graph.outputs}

        vpath = Path(td) / "i8.v"
        tb_path = Path(td) / "tb.v"
        vpath.write_text(verilog)

        out_names = sorted(out_widths)
        tb_lines = ["`timescale 1ns/1ps", "module tb;",
                    "  reg signed [3:0] x0, x1, x2;"]
        for n in out_names:
            tb_lines.append(
                f"  wire signed [{out_widths[n]-1}:0] {n.replace('.', '_')};"
            )
        tb_lines.append("  i8 dut (")
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

        sim_lines = [
            l for l in proc.stdout.splitlines() if l and l.split()[0].isdigit()
        ]
        for i, (a, bb, c) in enumerate(cases):
            line = next(l for l in sim_lines if l.startswith(f"{i} "))
            sim = [int(t) for t in line.split()[1:]]
            expected = py_eval([a, bb, c])
            assert sim == expected, (
                f"case {i} x={(a, bb, c)}: sim={sim} expected={expected}"
            )
