"""Tests for the bitnet_linear frontend."""

from __future__ import annotations

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
        # Each output is signed and wider than 8 bits (accumulator grows)
        assert "output wire signed [" in text


def test_multi_layer_chains():
    """Chained layers connect output of layer N to input of layer N+1."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        save_file(
            {
                # 2 -> 2
                "layers.0.weight": torch.tensor([[1, 1], [-1, 1]], dtype=torch.int8),
                # 2 -> 1
                "layers.1.weight": torch.tensor([[1, -1]], dtype=torch.int8),
            },
            str(p),
        )
        graph = registry.get("bitnet_linear")().parse(p, top="m")
        assert sorted(s.name for s in graph.outputs) == ["L1.y0"]
        # Should have references to L0.y0 and L0.y1 inside L1's MAC.
        l1_inputs = set()
        for g in graph.gates:
            if g.name.startswith("L1."):
                l1_inputs.update(g.inputs)
        assert "L0.y0" in l1_inputs
        assert "L0.y1" in l1_inputs


def test_bias_is_emitted():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 0]], bias=[5])
        graph = registry.get("bitnet_linear")().parse(p, top="m")
        # The bias-init constant gate should carry value 5
        init_gates = [g for g in graph.gates if g.kind == "constant"
                      and g.attrs.get("value") == 5]
        assert init_gates, "expected a constant gate with bias value 5"


def test_non_ternary_weight_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, 2]])  # 2 is not ternary
        with pytest.raises(ValueError, match="not ternary"):
            registry.get("bitnet_linear")().parse(p)


def test_emits_synthesizable_verilog():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.safetensors"
        _save_one_layer(p, weight=[[1, -1, 0, 1]])
        graph = registry.get("bitnet_linear")().parse(p, top="bn1", activation_bits=4)
        text = emit_module(graph)
        # Each of the 4 inputs is a 4-bit signed port
        for i in range(4):
            assert f"input wire signed [3:0] x{i}" in text
        # Output is a wider signed signal
        assert "output wire signed [" in text


def test_options_surfaced():
    """The bitnet_linear frontend should self-describe its CLI options."""
    cls = registry.get("bitnet_linear")
    opt_names = {o.name for o in cls.options()}
    assert "activation-bits" in opt_names
    assert "layer-prefix" in opt_names
