"""Tests for the threshold-logic frontend and the Verilog backend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from safetensors2verilog.core import registry
from safetensors2verilog.verilog import emit_module


def _build_and_or(path: Path) -> None:
    """Two gates: AND(a, b) and OR(a, b). Both use ternary weights."""
    sr = {"0": "#0", "1": "#1", "2": "$a", "3": "$b", "4": "and_ab", "5": "or_ab"}
    save_file(
        {
            "and_ab.weight": torch.tensor([1, 1], dtype=torch.int8),
            "and_ab.bias":   torch.tensor([-2], dtype=torch.int8),
            "and_ab.inputs": torch.tensor([2, 3], dtype=torch.int64),
            "or_ab.weight":  torch.tensor([1, 1], dtype=torch.int8),
            "or_ab.bias":    torch.tensor([-1], dtype=torch.int8),
            "or_ab.inputs":  torch.tensor([2, 3], dtype=torch.int64),
        },
        str(path),
        metadata={"signal_registry": json.dumps(sr)},
    )


def test_frontend_parses_simple_network():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        _build_and_or(p)
        graph = registry.get("threshold_logic")().parse(p, top="t")
        assert graph.top == "t"
        assert sorted(graph.inputs) == ["$a", "$b"]
        assert sorted(graph.outputs) == ["and_ab", "or_ab"]
        assert len(graph.gates) == 2


def test_backend_emits_synthesizable_verilog():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        _build_and_or(p)
        graph = registry.get("threshold_logic")().parse(p, top="and_or")
        text = emit_module(graph)
        # Critical structural properties
        assert "module and_or" in text
        assert "endmodule" in text
        assert "input wire _a" in text
        assert "input wire _b" in text
        assert "output wire and_ab" in text
        assert "output wire or_ab" in text
        # AND: H(a + b - 2) -> (a + b) >= 2
        assert "((_a + _b) >= 2)" in text
        # OR: H(a + b - 1) -> (a + b) >= 1
        assert "((_a + _b) >= 1)" in text


def test_topological_order_required():
    """The backend must reject graphs whose gates reference undeclared signals."""
    from safetensors2verilog.core import Gate, GateGraph

    bad = GateGraph(
        inputs=["$a"],
        outputs=["g2"],
        gates=[
            Gate(name="g2", pos=["g1"], bias=0),       # forward reference
            Gate(name="g1", pos=["$a"], bias=-1),
        ],
        top="t",
    )
    with pytest.raises(ValueError, match="topologically sorted"):
        emit_module(bad)


def test_constants_resolve_to_literals():
    """`#0` / `#1` should become Verilog literals, not wires."""
    from safetensors2verilog.core import Gate, GateGraph

    g = GateGraph(
        inputs=["$x"],
        outputs=["k"],
        gates=[Gate(name="k", pos=["$x", "#1"], neg=["#0"], bias=-1)],
        top="t",
    )
    text = emit_module(g)
    assert "1'b1" in text
    assert "1'b0" in text


def test_non_ternary_weight_expands_to_duplicates():
    """A weight of +2 should produce two `+1` contributions, not a multiplier."""
    sr = {"0": "#0", "1": "#1", "2": "$x", "3": "g"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "g.weight": torch.tensor([2], dtype=torch.int8),
                "g.bias":   torch.tensor([-1], dtype=torch.int8),
                "g.inputs": torch.tensor([2], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        graph = registry.get("threshold_logic")().parse(p)
        # Weight 2 -> two copies of the same signal, weight 1 each
        gate = graph.gates[0]
        assert gate.pos == ["$x", "$x"]
        text = emit_module(graph)
        assert "(_x + _x)" in text or "(_x+_x)" in text.replace(" ", "")
