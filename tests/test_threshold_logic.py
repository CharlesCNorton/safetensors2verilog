"""Tests for the threshold-logic frontend and the threshold gate lowering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from safetensors2verilog.core import Gate, GateGraph, Signal, registry
from safetensors2verilog.verilog import emit_module


def _build_and_or(path: Path) -> None:
    """Two single-layer threshold gates: AND(a,b) and OR(a,b)."""
    sr = {
        "0": "#0", "1": "#1", "2": "$a", "3": "$b",
        "4": "and_ab", "5": "or_ab",
    }
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
        assert sorted(s.name for s in graph.inputs) == ["$a", "$b"]
        assert sorted(s.name for s in graph.outputs) == ["and_ab", "or_ab"]
        assert len(graph.gates) == 2
        for g in graph.gates:
            assert g.kind == "threshold"
            assert g.attrs["weights"] == [1, 1]


def test_backend_emits_synthesizable_verilog():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        _build_and_or(p)
        graph = registry.get("threshold_logic")().parse(p, top="and_or")
        text = emit_module(graph)
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
        assert "`default_nettype none" in text


def test_topological_order_required():
    bad = GateGraph(
        inputs=[Signal("$a")],
        outputs=[Signal("g2")],
        gates=[
            Gate(name="g2", kind="threshold", inputs=["g1"],
                 attrs={"weights": [1], "bias": 0}),
            Gate(name="g1", kind="threshold", inputs=["$a"],
                 attrs={"weights": [1], "bias": -1}),
        ],
        top="t",
    )
    with pytest.raises(ValueError, match="topologically sorted"):
        emit_module(bad)


def test_constants_resolve_to_literals():
    g = GateGraph(
        inputs=[Signal("$x")],
        outputs=[Signal("k")],
        gates=[
            Gate(name="k", kind="threshold",
                 inputs=["$x", "#1", "#0"],
                 attrs={"weights": [1, 1, -1], "bias": -1}),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "1'b1" in text
    assert "1'b0" in text


def test_non_ternary_weight_is_not_expanded_to_duplicates():
    """Non-ternary weights should appear as `k*x`, not `x + x + ...`."""
    sr = {"0": "#0", "1": "#1", "2": "$x", "3": "g"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "g.weight": torch.tensor([5], dtype=torch.int8),
                "g.bias":   torch.tensor([-1], dtype=torch.int8),
                "g.inputs": torch.tensor([2], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        graph = registry.get("threshold_logic")().parse(p)
        text = emit_module(graph)
        assert "5*_x" in text
        assert "_x + _x + _x" not in text


def test_strict_mode_errors_on_unresolved_signal():
    """With strict=True, unresolved signal IDs raise instead of becoming anon ports."""
    sr = {"0": "#0", "1": "#1"}  # IDs 2, 3 missing
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "g.weight": torch.tensor([1, 1], dtype=torch.int8),
                "g.bias":   torch.tensor([-2], dtype=torch.int8),
                "g.inputs": torch.tensor([2, 3], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        with pytest.raises(ValueError, match="unresolved signal id"):
            registry.get("threshold_logic")().parse(p, strict=True)


def test_strict_mode_errors_on_stale_routing():
    """With strict=True, .inputs length mismatch raises instead of promoting to anon ports."""
    sr = {"0": "#0", "1": "#1", "2": "$a", "3": "$b"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "g.weight": torch.tensor([1, 1], dtype=torch.int8),
                "g.bias":   torch.tensor([-2], dtype=torch.int8),
                "g.inputs": torch.tensor([2], dtype=torch.int64),  # length 1, weight length 2
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        with pytest.raises(ValueError, match="stale routing"):
            registry.get("threshold_logic")().parse(p, strict=True)


def test_qat_float_weights_accepted():
    """Float weights with tiny rounding error should not raise."""
    sr = {"0": "#0", "1": "#1", "2": "$a", "3": "$b", "4": "g"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "g.weight": torch.tensor([1.0 + 1e-8, -1.0 - 1e-9], dtype=torch.float32),
                "g.bias":   torch.tensor([0.0], dtype=torch.float32),
                "g.inputs": torch.tensor([2, 3], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        graph = registry.get("threshold_logic")().parse(p)
        assert graph.gates[0].attrs["weights"] == [1, -1]


def test_packed_multi_gate_unpacks():
    """A packed [N, K] weight + length-N bias unpacks into N sub-gates."""
    sr = {"0": "#0", "1": "#1", "2": "$a", "3": "$b"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "pack.weight": torch.tensor([[1, 1], [-1, -1]], dtype=torch.int8),
                "pack.bias":   torch.tensor([-2, 1], dtype=torch.int8),
                "pack.inputs": torch.tensor([2, 3, 2, 3], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        graph = registry.get("threshold_logic")().parse(p)
        names = sorted(g.name for g in graph.gates)
        assert names == ["pack.bit0", "pack.bit1"]


def test_skip_memory_drops_memory_gates():
    sr = {"0": "#0", "1": "#1", "2": "$addr"}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.safetensors"
        save_file(
            {
                "memory.cell.weight": torch.tensor([1], dtype=torch.int8),
                "memory.cell.bias":   torch.tensor([-1], dtype=torch.int8),
                "memory.cell.inputs": torch.tensor([2], dtype=torch.int64),
                "logic.gate.weight": torch.tensor([1, 1], dtype=torch.int8),
                "logic.gate.bias":   torch.tensor([-1], dtype=torch.int8),
                "logic.gate.inputs": torch.tensor([2, 2], dtype=torch.int64),
            },
            str(p),
            metadata={"signal_registry": json.dumps(sr)},
        )
        graph_full = registry.get("threshold_logic")().parse(p)
        graph_skip = registry.get("threshold_logic")().parse(p, skip_memory=True)
        names_full = {g.name for g in graph_full.gates}
        names_skip = {g.name for g in graph_skip.gates}
        assert "memory.cell" in names_full
        assert "memory.cell" not in names_skip
        assert "logic.gate" in names_skip


def test_cycle_detection_raises():
    """Manually-constructed cycle through topo sort is rejected."""
    # Two gates referencing each other's outputs -> cycle.
    bad = GateGraph(
        inputs=[],
        outputs=[Signal("a")],
        gates=[
            Gate(name="a", kind="threshold", inputs=["b"],
                 attrs={"weights": [1], "bias": 0}),
            Gate(name="b", kind="threshold", inputs=["a"],
                 attrs={"weights": [1], "bias": 0}),
        ],
        top="t",
    )
    with pytest.raises(ValueError, match="topologically sorted"):
        emit_module(bad)
