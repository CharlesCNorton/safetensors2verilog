"""Tests for static-analysis utilities."""
from __future__ import annotations

from safetensors2verilog.analysis import (
    critical_path,
    fanout,
    gate_depths,
    summary,
)
from safetensors2verilog.core import Gate, GateGraph, Signal


def _chain(n: int) -> GateGraph:
    """Linear AND chain: g0 -> g1 -> ... -> g(n-1), depth = n."""
    gates = [
        Gate(name="g0", kind="and", inputs=["x", "y"], output_width=1)
    ]
    for i in range(1, n):
        gates.append(
            Gate(name=f"g{i}", kind="and", inputs=[f"g{i-1}", "y"],
                 output_width=1)
        )
    return GateGraph(
        inputs=[Signal("x"), Signal("y")],
        outputs=[Signal(f"g{n-1}")],
        gates=gates,
        top="chain",
    )


def test_depth_linear_chain():
    graph = _chain(5)
    d = gate_depths(graph)
    assert d["g0"] == 1
    assert d["g4"] == 5


def test_critical_path_chain():
    graph = _chain(4)
    cp = critical_path(graph)
    assert cp == ["g0", "g1", "g2", "g3"]


def test_register_resets_depth():
    """A register output starts a fresh combinational chain."""
    gates = [
        Gate(name="a", kind="and", inputs=["x", "y"]),
        Gate(name="r", kind="register", inputs=["a"],
             attrs={"clk": "clk"}),
        Gate(name="b", kind="and", inputs=["r", "y"]),
        Gate(name="c", kind="and", inputs=["b", "y"]),
    ]
    graph = GateGraph(
        inputs=[Signal("x"), Signal("y"), Signal("clk")],
        outputs=[Signal("c")],
        gates=gates,
        top="t",
    )
    d = gate_depths(graph)
    assert d["a"] == 1
    assert d["r"] == 1   # register breaks the chain
    assert d["b"] == 2
    assert d["c"] == 3


def test_fanout_counts():
    graph = _chain(3)
    f = fanout(graph)
    # y is consumed by all 3 gates
    assert f["y"] == 3
    # x is consumed only by g0
    assert f["x"] == 1
    # g0 is consumed by g1
    assert f["g0"] == 1


def test_summary_structure():
    graph = _chain(4)
    s = summary(graph)
    assert s["gates"] == 4
    assert s["max_depth"] == 4
    assert s["kinds"] == {"and": 4}
    assert s["fanout_max"]["signal"] == "y"
    assert len(s["critical_path"]) == 4
