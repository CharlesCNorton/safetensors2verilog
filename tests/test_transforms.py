"""Tests for graph-level transforms (pipeline insertion)."""
from __future__ import annotations

from safetensors2verilog.analysis import gate_depths
from safetensors2verilog.core import Gate, GateGraph, Signal
from safetensors2verilog.transforms import pipeline_at_depths, pipeline_every


def _chain(n: int) -> GateGraph:
    gates = [Gate(name="g0", kind="and", inputs=["x", "y"], output_width=1)]
    for i in range(1, n):
        gates.append(Gate(name=f"g{i}", kind="and",
                          inputs=[f"g{i-1}", "y"], output_width=1))
    return GateGraph(
        inputs=[Signal("x"), Signal("y")],
        outputs=[Signal(f"g{n-1}")],
        gates=gates,
        top="chain",
    )


def test_pipeline_every_inserts_registers():
    g = _chain(8)
    g2 = pipeline_every(g, period=4)
    register_gates = [x for x in g2.gates if x.kind == "register"]
    # depths 4 and 8 → 2 registers inserted
    assert len(register_gates) == 2


def test_pipeline_adds_clk_input():
    g = _chain(4)
    g2 = pipeline_every(g, period=2)
    assert any(s.name == "clk" for s in g2.inputs)


def test_pipeline_rewrites_consumers():
    """A gate downstream of a pipelined gate must read from the register."""
    g = _chain(4)
    g2 = pipeline_every(g, period=2)
    # find the register inserted at depth 2 (named g1__pipe2)
    reg_names = {x.name for x in g2.gates if x.kind == "register"}
    assert "g1__pipe2" in reg_names
    # g2 (depth 3) should now read from the register, not directly from g1
    g2_gate = next(x for x in g2.gates if x.name == "g2")
    assert "g1__pipe2" in g2_gate.inputs
    assert "g1" not in g2_gate.inputs


def test_pipeline_rewrites_outputs():
    """If an output port references a now-pipelined gate, point at register."""
    g = _chain(4)
    g2 = pipeline_every(g, period=4)
    # g3 was the output and is at depth 4 → pipelined. Output should
    # point at g3__pipe4 instead.
    output_names = {s.name for s in g2.outputs}
    assert "g3__pipe4" in output_names


def test_pipeline_at_depths_no_op_for_empty_cuts():
    g = _chain(4)
    g2 = pipeline_at_depths(g, [])
    assert len(g2.gates) == len(g.gates)


def test_pipeline_critical_path_shrinks():
    g = _chain(20)
    pre = max(gate_depths(g).values())
    g2 = pipeline_every(g, period=5)
    post = max(
        gate_depths(g2).values()
    )
    assert post < pre
    # Max combinational depth between cuts is `period` + 1 (the register
    # itself counts as one level in the analysis convention).
    assert post <= 6
