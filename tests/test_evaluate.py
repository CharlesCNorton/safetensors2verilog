"""Tests for the Python GateGraph evaluator."""

from __future__ import annotations

import pytest

from safetensors2verilog import evaluate_graph, step_graph
from safetensors2verilog.core import Gate, GateGraph, Signal


def test_evaluator_threshold_and():
    g = GateGraph(
        inputs=[Signal("$a"), Signal("$b")],
        outputs=[Signal("y")],
        gates=[Gate(name="y", kind="threshold", inputs=["$a", "$b"],
                    attrs={"weights": [1, 1], "bias": -2})],
        top="t",
    )
    for a, b, expected in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
        v = evaluate_graph(g, {"$a": a, "$b": b})
        assert v["y"] == expected, f"AND({a},{b}) -> {v['y']} expected {expected}"


def test_evaluator_linear():
    g = GateGraph(
        inputs=[Signal("a", width=8, signed=True),
                Signal("b", width=8, signed=True)],
        outputs=[Signal("y", width=16, signed=True)],
        gates=[Gate(name="y", kind="linear", inputs=["a", "b"],
                    attrs={"weights": [3, -2], "bias": 5},
                    output_width=16, output_signed=True)],
        top="t",
    )
    for a, b in [(0, 0), (1, 1), (-3, 4), (10, -5)]:
        v = evaluate_graph(g, {"a": a, "b": b})
        assert v["y"] == 3 * a - 2 * b + 5


def test_evaluator_arithmetic_and_bitwise():
    g = GateGraph(
        inputs=[Signal("a", width=4), Signal("b", width=4)],
        outputs=[Signal("s", width=5),
                 Signal("d", width=5, signed=True),
                 Signal("a_xor_b", width=4)],
        gates=[
            Gate(name="s", kind="add", inputs=["a", "b"], output_width=5),
            Gate(name="d", kind="sub", inputs=["a", "b"],
                 output_width=5, output_signed=True),
            Gate(name="a_xor_b", kind="xor", inputs=["a", "b"], output_width=4),
        ],
        top="t",
    )
    v = evaluate_graph(g, {"a": 5, "b": 3})
    assert v["s"] == 8
    assert v["d"] == 2
    assert v["a_xor_b"] == 5 ^ 3


def test_evaluator_mux():
    g = GateGraph(
        inputs=[Signal("sel", width=2),
                Signal("a", width=4), Signal("b", width=4),
                Signal("c", width=4), Signal("d", width=4)],
        outputs=[Signal("y", width=4)],
        gates=[Gate(name="y", kind="mux",
                    inputs=["sel", "a", "b", "c", "d"],
                    output_width=4)],
        top="t",
    )
    for sel, expected_key in [(0, "a"), (1, "b"), (2, "c"), (3, "d")]:
        v = evaluate_graph(g, {"sel": sel, "a": 1, "b": 2, "c": 3, "d": 4})
        assert v["y"] == {"a": 1, "b": 2, "c": 3, "d": 4}[expected_key]


def test_evaluator_relu_clamp():
    g = GateGraph(
        inputs=[Signal("x", width=8, signed=True)],
        outputs=[Signal("rl", width=8, signed=True),
                 Signal("cl", width=8, signed=True)],
        gates=[
            Gate(name="rl", kind="relu", inputs=["x"],
                 output_width=8, output_signed=True),
            Gate(name="cl", kind="clamp", inputs=["x"],
                 attrs={"lo": -16, "hi": 15},
                 output_width=8, output_signed=True),
        ],
        top="t",
    )
    for x, exp_rl, exp_cl in [(0, 0, 0), (10, 10, 10), (-5, 0, -5),
                              (50, 50, 15), (-100, 0, -16)]:
        v = evaluate_graph(g, {"x": x})
        assert v["rl"] == exp_rl
        assert v["cl"] == exp_cl


def test_evaluator_concat_slice():
    g = GateGraph(
        inputs=[Signal("a", width=4), Signal("b", width=4)],
        outputs=[Signal("cc", width=8), Signal("hi", width=4)],
        gates=[
            Gate(name="cc", kind="concat", inputs=["a", "b"], output_width=8),
            Gate(name="hi", kind="slice", inputs=["cc"],
                 attrs={"hi": 7, "lo": 4}, output_width=4),
        ],
        top="t",
    )
    v = evaluate_graph(g, {"a": 0xA, "b": 0x5})
    assert v["cc"] == 0xA5
    assert v["hi"] == 0xA


def test_evaluator_constant():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("c", width=8, signed=True)],
        gates=[Gate(name="c", kind="constant", attrs={"value": -7},
                    output_width=8, output_signed=True)],
        top="t",
    )
    v = evaluate_graph(g, {})
    assert v["c"] == -7


def test_evaluator_rom():
    g = GateGraph(
        inputs=[Signal("addr", width=2)],
        outputs=[Signal("d", width=8)],
        gates=[Gate(name="d", kind="rom", inputs=["addr"],
                    attrs={"init": [10, 20, 30, 40], "width": 8, "depth": 4},
                    output_width=8)],
        top="t",
    )
    for addr, expected in [(0, 10), (1, 20), (2, 30), (3, 40)]:
        v = evaluate_graph(g, {"addr": addr})
        assert v["d"] == expected


def test_evaluator_eq():
    g = GateGraph(
        inputs=[Signal("a", width=4), Signal("b", width=4)],
        outputs=[Signal("e")],
        gates=[Gate(name="e", kind="eq", inputs=["a", "b"], output_width=1)],
        top="t",
    )
    assert evaluate_graph(g, {"a": 5, "b": 5})["e"] == 1
    assert evaluate_graph(g, {"a": 5, "b": 4})["e"] == 0


def test_evaluator_step_graph_counter():
    """A counter built from add+register should advance one step per step_graph."""
    g = GateGraph(
        inputs=[],
        outputs=[Signal("counter", width=4)],
        gates=[
            Gate(name="one", kind="constant",
                 attrs={"value": 1}, output_width=4),
            Gate(name="counter_next", kind="add",
                 inputs=["counter", "one"], output_width=4),
            Gate(name="counter", kind="register",
                 inputs=["counter_next"],
                 attrs={"clk": "clk", "rst": "rst", "init": 0},
                 output_width=4),
        ],
        top="cnt",
    )
    state: dict[str, int] = {}
    for cycle in range(5):
        v = evaluate_graph(g, {}, register_state=state)
        assert v["counter"] == cycle
        state = step_graph(g, {}, state)


def test_evaluator_unknown_kind_raises():
    g = GateGraph(
        inputs=[Signal("x")],
        outputs=[Signal("y")],
        gates=[Gate(name="y", kind="not_a_real_kind",
                    inputs=["x"], output_width=1)],
        top="t",
    )
    with pytest.raises(NotImplementedError, match="not_a_real_kind"):
        evaluate_graph(g, {"x": 1})
