"""Tests for the formal-equivalence harness."""
from __future__ import annotations

import shutil

import pytest

from safetensors2verilog.core import Gate, GateGraph, Signal
from safetensors2verilog.equivalence import (
    emit_sby_equiv,
    emit_self_checking_tb,
    evaluate_python,
)


def _and_graph() -> GateGraph:
    """Threshold AND: out = (a + b - 2 >= 0); fires only when both inputs are 1."""
    return GateGraph(
        inputs=[Signal("a"), Signal("b")],
        outputs=[Signal("y")],
        gates=[
            Gate(name="y", kind="threshold", inputs=["a", "b"],
                 attrs={"weights": [1, 1], "bias": -2},
                 output_width=1),
        ],
        top="t",
    )


def test_evaluate_python_truth_table():
    g = _and_graph()
    assert evaluate_python(g, {"a": 0, "b": 0})["y"] == 0
    assert evaluate_python(g, {"a": 0, "b": 1})["y"] == 0
    assert evaluate_python(g, {"a": 1, "b": 0})["y"] == 0
    assert evaluate_python(g, {"a": 1, "b": 1})["y"] == 1


def test_emit_self_checking_tb_exhaustive():
    tb = emit_self_checking_tb(_and_graph(), dut_module="t")
    # 2 inputs -> 4 cases, exhaustive
    assert tb.count("PASS") == 1
    assert "$display" in tb
    assert "1'b1" in tb
    # Should drive every combination
    assert "a = 1'b0; b = 1'b0" in tb
    assert "a = 1'b1; b = 1'b1" in tb


def test_emit_sby_equiv_template():
    text = emit_sby_equiv("ref.v", "tgt.v", top="rc8", depth=20)
    assert "[options]" in text
    assert "mode prove" in text
    assert "depth 20" in text
    assert "ref.v" in text and "tgt.v" in text
    assert "equiv_make gold gate equiv" in text


def test_evaluate_unknown_input_raises():
    g = _and_graph()
    with pytest.raises(KeyError):
        evaluate_python(g, {"a": 0})  # missing b


def test_evaluate_unsupported_kind_raises():
    g = GateGraph(
        inputs=[Signal("a"), Signal("b")],
        outputs=[Signal("y")],
        gates=[Gate(name="y", kind="and", inputs=["a", "b"])],
        top="t",
    )
    with pytest.raises(ValueError, match="threshold gates only"):
        evaluate_python(g, {"a": 1, "b": 1})


@pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("vvp") is None,
    reason="iverilog/vvp not on PATH",
)
def test_run_iverilog_check_live():
    """End-to-end: build TB, compile, run, parse pass."""
    from safetensors2verilog.equivalence import run_iverilog_check
    from safetensors2verilog.verilog import emit_module
    g = _and_graph()
    v = emit_module(g)
    res = run_iverilog_check(g, v, dut_module="t")
    assert res["passed"]
    assert res["cases"] == 4
    assert res["fails"] == 0
