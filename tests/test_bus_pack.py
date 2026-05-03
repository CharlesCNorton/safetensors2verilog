"""Tests for --pack-buses port emission."""
from __future__ import annotations

from safetensors2verilog.core import Gate, GateGraph, Signal
from safetensors2verilog.verilog import emit_module


def _bus_graph() -> GateGraph:
    """8 single-bit external inputs $a[0..7] feeding one threshold gate."""
    inputs = [Signal(name=f"$a[{i}]", width=1) for i in range(8)]
    return GateGraph(
        inputs=inputs,
        outputs=[Signal("y")],
        gates=[Gate(
            name="y", kind="threshold",
            inputs=[f"$a[{i}]" for i in range(8)],
            attrs={"weights": [1] * 8, "bias": -4},
            output_width=1,
        )],
        top="b",
    )


def test_bus_packing_emits_packed_input():
    g = _bus_graph()
    text = emit_module(g, pack_buses=True)
    assert "input wire [7:0] a" in text
    # Only one input port now, not 8 flat scalars.
    assert text.count("\n  input wire") == 1
    # Body uses bit-selects on 'a'
    assert "a[0]" in text and "a[7]" in text


def test_bus_packing_off_by_default():
    g = _bus_graph()
    text = emit_module(g, pack_buses=False)
    # Without packing, every $a[i] becomes its own scalar port.
    assert text.count("\n  input wire") == 8


def test_bus_packing_skips_partial_bus():
    """Buses with non-contiguous indices stay flat."""
    inputs = [Signal(name="$a[0]"), Signal(name="$a[2]"),
              Signal(name="$a[3]")]
    g = GateGraph(
        inputs=inputs,
        outputs=[Signal("y")],
        gates=[Gate(
            name="y", kind="threshold",
            inputs=["$a[0]", "$a[2]", "$a[3]"],
            attrs={"weights": [1, 1, 1], "bias": -2},
            output_width=1,
        )],
        top="p",
    )
    text = emit_module(g, pack_buses=True)
    # Indices 0, 2, 3 — gap at 1, so packing must be declined.
    assert text.count("\n  input wire") == 3
