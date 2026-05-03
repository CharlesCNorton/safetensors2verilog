"""Property-based fuzz tests for the IR and backend.

Hypothesis is already a transitive dep of pytest-hypothesis on most
installs but we declare it as a dev dependency for clarity. These
tests probe the surface area where the backend or evaluator might
disagree on edge cases (extreme widths, negative constants, deeply
chained gates, etc.).
"""

from __future__ import annotations

import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from safetensors2verilog import evaluate_graph  # noqa: E402
from safetensors2verilog.core import Gate, GateGraph, Signal  # noqa: E402
from safetensors2verilog.verilog import emit_module  # noqa: E402

# ---- Strategies ------------------------------------------------------------


widths = st.integers(min_value=1, max_value=32)


@st.composite
def signed_int(draw, width):
    """An integer that fits in `width` signed bits."""
    bound = 1 << (width - 1)
    return draw(st.integers(min_value=-bound, max_value=bound - 1))


@st.composite
def unsigned_int(draw, width):
    bound = 1 << width
    return draw(st.integers(min_value=0, max_value=bound - 1))


# ---- Fuzz: constant gate matches its declared value -----------------------


@given(width=st.integers(min_value=1, max_value=32),
       value=st.integers(min_value=-(1 << 31), max_value=(1 << 31) - 1))
@settings(max_examples=50, deadline=2000)
def test_constant_gate_evaluates_correctly(width: int, value: int):
    g = GateGraph(
        inputs=[],
        outputs=[Signal("c", width=width, signed=True)],
        gates=[Gate(name="c", kind="constant", attrs={"value": value},
                    output_width=width, output_signed=True)],
        top="t",
    )
    v = evaluate_graph(g, {})
    # Two's-complement-mask the value to the declared width
    mask = (1 << width) - 1
    masked = value & mask
    if masked & (1 << (width - 1)):
        masked -= 1 << width
    assert v["c"] == masked
    # And the emitted module should still be parseable
    text = emit_module(g)
    assert "module t" in text


# ---- Fuzz: linear gate over random weights ---------------------------------


@given(
    width=st.integers(min_value=2, max_value=8),
    weights=st.lists(
        st.integers(min_value=-3, max_value=3),
        min_size=1, max_size=8,
    ),
    bias=st.integers(min_value=-32, max_value=32),
)
@settings(max_examples=40, deadline=2000)
def test_linear_gate_matches_python(width, weights, bias):
    n = len(weights)
    inputs_sigs = [Signal(f"x{i}", width=width, signed=True) for i in range(n)]
    out_width = width + n.bit_length() + 4  # generous
    g = GateGraph(
        inputs=inputs_sigs,
        outputs=[Signal("y", width=out_width, signed=True)],
        gates=[Gate(name="y", kind="linear",
                    inputs=[s.name for s in inputs_sigs],
                    attrs={"weights": list(weights), "bias": bias},
                    output_width=out_width, output_signed=True)],
        top="t",
    )
    text = emit_module(g)
    assert "module t" in text

    # Choose inputs that fit in `width`-bit signed range so the evaluator's
    # input masking doesn't change the value out from under us.
    bound = 1 << (width - 1)
    xs = {f"x{i}": ((i - n // 2) % bound) for i in range(n)}
    expected = sum(w * x for w, x in zip(weights, xs.values())) + bias
    v = evaluate_graph(g, xs)
    assert v["y"] == expected


# ---- Fuzz: counter pattern ------------------------------------------------


@given(width=st.integers(min_value=1, max_value=8),
       cycles=st.integers(min_value=1, max_value=20))
@settings(max_examples=20, deadline=2000)
def test_counter_increments_correctly(width: int, cycles: int):
    """Build a counter, step it N times, verify the value."""
    from safetensors2verilog import step_graph

    g = GateGraph(
        inputs=[],
        outputs=[Signal("counter", width=width)],
        gates=[
            Gate(name="one", kind="constant",
                 attrs={"value": 1}, output_width=width),
            Gate(name="counter_next", kind="add",
                 inputs=["counter", "one"], output_width=width),
            Gate(name="counter", kind="register",
                 inputs=["counter_next"],
                 attrs={"clk": "clk", "init": 0},
                 output_width=width),
        ],
        top="cnt",
    )
    state: dict[str, int] = {}
    for cycle in range(cycles):
        v = evaluate_graph(g, {}, register_state=state)
        # Counter wraps at 2^width
        assert v["counter"] == cycle % (1 << width)
        state = step_graph(g, {}, state)


# ---- Fuzz: random graph topo-sort acceptance ------------------------------


@given(
    n_gates=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=30, deadline=2000)
def test_chain_of_adds_emits_clean(n_gates: int, seed: int):
    """A long chain `a -> add(a, 1) -> add(prev, 1) -> ... -> y` parses."""
    width = 8
    gates: list[Gate] = [
        Gate(name="one", kind="constant",
             attrs={"value": 1}, output_width=width),
    ]
    prev = "x"
    for i in range(n_gates):
        next_name = "y" if i == n_gates - 1 else f"step{i}"
        gates.append(Gate(
            name=next_name, kind="add",
            inputs=[prev, "one"],
            output_width=width,
        ))
        prev = next_name
    g = GateGraph(
        inputs=[Signal("x", width=width)],
        outputs=[Signal("y", width=width)],
        gates=gates,
        top="t",
    )
    text = emit_module(g)
    assert "module t" in text
    v = evaluate_graph(g, {"x": seed % (1 << width)})
    expected = (seed % (1 << width)) + n_gates
    assert v["y"] == (expected & ((1 << width) - 1))
