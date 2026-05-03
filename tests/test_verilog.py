"""Tests for the Verilog backend: kind dispatch, multibit, sanitization, validation."""

from __future__ import annotations

import pytest

from safetensors2verilog.core import Gate, GateGraph, Signal
from safetensors2verilog.verilog import (
    _sanitize,
    _validate_module_name,
    emit_bram_template,
    emit_module,
    lowering,
    registered_kinds,
)


# ---- Module name validation -------------------------------------------------


def test_valid_module_name_accepted():
    _validate_module_name("foo_bar")
    _validate_module_name("_underscore")
    _validate_module_name("a")


def test_invalid_module_name_rejected():
    for bad in ("123start", "has-dash", "has space", "", "with.dot"):
        with pytest.raises(ValueError):
            _validate_module_name(bad)


def test_verilog_keyword_rejected_as_module_name():
    with pytest.raises(ValueError, match="keyword"):
        _validate_module_name("module")
    with pytest.raises(ValueError, match="keyword"):
        _validate_module_name("wire")


def test_emit_module_validates_top_name():
    g = GateGraph(inputs=[], outputs=[], gates=[], top="123bad")
    with pytest.raises(ValueError):
        emit_module(g)


# ---- Sanitization -----------------------------------------------------------


def test_sanitize_replaces_illegal_chars():
    assert _sanitize("$a[0]") == "_a_0_"
    assert _sanitize("foo.bar.baz") == "foo_bar_baz"
    assert _sanitize("123abc").startswith("_")
    assert _sanitize("") == "_anon"


def test_sanitize_collision_avoidance():
    used = set()
    a = _sanitize("$a[0]", used)
    b = _sanitize("$a_0", used)
    c = _sanitize("$a_0", used)
    assert a != b != c
    assert a in used and b in used and c in used


def test_sanitize_escapes_keyword():
    out = _sanitize("module")
    assert out != "module"
    assert out.endswith("_")


# ---- Multi-bit signals ------------------------------------------------------


def test_multibit_input_emits_bus_port():
    g = GateGraph(
        inputs=[Signal("x", width=8, signed=True)],
        outputs=[Signal("y", width=8, signed=True)],
        gates=[
            Gate(name="y", kind="add", inputs=["x", "x"],
                 output_width=8, output_signed=True),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "input wire signed [7:0] x" in text
    assert "output wire signed [7:0] y" in text


def test_multibit_constant():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("c", width=8, signed=True)],
        gates=[Gate(name="c", kind="constant",
                    attrs={"value": -7},
                    output_width=8, output_signed=True)],
        top="t",
    )
    text = emit_module(g)
    # -7 in 8-bit two's complement = 0xF9 = 249 unsigned
    assert "8'sd249" in text


# ---- Kind dispatch ----------------------------------------------------------


def test_unknown_kind_raises():
    g = GateGraph(
        inputs=[Signal("x")],
        outputs=[Signal("y")],
        gates=[Gate(name="y", kind="totally_made_up_op",
                    inputs=["x"], output_width=1)],
        top="t",
    )
    with pytest.raises(ValueError, match="no backend lowering"):
        emit_module(g)


def test_register_kind_emits_always_block():
    g = GateGraph(
        inputs=[Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[
            Gate(name="q", kind="register", inputs=["d"],
                 attrs={"clk": "clk", "rst": "rst", "init": 0},
                 output_width=4),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "input wire clk" in text
    assert "input wire rst" in text
    assert "output reg [3:0] q" in text
    assert "always @(posedge clk or posedge rst)" in text


def test_rom_kind_emits_init_block():
    g = GateGraph(
        inputs=[Signal("addr", width=2)],
        outputs=[Signal("data", width=8)],
        gates=[
            Gate(name="data", kind="rom", inputs=["addr"],
                 attrs={"init": [10, 20, 30, 40], "width": 8, "depth": 4},
                 output_width=8),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "reg [7:0] data_mem [0:3];" in text
    assert "data_mem[0] = 8'd10;" in text
    assert "data_mem[3] = 8'd40;" in text


def test_arithmetic_kinds_round_trip():
    g = GateGraph(
        inputs=[Signal("a", width=8, signed=True),
                Signal("b", width=8, signed=True)],
        outputs=[Signal("sum", width=8, signed=True),
                 Signal("diff", width=8, signed=True),
                 Signal("prod", width=16, signed=True)],
        gates=[
            Gate(name="sum", kind="add", inputs=["a", "b"],
                 output_width=8, output_signed=True),
            Gate(name="diff", kind="sub", inputs=["a", "b"],
                 output_width=8, output_signed=True),
            Gate(name="prod", kind="mul", inputs=["a", "b"],
                 output_width=16, output_signed=True),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "$signed(a) + $signed(b)" in text
    assert "$signed(a) - $signed(b)" in text
    assert "$signed(a) * $signed(b)" in text


def test_relu_clamp_concat_slice():
    g = GateGraph(
        inputs=[Signal("x", width=8, signed=True),
                Signal("a", width=4),
                Signal("b", width=4)],
        outputs=[Signal("rl", width=8, signed=True),
                 Signal("cl", width=8, signed=True),
                 Signal("cc", width=8),
                 Signal("sl", width=4)],
        gates=[
            Gate(name="rl", kind="relu", inputs=["x"],
                 output_width=8, output_signed=True),
            Gate(name="cl", kind="clamp", inputs=["x"],
                 attrs={"lo": -16, "hi": 15},
                 output_width=8, output_signed=True),
            Gate(name="cc", kind="concat", inputs=["a", "b"],
                 output_width=8),
            Gate(name="sl", kind="slice", inputs=["cc"],
                 attrs={"hi": 7, "lo": 4},
                 output_width=4),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "$signed(x) > 0" in text
    assert "{a, b}" in text
    assert "cc[7:4]" in text


# ---- Custom kind via @lowering ---------------------------------------------


def test_custom_kind_via_lowering_decorator():
    @lowering("__test_custom__")
    def lower(ctx, g):
        return [f"  assign {ctx.name(g.name)} = ~{ctx.name(g.inputs[0])};"]

    assert "__test_custom__" in registered_kinds()

    g = GateGraph(
        inputs=[Signal("x")],
        outputs=[Signal("y")],
        gates=[Gate(name="y", kind="__test_custom__", inputs=["x"],
                    output_width=1)],
        top="t",
    )
    text = emit_module(g)
    assert "assign y = ~x;" in text


# ---- BRAM template ----------------------------------------------------------


def test_bram_template_basic():
    text = emit_bram_template(addr_bits=10, data_bits=8, module_name="my_bram")
    assert "module my_bram" in text
    assert "[9:0]" in text
    assert "[7:0]" in text
    assert "0:1023" in text


def test_bram_template_validates_module_name():
    with pytest.raises(ValueError):
        emit_bram_template(module_name="123bad")


def test_bram_template_validates_widths():
    with pytest.raises(ValueError):
        emit_bram_template(addr_bits=0)
    with pytest.raises(ValueError):
        emit_bram_template(addr_bits=64)
    with pytest.raises(ValueError):
        emit_bram_template(data_bits=0)
