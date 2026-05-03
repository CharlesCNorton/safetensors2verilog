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


# ---- Shifts -----------------------------------------------------------------


def test_shift_left_emits_correct_operator():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="shift_left", inputs=["x"],
                    attrs={"amount": 3}, output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "x << 3" in text


def test_shift_right_unsigned_emits_logical_shift():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="shift_right", inputs=["x"],
                    attrs={"amount": 2}, output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert ">> 2" in text and ">>>" not in text


def test_shift_right_signed_emits_arithmetic_shift():
    g = GateGraph(
        inputs=[Signal("x", width=8, signed=True)],
        outputs=[Signal("y", width=8, signed=True)],
        gates=[Gate(name="y", kind="shift_right", inputs=["x"],
                    attrs={"amount": 1},
                    output_width=8, output_signed=True)],
        top="t",
    )
    text = emit_module(g)
    assert ">>> 1" in text
    assert "$signed(x)" in text


# ---- Multi-way mux ----------------------------------------------------------


def test_multi_way_mux_emits_chained_ternary():
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
    text = emit_module(g)
    # 4-way mux: chained ternary referencing sel == 0/1/2 plus default
    assert "(sel == 0)" in text
    assert "(sel == 1)" in text
    assert "(sel == 2)" in text


def test_mux_rejects_data_width_mismatch():
    g = GateGraph(
        inputs=[Signal("sel", width=1),
                Signal("a", width=8), Signal("b", width=4)],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="mux", inputs=["sel", "a", "b"],
                    output_width=8)],
        top="t",
    )
    with pytest.raises(ValueError, match="mux"):
        emit_module(g)


# ---- Slice / concat width validation ---------------------------------------


def test_slice_single_bit():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=1)],
        gates=[Gate(name="y", kind="slice", inputs=["x"],
                    attrs={"hi": 3, "lo": 3}, output_width=1)],
        top="t",
    )
    text = emit_module(g)
    assert "x[3]" in text


def test_slice_rejects_width_mismatch():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=4)],
        gates=[Gate(name="y", kind="slice", inputs=["x"],
                    attrs={"hi": 7, "lo": 0},   # would need width 8
                    output_width=4)],
        top="t",
    )
    with pytest.raises(ValueError, match="slice"):
        emit_module(g)


def test_slice_rejects_out_of_range():
    g = GateGraph(
        inputs=[Signal("x", width=4)],
        outputs=[Signal("y", width=2)],
        gates=[Gate(name="y", kind="slice", inputs=["x"],
                    attrs={"hi": 5, "lo": 4}, output_width=2)],
        top="t",
    )
    with pytest.raises(ValueError, match="only 4 bits"):
        emit_module(g)


def test_slice_rejects_hi_lt_lo():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=1)],
        gates=[Gate(name="y", kind="slice", inputs=["x"],
                    attrs={"hi": 2, "lo": 5}, output_width=1)],
        top="t",
    )
    with pytest.raises(ValueError, match="hi=2 < lo=5"):
        emit_module(g)


def test_concat_rejects_width_mismatch():
    g = GateGraph(
        inputs=[Signal("a", width=4), Signal("b", width=4)],
        outputs=[Signal("y", width=10)],   # 4+4 != 10
        gates=[Gate(name="y", kind="concat", inputs=["a", "b"],
                    output_width=10)],
        top="t",
    )
    with pytest.raises(ValueError, match="concat"):
        emit_module(g)


# ---- ROM extras -------------------------------------------------------------


def test_rom_zero_pads_when_depth_exceeds_init():
    g = GateGraph(
        inputs=[Signal("addr", width=2)],
        outputs=[Signal("d", width=8)],
        gates=[Gate(name="d", kind="rom", inputs=["addr"],
                    attrs={"init": [10, 20], "width": 8, "depth": 4},
                    output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "d_mem[0] = 8'd10;" in text
    assert "d_mem[1] = 8'd20;" in text
    assert "d_mem[2] = 8'd0;" in text
    assert "d_mem[3] = 8'd0;" in text


def test_rom_emits_ram_style_pragma():
    g = GateGraph(
        inputs=[Signal("addr", width=1)],
        outputs=[Signal("d", width=8)],
        gates=[Gate(name="d", kind="rom", inputs=["addr"],
                    attrs={"init": [0, 1], "width": 8, "depth": 2,
                           "ram_style": "block"},
                    output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert '(* ram_style = "block" *)' in text


def test_rom_rejects_init_larger_than_depth():
    g = GateGraph(
        inputs=[Signal("addr", width=1)],
        outputs=[Signal("d", width=8)],
        gates=[Gate(name="d", kind="rom", inputs=["addr"],
                    attrs={"init": [0, 1, 2, 3], "width": 8, "depth": 2},
                    output_width=8)],
        top="t",
    )
    with pytest.raises(ValueError, match="smaller than init"):
        emit_module(g)


def test_rom_rejects_width_mismatch():
    g = GateGraph(
        inputs=[Signal("addr", width=1)],
        outputs=[Signal("d", width=4)],
        gates=[Gate(name="d", kind="rom", inputs=["addr"],
                    attrs={"init": [0, 1], "width": 8, "depth": 2},
                    output_width=4)],
        top="t",
    )
    with pytest.raises(ValueError, match="output_width"):
        emit_module(g)


# ---- Register edge cases ----------------------------------------------------


def test_register_without_rst():
    g = GateGraph(
        inputs=[Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d"],
                    attrs={"clk": "clk"}, output_width=4)],
        top="t",
    )
    text = emit_module(g)
    assert "always @(posedge clk) q <= d;" in text
    assert "always @(posedge clk or" not in text


def test_register_feedback_counter():
    """A counter is a register whose D input feeds back through +1.

    Output -> add -> register -> output forms a sequential cycle that
    the topo validator must allow.
    """
    g = GateGraph(
        inputs=[],
        outputs=[Signal("counter", width=4)],
        gates=[
            Gate(name="one", kind="constant",
                 attrs={"value": 1}, output_width=4),
            Gate(name="counter_next", kind="add",
                 inputs=["counter", "one"],
                 output_width=4),
            Gate(name="counter", kind="register",
                 inputs=["counter_next"],
                 attrs={"clk": "clk", "rst": "rst"},
                 output_width=4),
        ],
        top="cnt",
    )
    text = emit_module(g)
    # Should emit clean Verilog with a feedback assign and an always block.
    assert "input wire clk" in text
    assert "input wire rst" in text
    assert "counter_next" in text
    assert "always @(posedge clk or posedge rst)" in text


def test_register_d_input_validation():
    """A register's D input must be produced by some gate or external input."""
    g = GateGraph(
        inputs=[],
        outputs=[Signal("q", width=1)],
        gates=[Gate(name="q", kind="register",
                    inputs=["nonexistent"],
                    attrs={"clk": "clk"}, output_width=1)],
        top="t",
    )
    with pytest.raises(ValueError, match="not produced"):
        emit_module(g)


def test_register_with_sync_reset():
    g = GateGraph(
        inputs=[Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d"],
                    attrs={"clk": "clk", "rst": "rst",
                           "reset_kind": "sync", "init": 0},
                    output_width=4)],
        top="t",
    )
    text = emit_module(g)
    assert "always @(posedge clk) begin" in text
    assert "if (rst) q <= 4'd0;" in text


def test_register_with_async_low_reset():
    g = GateGraph(
        inputs=[Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d"],
                    attrs={"clk": "clk", "rst": "rst_n",
                           "reset_polarity": "low", "init": 0},
                    output_width=4)],
        top="t",
    )
    text = emit_module(g)
    assert "always @(posedge clk or negedge rst_n)" in text
    assert "if (!rst_n)" in text


def test_register_with_enable():
    g = GateGraph(
        inputs=[Signal("d", width=4), Signal("en", width=1)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d", "en"],
                    attrs={"clk": "clk"}, output_width=4)],
        top="t",
    )
    text = emit_module(g)
    assert "q <= en ? d : q;" in text


def test_tristate_kind():
    g = GateGraph(
        inputs=[Signal("d", width=8), Signal("en", width=1)],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="tristate", inputs=["d", "en"],
                    output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "8'bz" in text
    assert "en ? d :" in text


def test_inout_signal_emits_inout_port():
    g = GateGraph(
        inputs=[Signal("io_data", width=8, direction="inout"),
                Signal("en", width=1)],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="add",
                    inputs=["io_data", "io_data"], output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "inout wire [7:0] io_data" in text


def test_parameter_kind_emits_localparam():
    g = GateGraph(
        inputs=[Signal("x", width=8)],
        outputs=[Signal("y", width=8)],
        gates=[
            Gate(name="MAGIC", kind="parameter",
                 attrs={"value": 42}, output_width=8),
            Gate(name="y", kind="add", inputs=["x", "MAGIC"], output_width=8),
        ],
        top="t",
    )
    text = emit_module(g)
    assert "localparam" in text
    assert "MAGIC" in text


def test_module_parameter_emits_in_hash_paren():
    """Signal with is_parameter=True becomes `module foo #(parameter ...)`."""
    g = GateGraph(
        inputs=[
            Signal("WIDTH", width=8, is_parameter=True, parameter_value=8),
            Signal("x", width=8),
        ],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="add", inputs=["x", "x"], output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "module t #(" in text
    assert "parameter [7:0] WIDTH" in text


def test_dsp_attribute_on_mul():
    g = GateGraph(
        inputs=[Signal("a", width=8, signed=True),
                Signal("b", width=8, signed=True)],
        outputs=[Signal("p", width=16, signed=True)],
        gates=[Gate(name="p", kind="mul", inputs=["a", "b"],
                    attrs={"use_dsp": True},
                    output_width=16, output_signed=True)],
        top="t",
    )
    text = emit_module(g)
    assert '(* use_dsp = "yes" *)' in text


def test_mux_case_form_for_many_inputs():
    """A mux with > 4 data inputs emits a case statement."""
    inputs_list = [Signal("sel", width=3)] + [
        Signal(f"d{i}", width=4) for i in range(6)
    ]
    g = GateGraph(
        inputs=inputs_list,
        outputs=[Signal("y", width=4)],
        gates=[Gate(name="y", kind="mux",
                    inputs=["sel"] + [f"d{i}" for i in range(6)],
                    output_width=4)],
        top="t",
    )
    text = emit_module(g)
    assert "case (sel)" in text
    assert "endcase" in text
    # The output must be declared as reg (always @(*) drives it)
    assert "output reg [3:0] y" in text


def test_systemverilog_target_uses_logic_keyword():
    g = GateGraph(
        inputs=[Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d"],
                    attrs={"clk": "clk"}, output_width=4)],
        top="t",
    )
    text = emit_module(g, target="sv")
    assert "logic" in text
    assert "always_ff @(posedge clk)" in text
    assert "wire" not in text or "wire " not in text  # crude check


def test_unknown_target_rejected():
    g = GateGraph(inputs=[], outputs=[], gates=[], top="t")
    with pytest.raises(ValueError, match="unknown target"):
        emit_module(g, target="vhdl")


def test_collect_clocks_finds_distinct_clocks():
    """collect_clocks returns the set of clk attrs across register gates."""
    from safetensors2verilog import collect_clocks, collect_resets
    g = GateGraph(
        inputs=[Signal("d1"), Signal("d2"),
                Signal("clk_fast"), Signal("clk_slow"),
                Signal("rst_a"), Signal("rst_b")],
        outputs=[Signal("q1"), Signal("q2")],
        gates=[
            Gate(name="q1", kind="register", inputs=["d1"],
                 attrs={"clk": "clk_fast", "rst": "rst_a"}),
            Gate(name="q2", kind="register", inputs=["d2"],
                 attrs={"clk": "clk_slow", "rst": "rst_b"}),
        ],
        top="t",
    )
    assert collect_clocks(g) == {"clk_fast", "clk_slow"}
    assert collect_resets(g) == {"rst_a", "rst_b"}


def test_emit_top_wrapper_links_core_and_bram():
    """The wrapper helper instantiates the carved-out core and a BRAM."""
    from safetensors2verilog import emit_top_wrapper
    text = emit_top_wrapper(
        core_module="threshold_cpu",
        bram_module="threshold_bram",
        addr_bits=10,
        data_bits=8,
        wrapper_name="cpu_with_bram",
    )
    assert "module cpu_with_bram" in text
    assert "threshold_cpu core" in text
    assert "threshold_bram bram" in text
    assert "[9:0] addr" in text
    assert "[7:0] data_in" in text
    assert ".clk(clk)" in text
    assert ".we(we)" in text


def test_emit_top_wrapper_validates_names():
    from safetensors2verilog import emit_top_wrapper
    with pytest.raises(ValueError):
        emit_top_wrapper(core_module="123_bad", bram_module="ok",
                         wrapper_name="ok2")


# ---- Constant edge cases ----------------------------------------------------


def test_negative_signed_constant_emits_twos_complement():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("c", width=4, signed=True)],
        gates=[Gate(name="c", kind="constant",
                    attrs={"value": -3},
                    output_width=4, output_signed=True)],
        top="t",
    )
    text = emit_module(g)
    # -3 in 4-bit two's complement = 13 (0b1101)
    assert "4'sd13" in text


def test_unsigned_constant_emits_unsigned_literal():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("c", width=8)],
        gates=[Gate(name="c", kind="constant",
                    attrs={"value": 200}, output_width=8)],
        top="t",
    )
    text = emit_module(g)
    assert "8'd200" in text
    assert "8'sd" not in text


# ---- Arity checks for previously-unchecked kinds ---------------------------


def test_shift_left_rejects_zero_inputs():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("y", width=8)],
        gates=[Gate(name="y", kind="shift_left", inputs=[],
                    attrs={"amount": 1}, output_width=8)],
        top="t",
    )
    with pytest.raises(ValueError, match="shift_left"):
        emit_module(g)


def test_slice_rejects_zero_inputs():
    g = GateGraph(
        inputs=[],
        outputs=[Signal("y", width=1)],
        gates=[Gate(name="y", kind="slice", inputs=[],
                    attrs={"hi": 0, "lo": 0}, output_width=1)],
        top="t",
    )
    with pytest.raises(ValueError, match="slice"):
        emit_module(g)
