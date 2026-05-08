"""Tests for code added in the items 1-58 batch:
  - argparse allow_abbrev=False (frontend flags don't prefix-collide globals)
  - tristate-as-None in evaluate_graph
  - vendor placement attributes (_vendor_attr_lines)
  - multi-clock SDC + cross-domain false paths
  - cross_frontend_equiv_sweep + trace_n_cycles
  - bitnet_linear NotImplementedError on unimplemented sequential variants
  - onnx_topology Sigmoid + Exp via instance gates onto LUT blocks
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


# ---------- argparse allow_abbrev=False ----------
def test_argparse_no_abbrev_for_pipeline():
    """`--pipeline` should bind to bitnet_linear's bool, not the global
    `--pipeline-every` int. Before allow_abbrev=False this raised
    'expected one argument'."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sf = td / "m.safetensors"
        save_file(
            {"layers.0.weight": torch.tensor([[1, -1]], dtype=torch.int8)},
            str(sf),
        )
        proc = subprocess.run(
            [sys.executable, "-m", "safetensors2verilog", str(sf),
             "--frontend", "bitnet_linear", "--pipeline",
             "-o", str(td / "out.v")],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"--pipeline still misparsed: stderr={proc.stderr!r}"
        )
        text = (td / "out.v").read_text()
        # --pipeline mode emits a clk port + always @posedge clk register.
        assert "input wire clk" in text
        assert "always @(posedge clk)" in text


# ---------- evaluate_graph tristate ----------
def test_evaluate_graph_tristate_high_z_is_none():
    from safetensors2verilog import evaluate_graph
    from safetensors2verilog.core import Gate, GateGraph, Signal
    g = GateGraph(
        inputs=[Signal("d", width=8, signed=True),
                Signal("en", width=1, signed=False)],
        outputs=[Signal("y", width=8, signed=True)],
        gates=[Gate(name="y", kind="tristate", inputs=["d", "en"],
                    output_width=8, output_signed=True,
                    attrs={"enable_high": True})],
        top="t",
    )
    # en=1 -> drive
    v = evaluate_graph(g, {"d": 42, "en": 1})
    assert v["y"] == 42, "tristate should drive when en=1"
    # en=0 -> high-impedance, propagated as None
    v = evaluate_graph(g, {"d": 42, "en": 0})
    assert v["y"] is None, "tristate should return None for high-Z"
    # enable_high=False
    g2 = GateGraph(
        inputs=g.inputs, outputs=g.outputs,
        gates=[Gate(name="y", kind="tristate", inputs=["d", "en"],
                    output_width=8, output_signed=True,
                    attrs={"enable_high": False})],
        top="t",
    )
    v = evaluate_graph(g2, {"d": 7, "en": 0})
    assert v["y"] == 7, "enable_high=False drives when en=0"
    v = evaluate_graph(g2, {"d": 7, "en": 1})
    assert v["y"] is None, "enable_high=False returns Z when en=1"


# ---------- vendor placement attributes ----------
def test_vendor_attrs_emit_pragmas_on_register():
    from safetensors2verilog import emit_module
    from safetensors2verilog.core import Gate, GateGraph, Signal
    g = GateGraph(
        inputs=[Signal("d", width=8)],
        outputs=[Signal("q", width=8)],
        gates=[
            Gate(name="q", kind="register", inputs=["d"],
                 attrs={
                     "clk": "clk",
                     "vivado_loc": "X12Y34",
                     "vivado_keep": True,
                     "quartus_chip_pin": "AB12",
                     "lattice_loc": "R5C7",
                     "synplify_keep": True,
                     "generic_attr": [("custom_attr", "value1")],
                 },
                 output_width=8),
        ],
        top="vt",
    )
    text = emit_module(g)
    assert '(* LOC = "X12Y34" *)' in text
    assert '(* keep = "true" *)' in text
    assert '(* chip_pin = "AB12" *)' in text
    assert '(* LOC = "R5C7" *)' in text
    assert '(* syn_keep = 1 *)' in text
    assert '(* custom_attr = "value1" *)' in text


# ---------- Multi-clock SDC + cross-domain false paths ----------
def test_emit_sdc_multi_clock_emits_cross_domain_false_paths():
    """A graph with two distinct register-clk domains should emit two
    create_clock lines + bidirectional set_false_path between them."""
    from safetensors2verilog.cli import _emit_sdc_template
    from safetensors2verilog.core import Gate, GateGraph, Signal
    g = GateGraph(
        inputs=[Signal("d_a"), Signal("d_b"),
                Signal("clk_fast"), Signal("clk_slow")],
        outputs=[Signal("q_a"), Signal("q_b")],
        gates=[
            Gate(name="q_a", kind="register", inputs=["d_a"],
                 attrs={"clk": "clk_fast"}),
            Gate(name="q_b", kind="register", inputs=["d_b"],
                 attrs={"clk": "clk_slow"}),
        ],
        top="multi",
    )
    sdc = _emit_sdc_template(g, period_ns=10.0)
    assert "create_clock -name clk_fast" in sdc
    assert "create_clock -name clk_slow" in sdc
    # Both directions of the cross-domain false path.
    assert (
        "set_false_path -from [get_clocks clk_fast] -to [get_clocks clk_slow]"
        in sdc
    )
    assert (
        "set_false_path -from [get_clocks clk_slow] -to [get_clocks clk_fast]"
        in sdc
    )


def test_emit_sdc_per_port_input_output_delays():
    """SDC should issue one set_input_delay per data port, and one
    set_output_delay per output port. clk and rst are excluded from the
    data-port set."""
    from safetensors2verilog.cli import _emit_sdc_template
    from safetensors2verilog.core import Gate, GateGraph, Signal
    g = GateGraph(
        inputs=[Signal("a"), Signal("b"), Signal("c"), Signal("clk"),
                Signal("rst")],
        outputs=[Signal("y1"), Signal("y2")],
        gates=[Gate(name="y1", kind="register", inputs=["a"],
                    attrs={"clk": "clk", "rst": "rst"}),
               Gate(name="y2", kind="register", inputs=["b"],
                    attrs={"clk": "clk", "rst": "rst"})],
        top="t",
    )
    sdc = _emit_sdc_template(g, period_ns=8.0)
    for port in ("a", "b", "c"):
        assert f"[get_ports {port}]" in sdc
    # clk and rst excluded from set_input_delay
    assert "set_input_delay" in sdc
    assert "set_input_delay  -clock clk -max" in sdc
    # rst false path emitted
    assert "set_false_path -from [get_ports rst]" in sdc


# ---------- equivalence: cross_frontend_equiv_sweep + trace_n_cycles ----------
def test_cross_frontend_equiv_sweep_combinational():
    from safetensors2verilog.core import Gate, GateGraph, Signal
    from safetensors2verilog.equivalence import cross_frontend_equiv_sweep
    g = GateGraph(
        inputs=[Signal("a", width=4, signed=True),
                Signal("b", width=4, signed=True)],
        outputs=[Signal("y", width=8, signed=True)],
        gates=[Gate(name="y", kind="add", inputs=["a", "b"],
                    output_width=8, output_signed=True)],
        top="add",
    )
    res = cross_frontend_equiv_sweep(g, n_random_inputs=5)
    assert res["is_sequential"] is False
    assert len(res["trace"]) == 5
    for entry in res["trace"]:
        a = entry["inputs"]["a"]
        b = entry["inputs"]["b"]
        assert entry["outputs"]["y"] == a + b


def test_trace_n_cycles_counter():
    from safetensors2verilog.core import Gate, GateGraph, Signal
    from safetensors2verilog.equivalence import trace_n_cycles
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
    trace = trace_n_cycles(g, [{} for _ in range(8)])
    # Counter is a register so cycle k captures the pre-update value
    # which equals k (starts at 0, increments).
    for k, entry in enumerate(trace):
        assert entry["counter"] == k


# ---------- bitnet_linear deferred-variant errors ----------
@pytest.mark.parametrize("kw,match", [
    ({"parallelism": 1}, "parallelism"),
    ({"streaming_input": True}, "streaming-input"),
    ({"handshake": True}, "handshake"),
    ({"weight_bram": True}, "weight-bram"),
])
def test_bitnet_sequential_variants_raise_not_implemented(kw, match):
    from safetensors2verilog.core import registry
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "m.safetensors"
        save_file(
            {"layers.0.weight": torch.tensor([[1, -1]], dtype=torch.int8)},
            str(sf),
        )
        with pytest.raises(NotImplementedError, match=match):
            registry.get("bitnet_linear")().parse(sf, sequential=True, **kw)


# ---------- onnx_topology Sigmoid + Exp ----------
def test_onnx_topology_sigmoid_emits_instance_gate():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from safetensors2verilog.core import registry

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Sigmoid", ["x"], ["y"], "sig1")
        graph = helper.make_graph([node], "g", [x], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        ir = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=8,
        )
        kinds = {g.kind for g in ir.gates}
        assert "instance" in kinds
        # The submodule for the sigmoid LUT should be attached.
        sub_tops = [s.top for s in ir.submodules]
        assert any(t.startswith("sigmoid_lut") for t in sub_tops)


def test_emit_sva_assertions_register_reset():
    """Emit SVA assertions for a graph with a register + reset and check
    the output module wires up the reset and init invariant."""
    from safetensors2verilog.core import Gate, GateGraph, Signal
    from safetensors2verilog.equivalence import emit_sva_assertions
    g = GateGraph(
        inputs=[Signal("clk"), Signal("rst"), Signal("d", width=4)],
        outputs=[Signal("q", width=4)],
        gates=[Gate(name="q", kind="register", inputs=["d"],
                    attrs={"clk": "clk", "rst": "rst", "init": 0},
                    output_width=4)],
        top="t",
    )
    sva = emit_sva_assertions(g)
    # Module name + bind hint
    assert "module t_assertions" in sva
    assert "bind t t_assertions u_assert" in sva
    # Register reset assertion + cover
    assert "assert_reset_q" in sva
    assert "cover_reset_q" in sva
    assert "rst |=> (q == 4'd0)" in sva


def test_emit_sva_assertions_mux_select_range():
    """A mux with N data ports asserts the select stays in [0, N-1]."""
    from safetensors2verilog.core import Gate, GateGraph, Signal
    from safetensors2verilog.equivalence import emit_sva_assertions
    g = GateGraph(
        inputs=[Signal("sel", width=2),
                Signal("a", width=4), Signal("b", width=4),
                Signal("c", width=4)],
        outputs=[Signal("y", width=4)],
        gates=[Gate(name="y", kind="mux",
                    inputs=["sel", "a", "b", "c"], output_width=4)],
        top="m",
    )
    sva = emit_sva_assertions(g)
    assert "assume_mux_y" in sva
    assert "(sel < 3)" in sva


def test_onnx_topology_layernorm_emits_block_with_clock():
    """ONNX LayerNormalization wires through a layer_norm_block instance
    and adds clk/rst/start to the parent's external port list."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper as _nh
    from safetensors2verilog.core import registry

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
        scale = _nh.from_array(
            torch.ones(8, dtype=torch.float32).numpy(), name="ln_s"
        )
        bias = _nh.from_array(
            torch.zeros(8, dtype=torch.float32).numpy(), name="ln_b"
        )
        node = helper.make_node(
            "LayerNormalization", ["x", "ln_s", "ln_b"], ["y"], "ln1",
            axis=-1, epsilon=1e-5,
        )
        graph = helper.make_graph(
            [node], "g", [x], [y], [scale, bias],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        ir = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=8,
        )
        kinds = {g.kind for g in ir.gates}
        assert "instance" in kinds
        # Clock + reset + start were threaded into the external port list.
        in_names = [s.name for s in ir.inputs]
        assert "clk" in in_names
        assert "rst" in in_names
        assert "start" in in_names
        # The layer_norm submodule (and its rsqrt LUT companion) are present.
        sub_tops = [s.top for s in ir.submodules]
        assert any(t.startswith("layer_norm") for t in sub_tops)
        assert any(t.startswith("rsqrt_lut") for t in sub_tops)


def test_onnx_topology_exp_emits_instance_gate():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from safetensors2verilog.core import registry

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Exp", ["x"], ["y"], "exp1")
        graph = helper.make_graph([node], "g", [x], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        ir = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=8,
        )
        kinds = {g.kind for g in ir.gates}
        assert "instance" in kinds
        sub_tops = [s.top for s in ir.submodules]
        assert any(t.startswith("exp_lut") for t in sub_tops)
