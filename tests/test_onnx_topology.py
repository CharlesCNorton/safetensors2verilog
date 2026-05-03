"""Tests for the onnx_topology frontend.

Skipped when the optional `onnx` package isn't installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from safetensors2verilog.core import registry
from safetensors2verilog.verilog import emit_module

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _build_linear_relu_linear(out_path: Path,
                              W1, B1, W2, B2,
                              in_size: int):
    """Build an ONNX model: input -> Gemm(W1, B1) -> Relu -> Gemm(W2, B2) -> output."""
    # Gemm uses transB=1 convention (weight stored as [out, in])
    out_size = len(B2)

    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, out_size]),
    ]

    initializers = [
        numpy_helper.from_array(
            torch.tensor(W1, dtype=torch.float32).numpy(), name="W1"
        ),
        numpy_helper.from_array(
            torch.tensor(B1, dtype=torch.float32).numpy(), name="B1"
        ),
        numpy_helper.from_array(
            torch.tensor(W2, dtype=torch.float32).numpy(), name="W2"
        ),
        numpy_helper.from_array(
            torch.tensor(B2, dtype=torch.float32).numpy(), name="B2"
        ),
    ]

    nodes = [
        helper.make_node(
            "Gemm", ["x", "W1", "B1"], ["h0"], "linear1", transB=1
        ),
        helper.make_node("Relu", ["h0"], ["h1"], "relu1"),
        helper.make_node(
            "Gemm", ["h1", "W2", "B2"], ["y"], "linear2", transB=1
        ),
    ]

    graph = helper.make_graph(nodes, "mlp", inputs, outputs, initializers)
    model = helper.make_model(
        graph, producer_name="safetensors2verilog-test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(out_path))


def test_onnx_topology_round_trip():
    """A 3 -> 2 -> 1 MLP with int weights converts to a multi-gate IR."""
    W1 = [[1, -1, 1], [1, 1, -1]]
    B1 = [0, 1]
    W2 = [[2, -3]]
    B2 = [-1]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        onnx_path = td / "model.onnx"
        st_path = td / "weights.safetensors"
        _build_linear_relu_linear(onnx_path, W1, B1, W2, B2, in_size=3)

        # Empty safetensors: the frontend will use ONNX initializers for weights.
        # safetensors needs at least one tensor; insert a dummy that the
        # frontend will ignore.
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(st_path))

        graph = registry.get("onnx_topology")().parse(
            st_path, top="mlp", onnx=str(onnx_path),
            activation_bits=4, weight_bits=4,
        )
        kinds = {g.kind for g in graph.gates}
        assert "linear" in kinds
        assert "relu" in kinds
        assert len(graph.inputs) == 3
        assert len(graph.outputs) == 1
        text = emit_module(graph)
        assert "module mlp" in text


def test_onnx_topology_safetensors_overrides_onnx_weights():
    """When the safetensors file provides a same-named tensor, it wins."""
    # ONNX has W1 with all 1s; safetensors overrides W1 to all -1.
    W_onnx = [[1, 1]]
    W_st = [[-1, -1]]
    B = [0]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        onnx_path = td / "model.onnx"
        st_path = td / "w.safetensors"

        # Build a single-Gemm ONNX
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
        inits = [
            numpy_helper.from_array(
                torch.tensor(W_onnx, dtype=torch.float32).numpy(), name="W1"
            ),
            numpy_helper.from_array(
                torch.tensor(B, dtype=torch.float32).numpy(), name="B1"
            ),
        ]
        nodes = [helper.make_node("Gemm", ["x", "W1", "B1"], ["y"], "g", transB=1)]
        graph = helper.make_graph(nodes, "g", [x], [y], inits)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(onnx_path))

        # safetensors override
        save_file(
            {"W1": torch.tensor(W_st, dtype=torch.int8)},
            str(st_path),
        )

        result = registry.get("onnx_topology")().parse(
            st_path, onnx=str(onnx_path), weight_bits=4
        )
        linear = next(g for g in result.gates if g.kind == "linear")
        assert linear.attrs["weights"] == [-1, -1]


def test_onnx_topology_unsupported_op_raises():
    """An unsupported op (e.g. Sigmoid) raises NotImplementedError naming the op."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        onnx_path = td / "model.onnx"
        st_path = td / "w.safetensors"

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Sigmoid", ["x"], ["y"], "sig")
        graph = helper.make_graph([node], "g", [x], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(onnx_path))

        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(st_path))

        with pytest.raises(NotImplementedError, match="Sigmoid"):
            registry.get("onnx_topology")().parse(st_path, onnx=str(onnx_path))


def test_onnx_topology_requires_onnx_path():
    """Calling parse() without --onnx raises a helpful error."""
    with tempfile.TemporaryDirectory() as td:
        st_path = Path(td) / "w.safetensors"
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(st_path))
        with pytest.raises(ValueError, match="--onnx"):
            registry.get("onnx_topology")().parse(st_path)


def test_onnx_topology_multi_input_graph():
    """A graph with two non-initializer inputs becomes two banks of ports."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 3])
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Add", ["a", "b"], ["y"], "add1")
        graph = helper.make_graph([node], "g", [a, b], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        # 2 inputs * 3 elements = 6 input ports
        in_names = sorted(s.name for s in result.inputs)
        assert len(in_names) == 6
        # And the names are prefixed since multi-input
        assert any(n.startswith("a_") for n in in_names)
        assert any(n.startswith("b_") for n in in_names)


def test_onnx_topology_scalar_broadcasting():
    """Add with one operand being a scalar (length 1) broadcasts."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        scalar = numpy_helper.from_array(
            torch.tensor([3], dtype=torch.float32).numpy(), name="bias"
        )
        node = helper.make_node("Add", ["x", "bias"], ["y"], "broadcast_add")
        graph = helper.make_graph([node], "g", [x], [y], [scalar])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        # 4 add gates (one per output element), each adding x[i] + bias_const
        adds = [g for g in result.gates if g.kind == "add"]
        assert len(adds) == 4


def test_onnx_topology_concat_split():
    """Concat then Split round-trips the signal list."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2])
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2])
        out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, 2])
        out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 2])
        nodes = [
            helper.make_node("Concat", ["a", "b"], ["cc"], "c", axis=1),
            helper.make_node(
                "Split", ["cc"], ["out0", "out1"], "s",
                axis=1, split=[2, 2],
            ),
        ]
        graph = helper.make_graph(
            nodes, "g", [a, b], [out0, out1], []
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        # Two ONNX outputs, each carrying 2 elements: 4 IR output signals.
        assert len(result.outputs) == 4


def test_onnx_topology_gather():
    """Gather with initializer indices produces a permutation."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
        idx = numpy_helper.from_array(
            torch.tensor([3, 0, 2], dtype=torch.int64).numpy(), name="idx"
        )
        node = helper.make_node("Gather", ["x", "idx"], ["y"], "g", axis=1)
        graph = helper.make_graph([node], "g", [x], [y], [idx])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))

        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        assert len(result.outputs) == 3


def test_onnx_topology_vector_vector_tile_broadcast():
    """When one Add operand is shorter and divides the other, it tiles."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6])
        # length-2 bias tiles 3x to match length-6 input
        bias = numpy_helper.from_array(
            torch.tensor([1, 2], dtype=torch.float32).numpy(), name="bias"
        )
        node = helper.make_node("Add", ["x", "bias"], ["y"], "tile_add")
        graph = helper.make_graph([node], "g", [x], [y], [bias])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))
        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        adds = [g for g in result.gates if g.kind == "add"]
        assert len(adds) == 6  # one per output element


def test_onnx_topology_dynamic_gather_emits_mux():
    """Gather with dynamic index ports lowers to per-output mux gates."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        idx = helper.make_tensor_value_info("idx", TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        node = helper.make_node("Gather", ["x", "idx"], ["y"], "g", axis=1)
        graph = helper.make_graph([node], "g", [x, idx], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))
        result = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=4
        )
        # Should produce 2 mux gates, one per output index
        muxes = [g for g in result.gates if g.kind == "mux"]
        assert len(muxes) == 2


def test_onnx_topology_unsupported_op_lists_alternatives():
    """The error message should list supported ops and known deferred ops."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Sigmoid", ["x"], ["y"], "sig")
        graph = helper.make_graph([node], "g", [x], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(op))
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(sp))
        with pytest.raises(NotImplementedError) as exc:
            registry.get("onnx_topology")().parse(sp, onnx=str(op))
        msg = str(exc.value)
        assert "Sigmoid" in msg
        assert "Supported:" in msg
        assert "Gemm" in msg
        assert "Conv" in msg  # mentioned as deferred


def _have_iverilog() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


@pytest.mark.skipif(not _have_iverilog(),
                    reason="iverilog/vvp not on PATH")
def test_onnx_topology_numeric_round_trip():
    """End-to-end: ONNX MLP -> Verilog -> iverilog matches Python ground truth."""
    W1 = [[1, -1, 1], [1, 1, -1]]
    B1 = [0, 1]
    W2 = [[2, -3]]
    B2 = [-1]

    def py_eval(x):
        h0 = [
            sum(W1[j][i] * x[i] for i in range(3)) + B1[j]
            for j in range(2)
        ]
        h1 = [max(0, v) for v in h0]
        y = [
            sum(W2[j][i] * h1[i] for i in range(2)) + B2[j]
            for j in range(1)
        ]
        return y

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        onnx_path = td / "model.onnx"
        st_path = td / "w.safetensors"
        _build_linear_relu_linear(onnx_path, W1, B1, W2, B2, in_size=3)
        save_file({"_unused": torch.tensor([0], dtype=torch.int8)}, str(st_path))

        graph = registry.get("onnx_topology")().parse(
            st_path, top="mlp", onnx=str(onnx_path),
            activation_bits=4, weight_bits=4,
        )
        verilog = emit_module(graph)
        out_widths = {s.name: s.width for s in graph.outputs}
        out_names = sorted(out_widths)

        vpath = td / "mlp.v"
        tb_path = td / "tb.v"
        vpath.write_text(verilog)

        tb_lines = ["`timescale 1ns/1ps", "module tb;",
                    "  reg signed [3:0] x0, x1, x2;"]
        for n in out_names:
            tb_lines.append(
                f"  wire signed [{out_widths[n]-1}:0] {n.replace('.', '_')};"
            )
        tb_lines.append("  mlp dut (")
        port_lines = ["    .x0(x0), .x1(x1), .x2(x2)"]
        for n in out_names:
            v = n.replace(".", "_")
            port_lines.append(f"    , .{v}({v})")
        tb_lines.extend(port_lines)
        tb_lines.append("  );")

        cases = [(0, 0, 0), (1, 1, 1), (-1, 0, 1), (3, -2, 1), (-4, 4, -4)]
        tb_lines.append("  initial begin")
        for i, (a, bb, c) in enumerate(cases):
            tb_lines.append(f"    x0 = {a}; x1 = {bb}; x2 = {c}; #1;")
            fmts = " ".join("%0d" for _ in out_names)
            args = ", ".join(n.replace(".", "_") for n in out_names)
            tb_lines.append(f"    $display(\"{i} {fmts}\", {args});")
        tb_lines.append("    $finish; end endmodule")
        tb_path.write_text("\n".join(tb_lines))

        vvp = td / "tb.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp), str(vpath), str(tb_path)],
            check=True,
        )
        proc = subprocess.run(
            ["vvp", str(vvp)], check=True, capture_output=True, text=True
        )

        sim_lines = [
            l for l in proc.stdout.splitlines()
            if l and l.split()[0].isdigit()
        ]
        for i, (a, bb, c) in enumerate(cases):
            line = next(l for l in sim_lines if l.startswith(f"{i} "))
            sim = [int(t) for t in line.split()[1:]]
            expected = py_eval([a, bb, c])
            assert sim == expected, (
                f"case {i} x={(a, bb, c)}: sim={sim} expected={expected}"
            )
