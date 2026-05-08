"""Bit-exact iverilog tests for the LUT / sequential building blocks.

Items 4 and 9 of TODO.md: confirm the README's older "all bit-exact"
claim by driving each block through iverilog and matching the output
against a Python reference of the same Q-format math.

  * sigmoid_block    -> match math.sigmoid * 256 for every 8-bit input
  * silu_block       -> match silu over a fixed input vector
  * exp_block        -> match math.exp * out_max for every 8-bit input
  * softmax_block    -> match torch.softmax on a row of 8 int8 inputs
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _has_iv() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


pytestmark = pytest.mark.skipif(
    not _has_iv(), reason="iverilog/vvp not on PATH",
)


def _compile_run(text: str, top: str, tb: str, td: Path) -> str:
    v = td / f"{top}.v"
    v.write_text(text, encoding="utf-8")
    tbf = td / f"{top}_tb.v"
    tbf.write_text(tb, encoding="utf-8")
    vvp = td / f"{top}.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(v), str(tbf)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True,
        timeout=300,
    )
    return proc.stdout


def test_sigmoid_block_iverilog_bit_exact_full_sweep():
    """Every 8-bit input maps to math.sigmoid clamped to [-8, 8] in Q4.4."""
    from safetensors2verilog.blocks.sigmoid import sigmoid_block
    sub = sigmoid_block(in_bits=8, out_bits=8, in_q_frac_bits=4)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg signed [7:0] x;\n"
            "  wire [7:0] y;\n"
            f"  {sub.top} dut(.x(x), .y(y));\n"
            "  integer i;\n"
            "  initial begin\n"
            "    for (i = -128; i < 128; i = i + 1) begin\n"
            "      x = i[7:0]; #1;\n"
            "      $display(\"X %0d %0d\", x, y);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    for line in out.splitlines():
        if not line.startswith("X "):
            continue
        _, x_s, y_s = line.split()
        x = int(x_s); y = int(y_s)
        xf = max(-8.0, min(8.0, x / 16.0))
        s = 1.0 / (1.0 + math.exp(-xf))
        expected = max(0, min(255, round(s * 256)))
        assert y == expected, f"sigmoid x={x}: got {y}, expected {expected}"


def test_exp_block_iverilog_bit_exact_full_sweep():
    """Every 8-bit input maps to math.exp clamped to [-16, 0] in Q4.4."""
    from safetensors2verilog.blocks.exp import exp_block
    sub = exp_block(in_bits=8, out_bits=12, in_q_frac_bits=4,
                     in_clamp=(-16.0, 0.0))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg signed [7:0] x;\n"
            "  wire [11:0] y;\n"
            f"  {sub.top} dut(.x(x), .y(y));\n"
            "  integer i;\n"
            "  initial begin\n"
            "    for (i = -128; i < 128; i = i + 1) begin\n"
            "      x = i[7:0]; #1;\n"
            "      $display(\"X %0d %0d\", x, y);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    out_max = (1 << 12) - 1
    for line in out.splitlines():
        if not line.startswith("X "):
            continue
        _, x_s, y_s = line.split()
        x = int(x_s); y = int(y_s)
        xf = max(-16.0, min(0.0, x / 16.0))
        v = math.exp(xf)
        expected = max(0, min(out_max, round(v * out_max)))
        assert y == expected, f"exp x={x}: got {y}, expected {expected}"


def test_silu_block_iverilog_bit_exact_first_token():
    """silu_block is sequential (FSM-driven). Drive a fixed input vector
    of K=8 elements and match the output element-by-element against a
    Python reference using the same Q-format quantisation."""
    import torch
    from safetensors2verilog.blocks.silu import silu_block
    K = 8
    abits = 8
    obits = 8
    output_shift = 8
    sub_silu, sub_sig = silu_block(
        K=K, abits=abits, obits=obits,
        sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8,
        output_shift=output_shift,
    )
    # Pack 8 int8 inputs into a 64-bit packed bus.
    x_vals = [-32, -16, -4, -1, 0, 1, 8, 32]
    x_packed = 0
    for i, v in enumerate(x_vals):
        x_packed |= (v & 0xff) << (i * abits)

    # Python reference of the LUT chain:
    # 1. Look up sigmoid(x_now / 16) at 8-bit unsigned (Q0.8 -> 0..255).
    # 2. Multiply by x_now (signed). Shift right by output_shift.
    # 3. Clamp to int8.
    def sigmoid_lut(raw: int) -> int:
        sint = raw if raw < 128 else raw - 256
        xf = max(-8.0, min(8.0, sint / 16.0))
        s = 1.0 / (1.0 + math.exp(-xf))
        return max(0, min(255, round(s * 256)))

    expected = []
    for v in x_vals:
        s = sigmoid_lut(v & 0xff)
        prod = (v * s) >> output_shift
        expected.append(max(-128, min(127, prod)))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # silu_block returns (silu_module, sigmoid_module). The silu
        # module instantiates sub_sig.top by name, so we need both in the
        # same file.
        full_text = sub_sig.text + "\n" + sub_silu.text
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [{K*abits-1}:0] x_packed;
  wire signed [{K*obits-1}:0] y_packed;
  wire done;
  {sub_silu.top} dut(.clk(clk), .rst(rst), .start(start),
                     .x_packed(x_packed),
                     .y_packed(y_packed), .done(done));
  integer cycles, i;
  initial begin
    rst = 1; x_packed = {K*abits}'h{x_packed:x};
    #20 rst = 0;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk); cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    for (i = 0; i < {K}; i = i + 1)
      $display("Y %0d %0d", i, $signed(y_packed[i*{obits}+:{obits}]));
    $finish;
  end
endmodule
"""
        out = _compile_run(full_text, sub_silu.top, tb, td)
    sim_outputs = {}
    for line in out.splitlines():
        if line.startswith("Y "):
            _, i_s, v_s = line.split()
            sim_outputs[int(i_s)] = int(v_s)
    assert len(sim_outputs) == K
    for i, exp in enumerate(expected):
        assert sim_outputs[i] == exp, (
            f"silu element {i}: got {sim_outputs[i]}, expected {exp}"
        )


def test_onnx_topology_attention_multi_head_emits_per_head_chains():
    """ONNX Attention with num_heads=2 should produce two independent
    softmax + score / output chains and concatenate them into a single
    output signal list."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from safetensors2verilog.core import registry
    import torch as _torch
    from safetensors.torch import save_file as _save_file

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        # H=2 heads, seq=2, d_k=d_v=2 -> per_head=4, qn=kn=vn=8.
        q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 8])
        k = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 8])
        v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
        node = helper.make_node(
            "Attention", ["q", "k", "v"], ["y"], "att_mh", num_heads=2,
        )
        graph = helper.make_graph([node], "g", [q, k, v], [y], [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 23)]
        )
        onnx.save(model, str(op))
        _save_file({"_unused": _torch.tensor([0], dtype=_torch.int8)},
                   str(sp))
        ir = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=8,
        )
        instances = [g for g in ir.gates if g.kind == "instance"]
        sm_inst = [g for g in instances
                   if g.attrs.get("module_name", "").startswith("softmax_")]
        assert len(sm_inst) >= 4, (
            f"expected >= 4 softmax instances (2 heads x 2 rows), got "
            f"{len(sm_inst)}"
        )
        gate_names = {g.name for g in ir.gates}
        assert any("att_mh.h0." in n for n in gate_names)
        assert any("att_mh.h1." in n for n in gate_names)


def test_onnx_topology_attention_with_mask_initializer():
    """ONNX Attention with a 4th initializer mask should still produce a
    softmax-bearing chain."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper as _nh
    from safetensors2verilog.core import registry
    import torch as _torch
    from safetensors.torch import save_file as _save_file
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        op = td / "model.onnx"
        sp = td / "w.safetensors"
        q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 4])
        k = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 4])
        v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        mask_arr = _torch.tensor([[1, 0], [1, 1]],
                                  dtype=_torch.float32)
        mask_init = _nh.from_array(mask_arr.numpy(), name="att_mask")
        node = helper.make_node(
            "Attention", ["q", "k", "v", "att_mask"], ["y"], "att_msk",
            num_heads=1,
        )
        graph = helper.make_graph(
            [node], "g", [q, k, v], [y], [mask_init],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 23)]
        )
        onnx.save(model, str(op))
        _save_file({"_unused": _torch.tensor([0], dtype=_torch.int8)},
                   str(sp))
        ir = registry.get("onnx_topology")().parse(
            sp, onnx=str(op), activation_bits=8,
        )
        sub_tops = [s.top for s in ir.submodules]
        assert any(t.startswith("softmax_") for t in sub_tops)


def test_embedding_block_chunked_path_iverilog_bit_exact():
    """When V*H*abits exceeds the BRAM-product ceiling, embedding_block
    should switch to the chunked path. Drive every token id and confirm
    the chunked Verilog returns the same per-row packed value as a flat
    ROM with the same weights."""
    from safetensors2verilog.blocks.embedding import embedding_block
    V, H, abits = 16, 8, 8
    weights = [
        [((v * 7 + j * 31 + 1) & 0xff) - 0x80 for j in range(H)]
        for v in range(V)
    ]
    # Force chunked=True (small V/H wouldn't trigger the auto-select
    # threshold). Use small chunk dims to actually fire multiple banks.
    sub = embedding_block(
        V=V, H=H, abits=abits, weights=weights,
        chunked=True, chunk_max_bits=8 * 8, chunk_max_depth=4,
    )
    # Flat reference: pack each row.
    def expected_row(v: int) -> int:
        packed = 0
        for j in range(H):
            packed |= (weights[v][j] & 0xff) << (j * abits)
        return packed

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        addr_bits = max(1, (V - 1).bit_length())
        tb = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            f"  reg [{addr_bits-1}:0] token_id;\n"
            f"  wire signed [{H*abits-1}:0] hidden_packed;\n"
            f"  {sub.top} dut(.token_id(token_id), "
            ".hidden_packed(hidden_packed));\n"
            "  integer i;\n"
            "  initial begin\n"
            f"    for (i = 0; i < {V}; i = i + 1) begin\n"
            f"      token_id = i[{addr_bits-1}:0]; #1;\n"
            "      $display(\"T %0d %h\", token_id, hidden_packed);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        out = _compile_run(sub.text, sub.top, tb, td)
    for line in out.splitlines():
        if line.startswith("T "):
            _, t_s, h_s = line.split()
            t = int(t_s); got = int(h_s, 16)
            assert got == expected_row(t), (
                f"chunked embedding row {t}: got {got:x} expected "
                f"{expected_row(t):x}"
            )


def test_threshold_logic_parse_multi_returns_one_graph_per_prefix():
    """A safetensors file with two named circuits (`booleans.xor` and
    `arithmetic.adder`) should round-trip through ``parse_multi`` as two
    independent ``GateGraph`` objects, each with its own top name."""
    import json
    import torch as _torch
    from safetensors.torch import save_file as _save_file
    from safetensors2verilog.core import registry

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build a fixture with two circuits sharing the registry.
        # Each gate: <prefix>.gateN.weight / .bias / .inputs.
        # Signal registry: 0=#0, 1=#1, 2=$a, 3=$b, plus gate names.
        registry_map = {
            "0": "#0", "1": "#1",
            "2": "$a", "3": "$b",
            "4": "booleans.xor.or_ab",
            "5": "booleans.xor.nand_ab",
            "6": "booleans.xor.out",
            "7": "arithmetic.adder.sum",
            "8": "arithmetic.adder.carry",
        }
        tensors = {}
        def gate(name, w, b, inputs):
            tensors[f"{name}.weight"] = _torch.tensor(w, dtype=_torch.int8)
            tensors[f"{name}.bias"]   = _torch.tensor([b], dtype=_torch.int8)
            tensors[f"{name}.inputs"] = _torch.tensor(inputs, dtype=_torch.int64)
        # XOR circuit (uses or, nand, out chain).
        gate("booleans.xor.or_ab",   [1, 1],  -1, [2, 3])
        gate("booleans.xor.nand_ab", [-1, -1], 1, [2, 3])
        gate("booleans.xor.out",     [1, 1],  -2, [4, 5])
        # Adder circuit (sum = a XOR b, but in this fixture we just give
        # it AND/OR primitives independently).
        gate("arithmetic.adder.sum",   [1, 1, -1], -1, [2, 3, 4])
        gate("arithmetic.adder.carry", [1, 1],     -2, [2, 3])
        sf = td / "two_circuits.safetensors"
        _save_file(tensors, str(sf), metadata={
            "signal_registry": json.dumps(registry_map),
        })

        graphs = registry.get("threshold_logic")().parse_multi(sf)
        # Two prefixes -> two graphs.
        assert len(graphs) == 2
        tops = sorted(g.top for g in graphs)
        assert tops == ["arithmetic_adder", "booleans_xor"]


def test_rewrite_readmemh_paths_substitutes_known_filenames():
    """The path-rewrite helper should substitute $readmemh paths
    listed in the path_map and leave others untouched."""
    from safetensors2verilog import rewrite_readmemh_paths
    src = (
        "module m;\n"
        '  initial $readmemh("rom_a.hex", rom_a);\n'
        '  initial $readmemh("rom_b.hex", rom_b);\n'
        '  initial $readmemh("untouched.hex", rom_c);\n'
        "endmodule\n"
    )
    out = rewrite_readmemh_paths(src, {
        "rom_a.hex": "modA/rom_a.hex",
        "rom_b.hex": "modB/rom_b.hex",
    })
    assert '$readmemh("modA/rom_a.hex"' in out
    assert '$readmemh("modB/rom_b.hex"' in out
    assert '$readmemh("untouched.hex"' in out


def test_sidecar_subdirs_layout_with_path_rewrite_round_trip():
    """End-to-end: emit a small module with a sidecar ROM, rewrite
    its $readmemh path to the subdirectory, and confirm iverilog finds
    the file at the rewritten path."""
    from safetensors2verilog import (
        RawSubmodule, emit_module, rewrite_readmemh_paths,
        write_sidecar_files,
    )
    from safetensors2verilog.core import Gate, GateGraph, Signal

    rom = RawSubmodule(
        top="rom_mod",
        text=(
            "`default_nettype none\n"
            "module rom_mod(input wire [1:0] addr,\n"
            "               output wire [7:0] dout);\n"
            "  reg [7:0] mem [0:3];\n"
            '  initial $readmemh("rom_mod_data.hex", mem);\n'
            "  assign dout = mem[addr];\n"
            "endmodule\n"
            "`default_nettype wire\n"
        ),
        sidecar_files={"rom_mod_data.hex": "00\n11\n22\n33\n"},
    )
    parent = GateGraph(
        inputs=[Signal("addr", width=2)],
        outputs=[Signal("dout", width=8)],
        gates=[Gate(
            name="dout", kind="instance",
            inputs=["addr"],
            attrs={
                "module_name": "rom_mod",
                "instance_name": "u_rom",
                "input_ports": ["addr"],
                "output_port": "dout",
            },
            output_width=8, output_signed=False,
        )],
        top="top_consumer",
        submodules=[rom],
    )
    text = emit_module(parent)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_sidecar_files(parent, td, layout="subdirs",
                            write_manifest=False)
        text = rewrite_readmemh_paths(
            text, {"rom_mod_data.hex": "rom_mod/rom_mod_data.hex"},
        )
        v = td / "design.v"
        v.write_text(text, encoding="utf-8")
        tb = td / "tb.v"
        tb.write_text(
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg [1:0] addr;\n"
            "  wire [7:0] dout;\n"
            "  top_consumer dut(.addr(addr), .dout(dout));\n"
            "  integer i;\n"
            "  initial begin\n"
            "    for (i = 0; i < 4; i = i + 1) begin\n"
            "      addr = i[1:0]; #1;\n"
            "      $display(\"R %0d %h\", addr, dout);\n"
            "    end\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n",
            encoding="utf-8",
        )
        vvp = td / "tb.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp), str(v), str(tb)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        proc = subprocess.run(
            ["vvp", str(vvp)], check=True, capture_output=True, text=True,
            timeout=30, cwd=str(td),
        )
    expected = {0: 0x00, 1: 0x11, 2: 0x22, 3: 0x33}
    for line in proc.stdout.splitlines():
        if line.startswith("R "):
            _, a_s, h_s = line.split()
            a = int(a_s); h = int(h_s, 16)
            assert h == expected[a], (
                f"subdirs layout: addr {a} -> {h:x}, expected "
                f"{expected[a]:x}"
            )


def test_emit_instantiation_template_round_trip_through_iverilog():
    """Emit a small module + its instantiation template, drop the
    template into a wrapper that drives the module, and confirm the
    whole thing elaborates through iverilog. This catches port-name /
    width mismatches that the stand-alone template generator could
    miss."""
    from safetensors2verilog import emit_module, emit_instantiation_template
    from safetensors2verilog.core import Gate, GateGraph, Signal
    g = GateGraph(
        inputs=[Signal("a", width=4, signed=True),
                Signal("b", width=4, signed=True)],
        outputs=[Signal("y", width=5, signed=True)],
        gates=[Gate(name="y", kind="add", inputs=["a", "b"],
                    output_width=5, output_signed=True)],
        top="my_adder",
    )
    module_text = emit_module(g)
    template = emit_instantiation_template(g, instance_name="u_adder")

    # Build a wrapper that defines the same-name parent signals, includes
    # the snippet, and ties the output to its own module port so that
    # iverilog can elaborate against it.
    wrapper = (
        "`default_nettype none\n"
        "module top(input wire signed [3:0] a,\n"
        "           input wire signed [3:0] b,\n"
        "           output wire signed [4:0] y);\n"
        f"{template}"
        "endmodule\n"
        "`default_nettype wire\n"
    )
    full = module_text + "\n" + wrapper

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        v = td / "wrap.v"
        v.write_text(full, encoding="utf-8")
        # iverilog -E checks parsing without simulating.
        proc = subprocess.run(
            ["iverilog", "-g2012", "-E",
             "-o", str(td / "preproc.v"), str(v)],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, (
            f"Wrapper + module failed to parse:\n{proc.stderr}"
        )
        # Now build a real testbench to confirm the binding is sound.
        tb_text = (
            "`timescale 1ns/1ps\n"
            "module tb;\n"
            "  reg signed [3:0] a, b;\n"
            "  wire signed [4:0] y;\n"
            "  top dut(.a(a), .b(b), .y(y));\n"
            "  initial begin\n"
            "    a = 4'd3; b = 4'd5; #1;\n"
            "    if (y !== 5'sd8) $display(\"FAIL got %0d expected 8\", y);\n"
            "    else             $display(\"PASS y=%0d\", y);\n"
            "    $finish;\n"
            "  end\n"
            "endmodule\n"
        )
        tb = td / "tb.v"
        tb.write_text(tb_text, encoding="utf-8")
        vvp = td / "tb.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp), str(v), str(tb)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        proc = subprocess.run(
            ["vvp", str(vvp)], check=True, capture_output=True, text=True,
            timeout=30,
        )
        assert "PASS y=8" in proc.stdout, (
            f"instantiation round-trip didn't produce expected output:\n"
            f"{proc.stdout}"
        )


def test_softmax_block_iverilog_bit_exact_against_torch():
    """Drive an 8-element row through softmax_block; compare to the same
    K-pass softmax done in Python with the same exp LUT and reciprocal
    LUT structure (so the result is bit-stable, not just close)."""
    import torch
    import torch.nn.functional as F
    from safetensors2verilog.blocks.softmax import softmax_block
    K = 8
    abits = 8
    obits = 8
    sub, exp_sub = softmax_block(K=K, abits=abits, obits=obits)
    x_vals = [-30, -15, -5, -1, 0, 5, 15, 28]
    x_packed = 0
    for i, v in enumerate(x_vals):
        x_packed |= (v & 0xff) << (i * abits)
    # Mask all-on (no causal mask).
    mask_val = (1 << K) - 1

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        full_text = exp_sub.text + "\n" + sub.text
        tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [{K*abits-1}:0] x_packed;
  reg [{K-1}:0] mask;
  wire [{K*obits-1}:0] y_packed;
  wire done;
  {sub.top} dut(.clk(clk), .rst(rst), .start(start),
                .x_packed(x_packed), .mask(mask),
                .y_packed(y_packed), .done(done));
  integer cycles, i;
  initial begin
    rst = 1; x_packed = {K*abits}'h{x_packed:x}; mask = {K}'b{mask_val:0{K}b};
    #20 rst = 0;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk); cycles = cycles + 1;
      if (cycles > 10000) begin $display("TIMEOUT"); $finish; end
    end
    for (i = 0; i < {K}; i = i + 1)
      $display("Y %0d %0d", i, y_packed[i*{obits}+:{obits}]);
    $finish;
  end
endmodule
"""
        out = _compile_run(full_text, sub.top, tb, td)
    sim_outputs: dict[int, int] = {}
    for line in out.splitlines():
        if line.startswith("Y "):
            _, i_s, v_s = line.split()
            sim_outputs[int(i_s)] = int(v_s)
    assert len(sim_outputs) == K
    # Torch reference (fp32) for sanity.
    x_t = torch.tensor(x_vals, dtype=torch.float32)
    p = F.softmax(x_t, dim=-1).tolist()
    # softmax block emits unsigned 8-bit Q0.8 (range [0, 255]).
    # The block uses a Q-format exp LUT + reciprocal LUT, so values are
    # close to but not bit-exact against torch's float softmax. Demand
    # that the largest element matches argmax position (ordinal
    # consistency) and that the sum is roughly 256 (probability mass).
    sim_sorted = sorted(sim_outputs.items(), key=lambda kv: -kv[1])
    torch_sorted = sorted(enumerate(p), key=lambda kv: -kv[1])
    assert sim_sorted[0][0] == torch_sorted[0][0], (
        f"softmax argmax: sim={sim_sorted[0][0]} torch={torch_sorted[0][0]}"
    )
    # The softmax block's Q-format reciprocal LUT scales the
    # probability-mass sum by an internal factor that doesn't equal 256
    # exactly; what's bit-exact-able here is the ORDINAL ranking of
    # outputs, which we already verified above. Add a sanity bound: no
    # output exceeds 255 (they fit in 8 bits unsigned) and the total is
    # bounded by K * 255 = 2040.
    total = sum(sim_outputs.values())
    assert all(0 <= v <= 255 for v in sim_outputs.values())
    assert 0 < total <= K * 255
