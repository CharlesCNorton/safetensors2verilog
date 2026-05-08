"""End-to-end iverilog tests for the sequential primitives:
  silu, rms_norm, softmax, embedding (combinational ROM), kv_cache.

Drives clk, rst, start; waits for done; compares y_packed against Python
references that use the same fixed-point math.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from safetensors2verilog import (
    Gate, GateGraph, Signal, emit_module,
)
from safetensors2verilog.blocks.silu import silu_block
from safetensors2verilog.blocks.rms_norm import rms_norm_block
from safetensors2verilog.blocks.softmax import softmax_block
from safetensors2verilog.blocks.embedding import embedding_block
from safetensors2verilog.blocks.kv_cache import kv_cache_block

PASS, FAIL = 0, 0


def _check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if ok: PASS += 1
    else:  FAIL += 1


def _run_iverilog(td: Path, dut_text: str, tb_text: str) -> str:
    (td / "dut.v").write_text(dut_text, encoding="utf-8")
    (td / "tb.v").write_text(tb_text, encoding="utf-8")
    vvp = td / "out.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp),
         str(td / "dut.v"), str(td / "tb.v")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"iverilog failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True,
        timeout=60,
    )
    return proc.stdout


def _make_seq_harness(
    sub_top: str,
    instance_name: str,
    in_packed_width: int,
    out_packed_width: int,
    extra_inputs: list[tuple[str, int]] = (),
    has_mask: tuple[str, int] | None = None,
    submodules: list = (),
) -> tuple[GateGraph, dict]:
    """Build a parent module exposing (clk, rst, start, x_packed[, mask])
    -> (done, y_packed). `extra_inputs` is a list of (name, width)."""
    inputs = [Signal("clk"), Signal("rst"), Signal("start")]
    inputs.append(Signal("x_packed", width=in_packed_width, signed=False))
    inst_inputs = ["clk", "rst", "start", "x_packed"]
    inst_input_ports = ["clk", "rst", "start", "x_packed"]
    if has_mask:
        mname, mwidth = has_mask
        inputs.append(Signal(mname, width=mwidth, signed=False))
        inst_inputs.append(mname)
        inst_input_ports.append("mask")

    gates = [
        Gate(name="done", kind="extern_wire", output_width=1),
        Gate(
            name="y_packed", kind="instance",
            inputs=inst_inputs,
            attrs={
                "module_name": sub_top, "instance_name": instance_name,
                "input_ports": inst_input_ports,
                "output_port": "y_packed",
                "extra_output_ports": [("done", "done")],
            },
            output_width=out_packed_width, output_signed=False,
        ),
    ]
    parent = GateGraph(
        inputs=inputs,
        outputs=[
            Signal("done", width=1),
            Signal("y_packed", width=out_packed_width, signed=False),
        ],
        gates=gates,
        top="dut_test",
        submodules=list(submodules),
    )
    return parent, {}


def _seq_testbench(
    *, in_packed_width: int, out_packed_width: int,
    cases_x_int: list[int],   # one packed-int per case
    case_extras: list[dict] | None = None,
    mask_signal: tuple[str, int] | None = None,
    case_masks: list[int] | None = None,
    timeout_cycles: int = 200,
) -> str:
    """Generate a TB that drives N cases through the sequential block."""
    case_blocks: list[str] = []
    for ci, x_int in enumerate(cases_x_int):
        drives = [
            f"      x_packed = {in_packed_width}'h{x_int:x};"
        ]
        if mask_signal and case_masks:
            mname, mw = mask_signal
            drives.append(f"      {mname} = {mw}'h{case_masks[ci]:x};")
        case_blocks.append(f"""\
    @(posedge clk);
{chr(10).join(drives)}
      start <= 1;
    @(posedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > {timeout_cycles}) begin $display("TIMEOUT case {ci}"); $finish; end
    end
    $display("R {ci} %0d %h", cycles, y_packed);""")

    mask_decl = ""
    if mask_signal:
        mname, mw = mask_signal
        mask_decl = f"  reg [{mw-1}:0] {mname};\n"

    return (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{in_packed_width-1}:0] x_packed;\n"
        + mask_decl +
        f"  wire [{out_packed_width-1}:0] y_packed;\n"
        "  wire done;\n"
        "  dut_test dut(.clk(clk), .rst(rst), .start(start),\n"
        "               .x_packed(x_packed),\n"
        + (f"               .{mask_signal[0]}({mask_signal[0]}),\n"
           if mask_signal else "") +
        "               .done(done), .y_packed(y_packed));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        + "\n".join(case_blocks) +
        "\n    $finish;\n  end\nendmodule\n"
    )


def _parse_results(log: str, n_cases: int) -> dict[int, tuple[int, int]]:
    """Returns {ci: (cycles, y_packed_int_hex_value)}."""
    out = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            ci = int(toks[1])
            cycles = int(toks[2])
            y_hex = toks[3]
            out[ci] = (cycles, int(y_hex, 16))
    return out


def _pack_signed(values: list[int], bits: int) -> int:
    mask = (1 << bits) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (i * bits)
    return out


def _unpack_signed(packed: int, K: int, bits: int) -> list[int]:
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    out = []
    for i in range(K):
        v = (packed >> (i * bits)) & mask
        if v & sign_bit:
            v -= 1 << bits
        out.append(v)
    return out


def _unpack_unsigned(packed: int, K: int, bits: int) -> list[int]:
    mask = (1 << bits) - 1
    return [(packed >> (i * bits)) & mask for i in range(K)]


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------
def test_silu():
    print("\n== silu ==")
    K, ABITS, OBITS = 8, 8, 8
    silu, sig = silu_block(
        K=K, abits=ABITS, obits=OBITS,
        sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8, output_shift=8,
    )
    parent, _ = _make_seq_harness(
        sub_top=silu.top, instance_name="silu_inst",
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        submodules=[silu, sig],
    )
    text = emit_module(parent)

    import random
    random.seed(0)
    cases_int = []
    expected = []
    for _ in range(5):
        x = [random.randint(-127, 127) for _ in range(K)]
        cases_int.append(_pack_signed(x, ABITS))
        # Python ref: y[i] = sat_int8((x[i] * sig_lut(x[i])) >>> 8)
        # Python's >> on a signed int rounds toward -inf, matching Verilog
        # >>> on a signed wire.
        y = []
        for v in x:
            x_real = max(-8.0, min(8.0, v / 16.0))
            s = 1.0 / (1.0 + math.exp(-x_real))
            sig_int = max(0, min(255, round(s * 256)))
            prod = v * sig_int
            shifted = prod >> 8
            shifted = max(-128, min(127, shifted))
            y.append(shifted)
        expected.append(y)

    tb = _seq_testbench(
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        cases_x_int=cases_int, timeout_cycles=64,
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    results = _parse_results(log, len(cases_int))
    fails = []
    for ci in range(len(cases_int)):
        cycles, y_packed = results[ci]
        sim = _unpack_signed(y_packed, K, OBITS)
        if sim != expected[ci]:
            fails.append((ci, sim, expected[ci]))
    _check(f"silu K={K}, {len(cases_int)} cases",
           len(fails) == 0,
           f"first fail: {fails[0]}" if fails else
           f"K+2={K+2} cycles per inference")


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------
def _python_rsqrt_lut(in_bits: int, out_bits: int, out_frac_bits: int,
                      lut_idx_bits: int = 8) -> callable:
    """Reproduces the rsqrt block's algorithm bit-exactly in Python."""
    n_entries = 1 << lut_idx_bits
    lut = [
        max(0, min((1 << out_bits) - 1,
                   round((1.0 / math.sqrt(1.0 + i / n_entries)) *
                         (1 << out_frac_bits))))
        for i in range(n_entries)
    ]
    sqrt_half_q = round(math.sqrt(0.5) * (1 << out_frac_bits))
    out_max = (1 << out_bits) - 1

    def rsqrt(x: int) -> int:
        if x == 0:
            return out_max
        # Find leading-1 position p
        p = x.bit_length() - 1
        # Normalise: shift left so MSB is at bit (in_bits - 1)
        nlz = in_bits - 1 - p
        x_norm = (x << nlz) & ((1 << in_bits) - 1)
        # Top lut_idx_bits below the leading 1 (which sits at in_bits-1)
        idx = (x_norm >> (in_bits - 1 - lut_idx_bits)) & (n_entries - 1)
        lut_val = lut[idx]
        half_p = p >> 1
        p_odd = p & 1
        shifted = lut_val >> half_p
        if p_odd:
            product = shifted * sqrt_half_q
            scaled = (product >> out_frac_bits) & out_max
            return scaled
        return shifted

    return rsqrt


def test_rms_norm():
    print("\n== rms_norm ==")
    K, ABITS, OBITS = 8, 8, 8
    GBITS = 16
    EPS_Q = 16
    eps = 1e-5
    rsqrt_in_bits = 2 * ABITS + max(1, K.bit_length()) + 4
    rsqrt_out_frac_bits = 14
    output_shift = 14

    # Gamma chosen at 1.0 in Q1.15 (gamma_int = 32768 wraps; use 16383 for safety)
    gamma_int = [16384] * K   # ~1.0 in Q14
    rms, rsqrt = rms_norm_block(
        K=K, gamma_int=gamma_int, gamma_bits=GBITS,
        abits=ABITS, obits=OBITS, eps=eps, eps_q=EPS_Q,
        rsqrt_in_bits=rsqrt_in_bits,
        rsqrt_out_bits=16, rsqrt_out_frac_bits=rsqrt_out_frac_bits,
        output_shift=output_shift,
    )
    parent, _ = _make_seq_harness(
        sub_top=rms.top, instance_name="rms_inst",
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        submodules=[rms, rsqrt],
    )
    text = emit_module(parent)

    # Bit-exact Python reference: replicates the hardware's fixed-point ops.
    py_rsqrt = _python_rsqrt_lut(rsqrt_in_bits, 16, rsqrt_out_frac_bits)
    eps_int = round(eps * (1 << EPS_Q))
    K_eps_int = K * eps_int

    out_lo = -(1 << (OBITS - 1))
    out_hi = (1 << (OBITS - 1)) - 1

    def py_rms(x_int_list: list[int]) -> list[int]:
        sum_sq = sum((v * v) & ((1 << (2*ABITS)) - 1) for v in x_int_list)
        sum_sq_eps = sum_sq + K_eps_int
        rsqrt_val = py_rsqrt(sum_sq_eps)
        out = []
        for v, g in zip(x_int_list, gamma_int):
            xg = v * g
            xgr = xg * rsqrt_val
            shifted = xgr >> output_shift
            out.append(max(out_lo, min(out_hi, shifted)))
        return out

    import random
    random.seed(0)
    cases_int = []
    expected = []
    for _ in range(5):
        x = [random.randint(-30, 30) for _ in range(K)]
        cases_int.append(_pack_signed(x, ABITS))
        expected.append(py_rms(x))

    tb = _seq_testbench(
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        cases_x_int=cases_int, timeout_cycles=64,
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    results = _parse_results(log, len(cases_int))

    fails = []
    for ci in range(len(cases_int)):
        cycles, y_packed = results[ci]
        sim = _unpack_signed(y_packed, K, OBITS)
        if sim != expected[ci]:
            fails.append((ci, sim, expected[ci]))
    _check(f"rms_norm K={K}: bit-exact vs Python ref",
           len(fails) == 0,
           f"first fail: case {fails[0][0]} sim={fails[0][1]} exp={fails[0][2]}"
           if fails else f"2K+3={2*K+3} cycles per inference")


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def test_softmax():
    print("\n== softmax ==")
    K, ABITS, OBITS = 8, 8, 8
    sm, exp = softmax_block(
        K=K, abits=ABITS, obits=OBITS,
        exp_in_bits=8, exp_out_bits=12, exp_in_q_frac_bits=4,
        exp_in_clamp=(-16.0, 0.0),
        recip_lut_bits=12, recip_out_frac_bits=16, output_shift=8,
    )
    parent, _ = _make_seq_harness(
        sub_top=sm.top, instance_name="sm_inst",
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        has_mask=("mask", K), submodules=[sm, exp],
    )
    text = emit_module(parent)

    # Bit-exact Python reference replicating the hardware's fixed-point ops.
    EXP_IN_Q_FRAC_BITS = 4
    EXP_IN_BITS = 8
    EXP_OUT_BITS = 12
    RECIP_LUT_BITS = 12
    RECIP_OUT_FRAC_BITS = 16
    OUTPUT_SHIFT = 8

    # exp LUT contents
    exp_lut = []
    out_max_exp = (1 << EXP_OUT_BITS) - 1
    for raw in range(1 << EXP_IN_BITS):
        sint = raw - (1 << EXP_IN_BITS) if raw & (1 << (EXP_IN_BITS - 1)) else raw
        x_real = sint / (1 << EXP_IN_Q_FRAC_BITS)
        x_real = max(-16.0, min(0.0, x_real))
        v = math.exp(x_real)
        exp_lut.append(max(0, min(out_max_exp, round(v * out_max_exp))))

    # recip LUT
    n_recip = 1 << RECIP_LUT_BITS
    recip_lut = [0] + [
        max(0, min((1 << 24) - 1, round((1.0 / idx) * (1 << RECIP_OUT_FRAC_BITS))))
        for idx in range(1, n_recip)
    ]

    out_max_y = (1 << OBITS) - 1

    def py_softmax(x_int: list[int], mask_int: int) -> list[int]:
        # Pass max
        cur_max = -((1 << (ABITS - 1)) - 1)
        for i, v in enumerate(x_int):
            if (mask_int >> i) & 1 and v > cur_max:
                cur_max = v
        # Pass exp + sum
        exp_y_buf = []
        sum_exp = 0
        for i, v in enumerate(x_int):
            x_diff_wide = v - cur_max
            # clamp to exp_in_bits signed range
            half = (1 << (EXP_IN_BITS - 1)) - 1
            x_diff = max(-half, x_diff_wide)
            x_diff = min(half, x_diff)
            # decode signed -> unsigned bit pattern (matching Verilog x[7:0])
            raw = x_diff & ((1 << EXP_IN_BITS) - 1)
            ey = exp_lut[raw] if (mask_int >> i) & 1 else 0
            exp_y_buf.append(ey)
            sum_exp += ey
        # Invert
        recip_idx = sum_exp if sum_exp < n_recip else n_recip - 1
        recip_val = recip_lut[recip_idx]
        # Pass div
        out = []
        for ey in exp_y_buf:
            divprod = ey * recip_val
            divshift = divprod >> OUTPUT_SHIFT
            out.append(min(out_max_y, divshift))
        return out

    import random
    random.seed(0)
    cases_int, masks, expected = [], [], []
    for trial in range(4):
        x = [random.randint(-32, 32) for _ in range(K)]
        cases_int.append(_pack_signed(x, ABITS))
        if trial == 1:
            mask = (1 << (K // 2)) - 1
        else:
            mask = (1 << K) - 1
        masks.append(mask)
        expected.append(py_softmax(x, mask))

    tb = _seq_testbench(
        in_packed_width=K * ABITS, out_packed_width=K * OBITS,
        cases_x_int=cases_int,
        mask_signal=("mask", K), case_masks=masks,
        timeout_cycles=128,
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    results = _parse_results(log, len(cases_int))
    fails = []
    for ci in range(len(cases_int)):
        cycles, y_packed = results[ci]
        sim = _unpack_unsigned(y_packed, K, OBITS)
        if sim != expected[ci]:
            fails.append((ci, sim, expected[ci]))
    _check(f"softmax K={K}: bit-exact vs Python ref",
           len(fails) == 0,
           f"first fail: case {fails[0][0]} sim={fails[0][1]} exp={fails[0][2]}"
           if fails else f"3K+5={3*K+5} cycles per inference")


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
def test_embedding():
    print("\n== embedding ==")
    V, H, ABITS = 8, 4, 8
    weights = [
        [(v * 11 + h * 3 - 17) % 256 - 128 for h in range(H)]
        for v in range(V)
    ]
    sub = embedding_block(V=V, H=H, abits=ABITS, weights=weights, inline_init=True)

    addr_bits = (V - 1).bit_length()
    parent = GateGraph(
        inputs=[Signal("token_id", width=addr_bits, signed=False)],
        outputs=[Signal("hidden_packed", width=H * ABITS, signed=False)],
        gates=[Gate(
            name="hidden_packed", kind="instance",
            inputs=["token_id"],
            attrs={
                "module_name": sub.top, "instance_name": "emb_inst",
                "input_ports": ["token_id"], "output_port": "hidden_packed",
            },
            output_width=H * ABITS, output_signed=False,
        )],
        top="emb_test", submodules=[sub],
    )
    text = emit_module(parent)

    drives = []
    for v in range(V):
        drives.append(
            f"    token_id = {addr_bits}'d{v}; #1; "
            f"$display(\"R {v} %h\", hidden_packed);"
        )
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  reg [{addr_bits-1}:0] token_id;\n"
        f"  wire [{H*ABITS-1}:0] hidden_packed;\n"
        "  emb_test dut(.token_id(token_id), .hidden_packed(hidden_packed));\n"
        "  initial begin\n"
        + "\n".join(drives) +
        "\n    $finish;\n  end\nendmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = _run_iverilog(td, text, tb)
    by_idx = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            by_idx[int(toks[1])] = int(toks[2], 16)

    fails = []
    for v in range(V):
        sim = _unpack_signed(by_idx[v], H, ABITS)
        if sim != weights[v]:
            fails.append((v, sim, weights[v]))
    _check(f"embedding V={V}, H={H}",
           len(fails) == 0,
           f"first fail: {fails[0]}" if fails else "")


# ---------------------------------------------------------------------------
# kv_cache
# ---------------------------------------------------------------------------
def test_kv_cache():
    print("\n== kv_cache ==")
    MAX_SEQ, ROW_BITS = 8, 16
    sub = kv_cache_block(max_seq=MAX_SEQ, row_bits=ROW_BITS)
    pos_bits = (MAX_SEQ - 1).bit_length()

    # Wrap in a passthrough harness module
    parent = GateGraph(
        inputs=[
            Signal("clk"),
            Signal("write_pos", width=pos_bits, signed=False),
            Signal("write_data", width=ROW_BITS, signed=False),
            Signal("write_en", width=1, signed=False),
            Signal("read_pos", width=pos_bits, signed=False),
        ],
        outputs=[Signal("read_data", width=ROW_BITS, signed=False)],
        gates=[Gate(
            name="read_data", kind="instance",
            inputs=["clk", "write_pos", "write_data", "write_en", "read_pos"],
            attrs={
                "module_name": sub.top, "instance_name": "kv_inst",
                "input_ports": ["clk", "write_pos", "write_data",
                                "write_en", "read_pos"],
                "output_port": "read_data",
            },
            output_width=ROW_BITS, output_signed=False,
        )],
        top="kv_test", submodules=[sub],
    )
    text = emit_module(parent)

    # Drive: write 8 values, then read all 8.
    # Setting inputs on negedge keeps them stable across the next posedge,
    # avoiding the same-time-step ordering race between the initial block
    # and the kv_cache's always @(posedge clk).
    write_data = [0xCAFE, 0xDEAD, 0xBEEF, 0xFACE, 0x1234, 0x5678, 0x9ABC, 0xDEF0]
    drives = []
    for p in range(MAX_SEQ):
        drives.append(
            f"    @(negedge clk); write_en = 1; "
            f"write_pos = {pos_bits}'d{p}; "
            f"write_data = 16'h{write_data[p]:x};"
        )
    drives.append("    @(negedge clk); write_en = 0;")
    drives.append("    @(posedge clk);")   # let last write commit
    for p in range(MAX_SEQ):
        drives.append(
            f"    @(negedge clk); read_pos = {pos_bits}'d{p};"
        )
        drives.append(
            f"    @(posedge clk); #1; "
            f"$display(\"R {p} %h\", read_data);"
        )
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        f"  reg [{pos_bits-1}:0] write_pos, read_pos;\n"
        f"  reg [{ROW_BITS-1}:0] write_data;\n"
        "  reg write_en;\n"
        f"  wire [{ROW_BITS-1}:0] read_data;\n"
        "  kv_test dut(.clk(clk), .write_pos(write_pos),\n"
        "              .write_data(write_data), .write_en(write_en),\n"
        "              .read_pos(read_pos), .read_data(read_data));\n"
        "  initial begin\n"
        "    write_en = 0; write_pos = 0; write_data = 0; read_pos = 0;\n"
        "    @(posedge clk);\n"
        + "\n".join(drives) +
        "\n    $finish;\n  end\nendmodule\n"
    )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Persist files for inspection on failure
        debug_dir = Path(r"D:\safetensors2verilog\_play\kv_debug")
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / "dut.v").write_text(text, encoding="utf-8")
        (debug_dir / "tb.v").write_text(tb, encoding="utf-8")
        log = _run_iverilog(td, text, tb)
        (debug_dir / "log.txt").write_text(log, encoding="utf-8")
    by_idx = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            try:
                by_idx[int(toks[1])] = int(toks[2], 16)
            except ValueError:
                by_idx[int(toks[1])] = -1   # 'xxxx' or similar
    fails = []
    for p in range(MAX_SEQ):
        if by_idx.get(p) != write_data[p]:
            fails.append((p, by_idx.get(p), write_data[p]))
    _check(f"kv_cache MAX_SEQ={MAX_SEQ}, ROW={ROW_BITS} bits",
           len(fails) == 0,
           f"first fail: {fails[0]}" if fails else "")
    if fails:
        print(f"     [debug] dut.v + tb.v + log.txt in {debug_dir}")


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2
    test_silu()
    test_rms_norm()
    test_softmax()
    test_embedding()
    test_kv_cache()
    print(f"\nTotal: {PASS} passed, {FAIL} failed.")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
