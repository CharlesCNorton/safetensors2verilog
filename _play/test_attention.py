"""Bit-exact iverilog test for multi_head_attention.

Drives 4 tokens through a tiny attention block (H=2, KV=1, D=4, MAX_SEQ=4)
sequentially, capturing the per-token output. A Python reference replicates
the hardware's fixed-point math (score shift, exp/recip LUTs of the inner
softmax, V accumulation, output saturation) and asserts bit-exact match
against every output element of every token.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import Gate, GateGraph, Signal, emit_module
from safetensors2verilog.blocks.attention import attention_step_block


# ---- Test config -----------------------------------------------------------
H, KV, D, MAX_SEQ = 2, 1, 4, 4
ABITS = 8
SCORE_BITS = 2 * ABITS + max(1, D.bit_length()) + 4   # default
SOFTMAX_OBITS = 8
OUT_ABITS = 8
SCORE_SHIFT = 0
OUT_SHIFT = 8

# softmax_block parameters (must match)
EXP_IN_BITS = 8
EXP_OUT_BITS = 12
EXP_IN_Q_FRAC_BITS = 4
EXP_IN_CLAMP = (-16.0, 0.0)
RECIP_LUT_BITS = 12
RECIP_OUT_FRAC_BITS = 16
SOFTMAX_OUTPUT_SHIFT = 8


# ---- Python reference ------------------------------------------------------
def _exp_lut() -> list[int]:
    out_max = (1 << EXP_OUT_BITS) - 1
    lut = []
    for raw in range(1 << EXP_IN_BITS):
        sint = raw - (1 << EXP_IN_BITS) if raw & (1 << (EXP_IN_BITS-1)) else raw
        x_real = sint / (1 << EXP_IN_Q_FRAC_BITS)
        x_real = max(EXP_IN_CLAMP[0], min(EXP_IN_CLAMP[1], x_real))
        v = math.exp(x_real)
        lut.append(max(0, min(out_max, round(v * out_max))))
    return lut


def _recip_lut() -> list[int]:
    n = 1 << RECIP_LUT_BITS
    lut = [0] + [
        max(0, min((1 << 24) - 1, round((1.0 / idx) * (1 << RECIP_OUT_FRAC_BITS))))
        for idx in range(1, n)
    ]
    return lut


EXP_LUT = _exp_lut()
RECIP_LUT = _recip_lut()


def py_softmax(scores: list[int], mask: list[int]) -> list[int]:
    """Replicate softmax_block's fixed-point math bit-exactly.

    scores: list of signed score_bits ints (length MAX_SEQ).
    mask:   list of 0/1 (length MAX_SEQ).
    Returns: list of unsigned softmax_obits ints (probabilities Q0.softmax_obits).
    """
    abits = SCORE_BITS
    half = (1 << (abits - 1)) - 1

    # Find max over masked entries
    cur_max = -((1 << (abits - 1)) - 1)
    for v, m in zip(scores, mask):
        if m and v > cur_max:
            cur_max = v

    exp_y_buf = []
    sum_exp = 0
    for v, m in zip(scores, mask):
        x_diff = v - cur_max
        # clamp to exp_in_bits signed range
        e_half = (1 << (EXP_IN_BITS - 1)) - 1
        x_diff = max(-e_half, min(e_half, x_diff))
        raw = x_diff & ((1 << EXP_IN_BITS) - 1)
        ey = EXP_LUT[raw] if m else 0
        exp_y_buf.append(ey)
        sum_exp += ey

    n_recip = 1 << RECIP_LUT_BITS
    recip_idx = sum_exp if sum_exp < n_recip else n_recip - 1
    recip_val = RECIP_LUT[recip_idx]
    out_max_y = (1 << SOFTMAX_OBITS) - 1
    out = []
    for ey in exp_y_buf:
        divprod = ey * recip_val
        divshift = divprod >> SOFTMAX_OUTPUT_SHIFT
        out.append(min(out_max_y, divshift))
    return out


def _arsh(x: int, n: int) -> int:
    return x >> n   # Python's >> on int is arithmetic for negatives


def _sat(x: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return max(lo, min(hi, x))


def py_attention_step(q, k_new, v_new, position,
                      k_cache, v_cache):
    """Bit-exact attention step. Mutates k_cache, v_cache.

    q, k_new, v_new: lists of ints in the appropriate shapes.
    Returns: out (list of H*D ints in [-128, 127]).
    """
    # Write current K, V at row `position`.
    k_cache[position] = list(k_new)
    v_cache[position] = list(v_new)

    out = [0] * (H * D)
    group = H // KV
    for h in range(H):
        h_kv = h // group
        # 1. Scores
        scores = []
        for k_pos in range(MAX_SEQ):
            acc = 0
            for d in range(D):
                q_v = q[h * D + d]
                # K_cache[k_pos] has KV * D elements; index for h_kv, d
                k_v = k_cache[k_pos][h_kv * D + d] if k_cache[k_pos] else 0
                acc += q_v * k_v
            scores.append(_arsh(acc, SCORE_SHIFT))
        # 2. Mask + softmax
        mask = [1 if k_pos <= position else 0 for k_pos in range(MAX_SEQ)]
        attn = py_softmax(scores, mask)
        # 3. V output
        for d in range(D):
            acc = 0
            for k_pos in range(MAX_SEQ):
                v_v = v_cache[k_pos][h_kv * D + d] if v_cache[k_pos] else 0
                acc += attn[k_pos] * v_v
            out[h * D + d] = _sat(_arsh(acc, OUT_SHIFT), OUT_ABITS)
    return out


# ---- Verilog harness + iverilog runner -------------------------------------
def _pack_signed(values, bits):
    mask = (1 << bits) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (i * bits)
    return out


def _unpack_signed(packed, K, bits):
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    o = []
    for i in range(K):
        v = (packed >> (i * bits)) & mask
        if v & sign_bit:
            v -= 1 << bits
        o.append(v)
    return o


def main() -> int:
    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; aborting.")
        return 2

    sub, deps = attention_step_block(
        num_q_heads=H, num_kv_heads=KV, head_dim=D, max_seq=MAX_SEQ,
        abits=ABITS, score_bits=SCORE_BITS, softmax_obits=SOFTMAX_OBITS,
        out_abits=OUT_ABITS, score_shift=SCORE_SHIFT, out_shift=OUT_SHIFT,
    )

    pos_bits = max(1, (MAX_SEQ - 1).bit_length() + 1)

    parent = GateGraph(
        inputs=[
            Signal("clk"), Signal("rst"), Signal("start"),
            Signal("position", width=pos_bits, signed=False),
            Signal("q_packed", width=H * D * ABITS, signed=True),
            Signal("k_new_packed", width=KV * D * ABITS, signed=True),
            Signal("v_new_packed", width=KV * D * ABITS, signed=True),
        ],
        outputs=[
            Signal("done", width=1),
            Signal("out_packed", width=H * D * OUT_ABITS, signed=True),
        ],
        gates=[
            Gate(name="done", kind="extern_wire", output_width=1),
            Gate(
                name="out_packed", kind="instance",
                inputs=["clk", "rst", "start", "position",
                        "q_packed", "k_new_packed", "v_new_packed"],
                attrs={
                    "module_name": sub.top, "instance_name": "att_inst",
                    "input_ports": ["clk", "rst", "start", "position",
                                    "q_packed", "k_new_packed", "v_new_packed"],
                    "output_port": "out_packed",
                    "extra_output_ports": [("done", "done")],
                },
                output_width=H * D * OUT_ABITS, output_signed=True,
            ),
        ],
        top="att_test",
        submodules=[sub] + deps,
    )

    text = emit_module(parent)
    print(f"emitted {len(text.splitlines())} lines of Verilog")

    import random
    random.seed(0)

    # Generate 4 tokens of Q/K/V data
    tokens = []
    for t in range(4):
        q   = [random.randint(-30, 30) for _ in range(H * D)]
        k_n = [random.randint(-30, 30) for _ in range(KV * D)]
        v_n = [random.randint(-30, 30) for _ in range(KV * D)]
        tokens.append((q, k_n, v_n))

    # Compute Python reference (mutates cache)
    k_cache = [[]] * MAX_SEQ
    v_cache = [[]] * MAX_SEQ
    expected = []
    for t, (q, k_n, v_n) in enumerate(tokens):
        ref = py_attention_step(q, k_n, v_n, t, k_cache, v_cache)
        expected.append(ref)

    # Build a testbench that drives 4 tokens sequentially and captures outputs.
    drives = []
    for t, (q, k_n, v_n) in enumerate(tokens):
        q_h = _pack_signed(q, ABITS)
        k_h = _pack_signed(k_n, ABITS)
        v_h = _pack_signed(v_n, ABITS)
        drives.append(f"""\
    @(negedge clk);
      position     = {pos_bits}'d{t};
      q_packed     = {H*D*ABITS}'h{q_h:x};
      k_new_packed = {KV*D*ABITS}'h{k_h:x};
      v_new_packed = {KV*D*ABITS}'h{v_h:x};
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 2000) begin $display("TIMEOUT t={t}"); $finish; end
    end
    $display("R {t} cyc=%0d %h", cycles, out_packed);""")

    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{pos_bits-1}:0] position;\n"
        f"  reg [{H*D*ABITS-1}:0] q_packed;\n"
        f"  reg [{KV*D*ABITS-1}:0] k_new_packed, v_new_packed;\n"
        "  wire done;\n"
        f"  wire [{H*D*OUT_ABITS-1}:0] out_packed;\n"
        "  att_test dut(.clk(clk), .rst(rst), .start(start),\n"
        "               .position(position), .q_packed(q_packed),\n"
        "               .k_new_packed(k_new_packed), "
        ".v_new_packed(v_new_packed),\n"
        "               .done(done), .out_packed(out_packed));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        + "\n".join(drives) +
        "\n    $finish;\n"
        "  end\nendmodule\n"
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        debug = Path(r"D:\safetensors2verilog\_play\att_debug")
        debug.mkdir(exist_ok=True)
        (td / "dut.v").write_text(text, encoding="utf-8")
        (td / "tb.v").write_text(tb, encoding="utf-8")
        (debug / "dut.v").write_text(text, encoding="utf-8")
        (debug / "tb.v").write_text(tb, encoding="utf-8")
        vvp = td / "out.vvp"
        proc = subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp),
             str(td / "dut.v"), str(td / "tb.v")],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("iverilog failed:")
            print(proc.stderr)
            return 1
        proc = subprocess.run(
            ["vvp", str(vvp)], capture_output=True, text=True, timeout=120,
        )
        log = proc.stdout
        (debug / "log.txt").write_text(log, encoding="utf-8")

    by_t = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            by_t[int(toks[1])] = (int(toks[2].split("=")[1]),
                                  int(toks[3], 16))

    fails = []
    for t in range(4):
        if t not in by_t:
            print(f"  t={t} MISSING from sim output")
            fails.append((t, None, expected[t]))
            continue
        cycles, out_packed = by_t[t]
        sim = _unpack_signed(out_packed, H * D, OUT_ABITS)
        ok = sim == expected[t]
        status = "OK" if ok else "FAIL"
        print(f"  t={t} cycles={cycles} {status}")
        if not ok:
            fails.append((t, sim, expected[t]))

    print(f"\n{4 - len(fails)}/4 tokens bit-exact")
    if fails:
        for f in fails[:1]:
            print(f"first fail: t={f[0]}\n  sim={f[1]}\n  exp={f[2]}")
        print(f"  [debug] dut.v + tb.v + log.txt in {Path(r'D:\safetensors2verilog\_play\att_debug')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
