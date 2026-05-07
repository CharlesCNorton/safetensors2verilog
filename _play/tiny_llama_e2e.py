"""Tiny LLaMA forward pass, hand-wired GateGraph + iverilog vs Python ref.

Shape: 1 layer, hidden=8, num_q_heads=2, num_kv_heads=1, head_dim=4,
intermediate=16, vocab=8, max_seq=4, abits=8 (signed Q4.4 throughout).

Per-block output_shift values are picked to keep activations in int8 range
for random integer weights/activations. The Python reference replicates
each block's fixed-point math bit-for-bit so the comparison is exact.

This is item 56 from TODO.md and the prerequisite for item 55 (the generic
hf_llama frontend, which is just this construction parameterised).
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from safetensors2verilog import (
    Gate, GateGraph, RawSubmodule, Signal, emit_module,
)
from safetensors2verilog.blocks.matmul import matmul_seq_block, matmul_seq_invoke
from safetensors2verilog.blocks.rms_norm import rms_norm_block
from safetensors2verilog.blocks.rope import rope_block
from safetensors2verilog.blocks.silu import silu_block
from safetensors2verilog.blocks.embedding import embedding_block
from safetensors2verilog.blocks.argmax import argmax_block
from safetensors2verilog.blocks.attention import attention_step_block
from safetensors2verilog.blocks.requantize import requantize_block

# ---------------- Shape ----------------
H = 2          # num_q_heads
KV = 1         # num_kv_heads
D = 4          # head_dim
HID = H * D    # hidden = 8
INTER = 16     # MLP intermediate
VOCAB = 8
MAX_SEQ = 4
ABITS = 8
WBITS = 8
ROPE_THETA = 10000.0


# ---------------- Random weights ----------------
random.seed(0)


def _randint_table(rows, cols, lo=-32, hi=32):
    return [[random.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def _randint_vec(n, lo=-32, hi=32):
    return [random.randint(lo, hi) for _ in range(n)]


# Embedding: vocab x hidden
embed_W = _randint_table(VOCAB, HID, -64, 64)

# RMSNorm gammas (Q14 fixed-point, ~1.0)
gamma1 = [16384] * HID
gamma2 = [16384] * HID
gamma_final = [16384] * HID

# Q/K/V projections: [out, in]
W_q = _randint_table(HID, HID)            # [8, 8]
W_k = _randint_table(KV * D, HID)         # [4, 8]
W_v = _randint_table(KV * D, HID)         # [4, 8]
W_o = _randint_table(HID, HID)            # [8, 8]

# MLP weights
W_gate = _randint_table(INTER, HID)       # [16, 8]
W_up   = _randint_table(INTER, HID)       # [16, 8]
W_down = _randint_table(HID, INTER)       # [8, 16]

# lm_head: [vocab, hidden]
W_lm = _randint_table(VOCAB, HID, -64, 64)


# ---------------- Per-block bit widths and shifts ----------------
# Each matmul's output bits is the lossless minimum:
#   out_bits = act_bits + weight_bits + ceil(log2(K_in)) + 1
# After matmul we requantize to ABITS (8-bit signed). The requantize
# multiplier is just 1; the shift is chosen so typical random outputs fit
# in int8. For K_in inputs of int8 magnitude:
#   max product magnitude ~= 127 * weight_max
#   sum of K of them ~= sqrt(K) * 127 * weight_max
# We just pick conservative shifts; real PTQ would calibrate.

def _matmul_obits(K_in, ab=ABITS, wb=WBITS):
    return ab + wb + max(1, (K_in - 1).bit_length()) + 1


def _qk_v_proj_shift(K_in):
    """Shift for Q/K/V/O/gate/up/down/lm_head matmul outputs to fit int8."""
    return WBITS + max(1, (K_in - 1).bit_length()) - 2  # heuristic


# ---------------- Python reference (bit-exact replica of hardware) ----------------
def _pack_signed(values, bits):
    mask = (1 << bits) - 1
    out = 0
    for i, v in enumerate(values):
        out |= (v & mask) << (i * bits)
    return out


def _unpack_signed(packed, K, bits):
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    out = []
    for i in range(K):
        v = (packed >> (i * bits)) & mask
        if v & sign_bit:
            v -= 1 << bits
        out.append(v)
    return out


def _arsh(x, n):
    return x >> n if n >= 0 else x << (-n)


def _sat(x, bits, signed=True):
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    return max(lo, min(hi, x))


def py_matmul(W, x, weight_bits=WBITS, act_bits=ABITS, out_bits=None):
    """Replicates matmul_seq's accumulator-with-bias=0."""
    M = len(W)
    K = len(W[0])
    if out_bits is None:
        out_bits = _matmul_obits(K, act_bits, weight_bits)
    out = []
    for j in range(M):
        acc = sum(W[j][i] * x[i] for i in range(K))
        # Accumulator wraps modulo 2**out_bits
        mask = (1 << out_bits) - 1
        acc &= mask
        # Convert to signed
        if acc & (1 << (out_bits - 1)):
            acc -= 1 << out_bits
        out.append(acc)
    return out


def py_requantize(x, mul, shift, in_bits, out_bits):
    """Replicates requantize_block."""
    out = []
    for v in x:
        prod = v * mul
        shifted = _arsh(prod, shift)
        out.append(_sat(shifted, out_bits))
    return out


def _python_rsqrt_lut(in_bits, out_bits, out_frac_bits, lut_idx_bits=8):
    n_entries = 1 << lut_idx_bits
    lut = [
        max(0, min((1 << out_bits) - 1,
                   round((1.0 / math.sqrt(1.0 + i / n_entries)) *
                         (1 << out_frac_bits))))
        for i in range(n_entries)
    ]
    sqrt_half_q = round(math.sqrt(0.5) * (1 << out_frac_bits))
    out_max = (1 << out_bits) - 1

    def rsqrt(x):
        if x == 0:
            return out_max
        p = x.bit_length() - 1
        nlz = in_bits - 1 - p
        x_norm = (x << nlz) & ((1 << in_bits) - 1)
        idx = (x_norm >> (in_bits - 1 - lut_idx_bits)) & (n_entries - 1)
        lut_val = lut[idx]
        half_p = p >> 1
        p_odd = p & 1
        shifted = lut_val >> half_p
        if p_odd:
            product = shifted * sqrt_half_q
            return (product >> out_frac_bits) & out_max
        return shifted
    return rsqrt


# RMSNorm constants (must match block parameters)
RMS_GBITS = 16
RMS_OBITS = 8
RMS_EPS = 1e-5
RMS_EPS_Q = 16
RMS_RSQRT_OUT_BITS = 16
RMS_RSQRT_OUT_FRAC = 14
RMS_OUTPUT_SHIFT = 14
RMS_RSQRT_IN_BITS = 2 * ABITS + max(1, HID.bit_length()) + 4

py_rsqrt = _python_rsqrt_lut(
    RMS_RSQRT_IN_BITS, RMS_RSQRT_OUT_BITS, RMS_RSQRT_OUT_FRAC,
)


def py_rms_norm(x, gamma):
    K = len(x)
    sum_sq = sum((v * v) & ((1 << (2 * ABITS)) - 1) for v in x)
    eps_int = round(RMS_EPS * (1 << RMS_EPS_Q))
    sum_sq_eps = sum_sq + K * eps_int
    rsqrt_val = py_rsqrt(sum_sq_eps)
    out = []
    for v, g in zip(x, gamma):
        xg = v * g
        xgr = xg * rsqrt_val
        shifted = xgr >> RMS_OUTPUT_SHIFT
        out.append(_sat(shifted, RMS_OBITS))
    return out


# RoPE constants
ROPE_SINCOS_BITS = 16
ROPE_SINCOS_FRAC = 14


def py_rope(x_int, position, head_dim=D, theta_base=ROPE_THETA):
    half = head_dim // 2
    sincos_max = (1 << (ROPE_SINCOS_BITS - 1)) - 1
    out = list(x_int)
    for i in range(half):
        theta = position / (theta_base ** (2 * i / head_dim))
        s_int = max(-sincos_max - 1,
                    min(sincos_max, round(math.sin(theta) * (1 << ROPE_SINCOS_FRAC))))
        c_int = max(-sincos_max - 1,
                    min(sincos_max, round(math.cos(theta) * (1 << ROPE_SINCOS_FRAC))))
        x_e = x_int[2 * i]
        x_o = x_int[2 * i + 1]
        y_e = _arsh(x_e * c_int - x_o * s_int, ROPE_SINCOS_FRAC)
        y_o = _arsh(x_e * s_int + x_o * c_int, ROPE_SINCOS_FRAC)
        out[2 * i] = _sat(y_e, ABITS)
        out[2 * i + 1] = _sat(y_o, ABITS)
    return out


# Per-head RoPE applied to packed buses of multiple heads
def py_rope_multihead(x_int, position, num_heads, head_dim=D):
    out = []
    for h in range(num_heads):
        chunk = x_int[h * head_dim:(h + 1) * head_dim]
        rotated = py_rope(chunk, position, head_dim=head_dim)
        out.extend(rotated)
    return out


# SiLU constants
SILU_OUTPUT_SHIFT = 8


def py_silu(x_int):
    out = []
    for v in x_int:
        x_real = max(-8.0, min(8.0, v / 16.0))
        s = 1.0 / (1.0 + math.exp(-x_real))
        sig_int = max(0, min(255, round(s * 256)))
        prod = v * sig_int
        out.append(_sat(prod >> SILU_OUTPUT_SHIFT, ABITS))
    return out


# Attention constants
SCORE_BITS = 2 * ABITS + max(1, D.bit_length()) + 4
SOFTMAX_OBITS = 8
ATT_SCORE_SHIFT = 0
ATT_OUT_SHIFT = 8


def _exp_lut_for(in_bits, out_bits, in_q_frac_bits, in_clamp):
    out_max = (1 << out_bits) - 1
    lut = []
    for raw in range(1 << in_bits):
        sint = raw - (1 << in_bits) if raw & (1 << (in_bits - 1)) else raw
        x_real = sint / (1 << in_q_frac_bits)
        x_real = max(in_clamp[0], min(in_clamp[1], x_real))
        v = math.exp(x_real)
        lut.append(max(0, min(out_max, round(v * out_max))))
    return lut


def _recip_lut_for(n, out_frac_bits):
    return [0] + [
        max(0, min((1 << 24) - 1,
                   round((1.0 / idx) * (1 << out_frac_bits))))
        for idx in range(1, n)
    ]


# Softmax sub-config (from softmax_block defaults)
SM_EXP_IN_BITS = 8
SM_EXP_OUT_BITS = 12
SM_EXP_IN_Q_FRAC = 4
SM_EXP_IN_CLAMP = (-16.0, 0.0)
SM_RECIP_LUT_BITS = 12
SM_RECIP_OUT_FRAC = 16
SM_OUTPUT_SHIFT = 8

EXP_LUT = _exp_lut_for(SM_EXP_IN_BITS, SM_EXP_OUT_BITS, SM_EXP_IN_Q_FRAC, SM_EXP_IN_CLAMP)
RECIP_LUT = _recip_lut_for(1 << SM_RECIP_LUT_BITS, SM_RECIP_OUT_FRAC)


def py_softmax_for_attn(scores, mask):
    abits = SCORE_BITS
    cur_max = -((1 << (abits - 1)) - 1)
    for v, m in zip(scores, mask):
        if m and v > cur_max:
            cur_max = v
    sum_exp = 0
    exp_buf = []
    e_half = (1 << (SM_EXP_IN_BITS - 1)) - 1
    for v, m in zip(scores, mask):
        x_diff = max(-e_half, min(e_half, v - cur_max))
        raw = x_diff & ((1 << SM_EXP_IN_BITS) - 1)
        ey = EXP_LUT[raw] if m else 0
        exp_buf.append(ey)
        sum_exp += ey
    n = 1 << SM_RECIP_LUT_BITS
    idx = sum_exp if sum_exp < n else n - 1
    rv = RECIP_LUT[idx]
    out_max_y = (1 << SOFTMAX_OBITS) - 1
    return [min(out_max_y, (e * rv) >> SM_OUTPUT_SHIFT) for e in exp_buf]


def py_attention_step(q, k_new, v_new, position, k_cache, v_cache):
    k_cache[position] = list(k_new)
    v_cache[position] = list(v_new)
    out = [0] * (H * D)
    group = H // KV
    for h in range(H):
        h_kv = h // group
        scores = []
        for k_pos in range(MAX_SEQ):
            acc = 0
            for d in range(D):
                acc += q[h * D + d] * k_cache[k_pos][h_kv * D + d]
            scores.append(_arsh(acc, ATT_SCORE_SHIFT))
        mask = [1 if k_pos <= position else 0 for k_pos in range(MAX_SEQ)]
        attn = py_softmax_for_attn(scores, mask)
        for d in range(D):
            acc = 0
            for k_pos in range(MAX_SEQ):
                acc += attn[k_pos] * v_cache[k_pos][h_kv * D + d]
            out[h * D + d] = _sat(_arsh(acc, ATT_OUT_SHIFT), ABITS)
    return out


# ---------------- Python reference forward pass ----------------
def py_forward(token_id_seq):
    """Run all `token_id_seq` tokens autoregressively. Returns list of logit
    vectors (one per token) in int after lm_head."""
    # KV cache
    k_cache = [[0] * (KV * D) for _ in range(MAX_SEQ)]
    v_cache = [[0] * (KV * D) for _ in range(MAX_SEQ)]

    logits_seq = []
    for pos, tid in enumerate(token_id_seq):
        # 1. Embedding
        hidden = list(embed_W[tid])

        # 2. RMSNorm 1
        norm1 = py_rms_norm(hidden, gamma1)

        # 3. Q/K/V projections (matmul -> requantize to ABITS)
        q_wide = py_matmul(W_q, norm1)
        k_wide = py_matmul(W_k, norm1)
        v_wide = py_matmul(W_v, norm1)
        qkv_shift = _qk_v_proj_shift(HID)
        q_int = py_requantize(q_wide, 1, qkv_shift,
                              _matmul_obits(HID), ABITS)
        k_int = py_requantize(k_wide, 1, qkv_shift,
                              _matmul_obits(HID), ABITS)
        v_int = py_requantize(v_wide, 1, qkv_shift,
                              _matmul_obits(HID), ABITS)

        # 4. RoPE on Q (per head) and K (per head)
        q_rot = py_rope_multihead(q_int, pos, num_heads=H)
        k_rot = py_rope_multihead(k_int, pos, num_heads=KV)

        # 5. Attention
        attn_out = py_attention_step(q_rot, k_rot, v_int, pos,
                                     k_cache, v_cache)

        # 6. Output projection
        o_wide = py_matmul(W_o, attn_out)
        o_int = py_requantize(o_wide, 1, qkv_shift,
                              _matmul_obits(HID), ABITS)

        # 7. Residual 1
        res1 = [_sat(h_v + o_v, ABITS) for h_v, o_v in zip(hidden, o_int)]

        # 8. RMSNorm 2
        norm2 = py_rms_norm(res1, gamma2)

        # 9. MLP
        gate_wide = py_matmul(W_gate, norm2)
        up_wide   = py_matmul(W_up,   norm2)
        gate_shift = _qk_v_proj_shift(HID)
        gate_int = py_requantize(gate_wide, 1, gate_shift,
                                 _matmul_obits(HID), ABITS)
        up_int   = py_requantize(up_wide,   1, gate_shift,
                                 _matmul_obits(HID), ABITS)
        silu_g = py_silu(gate_int)
        # Element-wise mul: silu_g * up_int -> wider, then >> 4 (Q4.4 product)
        silu_up = [_sat(_arsh(s * u, 4), ABITS)
                   for s, u in zip(silu_g, up_int)]
        down_wide = py_matmul(W_down, silu_up)
        down_shift = _qk_v_proj_shift(INTER)
        down_int = py_requantize(down_wide, 1, down_shift,
                                 _matmul_obits(INTER), ABITS)

        # 10. Residual 2
        res2 = [_sat(r + d, ABITS) for r, d in zip(res1, down_int)]

        # 11. Final RMSNorm
        final_norm = py_rms_norm(res2, gamma_final)

        # 12. lm_head
        lm_wide = py_matmul(W_lm, final_norm)
        # logits stay wide for argmax — argmax doesn't need int8
        logits_seq.append(lm_wide)

    return logits_seq


# ---------------- Hardware GateGraph builder ----------------
def _packed_add_gates(*, name_prefix, lhs, rhs, K, abits):
    """Element-wise add of two packed buses, output saturated to abits.

    Returns (gates, output_signal_name). Output is packed bus of K * abits bits.
    """
    gates = []
    elt_names = []
    for i in range(K):
        l_slice = f"{name_prefix}.lhs.{i}"
        r_slice = f"{name_prefix}.rhs.{i}"
        sum_name = f"{name_prefix}.sum.{i}"
        sat_name = f"{name_prefix}.sat.{i}"
        gates.append(Gate(name=l_slice, kind="slice", inputs=[lhs],
                          attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                          output_width=abits, output_signed=True))
        gates.append(Gate(name=r_slice, kind="slice", inputs=[rhs],
                          attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                          output_width=abits, output_signed=True))
        gates.append(Gate(name=sum_name, kind="add", inputs=[l_slice, r_slice],
                          output_width=abits + 1, output_signed=True))
        # Saturate to abits via clamp gate
        lo, hi = -(1 << (abits - 1)), (1 << (abits - 1)) - 1
        gates.append(Gate(name=sat_name, kind="clamp", inputs=[sum_name],
                          attrs={"lo": lo, "hi": hi},
                          output_width=abits, output_signed=True))
        elt_names.append(sat_name)
    out_name = f"{name_prefix}.packed"
    # `concat` is MSB-first; we want element 0 in LSBs, so reverse.
    gates.append(Gate(name=out_name, kind="concat",
                      inputs=list(reversed(elt_names)),
                      output_width=K * abits, output_signed=True))
    return gates, out_name


def _packed_mul_shift_gates(*, name_prefix, lhs, rhs, K, abits, shift):
    """Element-wise (lhs[i] * rhs[i]) >> shift saturated to abits."""
    gates = []
    elt_names = []
    for i in range(K):
        l_slice = f"{name_prefix}.lhs.{i}"
        r_slice = f"{name_prefix}.rhs.{i}"
        prod = f"{name_prefix}.prod.{i}"
        shifted = f"{name_prefix}.shifted.{i}"
        sat = f"{name_prefix}.sat.{i}"
        gates.append(Gate(name=l_slice, kind="slice", inputs=[lhs],
                          attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                          output_width=abits, output_signed=True))
        gates.append(Gate(name=r_slice, kind="slice", inputs=[rhs],
                          attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                          output_width=abits, output_signed=True))
        gates.append(Gate(name=prod, kind="mul", inputs=[l_slice, r_slice],
                          output_width=2 * abits, output_signed=True))
        gates.append(Gate(name=shifted, kind="shift_right", inputs=[prod],
                          attrs={"amount": shift},
                          output_width=2 * abits, output_signed=True))
        lo, hi = -(1 << (abits - 1)), (1 << (abits - 1)) - 1
        gates.append(Gate(name=sat, kind="clamp", inputs=[shifted],
                          attrs={"lo": lo, "hi": hi},
                          output_width=abits, output_signed=True))
        elt_names.append(sat)
    out_name = f"{name_prefix}.packed"
    gates.append(Gate(name=out_name, kind="concat",
                      inputs=list(reversed(elt_names)),
                      output_width=K * abits, output_signed=True))
    return gates, out_name


def _instance_gate(*, name, module, instance_name, input_ports, inputs,
                   output_port, output_width, output_signed=True,
                   extra_outputs=None):
    attrs = {
        "module_name": module, "instance_name": instance_name,
        "input_ports": input_ports, "output_port": output_port,
    }
    if extra_outputs:
        attrs["extra_output_ports"] = extra_outputs
    return Gate(name=name, kind="instance", inputs=list(inputs),
                attrs=attrs, output_width=output_width,
                output_signed=output_signed)


def build_tiny_llama_graph():
    """Build the GateGraph for the 1-layer tiny LLaMA (single forward step)."""
    submodules: list[RawSubmodule] = []
    gates: list[Gate] = []

    qkv_shift = _qk_v_proj_shift(HID)
    o_shift = _qk_v_proj_shift(HID)
    gate_shift = _qk_v_proj_shift(HID)
    down_shift = _qk_v_proj_shift(INTER)
    lm_shift = _qk_v_proj_shift(HID)

    # ------ Embedding (combinational) ------
    embed_sub = embedding_block(V=VOCAB, H=HID, abits=ABITS, weights=embed_W)
    submodules.append(embed_sub)
    gates.append(_instance_gate(
        name="hidden_packed", module=embed_sub.top, instance_name="embed",
        input_ports=["token_id"], inputs=["token_id"],
        output_port="hidden_packed",
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ RMSNorm 1 ------
    rms1_sub, rsqrt_sub = rms_norm_block(
        K=HID, gamma_int=gamma1, gamma_bits=RMS_GBITS,
        abits=ABITS, obits=ABITS, eps=RMS_EPS, eps_q=RMS_EPS_Q,
        rsqrt_in_bits=RMS_RSQRT_IN_BITS,
        rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
        rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
        output_shift=RMS_OUTPUT_SHIFT,
        module_suffix="r1",
    )
    submodules += [rms1_sub, rsqrt_sub]
    gates.append(Gate(name="rms1_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="norm1_packed", module=rms1_sub.top, instance_name="rms1",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "start", "hidden_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "rms1_done")],
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ Q/K/V projections (matmul + requantize) ------
    # Build matmul submodules
    q_matmul_sub = matmul_seq_block(
        weights=W_q, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="q_proj",
    )
    k_matmul_sub = matmul_seq_block(
        weights=W_k, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="k_proj",
    )
    v_matmul_sub = matmul_seq_block(
        weights=W_v, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="v_proj",
    )
    submodules += [q_matmul_sub, k_matmul_sub, v_matmul_sub]
    qkv_obits = _matmul_obits(HID)

    # Concat hidden's 8 elements into a packed bus of HID*ABITS for matmul.
    # rms1's output y_packed is already packed, but matmul_seq_invoke expects
    # x as ELEMENTS that it concats internally. So we call matmul_seq_invoke
    # on individual element signals. Easier: instantiate matmul_seq directly,
    # passing norm1_packed as x_packed (matmul_seq's x_packed is elem-LSB-first).
    # matmul_seq packs elements LSB-first in x_packed[i*ABITS +: ABITS].
    # rms_norm's output uses the same convention. So we can pass directly.
    gates.append(Gate(name="q_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="q_wide_packed", module=q_matmul_sub.top, instance_name="qm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rms1_done", "norm1_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "q_done")],
        output_width=HID * qkv_obits, output_signed=True,
    ))
    gates.append(Gate(name="k_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="k_wide_packed", module=k_matmul_sub.top, instance_name="km",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rms1_done", "norm1_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "k_done")],
        output_width=KV * D * qkv_obits, output_signed=True,
    ))
    gates.append(Gate(name="v_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="v_wide_packed", module=v_matmul_sub.top, instance_name="vm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rms1_done", "norm1_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "v_done")],
        output_width=KV * D * qkv_obits, output_signed=True,
    ))

    # Requantize: muls=[1]*K, shifts=[qkv_shift]*K
    rq_q_sub = requantize_block(
        K=HID, in_bits=qkv_obits, out_bits=ABITS,
        muls=[1] * HID, shifts=[qkv_shift] * HID, mul_bits=8,
        module_suffix="q",
    )
    rq_k_sub = requantize_block(
        K=KV * D, in_bits=qkv_obits, out_bits=ABITS,
        muls=[1] * (KV * D), shifts=[qkv_shift] * (KV * D), mul_bits=8,
        module_suffix="k",
    )
    rq_v_sub = requantize_block(
        K=KV * D, in_bits=qkv_obits, out_bits=ABITS,
        muls=[1] * (KV * D), shifts=[qkv_shift] * (KV * D), mul_bits=8,
        module_suffix="v",
    )
    submodules += [rq_q_sub, rq_k_sub, rq_v_sub]
    gates.append(_instance_gate(
        name="q_int_packed", module=rq_q_sub.top, instance_name="rq_q",
        input_ports=["x_packed"], inputs=["q_wide_packed"],
        output_port="y_packed",
        output_width=HID * ABITS, output_signed=True,
    ))
    gates.append(_instance_gate(
        name="k_int_packed", module=rq_k_sub.top, instance_name="rq_k",
        input_ports=["x_packed"], inputs=["k_wide_packed"],
        output_port="y_packed",
        output_width=KV * D * ABITS, output_signed=True,
    ))
    gates.append(_instance_gate(
        name="v_int_packed", module=rq_v_sub.top, instance_name="rq_v",
        input_ports=["x_packed"], inputs=["v_wide_packed"],
        output_port="y_packed",
        output_width=KV * D * ABITS, output_signed=True,
    ))

    # ------ RoPE on Q (per head) and K (per head) ------
    # rope_block operates on one head at a time. For multi-head we need to
    # call rope on each head's slice. Simpler: define rope per total length
    # and treat as single head_dim. But rope depends on head_dim parameter.
    # Approach: instantiate H rope blocks for Q, KV rope blocks for K.
    # Each gets a sliced packed bus.
    rope_sub = rope_block(
        head_dim=D, max_seq=MAX_SEQ, theta_base=ROPE_THETA,
        abits=ABITS, sincos_bits=ROPE_SINCOS_BITS,
        sincos_frac_bits=ROPE_SINCOS_FRAC,
    )
    submodules.append(rope_sub)

    # Slice q_int_packed into H pieces and k_int_packed into KV pieces, apply
    # RoPE to each, concat back.
    def slice_and_rope(src, num_heads, prefix):
        rotated_names = []
        for h in range(num_heads):
            slice_name = f"{prefix}.head{h}"
            rot_name = f"{prefix}.rot{h}"
            gates.append(Gate(name=slice_name, kind="slice", inputs=[src],
                              attrs={
                                  "hi": (h + 1) * D * ABITS - 1,
                                  "lo": h * D * ABITS,
                              },
                              output_width=D * ABITS, output_signed=True))
            gates.append(_instance_gate(
                name=rot_name, module=rope_sub.top,
                instance_name=f"{prefix}_inst{h}",
                input_ports=["x_packed", "position"],
                inputs=[slice_name, "position"],
                output_port="y_packed",
                output_width=D * ABITS, output_signed=True,
            ))
            rotated_names.append(rot_name)
        cat_name = f"{prefix}.cat"
        gates.append(Gate(name=cat_name, kind="concat",
                          inputs=list(reversed(rotated_names)),
                          output_width=num_heads * D * ABITS,
                          output_signed=True))
        return cat_name

    q_rot_packed = slice_and_rope("q_int_packed", H, "rope_q")
    k_rot_packed = slice_and_rope("k_int_packed", KV, "rope_k")

    # ------ Attention ------
    # Attention block expects k_new and v_new in pre-RoPE order, and applies
    # RoPE-equivalent transformations internally? No, my attention block
    # does NOT apply RoPE — it expects already-rotated K. Good.
    # But ALSO: the attention block expects start to fire after Q, K_rot, V
    # are all stable. Since Q comes via rms1_done -> q matmul -> rq_q (combinational)
    # -> rope (combinational), q is ready a few cycles after rms1_done.
    # For triggering: use V's done since V matmul is the slowest.
    att_sub, att_deps = attention_step_block(
        num_q_heads=H, num_kv_heads=KV, head_dim=D, max_seq=MAX_SEQ,
        abits=ABITS, score_bits=SCORE_BITS, softmax_obits=SOFTMAX_OBITS,
        out_abits=ABITS,
        score_shift=ATT_SCORE_SHIFT, out_shift=ATT_OUT_SHIFT,
    )
    submodules += [att_sub] + list(att_deps)

    # All three matmuls (q, k, v) take the same number of cycles (HID+1=9 each)
    # since they share K=HID. They start together on rms1_done. The latest
    # done is, say, v_done. RoPE is combinational. So use v_done as the trigger
    # for attention... but RoPE is NOT a sequential block. Q needs RoPE
    # applied first, but RoPE outputs combinationally from q_int_packed.
    # Once q_int (rq_q output, combinational from q_wide which is registered
    # by qm) is stable AND position is stable, RoPE result is stable.
    # So as soon as v_done fires (q,k,v all done), q_rot_packed and k_rot_packed
    # are also stable. Trigger attention on v_done.
    gates.append(Gate(name="att_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="attn_out_packed", module=att_sub.top, instance_name="att",
        input_ports=["clk", "rst", "start", "position",
                     "q_packed", "k_new_packed", "v_new_packed"],
        inputs=["clk", "rst", "v_done", "position",
                q_rot_packed, k_rot_packed, "v_int_packed"],
        output_port="out_packed",
        extra_outputs=[("done", "att_done")],
        output_width=H * D * ABITS, output_signed=True,
    ))

    # ------ Output projection ------
    o_matmul_sub = matmul_seq_block(
        weights=W_o, weight_bits=WBITS, act_bits=ABITS, module_suffix="o_proj",
    )
    submodules.append(o_matmul_sub)
    o_obits = _matmul_obits(HID)
    gates.append(Gate(name="o_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="o_wide_packed", module=o_matmul_sub.top, instance_name="om",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "att_done", "attn_out_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "o_done")],
        output_width=HID * o_obits, output_signed=True,
    ))
    rq_o_sub = requantize_block(
        K=HID, in_bits=o_obits, out_bits=ABITS,
        muls=[1] * HID, shifts=[o_shift] * HID, mul_bits=8,
        module_suffix="o",
    )
    submodules.append(rq_o_sub)
    gates.append(_instance_gate(
        name="o_int_packed", module=rq_o_sub.top, instance_name="rq_o",
        input_ports=["x_packed"], inputs=["o_wide_packed"],
        output_port="y_packed",
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ Residual 1: hidden + o_int ------
    res1_gates, res1_packed = _packed_add_gates(
        name_prefix="res1", lhs="hidden_packed", rhs="o_int_packed",
        K=HID, abits=ABITS,
    )
    gates += res1_gates

    # ------ RMSNorm 2 ------
    rms2_sub, rsqrt2_sub = rms_norm_block(
        K=HID, gamma_int=gamma2, gamma_bits=RMS_GBITS,
        abits=ABITS, obits=ABITS, eps=RMS_EPS, eps_q=RMS_EPS_Q,
        rsqrt_in_bits=RMS_RSQRT_IN_BITS,
        rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
        rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
        output_shift=RMS_OUTPUT_SHIFT,
        module_suffix="r2",
    )
    # rsqrt2_sub will dedupe with rsqrt_sub since same params
    submodules += [rms2_sub, rsqrt2_sub]
    gates.append(Gate(name="rms2_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="norm2_packed", module=rms2_sub.top, instance_name="rms2",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "o_done", res1_packed],
        output_port="y_packed",
        extra_outputs=[("done", "rms2_done")],
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ MLP gate / up projections ------
    gate_matmul_sub = matmul_seq_block(
        weights=W_gate, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="gate_proj",
    )
    up_matmul_sub = matmul_seq_block(
        weights=W_up, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="up_proj",
    )
    submodules += [gate_matmul_sub, up_matmul_sub]
    mlp_obits = _matmul_obits(HID)
    gates.append(Gate(name="gate_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="gate_wide_packed", module=gate_matmul_sub.top,
        instance_name="gm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rms2_done", "norm2_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "gate_done")],
        output_width=INTER * mlp_obits, output_signed=True,
    ))
    gates.append(Gate(name="up_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="up_wide_packed", module=up_matmul_sub.top, instance_name="um",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rms2_done", "norm2_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "up_done")],
        output_width=INTER * mlp_obits, output_signed=True,
    ))
    rq_gate_sub = requantize_block(
        K=INTER, in_bits=mlp_obits, out_bits=ABITS,
        muls=[1] * INTER, shifts=[gate_shift] * INTER, mul_bits=8,
        module_suffix="gate",
    )
    rq_up_sub = requantize_block(
        K=INTER, in_bits=mlp_obits, out_bits=ABITS,
        muls=[1] * INTER, shifts=[gate_shift] * INTER, mul_bits=8,
        module_suffix="up",
    )
    submodules += [rq_gate_sub, rq_up_sub]
    gates.append(_instance_gate(
        name="gate_int_packed", module=rq_gate_sub.top,
        instance_name="rq_g",
        input_ports=["x_packed"], inputs=["gate_wide_packed"],
        output_port="y_packed",
        output_width=INTER * ABITS, output_signed=True,
    ))
    gates.append(_instance_gate(
        name="up_int_packed", module=rq_up_sub.top, instance_name="rq_u",
        input_ports=["x_packed"], inputs=["up_wide_packed"],
        output_port="y_packed",
        output_width=INTER * ABITS, output_signed=True,
    ))

    # ------ SiLU(gate) ------
    silu_sub, sig_sub = silu_block(
        K=INTER, abits=ABITS, obits=ABITS,
        sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8,
        output_shift=SILU_OUTPUT_SHIFT,
    )
    submodules += [silu_sub, sig_sub]
    gates.append(Gate(name="silu_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="silu_g_packed", module=silu_sub.top, instance_name="silu",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "up_done", "gate_int_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "silu_done")],
        output_width=INTER * ABITS, output_signed=True,
    ))

    # ------ silu_g * up (element-wise, shifted) ------
    silu_up_gates, silu_up_packed = _packed_mul_shift_gates(
        name_prefix="silu_up", lhs="silu_g_packed", rhs="up_int_packed",
        K=INTER, abits=ABITS, shift=4,
    )
    gates += silu_up_gates

    # ------ down_proj ------
    down_matmul_sub = matmul_seq_block(
        weights=W_down, weight_bits=WBITS, act_bits=ABITS,
        module_suffix="down_proj",
    )
    submodules.append(down_matmul_sub)
    down_obits = _matmul_obits(INTER)
    gates.append(Gate(name="down_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="down_wide_packed", module=down_matmul_sub.top,
        instance_name="dm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "silu_done", silu_up_packed],
        output_port="y_packed",
        extra_outputs=[("done", "down_done")],
        output_width=HID * down_obits, output_signed=True,
    ))
    rq_down_sub = requantize_block(
        K=HID, in_bits=down_obits, out_bits=ABITS,
        muls=[1] * HID, shifts=[down_shift] * HID, mul_bits=8,
        module_suffix="down",
    )
    submodules.append(rq_down_sub)
    gates.append(_instance_gate(
        name="down_int_packed", module=rq_down_sub.top,
        instance_name="rq_d",
        input_ports=["x_packed"], inputs=["down_wide_packed"],
        output_port="y_packed",
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ Residual 2: res1 + down_int ------
    res2_gates, res2_packed = _packed_add_gates(
        name_prefix="res2", lhs=res1_packed, rhs="down_int_packed",
        K=HID, abits=ABITS,
    )
    gates += res2_gates

    # ------ Final RMSNorm ------
    rmsf_sub, rsqrtf_sub = rms_norm_block(
        K=HID, gamma_int=gamma_final, gamma_bits=RMS_GBITS,
        abits=ABITS, obits=ABITS, eps=RMS_EPS, eps_q=RMS_EPS_Q,
        rsqrt_in_bits=RMS_RSQRT_IN_BITS,
        rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
        rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
        output_shift=RMS_OUTPUT_SHIFT,
        module_suffix="rf",
    )
    submodules += [rmsf_sub, rsqrtf_sub]
    gates.append(Gate(name="rmsf_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="final_norm_packed", module=rmsf_sub.top, instance_name="rmsf",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "down_done", res2_packed],
        output_port="y_packed",
        extra_outputs=[("done", "rmsf_done")],
        output_width=HID * ABITS, output_signed=True,
    ))

    # ------ lm_head ------
    lm_matmul_sub = matmul_seq_block(
        weights=W_lm, weight_bits=WBITS, act_bits=ABITS, module_suffix="lm",
    )
    submodules.append(lm_matmul_sub)
    lm_obits = _matmul_obits(HID)
    gates.append(Gate(name="lm_done", kind="extern_wire", output_width=1))
    gates.append(_instance_gate(
        name="logits_packed", module=lm_matmul_sub.top, instance_name="lm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rmsf_done", "final_norm_packed"],
        output_port="y_packed",
        extra_outputs=[("done", "lm_done")],
        output_width=VOCAB * lm_obits, output_signed=True,
    ))

    # ------ argmax ------
    argmax_sub = argmax_block(K=VOCAB, abits=lm_obits)
    submodules.append(argmax_sub)
    idx_bits = max(1, (VOCAB - 1).bit_length())
    gates.append(_instance_gate(
        name="next_token_id", module=argmax_sub.top, instance_name="am",
        input_ports=["x_packed"], inputs=["logits_packed"],
        output_port="argmax_idx",
        output_width=idx_bits, output_signed=False,
    ))

    # ------ Build GateGraph ------
    pos_bits = max(1, (MAX_SEQ - 1).bit_length() + 1)
    parent = GateGraph(
        inputs=[
            Signal("clk"), Signal("rst"), Signal("start"),
            Signal("token_id", width=max(1, (VOCAB - 1).bit_length()),
                   signed=False),
            Signal("position", width=pos_bits, signed=False),
        ],
        outputs=[
            Signal("lm_done", width=1),
            Signal("logits_packed", width=VOCAB * lm_obits, signed=True),
            Signal("next_token_id", width=idx_bits, signed=False),
        ],
        gates=gates, top="tiny_llama",
        submodules=submodules,
    )
    return parent, lm_obits


def main() -> int:
    # Pick a token sequence
    tokens = [random.randint(0, VOCAB - 1) for _ in range(MAX_SEQ)]
    print(f"input tokens: {tokens}")

    # Run Python reference
    logits_seq = py_forward(tokens)
    for t, logits in enumerate(logits_seq):
        argmax_idx = max(range(VOCAB), key=lambda i: logits[i])
        print(f"  t={t} tid={tokens[t]} argmax(logits) = {argmax_idx}  "
              f"logits range [{min(logits):>8}, {max(logits):>8}]")

    # Build hardware
    print("\nBuilding hardware GateGraph...")
    graph, lm_obits = build_tiny_llama_graph()
    text = emit_module(graph)
    print(f"  emitted {len(text.splitlines())} lines of Verilog "
          f"({len(text)/1024:.0f} KB), {len(graph.submodules)} submodule slots")

    out_dir = Path(r"D:\safetensors2verilog\_play\tiny_llama_out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "tiny_llama.v").write_text(text, encoding="utf-8")

    if shutil.which("iverilog") is None:
        print("iverilog not on PATH; stopping here.")
        return 0

    print("Compiling with iverilog (may take a minute)...")
    # Build a testbench that drives the four tokens sequentially.
    pos_bits = max(1, (MAX_SEQ - 1).bit_length() + 1)
    tb_drives = []
    for t, tid in enumerate(tokens):
        tb_drives.append(f"""\
    @(negedge clk);
      token_id = {(VOCAB-1).bit_length()}'d{tid};
      position = {pos_bits}'d{t};
      start <= 1;
    @(negedge clk); start <= 0;
    cycles = 0;
    while (!lm_done) begin
      @(posedge clk);
      cycles = cycles + 1;
      if (cycles > 5000) begin $display("TIMEOUT t={t}"); $finish; end
    end
    $display("R {t} cyc=%0d nt=%0d logits=%h", cycles, next_token_id, logits_packed);""")

    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0; always #5 clk = ~clk;\n"
        "  reg rst = 1, start = 0;\n"
        f"  reg [{(VOCAB-1).bit_length()-1}:0] token_id;\n"
        f"  reg [{pos_bits-1}:0] position;\n"
        "  wire lm_done;\n"
        f"  wire signed [{VOCAB*lm_obits - 1}:0] logits_packed;\n"
        f"  wire [{(VOCAB-1).bit_length()-1}:0] next_token_id;\n"
        "  tiny_llama dut(.clk(clk), .rst(rst), .start(start),\n"
        "                 .token_id(token_id), .position(position),\n"
        "                 .lm_done(lm_done), .logits_packed(logits_packed),\n"
        "                 .next_token_id(next_token_id));\n"
        "  integer cycles;\n"
        "  initial begin\n"
        "    rst = 1; #20 rst = 0;\n"
        + "\n".join(tb_drives) +
        "\n    $finish;\n  end\nendmodule\n"
    )
    (out_dir / "tb.v").write_text(tb, encoding="utf-8")

    vvp = out_dir / "tb.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp),
         str(out_dir / "tiny_llama.v"), str(out_dir / "tb.v")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("iverilog compile failed:")
        print(proc.stderr[:5000])
        return 1
    print("Running simulation...")
    proc = subprocess.run(
        ["vvp", str(vvp)], capture_output=True, text=True, timeout=600,
    )
    log = proc.stdout
    (out_dir / "log.txt").write_text(log, encoding="utf-8")

    # Parse and compare
    by_t = {}
    for line in log.splitlines():
        if line.startswith("R "):
            toks = line.split()
            t = int(toks[1])
            cyc = int(toks[2].split("=")[1])
            nt = int(toks[3].split("=")[1])
            logits_hex = toks[4].split("=")[1]
            by_t[t] = (cyc, nt, int(logits_hex, 16))

    print("\n--- comparison ---")
    fails = 0
    for t, tid in enumerate(tokens):
        if t not in by_t:
            print(f"  t={t} MISSING")
            fails += 1
            continue
        cyc, nt_sim, logits_packed = by_t[t]
        sim_logits = _unpack_signed(logits_packed, VOCAB, lm_obits)
        py_logits = logits_seq[t]
        py_argmax = max(range(VOCAB), key=lambda i: py_logits[i])
        ok_logits = sim_logits == py_logits
        ok_argmax = nt_sim == py_argmax
        status = "OK" if ok_logits and ok_argmax else "FAIL"
        print(f"  t={t} cyc={cyc} sim_argmax={nt_sim} py_argmax={py_argmax} "
              f"{'logits OK' if ok_logits else 'logits MISMATCH'}  [{status}]")
        if not ok_logits:
            fails += 1
            for i in range(VOCAB):
                if sim_logits[i] != py_logits[i]:
                    print(f"    logit[{i}]: sim={sim_logits[i]} py={py_logits[i]}")
                    break
        if not ok_argmax:
            fails += 1

    if fails == 0:
        print(f"\n{MAX_SEQ}/{MAX_SEQ} tokens bit-exact (logits + argmax)")
        return 0
    print(f"\n{MAX_SEQ - fails}/{MAX_SEQ} tokens match")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
