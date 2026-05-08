"""Bit-exact Python mirrors of the fixed-point LUT blocks.

Item 6 of TODO.md. The Verilog blocks under ``safetensors2verilog.blocks``
use Q-format integer arithmetic with precomputed LUTs for transcendentals
(rsqrt, sigmoid, exp). The original ``llama_int_reference_one_layer`` in
``llama_reference.py`` used float math for those primitives, so its
output drifted by a few LSBs vs the Verilog at the final RMSNorm.

This module mirrors each LUT block step-by-step, so the Python output
matches the Verilog output bit-for-bit. The functions take the same
parameter names + defaults as the corresponding ``blocks/*`` factories
so callers can re-use the same hyperparameters.

  * ``rsqrt_lut_eval``    - mirror of ``blocks.rsqrt.rsqrt_block``
  * ``sigmoid_lut_eval``  - mirror of ``blocks.sigmoid.sigmoid_block``
  * ``exp_lut_eval``      - mirror of ``blocks.exp.exp_block``
  * ``silu_lut_eval``     - mirror of ``blocks.silu.silu_block`` (uses sigmoid_lut)
  * ``rms_norm_lut_eval`` - mirror of ``blocks.rms_norm.rms_norm_block``
                             (uses rsqrt_lut)

These mirror the Verilog *exactly* — the LUT ROM contents are computed
by the same ``round(...)`` formulas the block factories use, and the
sequential FSM math (e.g. RMSNorm's accumulate-then-rsqrt-then-multiply)
is replayed by the eval function in the same order with the same bit
widths. The result is a Python reference that callers can diff against
the Verilog for unit-test-quality bit-exact verification.
"""
from __future__ import annotations

import math
from functools import lru_cache


# --------------------------------------------------------------------------
# rsqrt LUT
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _make_rsqrt_lut(
    out_bits: int, out_frac_bits: int, lut_idx_bits: int,
) -> tuple[int, ...]:
    """LUT contents: ``rsqrt(1 + i / 2**lut_idx_bits) * 2**out_frac_bits``,
    rounded and clamped to ``out_bits`` unsigned. Same formula as the
    block factory so the LUT is identical."""
    n_entries = 1 << lut_idx_bits
    out_max = (1 << out_bits) - 1
    return tuple(
        max(
            0,
            min(
                out_max,
                round((1.0 / math.sqrt(1.0 + i / n_entries))
                      * (1 << out_frac_bits)),
            ),
        )
        for i in range(n_entries)
    )


def rsqrt_lut_eval(
    x: int,
    *,
    in_bits: int = 32,
    out_bits: int = 16,
    out_frac_bits: int = 14,
    lut_idx_bits: int = 8,
) -> int:
    """Bit-exact mirror of the rsqrt block's combinational output.

    Replicates the Verilog: count leading zeros, normalise, look up,
    apply power-of-two rescale (with an extra sqrt(0.5) multiply for
    odd p).
    """
    out_max = (1 << out_bits) - 1
    if x <= 0:
        return out_max
    # Count leading zeros: 0..in_bits-1.
    bit = in_bits - 1
    lz = 0
    while bit >= 0 and ((x >> bit) & 1) == 0:
        lz += 1
        bit -= 1
    p = in_bits - 1 - lz
    x_norm = x << lz
    # Mask to in_bits since shift may have grown beyond.
    x_norm &= (1 << in_bits) - 1
    # idx = x_norm[in_bits - 2 : in_bits - 1 - lut_idx_bits]
    shift_for_idx = in_bits - 1 - lut_idx_bits
    idx = (x_norm >> shift_for_idx) & ((1 << lut_idx_bits) - 1)

    lut = _make_rsqrt_lut(out_bits, out_frac_bits, lut_idx_bits)
    lut_val = lut[idx]

    p_odd = p & 1
    half_p = p >> 1
    shifted = lut_val >> half_p
    shifted &= out_max
    sqrt_half_q = round(math.sqrt(0.5) * (1 << out_frac_bits))
    product = shifted * sqrt_half_q
    # The Verilog `wire [out_bits + out_frac_bits - 1:0] product` so
    # mask to that width.
    product &= (1 << (out_bits + out_frac_bits)) - 1
    # scaled = product[out_bits + out_frac_bits - 1 : out_frac_bits]
    scaled = (product >> out_frac_bits) & out_max
    return scaled if p_odd else shifted


# --------------------------------------------------------------------------
# sigmoid LUT
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _make_sigmoid_lut(
    in_bits: int, out_bits: int, in_q_frac_bits: int,
    in_clamp_lo: float, in_clamp_hi: float,
) -> tuple[int, ...]:
    n_entries = 1 << in_bits
    out_max = (1 << out_bits) - 1
    out: list[int] = []
    for raw in range(n_entries):
        if raw & (1 << (in_bits - 1)):
            sint = raw - (1 << in_bits)
        else:
            sint = raw
        x = sint / (1 << in_q_frac_bits)
        x = max(in_clamp_lo, min(in_clamp_hi, x))
        s = 1.0 / (1.0 + math.exp(-x))
        out.append(max(0, min(out_max, round(s * (1 << out_bits)))))
    return tuple(out)


def sigmoid_lut_eval(
    raw: int,
    *,
    in_bits: int = 8,
    out_bits: int = 8,
    in_q_frac_bits: int = 4,
    in_clamp: tuple[float, float] = (-8.0, 8.0),
) -> int:
    """Bit-exact mirror of sigmoid_block. ``raw`` is the unsigned bit
    pattern of the signed input (range 0..2**in_bits-1)."""
    lut = _make_sigmoid_lut(
        in_bits, out_bits, in_q_frac_bits, in_clamp[0], in_clamp[1],
    )
    return lut[raw & ((1 << in_bits) - 1)]


# --------------------------------------------------------------------------
# exp LUT
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _make_exp_lut(
    in_bits: int, out_bits: int, in_q_frac_bits: int,
    in_clamp_lo: float, in_clamp_hi: float,
) -> tuple[int, ...]:
    n_entries = 1 << in_bits
    out_max = (1 << out_bits) - 1
    out: list[int] = []
    for raw in range(n_entries):
        if raw & (1 << (in_bits - 1)):
            sint = raw - (1 << in_bits)
        else:
            sint = raw
        x = sint / (1 << in_q_frac_bits)
        x = max(in_clamp_lo, min(in_clamp_hi, x))
        v = math.exp(x)
        out.append(max(0, min(out_max, round(v * out_max))))
    return tuple(out)


def exp_lut_eval(
    raw: int,
    *,
    in_bits: int = 8,
    out_bits: int = 16,
    in_q_frac_bits: int = 4,
    in_clamp: tuple[float, float] = (-16.0, 0.0),
) -> int:
    """Bit-exact mirror of exp_block."""
    lut = _make_exp_lut(
        in_bits, out_bits, in_q_frac_bits, in_clamp[0], in_clamp[1],
    )
    return lut[raw & ((1 << in_bits) - 1)]


# --------------------------------------------------------------------------
# SiLU sequential
# --------------------------------------------------------------------------


def silu_lut_eval(
    x: list[int],
    *,
    abits: int = 8,
    obits: int = 8,
    sigmoid_in_q_frac_bits: int = 4,
    sigmoid_out_bits: int = 8,
    output_shift: int = 8,
) -> list[int]:
    """Bit-exact mirror of the SiLU sequential block.

    For each element: ``y[i] = sat( (x[i] * sigmoid_lut(x[i])) >>> shift )``.
    The sigmoid LUT is keyed by the unsigned representation of the signed
    8-bit x (matching the Verilog's ``rom[x[in_bits-1:0]]`` indexing).
    """
    out_lo = -(1 << (obits - 1))
    out_hi = (1 << (obits - 1)) - 1
    abits_mask = (1 << abits) - 1
    out: list[int] = []
    for xi in x:
        # The Verilog uses `rom[x[in_bits-1:0]]` which is the unsigned
        # bit pattern; we encode signed as two's complement.
        raw = xi & abits_mask
        sig_y = sigmoid_lut_eval(
            raw, in_bits=abits, out_bits=sigmoid_out_bits,
            in_q_frac_bits=sigmoid_in_q_frac_bits,
            in_clamp=(-(1 << (abits - 1 - sigmoid_in_q_frac_bits)),
                      (1 << (abits - 1 - sigmoid_in_q_frac_bits)) - 1
                      + 1.0 / (1 << sigmoid_in_q_frac_bits)),
        )
        # The Verilog `prod = x * $signed({1'b0, sig_y})`. sig_y is
        # treated as a non-negative signed value, so just multiply.
        prod = xi * sig_y
        # Arithmetic right shift (Python's >> is arithmetic on negatives).
        shifted = prod >> output_shift
        out.append(max(out_lo, min(out_hi, shifted)))
    return out


# --------------------------------------------------------------------------
# RMSNorm sequential
# --------------------------------------------------------------------------


def rms_norm_lut_eval(
    x: list[int],
    gamma_int: list[int],
    *,
    K: int,
    abits: int = 8,
    gamma_bits: int = 16,
    obits: int = 8,
    eps: float = 1e-5,
    eps_q: int = 16,
    rsqrt_in_bits: int | None = None,
    rsqrt_out_bits: int = 16,
    rsqrt_out_frac_bits: int = 14,
    output_shift: int = 14,
) -> list[int]:
    """Bit-exact mirror of the RMSNorm sequential block.

    Computes ``y[i] = sat( (x[i] * gamma[i] * rsqrt(sum_sq + K*eps_int))
    >>> output_shift )`` with the rsqrt computed via the bit-exact LUT
    mirror (no float rsqrt).
    """
    if len(x) != K:
        raise ValueError(f"x length {len(x)} != K={K}")
    if len(gamma_int) != K:
        raise ValueError(f"gamma_int length {len(gamma_int)} != K={K}")
    sum_sq_bits = 2 * abits + max(1, K.bit_length())
    if rsqrt_in_bits is None:
        rsqrt_in_bits = sum_sq_bits + 4
    eps_int = round(eps * (1 << eps_q))
    K_eps_int = K * eps_int
    # The Verilog accumulator masks to sum_sq_bits.
    sum_sq_mask = (1 << sum_sq_bits) - 1
    rsqrt_in_mask = (1 << rsqrt_in_bits) - 1
    sq_mask = (1 << (2 * abits)) - 1

    sum_sq = 0
    for xi in x:
        # Verilog: sq = x_now * x_now (signed * signed, but the result
        # is non-negative). The accumulator uses sq[2*abits-1:0] which
        # is just the low 2*abits bits.
        sq = (xi * xi) & sq_mask
        sum_sq = (sum_sq + sq) & sum_sq_mask
    sum_sq_eps = (sum_sq + K_eps_int) & rsqrt_in_mask

    rsqrt_val = rsqrt_lut_eval(
        sum_sq_eps,
        in_bits=rsqrt_in_bits,
        out_bits=rsqrt_out_bits,
        out_frac_bits=rsqrt_out_frac_bits,
    )

    out_lo = -(1 << (obits - 1))
    out_hi = (1 << (obits - 1)) - 1
    y: list[int] = []
    for i in range(K):
        # xg = x_now * gamma_now (signed * signed).
        xg = x[i] * gamma_int[i]
        # xgr = xg * $signed({1'b0, rsqrt_val}). rsqrt_val is non-negative.
        xgr = xg * rsqrt_val
        # Arithmetic right shift.
        shifted = xgr >> output_shift
        y.append(max(out_lo, min(out_hi, shifted)))
    return y
