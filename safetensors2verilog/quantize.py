"""Post-training quantization utilities.

The frontends compile *integer* operations to Verilog. Modern transformer
weights ship as bf16, fp16, or fp8; this module bridges that gap with
post-training quantization (PTQ).

Two schemes are exposed:

  per_tensor_symmetric_int(t, bits)
    One scale for the whole tensor. Smallest scale-tensor footprint;
    works well for weight matrices that don't have outlier rows.

  per_channel_symmetric_int(t, axis, bits)
    One scale per output channel (default: the first axis, matching
    PyTorch's [out, in] linear-weight layout). Substantially better
    accuracy than per-tensor for transformer weights, especially with
    aggressive bit widths (4, 3, 2). The standard choice.

Both schemes are *symmetric* (no zero point) and *signed*: the
quantised range is ``[-(2**(bits-1) - 1), 2**(bits-1) - 1]`` so that
zero is exactly representable. This avoids zero-point arithmetic in
the hardware multipliers.

For activation quantization at runtime there are two common paths:

  *Static activation quant*: pick a scale per layer offline from a
    calibration set, bake it in. Used here.
  *Dynamic activation quant*: compute the scale from the current token
    each forward pass. Cheap on CPU/GPU, expensive on FPGA. Avoid.

The activation quantization scale is not chosen here; the LM frontend
picks it from a calibration pass over a small dataset (deferred). For
the early bring-up we use ``activation_scale = weight_scale.max() /
weight_scale.mean()`` heuristics — fine for verification, not for
quality.

Float-format coercion:

  load_to_bf16(tensor)
    Accepts bf16, fp16, fp32, fp64, float8_e4m3fn, float8_e5m2 (when
    PyTorch supports them) and casts to fp32 for downstream PTQ.
    fp8-quantised checkpoints decode their packed weights to fp32 with
    no extra hardware cost — the dequantisation happens in the host
    loader, not on the FPGA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class QuantizedTensor:
    """A quantised tensor plus the metadata needed to reconstruct it.

    int_values:   integer tensor (dtype int32 for headroom; values fit in
                  ``bits``-bit signed range)
    scales:       float tensor of scales; broadcasts against ``int_values``
                  along the quantised axes
    bits:         signed quantization width (the integer fits in
                  ``[-(2**(bits-1) - 1), 2**(bits-1) - 1]``)
    axis:         tensor axis along which scales vary; ``None`` for
                  per-tensor quantisation
    fp_max_abs:   max abs value of the original tensor (for diagnostics)
    rmse:         root-mean-square reconstruction error vs the original
                  (for diagnostics; computed if ``track_error=True``)
    """
    int_values: torch.Tensor
    scales: torch.Tensor
    bits: int
    axis: int | None
    fp_max_abs: float
    rmse: float | None = None


def load_to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Coerce any supported floating-point or fp8 tensor to fp32.

    PyTorch fp8 dtypes (``torch.float8_e4m3fn``, ``torch.float8_e5m2``)
    cast cleanly to fp32 with the same numeric value; the hardware impact
    of that decode happens once in the loader, not in the synthesised
    design. Integer tensors are returned as-is (cast through fp32 to
    avoid type confusion downstream).
    """
    if t.dtype.is_floating_point:
        return t.to(torch.float32)
    fp8_names = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
                 "float8_e5m2fnuz")
    dt = str(t.dtype).replace("torch.", "")
    if dt in fp8_names:
        return t.to(torch.float32)
    return t.to(torch.float32)


def per_tensor_symmetric_int(
    t: torch.Tensor, *, bits: int = 8, track_error: bool = False,
) -> QuantizedTensor:
    """One scale for the whole tensor."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    f = load_to_fp32(t)
    qmax = (1 << (bits - 1)) - 1   # symmetric, exclude -2**(bits-1)
    fp_max_abs = float(f.abs().max().item())
    if fp_max_abs == 0.0:
        scale = torch.tensor(1.0)
        int_values = torch.zeros_like(f, dtype=torch.int32)
    else:
        scale = torch.tensor(fp_max_abs / qmax)
        int_values = (
            f.div(scale).round().clamp(-qmax, qmax).to(torch.int32)
        )
    rmse = None
    if track_error:
        recon = int_values.float() * scale
        rmse = float((f - recon).pow(2).mean().sqrt().item())
    return QuantizedTensor(
        int_values=int_values,
        scales=scale,
        bits=bits,
        axis=None,
        fp_max_abs=fp_max_abs,
        rmse=rmse,
    )


def per_channel_symmetric_int(
    t: torch.Tensor, *, axis: int = 0, bits: int = 8,
    track_error: bool = False,
) -> QuantizedTensor:
    """One scale per slice along ``axis``.

    For PyTorch linear weights ``[out, in]``, ``axis=0`` gives one scale
    per output channel — the standard scheme.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    f = load_to_fp32(t)
    qmax = (1 << (bits - 1)) - 1
    if axis < 0:
        axis = f.dim() + axis
    if not (0 <= axis < f.dim()):
        raise ValueError(f"axis {axis} out of range for shape {tuple(f.shape)}")

    other_axes = tuple(i for i in range(f.dim()) if i != axis)
    fp_max = f.abs()
    for a in sorted(other_axes, reverse=True):
        fp_max = fp_max.amax(dim=a, keepdim=True)
    fp_max_abs = float(fp_max.max().item())

    fp_max_clamped = fp_max.clamp_min(1e-12)
    scales = fp_max_clamped / qmax

    int_values = (
        f.div(scales).round().clamp(-qmax, qmax).to(torch.int32)
    )
    rmse = None
    if track_error:
        recon = int_values.float() * scales
        rmse = float((f - recon).pow(2).mean().sqrt().item())

    # Squeeze scales down to a 1-D tensor along `axis` for cleaner storage
    scales_squeezed = scales.squeeze().reshape(-1)
    return QuantizedTensor(
        int_values=int_values,
        scales=scales_squeezed,
        bits=bits,
        axis=axis,
        fp_max_abs=fp_max_abs,
        rmse=rmse,
    )


# --------------------------------------------------------------------------
# fp8 e4m3 native quantisation
# --------------------------------------------------------------------------


def _fp32_to_fp8_e4m3_bits(x: float) -> int:
    """Convert one fp32 value to its 8-bit fp8 e4m3 bit pattern.

    Format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
    Special-cases:
      x = 0     -> 0x00
      x = +inf  -> 0x7e (one less than NaN, e4m3 doesn't have +/-inf)
      x = NaN   -> 0x7f
      saturating: values above the max representable (~448) clamp to max.

    Matches the OCP MX standard for FP8 E4M3 (no infinities; NaN is
    encoded as 0x7f / 0xff).
    """
    if x != x:  # NaN
        return 0x7f
    if x == 0.0:
        return 0x00
    sign = 0x80 if x < 0 else 0
    a = abs(x)
    # Max representable e4m3 = (1 + 7/8) * 2^8 = 15/8 * 256 = 480; the
    # OCP spec uses 448 as the "maxnormal" via the bit pattern 0x7e
    # (mantissa 0b110, exp 0b1111). Saturate above that.
    max_normal = 448.0
    if a >= max_normal:
        return sign | 0x7e
    # Decompose to mantissa/exponent.
    m, e = math.frexp(a)   # a = m * 2^e, 0.5 <= m < 1
    # Rebase to 1 <= m < 2.
    m *= 2.0
    e -= 1
    biased_e = e + 7
    if biased_e <= 0:
        # Subnormal: m_norm = a * 2^(7-1) / 2^biased_e = a / 2^(-7+1)
        # round the 3-bit mantissa.
        sub = round(a * (1 << 9))   # = a / 2^-9
        if sub == 0:
            return sign
        return sign | (sub & 0x7f)
    if biased_e >= 16:
        return sign | 0x7e   # saturate
    # Normal: mantissa m_int = round((m - 1) * 8) within [0, 7].
    m_int = round((m - 1.0) * 8.0)
    if m_int >= 8:
        m_int = 0
        biased_e += 1
        if biased_e >= 16:
            return sign | 0x7e
    return sign | (biased_e << 3) | m_int


def _fp8_e4m3_bits_to_fp32(bits: int) -> float:
    """Inverse: decode an 8-bit fp8 e4m3 pattern back to fp32."""
    bits &= 0xff
    sign = -1.0 if (bits & 0x80) else 1.0
    payload = bits & 0x7f
    if payload == 0x7f:
        return float("nan")
    biased_e = (payload >> 3) & 0x0f
    m_int = payload & 0x07
    if biased_e == 0:
        # Subnormal: value = sign * m_int * 2^-9
        return sign * m_int * (2 ** -9)
    e = biased_e - 7
    return sign * (1.0 + m_int / 8.0) * (2.0 ** e)


def fp8_e4m3_quantize(t: torch.Tensor) -> torch.Tensor:
    """Quantise an fp32 tensor to fp8 e4m3 bit patterns (returned as int8).

    Each element is independently rounded to the nearest representable
    fp8 e4m3 value. Saturating: values above the max representable
    magnitude (448) clamp to the maxnormal pattern. Output dtype is
    ``torch.int8`` carrying the 8-bit raw pattern.

    Use the ``fp8_e4m3_dequantize`` companion to round-trip back to
    fp32 for comparison; the round-trip error is the fp8 quantisation
    noise (substantially less than int8 PTQ noise on transformer
    weights, which is why fp8 is preferred).
    """
    f = t.to(torch.float32).flatten().tolist()
    bits = [_fp32_to_fp8_e4m3_bits(float(x)) for x in f]
    raw = torch.tensor(bits, dtype=torch.uint8).view(t.shape).to(torch.int8)
    return raw


def fp8_e4m3_dequantize(raw: torch.Tensor) -> torch.Tensor:
    """Decode fp8 e4m3 bit patterns (in int8 storage) back to fp32."""
    raw_u = raw.to(torch.int64) & 0xff
    flat = raw_u.flatten().tolist()
    out = [_fp8_e4m3_bits_to_fp32(int(b)) for b in flat]
    return torch.tensor(out, dtype=torch.float32).view(raw.shape)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct an approximation to the original fp32 tensor."""
    if qt.axis is None:
        return qt.int_values.to(torch.float32) * qt.scales
    shape = [1] * qt.int_values.dim()
    shape[qt.axis] = -1
    return qt.int_values.to(torch.float32) * qt.scales.reshape(shape)
