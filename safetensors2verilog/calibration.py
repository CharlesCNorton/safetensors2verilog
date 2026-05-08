"""Post-training activation calibration for the LLaMA-architecture frontend.

The ``hf_llama`` frontend wires every ``matmul -> requantize`` chain with a
single per-layer right-shift derived analytically from the matmul's K
dimension::

    _matmul_shift(K, wbits) = wbits + ceil(log2(K)) - 2

This is the right *upper bound* shift if every channel has roughly the
same dynamic range and the activations are uniformly distributed. Real
transformer activations are anything but uniform: a few channels carry
most of the energy and the rest are near-zero. Using a uniform shift
either saturates the loud channels (if the shift is too small) or
quantises the quiet channels to zero (if the shift is too large).

PTQ activation calibration fixes this by running a small calibration
batch through the model, observing each requantize site's pre-shift
accumulator distribution per channel, and choosing an asymmetric
``(mul, shift)`` pair per channel that maps the observed range into the
target ``out_bits`` range without saturation.

Public API:

  collect_activation_stats(config, state_dict, token_sequences, *,
                           abits=8, weight_bits=8) -> LlamaCalibration
      Run an int8-quantised reference forward pass over the calibration
      tokens. Returns a ``LlamaCalibration`` carrying per-site, per-channel
      observed-max and observed-99.5th-percentile values.

  derive_requantize_params(stats, *, target_max=120, mul_bits=8,
                           min_shift=0) -> dict[str, dict[str, list[int]]]
      Given the collected stats, pick (mul, shift) for every requantize
      site so that the observed peak maps to ``target_max`` (default 120
      leaves a small safety margin under the int8 saturation point of 127).

The two pieces are separate so callers can inspect the raw stats, swap in
a different percentile / target / safety policy, or re-derive parameters
without re-running the calibration forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .quantize import per_channel_symmetric_int


# --------------------------------------------------------------------------
# Stats collection
# --------------------------------------------------------------------------


REQUANTIZE_SITES = ("q", "k", "v", "o", "gate", "up", "down")


@dataclass
class SiteStats:
    """Per-channel accumulator statistics for one requantize site.

    abs_max:  per-channel maximum absolute accumulator value across the
              calibration tokens. A length-K list of non-negative ints.
    abs_p995: per-channel 99.5th-percentile absolute value (more robust
              against single-token outliers than abs_max). Length K.
    n_tokens: how many tokens contributed to these stats.
    """
    abs_max: list[int]
    abs_p995: list[int]
    n_tokens: int


@dataclass
class LayerCalibration:
    """All requantize sites for one transformer layer."""
    sites: dict[str, SiteStats] = field(default_factory=dict)


@dataclass
class LlamaCalibration:
    """Calibration data for an entire LLaMA-architecture model."""
    layers: list[LayerCalibration] = field(default_factory=list)
    n_tokens: int = 0
    abits: int = 8
    weight_bits: int = 8
    config: dict = field(default_factory=dict)


def _quantize_per_channel_int(
    t: torch.Tensor, *, weight_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric int quant of an [out, in] linear weight.

    Returns (int_weights:int32, scales:fp32 length out).
    """
    qt = per_channel_symmetric_int(t, axis=0, bits=weight_bits)
    return qt.int_values, qt.scales


def _quantize_input_tokens(
    embed_weight: torch.Tensor, abits: int,
) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric int quant of the embedding (matches hf_llama)."""
    qmax = (1 << (abits - 1)) - 1
    fp_max = float(embed_weight.abs().max().item())
    scale = fp_max / qmax if fp_max > 0 else 1.0
    q = (embed_weight / scale).round().clamp(-qmax, qmax).to(torch.int32)
    return q, scale


def _per_channel_stats(
    accumulators: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Collapse a [tokens, channels] int tensor into per-channel abs-max
    and per-channel 99.5th-percentile absolute value.
    """
    abs_t = accumulators.abs()
    abs_max = abs_t.amax(dim=0).tolist()
    if accumulators.shape[0] >= 4:
        # Per-channel quantile across the token axis.
        p995 = torch.quantile(
            abs_t.to(torch.float32), 0.995, dim=0,
        ).round().to(torch.int64).tolist()
    else:
        # Too few tokens for a meaningful quantile; fall back to abs_max.
        p995 = abs_max[:]
    return [int(v) for v in abs_max], [int(v) for v in p995]


def _rms_norm_int_reference(
    x: torch.Tensor, gamma_int: list[int], gamma_bits: int,
    eps: float, abits: int,
) -> torch.Tensor:
    """Bit-rough int reference of the RMSNorm block. Operates on a single
    [K] int activation vector and returns an int activation in ``abits``.

    The hardware RMSNorm uses a fixed-point rsqrt LUT; here we use float
    arithmetic and round at the end. Differences vs the hardware are at
    most a couple of LSBs and don't bias the calibration meaningfully.
    """
    x_f = x.to(torch.float32)
    mean_sq = (x_f * x_f).mean()
    inv_rms = 1.0 / math.sqrt(float(mean_sq.item()) + eps)
    gamma_f = torch.tensor(gamma_int, dtype=torch.float32) / (1 << 14)
    y = x_f * inv_rms * gamma_f
    qmax = (1 << (abits - 1)) - 1
    return y.round().clamp(-qmax, qmax).to(torch.int32)


def _silu_int(x: torch.Tensor, abits: int) -> torch.Tensor:
    """SiLU on an int activation: dequant by 2**(abits-1), apply silu, requant.

    Uses float math; matches the hardware silu LUT to within the Q-format
    rounding error.
    """
    qmax = (1 << (abits - 1)) - 1
    x_f = x.to(torch.float32) / qmax
    y_f = x_f * torch.sigmoid(x_f)
    return (y_f * qmax).round().clamp(-qmax, qmax).to(torch.int32)


def _gelu_int(x: torch.Tensor, abits: int) -> torch.Tensor:
    qmax = (1 << (abits - 1)) - 1
    x_f = x.to(torch.float32) / qmax
    y_f = 0.5 * x_f * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x_f + 0.044715 * x_f**3)))
    return (y_f * qmax).round().clamp(-qmax, qmax).to(torch.int32)


def collect_activation_stats(
    *,
    config: dict,
    state_dict: dict[str, torch.Tensor],
    token_sequences: list[list[int]],
    abits: int = 8,
    weight_bits: int = 8,
    n_layers_override: int | None = None,
    prev_params: list[dict[str, dict[str, list[int]]]] | None = None,
) -> LlamaCalibration:
    """Run an int-arithmetic reference forward pass over the calibration
    tokens and return per-channel accumulator statistics for every
    requantize site.

    Each token in ``token_sequences`` is fed through the embedding,
    through the configured number of transformer layers, and the
    accumulator value at each requantize site is recorded. The model
    weights are int8-quantised per-channel symmetric (matching the
    hf_llama frontend's quant scheme) so the recorded stats reflect the
    distribution the hardware will actually see.

    Notes:
      * KV cache is not modelled; calibration treats each token as
        position 0 of a fresh sequence. For a transformer layer this is
        a reasonable proxy because attention is computed against the
        single-token K/V (the same as the hardware's first-token path).
      * RoPE is skipped (it doesn't change accumulator magnitudes).
      * Attention output is approximated as the V projection itself
        (single-token case). This keeps the o_proj input distribution
        realistic for the first-token path.
    """
    HID = int(config["hidden_size"])
    H = int(config["num_attention_heads"])
    KV = int(config.get("num_key_value_heads", H))
    INTER = int(config["intermediate_size"])
    EPS = float(config.get("rms_norm_eps", 1e-5))
    HIDDEN_ACT = config.get("hidden_act", "silu")
    N_LAYERS = int(config["num_hidden_layers"])
    if n_layers_override is not None:
        N_LAYERS = min(int(n_layers_override), N_LAYERS)
    D = HID // H

    if HIDDEN_ACT not in ("silu", "gelu", "swiglu"):
        raise ValueError(
            f"unknown hidden_act={HIDDEN_ACT!r}; expected silu/gelu/swiglu"
        )
    act_fn = _silu_int if HIDDEN_ACT in ("silu", "swiglu") else _gelu_int

    # Quantise embedding (per-tensor symmetric, matches hf_llama).
    embed_w = state_dict["model.embed_tokens.weight"].to(torch.float32)
    embed_q, _embed_scale = _quantize_input_tokens(embed_w, abits)

    layer_calibs = [LayerCalibration() for _ in range(N_LAYERS)]

    n_tokens = sum(len(s) for s in token_sequences)
    if n_tokens == 0:
        raise ValueError("token_sequences is empty; need at least one token")

    # Accumulator tensors per site, per layer: each is [n_tokens, K].
    site_K = {
        "q": HID, "k": KV * D, "v": KV * D,
        "o": HID, "gate": INTER, "up": INTER, "down": HID,
    }
    accs = [
        {site: [] for site in REQUANTIZE_SITES}
        for _ in range(N_LAYERS)
    ]

    # Pre-quantise every layer's weights once.
    layer_W: list[dict[str, torch.Tensor]] = []
    for li in range(N_LAYERS):
        Wq, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.self_attn.q_proj.weight"],
            weight_bits=weight_bits,
        )
        Wk, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.self_attn.k_proj.weight"],
            weight_bits=weight_bits,
        )
        Wv, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.self_attn.v_proj.weight"],
            weight_bits=weight_bits,
        )
        Wo, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.self_attn.o_proj.weight"],
            weight_bits=weight_bits,
        )
        Wg, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.mlp.gate_proj.weight"],
            weight_bits=weight_bits,
        )
        Wu, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.mlp.up_proj.weight"],
            weight_bits=weight_bits,
        )
        Wd, _ = _quantize_per_channel_int(
            state_dict[f"model.layers.{li}.mlp.down_proj.weight"],
            weight_bits=weight_bits,
        )
        layer_W.append(dict(q=Wq, k=Wk, v=Wv, o=Wo, gate=Wg, up=Wu, down=Wd))

    # Helper: gamma -> int (Q1.14 signed) per the hf_llama scheme.
    def gamma_to_int(gamma_t: torch.Tensor, gbits: int = 16) -> list[int]:
        f = gamma_t.to(torch.float32)
        qmax = (1 << (gbits - 1)) - 1
        scale = 1 << 14
        return [
            max(-qmax - 1, min(qmax, int(round(v * scale))))
            for v in f.tolist()
        ]

    # Pre-extract gammas per layer.
    layer_gammas: list[tuple[list[int], list[int]]] = []
    for li in range(N_LAYERS):
        g1 = gamma_to_int(
            state_dict[f"model.layers.{li}.input_layernorm.weight"]
        )
        g2 = gamma_to_int(
            state_dict[f"model.layers.{li}.post_attention_layernorm.weight"]
        )
        layer_gammas.append((g1, g2))

    # Run per-token forward pass.
    for seq in token_sequences:
        for tok in seq:
            # Embedding: row tok of embed_q.
            hidden = embed_q[tok].clone()  # int32 [HID]
            for li in range(N_LAYERS):
                # input layernorm
                g1, g2 = layer_gammas[li]
                norm1 = _rms_norm_int_reference(
                    hidden, g1, gamma_bits=16, eps=EPS, abits=abits,
                )
                # q/k/v matmuls (int)
                Wq = layer_W[li]["q"]; Wk = layer_W[li]["k"]; Wv = layer_W[li]["v"]
                q_acc = (Wq.to(torch.int64) @ norm1.to(torch.int64))
                k_acc = (Wk.to(torch.int64) @ norm1.to(torch.int64))
                v_acc = (Wv.to(torch.int64) @ norm1.to(torch.int64))
                accs[li]["q"].append(q_acc.tolist())
                accs[li]["k"].append(k_acc.tolist())
                accs[li]["v"].append(v_acc.tolist())

                # In-chain requantize: prefer the previous-iteration
                # calibrated (mul, shift) when ``prev_params`` is supplied;
                # fall back to the analytical heuristic otherwise. The
                # iterative form (recalibrate using the previous pass's
                # shifts) lets deeper sites observe realistic distributions
                # rather than the underflowed cascade the heuristic alone
                # produces.
                qmax_a = (1 << (abits - 1)) - 1

                def _apply_chained(acc: torch.Tensor, site: str) -> torch.Tensor:
                    if prev_params is not None:
                        site_p = prev_params[li][site]
                        muls_t = torch.tensor(site_p["muls"], dtype=torch.int64)
                        shifts_l = list(site_p["shifts"])
                        # element-wise: (acc * mul) >> shift, per channel.
                        scaled = acc * muls_t
                        out = torch.empty_like(acc)
                        for ch in range(acc.shape[0]):
                            sh = int(shifts_l[ch])
                            v = int(scaled[ch].item())
                            out[ch] = v >> sh if sh >= 0 else v << -sh
                        return out.clamp(-qmax_a, qmax_a)
                    qkv_shift_local = weight_bits + max(
                        1, (acc.shape[0] - 1).bit_length()
                    ) - 2
                    return (acc >> qkv_shift_local).clamp(-qmax_a, qmax_a)

                qkv_shift = weight_bits + max(1, (HID - 1).bit_length()) - 2
                q_int = _apply_chained(q_acc, "q")
                v_int = _apply_chained(v_acc, "v")

                # Attention: single-token approximation -> attn_out = v
                # repeated to fill H heads. (For the first token the score
                # is just the self-attention against the single K vector.)
                # We fill H D-vectors by tiling the single-head V across
                # H/KV groups, matching how grouped-query attention shares
                # KV across query heads.
                attn_out = v_int.view(KV, D).repeat_interleave(H // KV, dim=0).flatten()
                # Cap in int8 range to mirror what the attention block emits.
                attn_int = attn_out.clamp(-qmax_a, qmax_a)

                # o_proj
                Wo = layer_W[li]["o"]
                o_acc = (Wo.to(torch.int64) @ attn_int.to(torch.int64))
                accs[li]["o"].append(o_acc.tolist())
                o_int = _apply_chained(o_acc, "o")

                # Residual 1
                hidden = (hidden + o_int).clamp(-qmax_a, qmax_a)

                # post-attention layernorm
                norm2 = _rms_norm_int_reference(
                    hidden, g2, gamma_bits=16, eps=EPS, abits=abits,
                )
                # gate/up matmuls
                Wg = layer_W[li]["gate"]; Wu = layer_W[li]["up"]
                g_acc = (Wg.to(torch.int64) @ norm2.to(torch.int64))
                u_acc = (Wu.to(torch.int64) @ norm2.to(torch.int64))
                accs[li]["gate"].append(g_acc.tolist())
                accs[li]["up"].append(u_acc.tolist())
                g_int = _apply_chained(g_acc, "gate")
                u_int = _apply_chained(u_acc, "up")
                # SiLU(gate) * up, with the elt_mul shift baked in.
                silu_g = act_fn(g_int, abits)
                # Elt-wise mul into product, shift back to int8.
                elt = ((silu_g * u_int) >> 4).clamp(-qmax_a, qmax_a)

                # down_proj
                Wd = layer_W[li]["down"]
                d_acc = (Wd.to(torch.int64) @ elt.to(torch.int64))
                accs[li]["down"].append(d_acc.tolist())
                d_int = _apply_chained(d_acc, "down")

                # Residual 2
                hidden = (hidden + d_int).clamp(-qmax_a, qmax_a)

    # Reduce: accumulators for each (layer, site) become per-channel stats.
    for li in range(N_LAYERS):
        for site in REQUANTIZE_SITES:
            stacked = torch.tensor(accs[li][site], dtype=torch.int64)
            abs_max, p995 = _per_channel_stats(stacked)
            layer_calibs[li].sites[site] = SiteStats(
                abs_max=abs_max, abs_p995=p995, n_tokens=stacked.shape[0],
            )

    return LlamaCalibration(
        layers=layer_calibs,
        n_tokens=n_tokens,
        abits=abits,
        weight_bits=weight_bits,
        config=dict(config),
    )


# --------------------------------------------------------------------------
# Parameter derivation
# --------------------------------------------------------------------------


def _pick_mul_shift(
    target_to_observed_ratio: float,
    mul_bits: int = 8,
    min_shift: int = 0,
) -> tuple[int, int]:
    """Pick (mul, shift) so that ``observed * mul / 2**shift ~= target``.

    Grows ``shift`` until the multiplier fills the available ``mul_bits``
    range (giving the highest-precision representation), then backs off
    if the rounded mul overflowed. This minimises the integer-rounding
    error in the projection ``observed * mul >> shift``; using a coarser
    mul (like 1) at a small shift can overshoot the target by up to 2x
    when ``observed >> shift`` itself is already close to ``target``.
    """
    if target_to_observed_ratio <= 0:
        return 0, max(0, min_shift)
    mul_max = (1 << (mul_bits - 1)) - 1
    s = abs(target_to_observed_ratio)
    sign = 1 if target_to_observed_ratio > 0 else -1
    # Grow shift while the unrounded mul is comfortably under mul_max.
    shift = max(0, min_shift)
    while shift < 62 and s * (1 << (shift + 1)) <= mul_max:
        shift += 1
    mul = sign * round(s * (1 << shift))
    # Back off one bit if rounding pushed us over mul_max.
    if abs(mul) > mul_max and shift > 0:
        shift -= 1
        mul = sign * round(s * (1 << shift))
    if abs(mul) > mul_max:
        mul = sign * mul_max
    if mul == 0:
        # Ratio is so small that even at shift=62 round(s * 2^62) = 0;
        # the channel will be all-zero post-requantize.
        mul = sign * max(1, round(s * (1 << shift))) if s > 0 else 0
    return int(mul), int(shift)


def derive_requantize_params(
    stats: LlamaCalibration,
    *,
    target_max: int = 120,
    mul_bits: int = 8,
    use_p995: bool = True,
    min_shift: int = 0,
) -> list[dict[str, dict[str, list[int]]]]:
    """Derive per-channel (muls, shifts) for every requantize site of every
    layer.

    target_max:  maximum int8 magnitude the requantize output should reach.
                 Default 120 leaves a small safety headroom under 127.
    mul_bits:    width of the multiplier (signed). The hf_llama frontend's
                 default requantize block is mul_bits=8.
    use_p995:    if True (default), calibrate against the 99.5th percentile
                 absolute value; if False, calibrate against the observed
                 max (less robust to single-token outliers but never
                 saturates on the calibration set).
    min_shift:   minimum shift to issue. Increase if the synth tool
                 complains about excessive multiplier widths or the
                 per-channel scales bunch too close together.

    Returns a list of length ``n_layers``; each entry maps site name
    (``q`` / ``k`` / ``v`` / ``o`` / ``gate`` / ``up`` / ``down``) to a
    dict ``{"muls": [...], "shifts": [...]}``.
    """
    out: list[dict[str, dict[str, list[int]]]] = []
    for layer_calib in stats.layers:
        layer_out: dict[str, dict[str, list[int]]] = {}
        for site, ss in layer_calib.sites.items():
            ref = ss.abs_p995 if use_p995 else ss.abs_max
            muls: list[int] = []
            shifts: list[int] = []
            for ch_max in ref:
                if ch_max == 0:
                    muls.append(0)
                    shifts.append(min_shift if min_shift > 0 else 0)
                    continue
                ratio = target_max / float(ch_max)
                m, s = _pick_mul_shift(
                    ratio, mul_bits=mul_bits, min_shift=min_shift,
                )
                muls.append(m)
                shifts.append(s)
            layer_out[site] = {"muls": muls, "shifts": shifts}
        out.append(layer_out)
    return out


def calibrate_iteratively(
    *,
    config: dict,
    state_dict: dict[str, torch.Tensor],
    token_sequences: list[list[int]],
    abits: int = 8,
    weight_bits: int = 8,
    n_iterations: int = 3,
    target_max: int = 80,
    mul_bits: int = 8,
    use_p995: bool = False,
    n_layers_override: int | None = None,
) -> tuple[LlamaCalibration, list[dict[str, dict[str, list[int]]]]]:
    """Run ``collect_activation_stats`` + ``derive_requantize_params`` in
    a fixed-point iteration.

    Iteration 0: heuristic shifts in the chain (no prior calibration).
    Iteration k: use iteration k-1's calibrated params for the chained
    requantizes, then re-derive params from the resulting stats.

    On synthetic tiny shapes (the test fixture) the iteration converges
    smoothly: each round lifts the down site out of underflow without
    blowing up the upstream sites. On real SmolLM2-shape models the
    iteration can oscillate because the residual path amplifies
    calibration changes across sites (round 1 over-amplifies o, round 2
    saturates the residual into post-attention-norm, round 3 underflows
    gate/up). A damped iteration that interpolates between rounds (for
    example ``new_mul = 0.5 * (old_mul + raw_new_mul)``) stabilises this;
    the primitive provided here is the undamped form, surfaced for
    callers that want to compose their own damping scheme on top.

    Returns (final stats, final params).
    """
    params: list[dict[str, dict[str, list[int]]]] | None = None
    stats: LlamaCalibration | None = None
    for _ in range(max(1, n_iterations)):
        stats = collect_activation_stats(
            config=config, state_dict=state_dict,
            token_sequences=token_sequences,
            abits=abits, weight_bits=weight_bits,
            n_layers_override=n_layers_override,
            prev_params=params,
        )
        params = derive_requantize_params(
            stats, target_max=target_max, mul_bits=mul_bits,
            use_p995=use_p995,
        )
    assert stats is not None and params is not None
    return stats, params


def saturation_summary(
    params: list[dict[str, dict[str, list[int]]]],
    stats: LlamaCalibration,
    *,
    out_max: int = 127,
) -> dict:
    """Summarise how the derived params would treat the calibration data.

    Returns ``{site: {"sat_pct": float, "underflow_pct": float}}`` averaged
    across layers; ``sat_pct`` is the fraction of (token, channel) pairs
    that would saturate at out_max under the derived (mul, shift), and
    ``underflow_pct`` is the fraction that would round to 0.
    """
    summary: dict[str, dict[str, float]] = {}
    for site in REQUANTIZE_SITES:
        sat = 0
        under = 0
        total = 0
        for li, layer_calib in enumerate(stats.layers):
            ss = layer_calib.sites.get(site)
            if ss is None:
                continue
            site_p = params[li][site]
            for ch, max_v in enumerate(ss.abs_max):
                m = site_p["muls"][ch]
                s = site_p["shifts"][ch]
                proj = (max_v * m) >> s if s >= 0 else (max_v * m) << -s
                if proj >= out_max:
                    sat += 1
                if max_v != 0 and proj < 1:
                    under += 1
                total += 1
        summary[site] = {
            "sat_pct": 100.0 * sat / max(1, total),
            "underflow_pct": 100.0 * under / max(1, total),
        }
    return summary
