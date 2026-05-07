"""HuggingFace LlamaForCausalLM frontend.

Emits a one-token-per-call forward pass of a LLaMA-architecture transformer
(works for Llama 2/3, Qwen 2/3, Mistral, SmolLM2 — they share the
``embed -> N x [rmsnorm, attention, residual, rmsnorm, swiglu, residual]
-> rmsnorm -> lm_head`` pattern).

Inputs:
  path     - .safetensors file containing the model weights (HF layout:
             ``model.embed_tokens.weight``, ``model.layers.N.*``, etc.)
  config   - path to the config.json sitting next to the safetensors. The
             frontend reads ``hidden_size``, ``num_hidden_layers``,
             ``num_attention_heads``, ``num_key_value_heads``,
             ``intermediate_size``, ``vocab_size``, ``max_position_embeddings``,
             ``rope_theta``, ``rms_norm_eps``, ``hidden_act``, and
             ``tie_word_embeddings``.

Quantization: per-channel symmetric int8 on weights via
``safetensors2verilog.quantize.per_channel_symmetric_int``. Activations
are int8 throughout the pipeline; per-block requantize gates apply a
fixed shift after each matmul. (Static activation calibration is a TODO;
the current scheme uses heuristic shifts that work for small synthetic
models. Real models need calibration to avoid saturation in the deeper
layers.)

The emitted top-level module's interface:

  module top (
    input  wire                            clk,
    input  wire                            rst,
    input  wire                            start,
    input  wire        [TOKEN_BITS-1:0]    token_id,
    input  wire        [POS_BITS-1:0]      position,
    output wire                            done,
    output wire signed [VOCAB*OUT_BITS-1:0] logits_packed,
    output wire        [TOKEN_BITS-1:0]    next_token_id   // argmax
  );

Sidecar files: large weight ROMs in matmul / embedding instances may emit
sidecar ``.hex`` files (collected via ``collect_sidecar_files``); the
caller is responsible for writing them next to the .v file.

The frontend is constructed to handle any number of layers; the same
shape parameters used for the tiny verification fixture (hidden=8, etc.)
also drive SmolLM2-135M (hidden=576, 30 layers, vocab=49152) by changing
the values, not the structure.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ..core import (
    Frontend, FrontendOption, Gate, GateGraph, RawSubmodule, Signal, registry,
)
from ..blocks.argmax import argmax_block
from ..blocks.attention import attention_step_block
from ..blocks.embedding import embedding_block
from ..blocks.matmul import matmul_seq_block
from ..blocks.requantize import requantize_block
from ..blocks.rms_norm import rms_norm_block
from ..blocks.rope import rope_block
from ..blocks.silu import silu_block
from ..quantize import per_channel_symmetric_int


# -------- Constants picked to match the tiny_llama_e2e fixture --------
RMS_GBITS = 16
RMS_EPS_Q = 16
RMS_RSQRT_OUT_BITS = 16
RMS_RSQRT_OUT_FRAC = 14
RMS_OUTPUT_SHIFT = 14

ROPE_SINCOS_BITS = 16
ROPE_SINCOS_FRAC = 14

SILU_OUTPUT_SHIFT = 8
ATT_SCORE_SHIFT = 0
ATT_OUT_SHIFT = 8

ELT_MUL_SHIFT = 4   # silu(gate) * up shift

DEFAULT_ACT_BITS = 8
DEFAULT_WEIGHT_BITS = 8


def _matmul_obits(K_in, abits, wbits):
    return abits + wbits + max(1, (K_in - 1).bit_length()) + 1


def _matmul_shift(K_in, wbits):
    """Heuristic requantize shift after a matmul to keep typical outputs in
    the int8 range. Real PTQ would calibrate from activation statistics."""
    return wbits + max(1, (K_in - 1).bit_length()) - 2


def _rsqrt_in_bits(K, abits):
    return 2 * abits + max(1, K.bit_length()) + 4


def _instance(*, name, module, instance_name, input_ports, inputs,
              output_port, output_width, output_signed=True,
              extra_outputs=None) -> Gate:
    attrs = {
        "module_name": module, "instance_name": instance_name,
        "input_ports": input_ports, "output_port": output_port,
    }
    if extra_outputs:
        attrs["extra_output_ports"] = extra_outputs
    return Gate(name=name, kind="instance", inputs=list(inputs),
                attrs=attrs, output_width=output_width,
                output_signed=output_signed)


def _packed_add(*, name_prefix, lhs, rhs, K, abits) -> tuple[list[Gate], str]:
    """Element-wise saturating add of two packed buses. Returns (gates, out_signal)."""
    gs: list[Gate] = []
    elt = []
    for i in range(K):
        ls = f"{name_prefix}.l.{i}"
        rs = f"{name_prefix}.r.{i}"
        sm = f"{name_prefix}.s.{i}"
        st = f"{name_prefix}.sat.{i}"
        gs.append(Gate(name=ls, kind="slice", inputs=[lhs],
                       attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                       output_width=abits, output_signed=True))
        gs.append(Gate(name=rs, kind="slice", inputs=[rhs],
                       attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                       output_width=abits, output_signed=True))
        gs.append(Gate(name=sm, kind="add", inputs=[ls, rs],
                       output_width=abits + 1, output_signed=True))
        lo, hi = -(1 << (abits - 1)), (1 << (abits - 1)) - 1
        gs.append(Gate(name=st, kind="clamp", inputs=[sm],
                       attrs={"lo": lo, "hi": hi},
                       output_width=abits, output_signed=True))
        elt.append(st)
    out = f"{name_prefix}.packed"
    gs.append(Gate(name=out, kind="concat",
                   inputs=list(reversed(elt)),
                   output_width=K * abits, output_signed=True))
    return gs, out


def _packed_mul_shift(*, name_prefix, lhs, rhs, K, abits, shift):
    gs: list[Gate] = []
    elt = []
    for i in range(K):
        ls = f"{name_prefix}.l.{i}"
        rs = f"{name_prefix}.r.{i}"
        pr = f"{name_prefix}.p.{i}"
        sh = f"{name_prefix}.sh.{i}"
        st = f"{name_prefix}.sat.{i}"
        gs.append(Gate(name=ls, kind="slice", inputs=[lhs],
                       attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                       output_width=abits, output_signed=True))
        gs.append(Gate(name=rs, kind="slice", inputs=[rhs],
                       attrs={"hi": (i + 1) * abits - 1, "lo": i * abits},
                       output_width=abits, output_signed=True))
        gs.append(Gate(name=pr, kind="mul", inputs=[ls, rs],
                       output_width=2 * abits, output_signed=True))
        gs.append(Gate(name=sh, kind="shift_right", inputs=[pr],
                       attrs={"amount": shift},
                       output_width=2 * abits, output_signed=True))
        lo, hi = -(1 << (abits - 1)), (1 << (abits - 1)) - 1
        gs.append(Gate(name=st, kind="clamp", inputs=[sh],
                       attrs={"lo": lo, "hi": hi},
                       output_width=abits, output_signed=True))
        elt.append(st)
    out = f"{name_prefix}.packed"
    gs.append(Gate(name=out, kind="concat",
                   inputs=list(reversed(elt)),
                   output_width=K * abits, output_signed=True))
    return gs, out


def _gamma_int_from_bf16(t: torch.Tensor, frac_bits: int = 14) -> list[int]:
    """Quantize an RMSNorm gamma vector to gamma_bits Q-format ints.

    gamma represents a per-channel scale near 1.0; we use Q1.frac_bits
    signed ints (range about [-2, 2)).
    """
    f = t.to(torch.float32)
    scale = 1 << frac_bits
    return [
        max(-(1 << (RMS_GBITS - 1)),
            min((1 << (RMS_GBITS - 1)) - 1, int(round(v * scale))))
        for v in f.tolist()
    ]


def _embed_int_from_bf16(t: torch.Tensor, abits: int = DEFAULT_ACT_BITS,
                         ) -> list[list[int]]:
    """Quantize the embedding table to per-token int8 vectors.

    Per-row symmetric quant (one scale per token) is the standard scheme.
    Here we use per-tensor for simplicity — for actual PTQ on a real model,
    swap to ``per_channel_symmetric_int`` along axis=0.
    """
    f = t.to(torch.float32)
    qmax = (1 << (abits - 1)) - 1
    fp_max = float(f.abs().max().item())
    scale = fp_max / qmax if fp_max > 0 else 1.0
    q = (f / scale).round().clamp(-qmax, qmax).to(torch.int32)
    return q.tolist()


def _quantize_linear_weight(t: torch.Tensor, weight_bits: int,
                            ) -> tuple[list[list[int]], list[float]]:
    """Per-channel int quantization for an [out, in] linear weight.

    Returns (int_weights, scales_per_output_channel).
    """
    qt = per_channel_symmetric_int(t, axis=0, bits=weight_bits)
    return qt.int_values.tolist(), qt.scales.tolist()


def build_llama_graph(
    *,
    config: dict,
    state_dict: dict[str, torch.Tensor],
    top: str = "llama_top",
    abits: int = DEFAULT_ACT_BITS,
    weight_bits: int = DEFAULT_WEIGHT_BITS,
) -> GateGraph:
    """Build the per-token forward-pass GateGraph for a LLaMA-architecture model.

    config: parsed config.json.
    state_dict: tensor name -> torch tensor (any float dtype).
    """
    HID = int(config["hidden_size"])
    N_LAYERS = int(config["num_hidden_layers"])
    H = int(config["num_attention_heads"])
    KV = int(config.get("num_key_value_heads", H))
    INTER = int(config["intermediate_size"])
    VOCAB = int(config["vocab_size"])
    MAX_SEQ = int(config["max_position_embeddings"])
    ROPE_THETA = float(config.get("rope_theta", 10000.0))
    EPS = float(config.get("rms_norm_eps", 1e-5))
    TIE = bool(config.get("tie_word_embeddings", False))

    if H % KV != 0:
        raise ValueError(
            f"num_attention_heads={H} not divisible by "
            f"num_key_value_heads={KV}"
        )
    assert HID % H == 0
    D = HID // H   # head_dim

    submodules: list[RawSubmodule] = []
    gates: list[Gate] = []

    # ---- Embedding ----
    embed_weight = state_dict["model.embed_tokens.weight"]
    if embed_weight.shape != (VOCAB, HID):
        raise ValueError(
            f"embed_tokens shape {tuple(embed_weight.shape)} != "
            f"({VOCAB}, {HID})"
        )
    embed_int = _embed_int_from_bf16(embed_weight, abits=abits)
    embed_sub = embedding_block(
        V=VOCAB, H=HID, abits=abits, weights=embed_int,
        # inline_init=None autoselects based on V*H size
    )
    submodules.append(embed_sub)
    gates.append(_instance(
        name="hidden_packed", module=embed_sub.top, instance_name="embed",
        input_ports=["token_id"], inputs=["token_id"],
        output_port="hidden_packed",
        output_width=HID * abits, output_signed=True,
    ))
    cur_hidden = "hidden_packed"
    cur_done = "_first_done_"   # placeholder; replaced before use

    # First done is the embedding output's "validity". Embed is combinational
    # off token_id; its value is stable as soon as token_id is set. We use
    # the parent module's `start` as the trigger for the first sequential
    # block (RMSNorm). To pass that through, alias start as cur_done.
    # (We can't directly use ``start`` as another block's done because they
    # have the same semantics: a one-cycle pulse.)
    cur_done = "start"

    qkv_shift = _matmul_shift(HID, weight_bits)
    o_shift = qkv_shift
    gate_shift = _matmul_shift(HID, weight_bits)
    down_shift = _matmul_shift(INTER, weight_bits)
    lm_shift = _matmul_shift(HID, weight_bits)

    qkv_obits = _matmul_obits(HID, abits, weight_bits)
    o_obits = qkv_obits
    mlp_obits = _matmul_obits(HID, abits, weight_bits)
    down_obits = _matmul_obits(INTER, abits, weight_bits)
    lm_obits = _matmul_obits(HID, abits, weight_bits)

    rope_sub = rope_block(
        head_dim=D, max_seq=MAX_SEQ, theta_base=ROPE_THETA,
        abits=abits, sincos_bits=ROPE_SINCOS_BITS,
        sincos_frac_bits=ROPE_SINCOS_FRAC,
    )
    submodules.append(rope_sub)

    for li in range(N_LAYERS):
        prefix = f"L{li}"

        # --- Input RMSNorm ---
        gamma1 = _gamma_int_from_bf16(
            state_dict[f"model.layers.{li}.input_layernorm.weight"],
        )
        rms1, rsq = rms_norm_block(
            K=HID, gamma_int=gamma1, gamma_bits=RMS_GBITS,
            abits=abits, obits=abits, eps=EPS, eps_q=RMS_EPS_Q,
            rsqrt_in_bits=_rsqrt_in_bits(HID, abits),
            rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
            rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
            output_shift=RMS_OUTPUT_SHIFT,
            module_suffix=f"{prefix}_in",
        )
        submodules += [rms1, rsq]
        gates.append(Gate(name=f"{prefix}.rms1.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.norm1", module=rms1.top,
            instance_name=f"rms1_{li}",
            input_ports=["clk", "rst", "start", "x_packed"],
            inputs=["clk", "rst", cur_done, cur_hidden],
            output_port="y_packed",
            extra_outputs=[("done", f"{prefix}.rms1.done")],
            output_width=HID * abits, output_signed=True,
        ))

        # --- Q/K/V/O projections ---
        Wq, _sq = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.self_attn.q_proj.weight"], weight_bits)
        Wk, _sk = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.self_attn.k_proj.weight"], weight_bits)
        Wv, _sv = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.self_attn.v_proj.weight"], weight_bits)
        Wo, _so = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.self_attn.o_proj.weight"], weight_bits)

        q_sub = matmul_seq_block(weights=Wq, weight_bits=weight_bits,
                                 act_bits=abits, module_suffix=f"{prefix}_q")
        k_sub = matmul_seq_block(weights=Wk, weight_bits=weight_bits,
                                 act_bits=abits, module_suffix=f"{prefix}_k")
        v_sub = matmul_seq_block(weights=Wv, weight_bits=weight_bits,
                                 act_bits=abits, module_suffix=f"{prefix}_v")
        o_sub = matmul_seq_block(weights=Wo, weight_bits=weight_bits,
                                 act_bits=abits, module_suffix=f"{prefix}_o")
        submodules += [q_sub, k_sub, v_sub, o_sub]

        for tag, sub, K_out_w in (("q", q_sub, HID), ("k", k_sub, KV * D),
                                  ("v", v_sub, KV * D)):
            done_name = f"{prefix}.{tag}.done"
            gates.append(Gate(name=done_name, kind="extern_wire",
                              output_width=1))
            gates.append(_instance(
                name=f"{prefix}.{tag}_wide", module=sub.top,
                instance_name=f"{tag}m_{li}",
                input_ports=["clk", "rst", "start", "x_packed"],
                inputs=["clk", "rst", f"{prefix}.rms1.done",
                        f"{prefix}.norm1"],
                output_port="y_packed",
                extra_outputs=[("done", done_name)],
                output_width=K_out_w * qkv_obits, output_signed=True,
            ))

        for tag, K_w in (("q", HID), ("k", KV * D), ("v", KV * D)):
            rq = requantize_block(
                K=K_w, in_bits=qkv_obits, out_bits=abits,
                muls=[1] * K_w, shifts=[qkv_shift] * K_w, mul_bits=8,
                module_suffix=f"{prefix}_{tag}",
            )
            submodules.append(rq)
            gates.append(_instance(
                name=f"{prefix}.{tag}_int", module=rq.top,
                instance_name=f"rq_{tag}_{li}",
                input_ports=["x_packed"],
                inputs=[f"{prefix}.{tag}_wide"],
                output_port="y_packed",
                output_width=K_w * abits, output_signed=True,
            ))

        # --- RoPE on Q (per head) and K (per head) ---
        def slice_and_rope(src, num_heads, tag):
            rotated = []
            for h in range(num_heads):
                slice_name = f"{prefix}.{tag}.head{h}"
                rot_name = f"{prefix}.{tag}.rot{h}"
                gates.append(Gate(name=slice_name, kind="slice", inputs=[src],
                                  attrs={"hi": (h + 1) * D * abits - 1,
                                         "lo": h * D * abits},
                                  output_width=D * abits, output_signed=True))
                gates.append(_instance(
                    name=rot_name, module=rope_sub.top,
                    instance_name=f"rope_{tag}_{li}_{h}",
                    input_ports=["x_packed", "position"],
                    inputs=[slice_name, "position"],
                    output_port="y_packed",
                    output_width=D * abits, output_signed=True,
                ))
                rotated.append(rot_name)
            cat = f"{prefix}.{tag}.rot.cat"
            gates.append(Gate(name=cat, kind="concat",
                              inputs=list(reversed(rotated)),
                              output_width=num_heads * D * abits,
                              output_signed=True))
            return cat

        q_rot = slice_and_rope(f"{prefix}.q_int", H, "q")
        k_rot = slice_and_rope(f"{prefix}.k_int", KV, "k")

        # --- Attention ---
        att_sub, att_deps = attention_step_block(
            num_q_heads=H, num_kv_heads=KV, head_dim=D, max_seq=MAX_SEQ,
            abits=abits, out_abits=abits,
            score_shift=ATT_SCORE_SHIFT, out_shift=ATT_OUT_SHIFT,
            module_suffix=f"{prefix}",
        )
        submodules += [att_sub] + list(att_deps)
        gates.append(Gate(name=f"{prefix}.att.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.attn_out", module=att_sub.top,
            instance_name=f"att_{li}",
            input_ports=["clk", "rst", "start", "position",
                         "q_packed", "k_new_packed", "v_new_packed"],
            inputs=["clk", "rst", f"{prefix}.v.done", "position",
                    q_rot, k_rot, f"{prefix}.v_int"],
            output_port="out_packed",
            extra_outputs=[("done", f"{prefix}.att.done")],
            output_width=H * D * abits, output_signed=True,
        ))

        # --- O projection + requantize ---
        gates.append(Gate(name=f"{prefix}.o.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.o_wide", module=o_sub.top,
            instance_name=f"om_{li}",
            input_ports=["clk", "rst", "start", "x_packed"],
            inputs=["clk", "rst", f"{prefix}.att.done", f"{prefix}.attn_out"],
            output_port="y_packed",
            extra_outputs=[("done", f"{prefix}.o.done")],
            output_width=HID * o_obits, output_signed=True,
        ))
        rq_o = requantize_block(
            K=HID, in_bits=o_obits, out_bits=abits,
            muls=[1] * HID, shifts=[o_shift] * HID, mul_bits=8,
            module_suffix=f"{prefix}_o",
        )
        submodules.append(rq_o)
        gates.append(_instance(
            name=f"{prefix}.o_int", module=rq_o.top,
            instance_name=f"rq_o_{li}",
            input_ports=["x_packed"], inputs=[f"{prefix}.o_wide"],
            output_port="y_packed",
            output_width=HID * abits, output_signed=True,
        ))

        # --- Residual 1: hidden + o_int ---
        res1_g, res1 = _packed_add(
            name_prefix=f"{prefix}.res1", lhs=cur_hidden, rhs=f"{prefix}.o_int",
            K=HID, abits=abits,
        )
        gates += res1_g

        # --- post_attention_layernorm (RMSNorm 2) ---
        gamma2 = _gamma_int_from_bf16(
            state_dict[f"model.layers.{li}.post_attention_layernorm.weight"],
        )
        rms2, rsq2 = rms_norm_block(
            K=HID, gamma_int=gamma2, gamma_bits=RMS_GBITS,
            abits=abits, obits=abits, eps=EPS, eps_q=RMS_EPS_Q,
            rsqrt_in_bits=_rsqrt_in_bits(HID, abits),
            rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
            rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
            output_shift=RMS_OUTPUT_SHIFT,
            module_suffix=f"{prefix}_post",
        )
        submodules += [rms2, rsq2]
        gates.append(Gate(name=f"{prefix}.rms2.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.norm2", module=rms2.top,
            instance_name=f"rms2_{li}",
            input_ports=["clk", "rst", "start", "x_packed"],
            inputs=["clk", "rst", f"{prefix}.o.done", res1],
            output_port="y_packed",
            extra_outputs=[("done", f"{prefix}.rms2.done")],
            output_width=HID * abits, output_signed=True,
        ))

        # --- MLP: gate_proj, up_proj, silu, mul, down_proj ---
        Wg, _sg = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.mlp.gate_proj.weight"], weight_bits)
        Wu, _su = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.mlp.up_proj.weight"], weight_bits)
        Wd, _sd = _quantize_linear_weight(
            state_dict[f"model.layers.{li}.mlp.down_proj.weight"], weight_bits)

        gate_sub = matmul_seq_block(weights=Wg, weight_bits=weight_bits,
                                    act_bits=abits,
                                    module_suffix=f"{prefix}_gate")
        up_sub = matmul_seq_block(weights=Wu, weight_bits=weight_bits,
                                  act_bits=abits,
                                  module_suffix=f"{prefix}_up")
        down_sub = matmul_seq_block(weights=Wd, weight_bits=weight_bits,
                                    act_bits=abits,
                                    module_suffix=f"{prefix}_down")
        submodules += [gate_sub, up_sub, down_sub]

        for tag, sub, K_out_w in (("gate", gate_sub, INTER),
                                  ("up", up_sub, INTER)):
            done_name = f"{prefix}.{tag}.done"
            gates.append(Gate(name=done_name, kind="extern_wire",
                              output_width=1))
            gates.append(_instance(
                name=f"{prefix}.{tag}_wide", module=sub.top,
                instance_name=f"{tag}m_{li}",
                input_ports=["clk", "rst", "start", "x_packed"],
                inputs=["clk", "rst", f"{prefix}.rms2.done",
                        f"{prefix}.norm2"],
                output_port="y_packed",
                extra_outputs=[("done", done_name)],
                output_width=K_out_w * mlp_obits, output_signed=True,
            ))

        for tag, K_w in (("gate", INTER), ("up", INTER)):
            rq = requantize_block(
                K=K_w, in_bits=mlp_obits, out_bits=abits,
                muls=[1] * K_w, shifts=[gate_shift] * K_w, mul_bits=8,
                module_suffix=f"{prefix}_{tag}",
            )
            submodules.append(rq)
            gates.append(_instance(
                name=f"{prefix}.{tag}_int", module=rq.top,
                instance_name=f"rq_{tag}_{li}",
                input_ports=["x_packed"], inputs=[f"{prefix}.{tag}_wide"],
                output_port="y_packed",
                output_width=K_w * abits, output_signed=True,
            ))

        silu_sub, sig_sub = silu_block(
            K=INTER, abits=abits, obits=abits,
            sigmoid_in_q_frac_bits=4, sigmoid_out_bits=8,
            output_shift=SILU_OUTPUT_SHIFT,
            module_suffix=f"{prefix}",
        )
        submodules += [silu_sub, sig_sub]
        gates.append(Gate(name=f"{prefix}.silu.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.silu_g", module=silu_sub.top,
            instance_name=f"silu_{li}",
            input_ports=["clk", "rst", "start", "x_packed"],
            inputs=["clk", "rst", f"{prefix}.up.done", f"{prefix}.gate_int"],
            output_port="y_packed",
            extra_outputs=[("done", f"{prefix}.silu.done")],
            output_width=INTER * abits, output_signed=True,
        ))

        sup_g, sup = _packed_mul_shift(
            name_prefix=f"{prefix}.silu_up",
            lhs=f"{prefix}.silu_g", rhs=f"{prefix}.up_int",
            K=INTER, abits=abits, shift=ELT_MUL_SHIFT,
        )
        gates += sup_g

        gates.append(Gate(name=f"{prefix}.down.done", kind="extern_wire",
                          output_width=1))
        gates.append(_instance(
            name=f"{prefix}.down_wide", module=down_sub.top,
            instance_name=f"dm_{li}",
            input_ports=["clk", "rst", "start", "x_packed"],
            inputs=["clk", "rst", f"{prefix}.silu.done", sup],
            output_port="y_packed",
            extra_outputs=[("done", f"{prefix}.down.done")],
            output_width=HID * down_obits, output_signed=True,
        ))
        rq_d = requantize_block(
            K=HID, in_bits=down_obits, out_bits=abits,
            muls=[1] * HID, shifts=[down_shift] * HID, mul_bits=8,
            module_suffix=f"{prefix}_down",
        )
        submodules.append(rq_d)
        gates.append(_instance(
            name=f"{prefix}.down_int", module=rq_d.top,
            instance_name=f"rq_d_{li}",
            input_ports=["x_packed"], inputs=[f"{prefix}.down_wide"],
            output_port="y_packed",
            output_width=HID * abits, output_signed=True,
        ))

        # --- Residual 2: res1 + down_int ---
        res2_g, res2 = _packed_add(
            name_prefix=f"{prefix}.res2", lhs=res1, rhs=f"{prefix}.down_int",
            K=HID, abits=abits,
        )
        gates += res2_g

        cur_hidden = res2
        cur_done = f"{prefix}.down.done"

    # --- Final RMSNorm ---
    gamma_final = _gamma_int_from_bf16(state_dict["model.norm.weight"])
    rmsf, rsqf = rms_norm_block(
        K=HID, gamma_int=gamma_final, gamma_bits=RMS_GBITS,
        abits=abits, obits=abits, eps=EPS, eps_q=RMS_EPS_Q,
        rsqrt_in_bits=_rsqrt_in_bits(HID, abits),
        rsqrt_out_bits=RMS_RSQRT_OUT_BITS,
        rsqrt_out_frac_bits=RMS_RSQRT_OUT_FRAC,
        output_shift=RMS_OUTPUT_SHIFT,
        module_suffix="final",
    )
    submodules += [rmsf, rsqf]
    gates.append(Gate(name="rmsf.done", kind="extern_wire", output_width=1))
    gates.append(_instance(
        name="final_norm", module=rmsf.top, instance_name="rmsf",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", cur_done, cur_hidden],
        output_port="y_packed",
        extra_outputs=[("done", "rmsf.done")],
        output_width=HID * abits, output_signed=True,
    ))

    # --- lm_head ---
    if TIE:
        # Use the embedding weights as lm_head
        lm_W = embed_int
    else:
        lm_W, _sl = _quantize_linear_weight(
            state_dict["lm_head.weight"], weight_bits)
    lm_sub = matmul_seq_block(weights=lm_W, weight_bits=weight_bits,
                              act_bits=abits, module_suffix="lm_head")
    submodules.append(lm_sub)
    gates.append(Gate(name="lm.done", kind="extern_wire", output_width=1))
    gates.append(_instance(
        name="logits_packed", module=lm_sub.top, instance_name="lm",
        input_ports=["clk", "rst", "start", "x_packed"],
        inputs=["clk", "rst", "rmsf.done", "final_norm"],
        output_port="y_packed",
        extra_outputs=[("done", "lm.done")],
        output_width=VOCAB * lm_obits, output_signed=True,
    ))

    # --- argmax (combinational) ---
    argmax_sub = argmax_block(K=VOCAB, abits=lm_obits)
    submodules.append(argmax_sub)
    idx_bits = max(1, (VOCAB - 1).bit_length())
    gates.append(_instance(
        name="next_token_id", module=argmax_sub.top, instance_name="am",
        input_ports=["x_packed"], inputs=["logits_packed"],
        output_port="argmax_idx",
        output_width=idx_bits, output_signed=False,
    ))

    # --- Build top-level GateGraph ---
    pos_bits = max(1, (MAX_SEQ - 1).bit_length() + 1)
    parent = GateGraph(
        inputs=[
            Signal("clk"), Signal("rst"), Signal("start"),
            Signal("token_id", width=max(1, (VOCAB - 1).bit_length()),
                   signed=False),
            Signal("position", width=pos_bits, signed=False),
        ],
        outputs=[
            Signal("done", width=1),
            Signal("logits_packed", width=VOCAB * lm_obits, signed=True),
            Signal("next_token_id", width=idx_bits, signed=False),
        ],
        gates=gates + [
            Gate(name="done", kind="or", inputs=["lm.done"],
                 output_width=1, output_signed=False),
        ],
        top=top, submodules=submodules,
    )
    return parent


@registry.register(
    "hf_llama",
    description=(
        "HuggingFace LlamaForCausalLM-architecture transformer "
        "(Llama / Qwen / Mistral / SmolLM2). Per-token forward pass."
    ),
    metadata_namespace="hf_llama",
)
class HFLlamaFrontend(Frontend):

    @classmethod
    def options(cls) -> list[FrontendOption]:
        return [
            FrontendOption(
                name="config",
                type=str,
                default=None,
                help="path to the model's config.json (default: config.json "
                     "next to the safetensors).",
                metavar="PATH",
            ),
            FrontendOption(
                name="activation-bits",
                type=int,
                default=DEFAULT_ACT_BITS,
                help="bit width of int activations.",
            ),
            FrontendOption(
                name="weight-bits",
                type=int,
                default=DEFAULT_WEIGHT_BITS,
                help="bit width of int weights (per-channel symmetric quant).",
            ),
        ]

    def parse(
        self,
        path: Path,
        top: str = "llama_top",
        config: str | None = None,
        activation_bits: int = DEFAULT_ACT_BITS,
        weight_bits: int = DEFAULT_WEIGHT_BITS,
        **options,
    ) -> GateGraph:
        path = Path(path)
        if config is None:
            config_path = path.parent / "config.json"
        else:
            config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found at {config_path}; pass --config PATH"
            )
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        state_dict: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k).clone()

        return build_llama_graph(
            config=cfg, state_dict=state_dict, top=top,
            abits=activation_bits, weight_bits=weight_bits,
        )
