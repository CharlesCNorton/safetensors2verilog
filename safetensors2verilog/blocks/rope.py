"""RoPE (Rotary Position Embedding).

Per pair (q[2i], q[2i+1]) at position p:
  q'[2i]   = q[2i] * cos(theta_i_p) - q[2i+1] * sin(theta_i_p)
  q'[2i+1] = q[2i] * sin(theta_i_p) + q[2i+1] * cos(theta_i_p)
where theta_i_p = p / theta_base ** (2i / head_dim).

Combinational block: all head_dim/2 pair rotations in parallel. sin and
cos ROMs baked in as ``max_seq * head_dim/2`` entries each, indexed by
``position * (head_dim/2) + i``. ROM contents are computed offline from
``theta_base`` (the model's ``rope_theta``).

For SmolLM2 (head_dim=64, max_seq=8192, theta=100000): 262144 entries
per ROM at 16-bit signed = 512 KB per ROM, 1 MB total per RoPE module.
Block-RAM friendly. The same module is shared across all positions
(only one instance per head_dim configuration).

Sin/cos are stored as Q1.15 signed integers (range [-32768, 32767]
representing [-1, ~1)). Output bits = act_bits.
"""
from __future__ import annotations

import math
from textwrap import dedent

from ..core import Gate, RawSubmodule


def rope_module_name(head_dim: int, max_seq: int, abits: int) -> str:
    return f"rope_apply_D{head_dim}_S{max_seq}_a{abits}"


def _signed_mask(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def rope_block(
    *,
    head_dim: int,
    max_seq: int,
    theta_base: float = 10000.0,
    abits: int = 8,
    sincos_bits: int = 16,
    sincos_frac_bits: int = 15,
    module_suffix: str = "",
) -> RawSubmodule:
    """Combinational RoPE for one (q or k) head.

    head_dim: must be even. SmolLM2 uses 64.
    max_seq:  positions 0..max_seq-1 (must cover the longest sequence).
    theta_base: model config's rope_theta (10000 for original LLaMA, 100000
                for SmolLM2 / Qwen, etc.).
    abits:    bit width of each input/output element (signed).
    sincos_bits: bit width of the sin/cos ROM entries (signed Q1.frac).
                 16 bits at frac=15 gives ~30 bits of effective precision.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    pos_bits = max(1, (max_seq - 1).bit_length())
    rom_depth = max_seq * half
    addr_bits = max(1, (rom_depth - 1).bit_length())

    base = rope_module_name(head_dim, max_seq, abits)
    name = base + (f"_{module_suffix}" if module_suffix else "")

    # Build sin and cos ROMs.
    sin_lut: list[int] = []
    cos_lut: list[int] = []
    sincos_max = (1 << (sincos_bits - 1)) - 1
    for p in range(max_seq):
        for i in range(half):
            theta = p / (theta_base ** (2 * i / head_dim))
            s_int = max(-sincos_max - 1,
                        min(sincos_max, round(math.sin(theta) * (1 << sincos_frac_bits))))
            c_int = max(-sincos_max - 1,
                        min(sincos_max, round(math.cos(theta) * (1 << sincos_frac_bits))))
            sin_lut.append(_signed_mask(s_int, sincos_bits))
            cos_lut.append(_signed_mask(c_int, sincos_bits))

    sin_init = "\n".join(
        f"    sin_rom[{i}] = {sincos_bits}'h{sin_lut[i]:x};"
        for i in range(rom_depth)
    )
    cos_init = "\n".join(
        f"    cos_rom[{i}] = {sincos_bits}'h{cos_lut[i]:x};"
        for i in range(rom_depth)
    )

    # Per-pair combinational logic
    pair_blocks: list[str] = []
    for i in range(half):
        idx_expr = f"position * {half} + {i}"
        even_lo = (2 * i) * abits
        even_hi = even_lo + abits - 1
        odd_lo = (2 * i + 1) * abits
        odd_hi = odd_lo + abits - 1
        pair_blocks.append(dedent(f"""\
          // -- pair {i} --
          wire signed [{abits-1}:0] x_even_{i} = $signed(x_packed[{even_hi}:{even_lo}]);
          wire signed [{abits-1}:0] x_odd_{i}  = $signed(x_packed[{odd_hi}:{odd_lo}]);
          wire signed [{sincos_bits-1}:0] s_{i} = $signed(sin_rom[{idx_expr}]);
          wire signed [{sincos_bits-1}:0] c_{i} = $signed(cos_rom[{idx_expr}]);
          wire signed [{abits + sincos_bits - 1}:0] e_cos_{i} = x_even_{i} * c_{i};
          wire signed [{abits + sincos_bits - 1}:0] e_sin_{i} = x_even_{i} * s_{i};
          wire signed [{abits + sincos_bits - 1}:0] o_cos_{i} = x_odd_{i}  * c_{i};
          wire signed [{abits + sincos_bits - 1}:0] o_sin_{i} = x_odd_{i}  * s_{i};
          wire signed [{abits + sincos_bits}:0] y_even_wide_{i} = e_cos_{i} - o_sin_{i};
          wire signed [{abits + sincos_bits}:0] y_odd_wide_{i}  = e_sin_{i} + o_cos_{i};
          wire signed [{abits + sincos_bits}:0] y_even_shift_{i} = y_even_wide_{i} >>> {sincos_frac_bits};
          wire signed [{abits + sincos_bits}:0] y_odd_shift_{i}  = y_odd_wide_{i}  >>> {sincos_frac_bits};
          // saturate to abits
          assign y_packed[{even_hi}:{even_lo}] =
            (y_even_shift_{i} > {(1 << (abits-1)) - 1})  ? {abits}'sd{(1 << (abits-1)) - 1}
          : (y_even_shift_{i} < -{(1 << (abits-1))}) ? -{abits}'sd{(1 << (abits-1))}
          : y_even_shift_{i}[{abits-1}:0];
          assign y_packed[{odd_hi}:{odd_lo}] =
            (y_odd_shift_{i} > {(1 << (abits-1)) - 1})  ? {abits}'sd{(1 << (abits-1)) - 1}
          : (y_odd_shift_{i} < -{(1 << (abits-1))}) ? -{abits}'sd{(1 << (abits-1))}
          : y_odd_shift_{i}[{abits-1}:0];
        """))

    text = dedent(f"""\
        // Generated by safetensors2verilog.blocks.rope.
        // RoPE for one head. head_dim = {head_dim}, max_seq = {max_seq},
        // theta_base = {theta_base}, abits = {abits}, sincos = Q1.{sincos_frac_bits}.
        // Combinational; all {half} rotations evaluate in parallel.

        `default_nettype none

        module {name} (
          input  wire signed [{head_dim*abits-1}:0]  x_packed,
          input  wire        [{pos_bits-1}:0]         position,
          output wire signed [{head_dim*abits-1}:0]  y_packed
        );
          (* ram_style = "block" *)
          reg [{sincos_bits-1}:0] sin_rom [0:{rom_depth-1}];
          (* ram_style = "block" *)
          reg [{sincos_bits-1}:0] cos_rom [0:{rom_depth-1}];

          initial begin
        {sin_init}
          end
          initial begin
        {cos_init}
          end

        """) + "\n".join(pair_blocks) + dedent(f"""

        endmodule

        `default_nettype wire
        """)

    return RawSubmodule(top=name, text=text)


def rope_invoke(
    *,
    instance_name: str,
    parent_x_packed: str,
    parent_position: str,
    head_dim: int,
    max_seq: int,
    theta_base: float = 10000.0,
    abits: int = 8,
    sincos_bits: int = 16,
    sincos_frac_bits: int = 15,
    y_signal: str | None = None,
) -> tuple[RawSubmodule, list[Gate]]:
    sub = rope_block(
        head_dim=head_dim, max_seq=max_seq, theta_base=theta_base,
        abits=abits, sincos_bits=sincos_bits, sincos_frac_bits=sincos_frac_bits,
    )
    if y_signal is None:
        y_signal = f"{instance_name}_y_packed"
    gates = [
        Gate(
            name=y_signal, kind="instance",
            inputs=[parent_x_packed, parent_position],
            attrs={
                "module_name": sub.top, "instance_name": instance_name,
                "input_ports": ["x_packed", "position"],
                "output_port": "y_packed",
            },
            output_width=head_dim * abits, output_signed=True,
        ),
    ]
    return sub, gates
