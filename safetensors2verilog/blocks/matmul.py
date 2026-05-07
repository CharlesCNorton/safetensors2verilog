"""Sequential-MAC matrix-multiply block.

Architecture:
  - M output neurons computed in parallel (one MAC unit per output).
  - K input dimensions visited sequentially over K cycles.
  - Per-output weight ROM, addressed by a shared cycle counter.
  - Activation buffer latched on ``start``; freed by ``done``.
  - Output accumulator preloads to bias on ``start``, accumulates
    weight*activation each COMPUTE cycle.

Latency: K + 1 cycles from ``start`` pulse to ``done`` pulse.
Resource cost (per matmul instance):
  - K * ABITS flops (activation buffer)
  - M * OBITS flops (accumulators)
  - M * K * WBITS bits of ROM (weights, BRAM-inferable)
  - M parallel signed multipliers (DSP-inferable)
  - 1 small FSM (state, counter)

Module name encodes the shape:
  matmul_seq_M{M}_K{K}_w{WBITS}_a{ABITS}_o{OBITS}

so a matmul with the same shape parameters used in two places in a graph
deduplicates to one Verilog module emission. Weight contents are baked in
via ``initial begin ... end`` for now (works for total weight count up to
a few hundred thousand entries); larger matmuls should switch to
``$readmemh`` sidecar files (deferred for the lm_head scale).

Public API:
  matmul_seq_block(weights, *, weight_bits, act_bits, out_bits, biases) -> RawSubmodule
      Build the parameterized Verilog module text for one matmul shape.

  matmul_seq_invoke(parent_module_signals, ..., weights, biases, ...)
      -> (RawSubmodule, list[Gate])
      Build both the submodule and the IR Gates that instantiate it inside
      a parent GateGraph (concat input scalars to packed bus, instance Gate
      with extra_output_ports for done, slice gates to unpack the output).
"""

from __future__ import annotations

import math
from textwrap import dedent

from ..core import Gate, RawSubmodule


def matmul_seq_module_name(
    M: int, K: int, weight_bits: int, act_bits: int, out_bits: int,
) -> str:
    """Canonical Verilog module name for a matmul of the given shape.

    Choosing the same shape twice yields the same module name and the
    backend deduplicates emission. Shapes differing only in weight values
    collide on this name — caller must distinguish such cases by passing
    a unique ``module_suffix`` to ``matmul_seq_block``.
    """
    return (
        f"matmul_seq_M{M}_K{K}_w{weight_bits}_a{act_bits}_o{out_bits}"
    )


def _required_out_bits(K: int, weight_bits: int, act_bits: int) -> int:
    """Minimum accumulator width for a lossless K-input signed dot product.

    Per-MAC product: act_bits + weight_bits bits.
    Sum of K such products: + ceil(log2(K)) bits.
    Plus 1 bit for bias addition headroom.
    """
    return act_bits + weight_bits + max(1, (K - 1).bit_length()) + 1


def _signed_mask(value: int, width: int) -> int:
    """Two's-complement bit pattern for `value` in `width` bits."""
    mask = (1 << width) - 1
    return value & mask


def matmul_seq_block(
    weights: list[list[int]],
    *,
    weight_bits: int,
    act_bits: int,
    out_bits: int | None = None,
    biases: list[int] | None = None,
    module_suffix: str = "",
    use_dsp: bool = True,
    use_block_ram: bool = True,
) -> RawSubmodule:
    """Emit a Verilog matmul module computing y = W @ x + b.

    weights:      [M][K] list of signed integers within ``weight_bits``
    weight_bits:  bit width of each weight (signed two's complement)
    act_bits:     bit width of each input activation (signed two's complement)
    out_bits:     bit width of each output element (signed). Defaults to
                  the lossless minimum: ceil(log2(K)) + weight_bits +
                  act_bits + 1.
    biases:       optional length-M list of integers (preloaded into the
                  accumulator on ``start``). Default: zero bias.
    module_suffix: optional string appended to the module name when two
                   matmuls share the (M, K, bit_widths) shape but have
                   different weight contents.
    use_dsp:      emit ``(* use_dsp = "yes" *)`` on the multiplier so
                  Vivado / Quartus map it to a hard DSP block.
    use_block_ram: emit ``(* ram_style = "block" *)`` on the weight ROMs
                  so they map to BRAM rather than LUTRAM.

    Returns a ``RawSubmodule`` whose ``top`` is the canonical module name
    (with ``module_suffix`` applied) and whose ``text`` is the full module
    source. Add it to your ``GateGraph.submodules`` and reference its
    ``top`` from an ``instance`` Gate.
    """
    M = len(weights)
    if M == 0:
        raise ValueError("weights must have at least one row")
    K = len(weights[0])
    if K == 0:
        raise ValueError("weights row must have at least one column")
    for j, row in enumerate(weights):
        if len(row) != K:
            raise ValueError(
                f"row {j} has length {len(row)}, expected {K}"
            )

    biases = list(biases) if biases is not None else [0] * M
    if len(biases) != M:
        raise ValueError(
            f"biases length {len(biases)} != M={M}"
        )

    weight_lo = -(1 << (weight_bits - 1))
    weight_hi = (1 << (weight_bits - 1)) - 1
    for j, row in enumerate(weights):
        for i, w in enumerate(row):
            if not (weight_lo <= w <= weight_hi):
                raise ValueError(
                    f"weights[{j}][{i}]={w} outside "
                    f"[{weight_lo}, {weight_hi}] for weight_bits={weight_bits}"
                )

    if out_bits is None:
        out_bits = _required_out_bits(K, weight_bits, act_bits)
    needed = _required_out_bits(K, weight_bits, act_bits)
    if out_bits < needed:
        raise ValueError(
            f"out_bits={out_bits} too narrow for K={K}, weight_bits="
            f"{weight_bits}, act_bits={act_bits}; needs >= {needed}"
        )

    bias_lo = -(1 << (out_bits - 1))
    bias_hi = (1 << (out_bits - 1)) - 1
    for j, b in enumerate(biases):
        if not (bias_lo <= b <= bias_hi):
            raise ValueError(
                f"biases[{j}]={b} outside [{bias_lo}, {bias_hi}] for "
                f"out_bits={out_bits}"
            )

    base_name = matmul_seq_module_name(M, K, weight_bits, act_bits, out_bits)
    module_name = base_name + (f"_{module_suffix}" if module_suffix else "")

    counter_bits = max(1, (K - 1).bit_length() + 1)

    ram_attr = '(* ram_style = "block" *) ' if use_block_ram else ""
    dsp_attr = '(* use_dsp = "yes" *) ' if use_dsp else ""

    # Per-output blocks: ROM init, MAC, accumulator, slice into y_packed.
    per_output_blocks: list[str] = []
    for j in range(M):
        rom_init = "\n".join(
            f"    rom_{j}[{i}] = {weight_bits}'h"
            f"{_signed_mask(weights[j][i], weight_bits):x};"
            for i in range(K)
        )
        bias_lit = (
            f"{out_bits}'h{_signed_mask(biases[j], out_bits):x}"
        )
        per_output_blocks.append(dedent(f"""\
          // -- output {j} --
          {ram_attr}reg signed [{weight_bits-1}:0] rom_{j} [0:{K-1}];
          reg signed [{out_bits-1}:0] acc_{j};
          {dsp_attr}wire signed [{act_bits+weight_bits-1}:0] prod_{j};
          initial begin
        {rom_init}
          end
          assign prod_{j} = $signed(rom_{j}[counter]) * x_now;
          always @(posedge clk or posedge rst) begin
            if (rst)
              acc_{j} <= 0;
            else if (state == STATE_IDLE && start)
              acc_{j} <= $signed({bias_lit});
            else if (state == STATE_COMPUTE)
              acc_{j} <= acc_{j} + prod_{j};
          end
          assign y_packed[{(j+1)*out_bits-1}:{j*out_bits}] = acc_{j};
        """))

    body = "\n".join(per_output_blocks)

    text = dedent(f"""\
        // Generated by safetensors2verilog.blocks.matmul.
        // Sequential-MAC matrix multiply: y = W @ x + b
        //   M (output dim)  = {M}
        //   K (input dim)   = {K}
        //   weight bits     = {weight_bits}  (signed)
        //   activation bits = {act_bits}      (signed)
        //   output bits     = {out_bits}      (signed)
        //   latency         = K + 1 = {K + 1} cycles from start pulse to done
        // Resource estimate (per instance):
        //   {K * act_bits} activation buffer flops
        //   {M * out_bits} accumulator flops
        //   {M * K * weight_bits} weight ROM bits  ({M} * {K} * {weight_bits})
        //   {M} signed {act_bits}x{weight_bits} multipliers (DSP-inferable)

        `default_nettype none

        module {module_name} (
          input  wire                              clk,
          input  wire                              rst,
          input  wire                              start,
          input  wire        [{K * act_bits - 1}:0]   x_packed,
          output wire                              done,
          output wire signed [{M * out_bits - 1}:0]   y_packed
        );

          // ---- State machine ---------------------------------------------------
          localparam [1:0] STATE_IDLE    = 2'd0;
          localparam [1:0] STATE_COMPUTE = 2'd1;
          localparam [1:0] STATE_DONE    = 2'd2;

          reg [1:0] state;
          reg [{counter_bits - 1}:0] counter;
          wire counter_at_max = (counter == {K - 1});

          always @(posedge clk or posedge rst) begin
            if (rst) begin
              state   <= STATE_IDLE;
              counter <= 0;
            end else case (state)
              STATE_IDLE: begin
                counter <= 0;
                if (start) state <= STATE_COMPUTE;
              end
              STATE_COMPUTE: begin
                counter <= counter + 1'b1;
                if (counter_at_max) state <= STATE_DONE;
              end
              STATE_DONE: state <= STATE_IDLE;
              default:    state <= STATE_IDLE;
            endcase
          end

          assign done = (state == STATE_DONE);

          // ---- Activation buffer (latched on start in IDLE) -------------------
          reg signed [{act_bits - 1}:0] x_buf [0:{K - 1}];
          integer i;
          always @(posedge clk) begin
            if (state == STATE_IDLE && start) begin
              for (i = 0; i < {K}; i = i + 1) begin
                x_buf[i] <= $signed(x_packed[i*{act_bits} +: {act_bits}]);
              end
            end
          end

          wire signed [{act_bits - 1}:0] x_now = x_buf[counter];

          // ---- {M} parallel MAC units ----------------------------------------
        {body}
        endmodule

        `default_nettype wire
        """)

    return RawSubmodule(top=module_name, text=text)


def matmul_seq_invoke(
    *,
    instance_name: str,
    parent_x_signals: list[str],
    parent_clk: str,
    parent_rst: str,
    parent_start: str,
    weights: list[list[int]],
    weight_bits: int,
    act_bits: int,
    out_bits: int | None = None,
    biases: list[int] | None = None,
    y_prefix: str | None = None,
    done_signal: str | None = None,
    use_dsp: bool = True,
    use_block_ram: bool = True,
) -> tuple[RawSubmodule, list[Gate]]:
    """Build both the matmul submodule and the IR Gates to instantiate it.

    The caller appends the returned submodule to ``graph.submodules`` and
    extends ``graph.gates`` with the returned Gate list. The Gates are:
      1. A ``concat`` gate packing ``parent_x_signals`` into one bus.
      2. An ``extern_wire`` gate for the ``done`` output.
      3. The ``instance`` Gate driving a packed ``y_packed`` output.
      4. M ``slice`` gates unpacking ``y_packed`` into individual outputs
         named ``f"{y_prefix}{j}"`` (default y_prefix: ``f"{instance_name}_y"``).

    parent_x_signals: K parent-graph signal names of width ``act_bits`` each.
    parent_clk:       parent-graph signal name driving the matmul's clk.
    parent_rst:       parent-graph signal name driving the matmul's rst.
    parent_start:     parent-graph signal name driving the matmul's start.
    weights, biases:  see ``matmul_seq_block``.
    y_prefix:         prefix for individual unpacked output signal names.
    done_signal:      name of the done output signal (default: ``f"{instance_name}_done"``).

    The unpacked y signals are signed by virtue of the ``slice`` gate's
    ``output_signed=True``; downstream consumers should rely on Verilog's
    signed-arithmetic rules.
    """
    K = len(parent_x_signals)
    if any(len(row) != K for row in weights):
        raise ValueError("weights inner dim must match len(parent_x_signals)")
    M = len(weights)

    sub = matmul_seq_block(
        weights, weight_bits=weight_bits, act_bits=act_bits,
        out_bits=out_bits, biases=biases,
        use_dsp=use_dsp, use_block_ram=use_block_ram,
    )

    # Recover out_bits if the caller didn't specify
    if out_bits is None:
        out_bits = _required_out_bits(K, weight_bits, act_bits)

    if y_prefix is None:
        y_prefix = f"{instance_name}_y"
    if done_signal is None:
        done_signal = f"{instance_name}_done"

    pack_name = f"{instance_name}_x_packed"
    y_pack_name = f"{instance_name}_y_packed"

    gates: list[Gate] = []

    # 1. Concat the K parent x signals into a packed bus.
    # `concat` semantics in this IR: MSB-first concat. The matmul module
    # expects element i at bits [i*ABITS +: ABITS], so element 0 sits in the
    # LSBs. To get LSB-first packing from MSB-first concat, reverse the input
    # list.
    gates.append(Gate(
        name=pack_name,
        kind="concat",
        inputs=list(reversed(parent_x_signals)),
        output_width=K * act_bits,
        output_signed=False,
    ))

    # 2. extern_wire for the done output (the instance drives it).
    gates.append(Gate(
        name=done_signal,
        kind="extern_wire",
        inputs=[],
        attrs={},
        output_width=1,
        output_signed=False,
    ))

    # 3. The instance gate, primary output is the packed y bus.
    gates.append(Gate(
        name=y_pack_name,
        kind="instance",
        inputs=[parent_clk, parent_rst, parent_start, pack_name],
        attrs={
            "module_name": sub.top,
            "instance_name": instance_name,
            "input_ports": ["clk", "rst", "start", "x_packed"],
            "output_port": "y_packed",
            "extra_output_ports": [("done", done_signal)],
        },
        output_width=M * out_bits,
        output_signed=True,
    ))

    # 4. Per-output slice gates extracting individual y_j signals.
    for j in range(M):
        gates.append(Gate(
            name=f"{y_prefix}{j}",
            kind="slice",
            inputs=[y_pack_name],
            attrs={"hi": (j + 1) * out_bits - 1, "lo": j * out_bits},
            output_width=out_bits,
            output_signed=True,
        ))

    return sub, gates
