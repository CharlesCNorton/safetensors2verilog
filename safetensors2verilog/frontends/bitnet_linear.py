"""BitNet b1.58 ternary linear frontend.

BitNet b1.58 represents `nn.Linear` weights as ternary {-1, 0, 1};
activations are multi-bit (typically int8). This frontend reads a
safetensors file describing one or more chained ternary linear
layers and emits a dataflow network that computes
``y = (W_n · ... · W_1 · x) + biases``.

Tensor naming convention (matches PyTorch's ``state_dict`` for a
``nn.Sequential([nn.Linear, ...])``):

  <prefix>.<n>.weight    int / float tensor with shape [out, in], values in {-1, 0, 1}
  <prefix>.<n>.bias      int / float tensor with shape [out] (optional)

Default ``prefix`` is ``layers``. Override with ``--layer-prefix``.

The emitted module has one signed input port per element of the first
layer's input vector and one signed output port per element of the
last layer's output vector. Activation width is set by
``--activation-bits``. Per-layer accumulator widths grow to keep the
worst-case MAC sum lossless.

Optional features:
  --output-clamp LO,HI  Wrap each layer's output in a clamp gate
                        (truncates the accumulator to LO..HI). When
                        combined with --activation-bits N, a typical
                        choice is LO=-(2^(N-1)) HI=2^(N-1)-1, which
                        keeps each layer's output back in N bits.
  --pipeline            Register each layer's output; the resulting
                        Verilog has a clk port and one cycle of
                        latency per layer. Combine with --output-clamp
                        for a saturating, pipelined inference core.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from ..core import Frontend, FrontendOption, Gate, GateGraph, Signal, registry


def bitnet_rom_parity_bits(rom_init: list[int]) -> list[int]:
    """Compute even-parity bits for a list of ROM entries.

    Used by callers that want to detect single-bit flips in BitNet's 2-bit
    signed weight ROMs. Returns one bit per entry; emit alongside the
    main ROM, then XOR the read value's bits against the parity bit on
    each access and raise a ``parity_error`` if it doesn't match.

    Note: this gives single-bit detection, not correction. For correction
    a SECDED Hamming code over the 2-bit value would need 3 extra bits
    per entry (5-bit total per weight), which is a higher-area trade.
    """
    return [(int(v) & 1) ^ ((int(v) >> 1) & 1) for v in rom_init]


def _is_ternary(t: torch.Tensor, atol: float = 1e-6) -> bool:
    if t.dtype.is_floating_point:
        tf = t.to(torch.float64)
        if not bool(torch.isclose(tf, tf.round(), atol=atol).all().item()):
            return False
        rounded = tf.round()
    else:
        rounded = t.to(torch.float64)
    return bool((rounded.abs() <= 1.0).all().item())


def _to_int_list_2d(t: torch.Tensor) -> list[list[int]]:
    return [
        [int(round(float(v))) for v in row.flatten().tolist()]
        for row in t
    ]


def _to_int_list_1d(t: torch.Tensor) -> list[int]:
    return [int(round(float(v))) for v in t.flatten().tolist()]


def _parse_clamp_arg(arg: str | None) -> tuple[int, int] | None:
    if arg is None or arg == "":
        return None
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"--output-clamp expects 'LO,HI'; got {arg!r}"
        )
    try:
        lo = int(parts[0])
        hi = int(parts[1])
    except ValueError as e:
        raise ValueError(f"--output-clamp values must be integers: {e}") from e
    if lo > hi:
        raise ValueError(f"--output-clamp lo={lo} > hi={hi}")
    return (lo, hi)


def _build_sequential(
    layers_data: list[tuple[list[list[int]], list[int] | None]],
    activation_bits: int,
    clamp_range: tuple[int, int] | None,
    top: str,
    *,
    parallelism: int = 0,
    streaming_input: bool = False,
    handshake: bool = False,
    weight_bram: bool = False,
    mac_sharing: bool = False,
) -> GateGraph:
    """Build a streaming-MAC architecture for chained ternary linear layers.

    Module interface:
      clk, rst              auto-added by the backend
      start                 input pulse to begin a new inference
      x[0..N0-1]            signed activation_bits inputs (held stable
                            from start until done)
      done                  output pulse asserted for one cycle when
                            the final layer's last MAC retires
      y[0..M-1]             signed mac_width outputs valid while done is
                            high; settle to that value after done.

    Architecture: a 3-state FSM (IDLE / COMPUTE / DONE), a per-layer
    counter that walks 0..in_size[L]-1, an out_size[L]-wide bank of
    ternary-weight ROMs and accumulators per layer. On `start`, all
    accumulators preload to their layer's bias value; the FSM enters
    COMPUTE for layer 0; each cycle reads one weight per output neuron
    and adds (or subtracts) the selected input. When the last input of
    a layer completes, the FSM advances to the next layer; layer L's
    accumulators feed layer L+1's input mux. After the final layer's
    last MAC, the FSM enters DONE for one cycle, then returns to IDLE.

    Reset semantics: an asynchronous `rst` clears all accumulators to 0
    (the register kind's default init). This is distinct from the
    `start` pulse, which preloads each accumulator to its bias before
    the first MAC of a new inference. Reading outputs while in IDLE
    (without ever asserting `start`) yields 0; outputs are only valid
    when `done` is asserted.

    Notes on the IR shape:
      - Ternary weights are stored as 2-bit signed values (-1, 0, +1).
      - Per-output ROMs are addressed by the shared counter so all
        out_size[L] MACs run in parallel during a layer's compute phase.
      - Accumulator widths grow per layer to keep the worst-case sum
        lossless.
      - Inter-layer dataflow is via the previous layer's accumulators
        (no separate buffer needed; once a layer's compute phase ends,
        its accumulators retain their values until the next `start`).
    """
    K = len(layers_data)
    in_sizes = [len(rows[0]) for rows, _b in layers_data]
    out_sizes = [len(rows) for rows, _b in layers_data]

    # Per-layer accumulator widths: lossless growth from one layer to the next.
    mac_widths: list[int] = []
    cur_w = activation_bits
    for L in range(K):
        grow = max(1, in_sizes[L].bit_length()) + 1
        cur_w = cur_w + grow
        mac_widths.append(cur_w)

    # state_w = 3 when streaming_input adds the FILLING state, else 2.
    state_w = 3 if streaming_input else 2
    layer_w = max(1, K.bit_length())
    counter_w = max(1, max(in_sizes).bit_length())
    # When streaming_input is on, fill_counter walks 0..in_sizes[0]-1.
    fill_counter_w = max(1, in_sizes[0].bit_length()) if streaming_input else 1

    gates: list[Gate] = []

    def add(name, kind, *, inputs=None, attrs=None,
            output_width=1, output_signed=False):
        gates.append(Gate(
            name=name, kind=kind,
            inputs=list(inputs) if inputs else [],
            attrs=dict(attrs) if attrs else {},
            output_width=output_width,
            output_signed=output_signed,
        ))

    # ---- Constants ----
    # State encoding:
    #   IDLE=0, COMPUTE=1, DONE=2 always present
    #   FILL=3 only when streaming_input is on (state_w=3 in that case)
    add("const.state.idle",    "constant", attrs={"value": 0}, output_width=state_w)
    add("const.state.compute", "constant", attrs={"value": 1}, output_width=state_w)
    add("const.state.done",    "constant", attrs={"value": 2}, output_width=state_w)
    if streaming_input:
        add("const.state.fill", "constant", attrs={"value": 3}, output_width=state_w)
        add("const.fill.zero",  "constant", attrs={"value": 0},
            output_width=fill_counter_w)
        add("const.fill.one",   "constant", attrs={"value": 1},
            output_width=fill_counter_w)
        add("const.fill.max",   "constant",
            attrs={"value": in_sizes[0] - 1},
            output_width=fill_counter_w)
    add("const.counter.zero",  "constant", attrs={"value": 0}, output_width=counter_w)
    add("const.counter.one",   "constant", attrs={"value": 1}, output_width=counter_w)
    add("const.layer.zero",    "constant", attrs={"value": 0}, output_width=layer_w)
    add("const.layer.one",     "constant", attrs={"value": 1}, output_width=layer_w)
    add("const.layer.last",    "constant", attrs={"value": K - 1}, output_width=layer_w)
    for L in range(K):
        add(f"const.layer.{L}",
            "constant", attrs={"value": L}, output_width=layer_w)
        add(f"const.in_max.{L}",
            "constant", attrs={"value": in_sizes[L] - 1},
            output_width=counter_w)

    # ---- FSM signals ----
    add("is_idle",    "eq", inputs=["state.curr", "const.state.idle"])
    add("is_compute", "eq", inputs=["state.curr", "const.state.compute"])
    add("is_done",    "eq", inputs=["state.curr", "const.state.done"])
    if streaming_input:
        add("is_fill", "eq", inputs=["state.curr", "const.state.fill"])
        # fill_at_max: fill_counter == in_sizes[0]-1 and valid_in is high.
        add("fill_at_max",
            "eq", inputs=["fill_counter.curr", "const.fill.max"])
        add("fill_at_max_and_valid", "and",
            inputs=["fill_at_max", "valid_in"])
        add("fill_progress", "and", inputs=["is_fill", "valid_in"])
        # idle_and_start now means "begin filling"
        add("filling_done", "and",
            inputs=["is_fill", "fill_at_max_and_valid"])

    # is_layer.<L>: layer_idx == L
    for L in range(K):
        add(f"is_layer.{L}",
            "eq", inputs=["layer_idx.curr", f"const.layer.{L}"])

    # counter_at_max.<L>: counter == in_sizes[L] - 1
    for L in range(K):
        add(f"counter_at_max.{L}",
            "eq", inputs=["counter.curr", f"const.in_max.{L}"])

    # is_last_input: counter_at_max for the active layer.
    if K == 1:
        # Alias counter_at_max.0; an `or` with a 1-bit input acts as a buffer.
        add("is_last_input", "or", inputs=["counter_at_max.0"])
    else:
        # mux over layer_idx, picking counter_at_max.<L>.
        mux_inputs = ["layer_idx.curr"] + [
            f"counter_at_max.{L}" for L in range(K)
        ]
        add("is_last_input", "mux", inputs=mux_inputs, output_width=1)

    add("is_last_layer", "eq", inputs=["layer_idx.curr", "const.layer.last"])
    # transition_layer = is_compute && is_last_input && !is_last_layer
    # finishing       = is_compute && is_last_input && is_last_layer
    # With parallelism, both also require output_group_at_max so that all
    # groups in the current layer have completed.
    add("not_last_layer", "not", inputs=["is_last_layer"])
    add("compute_and_last_input", "and",
        inputs=["is_compute", "is_last_input"])
    add("transition_layer", "and",
        inputs=["compute_and_last_input", "not_last_layer"])
    add("finishing", "and",
        inputs=["compute_and_last_input", "is_last_layer"])
    # idle_and_start = is_idle && start
    add("idle_and_start", "and", inputs=["is_idle", "start"])
    # not_idle_and_start = !idle_and_start
    add("not_idle_and_start", "not", inputs=["idle_and_start"])

    # ---- parallelism: precompute output_group_at_max and effective FSM
    # gating so the state.next / layer_idx.next stages below can reference
    # the gated signals.
    use_parallelism = parallelism > 0 and parallelism < max(out_sizes)
    if use_parallelism:
        max_groups = max((sz + parallelism - 1) // parallelism
                         for sz in out_sizes)
        group_w = max(1, (max_groups - 1).bit_length())
        add("const.group.zero", "constant",
            attrs={"value": 0}, output_width=group_w)
        add("const.group.one", "constant",
            attrs={"value": 1}, output_width=group_w)
        for L in range(K):
            ngroups = (out_sizes[L] + parallelism - 1) // parallelism
            add(f"const.group_max.{L}", "constant",
                attrs={"value": ngroups - 1}, output_width=group_w)
            add(f"output_group_at_max.{L}", "eq",
                inputs=["output_group.curr", f"const.group_max.{L}"])
        if K == 1:
            add("output_group_at_max", "or",
                inputs=["output_group_at_max.0"])
        else:
            add("output_group_at_max", "mux",
                inputs=["layer_idx.curr"]
                       + [f"output_group_at_max.{L}" for L in range(K)],
                output_width=1)
        # Effective FSM signals: layer-transition only happens when both
        # counter and output_group complete; finishing only when last
        # layer's last group's last input completes.
        add("eff_transition_layer", "and",
            inputs=["transition_layer", "output_group_at_max"])
        add("eff_finishing", "and",
            inputs=["finishing", "output_group_at_max"])
    else:
        # Aliases so downstream code can always reference eff_* names.
        add("eff_transition_layer", "or", inputs=["transition_layer"])
        add("eff_finishing", "or", inputs=["finishing"])

    # ---- next_state -----------------------------------------------------
    # Base FSM (no variants):
    #   IDLE -> on start -> COMPUTE
    #   COMPUTE -> on finishing -> DONE; else stay
    #   DONE -> IDLE
    #
    # With streaming_input: IDLE -> on start -> FILL; FILL -> on filling_done
    #   -> COMPUTE.
    # With handshake: DONE -> IDLE only when ready_in; else hold in DONE.
    if streaming_input:
        # IDLE: on start -> FILL; else stay
        add("from_idle_target", "mux",
            inputs=["idle_and_start",
                    "const.state.idle", "const.state.fill"],
            output_width=state_w)
        # FILL: on filling_done -> COMPUTE; else stay
        add("from_fill_target", "mux",
            inputs=["filling_done",
                    "const.state.fill", "const.state.compute"],
            output_width=state_w)
        # COMPUTE: on eff_finishing -> DONE; else stay
        add("from_compute_target", "mux",
            inputs=["eff_finishing", "const.state.compute", "const.state.done"],
            output_width=state_w)
        # priority: is_done > is_compute > is_fill > else (idle)
        add("state_next_no_done", "mux",
            inputs=["state.curr", "from_idle_target",
                    "from_compute_target", "from_fill_target",
                    "from_idle_target"],
            output_width=state_w)
    else:
        add("from_compute_target", "mux",
            inputs=["eff_finishing", "const.state.compute", "const.state.done"],
            output_width=state_w)
        add("from_idle_target", "mux",
            inputs=["idle_and_start",
                    "const.state.idle", "const.state.compute"],
            output_width=state_w)
        add("state_next_no_done", "mux",
            inputs=["is_compute", "from_idle_target", "from_compute_target"],
            output_width=state_w)

    # Handshake gating on the DONE -> IDLE transition.
    if handshake:
        # done_release = is_done && ready_in
        add("done_release", "and", inputs=["is_done", "ready_in"])
        # If is_done && !ready_in, hold in DONE; if is_done && ready_in,
        # transition to IDLE.
        add("from_done_target", "mux",
            inputs=["done_release",
                    "const.state.done", "const.state.idle"],
            output_width=state_w)
        add("state.next", "mux",
            inputs=["is_done", "state_next_no_done", "from_done_target"],
            output_width=state_w)
    else:
        # Default: DONE -> IDLE unconditionally.
        add("state.next", "mux",
            inputs=["is_done", "state_next_no_done", "const.state.idle"],
            output_width=state_w)
    add("state.curr", "register",
        inputs=["state.next"],
        attrs={"clk": "clk", "rst": "rst", "init": 0},
        output_width=state_w)

    # ---- next_counter ---------------------------------------------------
    # Counter increments by 1 during COMPUTE unless this is the last input
    # of the active layer (then resets to 0 for the next layer or for IDLE).
    # On idle_and_start, also reset to 0.
    add("counter_plus_one", "add",
        inputs=["counter.curr", "const.counter.one"],
        output_width=counter_w)
    # During COMPUTE: counter_inc_or_reset = is_last_input ? 0 : counter+1
    add("counter_inc_or_reset", "mux",
        inputs=["is_last_input", "counter_plus_one", "const.counter.zero"],
        output_width=counter_w)
    # If is_compute, take counter_inc_or_reset; else take 0.
    add("counter.next", "mux",
        inputs=["is_compute", "const.counter.zero", "counter_inc_or_reset"],
        output_width=counter_w)
    add("counter.curr", "register",
        inputs=["counter.next"],
        attrs={"clk": "clk", "rst": "rst", "init": 0},
        output_width=counter_w)

    # ---- next_layer_idx -------------------------------------------------
    # On idle_and_start: layer_idx = 0
    # On transition_layer: layer_idx += 1
    # Else: hold
    add("layer_idx_plus_one", "add",
        inputs=["layer_idx.curr", "const.layer.one"],
        output_width=layer_w)
    add("layer_idx_after_transition", "mux",
        inputs=["eff_transition_layer", "layer_idx.curr", "layer_idx_plus_one"],
        output_width=layer_w)
    add("layer_idx.next", "mux",
        inputs=["idle_and_start", "layer_idx_after_transition", "const.layer.zero"],
        output_width=layer_w)
    add("layer_idx.curr", "register",
        inputs=["layer_idx.next"],
        attrs={"clk": "clk", "rst": "rst", "init": 0},
        output_width=layer_w)

    # ---- parallelism: output_group register + per-MAC group gating -----
    # When parallelism > 0 and < max(out_sizes), the FSM walks through
    # ceil(out_size / parallelism) output groups within each layer; only
    # accumulators whose j // parallelism == output_group update each
    # cycle. The IR still emits every accumulator (no physical MAC
    # sharing in this lowering); the synth tool may infer sharing under
    # resource-sharing passes.
    if use_parallelism:
        # output_group walks 0..ngroups[L]-1 within each layer; resets on
        # idle_and_start and on layer transitions.
        add("output_group_plus_one", "add",
            inputs=["output_group.curr", "const.group.one"],
            output_width=group_w)
        add("counter_wrapping", "and",
            inputs=["is_compute", "is_last_input"])
        add("group_step", "and",
            inputs=["counter_wrapping", "is_compute"])
        add("output_group_after_step", "mux",
            inputs=["group_step",
                    "output_group.curr", "output_group_plus_one"],
            output_width=group_w)
        add("output_group_wrap", "and",
            inputs=["counter_wrapping", "output_group_at_max"])
        add("output_group_advance", "mux",
            inputs=["output_group_wrap",
                    "output_group_after_step", "const.group.zero"],
            output_width=group_w)
        add("output_group.next", "mux",
            inputs=["idle_and_start",
                    "output_group_advance", "const.group.zero"],
            output_width=group_w)
        add("output_group.curr", "register",
            inputs=["output_group.next"],
            attrs={"clk": "clk", "rst": "rst", "init": 0},
            output_width=group_w)
        # Per-(L, j) group-match constants: 1 iff j // parallelism ==
        # output_group.curr (for the current layer).
        for L in range(K):
            for j in range(out_sizes[L]):
                grp = j // parallelism
                add(f"const.group.{L}.{j}", "constant",
                    attrs={"value": grp}, output_width=group_w)
                add(f"group_match_L{L}_j{j}", "eq",
                    inputs=["output_group.curr", f"const.group.{L}.{j}"])

    # ---- Streaming-input: fill counter + in_buf register file -----------
    if streaming_input:
        # fill_counter increments while in_fill && valid_in.
        add("fill_counter_plus_one", "add",
            inputs=["fill_counter.curr", "const.fill.one"],
            output_width=fill_counter_w)
        add("fill_counter_after_step", "mux",
            inputs=["fill_progress",
                    "fill_counter.curr", "fill_counter_plus_one"],
            output_width=fill_counter_w)
        # On idle_and_start, reset fill_counter to 0.
        add("fill_counter.next", "mux",
            inputs=["idle_and_start",
                    "fill_counter_after_step", "const.fill.zero"],
            output_width=fill_counter_w)
        add("fill_counter.curr", "register",
            inputs=["fill_counter.next"],
            attrs={"clk": "clk", "rst": "rst", "init": 0},
            output_width=fill_counter_w)
        # in_buf[i]: write-enable when fill_progress && fill_counter.curr == i
        for i in range(in_sizes[0]):
            add(f"in_buf.idx_{i}", "constant",
                attrs={"value": i}, output_width=fill_counter_w)
            add(f"in_buf.match_{i}", "eq",
                inputs=["fill_counter.curr", f"in_buf.idx_{i}"])
            add(f"in_buf.we_{i}", "and",
                inputs=["fill_progress", f"in_buf.match_{i}"])
            # Register's enable input: writes x when we_{i} is high.
            add(f"in_buf.{i}.curr", "register",
                inputs=["x", f"in_buf.we_{i}"],
                attrs={"clk": "clk", "rst": "rst", "init": 0,
                       "enable": f"in_buf.we_{i}"},
                output_width=activation_bits, output_signed=True)
        # ready_out: high while in FILL state. Caller asserts valid_in to
        # clock data into in_buf at fill_counter; one element per cycle.
        add("ready_out", "or", inputs=["is_fill"])

    # ---- Per-layer input mux: input_for_layer[L][counter] ---------------
    # Layer 0 reads from external x[0..N0-1] (or in_buf when streaming_input).
    # Layer L>0 reads from previous layer's accumulators.
    # Each layer's input mux picks 1 input based on counter.
    # When mac_sharing is on, layer L+1 reads from L's storage register
    # file instead of the per-output accumulators. The acc registers
    # still exist and update each cycle; storage captures at end-of-group
    # so downstream layers see stable values.
    acc_signal_template = "L{L}.store{j}.curr" if mac_sharing else "L{L}.acc{j}.curr"

    for L in range(K):
        if L == 0:
            if streaming_input:
                sources = [f"in_buf.{i}.curr" for i in range(in_sizes[0])]
            else:
                sources = [f"x{i}" for i in range(in_sizes[0])]
        else:
            sources = [
                acc_signal_template.format(L=L - 1, j=j)
                for j in range(out_sizes[L - 1])
            ]
        # Pad sources to power-of-2 size for clean mux indexing? Not needed:
        # the mux walks counter; values beyond in_sizes[L]-1 won't be read
        # because counter never reaches them.
        sw = activation_bits if L == 0 else mac_widths[L - 1]
        if len(sources) == 1:
            # Single-input layer: rename the lone source to the canonical
            # input-mux name via an `add` with a zero constant.
            zero_name = f"L{L}.input_zero"
            add(zero_name, "constant",
                attrs={"value": 0}, output_width=sw, output_signed=True)
            add(f"L{L}.input_mux", "add",
                inputs=[sources[0], zero_name],
                output_width=sw, output_signed=True)
        else:
            add(f"L{L}.input_mux", "mux",
                inputs=["counter.curr"] + sources,
                output_width=sw, output_signed=True)

    # ---- weight_bram: per-ROM write-enable routing ----------------------
    if weight_bram:
        # weight_addr_layer selects which layer's ROMs accept writes;
        # weight_addr_output selects which output (j) within that layer;
        # weight_addr_position is the in-row position the ROM sees as its
        # write_addr (also driven by counter.curr for reads, but the RAM
        # primitive distinguishes read_addr from write_addr).
        for L in range(K):
            add(f"const.weight_layer.{L}", "constant",
                attrs={"value": L}, output_width=layer_w)
            add(f"weight_layer_match.{L}", "eq",
                inputs=["weight_addr_layer", f"const.weight_layer.{L}"])
            for j in range(out_sizes[L]):
                # output_idx_w widens to fit the largest layer; padding for
                # narrower layers is harmless because the equality check
                # against the layer-specific output index does the right
                # thing on zero-padded values.
                add(f"const.weight_output.{L}.{j}", "constant",
                    attrs={"value": j},
                    output_width=max(1, max(out_sizes).bit_length()))
                add(f"weight_output_match_L{L}_j{j}", "eq",
                    inputs=["weight_addr_output",
                            f"const.weight_output.{L}.{j}"])
                add(f"weight_lj_match_L{L}_j{j}", "and",
                    inputs=[f"weight_layer_match.{L}",
                            f"weight_output_match_L{L}_j{j}"])
                add(f"weight_we_L{L}_j{j}", "and",
                    inputs=["weight_we", f"weight_lj_match_L{L}_j{j}"])

    # ---- Per-(L, j): weight ROM, product, accumulator -------------------
    for L in range(K):
        rows, biases = layers_data[L]
        layer_input_width = activation_bits if L == 0 else mac_widths[L - 1]
        product_width = layer_input_width + 2  # 2-bit signed weight
        for j in range(out_sizes[L]):
            row = rows[j]
            bias_val = biases[j] if biases else 0

            # ROM: 2-bit signed weights. depth = in_sizes[L].
            # We need values that two's-complement-decode to {-1, 0, +1}.
            # The backend's `rom` lowering masks values into the declared
            # width unsigned, so we encode -1 as the 2-bit pattern 0b11
            # which `$signed(...)` reads as -1.
            rom_init = []
            for w in row:
                if w == 0:
                    rom_init.append(0)
                elif w == 1:
                    rom_init.append(1)
                elif w == -1:
                    rom_init.append(3)  # 0b11 == -1 in 2-bit signed
                else:
                    raise ValueError(
                        f"ternary check failed: weight {w} for L{L}.j{j}"
                    )
            if weight_bram:
                # Writable RAM: caller drives weight_addr_layer /
                # weight_addr_output / weight_addr_position / weight_data /
                # weight_we. The per-(L,j) gates above mask weight_we so
                # each ROM only accepts the writes intended for it.
                add(f"L{L}.rom{j}", "ram_writable",
                    inputs=["counter.curr",          # read_addr
                            "weight_addr_position",  # write_addr
                            "weight_data",           # write_data
                            f"weight_we_L{L}_j{j}"], # write_en (per-ROM)
                    attrs={"init": rom_init, "width": 2,
                           "depth": in_sizes[L], "clk": "clk"},
                    output_width=2, output_signed=True)
            else:
                add(f"L{L}.rom{j}", "rom",
                    inputs=["counter.curr"],
                    attrs={"init": rom_init, "width": 2,
                           "depth": in_sizes[L]},
                    output_width=2, output_signed=True)

            # product = weight * input  (signed mul; width = sum of operand widths)
            add(f"L{L}.prod{j}", "mul",
                inputs=[f"L{L}.rom{j}", f"L{L}.input_mux"],
                output_width=product_width, output_signed=True)

            # acc.next: complex update logic.
            # is_active_for_L = is_compute && is_layer.<L>
            # add_value = acc.curr + product
            # update_or_hold = is_active_for_L ? add_value : acc.curr
            # acc.next = idle_and_start ? 0 : update_or_hold
            if use_parallelism:
                # is_active for (L, j) requires the group match too.
                add(f"L{L}.is_active_pre{j}", "and",
                    inputs=["is_compute", f"is_layer.{L}"])
                add(f"L{L}.is_active{j}", "and",
                    inputs=[f"L{L}.is_active_pre{j}",
                            f"group_match_L{L}_j{j}"])
            else:
                add(f"L{L}.is_active{j}", "and",
                    inputs=["is_compute", f"is_layer.{L}"])
            add(f"L{L}.sum{j}", "add",
                inputs=[f"L{L}.acc{j}.curr", f"L{L}.prod{j}"],
                output_width=mac_widths[L], output_signed=True)
            add(f"L{L}.update_or_hold{j}", "mux",
                inputs=[
                    f"L{L}.is_active{j}",
                    f"L{L}.acc{j}.curr",
                    f"L{L}.sum{j}",
                ],
                output_width=mac_widths[L], output_signed=True)
            add(f"L{L}.acc{j}.zero", "constant",
                attrs={"value": 0},
                output_width=mac_widths[L], output_signed=True)
            # If we have a non-zero bias, the reset value should be the
            # bias itself, not zero. That way the accumulator starts pre-
            # loaded with the bias and ends with bias + Σ w*x.
            if bias_val != 0:
                add(f"L{L}.acc{j}.bias_const", "constant",
                    attrs={"value": bias_val},
                    output_width=mac_widths[L], output_signed=True)
                reset_value = f"L{L}.acc{j}.bias_const"
            else:
                reset_value = f"L{L}.acc{j}.zero"
            add(f"L{L}.acc{j}.next", "mux",
                inputs=[
                    "idle_and_start",
                    f"L{L}.update_or_hold{j}",
                    reset_value,
                ],
                output_width=mac_widths[L], output_signed=True)
            add(f"L{L}.acc{j}.curr", "register",
                inputs=[f"L{L}.acc{j}.next"],
                attrs={"clk": "clk", "rst": "rst", "init": 0},
                output_width=mac_widths[L], output_signed=True)

            # ---- mac_sharing storage register --------------------------
            # Captures the accumulator value at end-of-group for output j,
            # signaling to the synth tool's resource-sharing pass that
            # the accumulator is consumed only at the group boundary.
            # Combined with --parallelism's group gating, this gives the
            # architectural pattern where N MAC units feed an out_size
            # storage file. The accumulator values can then be reused
            # across groups in physical hardware.
            if mac_sharing:
                if use_parallelism:
                    # Capture only when this output's group is current.
                    add(f"L{L}.store_capture{j}", "and",
                        inputs=[f"L{L}.is_active{j}", "is_last_input"])
                else:
                    # Without parallelism, capture at every is_last_input
                    # within this layer.
                    add(f"L{L}.store_capture{j}", "and",
                        inputs=["compute_and_last_input", f"is_layer.{L}"])
                add(f"L{L}.store{j}.next", "mux",
                    inputs=[
                        f"L{L}.store_capture{j}",
                        f"L{L}.store{j}.curr",
                        f"L{L}.acc{j}.next",
                    ],
                    output_width=mac_widths[L], output_signed=True)
                add(f"L{L}.store{j}.curr", "register",
                    inputs=[f"L{L}.store{j}.next"],
                    attrs={"clk": "clk", "rst": "rst", "init": 0},
                    output_width=mac_widths[L], output_signed=True)

    # ---- Optional output clamp -----------------------------------------
    final_L = K - 1
    final_mac_width = mac_widths[final_L]
    final_outputs: list[str] = []
    final_acc_template = (
        "L{L}.store{j}.curr" if mac_sharing else "L{L}.acc{j}.curr"
    )
    if clamp_range is not None:
        lo, hi = clamp_range
        for j in range(out_sizes[final_L]):
            add(f"y{j}", "clamp",
                inputs=[final_acc_template.format(L=final_L, j=j)],
                attrs={"lo": lo, "hi": hi},
                output_width=final_mac_width, output_signed=True)
            final_outputs.append(f"y{j}")
    else:
        for j in range(out_sizes[final_L]):
            add(f"y{j}.zero", "constant",
                attrs={"value": 0},
                output_width=final_mac_width, output_signed=True)
            add(f"y{j}", "add",
                inputs=[final_acc_template.format(L=final_L, j=j),
                        f"y{j}.zero"],
                output_width=final_mac_width, output_signed=True)
            final_outputs.append(f"y{j}")

    # ---- 'done' signal --------------------------------------------------
    # done = is_done; alias via an `or` so it has a public name.
    add("done", "or", inputs=["is_done"])

    # ---- handshake: valid_out = is_done; ready_in is a module input ----
    if handshake:
        add("valid_out", "or", inputs=["is_done"])

    # ---- External signals ----------------------------------------------
    inputs = [Signal("start", width=1), Signal("rst", width=1)]
    if streaming_input:
        inputs.append(Signal("x", width=activation_bits, signed=True))
        inputs.append(Signal("valid_in", width=1))
    else:
        for i in range(in_sizes[0]):
            inputs.append(
                Signal(f"x{i}", width=activation_bits, signed=True)
            )
    if handshake:
        inputs.append(Signal("ready_in", width=1))
    if weight_bram:
        # Address ports widths: layer needs ceil(log2(K)), output needs
        # ceil(log2(max(out_sizes))), position needs ceil(log2(max(in_sizes))).
        out_idx_w = max(1, max(out_sizes).bit_length())
        pos_w = counter_w
        inputs.append(Signal("weight_addr_layer", width=layer_w))
        inputs.append(Signal("weight_addr_output", width=out_idx_w))
        inputs.append(Signal("weight_addr_position", width=pos_w))
        inputs.append(Signal("weight_data", width=2, signed=True))
        inputs.append(Signal("weight_we", width=1))

    outputs = [Signal("done", width=1)]
    if streaming_input:
        outputs.append(Signal("ready_out", width=1))
    if handshake:
        outputs.append(Signal("valid_out", width=1))
    for j in range(out_sizes[final_L]):
        outputs.append(Signal(f"y{j}", width=final_mac_width, signed=True))

    return GateGraph(inputs=inputs, outputs=outputs, gates=gates, top=top)


@registry.register(
    "bitnet_linear",
    description="BitNet b1.58-style ternary linear layers with multibit activations.",
    metadata_namespace="bitnet_linear",
)
class BitNetLinearFrontend(Frontend):

    @classmethod
    def options(cls) -> list[FrontendOption]:
        return [
            FrontendOption(
                name="activation-bits",
                type=int,
                default=8,
                help="bit width of input activations (signed two's complement).",
            ),
            FrontendOption(
                name="layer-prefix",
                type=str,
                default="layers",
                help=(
                    "tensor name prefix that identifies layer weights "
                    "(matches state_dict from torch.nn.Sequential)."
                ),
            ),
            FrontendOption(
                name="output-clamp",
                type=str,
                default=None,
                help=(
                    "clamp each layer output to LO,HI integers (e.g. -128,127 "
                    "for int8 saturation). Wraps each layer's MAC in a 'clamp' "
                    "gate. Width is inherited from the activation-bits choice."
                ),
            ),
            FrontendOption(
                name="pipeline",
                type=bool,
                default=False,
                help=(
                    "register each layer's output (one cycle of latency per "
                    "layer). Combinational mode only; mutually exclusive "
                    "with --sequential."
                ),
            ),
            FrontendOption(
                name="parallelism",
                type=int,
                default=0,
                help=(
                    "sequential-mode --parallelism N: time-multiplex "
                    "outputs to trade latency for area. With N < "
                    "out_size[L], the layer reuses N MAC units across "
                    "ceil(out_size[L]/N) output groups. 0 means full "
                    "parallelism (one MAC per output, current default)."
                ),
            ),
            FrontendOption(
                name="streaming-input",
                type=bool,
                default=False,
                help=(
                    "sequential-mode --streaming-input: replace the "
                    "per-input port bank with a single x port plus "
                    "valid_in / ready_out handshake, plus an internal "
                    "in_buf register file that fills before COMPUTE."
                ),
            ),
            FrontendOption(
                name="handshake",
                type=bool,
                default=False,
                help=(
                    "sequential-mode --handshake: full valid_out + "
                    "ready_in protocol on the output side. DONE state "
                    "holds outputs valid until ready_in fires."
                ),
            ),
            FrontendOption(
                name="weight-bram",
                type=bool,
                default=False,
                help=(
                    "sequential-mode --weight-bram: replace the per-"
                    "output ROMs with writable BRAMs and expose "
                    "weight_addr / weight_data / weight_we ports for "
                    "runtime weight reload."
                ),
            ),
            FrontendOption(
                name="sequential",
                type=bool,
                default=False,
                help=(
                    "emit a streaming architecture: one MAC per output neuron, "
                    "ROM-stored weights addressed by a shared counter, an FSM "
                    "stepping through layers. Ports gain start/done; latency "
                    "is sum(in_sizes) cycles per inference. Mutually exclusive "
                    "with --pipeline. Output buses are sized to the final "
                    "layer's accumulator width."
                ),
            ),
            FrontendOption(
                name="mac-sharing",
                type=bool,
                default=False,
                help=(
                    "sequential-mode --mac-sharing: emit per-output "
                    "storage registers that capture the accumulator "
                    "value at end-of-group, with inter-layer reads + "
                    "final outputs going through storage. Combined with "
                    "--parallelism N this gives the synth tool the "
                    "structural cues (active accumulators feeding storage "
                    "captured at group boundaries) to share MAC hardware "
                    "across output groups in the placed design."
                ),
            ),
        ]

    def parse(
        self,
        path: Path,
        top: str = "top",
        activation_bits: int = 8,
        layer_prefix: str = "layers",
        output_clamp: str | None = None,
        pipeline: bool = False,
        sequential: bool = False,
        parallelism: int = 0,
        streaming_input: bool = False,
        handshake: bool = False,
        weight_bram: bool = False,
        mac_sharing: bool = False,
        **options,
    ) -> GateGraph:
        if pipeline and sequential:
            raise ValueError(
                "--pipeline and --sequential are mutually exclusive; "
                "sequential mode is already inherently pipelined "
                "(one input per cycle into shared MAC hardware)."
            )

        # Sequential-bitnet variants are now wired through.
        if parallelism < 0:
            raise ValueError(
                f"--parallelism must be non-negative, got {parallelism}"
            )
        if (parallelism or streaming_input or handshake or weight_bram or mac_sharing) \
                and not sequential:
            raise ValueError(
                "--parallelism / --streaming-input / --handshake / "
                "--weight-bram / --mac-sharing only apply with --sequential."
            )

        clamp_range = _parse_clamp_arg(output_clamp)

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name).clone()

        prefix_dot = f"{layer_prefix}."
        layer_indices: list[int] = []
        for name in tensors:
            if name.startswith(prefix_dot) and name.endswith(".weight"):
                idx_part = name[len(prefix_dot):].split(".")[0]
                if idx_part.isdigit():
                    layer_indices.append(int(idx_part))
        layer_indices = sorted(set(layer_indices))
        if not layer_indices:
            raise ValueError(
                f"no layers found with prefix '{prefix_dot}<n>.weight'. "
                f"Tensors present: {sorted(tensors)[:8]}..."
            )

        # Validate every layer up front (both modes need this)
        layers_data: list[tuple[list[list[int]], list[int] | None]] = []
        prev_in = None
        for layer_idx in layer_indices:
            wkey = f"{layer_prefix}.{layer_idx}.weight"
            bkey = f"{layer_prefix}.{layer_idx}.bias"
            w = tensors[wkey]
            if w.dim() != 2:
                raise ValueError(
                    f"layer {layer_idx} weight not 2-D: {tuple(w.shape)}"
                )
            if not _is_ternary(w):
                raise ValueError(
                    f"layer {layer_idx} weights are not ternary {{-1, 0, 1}}"
                )
            if prev_in is not None and w.shape[1] != prev_in:
                raise ValueError(
                    f"layer {layer_idx}: in_features={w.shape[1]} but "
                    f"previous stage produced {prev_in} signals"
                )
            prev_in = w.shape[0]
            biases = (
                _to_int_list_1d(tensors[bkey]) if bkey in tensors else None
            )
            layers_data.append((_to_int_list_2d(w), biases))

        if sequential:
            return _build_sequential(
                layers_data, activation_bits, clamp_range, top,
                parallelism=parallelism,
                streaming_input=streaming_input,
                handshake=handshake,
                weight_bram=weight_bram,
                mac_sharing=mac_sharing,
            )

        gates: list[Gate] = []

        first_w = tensors[f"{layer_prefix}.{layer_indices[0]}.weight"]
        in_size = first_w.shape[1]

        input_signals = [
            Signal(name=f"x{i}", width=activation_bits, signed=True)
            for i in range(in_size)
        ]
        prev_outputs: list[str] = [s.name for s in input_signals]

        accumulator_width = activation_bits

        for layer_idx, (wrows, layer_biases) in zip(layer_indices, layers_data):
            out_size = len(wrows)
            layer_in = len(wrows[0])

            grow = max(1, layer_in.bit_length()) + 1
            mac_width = accumulator_width + grow

            this_outputs: list[str] = []
            for j in range(out_size):
                row = wrows[j]
                bias_val = layer_biases[j] if layer_biases else 0
                final = f"L{layer_idx}.y{j}"

                # Optional post-MAC stages: clamp and/or pipeline register.
                # The 'linear' gate emits the raw Σ w*x + b. If we need a
                # clamp or register after it, we emit it under a private
                # name and have the post-stage carry the public layer-output
                # name. If there are no post-stages, the linear gate IS
                # the public output.
                need_clamp = clamp_range is not None
                need_register = pipeline
                if need_register:
                    mac_name = f"L{layer_idx}.y{j}.mac"
                    if need_clamp:
                        clamped_name = f"L{layer_idx}.y{j}.clamped"
                    else:
                        clamped_name = mac_name
                elif need_clamp:
                    mac_name = f"L{layer_idx}.y{j}.mac"
                    clamped_name = final
                else:
                    mac_name = final
                    clamped_name = final

                gates.append(Gate(
                    name=mac_name, kind="linear",
                    inputs=list(prev_outputs),
                    attrs={"weights": row, "bias": bias_val},
                    output_width=mac_width, output_signed=True,
                ))

                if clamp_range is not None:
                    lo, hi = clamp_range
                    gates.append(Gate(
                        name=clamped_name, kind="clamp",
                        inputs=[mac_name],
                        attrs={"lo": lo, "hi": hi},
                        output_width=mac_width, output_signed=True,
                    ))

                if need_register:
                    gates.append(Gate(
                        name=final, kind="register",
                        inputs=[clamped_name],
                        attrs={"clk": "clk"},
                        output_width=mac_width, output_signed=True,
                    ))

                this_outputs.append(final)

            prev_outputs = this_outputs
            accumulator_width = mac_width

        gate_widths = {g.name: g.output_width for g in gates}
        gate_signed = {g.name: g.output_signed for g in gates}
        output_signals = [
            Signal(name=n, width=gate_widths[n], signed=gate_signed[n])
            for n in prev_outputs
        ]

        return GateGraph(
            inputs=input_signals,
            outputs=output_signals,
            gates=gates,
            top=top,
        )
