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
    accumulators reset to 0; the FSM enters COMPUTE for layer 0; each
    cycle reads one weight per output neuron and adds (or subtracts) the
    selected input. When the last input of a layer completes, the FSM
    advances to the next layer; layer L's accumulators feed layer L+1's
    input mux. After the final layer's last MAC, the FSM enters DONE
    for one cycle, then returns to IDLE.

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

    state_w = 2
    layer_w = max(1, K.bit_length())
    counter_w = max(1, max(in_sizes).bit_length())

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
    add("const.state.idle",    "constant", attrs={"value": 0}, output_width=state_w)
    add("const.state.compute", "constant", attrs={"value": 1}, output_width=state_w)
    add("const.state.done",    "constant", attrs={"value": 2}, output_width=state_w)
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
    add("not_last_layer", "not", inputs=["is_last_layer"])
    add("compute_and_last_input", "and",
        inputs=["is_compute", "is_last_input"])
    add("transition_layer", "and",
        inputs=["compute_and_last_input", "not_last_layer"])
    # finishing = is_compute && is_last_input && is_last_layer
    add("finishing", "and",
        inputs=["compute_and_last_input", "is_last_layer"])
    # idle_and_start = is_idle && start
    add("idle_and_start", "and", inputs=["is_idle", "start"])
    # not_idle_and_start = !idle_and_start
    add("not_idle_and_start", "not", inputs=["idle_and_start"])

    # ---- next_state -----------------------------------------------------
    # IDLE  : on start -> COMPUTE; else stay IDLE
    # COMPUTE : if finishing -> DONE; else stay COMPUTE
    # DONE  : -> IDLE
    #
    # Encoded as: state_next = mux on the priority list:
    #   if is_done       -> IDLE
    #   elif finishing   -> DONE
    #   elif idle_and_start -> COMPUTE
    #   elif is_compute and not finishing -> COMPUTE (stay)
    #   elif is_idle and not start -> IDLE
    #   else (unreachable) -> IDLE
    # We build it via chained 2-input muxes.
    #
    # Stage A: from_compute_target = is_compute && finishing ? DONE : COMPUTE
    add("from_compute_target", "mux",
        inputs=["finishing", "const.state.compute", "const.state.done"],
        output_width=state_w)
    # Stage B: from_idle_target = is_idle && start ? COMPUTE : IDLE
    add("from_idle_target", "mux",
        inputs=["idle_and_start", "const.state.idle", "const.state.compute"],
        output_width=state_w)
    # Stage C: combine with is_compute selector
    add("state_next_intermediate", "mux",
        inputs=["is_compute", "from_idle_target", "from_compute_target"],
        output_width=state_w)
    # Stage D: if is_done, override to IDLE
    add("state.next", "mux",
        inputs=["is_done", "state_next_intermediate", "const.state.idle"],
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
        inputs=["transition_layer", "layer_idx.curr", "layer_idx_plus_one"],
        output_width=layer_w)
    add("layer_idx.next", "mux",
        inputs=["idle_and_start", "layer_idx_after_transition", "const.layer.zero"],
        output_width=layer_w)
    add("layer_idx.curr", "register",
        inputs=["layer_idx.next"],
        attrs={"clk": "clk", "rst": "rst", "init": 0},
        output_width=layer_w)

    # ---- Per-layer input mux: input_for_layer[L][counter] ---------------
    # Layer 0 reads from external x[0..N0-1].
    # Layer L>0 reads from previous layer's accumulators.
    # Each layer's input mux picks 1 input based on counter.
    for L in range(K):
        if L == 0:
            sources = [f"x{i}" for i in range(in_sizes[0])]
        else:
            sources = [f"L{L-1}.acc{j}.curr" for j in range(out_sizes[L-1])]
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
            add(f"L{L}.rom{j}", "rom",
                inputs=["counter.curr"],
                attrs={"init": rom_init, "width": 2, "depth": in_sizes[L]},
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

    # ---- Optional output clamp -----------------------------------------
    final_L = K - 1
    final_mac_width = mac_widths[final_L]
    final_outputs: list[str] = []
    if clamp_range is not None:
        lo, hi = clamp_range
        for j in range(out_sizes[final_L]):
            add(f"y{j}", "clamp",
                inputs=[f"L{final_L}.acc{j}.curr"],
                attrs={"lo": lo, "hi": hi},
                output_width=final_mac_width, output_signed=True)
            final_outputs.append(f"y{j}")
    else:
        # Buffer the accumulators to a public 'y<j>' name via a +0.
        for j in range(out_sizes[final_L]):
            add(f"y{j}.zero", "constant",
                attrs={"value": 0},
                output_width=final_mac_width, output_signed=True)
            add(f"y{j}", "add",
                inputs=[f"L{final_L}.acc{j}.curr", f"y{j}.zero"],
                output_width=final_mac_width, output_signed=True)
            final_outputs.append(f"y{j}")

    # ---- 'done' signal --------------------------------------------------
    # done = is_done; alias via an `or` so it has a public name.
    add("done", "or", inputs=["is_done"])

    # ---- External signals ----------------------------------------------
    inputs = [Signal("start", width=1), Signal("rst", width=1)]
    for i in range(in_sizes[0]):
        inputs.append(Signal(f"x{i}", width=activation_bits, signed=True))

    outputs = [Signal("done", width=1)]
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
        **options,
    ) -> GateGraph:
        if pipeline and sequential:
            raise ValueError(
                "--pipeline and --sequential are mutually exclusive; "
                "sequential mode is already inherently pipelined "
                "(one input per cycle into shared MAC hardware)."
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
                layers_data, activation_bits, clamp_range, top
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
