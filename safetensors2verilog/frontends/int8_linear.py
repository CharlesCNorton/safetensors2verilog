"""Int8 linear-layer frontend.

Reads a safetensors file describing one or more chained quantized
linear layers with arbitrary signed integer weights. Each layer
becomes one ``linear`` gate per output neuron, with an optional
saturating clamp and optional pipeline register.

Tensor naming convention (matches PyTorch's ``state_dict`` for a
``nn.Sequential([nn.Linear, ...])``):

  <prefix>.<n>.weight   integer / float tensor with shape [out, in]
  <prefix>.<n>.bias     integer / float tensor with shape [out] (optional)

Default ``prefix`` is ``layers``. Override with ``--layer-prefix``.

Weights may be any integers (within ``--weight-bits``); ternary models
will work, but if all your weights are in {-1, 0, 1}, the more
constrained ``bitnet_linear`` frontend will reject non-ternary inputs
explicitly.

Activation width is set by ``--activation-bits`` (default 8). The
per-layer accumulator widens by ``weight-bits + ceil(log2(in_features))``
to keep the worst-case sum lossless.

Optional features:
  --output-clamp LO,HI   Wrap each output in a clamp gate.
  --pipeline             Register each layer's output (one cycle of
                         latency per layer).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from ..core import Frontend, FrontendOption, Gate, GateGraph, Signal, registry


def _is_integer_tensor(t: torch.Tensor, atol: float = 1e-6) -> bool:
    if not t.dtype.is_floating_point:
        return True
    tf = t.to(torch.float64)
    return bool(torch.isclose(tf, tf.round(), atol=atol).all().item())


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
        raise ValueError(f"--output-clamp expects 'LO,HI'; got {arg!r}")
    try:
        lo = int(parts[0])
        hi = int(parts[1])
    except ValueError as e:
        raise ValueError(f"--output-clamp values must be integers: {e}") from e
    if lo > hi:
        raise ValueError(f"--output-clamp lo={lo} > hi={hi}")
    return (lo, hi)


@registry.register(
    "int8_linear",
    description="Quantized linear layers with signed integer weights and multibit activations.",
    metadata_namespace="int8_linear",
)
class Int8LinearFrontend(Frontend):

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
                name="weight-bits",
                type=int,
                default=8,
                help=(
                    "expected bit width of weights (signed). Used to size each "
                    "layer's accumulator; weights outside [-2^(N-1), 2^(N-1)-1] "
                    "are rejected."
                ),
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
                    "gate."
                ),
            ),
            FrontendOption(
                name="pipeline",
                type=bool,
                default=False,
                help=(
                    "register each layer's output (one cycle of latency per "
                    "layer). Adds a 'clk' port to the emitted module."
                ),
            ),
        ]

    def parse(
        self,
        path: Path,
        top: str = "top",
        activation_bits: int = 8,
        weight_bits: int = 8,
        layer_prefix: str = "layers",
        output_clamp: str | None = None,
        pipeline: bool = False,
        **options,
    ) -> GateGraph:
        if weight_bits < 2:
            raise ValueError(f"weight-bits must be >= 2, got {weight_bits}")
        weight_lo = -(1 << (weight_bits - 1))
        weight_hi = (1 << (weight_bits - 1)) - 1
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

        gates: list[Gate] = []

        first_w = tensors[f"{layer_prefix}.{layer_indices[0]}.weight"]
        if first_w.dim() != 2:
            raise ValueError(
                f"layer {layer_indices[0]} weight must be 2-D [out, in]; "
                f"got {tuple(first_w.shape)}"
            )
        in_size = first_w.shape[1]

        input_signals = [
            Signal(name=f"x{i}", width=activation_bits, signed=True)
            for i in range(in_size)
        ]
        prev_outputs: list[str] = [s.name for s in input_signals]
        accumulator_width = activation_bits

        for layer_idx in layer_indices:
            wkey = f"{layer_prefix}.{layer_idx}.weight"
            bkey = f"{layer_prefix}.{layer_idx}.bias"
            w = tensors[wkey]
            if w.dim() != 2:
                raise ValueError(
                    f"layer {layer_idx} weight not 2-D: {tuple(w.shape)}"
                )
            if w.shape[1] != len(prev_outputs):
                raise ValueError(
                    f"layer {layer_idx}: in_features={w.shape[1]} but "
                    f"previous stage produced {len(prev_outputs)} signals"
                )
            if not _is_integer_tensor(w):
                raise ValueError(
                    f"layer {layer_idx} weights are not integer-valued"
                )

            wrows = _to_int_list_2d(w)
            for j, row in enumerate(wrows):
                for i, wij in enumerate(row):
                    if wij < weight_lo or wij > weight_hi:
                        raise ValueError(
                            f"layer {layer_idx} weight[{j}][{i}]={wij} is "
                            f"outside [{weight_lo}, {weight_hi}] for "
                            f"--weight-bits={weight_bits}"
                        )

            out_size, layer_in = w.shape
            biases: list[int] | None = None
            if bkey in tensors:
                if not _is_integer_tensor(tensors[bkey]):
                    raise ValueError(
                        f"layer {layer_idx} bias is not integer-valued"
                    )
                biases = _to_int_list_1d(tensors[bkey])

            grow = weight_bits + max(1, layer_in.bit_length())
            mac_width = accumulator_width + grow

            this_outputs: list[str] = []
            for j in range(out_size):
                row = wrows[j]
                bias_val = biases[j] if biases else 0
                final = f"L{layer_idx}.y{j}"

                need_clamp = clamp_range is not None
                need_register = pipeline
                if need_register:
                    mac_name = f"L{layer_idx}.y{j}.mac"
                    clamped_name = (
                        f"L{layer_idx}.y{j}.clamped" if need_clamp else mac_name
                    )
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
