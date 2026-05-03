"""BitNet b1.58 ternary linear frontend.

BitNet b1.58 represents `nn.Linear` weights as ternary {-1, 0, 1};
activations are multi-bit (typically int8). This frontend reads a
safetensors file describing one or more chained ternary linear
layers and emits a combinational dataflow network that computes
``y = (W_n · ... · W_1 · x) + biases``.

Tensor naming convention (matches PyTorch's `state_dict` for a
``nn.Sequential([nn.Linear, ...])``):

  <prefix>.<n>.weight    int / float tensor with shape [out, in], values in {-1, 0, 1}
  <prefix>.<n>.bias      int / float tensor with shape [out] (optional)

Default ``prefix`` is ``layers``. Override with ``--layer-prefix``.

The emitted module has one signed input port per element of the first
layer's input vector and one signed output port per element of the
last layer's output vector. Activation width is set by
``--activation-bits`` (default 8). Per-layer accumulator widths grow
to keep the worst-case MAC sum lossless; downstream re-quantization
is the user's responsibility.

This frontend exists primarily to demonstrate that the IR supports
multibit arithmetic; production BitNet inference would also want
saturating quantization, registered pipelining, and weight ROM
emission, all of which are expressible in the same IR via the
``clamp``, ``register``, and ``rom`` kinds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _to_int_list_2d(t: torch.Tensor) -> List[List[int]]:
    return [
        [int(round(float(v))) for v in row.flatten().tolist()]
        for row in t
    ]


def _to_int_list_1d(t: torch.Tensor) -> List[int]:
    return [int(round(float(v))) for v in t.flatten().tolist()]


def _bits_for(value: int) -> int:
    """Bit width needed to express max(1, |value|) plus a sign bit."""
    return max(1, abs(value).bit_length()) + 1


@registry.register(
    "bitnet_linear",
    description="BitNet b1.58-style ternary linear layers with multibit activations.",
    metadata_namespace="bitnet_linear",
)
class BitNetLinearFrontend(Frontend):

    @classmethod
    def options(cls) -> List[FrontendOption]:
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
        ]

    def parse(
        self,
        path: Path,
        top: str = "top",
        activation_bits: int = 8,
        layer_prefix: str = "layers",
        **options,
    ) -> GateGraph:
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name).clone()

        # ---- Discover layer indices ----
        prefix_dot = f"{layer_prefix}."
        layer_indices: List[int] = []
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

        # ---- Walk layers ----
        gates: List[Gate] = []

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
        prev_outputs: List[str] = [s.name for s in input_signals]

        accumulator_width = activation_bits
        # For each successive layer, the accumulator can grow by
        # ceil(log2(in_size)) + 1 bits in the worst case (Σ |w*x| with
        # |w|=1 and in_size terms). One extra bit for sign safety.
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
            if not _is_ternary(w):
                raise ValueError(
                    f"layer {layer_idx} weights are not ternary {{-1, 0, 1}}"
                )

            out_size, layer_in = w.shape
            biases: Optional[List[int]] = None
            if bkey in tensors:
                biases = _to_int_list_1d(tensors[bkey])

            wrows = _to_int_list_2d(w)

            # Worst-case extra bits: ceil(log2(in)) + 1 (for sign safety).
            grow = max(1, layer_in.bit_length()) + 1
            mac_width = accumulator_width + grow

            this_outputs: List[str] = []
            for j in range(out_size):
                row = wrows[j]
                non_zero: List[Tuple[int, int]] = [
                    (i, sign) for i, sign in enumerate(row) if sign != 0
                ]
                bias_val = biases[j] if biases else 0
                out_name = f"L{layer_idx}.y{j}"

                if not non_zero and bias_val == 0:
                    gates.append(Gate(
                        name=out_name, kind="constant",
                        attrs={"value": 0},
                        output_width=mac_width, output_signed=True,
                    ))
                    this_outputs.append(out_name)
                    continue

                # Initial accumulator: bias (always emit so MAC width
                # matches subsequent add/sub operands).
                cur = f"L{layer_idx}.y{j}.init"
                gates.append(Gate(
                    name=cur, kind="constant",
                    attrs={"value": bias_val},
                    output_width=mac_width, output_signed=True,
                ))

                if not non_zero:
                    # Bias-only output: rename via a +0 buffer so the
                    # public output signal has the canonical layer name.
                    zero_name = f"L{layer_idx}.y{j}.zero"
                    gates.append(Gate(
                        name=zero_name, kind="constant",
                        attrs={"value": 0},
                        output_width=mac_width, output_signed=True,
                    ))
                    gates.append(Gate(
                        name=out_name, kind="add",
                        inputs=[cur, zero_name],
                        output_width=mac_width, output_signed=True,
                    ))
                    this_outputs.append(out_name)
                    continue

                for k, (i, sign) in enumerate(non_zero):
                    is_last = k == len(non_zero) - 1
                    nxt = (
                        out_name if is_last
                        else f"L{layer_idx}.y{j}.acc{k}"
                    )
                    gates.append(Gate(
                        name=nxt,
                        kind="add" if sign == 1 else "sub",
                        inputs=[cur, prev_outputs[i]],
                        output_width=mac_width, output_signed=True,
                    ))
                    cur = nxt
                this_outputs.append(out_name)

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
