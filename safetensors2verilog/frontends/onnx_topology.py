"""ONNX-topology frontend.

Reads an ONNX model file (provided via the ``--onnx`` option) for the
graph topology and a safetensors file for the weights. Each ONNX
node is lowered to one or more IR gates.

Supported ONNX ops in the initial implementation:
  Gemm        Y = alpha * A * B + beta * C   (alpha=beta=1, transB=1
              is the standard nn.Linear shape)
  MatMul      Y = A * B
  Add         elementwise add
  Sub         elementwise sub
  Mul         elementwise mul (operand-by-operand)
  Relu        max(0, x)
  Identity    pass-through
  Constant    constant tensor (must be scalar or 1-D)

Unsupported ops raise NotImplementedError with the op type and the
node name in the message; that is the natural place to extend
coverage.

Weights:
  ONNX initializers carry both names and values. For each
  initializer referenced by a supported op, the frontend prefers a
  same-name tensor in the safetensors file (for the case where a
  caller wants to swap weights without re-generating the .onnx
  graph), and falls back to the value embedded in the ONNX file.

Activation width:
  Set with ``--activation-bits`` (default 8). All activations are
  signed two's-complement at this width. The IR's accumulator widths
  grow per layer to keep MAC sums lossless.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from safetensors import safe_open

from ..core import Frontend, FrontendOption, Gate, GateGraph, Signal, registry


def _import_onnx():
    """Lazy-import onnx so the package import doesn't require onnx installed."""
    try:
        return importlib.import_module("onnx")
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the onnx_topology frontend requires the 'onnx' package; "
            "install it with `pip install onnx`."
        ) from e


def _to_int_list_2d(t: torch.Tensor) -> list[list[int]]:
    return [
        [int(round(float(v))) for v in row.flatten().tolist()]
        for row in t
    ]


def _to_int_list_1d(t: torch.Tensor) -> list[int]:
    return [int(round(float(v))) for v in t.flatten().tolist()]


def _is_integer_tensor(t: torch.Tensor, atol: float = 1e-6) -> bool:
    if not t.dtype.is_floating_point:
        return True
    tf = t.to(torch.float64)
    return bool(torch.isclose(tf, tf.round(), atol=atol).all().item())


@registry.register(
    "onnx_topology",
    description="ONNX graph topology + safetensors weights; subset of ops supported.",
    metadata_namespace="onnx_topology",
)
class OnnxTopologyFrontend(Frontend):

    @classmethod
    def options(cls) -> list[FrontendOption]:
        return [
            FrontendOption(
                name="onnx",
                type=str,
                default=None,
                help="path to the .onnx model file (required).",
                metavar="PATH",
            ),
            FrontendOption(
                name="activation-bits",
                type=int,
                default=8,
                help="bit width of activations (signed two's complement).",
            ),
            FrontendOption(
                name="weight-bits",
                type=int,
                default=8,
                help="expected bit width of weights (signed).",
            ),
        ]

    def parse(
        self,
        path: Path,
        top: str = "top",
        onnx: str | None = None,
        activation_bits: int = 8,
        weight_bits: int = 8,
        **options,
    ) -> GateGraph:
        if onnx is None:
            raise ValueError(
                "onnx_topology frontend requires --onnx PATH (the .onnx model)"
            )
        onnx_path = Path(onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        onnx_mod = _import_onnx()
        from onnx import numpy_helper  # noqa: WPS433

        model = onnx_mod.load(str(onnx_path))
        graph = model.graph

        # ---- Index initializers ----
        # First, prefer values from the safetensors file when present.
        st_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():
                st_tensors[name] = f.get_tensor(name).clone()

        initializers: dict[str, torch.Tensor] = {}
        for init in graph.initializer:
            if init.name in st_tensors:
                initializers[init.name] = st_tensors[init.name]
            else:
                arr = numpy_helper.to_array(init)
                initializers[init.name] = torch.from_numpy(arr.copy())

        # ---- Resolve graph inputs (excluding initializers) ----
        init_names = {i.name for i in graph.initializer}
        graph_inputs: list[tuple[str, list[int]]] = []
        for vi in graph.input:
            if vi.name in init_names:
                continue
            shape = [
                int(d.dim_value)
                for d in vi.type.tensor_type.shape.dim
                if d.dim_value > 0
            ]
            graph_inputs.append((vi.name, shape))

        if not graph_inputs:
            raise ValueError("ONNX graph has no non-initializer inputs")

        # Each non-initializer input becomes its own bank of ports.
        for in_name, in_shape in graph_inputs:
            if not in_shape or len(in_shape) > 2:
                raise NotImplementedError(
                    f"input '{in_name}' has shape {in_shape}; "
                    f"only 1-D or 2-D-with-batch=1 supported"
                )

        # ---- IR construction ----
        gates: list[Gate] = []

        # Maps ONNX value names -> list of IR signal names (one per element)
        # plus the (width, signed) of each.
        val_signals: dict[str, list[str]] = {}
        val_widths: dict[str, int] = {}
        val_signed: dict[str, bool] = {}

        # External inputs: one bank of signed activation_bits ports per
        # graph input. Single-input models keep the legacy `x0..xN-1`
        # naming; multi-input models prefix with the input name to keep
        # ports unambiguous.
        external: list[Signal] = []
        single_input = len(graph_inputs) == 1
        for in_name, in_shape in graph_inputs:
            in_size = in_shape[-1]
            if single_input:
                names = [f"x{i}" for i in range(in_size)]
            else:
                safe = "".join(
                    c if c.isalnum() else "_" for c in in_name
                ).lower()
                names = [f"{safe}_{i}" for i in range(in_size)]
            for n in names:
                external.append(
                    Signal(name=n, width=activation_bits, signed=True)
                )
            val_signals[in_name] = names
            val_widths[in_name] = activation_bits
            val_signed[in_name] = True

        accumulator_width = activation_bits

        def _bind_constant(name: str, t: torch.Tensor) -> None:
            """Record a constant-valued signal vector (used for biases / 1-D consts)."""
            if t.dim() > 1:
                raise NotImplementedError(
                    f"constant '{name}' has shape {tuple(t.shape)}; "
                    f"only scalar or 1-D constants supported"
                )
            if not _is_integer_tensor(t):
                raise ValueError(f"constant '{name}' is not integer-valued")
            vals = _to_int_list_1d(t)
            sig_names = []
            for i, v in enumerate(vals):
                gname = f"const.{name}.{i}"
                gates.append(Gate(
                    name=gname, kind="constant",
                    attrs={"value": int(v)},
                    output_width=activation_bits, output_signed=True,
                ))
                sig_names.append(gname)
            val_signals[name] = sig_names
            val_widths[name] = activation_bits
            val_signed[name] = True

        # ---- Walk nodes in topological order ----
        for node in graph.node:
            op = node.op_type
            ins = list(node.input)
            outs = list(node.output)

            if op == "Gemm":
                # Y = alpha*A*B + beta*C; default alpha=beta=1, transB defaults
                # to 0 in ONNX but Linear export sets transB=1 for [out, in]
                # weight layout. We require alpha=beta=1.
                attrs = {a.name: a for a in node.attribute}
                alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
                beta = float(attrs["beta"].f) if "beta" in attrs else 1.0
                trans_a = bool(attrs["transA"].i) if "transA" in attrs else False
                trans_b = bool(attrs["transB"].i) if "transB" in attrs else False
                if alpha != 1.0 or beta != 1.0:
                    raise NotImplementedError(
                        f"Gemm node '{node.name}': alpha={alpha} beta={beta}; "
                        f"only alpha=beta=1 supported"
                    )
                if trans_a:
                    raise NotImplementedError(
                        f"Gemm node '{node.name}': transA=1 not supported"
                    )
                a_name, b_name = ins[0], ins[1]
                c_name = ins[2] if len(ins) > 2 else None
                w = initializers[b_name]
                if not trans_b:
                    # weight is [in, out]; we want [out, in]
                    w = w.T.contiguous()
                if w.dim() != 2:
                    raise ValueError(
                        f"Gemm node '{node.name}': weight is not 2-D"
                    )
                bias_t = initializers.get(c_name) if c_name else None
                _emit_linear_layer(
                    gates, val_signals, val_widths, val_signed,
                    inputs_name=a_name, weight_tensor=w,
                    bias_tensor=bias_t,
                    out_value_name=outs[0],
                    layer_label=node.name or f"gemm_{len(gates)}",
                    weight_bits=weight_bits,
                    activation_bits=accumulator_width,
                )
                accumulator_width = val_widths[outs[0]]

            elif op == "MatMul":
                a_name, b_name = ins[0], ins[1]
                w = initializers[b_name]
                # Convention: B is [in, out] for MatMul of [N, in] x [in, out]
                if w.dim() != 2:
                    raise ValueError(
                        f"MatMul node '{node.name}': weight not 2-D"
                    )
                _emit_linear_layer(
                    gates, val_signals, val_widths, val_signed,
                    inputs_name=a_name, weight_tensor=w.T.contiguous(),
                    bias_tensor=None,
                    out_value_name=outs[0],
                    layer_label=node.name or f"matmul_{len(gates)}",
                    weight_bits=weight_bits,
                    activation_bits=accumulator_width,
                )
                accumulator_width = val_widths[outs[0]]

            elif op in ("Add", "Sub", "Mul"):
                if any(n not in val_signals and n not in initializers for n in ins[:2]):
                    raise ValueError(
                        f"{op} node '{node.name}': unresolved input "
                        f"(have signals={list(val_signals)[:5]}..., "
                        f"initializers={list(initializers)[:5]}...)"
                    )
                # If an input is an initializer, materialize it as constant gates.
                for ix in ins[:2]:
                    if ix not in val_signals:
                        _bind_constant(ix, initializers[ix])

                a_sigs = list(val_signals[ins[0]])
                b_sigs = list(val_signals[ins[1]])
                # Scalar-vector broadcasting: a length-1 operand replicates
                # to match the length of the other.
                if len(a_sigs) != len(b_sigs):
                    if len(a_sigs) == 1:
                        a_sigs = a_sigs * len(b_sigs)
                    elif len(b_sigs) == 1:
                        b_sigs = b_sigs * len(a_sigs)
                    else:
                        raise NotImplementedError(
                            f"{op} node '{node.name}': vector-vector "
                            f"broadcasting ({len(a_sigs)} vs {len(b_sigs)}) "
                            f"not supported; only scalar-vector"
                        )
                width = max(val_widths[ins[0]], val_widths[ins[1]]) + 1
                kind = {"Add": "add", "Sub": "sub", "Mul": "mul"}[op]
                if op == "Mul":
                    width = val_widths[ins[0]] + val_widths[ins[1]]

                out_sigs = []
                label = node.name or f"{op.lower()}_{len(gates)}"
                for i, (a, b) in enumerate(zip(a_sigs, b_sigs)):
                    name = f"{label}.{i}"
                    gates.append(Gate(
                        name=name, kind=kind,
                        inputs=[a, b],
                        output_width=width, output_signed=True,
                    ))
                    out_sigs.append(name)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = width
                val_signed[outs[0]] = True

            elif op == "Reshape":
                # Our IR is bit-level / 1-D, so Reshape between 1-D shapes
                # of equal element count is a no-op alias.
                src = ins[0]
                # ONNX Reshape's second input is the shape (an initializer
                # we ignore beyond confirming the element count).
                if src not in val_signals:
                    raise ValueError(
                        f"Reshape node '{node.name}': input '{src}' has no signals"
                    )
                val_signals[outs[0]] = list(val_signals[src])
                val_widths[outs[0]] = val_widths[src]
                val_signed[outs[0]] = val_signed[src]

            elif op == "Concat":
                # Concatenate signal lists in input order.
                concat_sigs: list[str] = []
                widths_seen = set()
                signed_seen = set()
                for ix in ins:
                    if ix not in val_signals:
                        if ix in initializers:
                            _bind_constant(ix, initializers[ix])
                        else:
                            raise ValueError(
                                f"Concat node '{node.name}': input '{ix}' unresolved"
                            )
                    concat_sigs.extend(val_signals[ix])
                    widths_seen.add(val_widths[ix])
                    signed_seen.add(val_signed[ix])
                if len(widths_seen) > 1 or len(signed_seen) > 1:
                    raise NotImplementedError(
                        f"Concat node '{node.name}': inputs differ in element "
                        f"width or sign; not supported"
                    )
                val_signals[outs[0]] = concat_sigs
                val_widths[outs[0]] = next(iter(widths_seen))
                val_signed[outs[0]] = next(iter(signed_seen))

            elif op == "Split":
                # ONNX Split divides input into N equal (or `split` attr)
                # parts along an axis. For our 1-D case, divide the signal
                # list among len(outs) outputs.
                src = ins[0]
                if src not in val_signals:
                    raise ValueError(
                        f"Split node '{node.name}': input '{src}' unresolved"
                    )
                src_sigs = val_signals[src]
                attrs = {a.name: a for a in node.attribute}
                if "split" in attrs:
                    sizes = list(attrs["split"].ints)
                elif len(ins) > 1 and ins[1] in initializers:
                    sizes = [int(v) for v in initializers[ins[1]].flatten().tolist()]
                else:
                    if len(src_sigs) % len(outs) != 0:
                        raise NotImplementedError(
                            f"Split node '{node.name}': uneven split "
                            f"({len(src_sigs)} into {len(outs)}) without "
                            f"explicit sizes"
                        )
                    chunk = len(src_sigs) // len(outs)
                    sizes = [chunk] * len(outs)
                if sum(sizes) != len(src_sigs):
                    raise ValueError(
                        f"Split node '{node.name}': sizes sum {sum(sizes)} "
                        f"!= source length {len(src_sigs)}"
                    )
                offset = 0
                for out_name, size in zip(outs, sizes):
                    val_signals[out_name] = src_sigs[offset:offset + size]
                    val_widths[out_name] = val_widths[src]
                    val_signed[out_name] = val_signed[src]
                    offset += size

            elif op == "Gather":
                # data[indices]; we support 1-D data and integer indices
                # supplied as an initializer.
                data_name, idx_name = ins[0], ins[1]
                if data_name not in val_signals:
                    raise ValueError(
                        f"Gather node '{node.name}': data '{data_name}' "
                        f"unresolved"
                    )
                if idx_name not in initializers:
                    raise NotImplementedError(
                        f"Gather node '{node.name}': dynamic indices "
                        f"(non-initializer) not supported"
                    )
                indices = [
                    int(v) for v in initializers[idx_name].flatten().tolist()
                ]
                src_sigs = val_signals[data_name]
                gathered = []
                for k in indices:
                    if k < 0:
                        k += len(src_sigs)
                    if not 0 <= k < len(src_sigs):
                        raise ValueError(
                            f"Gather node '{node.name}': index {k} out of "
                            f"range for length {len(src_sigs)}"
                        )
                    gathered.append(src_sigs[k])
                val_signals[outs[0]] = gathered
                val_widths[outs[0]] = val_widths[data_name]
                val_signed[outs[0]] = val_signed[data_name]

            elif op == "Relu":
                in_sigs = val_signals[ins[0]]
                relu_width = val_widths[ins[0]]
                label = node.name or f"relu_{len(gates)}"
                out_sigs = []
                for i, sig in enumerate(in_sigs):
                    name = f"{label}.{i}"
                    gates.append(Gate(
                        name=name, kind="relu",
                        inputs=[sig],
                        output_width=relu_width, output_signed=True,
                    ))
                    out_sigs.append(name)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = relu_width
                val_signed[outs[0]] = True

            elif op == "Identity":
                # Pass-through: alias the output name to the input's signals.
                val_signals[outs[0]] = val_signals[ins[0]]
                val_widths[outs[0]] = val_widths[ins[0]]
                val_signed[outs[0]] = val_signed[ins[0]]

            elif op == "Constant":
                # An inline constant tensor.
                attrs = {a.name: a for a in node.attribute}
                if "value" not in attrs:
                    raise NotImplementedError(
                        f"Constant node '{node.name}': only 'value' attribute supported"
                    )
                arr = numpy_helper.to_array(attrs["value"].t)
                _bind_constant(outs[0], torch.from_numpy(arr.copy()))

            else:
                supported = (
                    "Gemm, MatMul, Add, Sub, Mul, Relu, Identity, Constant, "
                    "Reshape, Concat, Split, Gather"
                )
                deferred = (
                    "Conv, ConvTranspose (need 2-D windowed access), "
                    "LayerNorm, GroupNorm, BatchNorm (need fixed-point "
                    "sqrt/divide), Softmax, Sigmoid, Tanh, Exp (need "
                    "fixed-point transcendentals), Attention (composite "
                    "of softmax + matmul)"
                )
                raise NotImplementedError(
                    f"ONNX op '{op}' (node '{node.name}') not supported by "
                    f"the onnx_topology frontend.\n  Supported: {supported}.\n"
                    f"  Deferred: {deferred}.\n"
                    f"  Add a custom kind via @lowering(...) and a node "
                    f"branch here to extend coverage."
                )

        # ---- Resolve graph outputs ----
        graph_output_names = [vi.name for vi in graph.output]
        output_signals: list[Signal] = []
        for gn in graph_output_names:
            if gn not in val_signals:
                raise ValueError(
                    f"graph output '{gn}' was never produced by any node"
                )
            for sig_name in val_signals[gn]:
                output_signals.append(Signal(
                    name=sig_name,
                    width=val_widths[gn],
                    signed=val_signed[gn],
                ))

        return GateGraph(
            inputs=external,
            outputs=output_signals,
            gates=gates,
            top=top,
        )


def _emit_linear_layer(
    gates: list[Gate],
    val_signals: dict[str, list[str]],
    val_widths: dict[str, int],
    val_signed: dict[str, bool],
    *,
    inputs_name: str,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor | None,
    out_value_name: str,
    layer_label: str,
    weight_bits: int,
    activation_bits: int,
) -> None:
    """Emit one linear gate per output neuron."""
    if not _is_integer_tensor(weight_tensor):
        raise ValueError(f"layer '{layer_label}': weights are not integer-valued")
    weight_lo = -(1 << (weight_bits - 1))
    weight_hi = (1 << (weight_bits - 1)) - 1
    wrows = _to_int_list_2d(weight_tensor)
    for j, row in enumerate(wrows):
        for i, wij in enumerate(row):
            if wij < weight_lo or wij > weight_hi:
                raise ValueError(
                    f"layer '{layer_label}' weight[{j}][{i}]={wij} outside "
                    f"[{weight_lo}, {weight_hi}] for weight_bits={weight_bits}"
                )

    biases: list[int] | None = None
    if bias_tensor is not None:
        if not _is_integer_tensor(bias_tensor):
            raise ValueError(f"layer '{layer_label}' bias is not integer-valued")
        biases = _to_int_list_1d(bias_tensor)

    in_sigs = val_signals[inputs_name]
    layer_in = len(in_sigs)
    out_size = weight_tensor.shape[0]
    if weight_tensor.shape[1] != layer_in:
        raise ValueError(
            f"layer '{layer_label}': weight in_features={weight_tensor.shape[1]} "
            f"!= prior stage signals {layer_in}"
        )

    grow = weight_bits + max(1, layer_in.bit_length())
    mac_width = activation_bits + grow

    out_sigs: list[str] = []
    for j in range(out_size):
        row = wrows[j]
        bias_val = biases[j] if biases else 0
        name = f"{layer_label}.y{j}"
        gates.append(Gate(
            name=name, kind="linear",
            inputs=list(in_sigs),
            attrs={"weights": row, "bias": bias_val},
            output_width=mac_width, output_signed=True,
        ))
        out_sigs.append(name)

    val_signals[out_value_name] = out_sigs
    val_widths[out_value_name] = mac_width
    val_signed[out_value_name] = True
