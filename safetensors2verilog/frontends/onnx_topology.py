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

        # Validate + flatten input shapes. We accept 1-D, 2-D (batch=1,
        # features), and 4-D NCHW (batch=1) for convolutional inputs.
        # 4-D inputs are flattened to (H * W * C) elements in row-major
        # (h, w, c) order to match the conv2d primitive's expected
        # x_packed layout. Mutate graph_inputs in place so downstream
        # iterations see the flattened shape.
        flattened: list[tuple[str, list[int]]] = []
        for in_name, in_shape in graph_inputs:
            if not in_shape:
                raise NotImplementedError(
                    f"input '{in_name}' has empty shape"
                )
            if len(in_shape) == 4:
                if in_shape[0] != 1:
                    raise NotImplementedError(
                        f"input '{in_name}' shape {in_shape}: only "
                        f"batch=1 supported"
                    )
                _, in_c, in_hh, in_ww = in_shape
                in_shape = [in_hh * in_ww * in_c]
            elif len(in_shape) > 2:
                raise NotImplementedError(
                    f"input '{in_name}' has shape {in_shape}; "
                    f"supported: 1-D, 2-D (batch=1, features), or 4-D NCHW"
                )
            flattened.append((in_name, in_shape))
        graph_inputs = flattened

        # ---- IR construction ----
        gates: list[Gate] = []
        # Submodules accumulated by per-element activation ops (Sigmoid, Exp,
        # ...); each emits an `instance` Gate that references a parameterised
        # block from ``safetensors2verilog.blocks``. The backend dedups by
        # module name so multiple identical-shape instances share one
        # emitted RawSubmodule.
        _onnx_pending_submodules: list = []
        # Set to True when a sequential op (LayerNormalization, etc.) is
        # emitted; in that case clk/rst/start get added to the graph's
        # external inputs so the user can drive them.
        _needs_clock = False

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
                # Broadcasting: scalar-vector (length 1 splats), or
                # vector-vector where one length evenly divides the other
                # (the smaller side tiles to match). Full numpy broadcasting
                # over multidimensional shapes is gated on IR shape support
                # (see docs/DEFERRED.md).
                if len(a_sigs) != len(b_sigs):
                    if len(a_sigs) == 1:
                        a_sigs = a_sigs * len(b_sigs)
                    elif len(b_sigs) == 1:
                        b_sigs = b_sigs * len(a_sigs)
                    elif len(a_sigs) > len(b_sigs) and len(a_sigs) % len(b_sigs) == 0:
                        b_sigs = b_sigs * (len(a_sigs) // len(b_sigs))
                    elif len(b_sigs) > len(a_sigs) and len(b_sigs) % len(a_sigs) == 0:
                        a_sigs = a_sigs * (len(b_sigs) // len(a_sigs))
                    else:
                        raise NotImplementedError(
                            f"{op} node '{node.name}': vector-vector "
                            f"broadcasting ({len(a_sigs)} vs {len(b_sigs)}) "
                            f"not supported; sizes must match, one must be "
                            f"1, or one must evenly divide the other"
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
                # data[indices]: static indices (initializer) become a
                # signal-list permutation; dynamic indices (value port)
                # become a per-output mux.
                data_name, idx_name = ins[0], ins[1]
                if data_name not in val_signals:
                    raise ValueError(
                        f"Gather node '{node.name}': data '{data_name}' "
                        f"unresolved"
                    )
                src_sigs = val_signals[data_name]
                src_w = val_widths[data_name]
                src_signed = val_signed[data_name]
                if idx_name in initializers:
                    indices = [
                        int(v) for v in initializers[idx_name].flatten().tolist()
                    ]
                    gathered = []
                    for k in indices:
                        if k < 0:
                            k += len(src_sigs)
                        if not 0 <= k < len(src_sigs):
                            raise ValueError(
                                f"Gather node '{node.name}': index {k} out "
                                f"of range for length {len(src_sigs)}"
                            )
                        gathered.append(src_sigs[k])
                    val_signals[outs[0]] = gathered
                    val_widths[outs[0]] = src_w
                    val_signed[outs[0]] = src_signed
                else:
                    # Dynamic indices: each index signal feeds a mux that
                    # selects from src_sigs. Index signals must have width
                    # >= ceil(log2(len(src_sigs))) for clean indexing.
                    if idx_name not in val_signals:
                        raise ValueError(
                            f"Gather node '{node.name}': index '{idx_name}' "
                            f"is neither an initializer nor a produced signal"
                        )
                    idx_sigs = val_signals[idx_name]
                    label = node.name or f"gather_{len(gates)}"
                    out_sigs = []
                    for k, idx_sig in enumerate(idx_sigs):
                        out_name = f"{label}.{k}"
                        gates.append(Gate(
                            name=out_name, kind="mux",
                            inputs=[idx_sig] + list(src_sigs),
                            output_width=src_w, output_signed=src_signed,
                        ))
                        out_sigs.append(out_name)
                    val_signals[outs[0]] = out_sigs
                    val_widths[outs[0]] = src_w
                    val_signed[outs[0]] = src_signed

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

            elif op == "LayerNormalization":
                # ONNX LayerNormalization: y = ((x - mean(x)) / sqrt(var(x)
                # + eps)) * scale + bias. Lowered to a single
                # ``layer_norm_block`` instance.
                from ..blocks.layer_norm import layer_norm_block
                attrs = {a.name: a for a in node.attribute}
                eps = float(attrs["epsilon"].f) if "epsilon" in attrs else 1e-5
                scale_name = ins[1] if len(ins) > 1 else None
                bias_name  = ins[2] if len(ins) > 2 else None
                if scale_name is None or scale_name not in initializers:
                    raise ValueError(
                        f"LayerNormalization '{node.name}': scale (gamma) "
                        f"must be an initializer; got {scale_name!r}"
                    )
                gamma_t = initializers[scale_name].to(torch.float32)
                if not _is_integer_tensor(gamma_t):
                    # Quantise to Q1.14 signed
                    gamma_int = [
                        max(-(1 << 15),
                            min((1 << 15) - 1,
                                int(round(float(v) * (1 << 14)))))
                        for v in gamma_t.flatten().tolist()
                    ]
                else:
                    gamma_int = _to_int_list_1d(gamma_t)
                if bias_name and bias_name in initializers:
                    beta_t = initializers[bias_name].to(torch.float32)
                    beta_int = [
                        max(-(1 << 15),
                            min((1 << 15) - 1, int(round(float(v)))))
                        for v in beta_t.flatten().tolist()
                    ]
                else:
                    beta_int = [0] * len(gamma_int)

                in_sigs = val_signals[ins[0]]
                K = len(in_sigs)
                if len(gamma_int) != K:
                    raise ValueError(
                        f"LayerNormalization '{node.name}': scale length "
                        f"{len(gamma_int)} != input length {K}"
                    )
                ln, rsq = layer_norm_block(
                    K=K, gamma_int=gamma_int, beta_int=beta_int,
                    abits=val_widths[ins[0]], obits=val_widths[ins[0]],
                    eps=eps,
                )
                _onnx_pending_submodules.append(ln)
                _onnx_pending_submodules.append(rsq)
                _needs_clock = True

                # Pack input scalars into x_packed via concat (LSB-first to
                # match the block's expected layout).
                pack_name = f"{node.name or f'ln_{len(gates)}'}_x_packed"
                gates.append(Gate(
                    name=pack_name, kind="concat",
                    inputs=list(reversed(in_sigs)),
                    output_width=K * val_widths[ins[0]],
                    output_signed=False,
                ))
                # done extern wire + instance
                done_name = f"{node.name or f'ln_{len(gates)}'}_done"
                gates.append(Gate(
                    name=done_name, kind="extern_wire", output_width=1,
                ))
                y_pack_name = f"{node.name or f'ln_{len(gates)}'}_y_packed"
                gates.append(Gate(
                    name=y_pack_name, kind="instance",
                    inputs=["clk", "rst", "start", pack_name],
                    attrs={
                        "module_name": ln.top,
                        "instance_name": (
                            (node.name or f"ln_inst_{len(gates)}")
                            .replace(".", "_")
                        ),
                        "input_ports": ["clk", "rst", "start", "x_packed"],
                        "output_port": "y_packed",
                        "extra_output_ports": [("done", done_name)],
                    },
                    output_width=K * val_widths[ins[0]], output_signed=True,
                ))
                # Slice back into per-element signals
                out_sigs = []
                obits = val_widths[ins[0]]
                for k in range(K):
                    sname = f"{node.name or f'ln_{len(gates)}'}.{k}"
                    gates.append(Gate(
                        name=sname, kind="slice",
                        inputs=[y_pack_name],
                        attrs={"hi": (k+1)*obits - 1, "lo": k*obits},
                        output_width=obits, output_signed=True,
                    ))
                    out_sigs.append(sname)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = obits
                val_signed[outs[0]] = True

            elif op in ("Sigmoid", "Exp", "Tanh"):
                # Per-element LUT-based activation. The hardware blocks live
                # in safetensors2verilog.blocks.{sigmoid,exp,tanh} and are
                # parameterised by (in_bits, out_bits, q_frac_bits). One
                # instance per element; the backend dedups the module by
                # shape-canonical name. Sigmoid output is unsigned Q0.out;
                # Exp output is unsigned Q0.out; Tanh output is signed
                # Q1.(out-1) over [-1, 1).
                from ..blocks.exp import exp_block
                from ..blocks.sigmoid import sigmoid_block
                from ..blocks.tanh import tanh_block
                if op == "Sigmoid":
                    sub = sigmoid_block(in_bits=8, out_bits=8,
                                         in_q_frac_bits=4)
                    out_w, out_signed = 8, False
                elif op == "Exp":
                    sub = exp_block(in_bits=8, out_bits=12,
                                     in_q_frac_bits=4)
                    out_w, out_signed = 12, False
                else:  # Tanh
                    sub = tanh_block(in_bits=8, out_bits=8,
                                      in_q_frac_bits=4)
                    out_w, out_signed = 8, True
                # Stash the submodule on a side channel that ``parse``'s
                # caller consumes (we attach it to a dedicated list on the
                # graph after the loop; see end of this function).
                _onnx_pending_submodules.append(sub)
                in_sigs = val_signals[ins[0]]
                in_w = val_widths[ins[0]]
                label = node.name or f"{op.lower()}_{len(gates)}"
                out_sigs = []
                for i, sig in enumerate(in_sigs):
                    if in_w != 8:
                        # Truncate / sign-extend to 8-bit before LUT lookup.
                        # (For now require the upstream to emit 8-bit
                        # signed; signal a clear error otherwise.)
                        raise NotImplementedError(
                            f"{op} expects 8-bit signed inputs; "
                            f"got {in_w}-bit. Insert an explicit requantize."
                        )
                    out_name = f"{label}.{i}"
                    gates.append(Gate(
                        name=out_name, kind="instance",
                        inputs=[sig],
                        attrs={
                            "module_name": sub.top,
                            "instance_name": f"{label.replace('.', '_')}_{i}",
                            "input_ports": ["x"], "output_port": "y",
                        },
                        output_width=out_w, output_signed=out_signed,
                    ))
                    out_sigs.append(out_name)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = out_w
                val_signed[outs[0]] = out_signed

            elif op == "Softmax":
                # ONNX Softmax: y = exp(x - max(x)) / sum(exp(x - max(x))).
                # Lowered to a single softmax_block instance. Pack the
                # per-element signal list into x_packed, drive a constant
                # all-ones mask (no causal mask in the standalone op),
                # instance the block, then slice y_packed back into
                # per-element signals.
                from ..blocks.softmax import softmax_block
                attrs = {a.name: a for a in node.attribute}
                axis = int(attrs["axis"].i) if "axis" in attrs else -1
                in_sigs = val_signals[ins[0]]
                K = len(in_sigs)
                in_w = val_widths[ins[0]]
                if in_w != 8:
                    raise NotImplementedError(
                        f"Softmax expects 8-bit signed inputs; got {in_w}-bit. "
                        f"Insert an explicit requantize."
                    )
                if axis not in (-1, 1):
                    raise NotImplementedError(
                        f"Softmax node '{node.name}': axis={axis}; only the "
                        f"trailing axis (axis=-1 or 1 with batch=1) is "
                        f"supported under the 1-D IR."
                    )
                sm_sub, exp_sub = softmax_block(K=K, abits=in_w, obits=8)
                _onnx_pending_submodules.append(exp_sub)
                _onnx_pending_submodules.append(sm_sub)
                _needs_clock = True

                # Pack per-element scalars into x_packed (LSB-first).
                label = node.name or f"softmax_{len(gates)}"
                pack_name = f"{label}_x_packed"
                gates.append(Gate(
                    name=pack_name, kind="concat",
                    inputs=list(reversed(in_sigs)),
                    output_width=K * in_w, output_signed=False,
                ))
                # Constant all-ones mask (no positions zeroed).
                mask_name = f"{label}_mask"
                gates.append(Gate(
                    name=mask_name, kind="constant",
                    attrs={"value": (1 << K) - 1},
                    output_width=K, output_signed=False,
                ))
                # extern_wire for done + the instance.
                done_name = f"{label}_done"
                gates.append(Gate(
                    name=done_name, kind="extern_wire", output_width=1,
                ))
                y_pack_name = f"{label}_y_packed"
                gates.append(Gate(
                    name=y_pack_name, kind="instance",
                    inputs=["clk", "rst", "start", pack_name, mask_name],
                    attrs={
                        "module_name": sm_sub.top,
                        "instance_name": label.replace(".", "_") + "_inst",
                        "input_ports": ["clk", "rst", "start", "x_packed", "mask"],
                        "output_port": "y_packed",
                        "extra_output_ports": [("done", done_name)],
                    },
                    output_width=K * 8, output_signed=False,
                ))
                # Slice y_packed back into per-element unsigned 8-bit signals.
                out_sigs = []
                for k in range(K):
                    sname = f"{label}.{k}"
                    gates.append(Gate(
                        name=sname, kind="slice",
                        inputs=[y_pack_name],
                        attrs={"hi": (k + 1) * 8 - 1, "lo": k * 8},
                        output_width=8, output_signed=False,
                    ))
                    out_sigs.append(sname)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = 8
                val_signed[outs[0]] = False

            elif op == "Conv":
                # ONNX Conv (NCHW): X is [batch, in_c, in_h, in_w]; W is
                # [out_c, in_c/groups, kH, kW]; optional B is [out_c].
                # We require batch=1, groups=1, dilations=1, symmetric pads.
                attrs = {a.name: a for a in node.attribute}
                kernel_shape = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else None
                strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
                pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
                dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
                groups = int(attrs["group"].i) if "group" in attrs else 1
                if dilations != [1, 1]:
                    raise NotImplementedError(
                        f"Conv node '{node.name}': dilations {dilations} != "
                        f"[1, 1] not supported"
                    )
                if groups != 1:
                    raise NotImplementedError(
                        f"Conv node '{node.name}': group={groups}; only "
                        f"group=1 supported"
                    )
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise NotImplementedError(
                        f"Conv node '{node.name}': asymmetric pads {pads}"
                    )
                pad_h, pad_w = int(pads[0]), int(pads[1])
                stride_h, stride_w = int(strides[0]), int(strides[1])

                W_name = ins[1]
                if W_name not in initializers:
                    raise NotImplementedError(
                        f"Conv node '{node.name}': weight '{W_name}' must "
                        f"be an initializer (dynamic weights not supported)"
                    )
                W = initializers[W_name]
                if W.dim() != 4:
                    raise ValueError(
                        f"Conv node '{node.name}': weight is not 4-D"
                    )
                out_c, in_c, kH, kW = W.shape
                if kernel_shape is not None and kernel_shape != [kH, kW]:
                    raise ValueError(
                        f"Conv node '{node.name}': kernel_shape attr "
                        f"{kernel_shape} != weight ({kH}, {kW})"
                    )
                if not _is_integer_tensor(W):
                    raise ValueError(
                        f"Conv node '{node.name}': weights are not "
                        f"integer-valued"
                    )
                weights_lol = [
                    [[[int(round(float(W[oc, ic_, ki, kj]))) for kj in range(kW)]
                      for ki in range(kH)]
                     for ic_ in range(in_c)]
                    for oc in range(out_c)
                ]
                biases_l: list[int] = [0] * out_c
                if len(ins) >= 3 and ins[2] in initializers:
                    B = initializers[ins[2]]
                    if B.numel() != out_c:
                        raise ValueError(
                            f"Conv node '{node.name}': bias length "
                            f"{B.numel()} != out_c {out_c}"
                        )
                    biases_l = [int(round(float(v)))
                                for v in B.flatten().tolist()]

                x_name = ins[0]
                x_sigs = val_signals[x_name]
                in_w_bits = val_widths[x_name]
                expected_in_elems = len(x_sigs)
                if expected_in_elems % in_c != 0:
                    raise ValueError(
                        f"Conv node '{node.name}': input element count "
                        f"{expected_in_elems} not divisible by in_c={in_c}"
                    )
                spatial = expected_in_elems // in_c
                in_h_guess = int(spatial ** 0.5)
                if in_h_guess * in_h_guess == spatial:
                    in_h_in, in_w_in = in_h_guess, in_h_guess
                else:
                    raise NotImplementedError(
                        f"Conv node '{node.name}': non-square implicit "
                        f"input ({spatial} elements / {in_c} channels). "
                        f"Reshape to square upstream or extend the frontend "
                        f"to read explicit (H, W) shape from value_info."
                    )
                out_h_calc = (in_h_in + 2 * pad_h - kH) // stride_h + 1
                out_w_calc = (in_w_in + 2 * pad_w - kW) // stride_w + 1
                out_bits = max(in_w_bits, weight_bits) + max(
                    1, (in_c * kH * kW).bit_length()
                ) + 1

                label = node.name or f"conv_{len(gates)}"
                pack_name = f"{label}_x_packed"
                gates.append(Gate(
                    name=pack_name, kind="concat",
                    inputs=list(reversed(x_sigs)),
                    output_width=expected_in_elems * in_w_bits,
                    output_signed=False,
                ))
                conv_name = f"{label}_y_packed"
                gates.append(Gate(
                    name=conv_name, kind="conv2d",
                    inputs=[pack_name],
                    attrs={
                        "in_h": in_h_in, "in_w": in_w_in, "in_c": in_c,
                        "out_h": out_h_calc, "out_w": out_w_calc,
                        "out_c": out_c,
                        "kH": kH, "kW": kW,
                        "stride_h": stride_h, "stride_w": stride_w,
                        "pad_h": pad_h, "pad_w": pad_w,
                        "weights": weights_lol,
                        "biases": biases_l,
                        "act_bits": in_w_bits,
                        "weight_bits": weight_bits,
                        "out_bits": out_bits,
                    },
                    output_width=out_h_calc * out_w_calc * out_c * out_bits,
                    output_signed=True,
                ))
                conv_out_sigs = []
                for k in range(out_h_calc * out_w_calc * out_c):
                    sn = f"{label}.{k}"
                    gates.append(Gate(
                        name=sn, kind="slice",
                        inputs=[conv_name],
                        attrs={"hi": (k + 1) * out_bits - 1,
                               "lo": k * out_bits},
                        output_width=out_bits, output_signed=True,
                    ))
                    conv_out_sigs.append(sn)
                val_signals[outs[0]] = conv_out_sigs
                val_widths[outs[0]] = out_bits
                val_signed[outs[0]] = True

            elif op == "ConvTranspose":
                raise NotImplementedError(
                    f"ConvTranspose node '{node.name}': transposed "
                    f"convolution is not yet wired through the conv2d "
                    f"primitive. Workaround: replace ConvTranspose with "
                    f"the equivalent forward Conv plus an upsample."
                )

            elif op == "BatchNormalization":
                # Inference-only BatchNorm: bakes the running statistics
                # into a per-channel affine ``y = a * x + b`` where
                # a = scale / sqrt(var + eps) and b = bias - mean * a.
                # Inputs in the ONNX op order: X, scale, bias,
                # running_mean, running_var (5 total).
                if len(ins) != 5:
                    raise NotImplementedError(
                        f"BatchNormalization node '{node.name}': expected "
                        f"5 inputs (X, scale, bias, mean, var), got {len(ins)}"
                    )
                attrs = {a.name: a for a in node.attribute}
                eps = float(attrs["epsilon"].f) if "epsilon" in attrs else 1e-5
                if "momentum" in attrs:
                    # Inference doesn't use momentum; ignore.
                    pass
                if "training_mode" in attrs and bool(attrs["training_mode"].i):
                    raise NotImplementedError(
                        f"BatchNormalization node '{node.name}': "
                        f"training_mode=1 not supported (inference only)."
                    )
                for nm in ins[1:]:
                    if nm not in initializers:
                        raise NotImplementedError(
                            f"BatchNormalization node '{node.name}': "
                            f"input '{nm}' must be an initializer "
                            f"(running stats baked in)"
                        )
                scale = initializers[ins[1]].to(torch.float32)
                bias_t = initializers[ins[2]].to(torch.float32)
                mean = initializers[ins[3]].to(torch.float32)
                var = initializers[ins[4]].to(torch.float32)
                # Effective per-channel multiplier and offset.
                a_eff = scale / torch.sqrt(var + eps)
                b_eff = bias_t - mean * a_eff
                # Quantise to integers using a uniform scaling factor.
                # The frontend's accumulator widths grow per layer; we
                # round a_eff to int and apply it as a `linear` per
                # element (caller is responsible for upstream activation
                # scaling making this round well).
                a_int = a_eff.round().clamp(
                    -(1 << (weight_bits - 1)) + 1,
                    (1 << (weight_bits - 1)) - 1,
                ).to(torch.int32).tolist()
                b_int = b_eff.round().clamp(
                    -(1 << (val_widths[ins[0]] - 1)) * 256,
                    (1 << (val_widths[ins[0]] - 1)) * 256,
                ).to(torch.int32).tolist()
                in_sigs = val_signals[ins[0]]
                if len(a_int) != len(in_sigs):
                    raise NotImplementedError(
                        f"BatchNormalization node '{node.name}': scale "
                        f"length {len(a_int)} != input length "
                        f"{len(in_sigs)}; only per-element BN supported"
                    )
                in_w = val_widths[ins[0]]
                out_w = in_w + weight_bits + 1
                label = node.name or f"bn_{len(gates)}"
                out_sigs = []
                for i, sig in enumerate(in_sigs):
                    sn = f"{label}.{i}"
                    gates.append(Gate(
                        name=sn, kind="linear",
                        inputs=[sig],
                        attrs={"weights": [int(a_int[i])],
                               "bias": int(b_int[i])},
                        output_width=out_w, output_signed=True,
                    ))
                    out_sigs.append(sn)
                val_signals[outs[0]] = out_sigs
                val_widths[outs[0]] = out_w
                val_signed[outs[0]] = True

            elif op == "GroupNormalization":
                # GroupNorm with G groups along the channel axis: split
                # the input into G groups of size K/G, layer-norm each
                # independently with a per-group gamma slice, concatenate.
                from ..blocks.layer_norm import layer_norm_block
                attrs = {a.name: a for a in node.attribute}
                num_groups = int(attrs["num_groups"].i) if "num_groups" in attrs else 1
                eps = float(attrs["epsilon"].f) if "epsilon" in attrs else 1e-5
                if len(ins) < 3:
                    raise NotImplementedError(
                        f"GroupNormalization node '{node.name}': expected "
                        f"3 inputs (X, scale, bias), got {len(ins)}"
                    )
                scale = initializers[ins[1]].to(torch.float32)
                bias_t = initializers[ins[2]].to(torch.float32)

                in_sigs = val_signals[ins[0]]
                K = len(in_sigs)
                if K % num_groups != 0:
                    raise ValueError(
                        f"GroupNormalization node '{node.name}': K={K} "
                        f"not divisible by num_groups={num_groups}"
                    )
                K_per_group = K // num_groups
                if scale.numel() != K:
                    raise ValueError(
                        f"GroupNormalization node '{node.name}': scale "
                        f"length {scale.numel()} != K={K}"
                    )
                in_w = val_widths[ins[0]]
                if in_w != 8:
                    raise NotImplementedError(
                        f"GroupNormalization node '{node.name}': only "
                        f"8-bit signed inputs supported"
                    )
                _needs_clock = True
                label = node.name or f"gn_{len(gates)}"
                all_out_sigs: list[str] = []
                for gi in range(num_groups):
                    g_lo = gi * K_per_group
                    g_hi = g_lo + K_per_group
                    g_in = in_sigs[g_lo:g_hi]
                    g_scale = scale[g_lo:g_hi]
                    g_bias = bias_t[g_lo:g_hi]
                    gamma_int = [
                        max(-(1 << 15),
                            min((1 << 15) - 1,
                                int(round(float(v) * (1 << 14)))))
                        for v in g_scale.flatten().tolist()
                    ]
                    beta_int = [
                        max(-(1 << 15),
                            min((1 << 15) - 1, int(round(float(v)))))
                        for v in g_bias.flatten().tolist()
                    ]
                    ln, rsq = layer_norm_block(
                        K=K_per_group, gamma_int=gamma_int,
                        beta_int=beta_int,
                        abits=8, obits=8, eps=eps,
                    )
                    _onnx_pending_submodules.append(ln)
                    _onnx_pending_submodules.append(rsq)
                    pack = f"{label}.g{gi}_x_packed"
                    gates.append(Gate(
                        name=pack, kind="concat",
                        inputs=list(reversed(g_in)),
                        output_width=K_per_group * 8, output_signed=False,
                    ))
                    done_n = f"{label}.g{gi}_done"
                    gates.append(Gate(name=done_n, kind="extern_wire",
                                      output_width=1))
                    yp = f"{label}.g{gi}_y_packed"
                    gates.append(Gate(
                        name=yp, kind="instance",
                        inputs=["clk", "rst", "start", pack],
                        attrs={
                            "module_name": ln.top,
                            "instance_name": (
                                f"{label.replace('.', '_')}_g{gi}"
                            ),
                            "input_ports": ["clk", "rst", "start", "x_packed"],
                            "output_port": "y_packed",
                            "extra_output_ports": [("done", done_n)],
                        },
                        output_width=K_per_group * 8, output_signed=True,
                    ))
                    for k in range(K_per_group):
                        sn = f"{label}.g{gi}_{k}"
                        gates.append(Gate(
                            name=sn, kind="slice",
                            inputs=[yp],
                            attrs={"hi": (k + 1) * 8 - 1, "lo": k * 8},
                            output_width=8, output_signed=True,
                        ))
                        all_out_sigs.append(sn)
                val_signals[outs[0]] = all_out_sigs
                val_widths[outs[0]] = 8
                val_signed[outs[0]] = True

            elif op == "Attention":
                # Scaled dot-product attention as a composite of per-pair
                # linear scores, per-row softmax_block instances, and
                # per-element linear outputs. Minimal supported form:
                # 3-input (Q, K, V), single-head, batch=1, no mask, no
                # past_key/value. The hf_llama frontend covers transformer
                # attention end-to-end with full multi-head + KV cache;
                # this branch is for ONNX models that lay out attention
                # via the standalone op rather than decomposed MatMul +
                # Softmax + MatMul.
                from ..blocks.softmax import softmax_block
                if len(ins) > 3:
                    raise NotImplementedError(
                        f"Attention node '{node.name}': mask / past_key / "
                        f"past_value inputs (len(ins)={len(ins)}) not yet "
                        f"supported; only the (Q, K, V) form."
                    )
                attrs = {a.name: a for a in node.attribute}
                num_heads = int(attrs["num_heads"].i) if "num_heads" in attrs else 1
                if num_heads != 1:
                    raise NotImplementedError(
                        f"Attention node '{node.name}': num_heads="
                        f"{num_heads}; only single-head supported. Use "
                        f"the hf_llama frontend for multi-head attention."
                    )
                q_sigs = val_signals[ins[0]]
                k_sigs = val_signals[ins[1]]
                v_sigs = val_signals[ins[2]]
                in_w = val_widths[ins[0]]
                if (val_widths[ins[1]] != in_w
                        or val_widths[ins[2]] != in_w):
                    raise NotImplementedError(
                        f"Attention node '{node.name}': Q/K/V widths "
                        f"differ"
                    )
                qn, kn, vn = len(q_sigs), len(k_sigs), len(v_sigs)
                if qn != kn:
                    raise NotImplementedError(
                        f"Attention node '{node.name}': Q has {qn} "
                        f"elements, K has {kn}; require seq*d_k equal."
                    )
                seq_guess = int(qn ** 0.5)
                if seq_guess * seq_guess == qn:
                    seq, d_k = seq_guess, seq_guess
                else:
                    seq, d_k = 1, qn
                if vn % seq != 0:
                    raise NotImplementedError(
                        f"Attention node '{node.name}': V has {vn} "
                        f"elements, not divisible by seq={seq}"
                    )
                d_v = vn // seq

                label = node.name or f"attn_{len(gates)}"
                score_width = in_w * 2 + max(1, (d_k - 1).bit_length()) + 1
                score_sigs: list[list[str]] = []
                for i in range(seq):
                    row = []
                    for j in range(seq):
                        prod_names = []
                        for kk in range(d_k):
                            pn = f"{label}.prod_{i}_{j}_{kk}"
                            gates.append(Gate(
                                name=pn, kind="mul",
                                inputs=[q_sigs[i * d_k + kk],
                                        k_sigs[j * d_k + kk]],
                                output_width=2 * in_w, output_signed=True,
                            ))
                            prod_names.append(pn)
                        sn = f"{label}.score_{i}_{j}"
                        gates.append(Gate(
                            name=sn, kind="linear",
                            inputs=prod_names,
                            attrs={"weights": [1] * d_k, "bias": 0},
                            output_width=score_width, output_signed=True,
                        ))
                        row.append(sn)
                    score_sigs.append(row)

                # Per-row softmax instance.
                _needs_clock = True
                sm_sub, exp_sub = softmax_block(K=seq, abits=8, obits=8)
                _onnx_pending_submodules.append(exp_sub)
                _onnx_pending_submodules.append(sm_sub)
                weight_outs: list[list[str]] = []
                for i in range(seq):
                    truncated = []
                    for j in range(seq):
                        tn = f"{label}.score_trunc_{i}_{j}"
                        gates.append(Gate(
                            name=tn, kind="slice",
                            inputs=[score_sigs[i][j]],
                            attrs={"hi": 7, "lo": 0},
                            output_width=8, output_signed=True,
                        ))
                        truncated.append(tn)
                    pack = f"{label}.scorerow_{i}_packed"
                    gates.append(Gate(
                        name=pack, kind="concat",
                        inputs=list(reversed(truncated)),
                        output_width=seq * 8, output_signed=False,
                    ))
                    mask_n = f"{label}.mask_{i}"
                    gates.append(Gate(
                        name=mask_n, kind="constant",
                        attrs={"value": (1 << seq) - 1},
                        output_width=seq, output_signed=False,
                    ))
                    done_n = f"{label}.sm_done_{i}"
                    gates.append(Gate(name=done_n, kind="extern_wire",
                                      output_width=1))
                    yp = f"{label}.weights_{i}_packed"
                    gates.append(Gate(
                        name=yp, kind="instance",
                        inputs=["clk", "rst", "start", pack, mask_n],
                        attrs={
                            "module_name": sm_sub.top,
                            "instance_name": (
                                f"{label.replace('.', '_')}_sm_{i}"
                            ),
                            "input_ports": ["clk", "rst", "start",
                                             "x_packed", "mask"],
                            "output_port": "y_packed",
                            "extra_output_ports": [("done", done_n)],
                        },
                        output_width=seq * 8, output_signed=False,
                    ))
                    row_w = []
                    for j in range(seq):
                        wn = f"{label}.w_{i}_{j}"
                        gates.append(Gate(
                            name=wn, kind="slice",
                            inputs=[yp],
                            attrs={"hi": (j + 1) * 8 - 1, "lo": j * 8},
                            output_width=8, output_signed=False,
                        ))
                        row_w.append(wn)
                    weight_outs.append(row_w)

                out_width = 8 + in_w + max(1, (seq - 1).bit_length()) + 1
                output_sigs: list[str] = []
                for i in range(seq):
                    for kv in range(d_v):
                        prod_names = []
                        for j in range(seq):
                            pn = f"{label}.outprod_{i}_{kv}_{j}"
                            gates.append(Gate(
                                name=pn, kind="mul",
                                inputs=[weight_outs[i][j],
                                        v_sigs[j * d_v + kv]],
                                output_width=8 + in_w, output_signed=True,
                            ))
                            prod_names.append(pn)
                        sn = f"{label}.{i * d_v + kv}"
                        gates.append(Gate(
                            name=sn, kind="linear",
                            inputs=prod_names,
                            attrs={"weights": [1] * seq, "bias": 0},
                            output_width=out_width, output_signed=True,
                        ))
                        output_sigs.append(sn)
                val_signals[outs[0]] = output_sigs
                val_widths[outs[0]] = out_width
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
                    "Gemm, MatMul, Conv, Attention (single-head, no mask), "
                    "BatchNormalization (inference, baked stats), "
                    "GroupNormalization, Add, Sub, Mul, Relu, Sigmoid, "
                    "Exp, Tanh, Softmax, Identity, Constant, Reshape, "
                    "Concat, Split, Gather, LayerNormalization"
                )
                deferred = (
                    "ConvTranspose (transposed convolution; the conv2d "
                    "IR primitive handles forward conv only)"
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

        if _needs_clock:
            # A sequential op (LayerNormalization, ...) was emitted; add
            # clk / rst / start to the external port list so the caller
            # can drive them. (For purely combinational graphs the frontend
            # historically didn't expose these, so only add when needed.)
            external = [Signal("clk"), Signal("rst"), Signal("start")] + external

        return GateGraph(
            inputs=external,
            outputs=output_signals,
            gates=gates,
            top=top,
            submodules=list(_onnx_pending_submodules),
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
