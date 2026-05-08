"""Python evaluator for GateGraph IR.

`evaluate_graph(graph, inputs)` walks a GateGraph in topological order
and produces the value of every signal, including the graph's outputs.
This is the cross-check companion to `emit_module`: code generators
emit Verilog, this evaluator emits the reference values that an
iverilog simulation should match.

It supports every built-in kind in the Verilog backend with the
exception of:

  register   needs a clock; this is a combinational evaluator. Use
             `step_graph` for sequential graphs (drives one cycle).
  rom        evaluated as a constant lookup at the supplied address.
  tristate   high-impedance is represented as the literal `None`.
  parameter  evaluates to its compile-time value.

Inputs are passed as a {signal_name: int} dictionary. Width is honored
via two's-complement masking on signed signals and unsigned masking
on unsigned. Signal width information comes from the graph itself.
"""

from __future__ import annotations

from .core import Gate, GateGraph


def _mask(value: int, width: int, signed: bool) -> int:
    """Truncate `value` to `width` bits with proper sign extension."""
    width = max(1, width)
    mask = (1 << width) - 1
    v = value & mask
    if signed and (v & (1 << (width - 1))):
        v -= 1 << width
    return v


def _build_signal_info(graph: GateGraph) -> tuple[
    dict[str, int], dict[str, bool], dict[str, Gate]
]:
    widths: dict[str, int] = {}
    signed: dict[str, bool] = {}
    gate_by_name: dict[str, Gate] = {}
    for s in graph.inputs:
        widths[s.name] = max(1, s.width)
        signed[s.name] = s.signed
    for s in graph.outputs:
        widths[s.name] = max(1, s.width)
        signed[s.name] = s.signed
    for g in graph.gates:
        widths[g.name] = max(1, g.output_width)
        signed[g.name] = g.output_signed
        gate_by_name[g.name] = g
    return widths, signed, gate_by_name


def evaluate_graph(
    graph: GateGraph,
    inputs: dict[str, int],
    *,
    register_state: dict[str, int] | None = None,
    ram_state: dict[str, list[int]] | None = None,
) -> dict[str, int | None]:
    """Evaluate every signal in `graph` given external `inputs`.

    Returns a dict mapping every signal name (external inputs, internal
    gate outputs, and outputs) to its integer value. Signed signals
    return signed integers; unsigned return non-negative.

    For graphs that contain `register` gates, supply `register_state`
    as a {register_name: previous_value} dict. Registers' outputs come
    from the state; their D inputs are not evaluated by this function
    (use `step_graph` to advance state).

    For graphs that contain `ram_writable` gates, supply ``ram_state`` as
    a ``{ram_name: list[int]}`` dict; reads return ``ram_state[name][read_addr]``
    when present, falling back to the gate's ``attrs.init``. Use
    ``step_graph`` to advance writes per cycle.
    """
    widths, signed, gate_by_name = _build_signal_info(graph)
    register_state = register_state or {}
    ram_state = ram_state or {}

    # ``int | None`` because tristate gates return None for high-Z; the
    # value is otherwise an integer.
    values: dict[str, int | None] = {"#0": 0, "#1": 1}
    for s in graph.inputs:
        if s.name not in inputs and s.is_parameter:
            values[s.name] = s.parameter_value
        elif s.name in inputs:
            values[s.name] = _mask(inputs[s.name], widths[s.name], signed[s.name])
        elif s.name in ("clk", "rst"):
            # Clock and reset don't have meaningful values for combinational eval
            values[s.name] = 0
        else:
            raise KeyError(
                f"missing input '{s.name}' (graph requires "
                f"{[i.name for i in graph.inputs]})"
            )

    # Register outputs come from state (or 0 by default).
    for g in graph.gates:
        if g.kind == "register":
            values[g.name] = _mask(
                register_state.get(g.name, int(g.attrs.get("init", 0))),
                widths[g.name], signed[g.name],
            )

    for g in graph.gates:
        if g.kind == "register":
            continue
        try:
            if g.kind == "ram_writable" and g.name in ram_state:
                # Read from the live state, not the init contents.
                inps_for_ram = [values[s] for s in g.inputs]
                read_addr = inps_for_ram[0]
                if read_addr is None:
                    v = None
                else:
                    bank = ram_state[g.name]
                    depth = int(g.attrs["depth"])
                    if 0 <= int(read_addr) < depth:
                        v = bank[int(read_addr)] if int(read_addr) < len(bank) else 0
                    else:
                        v = 0
            else:
                v = _eval_gate(g, values, widths, signed)
        except KeyError as e:
            raise ValueError(
                f"gate '{g.name}' references undefined signal {e}"
            ) from e
        # ``None`` represents high-impedance (Z) from a tristate gate;
        # propagate as-is so downstream sinks see the disconnect.
        if v is None:
            values[g.name] = None
        else:
            values[g.name] = _mask(v, widths[g.name], signed[g.name])

    return values


def step_graph(
    graph: GateGraph,
    inputs: dict[str, int],
    register_state: dict[str, int],
    ram_state: dict[str, list[int]] | None = None,
) -> dict[str, int] | tuple[dict[str, int], dict[str, list[int]]]:
    """One simulation cycle: returns the next register state.

    Combines `evaluate_graph` with sampling each register's D input.
    The returned dict can be fed back as `register_state` for the next
    cycle.

    For graphs with ``ram_writable`` gates, supply ``ram_state`` and the
    return becomes ``(next_register_state, next_ram_state)`` so the
    caller can feed both back next cycle. When ``ram_state`` is None the
    legacy single-dict return is preserved.
    """
    ram_state_was_none = ram_state is None
    ram_state = ram_state or {}
    values = evaluate_graph(
        graph, inputs, register_state=register_state, ram_state=ram_state,
    )
    next_state: dict[str, int] = {}
    next_ram: dict[str, list[int]] = {
        name: list(bank) for name, bank in ram_state.items()
    }
    widths, signed, _ = _build_signal_info(graph)
    # Apply RAM writes for this cycle.
    for g in graph.gates:
        if g.kind != "ram_writable":
            continue
        if len(g.inputs) != 4:
            continue
        write_addr_v = values.get(g.inputs[1])
        write_data_v = values.get(g.inputs[2])
        write_en_v = values.get(g.inputs[3])
        if write_en_v is None or not int(write_en_v):
            # Initialise the bank from init contents on first use.
            if g.name not in next_ram:
                init = list(g.attrs.get("init", []))
                depth = int(g.attrs["depth"])
                bank = init + [0] * max(0, depth - len(init))
                next_ram[g.name] = bank[:depth]
            continue
        if g.name not in next_ram:
            init = list(g.attrs.get("init", []))
            depth = int(g.attrs["depth"])
            bank = init + [0] * max(0, depth - len(init))
            next_ram[g.name] = bank[:depth]
        if write_addr_v is None or write_data_v is None:
            continue
        depth = int(g.attrs["depth"])
        addr = int(write_addr_v)
        if 0 <= addr < depth:
            width = int(g.attrs["width"])
            mask = (1 << width) - 1
            next_ram[g.name][addr] = int(write_data_v) & mask
    for g in graph.gates:
        if g.kind != "register":
            continue
        d = g.inputs[0]
        # If a reset is high, the next state is the init value
        rst = g.attrs.get("rst")
        if rst is not None and values.get(rst, 0):
            next_state[g.name] = int(g.attrs.get("init", 0))
        else:
            d_val = values[d]
            if d_val is None:
                # D input is high-impedance: hold the previous state.
                next_state[g.name] = register_state.get(
                    g.name, int(g.attrs.get("init", 0))
                )
            else:
                next_state[g.name] = _mask(
                    d_val, widths[g.name], signed[g.name]
                )
    if ram_state_was_none:
        return next_state
    return next_state, next_ram


def _eval_gate(
    g: Gate,
    values: dict[str, int | None],
    widths: dict[str, int],
    signed: dict[str, bool],
) -> int | None:
    kind = g.kind
    inps_raw: list[int | None] = [values[s] for s in g.inputs]

    # Tristate handles None on its data input as Z passthrough; everything
    # else propagates Z (None) on any input to a None output.
    if kind == "tristate":
        if len(inps_raw) != 2:
            raise ValueError(
                f"gate '{g.name}' kind 'tristate' expects [data, en]"
            )
        ts_data = inps_raw[0]
        ts_en = inps_raw[1]
        if ts_en is None:
            return None
        enable_high = bool(g.attrs.get("enable_high", True))
        en_active = (ts_en != 0) if enable_high else (ts_en == 0)
        return ts_data if en_active else None

    if any(v is None for v in inps_raw):
        return None
    inps: list[int] = [int(v) for v in inps_raw if v is not None]

    if kind == "threshold":
        weights = list(g.attrs.get("weights", []))
        bias = int(g.attrs.get("bias", 0))
        s = sum(w * x for w, x in zip(weights, inps)) + bias
        return 1 if s >= 0 else 0

    if kind == "linear":
        weights = list(g.attrs.get("weights", []))
        bias = int(g.attrs.get("bias", 0))
        return sum(w * x for w, x in zip(weights, inps)) + bias

    if kind == "constant":
        return int(g.attrs["value"])

    if kind == "parameter":
        return int(g.attrs["value"])

    if kind == "add":
        return inps[0] + inps[1]
    if kind == "sub":
        return inps[0] - inps[1]
    if kind == "mul":
        return inps[0] * inps[1]

    if kind == "and":
        result = inps[0]
        for x in inps[1:]:
            result &= x
        return result
    if kind == "or":
        result = inps[0]
        for x in inps[1:]:
            result |= x
        return result
    if kind == "xor":
        result = inps[0]
        for x in inps[1:]:
            result ^= x
        return result
    if kind == "not":
        # Bitwise not, masked to width
        return ~inps[0]

    if kind == "shift_left":
        return inps[0] << int(g.attrs.get("amount", 1))
    if kind == "shift_right":
        return inps[0] >> int(g.attrs.get("amount", 1))

    if kind == "concat":
        # MSB first
        result = 0
        for s, v in zip(g.inputs, inps):
            w = widths.get(s, 1)
            result = (result << w) | (v & ((1 << w) - 1))
        return result

    if kind == "slice":
        hi = int(g.attrs["hi"])
        lo = int(g.attrs["lo"])
        return (inps[0] >> lo) & ((1 << (hi - lo + 1)) - 1)

    if kind == "mux":
        sel = inps[0]
        mux_data = inps[1:]
        if 0 <= sel < len(mux_data):
            return mux_data[sel]
        return mux_data[-1]

    if kind == "eq":
        return 1 if inps[0] == inps[1] else 0

    if kind == "relu":
        return max(0, inps[0]) if signed.get(g.inputs[0], False) else inps[0]

    if kind == "clamp":
        lo = int(g.attrs["lo"])
        hi = int(g.attrs["hi"])
        return max(lo, min(hi, inps[0]))

    if kind == "rom":
        addr = inps[0]
        init = list(g.attrs["init"])
        depth = int(g.attrs["depth"])
        if 0 <= addr < depth:
            return init[addr] if addr < len(init) else 0
        return 0

    if kind == "conv2d":
        a = g.attrs
        in_h = int(a["in_h"]); in_w = int(a["in_w"]); in_c = int(a["in_c"])
        out_h = int(a["out_h"]); out_w = int(a["out_w"]); out_c = int(a["out_c"])
        kH = int(a["kH"]); kW = int(a["kW"])
        stride_h = int(a.get("stride_h", 1))
        stride_w = int(a.get("stride_w", 1))
        pad_h = int(a.get("pad_h", 0))
        pad_w = int(a.get("pad_w", 0))
        act_bits = int(a["act_bits"])
        out_bits = int(a["out_bits"])
        weights = a["weights"]
        biases = a.get("biases") or [0] * out_c
        x_packed = inps[0] & ((1 << (in_h * in_w * in_c * act_bits)) - 1)

        def x_at(ih: int, iw: int, ic: int) -> int:
            if not (0 <= ih < in_h and 0 <= iw < in_w):
                return 0
            in_idx = (ih * in_w + iw) * in_c + ic
            lo = in_idx * act_bits
            v = (x_packed >> lo) & ((1 << act_bits) - 1)
            if v & (1 << (act_bits - 1)):
                v -= (1 << act_bits)
            return v

        out_packed = 0
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(out_c):
                    s = int(biases[oc])
                    for ic in range(in_c):
                        for ki in range(kH):
                            ih_ = oh * stride_h - pad_h + ki
                            for kj in range(kW):
                                iw_ = ow * stride_w - pad_w + kj
                                s += int(weights[oc][ic][ki][kj]) * x_at(ih_, iw_, ic)
                    out_idx = (oh * out_w + ow) * out_c + oc
                    mask = (1 << out_bits) - 1
                    out_packed |= (s & mask) << (out_idx * out_bits)
        return out_packed

    if kind == "conv_transpose2d":
        a = g.attrs
        in_h = int(a["in_h"]); in_w = int(a["in_w"]); in_c = int(a["in_c"])
        out_h = int(a["out_h"]); out_w = int(a["out_w"]); out_c = int(a["out_c"])
        kH = int(a["kH"]); kW = int(a["kW"])
        stride_h = int(a.get("stride_h", 1))
        stride_w = int(a.get("stride_w", 1))
        pad_h = int(a.get("pad_h", 0))
        pad_w = int(a.get("pad_w", 0))
        act_bits = int(a["act_bits"])
        out_bits = int(a["out_bits"])
        weights = a["weights"]
        biases = a.get("biases") or [0] * out_c
        x_packed = inps[0] & ((1 << (in_h * in_w * in_c * act_bits)) - 1)

        def _ct_x_at(ih: int, iw: int, ic: int) -> int:
            in_idx = (ih * in_w + iw) * in_c + ic
            lo = in_idx * act_bits
            v = (x_packed >> lo) & ((1 << act_bits) - 1)
            if v & (1 << (act_bits - 1)):
                v -= (1 << act_bits)
            return v

        out_acc = [[[0 for _ in range(out_c)] for _ in range(out_w)]
                   for _ in range(out_h)]
        for ih in range(in_h):
            for iw in range(in_w):
                for ic in range(in_c):
                    xv = _ct_x_at(ih, iw, ic)
                    for ki in range(kH):
                        oh = ih * stride_h - pad_h + ki
                        if not 0 <= oh < out_h:
                            continue
                        for kj in range(kW):
                            ow = iw * stride_w - pad_w + kj
                            if not 0 <= ow < out_w:
                                continue
                            for oc in range(out_c):
                                out_acc[oh][ow][oc] += int(weights[ic][oc][ki][kj]) * xv
        out_packed_ct = 0
        mask = (1 << out_bits) - 1
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(out_c):
                    s = out_acc[oh][ow][oc] + int(biases[oc])
                    out_idx = (oh * out_w + ow) * out_c + oc
                    out_packed_ct |= (s & mask) << (out_idx * out_bits)
        return out_packed_ct

    if kind == "ram_writable":
        # Combinational eval: read from the init contents only. Writes
        # require step_graph + register state, which would need a separate
        # state-bag for RAM contents (deferred); the read path is enough
        # for the equivalence harness's combinational sweeps.
        read_addr = inps[0]
        init = list(g.attrs.get("init", []))
        depth = int(g.attrs["depth"])
        if 0 <= read_addr < depth:
            return init[read_addr] if read_addr < len(init) else 0
        return 0

    # ``tristate`` is handled at the top of this function so its data
    # input can pass through None (Z) without being filtered.

    raise NotImplementedError(
        f"evaluate_graph does not implement kind '{kind}' "
        f"(gate '{g.name}'). Add a case in evaluate.py:_eval_gate."
    )


__all__ = ["evaluate_graph", "step_graph"]
