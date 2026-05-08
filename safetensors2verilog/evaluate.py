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
) -> dict[str, int]:
    """Evaluate every signal in `graph` given external `inputs`.

    Returns a dict mapping every signal name (external inputs, internal
    gate outputs, and outputs) to its integer value. Signed signals
    return signed integers; unsigned return non-negative.

    For graphs that contain `register` gates, supply `register_state`
    as a {register_name: previous_value} dict. Registers' outputs come
    from the state; their D inputs are not evaluated by this function
    (use `step_graph` to advance state).
    """
    widths, signed, gate_by_name = _build_signal_info(graph)
    register_state = register_state or {}

    values: dict[str, int] = {"#0": 0, "#1": 1}
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
) -> dict[str, int]:
    """One simulation cycle: returns the next register state.

    Combines `evaluate_graph` with sampling each register's D input.
    The returned dict can be fed back as `register_state` for the next
    cycle.
    """
    values = evaluate_graph(graph, inputs, register_state=register_state)
    next_state: dict[str, int] = {}
    widths, signed, _ = _build_signal_info(graph)
    for g in graph.gates:
        if g.kind != "register":
            continue
        d = g.inputs[0]
        # If a reset is high, the next state is the init value
        rst = g.attrs.get("rst")
        if rst is not None and values.get(rst, 0):
            next_state[g.name] = int(g.attrs.get("init", 0))
        else:
            next_state[g.name] = _mask(values[d], widths[g.name], signed[g.name])
    return next_state


def _eval_gate(
    g: Gate,
    values: dict[str, int],
    widths: dict[str, int],
    signed: dict[str, bool],
) -> int:
    kind = g.kind
    inps = [values[s] for s in g.inputs]

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
        data = inps[1:]
        if 0 <= sel < len(data):
            return data[sel]
        return data[-1]

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

    if kind == "tristate":
        # When the enable signal is asserted, drive the data through.
        # Otherwise return ``None`` to mark the bus as high-impedance;
        # downstream consumers must treat ``None`` as "don't read" or
        # propagate it. (The previous behaviour returned 0 unconditionally,
        # which silently hid Z propagation bugs in cross-checks.)
        enable_high = bool(g.attrs.get("enable_high", True))
        en_active = (inps[1] != 0) if enable_high else (inps[1] == 0)
        return inps[0] if en_active else None

    raise NotImplementedError(
        f"evaluate_graph does not implement kind '{kind}' "
        f"(gate '{g.name}'). Add a case in evaluate.py:_eval_gate."
    )


__all__ = ["evaluate_graph", "step_graph"]
