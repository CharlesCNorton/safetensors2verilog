"""Graph-level transforms over a `GateGraph`.

Currently provides:

  pipeline_at_depths(graph, cut_depths, clk='clk')
      Insert a 1-bit register after every gate at the given combinational
      depth(s); rewire consumers to read from the register instead. Each
      cut adds 1 cycle of latency in exchange for a shorter critical
      path. Multiple cuts compose.

  pipeline_every(graph, period, clk='clk')
      Convenience: pipeline at depths period, 2*period, 3*period... up to
      the deepest gate.

The transform is structural and only meaningful for combinational
threshold networks — designs that already contain `register` gates are
left untouched at those points (registers reset combinational depth).

Returns a new `GateGraph` rather than mutating in place. The new graph
adds `clk` to its inputs if not already present.
"""

from __future__ import annotations

from .analysis import gate_depths
from .core import Gate, GateGraph, Signal


def _ensure_clk(inputs: list[Signal], clk: str) -> list[Signal]:
    if any(s.name == clk for s in inputs):
        return inputs
    return list(inputs) + [Signal(name=clk, width=1)]


def pipeline_at_depths(
    graph: GateGraph, cut_depths: list[int], clk: str = "clk",
) -> GateGraph:
    """Insert pipeline registers immediately after every combinational gate
    whose depth equals one of ``cut_depths``.

    Each register samples its input on the rising edge of ``clk`` and
    presents the prior cycle's value to downstream consumers. The user
    must drive ``clk`` from outside; ``clk`` is added to the input port
    list when not already present.
    """
    if not cut_depths:
        return graph

    cut_set = set(int(d) for d in cut_depths)
    depths = gate_depths(graph)

    # Map of original-gate-name -> register-gate-name that downstream
    # consumers should read instead.
    rename: dict[str, str] = {}
    new_gates: list[Gate] = []

    for g in graph.gates:
        # Insert the gate first (its inputs may need rewriting if upstream
        # producers were already pipelined).
        rewritten_inputs = [rename.get(s, s) for s in g.inputs]
        new_gates.append(Gate(
            name=g.name, kind=g.kind,
            inputs=rewritten_inputs, attrs=dict(g.attrs),
            output_width=g.output_width, output_signed=g.output_signed,
        ))

        if g.kind == "register":
            # Already a register; cut_depth doesn't apply.
            continue
        d = depths.get(g.name, 0)
        if d in cut_set:
            reg_name = f"{g.name}__pipe{d}"
            new_gates.append(Gate(
                name=reg_name, kind="register",
                inputs=[g.name],
                attrs={"clk": clk},
                output_width=g.output_width,
                output_signed=g.output_signed,
            ))
            rename[g.name] = reg_name

    # Outputs that named a now-pipelined gate must point at the register.
    new_outputs = [
        Signal(name=rename.get(s.name, s.name),
               width=s.width, signed=s.signed,
               direction=s.direction)
        for s in graph.outputs
    ]

    return GateGraph(
        inputs=_ensure_clk(graph.inputs, clk),
        outputs=new_outputs,
        gates=new_gates,
        top=graph.top,
    )


def pipeline_every(
    graph: GateGraph, period: int, clk: str = "clk",
) -> GateGraph:
    """Pipeline at depth ``period``, ``2*period``, ... up to max depth."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    depths = gate_depths(graph)
    if not depths:
        return graph
    max_d = max(depths.values())
    cuts = list(range(period, max_d + 1, period))
    return pipeline_at_depths(graph, cuts, clk=clk)
