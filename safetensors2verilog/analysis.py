"""Static analysis over a GateGraph.

Currently provides:

  gate_depths(graph)       -> {gate_name: int}     longest combinational
                                                   path from any external
                                                   input to that gate.
  critical_path(graph)     -> list[str]            one path realising the
                                                   maximum depth.
  fanout(graph)            -> {signal_name: int}   how many gates consume
                                                   each produced signal.
  summary(graph)           -> dict                 high-level report
                                                   suitable for printing.

The depth measure ignores ``register`` gates: they break combinational
chains and always reset their consumer's depth to 1. This matches what
synthesis-time critical-path analysis cares about — "how many levels of
combinational logic between flip-flops or external inputs."
"""

from __future__ import annotations

from collections import defaultdict, deque

from .core import GateGraph


def gate_depths(graph: GateGraph) -> dict[str, int]:
    """Longest path (in gates) from any external input to each gate.

    External inputs and constants ('#0', '#1') have implicit depth 0.
    Register outputs reset to depth 1: their D input belongs to the
    previous clock cycle, so the combinational chain restarts.
    """
    depths: dict[str, int] = {}
    # Gates list is already topologically sorted (the backend enforces this).
    for g in graph.gates:
        if g.kind == "register":
            depths[g.name] = 1
            continue
        in_depths = []
        for src in g.inputs:
            if src in ("#0", "#1"):
                in_depths.append(0)
            elif src in depths:
                in_depths.append(depths[src])
            else:
                in_depths.append(0)
        depths[g.name] = max(in_depths, default=0) + 1
    return depths


def fanout(graph: GateGraph) -> dict[str, int]:
    """How many gates consume each produced signal."""
    counts: dict[str, int] = defaultdict(int)
    for g in graph.gates:
        for src in g.inputs:
            counts[src] += 1
    return dict(counts)


def critical_path(graph: GateGraph) -> list[str]:
    """One path through the graph achieving the maximum depth.

    Walks back from the deepest gate, picking at each step the input
    that contributed the depth. Returns the sequence of gate names from
    external-input side to deepest gate (last element is the deepest).
    """
    depths = gate_depths(graph)
    if not depths:
        return []
    end = max(depths, key=lambda n: depths[n])
    by_name = {g.name: g for g in graph.gates}
    path: deque[str] = deque([end])
    cur = end
    while True:
        g = by_name.get(cur)
        if g is None or g.kind == "register":
            break
        # find the input that achieved depth[cur]-1
        target = depths[cur] - 1
        prev = None
        for src in g.inputs:
            if src in by_name and depths.get(src, 0) == target:
                prev = src
                break
        if prev is None:
            break
        path.appendleft(prev)
        cur = prev
    return list(path)


def summary(graph: GateGraph) -> dict:
    """High-level report on the graph.

    Returns a dict with:
      gates            total gate count
      kinds            {kind: count}
      max_depth        deepest combinational path
      critical_path    list of gate names along that path
      fanout_max       (gate, count) for the highest-fanout signal
      inputs           number of external inputs
      outputs          number of external outputs
    """
    kinds: dict[str, int] = defaultdict(int)
    for g in graph.gates:
        kinds[g.kind] += 1
    depths = gate_depths(graph)
    fo = fanout(graph)
    fanout_max = max(fo.items(), key=lambda kv: kv[1]) if fo else (None, 0)
    return {
        "gates": len(graph.gates),
        "kinds": dict(kinds),
        "max_depth": max(depths.values(), default=0),
        "critical_path": critical_path(graph),
        "fanout_max": {"signal": fanout_max[0], "fanout": fanout_max[1]},
        "inputs": len(graph.inputs),
        "outputs": len(graph.outputs),
    }


def format_summary(graph: GateGraph) -> str:
    """Human-readable text rendering of summary(graph)."""
    s = summary(graph)
    lines: list[str] = []
    lines.append(f"top         : {graph.top}")
    lines.append(f"gates       : {s['gates']}")
    lines.append(f"inputs      : {s['inputs']}")
    lines.append(f"outputs     : {s['outputs']}")
    lines.append(f"max depth   : {s['max_depth']} (combinational levels)")
    if s["fanout_max"]["signal"]:
        lines.append(
            f"max fanout  : {s['fanout_max']['fanout']} "
            f"({s['fanout_max']['signal']})"
        )
    if s["kinds"]:
        kinds_str = ", ".join(
            f"{k}={v}" for k, v in sorted(s["kinds"].items(), key=lambda kv: -kv[1])
        )
        lines.append(f"by kind     : {kinds_str}")
    cp = s["critical_path"]
    if cp:
        lines.append(f"critical    : {' -> '.join(cp[:8])}"
                     + (f" ... ({len(cp)} hops)" if len(cp) > 8 else ""))
    return "\n".join(lines)
