"""Formal-equivalence harness for an emitted threshold-logic Verilog module.

Given a `GateGraph` (the IR the threshold_logic frontend produces) and the
emitted Verilog file, this module:

  1. Builds a Python evaluator from the GateGraph (combinational threshold
     gates only; fast enough for exhaustive checking on small circuits).
  2. Generates a self-checking Verilog testbench that drives every input
     combination (or a sample if exhaustive is too large) and compares the
     simulator's outputs against expected values computed in Python.
  3. Optionally writes a SymbiYosys ``equiv_check`` script template that
     compares the emitted Verilog against an ABC-synthesised reference.

For circuits with few enough single-bit external inputs (n <= 16 by
default) the harness drives all 2^n combinations. Larger circuits get
random sampling.

Public entry points:

  evaluate_python(graph, assignments) -> {output: 0|1}
      Compute outputs from a {input_name: 0|1} dict.

  emit_self_checking_tb(graph, dut_module, dut_path, max_exhaustive=16,
                        sample_size=1024, seed=0) -> str
      Verilog testbench source.

  emit_sby_equiv(reference_v, target_v, top, period_ns=10.0) -> str
      SymbiYosys ``equiv_check`` script.

The harness is intentionally restricted to combinational threshold gates;
sequential designs need a real driver.
"""

from __future__ import annotations

import random
import re

from .core import GateGraph


def _verilog_id(name: str) -> str:
    """Mirror the backend's identifier sanitization."""
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def evaluate_python(
    graph: GateGraph, assignments: dict[str, int]
) -> dict[str, int]:
    """Evaluate a combinational threshold-logic GateGraph in Python.

    Only ``threshold`` gates and the special ``#0`` / ``#1`` constants
    are supported; raises if the graph contains unsupported kinds.
    """
    values: dict[str, int] = {"#0": 0, "#1": 1}
    for s in graph.inputs:
        if s.name not in assignments:
            raise KeyError(f"input '{s.name}' not in assignments")
        values[s.name] = int(assignments[s.name]) & 1

    for g in graph.gates:
        if g.kind != "threshold":
            raise ValueError(
                f"evaluate_python supports threshold gates only; "
                f"gate '{g.name}' has kind '{g.kind}'"
            )
        weights = g.attrs.get("weights", [])
        bias = int(g.attrs.get("bias", 0))
        if len(weights) != len(g.inputs):
            raise ValueError(
                f"gate '{g.name}': weight/input length mismatch"
            )
        total = 0
        for w, src in zip(weights, g.inputs):
            v = values.get(src)
            if v is None:
                raise KeyError(
                    f"gate '{g.name}' references unknown signal '{src}'"
                )
            total += int(w) * v
        total += bias
        values[g.name] = 1 if total >= 0 else 0
    return values


def _generate_assignments(
    graph: GateGraph, max_exhaustive: int, sample_size: int, seed: int
) -> list[dict[str, int]]:
    inputs = [s.name for s in graph.inputs]
    n = len(inputs)
    if n <= max_exhaustive:
        out: list[dict[str, int]] = []
        for i in range(1 << n):
            out.append({inputs[k]: (i >> k) & 1 for k in range(n)})
        return out
    rng = random.Random(seed)
    out = []
    for _ in range(sample_size):
        out.append({nm: rng.randint(0, 1) for nm in inputs})
    return out


def emit_self_checking_tb(
    graph: GateGraph,
    dut_module: str,
    *,
    max_exhaustive: int = 16,
    sample_size: int = 1024,
    seed: int = 0,
    timescale: str = "1ns/1ps",
    pack_buses: bool = False,
) -> str:
    """Build a Verilog testbench that drives every input combination
    (or a sample) and compares against Python-evaluated truth.

    The testbench:
      - declares 1-bit reg for each input and 1-bit wire for each output,
      - instantiates ``dut_module`` with named-port association,
      - sets each input combination, waits #1, and checks every output
        against a baked-in expected value table.

    Returns the testbench Verilog text (a single module named ``tb``).
    The caller compiles it together with the original Verilog file via
    iverilog and runs it through vvp.
    """
    inputs = [s.name for s in graph.inputs]
    outputs = [s.name for s in graph.outputs]

    cases = _generate_assignments(
        graph, max_exhaustive, sample_size, seed
    )
    expected = []
    for asg in cases:
        out_vals = evaluate_python(graph, asg)
        expected.append({o: out_vals[o] for o in outputs})

    # Mirror the backend's bus-packing logic so the testbench's port
    # association matches what the emitted module actually exposes.
    bus_for: dict[str, tuple[str, int]] = {}   # input name -> (bus_id, idx)
    bus_widths: dict[str, int] = {}            # bus_id -> width
    if pack_buses:
        from .verilog import _detect_buses, _sanitize
        buses, _members = _detect_buses(graph.inputs)
        used: set[str] = set()
        for base, items in buses.items():
            bus_id = _sanitize(
                base[1:] if base.startswith("$") else base, used
            )
            bus_widths[bus_id] = len(items)
            for idx, original in items:
                bus_for[original] = (bus_id, idx)

    in_ids = [_verilog_id(n) for n in inputs]
    out_ids = [_verilog_id(n) for n in outputs]

    lines: list[str] = []
    lines.append(f"`timescale {timescale}")
    lines.append("module tb;")

    # Per-bit storage for inputs (used by Python truth table). For
    # bus-packed inputs we additionally declare a packed reg and assign
    # each bit-select from the per-bit reg.
    for nm in in_ids:
        lines.append(f"  reg  {nm};")
    for nm in out_ids:
        lines.append(f"  wire {nm};")
    for bus_id, width in bus_widths.items():
        lines.append(f"  wire [{width-1}:0] {bus_id};")
    for original_in, (bus_id, idx) in bus_for.items():
        lines.append(
            f"  assign {bus_id}[{idx}] = {_verilog_id(original_in)};"
        )

    # Build port association: outputs always connect by their flat names,
    # inputs that participate in a bus connect by the packed bus name.
    assocs: list[str] = []
    seen_buses: set[str] = set()
    for inp in inputs:
        if inp in bus_for:
            bus_id, _ = bus_for[inp]
            if bus_id in seen_buses:
                continue
            assocs.append(f".{bus_id}({bus_id})")
            seen_buses.add(bus_id)
        else:
            vid = _verilog_id(inp)
            assocs.append(f".{vid}({vid})")
    for o in outputs:
        vid = _verilog_id(o)
        assocs.append(f".{vid}({vid})")

    lines.append(f"  {dut_module} dut({', '.join(assocs)});")
    lines.append("  integer fails;")
    lines.append("  initial begin")
    lines.append("    fails = 0;")

    for asg, exp in zip(cases, expected):
        bind_lines = "; ".join(
            f"{_verilog_id(n)} = 1'b{int(asg[n])}" for n in inputs
        )
        lines.append(f"    {bind_lines};")
        lines.append("    #1;")
        for n in outputs:
            vid = _verilog_id(n)
            ev = int(exp[n])
            lines.append(
                f"    if ({vid} !== 1'b{ev}) begin fails = fails + 1; "
                f'$display("FAIL {n} got=%b exp=%b", {vid}, 1\'b{ev}); end'
            )

    lines.append('    if (fails == 0) $display("PASS %0d cases", '
                 + str(len(cases)) + ");")
    lines.append('    else $display("FAILS %0d / %0d", fails, '
                 + str(len(cases)) + ");")
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def emit_sby_equiv(
    reference_v: str,
    target_v: str,
    top: str,
    *,
    period_ns: float = 10.0,
    depth: int = 20,
) -> str:
    """Emit a SymbiYosys ``equiv`` task script template.

    The user runs ``sby equiv.sby`` to invoke Yosys's ``equiv_check``.
    Reference and target Verilog files are both compiled, then ``equiv``
    proves output equivalence over ``depth`` clock cycles.

    Combinational designs (no register gates) work without a clock; the
    template still emits a clock since equiv_check tolerates an unused
    one and many tools assume its presence.
    """
    return (
        f"[options]\n"
        f"mode prove\n"
        f"depth {depth}\n"
        f"\n"
        f"[engines]\n"
        f"smtbmc\n"
        f"\n"
        f"[script]\n"
        f"read_verilog -formal {reference_v}\n"
        f"prep -top {top}\n"
        f"design -stash gold\n"
        f"\n"
        f"read_verilog -formal {target_v}\n"
        f"prep -top {top}\n"
        f"design -stash gate\n"
        f"\n"
        f"design -copy-from gold -as gold {top}\n"
        f"design -copy-from gate -as gate {top}\n"
        f"equiv_make gold gate equiv\n"
        f"prep -top equiv\n"
        f"\n"
        f"[files]\n"
        f"{reference_v}\n"
        f"{target_v}\n"
    )


def run_iverilog_check(
    graph: GateGraph,
    dut_verilog: str,
    dut_module: str,
    *,
    max_exhaustive: int = 16,
    sample_size: int = 1024,
    seed: int = 0,
    iverilog_path: str = "iverilog",
    vvp_path: str = "vvp",
    timeout: float = 600.0,
    pack_buses: bool = False,
) -> dict:
    """End-to-end equivalence run: emit testbench, compile with iverilog,
    run with vvp, return result.

    Returns a dict with keys 'passed' (bool), 'cases', 'fails', and
    'output' (the raw simulator log).
    """
    import subprocess
    import tempfile
    from pathlib import Path

    tb_src = emit_self_checking_tb(
        graph, dut_module=dut_module,
        max_exhaustive=max_exhaustive, sample_size=sample_size, seed=seed,
        pack_buses=pack_buses,
    )
    with tempfile.TemporaryDirectory(prefix="s2v_equiv_") as tmpdir:
        td = Path(tmpdir)
        (td / "dut.v").write_text(dut_verilog, encoding="utf-8")
        (td / "tb.v").write_text(tb_src, encoding="utf-8")
        compiled = td / "test.vvp"
        proc = subprocess.run(
            [iverilog_path, "-o", str(compiled), str(td / "dut.v"),
             str(td / "tb.v")],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"iverilog failed:\n{proc.stdout}\n{proc.stderr}"
            )
        run = subprocess.run(
            [vvp_path, str(compiled)],
            capture_output=True, text=True, timeout=timeout,
        )
        log = run.stdout + run.stderr

    passed = "PASS" in log and "FAIL" not in log
    fails = 0
    cases = 0
    m = re.search(r"PASS (\d+) cases", log)
    if m:
        cases = int(m.group(1))
    m2 = re.search(r"FAILS (\d+) / (\d+)", log)
    if m2:
        fails = int(m2.group(1))
        cases = int(m2.group(2))
    return {
        "passed": passed,
        "cases": cases,
        "fails": fails,
        "output": log,
    }
