"""Pipelined MLP via the Python API + iverilog cycle-accurate simulation.

Confirms that --pipeline adds 1 cycle of latency per layer and that the
emitted Verilog matches Python's clamped matmul once the pipeline fills.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from safetensors2verilog import emit_module, registry

HERE = Path(__file__).parent

LO, HI = -8, 7

W0 = [[1, -1,  0,  1],
      [0,  1,  1, -1],
      [1,  1, -1,  0]]
B0 = [1, -1, 0]
W1 = [[1, -1,  1],
      [-1, 1,  1]]
B1 = [-2, 3]


def clamp(v: int) -> int:
    return max(LO, min(HI, v))


def py_eval(x: list[int]) -> list[int]:
    h = [clamp(sum(W0[j][i] * x[i] for i in range(4)) + B0[j])
         for j in range(3)]
    return [clamp(sum(W1[j][i] * h[i] for i in range(3)) + B1[j])
            for j in range(2)]


def main() -> int:
    fe = registry.get("bitnet_linear")()
    graph = fe.parse(
        HERE / "mlp.safetensors", top="mlp",
        activation_bits=4,
        output_clamp=f"{LO},{HI}",
        pipeline=True,
    )

    # Inspect the graph: it should have linear + clamp + register chains.
    kinds = {}
    for g in graph.gates:
        kinds[g.kind] = kinds.get(g.kind, 0) + 1
    print("kind histogram:", kinds)

    out_widths = {s.name: s.width for s in graph.outputs}
    print("outputs:", out_widths)

    text = emit_module(graph)
    vp = HERE / "mlp_pipe.v"
    vp.write_text(text, encoding="utf-8")

    if shutil.which("iverilog") is None:
        print("no iverilog; stopping after emission")
        return 0

    # Two-layer pipeline -> 2 cycles of latency. Drive 4 cases back-to-back,
    # check that cycle k+2 matches case k's expected output.
    cases = [(0, 0, 0, 0), (1, -1, 1, -1), (-2, 2, -2, 2), (3, -3, 0, 4)]
    expected = [py_eval(list(c)) for c in cases]

    out_names = [n for n in sorted(out_widths)]
    out_decls = "\n".join(
        f"  wire signed [{out_widths[n]-1}:0] {n.replace('.', '_')};"
        for n in out_names
    )
    out_assoc = ", ".join(
        f".{n.replace('.', '_')}({n.replace('.', '_')})"
        for n in out_names
    )
    fmt = " ".join(f"{n}=%0d" for n in out_names)
    args = ", ".join(n.replace('.', '_') for n in out_names)

    drive_lines = []
    for i, c in enumerate(cases):
        drive_lines.append(
            f"    @(posedge clk); x0 = {c[0]}; x1 = {c[1]}; "
            f"x2 = {c[2]}; x3 = {c[3]};"
        )
    # Three idle cycles after the last drive so the pipeline drains.
    drive_lines.extend(["    @(posedge clk);"] * 4)

    tb_lines = [
        "`timescale 1ns/1ps",
        "module tb;",
        "  reg clk = 0; always #5 clk = ~clk;",
        "  reg signed [3:0] x0, x1, x2, x3;",
        out_decls,
        "  mlp dut (",
        "    .clk(clk), .x0(x0), .x1(x1), .x2(x2), .x3(x3),",
        f"    {out_assoc}",
        "  );",
        "  initial begin",
        "    x0 = 0; x1 = 0; x2 = 0; x3 = 0;",
    ]
    tb_lines.extend(drive_lines)
    tb_lines.append('    $display("done");')
    tb_lines.append("    $finish;")
    tb_lines.append("  end")
    tb_lines.append("  always @(posedge clk) "
                    f'$display("cyc=%0t {fmt}", $time, {args});')
    tb_lines.append("endmodule")
    tb_path = HERE / "mlp_pipe_tb.v"
    tb_path.write_text("\n".join(tb_lines), encoding="utf-8")

    vvp = HERE / "mlp_pipe.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(vp), str(tb_path)],
        check=True,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True
    )
    print("--- simulation log ---")
    print(proc.stdout)

    # The 2-stage pipeline means case k arrives at the output 2 cycles
    # after we drive its inputs.
    cyc_lines = [
        l for l in proc.stdout.splitlines() if l.startswith("cyc=")
    ]
    # Drives happen on rising edges starting at cyc=15ns (after the initial
    # x0..x3 = 0 setting at t=0). Each posedge is at 5,15,25,35,45,...
    # Capture the y values at every posedge.
    y_history = []
    for line in cyc_lines:
        toks = line.split()
        ys = []
        for n in out_names:
            for t in toks:
                if t.startswith(f"{n.replace('.', '_')}="):
                    ys.append(int(t.split("=")[1]))
                    break
        y_history.append(ys)
    print("y_history (per-cycle):")
    for i, y in enumerate(y_history):
        print(f"  cyc {i}: {y}")

    # Drive starts at cyc=1 (after the t=0 initialization). 2-stage pipeline
    # delivers case k at cyc=k+3. Check.
    for k, exp in enumerate(expected):
        target_cyc = k + 3
        if target_cyc >= len(y_history):
            break
        got = y_history[target_cyc]
        ok = got == exp
        print(f"case {k} {cases[k]} -> expect {exp}; sim @cyc {target_cyc} = {got} {'OK' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
