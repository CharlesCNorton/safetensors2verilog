"""Sequential bitnet: FSM, ROMs, start/done handshake. Drive a few cases
through iverilog and check answers come out per the Python ref.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from safetensors2verilog import emit_module, registry

HERE = Path(__file__).parent

W0 = [[1, -1,  0,  1],
      [0,  1,  1, -1],
      [1,  1, -1,  0]]
B0 = [1, -1, 0]
W1 = [[1, -1,  1],
      [-1, 1,  1]]
B1 = [-2, 3]


def py_eval(x):
    h = [sum(W0[j][i] * x[i] for i in range(4)) + B0[j] for j in range(3)]
    return [sum(W1[j][i] * h[i] for i in range(3)) + B1[j] for j in range(2)]


def main() -> int:
    fe = registry.get("bitnet_linear")()
    graph = fe.parse(
        HERE / "mlp.safetensors", top="seq_mlp",
        activation_bits=4, sequential=True,
    )
    by_kind = {}
    for g in graph.gates:
        by_kind[g.kind] = by_kind.get(g.kind, 0) + 1
    print("seq IR kind histogram:", by_kind)
    print(f"total gates: {len(graph.gates)}, "
          f"inputs: {len(graph.inputs)}, outputs: {len(graph.outputs)}")
    print("input ports:", [s.name for s in graph.inputs])
    print("output ports + widths:",
          [(s.name, s.width) for s in graph.outputs])

    text = emit_module(graph)
    vp = HERE / "seq_mlp.v"
    vp.write_text(text, encoding="utf-8")
    print(f"verilog: {len(text.splitlines())} lines, {len(text)} chars")

    if shutil.which("iverilog") is None:
        return 0

    cases = [
        (0, 0, 0, 0),
        (1, -1, 1, -1),
        (-2, 2, -2, 2),
        (3, -3, 0, 4),
        (-4, 4, -4, 4),
    ]
    out_widths = {s.name: s.width for s in graph.outputs if s.name != "done"}
    out_names = sorted(out_widths)
    out_decls = "\n".join(
        f"  wire signed [{out_widths[n]-1}:0] {n};" for n in out_names
    )
    out_assoc = ", ".join(f".{n}({n})" for n in out_names)

    case_calls = "\n".join(
        f"    drive({c[0]}, {c[1]}, {c[2]}, {c[3]});" for c in cases
    )

    tb = f"""\
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg rst = 1, start = 0;
  reg signed [3:0] x0, x1, x2, x3;
  wire done;
{out_decls}

  seq_mlp dut (
    .clk(clk), .rst(rst), .start(start),
    .x0(x0), .x1(x1), .x2(x2), .x3(x3),
    .done(done), {out_assoc}
  );

  integer cycles;
  task drive(input signed [3:0] a, input signed [3:0] b,
             input signed [3:0] c, input signed [3:0] d);
    begin
      cycles = 0;
      x0 = a; x1 = b; x2 = c; x3 = d;
      @(posedge clk); start = 1;
      @(posedge clk); start = 0;
      while (!done) begin
        @(posedge clk);
        cycles = cycles + 1;
        if (cycles > 200) begin $display("TIMEOUT"); $finish; end
      end
      $display("RESULT in=(%0d,%0d,%0d,%0d) y0=%0d y1=%0d cycles=%0d",
               a, b, c, d, y0, y1, cycles);
    end
  endtask

  initial begin
    rst = 1; #20 rst = 0;
{case_calls}
    $finish;
  end
endmodule
"""
    tb_path = HERE / "seq_mlp_tb.v"
    tb_path.write_text(tb, encoding="utf-8")

    vvp = HERE / "seq_mlp.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), str(vp), str(tb_path)],
        check=True,
    )
    proc = subprocess.run(
        ["vvp", str(vvp)], check=True, capture_output=True, text=True
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT")]
    print("\n--- sequential simulation ---")
    fails = 0
    for case, line in zip(cases, lines):
        toks = dict(t.split("=") for t in line.split() if "=" in t)
        sim = [int(toks["y0"]), int(toks["y1"])]
        cycles = int(toks["cycles"])
        exp = py_eval(list(case))
        ok = sim == exp
        print(f"  in={case} sim={sim} exp={exp} cycles={cycles} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails += 1
    print(f"\n{len(cases) - fails}/{len(cases)} match")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
