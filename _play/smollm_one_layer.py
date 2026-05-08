"""Extract one Linear from SmolLM2-135M (k_proj of layer 0, first 16 of 192
outputs), int8-symmetric-quantize, run through the int8_linear frontend,
simulate in iverilog, compare to PyTorch ground truth.

This is the largest faithful slice of SmolLM2 that the current tool can
emit without TODO items 38-53.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from safetensors2verilog import emit_module, registry

ROOT = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
HERE = Path(__file__).parent

# -- Choose the slice ---------------------------------------------------------
LAYER_NAME = "model.layers.0.self_attn.k_proj.weight"
OUTPUTS_KEPT = 16   # of the 192 k_proj outputs (one full KV head + a bit)
ACT_BITS = 8        # int8 activations (signed, [-128, 127])

with safe_open(str(ROOT / "model.safetensors"), framework="pt") as f:
    W_full = f.get_tensor(LAYER_NAME).to(torch.float32)   # [192, 576]
print(f"loaded {LAYER_NAME}: {tuple(W_full.shape)}, "
      f"min={W_full.min():.4f} max={W_full.max():.4f}")

W = W_full[:OUTPUTS_KEPT]   # [16, 576]
in_dim = W.shape[1]
out_dim = W.shape[0]

# -- int8 symmetric per-tensor quantization of the weights --------------------
w_scale = W.abs().max().item() / 127.0
W_q = W.div(w_scale).round().clamp(-128, 127).to(torch.int8)
print(f"weight scale: {w_scale:.6f}")
print(f"W_q range: [{W_q.min().item()}, {W_q.max().item()}]  "
      f"unique values: {len(W_q.unique())}")
# Reconstruction error (just to see how lossy int8 is on this slice)
W_dq = W_q.to(torch.float32) * w_scale
err = (W - W_dq).abs().mean().item()
print(f"mean abs reconstruction error: {err:.6f} "
      f"(W std = {W.std().item():.4f})")

# -- Save in int8_linear's expected format ------------------------------------
fixture = HERE / "smollm_k_proj_slice.safetensors"
save_file(
    {
        "layers.0.weight": W_q.to(torch.int32),  # frontend reads as int
        # No bias on k_proj in LLaMA-style models
    },
    str(fixture),
)
print(f"wrote {fixture} ({fixture.stat().st_size/1024:.1f} KB)")

# -- Compile ------------------------------------------------------------------
fe = registry.get("int8_linear")()
graph = fe.parse(
    fixture, top="smollm_kproj16",
    activation_bits=ACT_BITS, weight_bits=8,
)
print(f"IR: {len(graph.gates)} gates, {len(graph.inputs)} inputs, "
      f"{len(graph.outputs)} outputs")
text = emit_module(graph)
v_path = HERE / "smollm_kproj16.v"
v_path.write_text(text, encoding="utf-8")
print(f"wrote Verilog: {len(text.splitlines())} lines, "
      f"{v_path.stat().st_size/1024:.1f} KB")

if shutil.which("iverilog") is None:
    print("no iverilog; stopping.")
    raise SystemExit(0)

# -- Generate test cases (random int8 activation vectors) ---------------------
torch.manual_seed(0)
N_CASES = 5
x_cases = torch.randint(-32, 32, (N_CASES, in_dim), dtype=torch.int32)

# Ground truth: pure int matmul with the int8 weights, no scale (since we
# verify the frontend's emitted Verilog reproduces the integer arithmetic).
expected = (x_cases.to(torch.int64) @ W_q.to(torch.int64).t()).tolist()

# -- Build testbench ----------------------------------------------------------
out_widths = {s.name: s.width for s in graph.outputs}
# Sort by the integer suffix after "y", not lexically: y0, y1, y2 ... y15
# (lexical sort would interleave y10-y15 between y1 and y2).
import re as _re
def _y_idx(name: str) -> int:
    m = _re.search(r"y(\d+)$", name)
    return int(m.group(1)) if m else -1
out_names = sorted(out_widths, key=_y_idx)
out_ids = [n.replace(".", "_") for n in out_names]

x_decl = "\n".join(
    f"  reg signed [{ACT_BITS-1}:0] x{i};" for i in range(in_dim)
)
y_decl = "\n".join(
    f"  wire signed [{out_widths[n]-1}:0] {oid};"
    for n, oid in zip(out_names, out_ids)
)
x_assoc = ", ".join(f".x{i}(x{i})" for i in range(in_dim))
y_assoc = ", ".join(f".{oid}({oid})" for oid in out_ids)
y_args = ", ".join(out_ids)
y_fmt = " ".join("%0d" for _ in out_ids)

case_blocks = []
for ci, x in enumerate(x_cases):
    drives = "\n".join(f"    x{i} = {int(v)};" for i, v in enumerate(x))
    case_blocks.append(
        f"    // case {ci}\n{drives}\n    #1;\n"
        f"    $display(\"CASE {ci} {y_fmt}\", {y_args});"
    )

tb = f"""\
`timescale 1ns/1ps
module tb;
{x_decl}
{y_decl}

  smollm_kproj16 dut (
    {x_assoc},
    {y_assoc}
  );

  initial begin
{chr(10).join(case_blocks)}
    $finish;
  end
endmodule
"""
tb_path = HERE / "smollm_kproj16_tb.v"
tb_path.write_text(tb, encoding="utf-8")

vvp = HERE / "smollm_kproj16.vvp"
print("compiling with iverilog...")
subprocess.run(
    ["iverilog", "-g2012", "-o", str(vvp), str(v_path), str(tb_path)],
    check=True,
)
print("running...")
proc = subprocess.run(
    ["vvp", str(vvp)], check=True, capture_output=True, text=True
)
sim_lines = [l for l in proc.stdout.splitlines() if l.startswith("CASE")]

# -- Compare ------------------------------------------------------------------
print(f"\n{'case':<6} {'iverilog == pytorch':<22} {'first 4 outputs (sim/exp)':}")
fails = 0
for ci, (line, exp) in enumerate(zip(sim_lines, expected)):
    sim = [int(t) for t in line.split()[2:]]
    ok = sim == exp
    if not ok:
        fails += 1
    diff_preview = " / ".join(
        f"{s}~{e}" for s, e in list(zip(sim, exp))[:4]
    )
    print(f"  {ci:<4} {'OK' if ok else 'FAIL':<22} {diff_preview}")

print(
    f"\n{N_CASES - fails}/{N_CASES} cases match PyTorch int matmul exactly."
)
print(
    f"What ran: 16 of layer 0's 192 k_proj output neurons of "
    f"SmolLM2-135M, int8-quantized, compiled to {len(text.splitlines())} "
    f"lines of Verilog, simulated cycle-by-cycle in iverilog."
)
