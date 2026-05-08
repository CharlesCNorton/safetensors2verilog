"""Emit SmolLM2 1-layer Verilog only (no iverilog/vvp invocation)."""
import time
from pathlib import Path
import sys

t0 = time.time()
from safetensors2verilog import collect_sidecar_files, emit_module
from safetensors2verilog.core import registry

SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT = Path(r"D:\safetensors2verilog\_play\smollm_out")
OUT.mkdir(exist_ok=True)
print(f"{time.time()-t0:.1f}s deps", flush=True)

t0 = time.time()
fe = registry.get("hf_llama")()
graph = fe.parse(
    SMOLLM_DIR / "model.safetensors",
    top="smollm_l1",
    config=str(SMOLLM_DIR / "config.json"),
    activation_bits=8, weight_bits=8,
    num_layers_override=1, max_seq_override=4,
    skip_lm_head=True,
)
print(f"{time.time()-t0:.1f}s parse", flush=True)

t0 = time.time()
text = emit_module(graph)
print(f"{time.time()-t0:.1f}s emit ({len(text.splitlines())} lines, {len(text)/1e6:.1f} MB)", flush=True)

t0 = time.time()
v_path = OUT / "smollm_l1.v"
v_path.write_text(text, encoding="utf-8")
print(f"{time.time()-t0:.1f}s write Verilog", flush=True)

t0 = time.time()
sidecar = collect_sidecar_files(graph)
print(f"{time.time()-t0:.1f}s collect sidecar ({len(sidecar)} files)", flush=True)

t0 = time.time()
total = 0
for fn, contents in sidecar.items():
    (OUT / fn).write_text(contents, encoding="utf-8")
    total += len(contents)
print(f"{time.time()-t0:.1f}s write sidecar ({total/1e6:.1f} MB)", flush=True)

# Build a minimal testbench (matches the existing tb format)
HID = 576
ABITS = 8
VOCAB = 49152
pos_bits = max(1, (4 - 1).bit_length() + 1)
token_bits = max(1, (VOCAB - 1).bit_length())

tb = (
    "`timescale 1ns/1ps\n"
    "module tb;\n"
    "  reg clk = 0; always #5 clk = ~clk;\n"
    "  reg rst = 1, start = 0;\n"
    f"  reg [{token_bits-1}:0] token_id;\n"
    f"  reg [{pos_bits-1}:0] position;\n"
    "  wire done;\n"
    f"  wire signed [{HID*ABITS-1}:0] final_norm;\n"
    "  smollm_l1 dut(.clk(clk), .rst(rst), .start(start),\n"
    "                .token_id(token_id), .position(position),\n"
    "                .done(done), .final_norm(final_norm));\n"
    "  integer cycles;\n"
    "  initial begin\n"
    "    rst = 1; #20 rst = 0;\n"
    "    @(negedge clk);\n"
    f"      token_id = {token_bits}'d42;\n"
    "      position = 0;\n"
    "      start = 1;\n"
    "    @(negedge clk); start = 0;\n"
    "    cycles = 0;\n"
    "    while (!done) begin\n"
    "      @(posedge clk);\n"
    "      cycles = cycles + 1;\n"
    "      if (cycles > 200000) begin $display(\"TIMEOUT\"); $finish; end\n"
    "    end\n"
    "    $display(\"DONE %0d cycles  final_norm=%h\", cycles, final_norm);\n"
    "    $finish;\n"
    "  end\n"
    "endmodule\n"
)
(OUT / "tb.v").write_text(tb, encoding="utf-8")
print("wrote testbench", flush=True)
