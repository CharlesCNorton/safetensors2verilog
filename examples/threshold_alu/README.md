# Half-adder example

End-to-end demonstration: build a small threshold network, convert it to Verilog, simulate the result with Icarus Verilog, and cross-check against a Python evaluation of the same network.

The network is a half-adder with sum and carry outputs. The XOR (sum) is built from `OR + NAND -> AND`, the AND (carry) is direct. All weights are in `{-1, 0, 1}`; biases are small integers. No multipliers appear in the synthesized form.

```bash
python run.py
```

Expected output: one line per stimulus showing the simulator's output, the Python evaluator's output, the arithmetic ground truth, and a `[OK]` if all three match. After the script finishes you'll find:

- `output/half_adder.safetensors` — the source threshold network
- `output/half_adder.v` — the generated Verilog
- `output/half_adder_tb.v` — the testbench
- `output/tb.vvp` — the iverilog-compiled simulation

The generated `half_adder.v` is fully synthesizable. Inspecting it shows what the converter produces: each threshold gate is one `wire` plus one `assign` of the form `(Σ positive inputs + bias_pos) >= (Σ negative inputs + bias_neg)`.
