# safetensors2verilog

Compile safetensors-stored networks to synthesis-ready Verilog.

The HuggingFace ecosystem stores model weights in `.safetensors` files. The FPGA ecosystem speaks Verilog. There is currently no bridge between them. This tool aims to be that bridge.

## What it does

Given a `.safetensors` file containing a network's weights and (where needed) a description of how those weights wire together, this tool emits Verilog modules that compute the same function the network does. The output is intended for synthesis through the standard FPGA toolchains (Yosys, Vivado, Quartus) and for simulation through Icarus Verilog or Verilator.

The tool is structured around **frontends**: one per model class. Each frontend knows how to interpret a particular family of safetensors files (which tensors are weights, which are biases, what activation function applies, how the dataflow connects). The shared backend handles Verilog emission — declarations, hierarchical modules, signed arithmetic, common patterns like popcount.

## Status

| Frontend | Status | Notes |
|----------|--------|-------|
| `threshold_logic` | working | Threshold-gate networks with named-circuit hierarchy and a `signal_registry` metadata map. Ternary weights become direct-sum comparisons; no multipliers in the generated RTL. Tested against the [8bit-threshold-computer](https://huggingface.co/phanerozoic/8bit-threshold-computer) family. |
| `bitnet_linear` | planned | BitNet b1.58-style ternary linear layers; activations passed in as fixed-point. |
| `int8_linear` | planned | Standard quantized `nn.Linear` with int8 weights. |
| `onnx_topology` | planned | Use an ONNX file to describe the dataflow, fetch weights from safetensors. |

Adding a new frontend means subclassing `Frontend` and implementing two methods: parse the safetensors file into a graph of named operations, and translate each operation kind into a Verilog snippet. The shared backend handles the rest.

## Install

```bash
git clone https://github.com/CharlesCNorton/safetensors2verilog.git
cd safetensors2verilog
pip install -e .
```

Requires Python 3.10+, `torch`, `safetensors`. For simulation in the test suite: `iverilog`.

## Usage

```bash
# Threshold-logic frontend
python -m safetensors2verilog input.safetensors --frontend threshold_logic -o output.v

# With a custom top-level module name
python -m safetensors2verilog input.safetensors --frontend threshold_logic -o output.v --top my_design

# List available frontends
python -m safetensors2verilog --list-frontends
```

## Worked example

The `examples/threshold_alu/` directory walks through converting a small piece of the threshold-computer's 8-bit ALU (the boolean gates and ripple-carry adder), simulating the resulting Verilog with Icarus Verilog, and cross-checking output against a Python evaluation of the same threshold network on the same inputs. Run it with `python examples/threshold_alu/run.py`.

## How threshold-logic conversion works

A threshold gate computes `output = 1 if (Σ wᵢ·xᵢ + b) ≥ 0 else 0`. With ternary weights `wᵢ ∈ {-1, 0, 1}`, the weighted sum is `popcount(positive-weighted inputs) − popcount(negative-weighted inputs)`. Comparing that to `−b` is a single integer comparison. So each gate compiles to one Verilog `wire` plus one `assign`:

```verilog
wire gate_X = ((pos_a + pos_b + ...) >= (neg_c + neg_d + ... + threshold));
```

No multipliers, no floating point, no signed arithmetic in the synthesized form. Synthesis tools optimize the popcount sums into adder trees that map directly to LUTs and carry chains.

The frontend reads the safetensors file's `signal_registry` metadata (a JSON map from integer signal IDs to symbolic names), uses each gate's `.inputs` tensor to look up its input signals, and emits gates in topologically-sorted order so that every reference is to an already-declared wire. External inputs (signal names starting with `$`) become module ports.

## Contributing

New frontends are the highest-impact contribution. Pick a model class (BitNet, int8 quantized linear, an ONNX model, anything storable in safetensors) and implement the `Frontend` interface; the backend handles the Verilog plumbing. See `safetensors2verilog/frontends/threshold_logic.py` as a reference.

## License

MIT.
