# safetensors2verilog

Compile safetensors-stored networks to synthesis-ready Verilog.

The HuggingFace ecosystem stores model weights in `.safetensors`. The FPGA ecosystem speaks Verilog. This tool is the bridge.

## What it does

Given a `.safetensors` file containing a network's weights and (where needed) a description of how those weights wire together, this tool emits Verilog modules that compute the same function the network does. The output is intended for synthesis through the standard FPGA toolchains (Yosys, Vivado, Quartus) and for simulation through Icarus Verilog or Verilator.

The tool is structured as a small intermediate representation with a pluggable frontend interface and a kind-dispatched Verilog backend:

- **Frontends** read a particular family of safetensors files and produce a `GateGraph` of typed `Gate` nodes.
- **Gate kinds** include `threshold`, `add`, `sub`, `mul`, `and`, `or`, `xor`, `not`, `shift_left`, `shift_right`, `concat`, `slice`, `mux`, `relu`, `clamp`, `register`, `rom`, and `constant`. Custom kinds register via `@lowering(kind)`.
- **Signals** carry width and signedness, so multi-bit signed buses, ROMs, and registered logic all flow through the same emit pipeline.

## Status

| Frontend | Status | Notes |
|----------|--------|-------|
| `threshold_logic` | working | Threshold-gate networks with named-circuit hierarchy and a `signal_registry` metadata map. Ternary weights become direct-sum comparisons; integer weights become constant-coefficient sums (`k*x`). Tested against the [8bit-threshold-computer](https://huggingface.co/phanerozoic/8bit-threshold-computer) family. |
| `bitnet_linear` | working | BitNet b1.58-style ternary linear layers with multi-bit signed activations. Reads state-dict-style tensor layout (`<prefix>.<n>.weight`, optional `<prefix>.<n>.bias`). Emits one `linear` gate per output neuron over signed buses; supports `--output-clamp LO,HI` for per-layer saturation and `--pipeline` for registered outputs. |
| `int8_linear` | working | Quantized linear layers with arbitrary signed integer weights. Same shape as `bitnet_linear` but accepts any integers within `--weight-bits`; same `--output-clamp` and `--pipeline` options. |
| `onnx_topology` | working (subset) | ONNX file gives the topology, safetensors gives weights. Supported ops: `Gemm`, `MatMul`, `Add`, `Sub`, `Mul`, `Relu`, `Identity`, `Constant`. Anything else raises `NotImplementedError` naming the op. Requires `pip install -e .[onnx]`. |

Adding a new frontend means subclassing `Frontend` and implementing `parse(path) -> GateGraph`. Operations are expressed as `Gate(kind, inputs, attrs, output_width, output_signed)`. The Verilog backend handles signal sanitization, port emission, topological-order validation, and per-kind lowering. See `safetensors2verilog/frontends/threshold_logic.py` and `bitnet_linear.py` as references.

## Install

```bash
git clone https://github.com/CharlesCNorton/safetensors2verilog.git
cd safetensors2verilog
pip install -e .
```

For the test suite:

```bash
pip install -e .[test]
```

Requires Python 3.10+, `torch`, `safetensors`. The integration test (`examples/threshold_alu/run.py`) and most simulation work also need `iverilog` on `PATH`.

## Usage

```bash
# Compile via the threshold-logic frontend
python -m safetensors2verilog input.safetensors --frontend threshold_logic -o output.v

# Custom top-level module name
python -m safetensors2verilog input.safetensors --frontend threshold_logic -o output.v --top my_design

# List available frontends
python -m safetensors2verilog --list-frontends

# List a frontend's specific options
python -m safetensors2verilog --list-frontend-options threshold_logic
python -m safetensors2verilog --list-frontend-options bitnet_linear
```

### Per-frontend options

Each frontend self-describes its CLI flags via `Frontend.options()`. After choosing `--frontend`, those flags become available on the same command line.

```bash
# threshold_logic: drop memory.* gates and emit a vendor BRAM template
python -m safetensors2verilog cpu.safetensors \
    --frontend threshold_logic \
    --skip-memory \
    --emit-bram-template bram.v \
    -o cpu_core.v

# threshold_logic: error out on stale or unresolved routing instead of
# promoting affected inputs to anonymous external ports
python -m safetensors2verilog cpu.safetensors --strict -o cpu.v

# bitnet_linear: 4-bit signed activations, per-layer saturating clamp,
# pipelined output (one cycle of latency per layer)
python -m safetensors2verilog model.safetensors \
    --frontend bitnet_linear \
    --activation-bits 4 \
    --output-clamp -8,7 \
    --pipeline \
    -o model.v

# int8_linear: 8-bit weights, 4-bit activations, custom layer prefix
python -m safetensors2verilog model.safetensors \
    --frontend int8_linear \
    --weight-bits 8 \
    --activation-bits 4 \
    --layer-prefix backbone.linear \
    -o model.v

# onnx_topology: ONNX file gives the graph, safetensors gives weights
python -m safetensors2verilog weights.safetensors \
    --frontend onnx_topology \
    --onnx model.onnx \
    --activation-bits 8 --weight-bits 8 \
    -o model.v
```

### Other CLI flags

```bash
# Inspect what a frontend produces, without emitting Verilog
python -m safetensors2verilog input.safetensors --emit-ir json -o ir.json

# Validate that a frontend accepts the input but emit nothing
python -m safetensors2verilog input.safetensors --dry-run

# Suppress progress messages on stderr
python -m safetensors2verilog input.safetensors -o out.v --quiet
```

## Worked examples

Two example scripts run end-to-end (build → convert → simulate → cross-check) and are part of CI:

```bash
python examples/threshold_alu/run.py     # threshold-network half-adder
python examples/bitnet_linear/run.py     # 3-input -> 2-output ternary linear
```

Both build a small safetensors fixture, convert it through the appropriate frontend, simulate the resulting Verilog with Icarus Verilog over a sweep of input combinations, and cross-check the simulator output against a Python evaluator of the same network.

## How the threshold-logic frontend works

A threshold gate computes `output = 1 if (Σ wᵢ·xᵢ + b) ≥ 0 else 0`. With ternary weights `wᵢ ∈ {-1, 0, 1}`, the weighted sum is `popcount(positive-weighted inputs) − popcount(negative-weighted inputs)`. Comparing that to `−b` is one integer comparison; each gate compiles to one Verilog `wire` plus one `assign`:

```verilog
wire gate_X = ((pos_a + pos_b + ...) >= (neg_c + neg_d + ... + threshold));
```

No multipliers, no floating point, no signed arithmetic in the synthesized form. Yosys/ABC fold the popcount sums into adder trees that map directly to LUTs and carry chains.

The frontend reads the safetensors file's `signal_registry` metadata (a JSON map from integer signal IDs to symbolic names), uses each gate's `.inputs` tensor to look up its input signals, and emits gates in topologically-sorted order. External inputs (signal names starting with `$`) become module ports; constant `#0` / `#1` become Verilog literals.

Non-ternary integer weights are kept as integer coefficients in the IR (`k*x` rather than k repetitions of `x`), so an int8-weighted threshold gate compiles to one short sum expression rather than a list of duplicated terms.

## How the bitnet_linear frontend works

BitNet b1.58 represents `nn.Linear` weights as ternary `{-1, 0, 1}`; activations are multi-bit (typically int8). The frontend reads tensors named:

```
<prefix>.<n>.weight    int / float tensor with shape [out, in], values in {-1, 0, 1}
<prefix>.<n>.bias      int / float tensor with shape [out] (optional)
```

For each output neuron, the frontend emits a chain: a `constant` gate carrying the bias as the initial accumulator, then one `add` or `sub` gate per non-zero weight. The accumulator width grows by `ceil(log2(in_features)) + 1` bits per layer to keep the worst-case MAC sum lossless; downstream re-quantization (clamp, register, ROM) is composable in the same IR.

End-to-end smoke: a 3-input → 2-output ternary linear with 4-bit activations, simulated in iverilog, matches Python ground truth on every test case.

## Adding a new frontend

```python
from safetensors2verilog.core import (
    Frontend, FrontendOption, Gate, GateGraph, Signal, registry,
)


@registry.register(
    "my_frontend",
    description="One-line description that shows in --list-frontends.",
    metadata_namespace="my_frontend",  # reserved metadata key prefix
)
class MyFrontend(Frontend):

    @classmethod
    def options(cls):
        return [
            FrontendOption(
                name="some-flag",
                type=int,
                default=8,
                help="surfaces as --some-flag on the CLI.",
            ),
        ]

    def parse(self, path, top="top", some_flag=8, **opts) -> GateGraph:
        # Read tensors, build Gate nodes, return GateGraph.
        gates = [
            Gate(name="y", kind="add", inputs=["a", "b"],
                 output_width=8, output_signed=True),
        ]
        return GateGraph(
            inputs=[Signal("a", width=8, signed=True),
                    Signal("b", width=8, signed=True)],
            outputs=[Signal("y", width=8, signed=True)],
            gates=gates,
            top=top,
        )
```

Drop the file under `safetensors2verilog/frontends/`, add a matching `from . import my_frontend` in `frontends/__init__.py`, and the frontend appears in `--list-frontends`.

If your frontend needs a Verilog operation the built-in lowerings don't cover, register one with `@lowering(kind)`:

```python
from safetensors2verilog.verilog import lowering


@lowering("my_op")
def lower_my_op(ctx, gate):
    return [f"  assign {ctx.name(gate.name)} = ...;"]
```

## Contributing

New frontends are the highest-impact contribution. The IR supports multi-bit signed arithmetic, parameter ROMs, registers, activations, and threshold logic; if you have a quantization scheme that fits, write a frontend.

## License

MIT.
