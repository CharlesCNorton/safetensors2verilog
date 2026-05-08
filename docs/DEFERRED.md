# Deferred features

Items intentionally scoped out of the current implementation, with the rationale.

## Conv2D (and other 2-D windowed ops)

The IR is bit-level / 1-D: each `Signal` is a flat width, and gate inputs are flat lists of signal names. Conv2D needs a windowed access pattern over a 2-D image: for each output pixel, you read a `kH x kW` block of the input. Two paths exist:

1. Lower Conv2D to im2col + matmul. The frontend reshapes the input into a 2-D matrix where each row is a flattened window, then emits a `linear` gate per output channel. This works but explodes the gate count for any non-trivial image and stride.
2. Add a `conv2d` kind to the IR with explicit shape attributes (`(in_h, in_w, in_c, out_c, kH, kW, stride_h, stride_w, pad_h, pad_w)`) and a backend lowering that emits the windowed access. This is the right design but requires the IR to track shapes, which it currently doesn't.

Either approach is a real engineering exercise. Useful for vision models (small CNNs at the edge), but not for the threshold-computer or BitNet-style MLP families that the project targets first.

## Batched / sequence dimensions

Most NN frontends carry shape info: a `Linear` layer is `[batch, in] -> [batch, out]`; a transformer block is `[batch, seq, hidden]`. The current IR has no batch or sequence axis. For combinational designs that's fine because you always emit a single forward pass; for sequential / streaming designs, batched throughput is what you'd measure.

Implementing this means widening `Signal` to carry a multidimensional shape, and propagating that through every kind's lowering. It's invasive. Plan: tackle alongside Conv2D.

## Floorplan / placement hints

Vendor-specific attributes like `(* LOC = "X12Y34" *)` (Vivado), `(* keep = "true" *)`, `(* ramstyle = "MLAB" *)` (Quartus), `(* attribute syn_keep ... *)` (Synplify). Each toolchain has its own syntax and semantics. The repo currently emits one (`(* ram_style *)` for ROMs targeting Vivado) but generalizing to a placement / floorplan attribute system means picking a target dialect and propagating user-visible knobs through the IR. Lower priority than the LLM-scale roadmap; deferred until someone needs it for a specific board.

## ONNX op coverage gaps

The `onnx_topology` frontend supports `Gemm`, `MatMul`, `Add`, `Sub`, `Mul`, `Relu`, `Identity`, `Constant`, `Reshape`, `Concat`, `Split`, and `Gather`. Several common ops are deferred:

- **Conv, ConvTranspose**: need 2-D windowed access (see above).
- **LayerNorm, GroupNorm, BatchNorm**: need fixed-point sqrt and divide. These are doable via lookup tables or iterative algorithms (Newton's method), but the IR has no fixed-point primitive; adding one means defining the format (Q-format, scale, bias) and a small library of fixed-point ops.
- **Softmax**: the underlying `softmax_block` exists, but it expects a packed K-element bus and the ONNX path emits one signal per element; needs the pack / instance / slice adapter pattern that `LayerNormalization` already uses. (`Sigmoid`, `Exp`, and `Tanh` are now wired through `sigmoid_block` / `exp_block` / `tanh_block` per-element instances.)
- **Attention**: composite of softmax + matmul over batched tensors; depends on the above plus batched primitives.

The path forward for any of these is the same: pick a fixed-point format, add the necessary primitive kinds (`fp_sqrt`, `fp_div`, `fp_exp`, `lut`), and either inline the algorithm in the frontend or expose a single composite IR kind with a complete lowering. The choice is a real design decision and deserves its own session.

## Sequential bitnet feature gaps

`bitnet_linear --sequential` ships with a fixed parallelism: one MAC per output neuron, with all out_size MACs running in parallel during a layer's compute phase. Latency is `sum(in_sizes)` cycles per inference. Below is what each feature would entail and the natural extension path.

### `--parallelism N`: time-multiplex outputs

Trade latency for area. Default behavior corresponds to `N = out_size[L]`. With `N < out_size[L]`, the layer reuses N MAC units to compute groups of N outputs at a time, looping `ceil(out_size[L] / N)` times before advancing to the next layer.

Architecture changes:
- Add an `output_group` register: 0..ceil(out_size[L]/N)-1 per layer.
- Per-layer ROM is addressed by `(output_group * N * in_size[L] + local_j * in_size[L] + counter)`, i.e. a flat row-major store of all out_size[L] * in_size[L] weights, with the per-MAC offset selected by `local_j` (0..N-1) and the per-cycle position by `counter`.
- Accumulator file: out_size[L] registers, but only the N corresponding to the current output_group update each cycle.
- FSM: a third counter wraps `output_group` once `counter == in_size[L]-1`; layer transitions when `output_group == ceil(out_size[L]/N)-1` and `counter == in_size[L]-1`.

Latency: `sum(in_sizes[L] * ceil(out_sizes[L]/N))`. Area: roughly N MAC units per layer.

### `--streaming-input`: single port + valid strobe

Replace the `x[0..N0-1]` port bank with a single `x` data bus, a `valid_in` strobe, and a `ready_out` backpressure signal.

Architecture changes:
- Replace the bank of input ports with one `x` port of activation_bits.
- Add a per-input register file (`in_buf[0..N0-1]`) that fills from `x` each cycle that `valid_in && ready_out`.
- The FSM stays in IDLE while `in_buf` is being filled; transitions to COMPUTE only when all inputs have arrived.
- For non-blocking operation, `ready_out` should reflect whether the pipeline can accept the next input (true while filling `in_buf` or while DONE; false during COMPUTE).

This protocol has flavors. The simplest is "fill all inputs, then compute, then drain" (latency = N + sum(in_sizes) + out_size). A more elaborate version pipelines the input fill with a previous inference's compute.

### `--handshake`: full ready/valid output protocol

Add `valid_out` and `ready_in` to the output side. The DONE state holds outputs valid until `ready_in` is asserted, then transitions back to IDLE. With both `--streaming-input` and `--handshake`, the module looks like a standard AXI-Stream-shaped block.

Architecture: a 4-state FSM (IDLE / FILLING / COMPUTING / VALID_WAITING). Output buffer registers hold final values across the VALID_WAITING state until `ready_in` fires.

### `--weight-bram`: runtime-loadable weights

Replace the per-output ROM with a writable BRAM, so the host can reload weights after synthesis (e.g. for fine-tuning or model swap).

Architecture changes:
- Add a `bram` IR kind: like `rom` but with `inputs = [read_addr, write_addr, write_data, write_en, clk]` and a synchronous write path. Or extend `rom` with optional write attributes.
- In `--weight-bram` mode, expose `weight_addr`, `weight_data`, `weight_we` as external module ports.
- ROM init becomes optional (zeros at reset), or kept as default values that can be overwritten.
- The user is responsible for loading weights before asserting `start`.

The Verilog template would mirror the existing `emit_bram_template`: synchronous single-port BRAM with vendor-friendly inference, plus a small write address mux that selects between layers.

---

All four are designed but not built. The current parallel-MAC sequential mode is the right default for small networks (sub-thousand-weight layers). For LLM-scale layers, `--parallelism 1` is the most needed of the four; the rest matter only when integrating into a streaming SoC.
