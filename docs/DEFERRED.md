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
- **Softmax, Sigmoid, Tanh, Exp**: need fixed-point transcendentals. Same blocker as the norms.
- **Attention**: composite of softmax + matmul over batched tensors; depends on the above plus batched primitives.

The path forward for any of these is the same: pick a fixed-point format, add the necessary primitive kinds (`fp_sqrt`, `fp_div`, `fp_exp`, `lut`), and either inline the algorithm in the frontend or expose a single composite IR kind with a complete lowering. The choice is a real design decision and deserves its own session.

## Sequential bitnet feature gaps

`bitnet_linear --sequential` ships with a fixed parallelism (one MAC per output, parallel across all outputs of the active layer). The list below is what's needed to scale to LLM-size layers:

- `--parallelism N`: time-multiplex outputs to trade latency for area. Doable: add an output-iterator counter alongside the input counter.
- `--streaming-input`: replace the `x[0..N-1]` port bank with a single input bus and a `valid_in` strobe. Doable but the protocol design (whether to buffer, whether to gate on `ready_out`) matters.
- `--handshake`: full Ready/Valid streaming protocol for both ends of the pipeline.
- `--weight-bram`: replace the per-output ROM with a writable BRAM, exposing a write port for runtime weight reload. Doable; needs a separate `bram` IR kind (or extending the `rom` kind to optionally accept write inputs).

These are substantial features; each warrants its own change with careful protocol thought. Not needed for small networks where the current parallel-MAC sequential mode works fine.
