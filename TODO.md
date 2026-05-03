# TODO

Open items, renumbered. Completed items have been removed; the implementations live in the source tree and the rationale for each is in the relevant commit message.

1. ONNX `Conv` / `ConvTranspose` — needs a 2-D windowed access primitive in the IR (item 9 below).
2. ONNX `LayerNorm`, `GroupNorm`, `BatchNorm` — need fixed-point sqrt and divide primitives.
3. ONNX `Softmax`, `Sigmoid`, `Tanh`, `Exp` — need fixed-point transcendentals (lookup tables or iterative algorithms).
4. ONNX `Attention` — composite of softmax + batched matmul; depends on items 2, 3, and 10.
5. Sequential bitnet `--parallelism N` (time-multiplex outputs to trade latency for area). Design in `docs/DEFERRED.md`.
6. Sequential bitnet `--streaming-input` (single x port + valid_in/ready_out). Design in `docs/DEFERRED.md`.
7. Sequential bitnet `--handshake` (full ready/valid output protocol). Design in `docs/DEFERRED.md`.
8. Sequential bitnet `--weight-bram` (writable BRAM with runtime weight reload port). Design in `docs/DEFERRED.md`.
9. `Conv2D` IR primitive with explicit shape attributes.
10. Batched / sequence-dimension primitives in the IR (`Signal` carries a multidimensional shape).
11. Vendor floorplan / placement attributes (`(* LOC = ... *)` etc.) propagated through the IR.
12. Real-world safetensors fixture in CI (download a tiny canonical model, round-trip through every applicable frontend).
13. Module hierarchy: `emit_module` produces one flat module; no submodule instantiation, no hierarchy preservation from the safetensors namespace.
14. Compose two `GateGraph`s (instantiate one as a sub-module of another).
15. `lut_ff_estimate.py` analytical model is ~17× off vs yosys actual; add a calibration pass or replace with a yosys-driven estimator.
16. Mypy strict mode (currently `ignore_missing_imports = true`); full `--strict` would surface more issues.
17. Frontend-author walkthrough in `CONTRIBUTING.md` is still threshold-centric; add a non-threshold worked example showing a custom kind via `@lowering`.
18. README has no demo image or screenshot.
19. No FPGA target documentation (Lattice / Xilinx / Intel-specific notes, recommended tooling versions, expected utilization).
20. No CHANGELOG.
21. No release tags or PyPI publish workflow.
22. CI actions on Node 20 (`actions/checkout@v4`, `actions/setup-python@v5`) will need bumping before June 2026.
23. Third-party contributor pipeline untested; no external frontend has been authored.
24. No users; deployment scenarios beyond the threshold-computer family and BitNet are speculative.
25. No FPGA deployment proof (utilization/Fmax numbers on a real Lattice or Xilinx board).
26. No published synthesis results for the sequential bitnet on any LLM-scale layer size.
27. No integration test that runs `yosys` synth and checks utilization is within a budget (CI runs synth but doesn't gate on numbers).
28. SystemVerilog mode is a string-rewrite over the Verilog output; a native SV emitter would handle SV-specific constructs (interfaces, packed structs, `unique case`).
29. No SDC features beyond the basic single-clock starter; multi-clock SDC, false paths between domains, and IO standard constraints aren't generated.
30. `--emit-sdc` doesn't differentiate between input/output ports for clock association, only emits a blanket constraint.
31. No equivalence-checking driver that automates iverilog round-trip across all frontends with a parametric sweep (each frontend has its own bespoke test today).
32. `evaluate_graph` doesn't handle `tristate` high-impedance properly (substitutes 0); structurally fine for cross-checking but not a faithful simulation.
33. No formal verification harness (SVA assertions, equivalence-checking against a golden model).
34. `Frontend.parse` returns a single `GateGraph`; multi-output workflows (one frontend emitting several modules with shared types) aren't expressible.
35. No structured way to express "this frontend generates module X parameterized by Y, which the user instantiates N times"; each frontend builds a flat top.
36. The bitnet_linear ROM stores 2-bit signed weights; no ECC, no parity, no integrity checks if the BRAM bit-flips.
37. Documentation for the `evaluate_graph` register-state contract is informal; a helper that walks N cycles of a sequential graph and returns a trace would make sequential testing cleaner.

Categories (rough): items 1–11 are scoped-out features needing IR or frontend extensions; 12–17 are followups on existing surface area; 18–22 are docs/release polish; 23–26 are external validation; 27–37 are smaller architectural gaps.
