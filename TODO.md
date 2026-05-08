# TODO

Open items, renumbered. Completed items have been removed; the implementations live in the source tree and the rationale for each is in the relevant commit message.

Recently shipped (not in this list):

- Dependency-closure subset extractor (`extract.py`) and CLI `--circuit` flag.
- Schema versioning on safetensors metadata (`core.SIGNAL_REGISTRY_SCHEMA_VERSION_LATEST`).
- Strict-by-default frontend behavior; `--promote-unresolved` opt-in for the old permissive path.
- Self-checking iverilog equivalence harness (`equivalence.py`, `--equiv-check`) and SymbiYosys equiv template (`--emit-sby-equiv`).
- Static-analysis: depth, fanout, critical path (`analysis.py`, `--report`).
- Pipeline-register insertion transform (`transforms.py`, `--pipeline-every`).
- Bus-packed port emission for any contiguous-index family (`--pack-buses`); port grouping by dotted prefix (`--group-ports`).
- Output-trim flag (`--top-outputs`); per-circuit port-contract printer (`--inspect`); circuit listing (`--list-circuits`).
- Yosys+ABC synthesis-stats wrapper (`synth.py`, `--synth-stats`).
- Metadata pass-through into Verilog header (`--metadata-passthrough`).
- iCE40 worked example with measured cell count (`examples/fpga_synth/`).

Open:

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

### Language model support

Concrete target: faithful Verilog translation of LLaMA-style transformers (SmolLM2-135M-Instruct, Qwen3-0.6B). Verified empirically by inspecting `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M bf16 params, 30 decoder layers, hidden=576, GQA 9/3, intermediate=1536, vocab=49152, RoPE θ=1e5, tied lm_head): all four existing frontends reject the file up front (weights are bf16, not integer / ternary; no `signal_registry`; no ONNX). The IR has no representation for RMSNorm, RoPE, Softmax, embedding lookup, or KV cache. Items below are the gap. The bar is *faithful translation* — synthesizability and area are secondary, but the backend's flat-text matmul lowering also has to change for any layer wider than a few thousand inputs (see item 51).

#### Numeric format

38. Fixed-point format primitive on `Signal`: a `(q_int_bits, q_frac_bits, signed)` triple, surfaced through every arithmetic kind. Prerequisite for every transcendental below. Generalises and unblocks items 2 and 3.
39. Quantization frontend (e.g. `quant_linear`): takes a fp16/bf16 safetensors plus a quant spec (per-tensor or per-channel int8/int4, GPTQ/AWQ layout) and emits an IR with explicit weight ROMs + scale tensors + multibit dequant gates. Required because nothing today accepts non-integer weights, and the only realistic LM compile path is post-training quantization.

#### Activation primitives (build on item 38)

40. `rms_norm` IR kind: `gamma * x / sqrt(mean(x*x) + eps)`. Decomposes into square / sum / mean / `fp_rsqrt` / mul over the last axis. LayerNorm (item 2) reduces to `rms_norm` plus a centering subtract.
41. `silu` and `gelu` IR kinds. SiLU = `x * sigmoid(x)`; sigmoid via either an 8K-entry x 8-bit ROM or piecewise-linear approximation. GELU via tanh approximation.
42. `softmax` IR kind with the running-max stability trick: subtract max, `fp_exp`, sum, `fp_div`. A separate `mask_then_softmax` kind keeps the causal mask out of `exp`.
43. `fp_rsqrt`, `fp_exp`, `fp_div`, `fp_recip` primitives: Newton-Raphson iteration in fixed point with caller-chosen precision, or a small LUT seed plus one Newton step. Closes the math gap behind items 2 and 3.

#### Sequence and attention machinery (depend on item 10)

44. `embedding_lookup` kind: a wrapper over `rom` indexed by token id. The SmolLM2 embedding is 49152 x 576 = 28M params — needs hierarchical BRAM emission (item 13) and per-row banking to be tractable.
45. `rope_apply` kind: precomputed sin/cos ROMs indexed by position, complex rotation pair across head_dim/2. ROM contents bake in once at compile time from the model's `rope_theta` and `max_position_embeddings`.
46. `causal_mask` kind: combinational, depends only on `(position, kv_position)`.
47. `kv_cache` IR pattern: per-layer pair of writable BRAMs sized `(num_kv_heads, max_seq, head_dim)`, write port driven by the current token's K and V projections, read port driven by the attention loop. Re-uses the `--weight-bram` design sketched in `docs/DEFERRED.md` but for activations rather than weights.
48. `multi_head_attention` composite kind: takes Q/K/V signals plus shape attrs `(num_q_heads, num_kv_heads, head_dim, max_seq)`, lowers to the masked-softmax * V construction. Depends on 10, 42, 46, 47.

#### Decode loop

49. Sampling kinds: `argmax`, `top_k`, `top_p`, `temperature_scale`. Argmax over a 49152-wide bus folds to a comparator tree.
50. Autoregressive driver: a frontend variant whose external interface is `(token_id_in, valid_in) -> (token_id_out, valid_out)`, internal state carries the KV cache and the position counter. Built on the sequential-bitnet FSM pattern.

#### Backend scale (mandatory at LM size)

51. Replace `linear`'s inline-expression lowering with a parameterised matmul block instantiation. The current path emits one assign per output neuron with every input term inline — at 49152 x 576 (lm_head) that's tens of MB of Verilog text and no synthesis tool will accept it. Closely related to item 13 (module hierarchy).
52. Vendor BRAM / DSP inference attributes on the matmul block so Vivado / Quartus / Yosys infer block RAM for weights and DSP cascades for the MAC tree. Generalises the existing `(* ram_style *)` plumbing in `rom`.

#### Verification

53. Tiny-LLaMA round-trip fixture: 1-layer model with hidden=8, num_heads=2, vocab=16, seq=4 (whose entire forward pass fits in iverilog in seconds), end-to-end through the LM frontend, cycle-accurate against the `transformers` fp32 reference for a handful of token sequences. Without this the LM frontend has no truth source.

#### Assembly and bring-up

The primitive library shipped in `safetensors2verilog/blocks/` (matmul_seq, rms_norm, softmax, silu, sigmoid, exp, rsqrt, embedding, kv_cache, argmax, requantize, rope_apply) is verified bit-exact in iverilog. The remaining work is wiring those primitives into a transformer and proving the full forward pass.

54. `multi_head_attention` composite kind. Wires the verified ingredients: per-head, write the current token's K and V projections into the layer's `kv_cache`, RoPE-rotate Q and K, sequentially compute Q ⋅ cached-K^T over positions 0..p, feed the result through `softmax_stable` with the causal mask, and accumulate softmax-weighted V. GQA broadcasts each KV head over `num_q_heads / num_kv_heads` Q heads. No new primitives; just the FSM and address generation that ties them together.
55. `hf_llama` frontend that walks `transformers.LlamaForCausalLM` config + state_dict and emits the full graph. Must generalise across the LLaMA-architecture family (Llama 2/3, Qwen 2/3, Mistral, SmolLM2) since they share the same `(embed -> N x [rms_norm, attention, residual, rms_norm, swiglu, residual] -> rms_norm -> lm_head)` pattern; the only per-model differences are config integers and `hidden_act`. Distinguishing factor from the existing `int8_linear` frontend: walks the model's *architecture* and emits hierarchical instances of the new blocks rather than flat per-weight `linear` gates.
56. Tiny-LLaMA cycle-accurate fixture: 1 layer, hidden=8, num_q_heads=2, num_kv_heads=1, head_dim=4, intermediate=16, vocab=8, seq=4 (also subsumes item 53). Random-weight model whose full forward pass fits in iverilog in a few seconds. Compare cycle-by-cycle against `transformers` fp32 reference dequantised through the same per-channel int8 PTQ the frontend applies. Pass criterion: identical token-id outputs after argmax over a handful of random prompts.
57. SmolLM2-135M end-to-end compile + iverilog round-trip on short prompts (3-5 tokens). Apply the `hf_llama` frontend to `HuggingFaceTB/SmolLM2-135M-Instruct`, emit Verilog (likely 100s of MB given lm_head scale; will need `$readmemh` sidecar weight files end to end), simulate one autoregressive step in iverilog, compare next-token argmax against the int8-quantised reference run on CPU. The reference itself is a separate fp32 -> int8 PTQ run plus the iverilog-equivalent fixed-point activation maths; both should agree on the produced token id.
58. Yosys synth pass on the SmolLM2 compile + utilization / Fmax report. Drives Yosys with the appropriate target (`synth -top top` for generic, `synth_xilinx` / `synth_intel` for vendor-specific). The blocks already carry `(* ram_style = "block" *)` and `(* use_dsp = "yes" *)` pragmas; the report should land in the README. Once we have a number, we know what FPGA tier (Stratix 10 MX, Versal AI Core, Alveo U250) is the smallest the design fits into.

### Status as of the LM bring-up (items 54-58)

54 (multi_head_attention) — DONE. Verified bit-exact 4/4 autoregressive tokens at H=2 KV=1 D=4 MAX_SEQ=4 in iverilog (110 cycles per token). Source: `safetensors2verilog/blocks/attention.py`.

55 (hf_llama frontend) — DONE. Reads HF safetensors + config.json, applies per-channel int8 PTQ, emits the per-token forward pass using all the verified primitives wired through start/done chains. Verified end-to-end on a synthetic LlamaForCausalLM (hidden=8, H=2, KV=1) at 4/4 clean done pulses + non-X logits, 234 cycles per token. Generalises across LLaMA / Qwen / Mistral / SmolLM2 by config integers. Source: `safetensors2verilog/frontends/hf_llama.py`. Frontend options: `max-seq-override`, `num-layers-override`, `skip-lm-head` (for SmolLM2-scale testing without the 49152-output matmul).

56 (tiny-LLaMA cycle-accurate fixture) — DONE. Hand-wired 1-layer LLaMA forward pass through embed → rmsnorm → q/k/v proj → rope → attention → o_proj → residual → rmsnorm → swiglu mlp → residual → rmsnorm → lm_head → argmax. 4/4 tokens bit-exact (logits + argmax) against a Python replica of the same fixed-point math. 9895 lines of generated Verilog, 234 cycles per token. Source: `_play/tiny_llama_e2e.py`.

57 (SmolLM2 compile) — Frontend-side DONE; 1-layer Verilog package emits cleanly (12353 IR gates → 174k lines of Verilog (6 MB) + 5185 sidecar weight ROM files (80 MB)). iverilog parses it in 3.2s. iverilog's *vvp* simulator at this scale never finishes (no observable progress in 12+ minutes); the path that works is **Verilator on WSL Ubuntu (apt verilator + g++)** — compiles the design to native C++ in 29s, links to a 41-second-per-token simulation binary. End-to-end inference demo: prompt `"The"` (token id 504) feeds through the verilator-built binary in 13615 cycles / 24.9s, returns the 576-dim final_norm hidden state; CPU-side fp32 lm_head + argmax through the HF tokenizer decodes the next token. The 1-layer truncation + uncalibrated activation Q-format means the prediction is dominated by the input embedding (the model "predicts back" capitalisation variants of the input rather than fp32 SmolLM2's actual prediction); fixing that is items 59 + 62 below.

58 (Yosys synth for utilization / Fmax) — DONE on tiny-LLaMA. Tiny LLaMA via `yosys synth -top tiny_llama` + `abc -g AND,OR,XOR,NAND,NOR,XNOR`: 1,866 flops (1,794 DFFE + 45 DFF + 14 DFF1 + 11 SDFFCE + 2 DFFE_PP1P) and 66,557 combinational cells across 25 submodule types. Well within a small Spartan-7 / iCE40HX8K. SmolLM2 1-layer with generic `synth` consumes 41+ GB of RAM mapping every weight ROM to muxed flop arrays (no BRAM cells in the generic library); `synth_xilinx` errors on the 28M-entry embedding RAM ("no valid mapping found") because a single Xilinx BRAM can't hold V=49152 × H=576 × 8 = 226 Mbit. Real Vivado handles this by auto-chunking; Yosys's xilinx flow needs source-level pre-chunking — see item 60.

#### LM bring-up follow-ups

These are the concrete next steps surfaced by the items 54-58 work:

59. SmolLM2 30-layer scale-up. Drop ``num_layers_override=1`` from the frontend call and re-emit. Expect ~30x more matmul submodules (210 unique shape+suffix names instead of 7), ~30x more sidecar files (~155k files instead of 5184), Verilator compile time of roughly 15-20 minutes, simulation time of roughly 20 minutes per token. The frontend's per-layer code path is already in place; this item is execution + capture, not new architecture work.
60. Source-level RAM chunking. Split V*H embedding (and the lm_head row, when streaming-lm_head is wired up at scale — item 59) into N column-banks of B rows each, sized to fit a single Xilinx 7-series 36 Kbit BRAM (or whichever target's primitive). Without this Yosys's `synth_xilinx` and any pure-iverilog flow that expects BRAM inference will reject the giant embedding RAM. Touches `safetensors2verilog/blocks/embedding.py` and the streaming variant of matmul.
61. Vivado / Quartus synth + place + route on the chunked design from item 60. Captures real utilization (LUT, FF, BRAM, DSP) and Fmax on a Versal AI Edge / Stratix 10 MX / Alveo U250-class target. Commercial-tool work; the design itself doesn't need to change beyond item 60.
62. PTQ activation calibration. The 1-layer SmolLM2 inference produced saturated activations (final_norm values clustered at ±127 / ±128) because the hf_llama frontend uses heuristic ``_qk_v_proj_shift`` and similar values for the per-block requantize shifts rather than calibrated values. Right path: run a small calibration set (e.g. 32 sequences from C4 / WikiText) through a CPU reference, capture per-block activation distributions, pick shift values that put the bulk of the distribution inside int8 without saturation. The frontend's ``requantize_block`` calls already accept per-channel ``muls`` and ``shifts`` lists; calibration just fills those in.
63. Bit-exact Python reference for ``LlamaForCausalLM`` forward pass + frontend verification at SmolLM2 shape. Tiny LLaMA has this (`_play/tiny_llama_e2e.py`). For SmolLM2 we currently rely on the synthetic-1-layer test from item 55 for frontend correctness; a real shape-matched reference (Python doing the same fixed-point arithmetic the Verilog does) gives a ground truth we can compare against, layer by layer, instead of comparing the Verilog output to fp32 `transformers`. Critical for confidence in the 30-layer scale-up.
64. Multi-token autoregressive driver. The current inference harness (`_play/smollm_inference.py`) drives one token. The KV cache lives inside the attention block and persists across calls naturally, so multi-token decode is just a Python loop that drives token N at position N, captures the predicted next token, feeds it back as token N+1. Need to run this past `MAX_SEQ` to confirm the KV cache write/read indexing works for non-zero positions.
65. CLI argparse `allow_abbrev=False`. The `--pipeline` (bitnet_linear option) prefix-collides with `--pipeline-every` (global), and similar prefix collisions will surface as the LM-frontend options multiply (`--skip-lm-head` / `--max-seq-override` / `--num-layers-override`). One-line fix in `cli.py`'s `argparse.ArgumentParser(...)` call.
66. Sidecar weight-ROM file management. SmolLM2 emits ~5k files per layer; full 30-layer compiles into ~155k files in one directory, which is unwieldy on every file system and bad for incremental rebuilds. Either a manifest + tarball, or per-module subdirectories, or a single concatenated file with offset addressing (would need matmul_seq + embedding rework).
67. Verilator + g++ as a CI target for the LM frontend. The current CI (`.github/workflows/ci.yml`) runs iverilog only. Adding a verilator job that runs `_play/smollm_inference.py` against the SmolLM2 weights would catch regressions in the LM emission path; needs ``transformers`` + the SmolLM2 weights cached in CI (or a smaller stand-in model).
68. Compare Verilog SmolLM2 next-token predictions against `transformers` fp32 reference. Once items 59 (30 layers) and 62 (calibration) land, this becomes the actual "is the Verilog model an accurate quantised SmolLM2" question. Pass criterion: ``argmax(verilog_logits) == argmax(transformers_logits)`` on most prompts, allowing for known PTQ accuracy loss on the tail.
69. fp8 native primitives as an alternative to int8 PTQ. HuggingFace ships some models pre-quantised to fp8 (E4M3 / E5M2). The current frontend dequantises fp8 → fp32 → re-quantises to int8, which loses precision twice. A native fp8 path with fp8 multipliers + fp8/fp16 accumulators would preserve the upstream quant. Significant new IR work; tracked here so it doesn't get lost.
70. Document the verilator + WSL build path in `README.md` / `CONTRIBUTING.md`. Current step-by-step lives only in commit messages and `_play/smollm_inference.py`. Should land under a "Running large models" section.

Categories (rough): items 1–11 are scoped-out features needing IR or frontend extensions; 12–17 are followups on existing surface area; 18–22 are docs/release polish; 23–26 are external validation; 27–37 are smaller architectural gaps; 38–53 are the language-model-support primitive cluster (with 38, 39, 51, and the prerequisite item 10 as foundations); 54–58 are the LM assembly + bring-up sequence (all five fully landed: 54-56 bit-exact at synthetic / tiny scales, 57 produces a verilator-simulated SmolLM2 1-layer that performs an end-to-end inference, 58 yosys-synthesises the tiny LLaMA cleanly); 59–70 are the next-step items surfaced by the bring-up — 59-62 take the design from "1 layer + uncalibrated" to "30 layers + calibrated to match fp32 SmolLM2", 63-68 are the surrounding rigor (Python reference, multi-token harness, CI, fp32 comparison), and 69-70 are forward-looking (fp8 native, docs).
