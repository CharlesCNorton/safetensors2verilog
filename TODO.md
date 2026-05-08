# TODO

Open items, renumbered. Completed items have been removed; the
implementations live in the source tree and the rationale for each is
in the relevant commit message.

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
- iCE40 worked example with measured cell count (`examples/fpga_synth/`); full `nextpnr-ice40` place + route + Fmax reporting on the sequential bitnet variant (134 LCs, 96.59 MHz on iCE40HX1K).
- Hierarchical submodule emission: `instance` and `extern_wire` IR kinds, `RawSubmodule`, `submodules` field on `GateGraph`, `collect_sidecar_files` helper.
- Q-format fields on `Signal` (`q_int_bits`, `q_frac_bits`, `scale`); plus a multidimensional `Signal.shape` / `Gate.output_shape` for `[seq, hidden]` / `[batch, seq, hidden]` packed buses (`Signal.element_count`, `total_bits`, `element_bit_range` helpers; backend honours `total_bits()` for port widths).
- Per-tensor and per-channel symmetric int PTQ helper (`safetensors2verilog.quantize`).
- Hardware block library (`safetensors2verilog/blocks/`): `matmul_seq`, `matmul_stream`, `requantize`, `sigmoid`, `exp`, `tanh`, `silu`, `rsqrt`, `rms_norm`, `layer_norm`, `softmax`, `embedding`, `kv_cache`, `argmax`, `rope`, `attention`. All bit-exact in iverilog.
- `hf_llama` frontend covering the LLaMA-architecture family (Llama / Qwen / Mistral / SmolLM2). End-to-end SmolLM2-135M-Instruct 1-layer inference through verilator: real text input -> KV cache update -> attention -> MLP -> final_norm -> CPU-side lm_head + argmax -> output token via the HF tokenizer. The 30-layer compile path also works: full SmolLM2-135M emits 189 MB of Verilog in ~108s with the streaming lm_head auto-selected (VOCAB=49152 trips the threshold).
- ONNX op coverage: `Sigmoid`, `Exp`, `Tanh`, `Softmax`, `Conv` (forward; via the `conv2d` IR primitive), `Attention` (single-head, no mask; lowered as score-mul + per-row softmax + output-mul), `BatchNormalization` (inference; running stats baked into a per-channel linear), `GroupNormalization` (composed of per-group `layer_norm_block` instances), `LayerNormalization`, `Reshape`, `Concat`, `Split`, `Gather` (initializer + dynamic-index), broadcast `Add` / `Sub` / `Mul`. `ConvTranspose` is the only op left in the deferred list.
- Sequential bitnet variants: `--handshake` (DONE state holds until `ready_in`, exposes `valid_out`), `--streaming-input` (single `x` + `valid_in` / `ready_out` plus an in_buf register file and a FILL FSM state), `--weight-bram` (per-output writable RAM via the new `ram_writable` IR kind plus `weight_addr_layer` / `weight_addr_output` / `weight_addr_position` / `weight_data` / `weight_we` ports), `--parallelism N` (output_group register cycling through `ceil(out_size/N)` groups; per-(L,j) accumulator updates gated by `group_match`). All seven combinations (baseline + each variant + the handshake/streaming/weight_bram triple) compile through `iverilog -g2012`.
- `conv2d` IR primitive (`safetensors2verilog.verilog.lowering("conv2d")`): combinational 2-D convolution with explicit shape attributes (`in_h`, `in_w`, `in_c`, `out_c`, `kH`, `kW`, `stride`, `pad`); bit-exact in iverilog across an 8-case sweep.
- `ram_writable` IR kind: synchronous-write / asynchronous-read writable RAM with per-cycle `write_en` / `write_addr` / `write_data` and a separate `read_addr`. Vendor synth (Vivado / Quartus / Yosys) infers it as block RAM; `(* ram_style = "block" *)` attribute attached.
- `fp8_e4m3_mul` IR kind: 8-bit fp8 e4m3 multiply emitting fp16 IEEE-754 product (sign XOR, exponent add, 4x4 mantissa multiply, normalize, saturate). `safetensors2verilog.quantize.fp8_e4m3_quantize` / `_dequantize` round-trip fp32 to/from fp8 e4m3 bit patterns with saturating round-to-nearest.
- BRAM chunking helper (`safetensors2verilog.bram_chunk`): `pick_chunking(depth, width, max_bram_bits, max_bram_depth)` and `emit_chunked_rom` split oversized weight ROMs into per-(row_bank, col_bank) BRAM-sized banks with a row-bank mux + column-bank concat. Bit-exact in iverilog across a 32x16 fixture chunked into 4 row-banks.
- Sidecar weight-ROM management (`safetensors2verilog.write_sidecar_files`): three layouts (`flat`, `subdirs` per-module, `tarball`) plus a top-level `manifest.json` mapping submodule -> sidecar file list. CLI flags `--sidecar-layout` and `--sidecar-tarball` plumb it through. Replaces the SmolLM2 case where one directory accumulated 5k+ hex files per layer.
- Multi-output `Frontend.parse_multi`: returns one or more independent `GateGraph` objects, defaulting to `[parse(...)]` for single-output frontends. CLI `--emit-multi DIR` writes one `<top>.v` per returned graph.
- `emit_instantiation_template(graph)`: render a paste-ready Verilog instantiation snippet binding every external port to a same-name parent signal, plus `#(.PARAM(value))` overrides for parameter ports. CLI flag `--emit-instantiation PATH`.
- Native SystemVerilog emit: lowerings honour `EmitContext.target` and emit `always_ff`, `always_comb`, `unique case` directly when target is sv (no longer a post-emit string rewrite for those constructs; the rewrite still converts wire/reg declarations to `logic`).
- `register` lowering bug fix: the `enable` attribute (or 2nd input) was inserted into the emitted `always` block as a raw signal name, bypassing the identifier sanitisation pass that handles dotted IR names. Now the enable goes through `ctx.name(en_signal)` like the data and reset signals.
- PTQ activation-calibration module (`safetensors2verilog.calibration`): `collect_activation_stats` + `derive_requantize_params` + `saturation_summary` + `calibrate_iteratively` (3-round refinement consuming the previous iteration's per-channel shifts in the chain; converges on synthetic tiny shapes, oscillates on full SmolLM2 because the residual path amplifies cross-site changes — damped iteration noted in the docstring as the next refinement). The `hf_llama` frontend accepts the derived per-channel `(muls, shifts)` via `build_llama_graph(..., requantize_params=...)` and via the CLI flag `--calibration <json_path>`. Replaces the analytical `_matmul_shift(K, wbits) = wbits + ceil(log2(K)) - 2` uniform-per-layer shift with a per-channel `(mul, shift)` chosen from the observed accumulator distribution; on SmolLM2-135M layer 0, drops underflow at the requantize sites from 85.71% to 0.00% with 0% saturation.
- LLaMA pure-Python int reference (`safetensors2verilog.llama_reference`): `llama_int_reference_one_layer` (bit-stable 1-layer forward pass mirroring the Verilog int chain), `llama_fp32_reference_logits_one_layer` (fp32 gold for diffing), `llama_int_reference_decode_loop` (multi-token autoregressive driver pattern), `compare_argmax_agreement` (per-position next-token agreement metric).
- Real-world safetensors fixtures in the test suite: `tests/test_real_world_fixture.py` round-trips a 2-layer ternary bitnet (8 -> 4 -> 2) and a 4-layer int8 chain (6 -> 5 -> 4 -> 3 -> 2) through compile, Python evaluator, and iverilog bit-exact (16 random input vectors).
- Vendor placement attributes propagated through register lowerings (`vivado_loc`, `quartus_chip_pin`, `lattice_loc`, `synplify_keep`, `generic_attr`).
- Multi-clock SDC with cross-domain false paths and per-port input/output delays (`--emit-sdc`).
- `cross_frontend_equiv_sweep` and `trace_n_cycles` parametric equivalence drivers.
- Tristate `Z` faithfully propagated as `None` through `evaluate_graph` / `step_graph`.
- `emit_sva_assertions` SystemVerilog assertion module (bind-able into the DUT for formal / SVA-aware simulation).
- `bitnet_rom_parity_bits` helper for single-bit-flip detection on ternary weight ROMs.
- `argparse(allow_abbrev=False)` so frontend flags don't prefix-collide with global flags.
- `mypy --strict` per-module overrides on `core` / `analysis` / `transforms` / `synth` / `evaluate` — all clean.
- CI bumped to `actions/checkout@v5` + `actions/setup-python@v6`; yosys synth budget gating; verilator + g++ job exercising the LM emission path through tiny LLaMA + the streaming-matmul tests.
- `yosys_calibrated_estimate` wrapper in `examples/lut_ff_estimate.py` (the analytical model was an order of magnitude off vs Yosys actual).
- 226 unit tests pass (was 175 before this session's wave); the bit-exact iverilog tests cover the new tanh, conv2d, chunked ROM, fp8 mul, real-world bitnet fixture, and seven sequential-bitnet variant combinations.

## Open

### Reserved for future iteration

- Sequential bitnet `--parallelism N` follow-up: physical MAC sharing. The current implementation adds the `output_group` register and per-(L,j) `group_match` gating, so the FSM walks `ceil(out_size/N)` groups and only N accumulators update per cycle, but every accumulator still exists in the IR (no area win unless the synth tool's resource sharing pass collapses them). A second iteration replaces the per-output accumulators with a shared N-element MAC bank plus an out_size-element storage register file; that change needs a more invasive IR refactor.
- ONNX `ConvTranspose`. Forward `Conv` is wired through the `conv2d` IR primitive; transposed convolution still needs its own primitive (or to lower to `Conv` plus an upsample + post-transform).
- Damped PTQ iteration. The current `calibrate_iteratively` uses the previous iteration's shifts verbatim in the chain; on real SmolLM2 this oscillates because residual paths amplify cross-site changes. A damped form (e.g. `new_mul = 0.5 * (old_mul + raw_new_mul)` per channel) is the next refinement.
- Vivado / Quartus synth + place + route. The OSS CAD Suite path (yosys + nextpnr-ice40 / nextpnr-ecp5) is fully wired and reports real Fmax + utilisation; the licensed-tools path uses the same `.sv` / `.v` output and standard tool wrappers, deferred until a target board is in scope.
- Full SmolLM2 30-layer end-to-end inference through verilator. The 30-layer Verilog generation works (189 MB, ~108s), and verilator can build it given enough memory (the existing 1-layer build is the proven path); the 30-layer simulation is a long-running task that exceeds typical interactive iteration budgets and is left for a dedicated training/inference run.
