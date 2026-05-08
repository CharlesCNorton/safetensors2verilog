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
- Vendor synth/PnR script generators (`safetensors2verilog.synth_vendor`): `emit_vivado_tcl` (Vivado batch synth_design + place_design + route_design + utilization/timing reports), `emit_quartus_qsf` (Quartus QSF + SDC + flow.tcl bundle for `quartus_sh -t`), `emit_synopsys_dc_tcl` (Design Compiler `compile_ultra` + area/timing reports). CLI flags `--emit-vivado-tcl`, `--emit-quartus`, `--emit-synopsys-dc-tcl` plumb them through.
- Metadata pass-through into Verilog header (`--metadata-passthrough`).
- iCE40 worked example with measured cell count (`examples/fpga_synth/`); full `nextpnr-ice40` place + route + Fmax reporting on the sequential bitnet variant (134 LCs, 96.59 MHz on iCE40HX1K).
- Hierarchical submodule emission: `instance` and `extern_wire` IR kinds, `RawSubmodule`, `submodules` field on `GateGraph`, `collect_sidecar_files` helper.
- Q-format fields on `Signal` (`q_int_bits`, `q_frac_bits`, `scale`); plus a multidimensional `Signal.shape` / `Gate.output_shape` for `[seq, hidden]` / `[batch, seq, hidden]` packed buses (`Signal.element_count`, `total_bits`, `element_bit_range` helpers; backend honours `total_bits()` for port widths).
- Per-tensor and per-channel symmetric int PTQ helper (`safetensors2verilog.quantize`).
- Hardware block library (`safetensors2verilog/blocks/`): `matmul_seq`, `matmul_stream`, `requantize`, `sigmoid`, `exp`, `tanh`, `silu`, `rsqrt`, `rms_norm`, `layer_norm`, `softmax`, `embedding`, `kv_cache`, `argmax`, `rope`, `attention`. All bit-exact in iverilog.
- `hf_llama` frontend covering the LLaMA-architecture family (Llama / Qwen / Mistral / SmolLM2). End-to-end SmolLM2-135M-Instruct 1-layer inference through verilator. The 30-layer compile path also works: full SmolLM2-135M emits 189 MB of Verilog in ~108s with the streaming lm_head auto-selected (VOCAB=49152 trips the threshold). Multi-layer parser scaling verified at 1, 2, and 4 layers (each parses through `iverilog -E` cleanly: 20 MB / 26 MB / 37 MB respectively).
- ONNX op coverage: `Sigmoid`, `Exp`, `Tanh`, `Softmax`, `Conv` (forward; via the `conv2d` IR primitive), `ConvTranspose` (via the `conv_transpose2d` IR primitive), `Attention` (single-head, no mask; lowered as score-mul + per-row softmax + output-mul), `BatchNormalization` (inference; running stats baked into a per-channel linear), `GroupNormalization` (composed of per-group `layer_norm_block` instances), `LayerNormalization`, `Reshape`, `Concat`, `Split`, `Gather` (initializer + dynamic-index), broadcast `Add` / `Sub` / `Mul`. No ONNX ops remain in a deferred list; coverage is extended by adding a branch to `onnx_topology.py` and a backend lowering in `verilog.py`.
- Sequential bitnet variants: `--handshake` (DONE state holds until `ready_in`, exposes `valid_out`), `--streaming-input` (single `x` + `valid_in` / `ready_out` plus an in_buf register file and a FILL FSM state), `--weight-bram` (per-output writable RAM via the new `ram_writable` IR kind plus `weight_addr_layer` / `weight_addr_output` / `weight_addr_position` / `weight_data` / `weight_we` ports), `--parallelism N` (output_group register cycling through `ceil(out_size/N)` groups; per-(L,j) accumulator updates gated by `group_match`), `--mac-sharing` (per-output storage registers capturing accumulator values at end-of-group; final outputs and inter-layer reads route through storage so synth's resource-sharing pass has the structural cues to share MAC hardware). All combinations compile through `iverilog -g2012`.
- `conv2d` and `conv_transpose2d` IR primitives (`safetensors2verilog.verilog`): combinational 2-D forward and transposed convolution with explicit shape attributes (`in_h`, `in_w`, `in_c`, `out_c`, `kH`, `kW`, `stride`, `pad`, plus `output_padding` for ConvTranspose); both bit-exact in iverilog across multi-case sweeps.
- `ram_writable` IR kind: synchronous-write / asynchronous-read writable RAM with per-cycle `write_en` / `write_addr` / `write_data` and a separate `read_addr`. Vendor synth (Vivado / Quartus / Yosys) infers it as block RAM; `(* ram_style = "block" *)` attribute attached.
- `fp8_e4m3_mul` IR kind: 8-bit fp8 e4m3 multiply emitting fp16 IEEE-754 product (sign XOR, exponent add, 4x4 mantissa multiply, normalize, saturate). `safetensors2verilog.quantize.fp8_e4m3_quantize` / `_dequantize` round-trip fp32 to/from fp8 e4m3 bit patterns with saturating round-to-nearest.
- BRAM chunking helper (`safetensors2verilog.bram_chunk`): `pick_chunking(depth, width, max_bram_bits, max_bram_depth)` and `emit_chunked_rom` split oversized weight ROMs into per-(row_bank, col_bank) BRAM-sized banks with a row-bank mux + column-bank concat. Bit-exact in iverilog across a 32x16 fixture chunked into 4 row-banks.
- Sidecar weight-ROM management (`safetensors2verilog.write_sidecar_files`): three layouts (`flat`, `subdirs` per-module, `tarball`) plus a top-level `manifest.json` mapping submodule -> sidecar file list. CLI flags `--sidecar-layout` and `--sidecar-tarball` plumb it through.
- Multi-output `Frontend.parse_multi`: returns one or more independent `GateGraph` objects, defaulting to `[parse(...)]` for single-output frontends. CLI `--emit-multi DIR` writes one `<top>.v` per returned graph.
- `emit_instantiation_template(graph)`: render a paste-ready Verilog instantiation snippet binding every external port to a same-name parent signal, plus `#(.PARAM(value))` overrides for parameter ports. CLI flag `--emit-instantiation PATH`.
- Native SystemVerilog emit: lowerings honour `EmitContext.target` and emit `always_ff`, `always_comb`, `unique case` directly when target is sv (no longer a post-emit string rewrite for those constructs; the rewrite still converts wire/reg declarations to `logic`).
- `register` lowering bug fix: the `enable` attribute (or 2nd input) was inserted into the emitted `always` block as a raw signal name, bypassing the identifier sanitisation pass that handles dotted IR names. Now the enable goes through `ctx.name(en_signal)` like the data and reset signals.
- PTQ activation-calibration module (`safetensors2verilog.calibration`): `collect_activation_stats` + `derive_requantize_params` + `saturation_summary` + `calibrate_iteratively` (undamped 3-round refinement; converges on synthetic tiny shapes) + `calibrate_iteratively_damped` (per-channel EMA between iterations to smooth the cross-site oscillation that the undamped form exhibits on residual-amplified architectures like SmolLM2). The `hf_llama` frontend accepts the derived per-channel `(muls, shifts)` via `build_llama_graph(..., requantize_params=...)` and via the CLI flag `--calibration <json_path>`. On SmolLM2-135M layer 0 the single-pass form drops underflow at the requantize sites from 85.71% to 0.00% with 0% saturation.
- LLaMA pure-Python int reference (`safetensors2verilog.llama_reference`): `llama_int_reference_one_layer` (bit-stable 1-layer forward pass mirroring the Verilog int chain), `llama_fp32_reference_logits_one_layer` (fp32 gold for diffing), `llama_int_reference_decode_loop` (multi-token autoregressive driver pattern), `compare_argmax_agreement` (per-position next-token agreement metric).
- Real-world safetensors fixtures in the test suite: `tests/test_real_world_fixture.py` round-trips a 2-layer ternary bitnet (8 -> 4 -> 2) and a 4-layer int8 chain (6 -> 5 -> 4 -> 3 -> 2) through compile, Python evaluator, and iverilog bit-exact (16 random input vectors).
- Vendor placement attributes propagated through register lowerings (`vivado_loc`, `quartus_chip_pin`, `lattice_loc`, `synplify_keep`, `generic_attr`).
- Multi-clock SDC with cross-domain false paths and per-port input/output delays (`--emit-sdc`).
- `cross_frontend_equiv_sweep` and `trace_n_cycles` parametric equivalence drivers.
- Tristate `Z` faithfully propagated as `None` through `evaluate_graph` / `step_graph`.
- `emit_sva_assertions` SystemVerilog assertion module (bind-able into the DUT for formal / SVA-aware simulation).
- `bitnet_rom_parity_bits` helper for single-bit-flip detection on ternary weight ROMs.
- `argparse(allow_abbrev=False)` so frontend flags don't prefix-collide with global flags.
- `mypy --strict` per-module overrides on `core` / `analysis` / `transforms` / `synth` / `evaluate`.
- `yosys_calibrated_estimate` wrapper in `examples/lut_ff_estimate.py`.
- 236 unit tests pass; bit-exact iverilog coverage on tanh, conv2d, conv_transpose2d, chunked ROM, fp8 mul, real-world bitnet fixture, and the sequential-bitnet variant combinations including mac_sharing.
