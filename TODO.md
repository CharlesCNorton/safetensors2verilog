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
- iCE40 worked example with measured cell count (`examples/fpga_synth/`).
- Hierarchical submodule emission: `instance` and `extern_wire` IR kinds, `RawSubmodule`, `submodules` field on `GateGraph`, `collect_sidecar_files` helper.
- Q-format fields on `Signal` (`q_int_bits`, `q_frac_bits`, `scale`).
- Per-tensor and per-channel symmetric int PTQ helper (`safetensors2verilog.quantize`).
- Hardware block library (`safetensors2verilog/blocks/`): `matmul_seq`, `matmul_stream`, `requantize`, `sigmoid`, `exp`, `silu`, `rsqrt`, `rms_norm`, `layer_norm`, `softmax`, `embedding`, `kv_cache`, `argmax`, `rope`, `attention`. All bit-exact in iverilog.
- `hf_llama` frontend covering the LLaMA-architecture family (Llama / Qwen / Mistral / SmolLM2). End-to-end SmolLM2-135M-Instruct 1-layer inference through verilator: real text input -> KV cache update -> attention -> MLP -> final_norm -> CPU-side lm_head + argmax -> output token via the HF tokenizer.
- ONNX op coverage: `Sigmoid`, `Exp`, `LayerNormalization`, `Reshape`, `Concat`, `Split`, `Gather` (initializer + dynamic-index), broadcast `Add` / `Sub` / `Mul`.
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
- 175 unit tests + 5 combinational + 5 sequential primitive iverilog tests + 4/4 tiny LLaMA tokens bit-exact + 5/5 layer_norm cases bit-exact + SmolLM2 1-layer end-to-end through verilator.

## Open

### IR and frontend op coverage

1. ONNX `Conv` / `ConvTranspose`. Needs the Conv2D IR primitive (#6) plus the shape system (#7).
2. ONNX `GroupNormalization` and `BatchNormalization`. GroupNorm composes the existing `layer_norm` per-group; BatchNorm needs running-mean / running-var initialiser handling and an inference-only path that bakes them in.
3. ONNX `Softmax` (the standalone op, not the one inside attention) and `Tanh`. The softmax block exists but expects a packed K-element input bus; Tanh needs a LUT block (mirror of `sigmoid_block` with tanh contents).
4. ONNX `Attention`. Composite of softmax + batched matmul; depends on #2 and #7. The `hf_llama` frontend already covers transformer-architecture attention end-to-end via the `attention_step_block`; this item is the ONNX-graph-driven path for non-LLaMA-shaped attention.
5. Sequential bitnet `--parallelism N`. Time-multiplex outputs (N MAC units, `ceil(out_size[L] / N)` output groups). Architecture in `docs/DEFERRED.md`.
6. Sequential bitnet `--streaming-input`. Single `x` port + `valid_in` / `ready_out` handshake plus an internal `in_buf` register file. Architecture in `docs/DEFERRED.md`.
7. Sequential bitnet `--handshake`. Full `valid_out` / `ready_in` protocol on the output side; 4-state FSM with `VALID_WAITING`. Architecture in `docs/DEFERRED.md`.
8. Sequential bitnet `--weight-bram`. Per-output writable BRAMs plus `weight_addr` / `weight_data` / `weight_we` ports for runtime weight reload. Architecture in `docs/DEFERRED.md`.
9. `Conv2D` IR primitive with explicit shape attributes (`in_h`, `in_w`, `in_c`, `out_c`, `kH`, `kW`, `stride`, `pad`).
10. Batched / sequence-dimension primitives in the IR. `Signal` widens to carry a multidimensional shape so transformer ops can express their natural `[seq, hidden]` / `[batch, seq, hidden]` axes instead of packed flat buses.
11. Real-world safetensors fixture in CI. Pull a tiny canonical model and round-trip it through every applicable frontend on every CI run.
12. Native SystemVerilog emitter. The current `target=sv` is a string-rewrite over the Verilog output; a native emitter would handle `interface` blocks, packed `struct` types, and `unique case`.
13. `Frontend.parse` returns a single `GateGraph`. Multi-output workflows (one frontend emitting several top modules with shared submodule types) aren't expressible.
14. No structured way to express "this frontend generates module X parameterised by Y, which the user instantiates N times in their wrapper."

### Backend and scale

15. Source-level RAM chunking. Split `V*H` embeddings (and the lm_head row, once streaming-lm_head lands at scale) into N column-banks of B rows each, sized to fit a single Xilinx 7-series 36 Kbit BRAM. Without this `synth_xilinx` rejects the SmolLM2-scale embedding RAM with "no valid mapping found."
16. Sidecar weight-ROM file management. SmolLM2 emits ~5k hex files per layer; a full 30-layer compile lands ~155k files in one directory. Replace with a manifest + tarball, per-module subdirectories, or one concatenated file with offset addressing.

### LM bring-up

17. SmolLM2 30-layer scale-up. Drop `num_layers_override=1` from the frontend call, wire the streaming lm_head through the auto-select threshold, capture the full forward pass through verilator.
18. PTQ activation calibration. The 1-layer SmolLM2 forward pass produced saturated activations because the frontend uses heuristic per-layer shifts. Run a small calibration set (~32 sequences from C4 / WikiText), capture per-block activation distributions, fill the `requantize_block` calls' `muls` / `shifts` lists with calibrated values.
19. Bit-exact Python reference for `LlamaForCausalLM` forward pass at SmolLM2 shape (analogous to `_play/tiny_llama_e2e.py`'s reference for the synthetic shape). Layer-by-layer comparison against the Verilog output.
20. Multi-token autoregressive driver. The current inference harness drives one token; the KV cache persists across calls inside the attention block so multi-token decode is a Python loop that drives token N at position N.
21. Compare Verilog SmolLM2 next-token predictions against `transformers` fp32 reference. Pass criterion: `argmax(verilog_logits) == argmax(transformers_logits)` on most prompts (allowing for known PTQ accuracy loss on the tail).
22. fp8 native primitives as an alternative to int8 PTQ. Hugging Face ships some models pre-quantised to fp8; the current frontend dequantises fp8 → fp32 → int8, losing precision twice. Native fp8 multipliers + fp8/fp16 accumulators preserve the upstream quant.
23. Vivado / Quartus synth + place + route on the chunked design from #15. Captures real utilization (LUT, FF, BRAM, DSP) and Fmax on a Versal AI Edge / Stratix 10 MX / Alveo U250 target.

Categories (rough): 1-4 are ONNX op coverage; 5-8 are sequential-bitnet variants whose architecture is sketched in `docs/DEFERRED.md`; 9-10 and 12-14 are IR generality; 11 is a CI fixture; 15-16 are backend scaling knobs the LM compile depends on; 17-23 are LM bring-up to fp32-matching prediction quality.
