# Worked example: threshold network → FPGA synthesis

End-to-end demonstration that a threshold-logic safetensors compiles to a real FPGA target.

## Pipeline

1. Extract a single circuit (8-bit ripple-carry adder) from the canonical 8bit-threshold-computer file.
2. Compile it through `safetensors2verilog` with packed-bus port emission.
3. Run Yosys + iCE40 tech-mapping (open-source, no Lattice tools needed).
4. Read back actual LUT utilization on the iCE40HX1K target.

## Reproduce

Requires Yosys ≥ 0.40 (OSS CAD Suite is the easiest source).

```bash
# 1. Extract + compile
python -m safetensors2verilog \
  ../../neural_alu8.safetensors \
  --frontend threshold_logic \
  --circuit arithmetic.ripplecarry8bit \
  --pack-buses \
  --equiv-check \
  -o rc8.v --top rc8

# 2. Synthesize for iCE40
yosys -s synth_rc8_ice40.ys
```

`synth_rc8_ice40.ys`:

```text
read_verilog rc8.v
hierarchy -check -top rc8
synth_ice40 -top rc8
stat
write_blif rc8_ice40.blif
```

## Measured result (iCE40HX1K target, Yosys 0.63)

```
=== rc8 ===

       21 wires
       47 wire bits
       21 public wires
       47 public wire bits
       11 ports
       25 port bits
       15 cells
       15   SB_LUT4
```

**15 LUT4s** for the full 8-bit add. Carry-chain merging by ABC collapses all 72 threshold gates of the source threshold network. The iCE40HX1K has 1280 LUTs, so this consumes 1.2% of the smallest hobbyist iCE40.

The same Verilog is also accepted by Vivado (Xilinx) and Quartus (Intel). A Vivado run would replace the Yosys script with:

```tcl
# Vivado .tcl
read_verilog rc8.v
synth_design -top rc8 -part xc7a35tcpg236-1
report_utilization
```

## Closing the loop: iverilog cross-check

The `--equiv-check` flag included above runs a Python-evaluator vs. iverilog cross-check on the emitted Verilog:

```
$ python -m safetensors2verilog ... --equiv-check ...
--equiv-check: PASS 65536 cases (0 fail)
```

All 2^16 input combinations of the 8-bit adder match between the in-memory threshold-network evaluator and the iverilog simulation of the synthesizable Verilog.

## Where to go next

- Larger circuits: drop `--circuit arithmetic.ripplecarry8bit` and let it emit the full ALU8 (~56k threshold gates → still a few thousand LUT4s).
- BitNet linear layers: see `../bitnet_linear/` for ternary `nn.Linear` to Verilog.
- Full-CPU synthesis: extract via `--circuit` with a CPU-level prefix, then add `--skip-memory` plus `--emit-bram-template` to swap the threshold-network memory out for vendor BRAM.
