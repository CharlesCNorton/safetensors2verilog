"""Estimate LUT and flip-flop counts per variant.

LUT model: each gate compiles to a popcount-and-compare. For fan-in F
(sum of |weight_i|) the gate consumes roughly:
    F <=  6:  1 LUT      (one LUT6 absorbs up to 6 binary inputs)
    F <= 12:  2 LUTs
    F <= 24:  4 LUTs
    F <= 64: 12 LUTs
    F  > 64: ceil(F/5) LUTs (adder tree)

This is an order-of-magnitude estimate calibrated to typical Xilinx /
Lattice / Intel FPGA mappings; exact numbers depend on the tool and
target. The model under-counts when many bias-only adders share a
common subexpression (synthesis CSE usually catches that) and over-
counts when the comparator collapses to a single LUT regardless of
sum width (which Vivado often does).

FF model: the threshold network itself is purely combinational. To run
the CPU sequentially you need a state register sized to the manifest's
state vector:

    state_bits = PC + IR + 4*REG_BITS + FLAG_BITS + SP + CTRL + MEM*8

With CPU memory mapped to threshold-gated cells (the `memory.write.*`
family), every memory bit is already absorbed into the combinational
graph and you'd register state externally; FF count = state_bits.
Mapping memory to a vendor BRAM block instead pulls those bits out of
the LUT count entirely.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_VARIANTS_DIR = Path("D:/8bit-threshold-computer/variants")


def luts_for_fanin(f: int) -> int:
    if f <= 6: return 1
    if f <= 12: return 2
    if f <= 24: return 4
    if f <= 64: return 12
    return max(12, (f + 4) // 5)


def gate_count_and_lut_estimate(path: Path):
    gates = 0
    total_fanin = 0
    luts = 0
    fanin_buckets = {"<=6": 0, "7-12": 0, "13-24": 0, "25-64": 0, ">64": 0}

    with safe_open(str(path), framework="pt") as f:
        manifest = {}
        for k in f.keys():
            if k.startswith("manifest.") and f.get_tensor(k).numel() == 1:
                manifest[k.split(".", 1)[1]] = int(f.get_tensor(k).item())

        for name in f.keys():
            if not name.endswith(".weight"):
                continue
            if name.startswith("manifest."):
                continue
            t = f.get_tensor(name)
            if t.dim() == 0:
                continue
            tf = t.float()
            # For packed (multi-row) tensors, each row is one gate.
            # The bias is one scalar per row; weights are concatenated.
            bias_key = name[: -len(".weight")] + ".bias"
            try:
                rows = f.get_tensor(bias_key).numel()
            except Exception:
                rows = 1

            if rows > 1 and tf.numel() % rows == 0:
                per_row = tf.view(rows, -1)
                row_fanins = per_row.abs().sum(dim=1).long().tolist()
            else:
                row_fanins = [int(tf.abs().sum().item())]

            for fanin in row_fanins:
                gates += 1
                total_fanin += fanin
                luts += luts_for_fanin(fanin)
                if fanin <= 6: fanin_buckets["<=6"] += 1
                elif fanin <= 12: fanin_buckets["7-12"] += 1
                elif fanin <= 24: fanin_buckets["13-24"] += 1
                elif fanin <= 64: fanin_buckets["25-64"] += 1
                else: fanin_buckets[">64"] += 1

    return {
        "gates": gates,
        "avg_fanin": total_fanin / gates if gates else 0,
        "luts_logic": luts,
        "fanin_dist": fanin_buckets,
        "manifest": manifest,
    }


def state_bits(m: dict) -> int:
    pc = m.get("pc_width", m.get("addr_bits", 0))
    ir = 16
    regs = 4 * 8
    flags = 4
    sp = pc
    ctrl = 4
    mem_bits = m.get("memory_bytes", 0) * 8
    return pc + ir + regs + flags + sp + ctrl + mem_bits


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variants_dir", nargs="?", type=Path, default=DEFAULT_VARIANTS_DIR,
        help=f"directory containing .safetensors variants "
             f"(default: {DEFAULT_VARIANTS_DIR})",
    )
    args = parser.parse_args()
    variants_dir = args.variants_dir

    if not variants_dir.exists():
        print(f"variants dir not found: {variants_dir}", file=sys.stderr)
        return 2

    files = sorted(variants_dir.glob("*.safetensors"))
    print(f"{'variant':<42} {'gates':>10} {'avg_fan':>8} {'LUTs (logic)':>14} {'FFs (state)':>12}  {'fan-in dist <=6/7-12/13-24/25-64/>64':>40}")
    print("-" * 140)
    for f in files:
        info = gate_count_and_lut_estimate(f)
        sb = state_bits(info["manifest"])
        d = info["fanin_dist"]
        dist = f"{d['<=6']}/{d['7-12']}/{d['13-24']}/{d['25-64']}/{d['>64']}"
        print(f"{f.name:<42} {info['gates']:>10,} {info['avg_fanin']:>8.2f} "
              f"{info['luts_logic']:>14,} {sb:>12,}  {dist:>40}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
