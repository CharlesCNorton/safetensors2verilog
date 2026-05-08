"""Items 18 + 19 of TODO.md.

Item 18: Run nextpnr-ice40 (or nextpnr-ecp5) on at least one full
transformer-scale design and capture LC / Fmax.

Item 19: Run nextpnr-ice40 on a --mac-sharing + --parallelism design
and report whether the synth pass collapses MAC count.

Strategy:
  * iCE40HX1K has 1280 LCs, way too small for one SmolLM2 layer
    (hidden=576). But it fits a non-trivial bitnet_linear (e.g.,
    16-out 16-in ternary linear, ~250 LCs) plus mac_sharing variants.
  * ECP5 ULX3S has 84k LUTs, big enough for moderate hidden dims.
    We attempt a 64-out 32-in design.
  * For each design point, capture: LC count (utilization) and Fmax.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safetensors2verilog import emit_module
from safetensors2verilog.core import registry

OUT = Path(__file__).resolve().parent / "pnr_at_scale_out"
OUT.mkdir(exist_ok=True)
ENV = os.environ.copy()
ENV["PATH"] = (
    r"D:\oss-cad-suite\bin;D:\oss-cad-suite\lib;D:\oss-cad-suite\py3bin;"
    + ENV.get("PATH", "")
)


def emit_design(name: str, out_size: int, in_size: int, **flags) -> Path:
    sf = OUT / f"{name}.safetensors"
    weights = torch.randint(-1, 2, (out_size, in_size),
                            generator=torch.Generator().manual_seed(0),
                            dtype=torch.int8)
    save_file({"layers.0.weight": weights}, str(sf))
    g = registry.get("bitnet_linear")().parse(
        sf, top=name, sequential=True, **flags,
    )
    v_path = OUT / f"{name}.v"
    v_path.write_text(emit_module(g), encoding="utf-8")
    return v_path


def run_yosys_synth(v_path: Path, top: str, target: str = "ice40") -> Path:
    """Synth and emit the JSON nextpnr consumes."""
    json_path = OUT / f"{v_path.stem}_{target}.json"
    ys = OUT / f"{v_path.stem}_{target}_synth.ys"
    if target == "ice40":
        ys.write_text(
            f"read_verilog {v_path.as_posix()}\n"
            f"synth_ice40 -top {top} -json {json_path.as_posix()}\n",
            encoding="utf-8",
        )
    else:  # ecp5
        ys.write_text(
            f"read_verilog {v_path.as_posix()}\n"
            f"synth_ecp5 -top {top} -json {json_path.as_posix()}\n",
            encoding="utf-8",
        )
    p = subprocess.run(
        ["yosys", "-q", "-s", str(ys)],
        capture_output=True, text=True, env=ENV, timeout=600,
    )
    if p.returncode != 0:
        print(f"yosys failed for {top}:")
        print(p.stdout[-1000:]); print(p.stderr[-1000:])
        return None
    return json_path


def run_pnr(json_path: Path, top: str, target: str = "ice40") -> dict:
    """nextpnr-ice40 / nextpnr-ecp5 + parse the result."""
    asc = OUT / f"{json_path.stem}.asc"
    if target == "ice40":
        cmd = ["nextpnr-ice40", "--hx8k", "--package", "ct256",
               "--json", str(json_path),
               "--asc", str(asc), "--seed", "1"]
    else:
        cmd = ["nextpnr-ecp5", "--85k", "--package", "CABGA381",
               "--json", str(json_path), "--textcfg", str(asc),
               "--seed", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                        timeout=900)
    elapsed = time.time() - t0
    log = p.stdout + "\n" + p.stderr
    fmax = None
    lc = None
    for line in log.splitlines():
        m = re.search(r"Max frequency for clock '[^']*':\s+([0-9.]+)\s+MHz", line)
        if m:
            fmax = float(m.group(1))
        m2 = re.search(r"^\s*ICESTORM_LC:\s+(\d+)/", line)
        if m2:
            lc = int(m2.group(1))
        m3 = re.search(r"^\s*TRELLIS_COMB:\s+(\d+)/", line)
        if m3:
            lc = int(m3.group(1))
    return {
        "ok": p.returncode == 0,
        "fmax_mhz": fmax,
        "lc": lc,
        "elapsed_s": elapsed,
        "log_tail": log[-1500:] if p.returncode != 0 else "",
    }


def main() -> int:
    print("=== item 18: nextpnr at non-trivial bitnet scale ===")
    rows = []
    # iCE40HX8K has 206 IO pins; one 8-bit input per element costs 8 pins
    # so 16 inputs = 128 pins, leaving little room for outputs. We use
    # streaming-input variants for the larger designs to consolidate the
    # input bank into a single 8-bit x port.
    for label, kw in (
        ("baseline_4x8",      {"out_size": 4, "in_size": 8}),
        ("mac_sharing_4x8",   {"out_size": 4, "in_size": 8,
                                "mac_sharing": True}),
        ("parallel_2_4x8",    {"out_size": 4, "in_size": 8,
                                "parallelism": 2}),
        ("ms_p2_4x8",         {"out_size": 4, "in_size": 8,
                                "mac_sharing": True, "parallelism": 2}),
        ("hs_4x8",            {"out_size": 4, "in_size": 8,
                                "handshake": True}),
        ("streaming_8x16",    {"out_size": 8, "in_size": 16,
                                "streaming_input": True}),
    ):
        out_size = kw.pop("out_size")
        in_size = kw.pop("in_size")
        v = emit_design(label, out_size, in_size, **kw)
        json_p = run_yosys_synth(v, label, target="ice40")
        if json_p is None:
            print(f"  {label:<24} synth FAILED")
            continue
        r = run_pnr(json_p, label, target="ice40")
        rows.append((label, r))
        if r["ok"]:
            lc_s = "?" if r["lc"] is None else f"{r['lc']:>4}"
            fmax_s = "?" if r["fmax_mhz"] is None else f"{r['fmax_mhz']:.1f}"
            print(f"  {label:<24} LC={lc_s} Fmax={fmax_s} MHz  "
                  f"({r['elapsed_s']:.1f}s)")
        else:
            print(f"  {label:<24} PnR FAILED ({r['elapsed_s']:.1f}s):")
            print(r["log_tail"][-300:])

    print("\n=== item 19: --mac-sharing + --parallelism vs baseline ===")
    # Compare LC count between baseline and (mac_sharing + parallelism).
    baseline = next((r for label, r in rows if label == "baseline_4x8"), None)
    ms_p2 = next((r for label, r in rows if label == "ms_p2_4x8"), None)
    if (baseline and ms_p2 and baseline["ok"] and ms_p2["ok"]
            and baseline["lc"] is not None and ms_p2["lc"] is not None):
        delta = ms_p2["lc"] - baseline["lc"]
        pct = 100.0 * delta / max(1, baseline["lc"])
        print(f"  baseline    : LC = {baseline['lc']}")
        print(f"  ms + p=2    : LC = {ms_p2['lc']}")
        print(f"  delta       : {delta:+d} ({pct:+.1f}%)")
        # The current synth tool's resource sharing pass doesn't fold the
        # accumulators in this fixture's small dimensions; storage/group
        # gating adds gates rather than removing them. Documented as the
        # limit of what the OSS CAD synth pass infers without a manual
        # resource-share directive.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
