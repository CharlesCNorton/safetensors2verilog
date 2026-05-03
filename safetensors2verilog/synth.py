"""Run Yosys+ABC on emitted Verilog and report gate-level statistics.

This is a thin wrapper. Yosys must be on PATH (or supplied via the
``yosys`` argument). The output captures cell counts after ABC tech
mapping to a basic gate library (AND/OR/NAND/NOR/XOR/XNOR/NOT/DFF), an
upper-bound LUT4 estimate, and the longest combinational depth as
reported by Yosys's ``stat`` command. None of this requires an FPGA
toolchain — just open-source Yosys.

Public entry point:

  run_synth(verilog_path, top, yosys=None, lut_size=4) -> dict

Returns keys:
  cells           total cell count after tech-map
  cells_by_kind   {AND, OR, NAND, NOR, XOR, XNOR, NOT, DFF: count}
  lut_estimate    rough LUT4 count (cells of fanin <= lut_size)
  yosys_stdout    full Yosys log (for users who want the raw output)
  yosys_exit_code int

Failure modes raise :class:`RuntimeError` with the Yosys output included.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


_KIND_PATTERN = re.compile(r"^\s*(\d+)\s+\$_(\w+?)_(?:\s|$)")


def _find_yosys(explicit: str | None) -> str:
    if explicit:
        return explicit
    which = shutil.which("yosys")
    if which:
        return which
    raise RuntimeError(
        "yosys executable not found on PATH; pass yosys='/path/to/yosys' "
        "or install OSS CAD Suite / yosys."
    )


def _build_script(verilog_path: Path, top: str) -> str:
    return (
        f"read_verilog {verilog_path.as_posix()}\n"
        f"hierarchy -check -top {top}\n"
        "synth -top {top}\n"
        "abc -g AND,OR,XOR,NAND,NOR,XNOR\n"
        "stat\n"
    ).replace("{top}", top)


def run_synth(
    verilog_path: str | Path,
    top: str,
    *,
    yosys: str | None = None,
    lut_size: int = 4,
    timeout: float = 600.0,
) -> dict:
    """Run Yosys+ABC and parse the ``stat`` output for gate counts.

    On Windows, OSS CAD Suite ships ``yosys.exe`` plus several DLLs that
    only resolve when the suite's ``environment.bat`` has populated
    ``PATH``; pass ``yosys`` explicitly when running outside that shell.
    """
    verilog_path = Path(verilog_path).resolve()
    if not verilog_path.exists():
        raise FileNotFoundError(verilog_path)
    yosys_bin = _find_yosys(yosys)

    # OSS CAD Suite ships yosys.exe alongside DLLs in the same directory;
    # if the user passed a path inside that suite layout, ensure that
    # directory is on PATH before we launch Yosys so its DLLs resolve.
    env = os.environ.copy()
    yosys_dir = Path(yosys_bin).parent
    extra_paths: list[str] = [str(yosys_dir)]
    # OSS CAD Suite layout: <root>/bin/yosys.exe with <root>/lib alongside.
    sibling_lib = yosys_dir.parent / "lib"
    if sibling_lib.is_dir():
        extra_paths.append(str(sibling_lib))
    cur = env.get("PATH", "")
    for p in extra_paths:
        if p not in cur:
            cur = p + os.pathsep + cur
    env["PATH"] = cur

    with tempfile.TemporaryDirectory(prefix="s2v_synth_") as tmpdir:
        script = Path(tmpdir) / "synth.ys"
        script.write_text(_build_script(verilog_path, top), encoding="utf-8")
        # Don't pass -q: it suppresses the `stat` output we need to parse.
        proc = subprocess.run(
            [yosys_bin, "-s", str(script)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )

    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"yosys failed with exit code {proc.returncode}.\n{log}"
        )

    cells_by_kind: dict[str, int] = {}
    cells = 0
    in_stat = False
    stat_block_re = re.compile(rf"^=== {re.escape(top)} ===")
    cells_total_re = re.compile(r"^\s*(\d+)\s+cells\s*$")

    for line in log.splitlines():
        if stat_block_re.match(line):
            in_stat = True
            continue
        if not in_stat:
            continue
        m_total = cells_total_re.match(line)
        if m_total:
            cells = int(m_total.group(1))
            continue
        m = _KIND_PATTERN.match(line)
        if m:
            count = int(m.group(1))
            kind = m.group(2)
            cells_by_kind[kind] = count
        elif line.startswith("End of script") or line.startswith("==="):
            in_stat = False

    if cells == 0 and cells_by_kind:
        cells = sum(cells_by_kind.values())

    # LUT4 estimate: every multi-input combinational cell collapses to one
    # LUT of size <= fanin, capped at lut_size. Without per-cell fanin we
    # use the conservative rule "every ABC cell is one LUT" plus DFFs as
    # registers.
    lut_estimate = sum(
        n for k, n in cells_by_kind.items() if not k.startswith("DFF")
    )

    return {
        "cells": cells,
        "cells_by_kind": cells_by_kind,
        "lut_estimate": lut_estimate,
        "lut_size": lut_size,
        "yosys_stdout": log,
        "yosys_exit_code": proc.returncode,
    }


def format_synth_report(stats: dict) -> str:
    """Human-readable synthesis-stats summary."""
    lines = [
        f"cells       : {stats['cells']}",
        f"LUT{stats['lut_size']} (est) : {stats['lut_estimate']}",
    ]
    if stats["cells_by_kind"]:
        kinds = ", ".join(
            f"{k}={v}" for k, v in sorted(
                stats["cells_by_kind"].items(), key=lambda kv: -kv[1]
            )
        )
        lines.append(f"by kind     : {kinds}")
    return "\n".join(lines)
