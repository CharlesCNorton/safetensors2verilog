"""Tests for synth.py — the Yosys+ABC wrapper.

Most users won't have Yosys installed; these tests exercise the parser
on captured Yosys output rather than running Yosys live. One smoke test
runs Yosys when it's on PATH and is otherwise skipped.
"""
from __future__ import annotations

import shutil

import pytest

SAMPLE_YOSYS_LOG = """
1. Executing Verilog-2005 frontend.
...
=== rc8 ===

        +----------Local Count, excluding submodules.
        |
      110 wires
      110 wire bits
       29 public wires
       29 public wire bits
       25 ports
       25 port bits
       37 cells
        1   $_AND_
       21   $_NAND_
       15   $_XOR_

End of script. Logfile hash: deadbeef
"""


def test_parse_stat_block(monkeypatch):
    """Patched run that returns a captured log; verify the parse path."""
    from safetensors2verilog import synth

    def fake_run(*a, **kw):
        class P:
            returncode = 0
            stdout = SAMPLE_YOSYS_LOG
            stderr = ""
        return P()

    monkeypatch.setattr(synth, "_find_yosys", lambda explicit: "yosys")
    monkeypatch.setattr(synth.subprocess, "run", fake_run)

    # We need a real file path so the existence check passes.
    import os
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".v", delete=False)
    f.write(b"module rc8(); endmodule\n")
    f.close()
    try:
        stats = synth.run_synth(f.name, top="rc8", yosys="yosys")
    finally:
        os.unlink(f.name)

    assert stats["cells"] == 37
    assert stats["cells_by_kind"] == {"AND": 1, "NAND": 21, "XOR": 15}
    assert stats["lut_estimate"] == 37


def test_format_synth_report():
    from safetensors2verilog.synth import format_synth_report
    text = format_synth_report({
        "cells": 37,
        "cells_by_kind": {"AND": 1, "NAND": 21, "XOR": 15},
        "lut_estimate": 37,
        "lut_size": 4,
    })
    assert "cells" in text
    assert "37" in text
    assert "NAND=21" in text


@pytest.mark.skipif(
    shutil.which("yosys") is None,
    reason="yosys not on PATH",
)
def test_live_yosys_smoke(tmp_path):
    """Smoke test: when yosys is installed, run it on a tiny module."""
    from safetensors2verilog.synth import run_synth
    v = tmp_path / "tiny.v"
    v.write_text(
        "module tiny(input wire a, input wire b, output wire y);\n"
        "  assign y = a & b;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    stats = run_synth(str(v), top="tiny")
    assert stats["cells"] >= 1
