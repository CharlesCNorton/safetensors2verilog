"""Tests for the CLI: frontend registration, option discovery, BRAM emission."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


def _make_simple_safetensors(path: Path):
    sr = {"0": "#0", "1": "#1", "2": "$a", "3": "$b"}
    save_file(
        {
            "and_ab.weight": torch.tensor([1, 1], dtype=torch.int8),
            "and_ab.bias":   torch.tensor([-2], dtype=torch.int8),
            "and_ab.inputs": torch.tensor([2, 3], dtype=torch.int64),
        },
        str(path),
        metadata={"signal_registry": json.dumps(sr)},
    )


def _run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "safetensors2verilog", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_list_frontends_includes_threshold_logic_and_bitnet():
    r = _run_cli("--list-frontends")
    assert r.returncode == 0
    assert "threshold_logic" in r.stdout
    assert "bitnet_linear" in r.stdout


def test_list_frontend_options_describes_threshold_flags():
    r = _run_cli("--list-frontend-options", "threshold_logic")
    assert r.returncode == 0
    assert "skip-memory" in r.stdout
    assert "strict" in r.stdout


def test_list_frontend_options_describes_bitnet_flags():
    r = _run_cli("--list-frontend-options", "bitnet_linear")
    assert r.returncode == 0
    assert "activation-bits" in r.stdout
    assert "layer-prefix" in r.stdout


def test_compile_threshold_logic_to_file():
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.v"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op), "--top", "and_only")
        assert r.returncode == 0, r.stderr
        text = op.read_text()
        assert "module and_only" in text
        assert "((_a + _b) >= 2)" in text


def test_runs_via_python_dash_m_from_unrelated_cwd():
    """Regression for the namespace-package collision bug: the CLI must
    register frontends regardless of where it is invoked from."""
    with tempfile.TemporaryDirectory() as td:
        # Drop a sibling directory named 'safetensors2verilog' (no init)
        # to mimic the original namespace-package collision.
        sibling = Path(td) / "safetensors2verilog"
        sibling.mkdir()
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.v"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op), cwd=td)
        assert r.returncode == 0, r.stderr
        assert op.exists()


def test_per_frontend_flag_is_accepted():
    """The CLI re-parses with the chosen frontend's options registered."""
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.v"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op), "--frontend", "threshold_logic",
                     "--skip-memory")
        assert r.returncode == 0, r.stderr


def test_emit_sdc_writes_starter_constraints():
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.v"
        sdc = Path(td) / "out.sdc"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op),
                     "--emit-sdc", str(sdc),
                     "--sdc-period-ns", "8.0")
        assert r.returncode == 0, r.stderr
        assert sdc.exists()
        text = sdc.read_text()
        # No clk in this trivial graph (combinational), so SDC notes that
        assert "no clk port detected" in text or "create_clock" in text


def test_target_sv_emits_systemverilog():
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.sv"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op), "--target", "sv")
        assert r.returncode == 0, r.stderr
        text = op.read_text()
        assert "logic" in text


def test_dry_run_emits_no_output_file():
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.safetensors"
        op = Path(td) / "out.v"
        _make_simple_safetensors(ip)
        r = _run_cli(str(ip), "-o", str(op), "--dry-run")
        assert r.returncode == 0, r.stderr
        assert not op.exists()
