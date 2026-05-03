"""Tests for the dependency-closure subset extractor."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from safetensors2verilog.extract import closure_keys, extract_subset


def _build_fixture(tmp_path: Path) -> Path:
    """Three circuits where C depends on B which depends on A.

    A.gate -> reads $x, $y         (no producer dependency)
    B.gate -> reads A.gate         (depends on A)
    C.gate -> reads B.gate         (depends on B, transitively on A)
    plus an unrelated D.gate       (must NOT be pulled in)
    """
    tensors = {
        # A.gate: 2-input AND-like
        "A.gate.weight": torch.tensor([1, 1], dtype=torch.int64),
        "A.gate.bias":   torch.tensor([-1], dtype=torch.int64),
        "A.gate.inputs": torch.tensor([10, 11], dtype=torch.int64),  # $x, $y
        # B.gate: reads A.gate
        "B.gate.weight": torch.tensor([1], dtype=torch.int64),
        "B.gate.bias":   torch.tensor([0], dtype=torch.int64),
        "B.gate.inputs": torch.tensor([20], dtype=torch.int64),       # A.gate
        # C.gate: reads B.gate
        "C.gate.weight": torch.tensor([1], dtype=torch.int64),
        "C.gate.bias":   torch.tensor([0], dtype=torch.int64),
        "C.gate.inputs": torch.tensor([21], dtype=torch.int64),       # B.gate
        # D.gate: unrelated, reads $x
        "D.gate.weight": torch.tensor([1], dtype=torch.int64),
        "D.gate.bias":   torch.tensor([0], dtype=torch.int64),
        "D.gate.inputs": torch.tensor([10], dtype=torch.int64),
    }
    signal_registry = {
        10: "$x", 11: "$y",
        20: "A.gate", 21: "B.gate", 22: "C.gate", 23: "D.gate",
    }
    md = {"signal_registry": json.dumps({str(k): v for k, v in signal_registry.items()})}
    out = tmp_path / "fix.safetensors"
    save_file(tensors, str(out), metadata=md)
    return out


def test_closure_pulls_transitive_deps(tmp_path):
    src = _build_fixture(tmp_path)
    dst = tmp_path / "out.safetensors"
    stats = extract_subset(src, ["C.gate"], dst, quiet=True)
    assert stats["closure_gates"] == 3   # C, B, A
    assert stats["seed_gates"] == 1      # only C matches the prefix

    # The output safetensors should contain A, B, C tensors but not D.
    from safetensors.torch import load_file
    out = load_file(str(dst))
    keys = set(out.keys())
    assert "A.gate.weight" in keys
    assert "B.gate.weight" in keys
    assert "C.gate.weight" in keys
    assert not any(k.startswith("D.gate") for k in keys)


def test_closure_prefix_matches_dotted_descendants(tmp_path):
    src = _build_fixture(tmp_path)
    dst = tmp_path / "out2.safetensors"
    # 'A' should match 'A.gate' (dotted descendant)
    stats = extract_subset(src, ["A"], dst, quiet=True)
    assert stats["seed_gates"] == 1
    assert stats["closure_gates"] == 1   # A has no further deps


def test_closure_unknown_prefix_raises(tmp_path):
    src = _build_fixture(tmp_path)
    dst = tmp_path / "out3.safetensors"
    import pytest
    with pytest.raises(ValueError, match="no gates matched"):
        extract_subset(src, ["Z.nonexistent"], dst, quiet=True)


def test_closure_keys_direct(tmp_path):
    src = _build_fixture(tmp_path)
    from safetensors.torch import load_file
    from safetensors import safe_open
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata()
    sr = {int(k): v for k, v in json.loads(meta["signal_registry"]).items()}
    tensors = load_file(str(src))
    keep, stats = closure_keys(tensors, sr, ["B.gate"])
    assert stats["closure_gates"] == 2   # B + A
    assert "A.gate.weight" in keep
    assert "B.gate.weight" in keep
    assert "C.gate.weight" not in keep
    assert "D.gate.weight" not in keep


def test_schema_version_unsupported_raises(tmp_path):
    """Files declaring an unknown schema_version must be rejected."""
    import json
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "future.safetensors"
    save_file(
        {"g.weight": torch.tensor([1, 1], dtype=torch.int64),
         "g.bias":   torch.tensor([0], dtype=torch.int64),
         "g.inputs": torch.tensor([10, 11], dtype=torch.int64)},
        str(src),
        metadata={
            "schema_version": "999",
            "signal_registry": json.dumps({"10": "$x", "11": "$y"}),
        },
    )
    from safetensors2verilog.frontends.threshold_logic import ThresholdLogicFrontend
    import pytest
    with pytest.raises(ValueError, match="unsupported schema_version"):
        ThresholdLogicFrontend().parse(src)


def test_schema_version_v1_explicit_accepted(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "v1.safetensors"
    save_file(
        {"g.weight": torch.tensor([1, 1], dtype=torch.int64),
         "g.bias":   torch.tensor([0], dtype=torch.int64),
         "g.inputs": torch.tensor([10, 11], dtype=torch.int64)},
        str(src),
        metadata={
            "schema_version": "1",
            "signal_registry": json.dumps({"10": "$x", "11": "$y"}),
        },
    )
    from safetensors2verilog.frontends.threshold_logic import ThresholdLogicFrontend
    graph = ThresholdLogicFrontend().parse(src)
    assert len(graph.gates) == 1
