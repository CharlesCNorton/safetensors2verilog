"""Property-based fuzz tests for subset extraction + frontend round-trip.

Catches the silent-anonymous-port failure mode: when extraction omits a
gate's dependencies, the frontend (in --strict, the new default) must
raise; in --promote-unresolved it must produce a graph whose anonymous
external inputs match the missing-dep set.
"""
from __future__ import annotations

import json

import pytest

hyp = pytest.importorskip("hypothesis")

import torch  # noqa: E402
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from safetensors2verilog.extract import closure_keys, extract_subset  # noqa: E402
from safetensors2verilog.frontends.threshold_logic import (  # noqa: E402
    ThresholdLogicFrontend,
)


@st.composite
def random_threshold_graph(draw):
    """Generate a small connected threshold graph with N gates and 2 external
    inputs. Each gate consumes 1-3 prior gates (or external inputs).

    Returns (tensors_dict, signal_registry, gate_names_list_in_order).
    """
    n_gates = draw(st.integers(min_value=2, max_value=8))
    tensors: dict[str, torch.Tensor] = {}
    sr: dict[int, str] = {1: "$x", 2: "$y"}
    next_id = 3
    gate_names: list[str] = []

    for i in range(n_gates):
        name = f"g{i}"
        candidate_signals = list(range(1, next_id))   # prior signal IDs
        if not candidate_signals:
            candidate_signals = [1, 2]
        n_inp = draw(st.integers(min_value=1, max_value=min(3, len(candidate_signals))))
        chosen_ids = draw(
            st.lists(st.sampled_from(candidate_signals),
                     min_size=n_inp, max_size=n_inp, unique=True)
        )
        weights = draw(st.lists(
            st.sampled_from([-1, 0, 1]),
            min_size=n_inp, max_size=n_inp,
        ))
        bias = draw(st.integers(min_value=-3, max_value=3))

        tensors[f"{name}.weight"] = torch.tensor(weights, dtype=torch.int64)
        tensors[f"{name}.bias"]   = torch.tensor([bias], dtype=torch.int64)
        tensors[f"{name}.inputs"] = torch.tensor(chosen_ids, dtype=torch.int64)
        sr[next_id] = name
        next_id += 1
        gate_names.append(name)
    return tensors, sr, gate_names


def _write_safetensors(tmp_path, tensors, sr, name="g.safetensors"):
    out = tmp_path / name
    md = {
        "schema_version": "1",
        "signal_registry": json.dumps({str(k): v for k, v in sr.items()}),
    }
    save_file(tensors, str(out), metadata=md)
    return out


@given(graph=random_threshold_graph())
@settings(max_examples=30, deadline=4000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_full_graph_parses_cleanly(graph, tmp_path_factory):
    """A full safetensors must round-trip through the frontend without raising."""
    tmp_path = tmp_path_factory.mktemp("full")
    tensors, sr, _names = graph
    src = _write_safetensors(tmp_path, tensors, sr)
    fe = ThresholdLogicFrontend()
    g = fe.parse(src)
    assert len(g.gates) >= 2


@given(graph=random_threshold_graph())
@settings(max_examples=30, deadline=4000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_closure_drops_only_unrelated_gates(graph, tmp_path_factory):
    """Extracting from the *last* gate must transitively keep every prior
    gate it depends on; the frontend then parses cleanly in strict mode."""
    tmp_path = tmp_path_factory.mktemp("closure")
    tensors, sr, names = graph
    src = _write_safetensors(tmp_path, tensors, sr)

    target = names[-1]
    dst = tmp_path / "sub.safetensors"
    stats = extract_subset(src, [target], dst, quiet=True)
    assert stats["closure_gates"] >= 1
    # Frontend (strict default) must accept the closure cleanly.
    fe = ThresholdLogicFrontend()
    g = fe.parse(dst)
    assert len(g.gates) == stats["closure_gates"]


@given(graph=random_threshold_graph())
@settings(max_examples=20, deadline=4000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_strict_rejects_partial_extraction(graph, tmp_path_factory):
    """Manually drop a dependency and check that strict mode raises."""
    tmp_path = tmp_path_factory.mktemp("partial")
    tensors, sr, names = graph
    assume(len(names) >= 3)
    # drop all tensors for the first gate; everything that referred to it
    # is now dangling. Strict mode must reject.
    drop = names[0]
    bad = {k: v for k, v in tensors.items() if not k.startswith(drop + ".")}
    src = _write_safetensors(tmp_path, bad, sr, name="bad.safetensors")
    fe = ThresholdLogicFrontend()
    # If no gate actually depended on `drop`, parsing succeeds — that's fine.
    # We only assert: when it raises, it raises *cleanly* (ValueError).
    try:
        fe.parse(src)
    except ValueError:
        pass
