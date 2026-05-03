"""Dependency-closure subset extraction for threshold-logic safetensors.

The 8bit-threshold-computer family (and any threshold_logic source) stores
hundreds of named circuits in a single safetensors file with a shared
``signal_registry`` mapping integer signal IDs to symbolic names. A circuit
like ``arithmetic.multiplier8x8`` references signals produced by other
circuits (e.g. shared ``arithmetic.fulladder`` outputs) via numeric IDs in
its ``.inputs`` tensors. Naively prefix-filtering by circuit name leaves
those references dangling: the threshold_logic frontend then either errors
(strict default) or fabricates anonymous external ports (--promote-unresolved).

This module extracts a circuit *with its dependency closure*: starting from
all gates whose names match the requested prefix(es), walk every signal ID
their gates consume and pull in the gates that *produce* those signals,
transitively.

Public entry points:

  closure_keys(tensors, signal_registry, circuit_prefixes)
      Return the set of tensor keys (``<gate>.weight`` etc.) that should be
      copied to satisfy the closure.

  extract_subset(src_path, circuits, dst_path)
      Read ``src_path``, compute closure for each prefix in ``circuits``,
      write the trimmed safetensors to ``dst_path`` (preserves original
      metadata, including ``signal_registry``).

The metadata's ``signal_registry`` is preserved as-is: signal IDs are
globally unique within a file, and the threshold_logic frontend resolves
external inputs (signals starting with ``$``) on demand. Carrying the full
registry costs a few hundred KB at most and avoids renumbering every
``.inputs`` tensor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_SUFFIXES = (".weight", ".bias", ".inputs")


def _gate_name_from_key(key: str) -> str | None:
    for suf in _SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return None


def _collect_gate_index(
    keys: Iterable[str],
) -> dict[str, set[str]]:
    """Group safetensors keys by gate name."""
    out: dict[str, set[str]] = {}
    for k in keys:
        gname = _gate_name_from_key(k)
        if gname is None:
            continue
        out.setdefault(gname, set()).add(k)
    return out


def _producer_index(
    signal_registry: dict[int, str], gate_names: set[str]
) -> dict[int, str]:
    """Map signal ID -> gate name that produces it.

    A gate produces a signal whose registry name *equals* the gate name.
    External inputs (``$...``) and constants (``#0``, ``#1``) have no
    producer — they're inputs to the design, not gates.
    """
    name_to_id_gate: dict[int, str] = {}
    for sid, sname in signal_registry.items():
        if sname.startswith("$") or sname in ("#0", "#1"):
            continue
        if sname in gate_names:
            name_to_id_gate[sid] = sname
    return name_to_id_gate


def _seed_gates(
    gate_names: set[str], prefixes: list[str]
) -> set[str]:
    """Start with every gate whose name equals or starts with one of the prefixes."""
    seeds: set[str] = set()
    for g in gate_names:
        for p in prefixes:
            if g == p or g.startswith(p + "."):
                seeds.add(g)
                break
    return seeds


def closure_keys(
    tensors: dict[str, torch.Tensor],
    signal_registry: dict[int, str],
    circuit_prefixes: list[str],
) -> tuple[set[str], dict[str, int]]:
    """Compute the dependency closure for the requested circuit prefixes.

    Returns (tensor_keys_to_keep, stats). Stats has 'seed_gates',
    'closure_gates', 'unresolved_signals'. Unresolved signals are non-dollar
    non-hash names referenced by closure gates that no in-file gate produces
    — these will become external inputs on the extracted module.
    """
    if not circuit_prefixes:
        raise ValueError("circuit_prefixes must be non-empty")

    by_gate = _collect_gate_index(tensors.keys())
    all_gate_names = set(by_gate)
    seeds = _seed_gates(all_gate_names, circuit_prefixes)
    if not seeds:
        raise ValueError(
            f"no gates matched circuit prefixes {circuit_prefixes}; "
            f"check the safetensors file's gate names"
        )

    sid_to_producer = _producer_index(signal_registry, all_gate_names)

    # BFS over the producer graph
    closure: set[str] = set(seeds)
    frontier: list[str] = list(seeds)
    unresolved: set[str] = set()

    while frontier:
        gname = frontier.pop()
        ikey = gname + ".inputs"
        if ikey not in tensors:
            continue
        ids = [int(round(float(v))) for v in tensors[ikey].flatten().tolist()]
        for sid in ids:
            if sid in sid_to_producer:
                producer = sid_to_producer[sid]
                if producer not in closure:
                    closure.add(producer)
                    frontier.append(producer)
            else:
                # External ($), constant (#), or unresolvable: not a gate.
                sname = signal_registry.get(sid)
                if sname is None:
                    unresolved.add(f"<id={sid}>")
                elif not (sname.startswith("$") or sname in ("#0", "#1")):
                    unresolved.add(sname)

    keep: set[str] = set()
    for gname in closure:
        keep.update(by_gate[gname])

    # Always keep manifest tensors so downstream tooling can read width info.
    for k in tensors:
        if k.startswith("manifest."):
            keep.add(k)

    stats = {
        "seed_gates": len(seeds),
        "closure_gates": len(closure),
        "unresolved_signals": len(unresolved),
    }
    return keep, stats


def extract_subset(
    src_path: Path,
    circuits: list[str],
    dst_path: Path,
    *,
    quiet: bool = False,
) -> dict[str, int]:
    """Extract a dependency-closed subset of a threshold_logic safetensors.

    src_path: input safetensors (e.g., neural_alu8.safetensors)
    circuits: list of circuit prefixes; each entry matches a gate whose
              name equals or is a dotted descendant of the prefix
              (e.g., 'arithmetic.multiplier8x8' or 'boolean.xor').
    dst_path: output safetensors with closure-only tensors and full metadata.

    Returns the stats dict from closure_keys.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with safe_open(str(src_path), framework="pt") as f:
        meta = f.metadata() or {}
    if "signal_registry" not in meta:
        raise ValueError(
            f"{src_path}: no signal_registry metadata; "
            f"this isn't a threshold_logic safetensors file"
        )
    signal_registry = {
        int(k): v for k, v in json.loads(meta["signal_registry"]).items()
    }

    tensors = load_file(str(src_path))
    keep, stats = closure_keys(tensors, signal_registry, circuits)
    sub = {k: tensors[k] for k in keep}

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sub, str(dst_path), metadata=meta)

    if not quiet:
        import sys
        print(
            f"extracted {stats['seed_gates']} seed gate(s) and "
            f"{stats['closure_gates'] - stats['seed_gates']} dependency gate(s) "
            f"({stats['closure_gates']} total) into {dst_path}",
            file=sys.stderr,
        )
        if stats["unresolved_signals"]:
            print(
                f"  note: {stats['unresolved_signals']} signal(s) remain "
                f"unresolved and will become external inputs",
                file=sys.stderr,
            )
    return stats
