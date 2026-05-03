"""Threshold-logic frontend.

Targets safetensors files whose tensors are named `<gate>.weight`,
`<gate>.bias`, and (optionally) `<gate>.inputs`, with a JSON metadata
field `signal_registry` mapping integer signal IDs to symbolic names.

Each gate's `.weight` is a 1-D integer tensor; `.bias` is a length-1
integer tensor; `.inputs` (when present) is a 1-D int tensor of signal
IDs whose length matches `.weight`. Signal names beginning with `$`
are treated as external inputs to the module; `#0` and `#1` are
constant zero/one wires; all others are gate outputs.

This is the format produced by the 8bit-threshold-computer family at
https://huggingface.co/phanerozoic/8bit-threshold-computer but the
schema is general enough to cover any threshold-gate hierarchical
network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from safetensors import safe_open

from ..core import Frontend, Gate, GateGraph, registry


def _is_integer_tensor(t: torch.Tensor) -> bool:
    if not t.dtype.is_floating_point:
        return True
    return torch.equal(t.float().round(), t.float())


def _is_ternary(t: torch.Tensor) -> bool:
    """Check whether all values are in {-1, 0, 1}."""
    tf = t.float()
    return bool((tf.abs() <= 1.0).all().item()) and _is_integer_tensor(t)


def _as_int_list(t: torch.Tensor) -> List[int]:
    return [int(v) for v in t.flatten().tolist()]


@registry.register(
    "threshold_logic",
    "Threshold-gate networks with named-circuit hierarchy and signal_registry metadata.",
)
class ThresholdLogicFrontend(Frontend):

    def parse(self, path: Path, top: str = "top", skip_memory: bool = False, **options) -> GateGraph:
        tensors: Dict[str, torch.Tensor] = {}
        signal_registry: Dict[int, str] = {}
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            if "signal_registry" in meta:
                raw = json.loads(meta["signal_registry"])
                # Stored as {"0": "#0", "1": "#1", "2": "$a", ...}
                signal_registry = {int(k): v for k, v in raw.items()}
            for name in f.keys():
                tensors[name] = f.get_tensor(name).clone()

        # Collect all gate names by stripping known suffixes
        gate_names: Set[str] = set()
        for name in tensors:
            for suffix in (".weight", ".bias", ".inputs"):
                if name.endswith(suffix):
                    gate_names.add(name[: -len(suffix)])
                    break

        # Manifest tensors are not gates
        gate_names = {g for g in gate_names if not g.startswith("manifest.")}

        # Build a parse-time list of gate descriptors
        gates_raw: List[Tuple[str, List[int], List[str], int]] = []
        external_inputs: Set[str] = set()
        non_ternary_warnings: List[str] = []

        for gname in sorted(gate_names):
            w_key = gname + ".weight"
            b_key = gname + ".bias"
            i_key = gname + ".inputs"
            if w_key not in tensors or b_key not in tensors:
                continue
            w = tensors[w_key]
            b = tensors[b_key]
            if not _is_integer_tensor(w):
                raise ValueError(f"gate '{gname}': weights are not integer-valued")
            if not _is_integer_tensor(b):
                raise ValueError(f"gate '{gname}': bias is not integer-valued")

            # Detect packed multi-gate tensors. The convention is:
            # bias holds N per-gate biases, weight holds N*K per-gate weights
            # (concatenated, possibly across additional dims). Unpack into
            # N gates named "{gname}.bit{i}" with K weights each.
            if b.numel() > 1 and w.numel() % b.numel() == 0:
                n_gates = b.numel()
                k = w.numel() // n_gates
                bias_flat = _as_int_list(b)
                weight_flat = _as_int_list(w)
                inputs_flat: List[int] = (
                    _as_int_list(tensors[i_key]) if i_key in tensors else []
                )
                inputs_per_gate = len(inputs_flat) // n_gates if inputs_flat else 0
                if inputs_per_gate not in (0, k):
                    inputs_flat = []  # routing length doesn't match; treat as missing
                    inputs_per_gate = 0
                self._unpacked_count = getattr(self, "_unpacked_count", 0) + n_gates
                for i in range(n_gates):
                    sub_name = f"{gname}.bit{i}"
                    sub_weights = weight_flat[i * k : (i + 1) * k]
                    sub_bias = bias_flat[i]
                    if inputs_per_gate:
                        sub_input_ids = inputs_flat[i * k : (i + 1) * k]
                        sub_input_names: List[str] = []
                        for sid in sub_input_ids:
                            if sid in signal_registry:
                                sub_input_names.append(signal_registry[sid])
                            else:
                                ph = f"_unresolved_{sub_name}_id{sid}"
                                sub_input_names.append(ph)
                                external_inputs.add(ph)
                    else:
                        sub_input_names = [f"{sub_name}.in{j}" for j in range(k)]
                        for nm in sub_input_names:
                            external_inputs.add(nm)
                        self._missing_routing_count = getattr(self, "_missing_routing_count", 0) + 1
                    for nm in sub_input_names:
                        if nm.startswith("$"):
                            external_inputs.add(nm)
                    if not all(abs(wv) <= 1 for wv in sub_weights):
                        non_ternary_warnings.append(sub_name)
                    gates_raw.append((sub_name, sub_weights, sub_input_names, sub_bias))
                continue

            # Single-gate path: w must be 1-D, b must be a scalar.
            if w.dim() != 1 or b.numel() != 1:
                self._packed_count = getattr(self, "_packed_count", 0) + 1
                continue
            if not _is_ternary(w):
                non_ternary_warnings.append(gname)

            weights = _as_int_list(w)
            bias = int(b.float().item())

            # Resolve input names
            input_names: List[str]
            stale_inputs = False
            if i_key in tensors:
                ids = _as_int_list(tensors[i_key])
                if len(ids) != len(weights):
                    # Stale routing: tensor was rebuilt (different fan-in) but
                    # .inputs wasn't regenerated. Treat as if no routing info.
                    stale_inputs = True
                    input_names = [f"{gname}.in{i}" for i in range(len(weights))]
                else:
                    input_names = []
                    for sid in ids:
                        if sid in signal_registry:
                            input_names.append(signal_registry[sid])
                        else:
                            # Unresolved IDs (e.g. -1 placeholders) become
                            # synthetic external inputs.
                            placeholder = f"_unresolved_{gname}_id{sid}"
                            input_names.append(placeholder)
                            external_inputs.add(placeholder)
            else:
                # No routing info; the gate's logical inputs are unknown.
                input_names = [f"{gname}.in{i}" for i in range(len(weights))]
                self._missing_routing_count = getattr(self, "_missing_routing_count", 0) + 1

            if stale_inputs:
                self._stale_count = getattr(self, "_stale_count", 0) + 1

            for inp in input_names:
                if inp.startswith("$"):
                    external_inputs.add(inp)
                elif inp.startswith(f"{gname}.in"):
                    # Anonymous placeholder for a gate without proper routing
                    # metadata: promote to external so the file is at least
                    # synthesizable. Caller should regenerate .inputs.
                    external_inputs.add(inp)

            gates_raw.append((gname, weights, input_names, bias))

        # ----- BRAM carve-out: drop the memory.* gates and let a real
        # vendor block-RAM serve those reads/writes. The signals that
        # non-memory gates were consuming from memory.* (read data bits)
        # become external module inputs, ready to be wired to BRAM
        # data_out. The signals that memory.* gates consume ($addr,
        # $data, $we, etc.) are already external and become natural
        # outputs of the resulting CPU core.
        if skip_memory:
            mem_prefix = "memory."
            mem_gate_set = {n for n, *_ in gates_raw if n.startswith(mem_prefix)}
            promoted: Set[str] = set()
            for name, _w, inputs, _b in gates_raw:
                if name.startswith(mem_prefix):
                    continue
                for inp in inputs:
                    if inp in mem_gate_set:
                        promoted.add(inp)
                        external_inputs.add(inp)
            gates_raw = [g for g in gates_raw if not g[0].startswith(mem_prefix)]
            self._memory_skipped = len(mem_gate_set)
            self._memory_promoted = len(promoted)

            referenced: Set[str] = set()
            for _n, _w, inputs, _b in gates_raw:
                referenced.update(inputs)
            external_inputs = {x for x in external_inputs if x in referenced}

        if non_ternary_warnings:
            print(
                f"warning: {len(non_ternary_warnings)} gate(s) have weights "
                f"outside {{-1, 0, 1}}; the backend handles them but the "
                f"generated RTL will use larger adder trees for those gates. "
                f"First few: {non_ternary_warnings[:5]}"
            )
        if getattr(self, "_memory_skipped", 0):
            print(
                f"info: skipped {self._memory_skipped} memory.* gate(s); "
                f"{self._memory_promoted} read-side signal(s) promoted to "
                f"external inputs. Wire them to a vendor BRAM block."
            )
        if getattr(self, "_stale_count", 0):
            print(
                f"warning: {self._stale_count} gate(s) have stale .inputs "
                f"metadata (length mismatch with .weight); their inputs "
                f"have been promoted to external module ports. Regenerate "
                f"the file's .inputs metadata for accurate routing."
            )
        if getattr(self, "_unpacked_count", 0):
            print(
                f"info: unpacked {self._unpacked_count} sub-gate(s) from "
                f"packed tensors (one bias per row, weights concatenated)."
            )
        if getattr(self, "_packed_count", 0):
            print(
                f"warning: skipped {self._packed_count} packed tensor(s) "
                f"with non-rectangular layout (weight/bias size mismatch)."
            )

        # Topological sort
        gate_set = {name for name, *_ in gates_raw}
        deps: Dict[str, Set[str]] = {}
        for gname, _w, inputs, _b in gates_raw:
            deps[gname] = {inp for inp in inputs if inp in gate_set}

        sorted_names: List[str] = []
        marked: Set[str] = set()
        in_progress: Set[str] = set()

        def visit(node: str) -> None:
            if node in marked:
                return
            if node in in_progress:
                raise ValueError(
                    f"cycle detected in gate dependency graph involving '{node}'"
                )
            in_progress.add(node)
            for dep in deps.get(node, ()):
                visit(dep)
            in_progress.discard(node)
            marked.add(node)
            sorted_names.append(node)

        gate_by_name = {name: (w, inputs, bias) for name, w, inputs, bias in gates_raw}
        for name, *_ in gates_raw:
            visit(name)

        # Translate to Gate dataclasses
        gates: List[Gate] = []
        for name in sorted_names:
            w, inputs, bias = gate_by_name[name]
            pos: List[str] = []
            neg: List[str] = []
            for weight, src in zip(w, inputs):
                if weight > 0:
                    for _ in range(weight):
                        pos.append(src)
                elif weight < 0:
                    for _ in range(-weight):
                        neg.append(src)
                # weight == 0: drop the connection
            gates.append(Gate(name=name, pos=pos, neg=neg, bias=bias))

        # Outputs: gates that aren't anyone's input
        used: Set[str] = set()
        for g in gates:
            for src in g.pos + g.neg:
                used.add(src)
        outputs = sorted(g.name for g in gates if g.name not in used)

        # External inputs: anything referenced but not produced by any gate.
        # Constants `#0` / `#1` are handled as Verilog literals, not ports.
        produced = {g.name for g in gates}
        for g in gates:
            for src in g.pos + g.neg:
                if src in ("#0", "#1"):
                    continue
                if src not in produced:
                    external_inputs.add(src)

        inputs_list = sorted(external_inputs)

        return GateGraph(
            inputs=inputs_list,
            outputs=outputs,
            gates=gates,
            top=top,
        )
