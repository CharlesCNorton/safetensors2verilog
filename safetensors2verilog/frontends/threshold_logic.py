"""Threshold-logic frontend.

Targets safetensors files whose tensors are named:
  <gate>.weight   1-D integer tensor (or packed multi-row [N, K])
  <gate>.bias     length-1 integer tensor (or length-N for packed)
  <gate>.inputs   1-D int tensor of signal IDs (optional)

Plus a JSON metadata field ``signal_registry`` mapping integer signal
IDs to symbolic names. Names beginning with ``$`` are external module
inputs; ``#0`` / ``#1`` are constant zero / one wires; all others are
gate outputs.

Manifest tensors (prefix ``manifest.``) are ignored.

This is the format produced by the 8bit-threshold-computer family at
https://huggingface.co/phanerozoic/8bit-threshold-computer, but the
schema is general enough to cover any threshold-gate hierarchical
network.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from ..core import Frontend, FrontendOption, Gate, GateGraph, Signal, registry


def _is_integer_tensor(t: torch.Tensor, atol: float = 1e-6) -> bool:
    """Allow rounding error from QAT-exported floats."""
    if not t.dtype.is_floating_point:
        return True
    tf = t.to(torch.float64)
    return bool(torch.isclose(tf, tf.round(), atol=atol).all().item())


def _is_ternary(t: torch.Tensor) -> bool:
    if not _is_integer_tensor(t):
        return False
    return bool((t.to(torch.float64).abs() <= 1.0).all().item())


def _as_int_list(t: torch.Tensor) -> list[int]:
    return [int(round(float(v))) for v in t.flatten().tolist()]


@registry.register(
    "threshold_logic",
    description="Threshold-gate networks with named-circuit hierarchy and signal_registry metadata.",
    metadata_namespace="threshold_logic",
)
class ThresholdLogicFrontend(Frontend):

    @classmethod
    def options(cls) -> list[FrontendOption]:
        return [
            FrontendOption(
                name="skip-memory",
                type=bool,
                default=False,
                help=(
                    "drop memory.* gates so they can be served by a vendor BRAM "
                    "block. Internal references to memory.* signals become external "
                    "inputs of the resulting CPU core; address / data / write-enable "
                    "signals appear as external inputs from external code."
                ),
            ),
            FrontendOption(
                name="strict",
                type=bool,
                default=True,
                help=(
                    "error out on stale or missing routing metadata instead of "
                    "promoting affected inputs to anonymous external ports. "
                    "On by default; pass --promote-unresolved to opt in to the "
                    "permissive behavior (silently fabricates external ports)."
                ),
            ),
            FrontendOption(
                name="promote-unresolved",
                type=bool,
                default=False,
                help=(
                    "permissive complement of --strict: promote stale or missing "
                    "routing references to anonymous external ports rather than "
                    "raising. Useful for debugging partial extractions; do not "
                    "use for production compilation."
                ),
            ),
        ]

    def parse(
        self,
        path: Path,
        top: str = "top",
        skip_memory: bool = False,
        strict: bool = True,
        promote_unresolved: bool = False,
        **options,
    ) -> GateGraph:
        # --promote-unresolved overrides the default strict=True.
        if promote_unresolved:
            strict = False
        from ..core import check_schema_version, validate_metadata_namespace

        # ---- Load tensors and signal registry ----
        tensors: dict[str, torch.Tensor] = {}
        signal_registry: dict[int, str] = {}
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            check_schema_version(meta, "threshold_logic")
            validate_metadata_namespace(type(self), meta)
            if "signal_registry" in meta:
                raw = json.loads(meta["signal_registry"])
                signal_registry = {int(k): v for k, v in raw.items()}
            for name in f.keys():
                tensors[name] = f.get_tensor(name).clone()

        # ---- Identify gate names ----
        gate_names: set[str] = set()
        for name in tensors:
            for suffix in (".weight", ".bias", ".inputs"):
                if name.endswith(suffix):
                    gate_names.add(name[: -len(suffix)])
                    break
        gate_names = {g for g in gate_names if not g.startswith("manifest.")}

        # ---- Walk gates ----
        gates_raw: list[tuple[str, list[int], list[str], int]] = []
        external_inputs: set[str] = set()
        non_ternary: list[str] = []
        stats = {
            "unpacked": 0,
            "skipped_packed": 0,
            "stale_routing": 0,
            "missing_routing": 0,
        }

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

            # Packed multi-gate path: bias has N entries, weight has N*K entries.
            if b.numel() > 1 and w.numel() % b.numel() == 0:
                n_gates = b.numel()
                k = w.numel() // n_gates
                bias_flat = _as_int_list(b)
                weight_flat = _as_int_list(w)
                inputs_flat = (
                    _as_int_list(tensors[i_key]) if i_key in tensors else []
                )
                inputs_per = (
                    len(inputs_flat) // n_gates if inputs_flat else 0
                )
                if inputs_per not in (0, k):
                    inputs_flat = []
                    inputs_per = 0
                stats["unpacked"] += n_gates

                for i in range(n_gates):
                    sub_name = f"{gname}.bit{i}"
                    sub_w = weight_flat[i * k : (i + 1) * k]
                    sub_b = bias_flat[i]
                    if inputs_per:
                        sub_ids = inputs_flat[i * k : (i + 1) * k]
                        sub_inputs = self._resolve_ids(
                            sub_ids, signal_registry, sub_name,
                            external_inputs, strict,
                        )
                    else:
                        sub_inputs = self._anon_inputs(
                            sub_name, k, external_inputs, strict
                        )
                        stats["missing_routing"] += 1
                    if not all(abs(wv) <= 1 for wv in sub_w):
                        non_ternary.append(sub_name)
                    gates_raw.append((sub_name, sub_w, sub_inputs, sub_b))
                continue

            if w.dim() != 1 or b.numel() != 1:
                stats["skipped_packed"] += 1
                continue
            if not _is_ternary(w):
                non_ternary.append(gname)

            weights = _as_int_list(w)
            bias = int(round(float(b.item())))

            if i_key in tensors:
                ids = _as_int_list(tensors[i_key])
                if len(ids) != len(weights):
                    stats["stale_routing"] += 1
                    inputs = self._anon_inputs(
                        gname, len(weights), external_inputs, strict
                    )
                else:
                    inputs = self._resolve_ids(
                        ids, signal_registry, gname, external_inputs, strict
                    )
            else:
                stats["missing_routing"] += 1
                inputs = self._anon_inputs(
                    gname, len(weights), external_inputs, strict
                )

            gates_raw.append((gname, weights, inputs, bias))

        # ---- Memory carve-out ----
        if skip_memory:
            mem_prefix = "memory."
            mem_set = {n for n, *_ in gates_raw if n.startswith(mem_prefix)}
            promoted = 0
            for name, _w, ins, _b in gates_raw:
                if name.startswith(mem_prefix):
                    continue
                for src in ins:
                    if src in mem_set:
                        external_inputs.add(src)
                        promoted += 1
            gates_raw = [
                g for g in gates_raw if not g[0].startswith(mem_prefix)
            ]
            referenced: set[str] = set()
            for _n, _w, ins, _b in gates_raw:
                referenced.update(ins)
            external_inputs = {x for x in external_inputs if x in referenced}
            stats["mem_skipped"] = len(mem_set)
            stats["mem_promoted"] = promoted

        # ---- Diagnostics ----
        self._print_warnings(non_ternary, stats)

        # ---- Topological sort (iterative; tolerates deep chains) ----
        sorted_raw = self._topo_sort(gates_raw)

        # ---- Build Gate IR ----
        gates: list[Gate] = [
            Gate(
                name=name,
                kind="threshold",
                inputs=inputs,
                attrs={"weights": weights, "bias": bias},
                output_width=1,
                output_signed=False,
            )
            for name, weights, inputs, bias in sorted_raw
        ]

        consumed: set[str] = set()
        produced: set[str] = set()
        for g in gates:
            produced.add(g.name)
            for s in g.inputs:
                consumed.add(s)
        output_names = sorted(g.name for g in gates if g.name not in consumed)

        for g in gates:
            for s in g.inputs:
                if s in ("#0", "#1"):
                    continue
                if s not in produced:
                    external_inputs.add(s)

        input_signals = [
            Signal(name=n, width=1, signed=False)
            for n in sorted(external_inputs)
        ]
        output_signals = [
            Signal(name=n, width=1, signed=False) for n in output_names
        ]

        return GateGraph(
            inputs=input_signals,
            outputs=output_signals,
            gates=gates,
            top=top,
        )

    # ---- Helpers ----

    @staticmethod
    def _resolve_ids(
        ids: list[int],
        reg: dict[int, str],
        gname: str,
        external: set[str],
        strict: bool,
    ) -> list[str]:
        names: list[str] = []
        for sid in ids:
            if sid in reg:
                nm = reg[sid]
            else:
                if strict:
                    raise ValueError(
                        f"gate '{gname}' references unresolved signal id {sid}"
                    )
                nm = f"_unresolved_{gname}_id{sid}"
                external.add(nm)
            if nm.startswith("$"):
                external.add(nm)
            names.append(nm)
        return names

    @staticmethod
    def _anon_inputs(
        gname: str, n: int, external: set[str], strict: bool
    ) -> list[str]:
        if strict:
            raise ValueError(
                f"gate '{gname}' has missing or stale routing metadata "
                f"(use strict=False to promote to anonymous external inputs)"
            )
        names = [f"{gname}.in{i}" for i in range(n)]
        for nm in names:
            external.add(nm)
        return names

    @staticmethod
    def _print_warnings(non_ternary: list[str], stats: dict[str, int]) -> None:
        import warnings

        if non_ternary:
            warnings.warn(
                f"{len(non_ternary)} gate(s) have non-ternary weights; "
                f"the IR carries them as integer weights (kind='threshold' with "
                f"weights:list[int]). Synthesis lowers k*x as a constant-coefficient "
                f"multiplier. First few: {non_ternary[:5]}",
                UserWarning, stacklevel=2,
            )
        if stats.get("unpacked"):
            warnings.warn(
                f"unpacked {stats['unpacked']} sub-gate(s) from packed tensors.",
                UserWarning, stacklevel=2,
            )
        if stats.get("skipped_packed"):
            warnings.warn(
                f"skipped {stats['skipped_packed']} packed tensor(s) "
                f"with non-rectangular layout.",
                UserWarning, stacklevel=2,
            )
        if stats.get("stale_routing"):
            warnings.warn(
                f"{stats['stale_routing']} gate(s) had .inputs metadata "
                f"out of sync with .weight; their inputs were promoted to anonymous "
                f"external ports. Regenerate the safetensors' routing metadata.",
                UserWarning, stacklevel=2,
            )
        if stats.get("missing_routing"):
            warnings.warn(
                f"{stats['missing_routing']} gate(s) had no .inputs "
                f"metadata; their inputs were promoted to anonymous external ports.",
                UserWarning, stacklevel=2,
            )
        if stats.get("mem_skipped"):
            warnings.warn(
                f"skipped {stats['mem_skipped']} memory.* gate(s); "
                f"{stats.get('mem_promoted', 0)} read-side reference(s) were "
                f"promoted to external inputs. Wire them to a vendor BRAM block "
                f"(use --emit-bram-template to get a starter module).",
                UserWarning, stacklevel=2,
            )

    @staticmethod
    def _topo_sort(
        gates_raw: list[tuple[str, list[int], list[str], int]],
    ) -> list[tuple[str, list[int], list[str], int]]:
        """Iterative DFS so very deep gate chains do not hit Python's recursion limit."""
        gate_set = {name for name, *_ in gates_raw}
        gate_by_name = {name: (w, ins, b) for name, w, ins, b in gates_raw}
        deps: dict[str, list[str]] = {
            name: [s for s in ins if s in gate_set]
            for name, _w, ins, _b in gates_raw
        }

        order: list[str] = []
        marked: set[str] = set()
        in_progress: set[str] = set()

        for start in [name for name, *_ in gates_raw]:
            if start in marked:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, visited = stack.pop()
                if visited:
                    in_progress.discard(node)
                    if node not in marked:
                        marked.add(node)
                        order.append(node)
                    continue
                if node in marked:
                    continue
                if node in in_progress:
                    raise ValueError(
                        f"cycle detected in gate graph involving '{node}'"
                    )
                in_progress.add(node)
                stack.append((node, True))
                for dep in deps.get(node, ()):
                    if dep not in marked:
                        stack.append((dep, False))

        return [(name, *gate_by_name[name]) for name in order]
