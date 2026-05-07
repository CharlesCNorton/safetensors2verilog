"""Frontend interface, IR, and backend dispatch.

The IR is a dataflow graph of `Gate` operations. Each Gate has a `kind`
that selects a backend lowering rule. Built-in kinds include:

  threshold     Σ wᵢ·xᵢ + bias ≥ 0   (1-bit output)
  add, sub, mul multibit signed/unsigned arithmetic
  and, or, xor, not  bitwise logic
  shift_left, shift_right       constant shifts
  concat, slice                 width plumbing
  mux                           N-way multiplexer
  constant                      multibit integer constant
  rom                           parameter ROM with init
  register                      synchronous flip-flop
  relu, clamp                   activation primitives

Frontends emit Gates whose kind is registered with the Verilog backend.
Custom kinds can be added via the @lowering(kind) decorator in
`safetensors2verilog.verilog`.

Signal widths and sign-ness travel as `Signal(name, width, signed)` on
the GateGraph's external ports and as `output_width` / `output_signed`
on each Gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Signal:
    """An external port of the design.

    name:        free-form symbolic name; the backend sanitises for Verilog
    width:       bit width (1 = single bit, >1 = bus)
    signed:      if True, declared 'signed' and arithmetic uses $signed
    direction:   "in", "out", or "inout"; "inout" enables tristate emission
    is_parameter: if True, declared as a Verilog parameter rather than wire/reg
                  (used for compile-time constants exposed at the module port)
    parameter_value: integer default for parameter ports

    Q-format annotation (informational; hardware operates on raw bits):
      q_int_bits, q_frac_bits — fractional fixed-point split. The bit
        pattern represents the value ``raw / 2**q_frac_bits``; total
        ``width = (1 if signed else 0) + q_int_bits + q_frac_bits`` for a
        canonical Q-format signal, but this is not enforced because some
        carriers (matmul accumulators, packed buses) intentionally have
        slack bits or pack heterogeneous elements. When both Q fields are
        zero the signal is interpreted as a plain integer.
      scale — dequantisation multiplier (a float64) recording the mapping
        from the integer bit pattern to the original real-valued tensor.
        Hardware does not consume ``scale``; downstream tooling, golden
        models, and frontend authors use it to track quantisation error.
    """
    name: str
    width: int = 1
    signed: bool = False
    direction: str = "auto"
    is_parameter: bool = False
    parameter_value: int = 0
    q_int_bits: int = 0
    q_frac_bits: int = 0
    scale: float = 1.0


@dataclass
class Gate:
    """A dataflow node.

    name:           unique identifier within the design (also the produced signal)
    kind:           dispatch tag for the backend lowering rule
    inputs:         signal names this node consumes
    attrs:          kind-specific data (weights, slice indices, init, etc.)
    output_width:   bit width of the produced signal
    output_signed:  if True, the produced signal is two's-complement

    For a threshold gate, `attrs` carries:
        weights : list[int], same length as `inputs`
        bias    : int

    Frontends construct Gates directly. The frontend is responsible for
    knowing which kinds the backend (or its own registered lowerings)
    supports.
    """
    name: str
    kind: str = "threshold"
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    output_width: int = 1
    output_signed: bool = False


@dataclass
class GateGraph:
    """The frontend-produced IR the backend lowers to Verilog.

    inputs:     external input ports (Signal: name + width + sign)
    outputs:    external output ports
    gates:      dataflow nodes, must be topologically sorted
    top:        module name in the generated Verilog
    submodules: additional GateGraphs to emit as separate Verilog modules
                in the same output file. Each submodule's `top` becomes a
                module name that the parent's gates may reference via the
                ``instance`` kind. Used for hierarchical compilation
                (matmul blocks, RMSNorm units, attention heads, etc.) so
                that the parent module stays small while large repeated
                primitives become reusable parameterized modules. Order
                matters only insofar as nested ``instance`` references
                must point at modules that appear earlier in the depth-
                first traversal; the backend emits leaves first.
    """
    inputs: list[Signal]
    outputs: list[Signal]
    gates: list[Gate]
    top: str = "top"
    submodules: list["GateGraph | RawSubmodule"] = field(default_factory=list)


@dataclass
class RawSubmodule:
    """A pre-written Verilog module included verbatim alongside the parent.

    Use this for parameterized templates (matmul block, RMSNorm unit,
    attention head, etc.) whose IR-graph representation would be unwieldy
    or whose synthesis behavior depends on direct vendor pragma placement.
    The backend emits ``text`` verbatim before any module that instantiates
    it via the ``instance`` IR kind.

    top:           module name (must match the ``module`` keyword in ``text``)
    text:          full Verilog source for the module, including ``module``
                   / ``endmodule`` and any necessary `default_nettype
                   directives.
    sidecar_files: filename -> file contents map for any external files the
                   module references (e.g. weight ROMs loaded via
                   ``$readmemh``). Callers writing the emitted Verilog to
                   disk should also write each sidecar file in the same
                   directory; ``collect_sidecar_files(graph)`` walks a
                   GateGraph tree and returns the merged map.
    """
    top: str
    text: str
    sidecar_files: dict[str, str] = field(default_factory=dict)


def collect_sidecar_files(
    graph: "GateGraph",
) -> dict[str, str]:
    """Walk ``graph.submodules`` recursively and merge every ``RawSubmodule``'s
    ``sidecar_files`` into a single dict for the caller to write to disk.

    Duplicate filenames across different RawSubmodules raise ValueError so
    the caller doesn't silently lose ROM contents. Filenames are intended
    to be unique per (module_name, role) combination — see the matmul block
    factory for the canonical naming convention.
    """
    out: dict[str, str] = {}

    def walk(g: "GateGraph") -> None:
        for sub in g.submodules:
            if isinstance(sub, RawSubmodule):
                for fn, contents in sub.sidecar_files.items():
                    if fn in out and out[fn] != contents:
                        raise ValueError(
                            f"sidecar filename collision: '{fn}' "
                            f"(submodule '{sub.top}')"
                        )
                    out[fn] = contents
            else:
                walk(sub)

    walk(graph)
    return out


# ---- Frontend abstraction ---------------------------------------------------


@dataclass
class FrontendOption:
    """One per-frontend CLI option, surfaced by the CLI driver.

    name:    flag name (without leading dashes); becomes --<name>
    type:    str / int / float, or `bool` for store_true flags
    default: value when flag is absent
    help:    short --help description
    metavar: optional argparse metavar override
    """
    name: str
    type: type = str
    default: Any = None
    help: str = ""
    metavar: str | None = None


class Frontend:
    """Subclass and implement parse() to add support for a model class.

    Class attributes (set by the @registry.register decorator):
      name                public CLI name
      description         short blurb
      metadata_namespace  reserved metadata key prefix; avoids collisions
                          when multiple frontends share a safetensors file
    """

    name: str = "base"
    description: str = ""
    metadata_namespace: str = ""

    @classmethod
    def options(cls) -> list[FrontendOption]:
        """Per-frontend CLI options. Override to expose flags."""
        return []

    def parse(self, path: Path, top: str = "top", **options) -> GateGraph:
        raise NotImplementedError("subclasses must implement parse()")


# ---- Frontend registry ------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self._frontends: dict[str, type[Frontend]] = {}

    def register(
        self,
        name: str,
        description: str = "",
        metadata_namespace: str = "",
    ) -> Callable[[type[Frontend]], type[Frontend]]:
        def deco(cls: type[Frontend]) -> type[Frontend]:
            cls.name = name
            cls.description = description
            cls.metadata_namespace = metadata_namespace or name
            self._frontends[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[Frontend]:
        if name not in self._frontends:
            raise KeyError(
                f"unknown frontend '{name}'. Registered: {sorted(self._frontends)}"
            )
        return self._frontends[name]

    def names(self) -> list[tuple[str, str]]:
        return [
            (name, cls.description)
            for name, cls in sorted(self._frontends.items())
        ]


registry = _Registry()


# ---- Metadata-namespace enforcement ---------------------------------------


# Reserved metadata keys that the safetensors2verilog tool itself uses,
# regardless of frontend. Frontends must not write to these.
_RESERVED_METADATA_KEYS = frozenset({
    "format", "encoding", "version", "schema_version",
})

# Schema version for the threshold-logic / signal_registry metadata
# convention. Frontends that read signal_registry consult this so future
# format changes can be detected and rejected with a clear message rather
# than silently misinterpreted.
#
#   1   - original layout (8bit-threshold-computer through May 2026)
#         signal_registry is a JSON object mapping str(int) -> name.
#         External inputs prefixed "$"; constants "#0" / "#1".
#         Gate tensors: <gate>.weight (1-D or [N,K] packed),
#                       <gate>.bias (length-1 or length-N packed),
#                       <gate>.inputs (1-D int signal IDs).
SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED: tuple[int, ...] = (1,)
SIGNAL_REGISTRY_SCHEMA_VERSION_LATEST: int = 1


def check_schema_version(metadata: dict[str, Any], frontend_name: str) -> int:
    """Validate ``schema_version`` in safetensors metadata.

    Returns the parsed integer schema version. Files without the key are
    treated as version 1 for backward compatibility (the value the
    8bit-threshold-computer family uses today). Files whose declared
    version is not in :data:`SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED`
    raise :class:`ValueError` so the frontend never silently misreads a
    future format.
    """
    raw = metadata.get("schema_version")
    if raw is None:
        return 1
    try:
        version = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"frontend '{frontend_name}': metadata schema_version "
            f"{raw!r} is not an integer"
        ) from e
    if version not in SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED:
        raise ValueError(
            f"frontend '{frontend_name}': unsupported schema_version "
            f"{version}; supported versions: "
            f"{list(SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED)}"
        )
    return version


def validate_metadata_namespace(
    frontend: type[Frontend], metadata: dict[str, Any]
) -> None:
    """Check that every metadata key either lives under the frontend's
    namespace, is a reserved global key, or is a special shared key
    (currently: ``signal_registry``, used by both threshold_logic and
    any future frontend that wants symbolic-name lookup).

    Frontends are encouraged to use only keys with the prefix
    ``"<metadata_namespace>."``; everything else triggers a warning.
    Set ``metadata_namespace`` via the @registry.register decorator.
    """
    import warnings

    ns = frontend.metadata_namespace
    shared = {"signal_registry"}
    for key in metadata:
        if key in _RESERVED_METADATA_KEYS or key in shared:
            continue
        if not ns:
            warnings.warn(
                f"frontend '{frontend.name}' has no metadata_namespace and "
                f"is reading metadata key {key!r}; collisions with other "
                f"frontends are possible.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not (key == ns or key.startswith(ns + ".")):
            warnings.warn(
                f"frontend '{frontend.name}' (namespace '{ns}') is reading "
                f"metadata key {key!r} that lives outside its namespace; "
                f"prefer keys named '{ns}.<sub>'.",
                UserWarning,
                stacklevel=2,
            )
