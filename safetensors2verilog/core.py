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
    """
    name: str
    width: int = 1
    signed: bool = False
    direction: str = "auto"
    is_parameter: bool = False
    parameter_value: int = 0


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

    inputs:  external input ports (Signal: name + width + sign)
    outputs: external output ports
    gates:   dataflow nodes, must be topologically sorted
    top:     module name in the generated Verilog
    """
    inputs: list[Signal]
    outputs: list[Signal]
    gates: list[Gate]
    top: str = "top"


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
_RESERVED_METADATA_KEYS = frozenset({"format", "encoding", "version"})


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
