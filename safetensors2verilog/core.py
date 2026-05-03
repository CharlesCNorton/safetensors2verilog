"""Frontend interface and the gate graph the backend ingests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Gate:
    """One node in the dataflow graph.

    name:    unique identifier within the design
    pos:     list of input signal names contributing with weight +1
    neg:     list of input signal names contributing with weight -1
    bias:    integer bias added before the >=0 comparison
    kind:    free-form tag the backend can use to specialise emission
             (e.g. "and", "or", "xor", "popcount_compare"); frontends
             can use a default kind of "threshold" if no specialisation
             is needed
    """
    name: str
    pos: List[str] = field(default_factory=list)
    neg: List[str] = field(default_factory=list)
    bias: int = 0
    kind: str = "threshold"


@dataclass
class GateGraph:
    """The frontend-produced graph the Verilog backend lowers.

    inputs:  external input signal names (module ports)
    outputs: external output signal names (module ports)
    gates:   ordered list of Gate; must be topologically sorted so that
             every input to a gate is either an external input or a
             previously-declared gate
    top:     module name to use in the generated Verilog
    """
    inputs: List[str]
    outputs: List[str]
    gates: List[Gate]
    top: str = "top"


class Frontend:
    """Subclass and implement `parse` to add support for a new model class.

    A frontend is responsible for reading a safetensors file (and any
    auxiliary information) and producing a `GateGraph` that the backend
    can emit as Verilog. The backend assumes nothing about model
    semantics beyond what GateGraph carries.
    """

    name: str = "base"
    description: str = ""

    def parse(self, path: Path, **options) -> GateGraph:
        raise NotImplementedError("subclasses must implement parse()")


# Frontend registry. Frontends register themselves by decorating their
# class definition with @registry.register(name, description).
class _Registry:
    def __init__(self):
        self._frontends: Dict[str, type[Frontend]] = {}

    def register(self, name: str, description: str = "") -> Callable[[type[Frontend]], type[Frontend]]:
        def deco(cls: type[Frontend]) -> type[Frontend]:
            cls.name = name
            cls.description = description
            self._frontends[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type[Frontend]:
        if name not in self._frontends:
            raise KeyError(
                f"unknown frontend '{name}'. Registered: {sorted(self._frontends)}"
            )
        return self._frontends[name]

    def names(self) -> List[Tuple[str, str]]:
        return [(name, cls.description) for name, cls in sorted(self._frontends.items())]


registry = _Registry()
