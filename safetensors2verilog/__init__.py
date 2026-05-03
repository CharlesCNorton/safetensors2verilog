"""safetensors2verilog: compile safetensors networks to synthesis-ready Verilog."""

from .core import (
    Frontend,
    FrontendOption,
    Gate,
    GateGraph,
    Signal,
    registry,
)
from .verilog import (
    emit_bram_template,
    emit_module,
    lowering,
    registered_kinds,
)

# Importing each frontend module registers it via the registry decorator.
from . import frontends  # noqa: F401

__all__ = [
    "Frontend",
    "FrontendOption",
    "Gate",
    "GateGraph",
    "Signal",
    "registry",
    "emit_module",
    "emit_bram_template",
    "lowering",
    "registered_kinds",
]
