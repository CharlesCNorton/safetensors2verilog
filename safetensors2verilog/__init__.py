"""safetensors2verilog: compile safetensors-stored networks to synthesis-ready Verilog."""

from .core import Frontend, Gate, GateGraph, registry
from .verilog import emit_module, emit_bram_template

# Importing each frontend module registers it via the registry decorator.
from .frontends import threshold_logic  # noqa: F401

__all__ = ["Frontend", "Gate", "GateGraph", "registry", "emit_module", "emit_bram_template"]
__version__ = "0.1.0"
