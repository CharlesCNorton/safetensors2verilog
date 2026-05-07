"""safetensors2verilog: compile safetensors networks to synthesis-ready Verilog."""

# Importing each frontend module registers it via the registry decorator.
from . import frontends  # noqa: F401
from .core import (
    Frontend,
    FrontendOption,
    Gate,
    GateGraph,
    RawSubmodule,
    Signal,
    collect_sidecar_files,
    registry,
)
from .evaluate import evaluate_graph, step_graph
from .verilog import (
    collect_clocks,
    collect_resets,
    emit_bram_template,
    emit_module,
    emit_top_wrapper,
    lowering,
    registered_kinds,
)

__all__ = [
    "Frontend",
    "FrontendOption",
    "Gate",
    "GateGraph",
    "RawSubmodule",
    "Signal",
    "registry",
    "collect_sidecar_files",
    "emit_module",
    "emit_bram_template",
    "emit_top_wrapper",
    "collect_clocks",
    "collect_resets",
    "lowering",
    "registered_kinds",
    "evaluate_graph",
    "step_graph",
]
