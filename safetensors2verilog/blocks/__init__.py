"""Higher-level RTL building blocks emitted as ``RawSubmodule`` primitives.

These exist because some hardware primitives (matmul arrays, RMSNorm units,
attention heads, embedding ROMs) are too large or too vendor-specific to
express clearly as gate-by-gate IR; we emit them as parameterized Verilog
text and let the IR ``instance`` kind wire them into the parent dataflow.

Each block exposes a factory that returns a ``(submodule, instance_gates,
extern_wires)`` triple the frontend can splice directly into a GateGraph.
"""

from .matmul import matmul_seq_block, matmul_seq_invoke

__all__ = ["matmul_seq_block", "matmul_seq_invoke"]
