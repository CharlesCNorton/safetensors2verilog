"""Verilog backend: lowers a `GateGraph` to synthesizable Verilog.

`emit_module(graph)` dispatches each Gate.kind to a registered lowering
function. Built-in kinds cover threshold gates, multi-bit arithmetic
(add/sub/mul), bitwise logic (and/or/xor/not), constant shifts, slicing
and concatenation, multiplexers, ROMs, registers, and a couple of
activation primitives (relu, clamp).

To add a new kind, decorate a function with @lowering(kind):

    from safetensors2verilog.verilog import lowering

    @lowering("my_op")
    def lower_my_op(ctx, gate):
        return [f"  assign {ctx.name(gate.name)} = ...;"]
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set

from .core import Gate, GateGraph, Signal


# ---- Identifier handling ----------------------------------------------------


_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERILOG_KEYWORDS = frozenset({
    "always", "and", "assign", "begin", "case", "casex", "casez",
    "default", "deassign", "disable", "edge", "else", "end", "endcase",
    "endconfig", "endfunction", "endgenerate", "endmodule", "endprimitive",
    "endspecify", "endtable", "endtask", "event", "for", "force", "forever",
    "fork", "function", "generate", "genvar", "if", "ifnone", "include",
    "initial", "inout", "input", "integer", "join", "module", "negedge",
    "or", "output", "parameter", "posedge", "primitive", "real", "realtime",
    "reg", "release", "repeat", "rnmos", "rpmos", "rtran", "rtranif0",
    "rtranif1", "scalared", "signed", "specify", "specparam", "table",
    "task", "time", "tran", "tranif0", "tranif1", "tri", "tri0", "tri1",
    "triand", "trior", "trireg", "vectored", "wait", "wand", "weak0",
    "weak1", "while", "wire", "wor", "xnor", "xor",
})


def _validate_module_name(name: str) -> None:
    if not name:
        raise ValueError("module name must not be empty")
    if not _MODULE_NAME_RE.match(name):
        raise ValueError(
            f"invalid module name '{name}': must start with letter/underscore "
            f"and contain only [A-Za-z0-9_]"
        )
    if name in _VERILOG_KEYWORDS:
        raise ValueError(f"module name '{name}' is a Verilog keyword")


def _sanitize(name: str, used: Optional[Set[str]] = None) -> str:
    """Map a free-form signal name to a Verilog-legal identifier.

    Replaces [^A-Za-z0-9_] with '_'; ensures the result starts with a
    non-digit. If `used` is supplied, the result is suffixed with
    '_u<N>' until it does not collide with anything in `used`, and the
    final name is added to `used`.
    """
    if not name:
        out = "_anon"
    else:
        out = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not out or out[0].isdigit():
            out = "_" + out
    if out in _VERILOG_KEYWORDS:
        out = out + "_"

    if used is None:
        return out

    base = out
    n = 0
    while out in used:
        n += 1
        out = f"{base}_u{n}"
    used.add(out)
    return out


def _signal_decl(
    direction: Optional[str],
    keyword: str,
    sig_name: str,
    width: int,
    signed: bool,
) -> str:
    parts: List[str] = []
    if direction:
        parts.append(direction)
    parts.append(keyword)
    if signed:
        parts.append("signed")
    if width > 1:
        parts.append(f"[{width - 1}:0]")
    parts.append(sig_name)
    return "  " + " ".join(parts)


# ---- Emit context -----------------------------------------------------------


class EmitContext:
    """State threaded into per-gate lowering functions."""

    def __init__(
        self,
        sigmap: Dict[str, str],
        widths: Dict[str, int],
        signed: Dict[str, bool],
    ):
        self.sigmap = sigmap
        self.widths = widths
        self.signed = signed

    def name(self, sig: str) -> str:
        """Resolve a signal name to its Verilog identifier or constant literal."""
        if sig == "#0":
            return "1'b0"
        if sig == "#1":
            return "1'b1"
        return self.sigmap.get(sig, _sanitize(sig))

    def width(self, sig: str) -> int:
        if sig in ("#0", "#1"):
            return 1
        return self.widths.get(sig, 1)

    def is_signed(self, sig: str) -> bool:
        if sig in ("#0", "#1"):
            return False
        return self.signed.get(sig, False)


# ---- Lowering registry ------------------------------------------------------


LoweringFn = Callable[[EmitContext, Gate], List[str]]
_lowerings: Dict[str, LoweringFn] = {}


def lowering(kind: str) -> Callable[[LoweringFn], LoweringFn]:
    """Decorator that registers a backend lowering for a Gate.kind."""
    def deco(fn: LoweringFn) -> LoweringFn:
        _lowerings[kind] = fn
        return fn
    return deco


def registered_kinds() -> List[str]:
    return sorted(_lowerings)


# ---- Built-in lowerings -----------------------------------------------------


def _format_sum(terms: List[str]) -> str:
    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "(" + " + ".join(terms) + ")"


@lowering("threshold")
def _lower_threshold(ctx: EmitContext, g: Gate) -> List[str]:
    """Σ wᵢ·xᵢ + bias ≥ 0. attrs: weights:list[int], bias:int. 1-bit output."""
    weights = list(g.attrs.get("weights", []))
    bias = int(g.attrs.get("bias", 0))
    if len(weights) != len(g.inputs):
        raise ValueError(
            f"gate '{g.name}': len(weights)={len(weights)} != "
            f"len(inputs)={len(g.inputs)}"
        )

    pos: List[str] = []
    neg: List[str] = []
    for w, sig in zip(weights, g.inputs):
        if w == 0:
            continue
        s = ctx.name(sig)
        if w == 1:
            pos.append(s)
        elif w == -1:
            neg.append(s)
        elif w > 1:
            pos.append(f"{w}*{s}")
        else:
            neg.append(f"{-w}*{s}")

    if bias > 0:
        pos.append(str(bias))
    elif bias < 0:
        neg.append(str(-bias))

    lhs = _format_sum(pos)
    rhs = _format_sum(neg)
    return [f"  assign {ctx.name(g.name)} = ({lhs} >= {rhs});"]


@lowering("constant")
def _lower_constant(ctx: EmitContext, g: Gate) -> List[str]:
    """attrs: value:int."""
    value = int(g.attrs["value"])
    width = max(1, g.output_width)
    sign_letter = "s" if g.output_signed else ""
    mask = (1 << width) - 1
    return [
        f"  assign {ctx.name(g.name)} = {width}'{sign_letter}d{value & mask};"
    ]


def _arith2(op: str, ctx: EmitContext, g: Gate) -> List[str]:
    if len(g.inputs) != 2:
        raise ValueError(
            f"gate '{g.name}' kind '{g.kind}' expects 2 inputs, got {len(g.inputs)}"
        )
    a, b = g.inputs
    sa = "$signed" if ctx.is_signed(a) or g.output_signed else ""
    sb = "$signed" if ctx.is_signed(b) or g.output_signed else ""
    return [
        f"  assign {ctx.name(g.name)} = {sa}({ctx.name(a)}) {op} {sb}({ctx.name(b)});"
    ]


@lowering("add")
def _lower_add(ctx, g): return _arith2("+", ctx, g)


@lowering("sub")
def _lower_sub(ctx, g): return _arith2("-", ctx, g)


@lowering("mul")
def _lower_mul(ctx, g): return _arith2("*", ctx, g)


def _bitwise(op: str, ctx: EmitContext, g: Gate) -> List[str]:
    if not g.inputs:
        raise ValueError(f"gate '{g.name}' kind '{g.kind}' needs at least 1 input")
    expr = (" " + op + " ").join(ctx.name(s) for s in g.inputs)
    return [f"  assign {ctx.name(g.name)} = {expr};"]


@lowering("and")
def _lower_and(ctx, g): return _bitwise("&", ctx, g)


@lowering("or")
def _lower_or(ctx, g): return _bitwise("|", ctx, g)


@lowering("xor")
def _lower_xor(ctx, g): return _bitwise("^", ctx, g)


@lowering("not")
def _lower_not(ctx: EmitContext, g: Gate) -> List[str]:
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'not' expects 1 input")
    return [f"  assign {ctx.name(g.name)} = ~{ctx.name(g.inputs[0])};"]


@lowering("shift_left")
def _lower_shl(ctx: EmitContext, g: Gate) -> List[str]:
    """attrs: amount:int (default 1)."""
    amt = int(g.attrs.get("amount", 1))
    return [f"  assign {ctx.name(g.name)} = {ctx.name(g.inputs[0])} << {amt};"]


@lowering("shift_right")
def _lower_shr(ctx: EmitContext, g: Gate) -> List[str]:
    amt = int(g.attrs.get("amount", 1))
    op = ">>>" if g.output_signed else ">>"
    s = "$signed" if g.output_signed else ""
    return [
        f"  assign {ctx.name(g.name)} = {s}({ctx.name(g.inputs[0])}) {op} {amt};"
    ]


@lowering("concat")
def _lower_concat(ctx: EmitContext, g: Gate) -> List[str]:
    if not g.inputs:
        raise ValueError(f"gate '{g.name}' kind 'concat' needs at least 1 input")
    items = ", ".join(ctx.name(s) for s in g.inputs)
    return [f"  assign {ctx.name(g.name)} = {{{items}}};"]


@lowering("slice")
def _lower_slice(ctx: EmitContext, g: Gate) -> List[str]:
    """attrs: hi:int, lo:int."""
    hi = int(g.attrs["hi"])
    lo = int(g.attrs["lo"])
    src = ctx.name(g.inputs[0])
    if hi == lo:
        return [f"  assign {ctx.name(g.name)} = {src}[{hi}];"]
    return [f"  assign {ctx.name(g.name)} = {src}[{hi}:{lo}];"]


@lowering("mux")
def _lower_mux(ctx: EmitContext, g: Gate) -> List[str]:
    """inputs[0]=sel, inputs[1..N]=data choices indexed by sel."""
    if len(g.inputs) < 3:
        raise ValueError(f"gate '{g.name}' kind 'mux' needs sel + at least 2 data inputs")
    sel = ctx.name(g.inputs[0])
    data = g.inputs[1:]
    if len(data) == 2:
        return [
            f"  assign {ctx.name(g.name)} = {sel} ? "
            f"{ctx.name(data[1])} : {ctx.name(data[0])};"
        ]
    expr = ctx.name(data[-1])
    for i in range(len(data) - 2, -1, -1):
        expr = f"({sel} == {i}) ? {ctx.name(data[i])} : ({expr})"
    return [f"  assign {ctx.name(g.name)} = {expr};"]


@lowering("relu")
def _lower_relu(ctx: EmitContext, g: Gate) -> List[str]:
    """y = max(0, x). For unsigned x this is identity (always >= 0)."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'relu' expects 1 input")
    inp = ctx.name(g.inputs[0])
    if g.output_signed or ctx.is_signed(g.inputs[0]):
        width = max(1, g.output_width)
        return [
            f"  assign {ctx.name(g.name)} = ($signed({inp}) > 0) ? {inp} : "
            f"{width}'sd0;"
        ]
    return [f"  assign {ctx.name(g.name)} = {inp};"]


@lowering("clamp")
def _lower_clamp(ctx: EmitContext, g: Gate) -> List[str]:
    """attrs: lo:int, hi:int. y = clamp(x, lo, hi)."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'clamp' expects 1 input")
    inp = ctx.name(g.inputs[0])
    lo = int(g.attrs["lo"])
    hi = int(g.attrs["hi"])
    s = "$signed" if g.output_signed or ctx.is_signed(g.inputs[0]) else ""
    return [
        f"  assign {ctx.name(g.name)} = ({s}({inp}) > {hi}) ? {hi} : "
        f"(({s}({inp}) < {lo}) ? {lo} : {inp});"
    ]


@lowering("register")
def _lower_register(ctx: EmitContext, g: Gate) -> List[str]:
    """Synchronous flip-flop. inputs=[d]. attrs: clk='clk', rst (optional), init=0."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'register' expects 1 input")
    d = ctx.name(g.inputs[0])
    clk = g.attrs.get("clk", "clk")
    rst = g.attrs.get("rst")
    init = int(g.attrs.get("init", 0))
    name = ctx.name(g.name)
    width = max(1, g.output_width)
    init_lit = f"{width}'d{init & ((1 << width) - 1)}"
    if rst:
        return [
            f"  always @(posedge {clk} or posedge {rst}) begin",
            f"    if ({rst}) {name} <= {init_lit};",
            f"    else {name} <= {d};",
            f"  end",
        ]
    return [f"  always @(posedge {clk}) {name} <= {d};"]


@lowering("rom")
def _lower_rom(ctx: EmitContext, g: Gate) -> List[str]:
    """Parameter ROM. inputs=[addr]. attrs: init:list[int], width:int, depth:int."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'rom' expects 1 input (address)")
    addr = ctx.name(g.inputs[0])
    init: List[int] = list(g.attrs["init"])
    width = int(g.attrs["width"])
    depth = int(g.attrs["depth"])
    if depth != len(init):
        raise ValueError(
            f"gate '{g.name}' rom: depth={depth} but init has {len(init)} entries"
        )
    name = ctx.name(g.name)
    rom = f"{name}_mem"
    mask = (1 << width) - 1
    lines = [f"  reg [{width-1}:0] {rom} [0:{depth-1}];", "  initial begin"]
    for i, v in enumerate(init):
        lines.append(f"    {rom}[{i}] = {width}'d{int(v) & mask};")
    lines.append("  end")
    lines.append(f"  assign {name} = {rom}[{addr}];")
    return lines


# ---- Top-level emit ---------------------------------------------------------


def emit_module(graph: GateGraph) -> str:
    """Emit the full Verilog text for a GateGraph.

    Gates must be topologically sorted: every input of every gate must
    be either a constant sentinel ("#0", "#1"), an external input, or
    the output of an earlier gate. Module name and signal names are
    sanitized; collisions are resolved by appending '_u<N>'.
    """
    _validate_module_name(graph.top)

    used: Set[str] = set()
    sigmap: Dict[str, str] = {}

    for s in graph.inputs:
        sigmap[s.name] = _sanitize(s.name, used)
    for s in graph.outputs:
        if s.name not in sigmap:
            sigmap[s.name] = _sanitize(s.name, used)
    for g in graph.gates:
        if g.name not in sigmap:
            sigmap[g.name] = _sanitize(g.name, used)

    widths: Dict[str, int] = {}
    signed: Dict[str, bool] = {}
    for s in list(graph.inputs) + list(graph.outputs):
        widths[s.name] = max(1, s.width)
        signed[s.name] = s.signed
    for g in graph.gates:
        widths[g.name] = max(1, g.output_width)
        signed[g.name] = g.output_signed

    declared: Set[str] = {s.name for s in graph.inputs} | {"#0", "#1"}
    for g in graph.gates:
        if g.kind not in _lowerings:
            raise ValueError(
                f"no backend lowering registered for kind '{g.kind}' "
                f"(gate '{g.name}'). Registered: {registered_kinds()}"
            )
        for src in g.inputs:
            if src not in declared:
                raise ValueError(
                    f"gate '{g.name}' references undeclared signal '{src}'. "
                    f"GateGraph.gates must be topologically sorted."
                )
        declared.add(g.name)

    for s in graph.outputs:
        if s.name not in declared:
            raise ValueError(f"output '{s.name}' is never produced by any gate")

    has_register = any(g.kind == "register" for g in graph.gates)
    has_reset = any(
        g.kind == "register" and g.attrs.get("rst") for g in graph.gates
    )

    ctx = EmitContext(sigmap, widths, signed)

    lines: List[str] = []
    lines.append("// Generated by safetensors2verilog.")
    lines.append("`default_nettype none")
    lines.append("")

    port_decls: List[str] = []
    explicit_input_names = {s.name for s in graph.inputs}
    if has_register and "clk" not in explicit_input_names:
        port_decls.append("  input wire clk")
    if has_reset and "rst" not in explicit_input_names:
        port_decls.append("  input wire rst")

    register_outputs = {g.name for g in graph.gates if g.kind == "register"}

    for s in graph.inputs:
        port_decls.append(
            _signal_decl("input", "wire", sigmap[s.name],
                         max(1, s.width), s.signed)
        )
    for s in graph.outputs:
        kw = "reg" if s.name in register_outputs else "wire"
        port_decls.append(
            _signal_decl("output", kw, sigmap[s.name],
                         max(1, s.width), s.signed)
        )

    lines.append(f"module {graph.top} (")
    lines.append(",\n".join(port_decls))
    lines.append(");")
    lines.append("")

    output_names = {s.name for s in graph.outputs}
    internals = [g for g in graph.gates if g.name not in output_names]
    if internals:
        lines.append("  // internal nets")
        for g in internals:
            kw = "reg" if g.kind == "register" else "wire"
            lines.append(
                _signal_decl(None, kw, sigmap[g.name],
                             max(1, g.output_width), g.output_signed) + ";"
            )
        lines.append("")

    lines.append("  // gate evaluations")
    for g in graph.gates:
        lines.extend(_lowerings[g.kind](ctx, g))

    lines.append("")
    lines.append(f"endmodule // {graph.top}")
    lines.append("`default_nettype wire")
    lines.append("")

    return "\n".join(lines)


# ---- BRAM template (for memory carve-out) -----------------------------------


def emit_bram_template(
    addr_bits: int = 16, data_bits: int = 8, module_name: str = "threshold_bram"
) -> str:
    """Synchronous single-port BRAM that vendor synth tools infer as block RAM."""
    _validate_module_name(module_name)
    if addr_bits < 1 or addr_bits > 30:
        raise ValueError(f"addr_bits must be in [1, 30], got {addr_bits}")
    if data_bits < 1 or data_bits > 1024:
        raise ValueError(f"data_bits must be in [1, 1024], got {data_bits}")
    mem_size = 1 << addr_bits
    return (
        "// Generated by safetensors2verilog. Single-port synchronous BRAM template.\n"
        "// Vendor synthesis tools will infer this as block RAM (Xilinx XPM,\n"
        "// Lattice EBR/SPRAM, Intel M9K, etc.).\n"
        "\n"
        "`default_nettype none\n"
        "\n"
        f"module {module_name} (\n"
        "  input  wire                       clk,\n"
        "  input  wire                       we,\n"
        f"  input  wire [{addr_bits - 1}:0]              addr,\n"
        f"  input  wire [{data_bits - 1}:0]               data_in,\n"
        f"  output reg  [{data_bits - 1}:0]               data_out\n"
        ");\n"
        "\n"
        f"  reg [{data_bits - 1}:0] mem [0:{mem_size - 1}];\n"
        "\n"
        "  always @(posedge clk) begin\n"
        "    if (we) begin\n"
        "      mem[addr] <= data_in;\n"
        "    end\n"
        "    data_out <= mem[addr];\n"
        "  end\n"
        "\n"
        f"endmodule // {module_name}\n"
        "`default_nettype wire\n"
    )
