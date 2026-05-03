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
from collections.abc import Callable

from .core import Gate, GateGraph

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


def _sanitize(name: str, used: set[str] | None = None) -> str:
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
    direction: str | None,
    keyword: str,
    sig_name: str,
    width: int,
    signed: bool,
) -> str:
    parts: list[str] = []
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
        sigmap: dict[str, str],
        widths: dict[str, int],
        signed: dict[str, bool],
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


LoweringFn = Callable[[EmitContext, Gate], list[str]]
_lowerings: dict[str, LoweringFn] = {}


def lowering(kind: str) -> Callable[[LoweringFn], LoweringFn]:
    """Decorator that registers a backend lowering for a Gate.kind."""
    def deco(fn: LoweringFn) -> LoweringFn:
        _lowerings[kind] = fn
        return fn
    return deco


def registered_kinds() -> list[str]:
    return sorted(_lowerings)


# ---- Built-in lowerings -----------------------------------------------------


def _format_sum(terms: list[str]) -> str:
    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "(" + " + ".join(terms) + ")"


@lowering("threshold")
def _lower_threshold(ctx: EmitContext, g: Gate) -> list[str]:
    """Σ wᵢ·xᵢ + bias ≥ 0. attrs: weights:list[int], bias:int. 1-bit output."""
    weights = list(g.attrs.get("weights", []))
    bias = int(g.attrs.get("bias", 0))
    if len(weights) != len(g.inputs):
        raise ValueError(
            f"gate '{g.name}': len(weights)={len(weights)} != "
            f"len(inputs)={len(g.inputs)}"
        )

    pos: list[str] = []
    neg: list[str] = []
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
def _lower_constant(ctx: EmitContext, g: Gate) -> list[str]:
    """attrs: value:int."""
    value = int(g.attrs["value"])
    width = max(1, g.output_width)
    sign_letter = "s" if g.output_signed else ""
    mask = (1 << width) - 1
    return [
        f"  assign {ctx.name(g.name)} = {width}'{sign_letter}d{value & mask};"
    ]


@lowering("linear")
def _lower_linear(ctx: EmitContext, g: Gate) -> list[str]:
    """y = Σ wᵢ·xᵢ + bias at multibit precision.

    attrs:
      weights : list[int], same length as inputs
      bias    : int

    Like 'threshold' but emits the raw weighted sum instead of a
    comparison. Multibit, signed when output_signed is true. Per-input
    signedness is honored via $signed() casts. Zero-weighted inputs
    are dropped from the expression.
    """
    weights = list(g.attrs.get("weights", []))
    bias = int(g.attrs.get("bias", 0))
    if len(weights) != len(g.inputs):
        raise ValueError(
            f"gate '{g.name}' kind 'linear': len(weights)={len(weights)} != "
            f"len(inputs)={len(g.inputs)}"
        )

    terms: list[str] = []
    for w, sig in zip(weights, g.inputs):
        if w == 0:
            continue
        s = ctx.name(sig)
        is_signed = ctx.is_signed(sig) or g.output_signed
        cast = "$signed" if is_signed else ""
        if w == 1:
            terms.append(f"{cast}({s})" if cast else s)
        elif w == -1:
            terms.append(f"-{cast}({s})" if cast else f"-{s}")
        else:
            terms.append(f"{w}*{cast}({s})" if cast else f"{w}*{s}")

    if bias != 0:
        terms.append(str(bias))
    if not terms:
        terms.append("0")

    return [
        f"  assign {ctx.name(g.name)} = " + " + ".join(terms) + ";"
    ]


def _arith2(op: str, ctx: EmitContext, g: Gate) -> list[str]:
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
def _lower_mul(ctx, g):
    """Multibit multiply. Set attrs={'use_dsp': True} to emit a synthesis
    pragma directing Vivado / Quartus to place the operation in a DSP block.
    """
    lines = _arith2("*", ctx, g)
    if g.attrs.get("use_dsp"):
        lines.insert(0, '  (* use_dsp = "yes" *)')
    return lines


def _bitwise(op: str, ctx: EmitContext, g: Gate) -> list[str]:
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
def _lower_not(ctx: EmitContext, g: Gate) -> list[str]:
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'not' expects 1 input")
    return [f"  assign {ctx.name(g.name)} = ~{ctx.name(g.inputs[0])};"]


@lowering("shift_left")
def _lower_shl(ctx: EmitContext, g: Gate) -> list[str]:
    """attrs: amount:int (default 1)."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'shift_left' expects 1 input")
    amt = int(g.attrs.get("amount", 1))
    return [f"  assign {ctx.name(g.name)} = {ctx.name(g.inputs[0])} << {amt};"]


@lowering("shift_right")
def _lower_shr(ctx: EmitContext, g: Gate) -> list[str]:
    """attrs: amount:int (default 1). Arithmetic shift when output_signed."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'shift_right' expects 1 input")
    amt = int(g.attrs.get("amount", 1))
    op = ">>>" if g.output_signed else ">>"
    s = "$signed" if g.output_signed else ""
    return [
        f"  assign {ctx.name(g.name)} = {s}({ctx.name(g.inputs[0])}) {op} {amt};"
    ]


@lowering("concat")
def _lower_concat(ctx: EmitContext, g: Gate) -> list[str]:
    """Concatenate inputs MSB-first; output_width must equal sum of input widths."""
    if not g.inputs:
        raise ValueError(f"gate '{g.name}' kind 'concat' needs at least 1 input")
    total = sum(ctx.width(s) for s in g.inputs)
    if total != g.output_width:
        raise ValueError(
            f"gate '{g.name}' kind 'concat': sum of input widths = {total} "
            f"but output_width = {g.output_width}"
        )
    items = ", ".join(ctx.name(s) for s in g.inputs)
    return [f"  assign {ctx.name(g.name)} = {{{items}}};"]


@lowering("slice")
def _lower_slice(ctx: EmitContext, g: Gate) -> list[str]:
    """attrs: hi:int, lo:int. output_width must equal hi - lo + 1."""
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'slice' expects 1 input")
    hi = int(g.attrs["hi"])
    lo = int(g.attrs["lo"])
    if hi < lo:
        raise ValueError(f"gate '{g.name}' kind 'slice': hi={hi} < lo={lo}")
    expected_width = hi - lo + 1
    if expected_width != g.output_width:
        raise ValueError(
            f"gate '{g.name}' kind 'slice': hi-lo+1 = {expected_width} "
            f"but output_width = {g.output_width}"
        )
    src_width = ctx.width(g.inputs[0])
    if hi >= src_width:
        raise ValueError(
            f"gate '{g.name}' kind 'slice': hi={hi} but input '{g.inputs[0]}' "
            f"is only {src_width} bits wide"
        )
    src = ctx.name(g.inputs[0])
    if hi == lo:
        return [f"  assign {ctx.name(g.name)} = {src}[{hi}];"]
    return [f"  assign {ctx.name(g.name)} = {src}[{hi}:{lo}];"]


@lowering("mux")
def _lower_mux(ctx: EmitContext, g: Gate) -> list[str]:
    """inputs[0]=sel, inputs[1..N]=data choices indexed by sel.

    All data inputs must match output_width. For more than 4 data
    cases the lowering emits a `case` statement rather than a chained
    ternary; vendor synthesis tends to recognise the case form more
    cleanly as a hardware mux.
    """
    if len(g.inputs) < 3:
        raise ValueError(
            f"gate '{g.name}' kind 'mux' needs sel + at least 2 data inputs"
        )
    sel = ctx.name(g.inputs[0])
    data = g.inputs[1:]
    for d in data:
        w = ctx.width(d)
        if w != g.output_width:
            raise ValueError(
                f"gate '{g.name}' kind 'mux': data input '{d}' is {w} bits "
                f"but output_width = {g.output_width}"
            )
    if len(data) == 2:
        return [
            f"  assign {ctx.name(g.name)} = {sel} ? "
            f"{ctx.name(data[1])} : {ctx.name(data[0])};"
        ]
    if len(data) <= 4:
        # chained ternary: small enough to read as a single line
        expr = ctx.name(data[-1])
        for i in range(len(data) - 2, -1, -1):
            expr = f"({sel} == {i}) ? {ctx.name(data[i])} : ({expr})"
        return [f"  assign {ctx.name(g.name)} = {expr};"]
    # case form for larger muxes
    out = ctx.name(g.name)
    lines = ["  always @(*) begin"]
    lines.append(f"    case ({sel})")
    for i, d in enumerate(data):
        lines.append(f"      {i}: {out} = {ctx.name(d)};")
    lines.append(f"      default: {out} = {ctx.name(data[-1])};")
    lines.append("    endcase")
    lines.append("  end")
    return lines


@lowering("eq")
def _lower_eq(ctx: EmitContext, g: Gate) -> list[str]:
    """1-bit equality comparison: y = (a == b)."""
    if len(g.inputs) != 2:
        raise ValueError(f"gate '{g.name}' kind 'eq' expects 2 inputs")
    a, b = g.inputs
    return [
        f"  assign {ctx.name(g.name)} = ({ctx.name(a)} == {ctx.name(b)});"
    ]


@lowering("relu")
def _lower_relu(ctx: EmitContext, g: Gate) -> list[str]:
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
def _lower_clamp(ctx: EmitContext, g: Gate) -> list[str]:
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
def _lower_register(ctx: EmitContext, g: Gate) -> list[str]:
    """Synchronous flip-flop.

    inputs : [d]  (or [d, en] when an enable signal is supplied)

    attrs:
      clk             clock signal name (default "clk")
      rst             reset signal name (optional)
      init            integer reset value (default 0)
      reset_polarity  "high" (default) or "low"
      reset_kind      "async" (default) or "sync"
      enable          enable signal name (optional); takes precedence
                      over the second `inputs` element when both supplied
    """
    if len(g.inputs) not in (1, 2):
        raise ValueError(
            f"gate '{g.name}' kind 'register' expects 1 or 2 inputs"
        )
    d = ctx.name(g.inputs[0])
    en_signal = g.attrs.get("enable")
    if en_signal is None and len(g.inputs) == 2:
        en_signal = g.inputs[1]
    clk = g.attrs.get("clk", "clk")
    rst = g.attrs.get("rst")
    init = int(g.attrs.get("init", 0))
    polarity = g.attrs.get("reset_polarity", "high")
    kind = g.attrs.get("reset_kind", "async")
    if polarity not in ("high", "low"):
        raise ValueError(
            f"gate '{g.name}' register: reset_polarity must be 'high' or 'low'"
        )
    if kind not in ("async", "sync"):
        raise ValueError(
            f"gate '{g.name}' register: reset_kind must be 'async' or 'sync'"
        )

    name = ctx.name(g.name)
    width = max(1, g.output_width)
    init_lit = f"{width}'d{init & ((1 << width) - 1)}"

    rst_test = f"{rst}" if polarity == "high" else f"!{rst}"
    rst_edge = f"posedge {rst}" if polarity == "high" else f"negedge {rst}"

    body_assign = (
        f"{name} <= {en_signal} ? {d} : {name};"
        if en_signal else f"{name} <= {d};"
    )

    if rst:
        if kind == "async":
            return [
                f"  always @(posedge {clk} or {rst_edge}) begin",
                f"    if ({rst_test}) {name} <= {init_lit};",
                f"    else {body_assign}",
                "  end",
            ]
        return [
            f"  always @(posedge {clk}) begin",
            f"    if ({rst_test}) {name} <= {init_lit};",
            f"    else {body_assign}",
            "  end",
        ]
    return [f"  always @(posedge {clk}) {body_assign}"]


@lowering("tristate")
def _lower_tristate(ctx: EmitContext, g: Gate) -> list[str]:
    """3-state buffer: y = en ? data : 'bz.

    inputs : [data, en]
    attrs:
      enable_high   if True (default), drive data when en==1; else when en==0
    """
    if len(g.inputs) != 2:
        raise ValueError(
            f"gate '{g.name}' kind 'tristate' expects [data, en]"
        )
    data = ctx.name(g.inputs[0])
    en = ctx.name(g.inputs[1])
    enable_high = bool(g.attrs.get("enable_high", True))
    width = max(1, g.output_width)
    z_lit = f"{width}'bz" if width > 1 else "1'bz"
    if enable_high:
        return [f"  assign {ctx.name(g.name)} = {en} ? {data} : {z_lit};"]
    return [f"  assign {ctx.name(g.name)} = {en} ? {z_lit} : {data};"]


@lowering("parameter")
def _lower_parameter(ctx: EmitContext, g: Gate) -> list[str]:
    """Compile-time constant emitted as a Verilog `localparam` declaration.

    attrs: value:int. Width is taken from output_width.
    """
    value = int(g.attrs["value"])
    width = max(1, g.output_width)
    sign_letter = "s" if g.output_signed else ""
    mask = (1 << width) - 1
    return [
        f"  localparam [{width-1}:0] {ctx.name(g.name)} = "
        f"{width}'{sign_letter}d{value & mask};"
    ]


@lowering("rom")
def _lower_rom(ctx: EmitContext, g: Gate) -> list[str]:
    """Parameter ROM. inputs=[addr].

    attrs:
      init       list[int]; entries beyond len(init) are zero-padded up to depth
      width      int, bit width of each entry
      depth      int, number of entries (>= len(init))
      ram_style  optional str, e.g. 'block' or 'distributed' (vendor-specific
                 attribute; rendered as a synthesis pragma when present)
    """
    if len(g.inputs) != 1:
        raise ValueError(f"gate '{g.name}' kind 'rom' expects 1 input (address)")
    addr = ctx.name(g.inputs[0])
    init: list[int] = list(g.attrs["init"])
    width = int(g.attrs["width"])
    depth = int(g.attrs["depth"])
    if depth < len(init):
        raise ValueError(
            f"gate '{g.name}' rom: depth={depth} smaller than init size {len(init)}"
        )
    if width < 1:
        raise ValueError(f"gate '{g.name}' rom: width must be >= 1, got {width}")
    if width != g.output_width:
        raise ValueError(
            f"gate '{g.name}' rom: width={width} != output_width={g.output_width}"
        )
    name = ctx.name(g.name)
    rom = f"{name}_mem"
    mask = (1 << width) - 1
    lines: list[str] = []
    ram_style = g.attrs.get("ram_style")
    if ram_style:
        lines.append(f'  (* ram_style = "{ram_style}" *)')
    lines.append(f"  reg [{width-1}:0] {rom} [0:{depth-1}];")
    lines.append("  initial begin")
    for i in range(depth):
        v = init[i] if i < len(init) else 0
        lines.append(f"    {rom}[{i}] = {width}'d{int(v) & mask};")
    lines.append("  end")
    lines.append(f"  assign {name} = {rom}[{addr}];")
    return lines


# ---- Top-level emit ---------------------------------------------------------


def _gate_needs_reg(g: Gate) -> bool:
    """Gates whose lowering emits an `always` block need a `reg` declaration."""
    if g.kind == "register":
        return True
    if g.kind == "mux":
        # The mux lowering switches to case-form for >4 data inputs (>5 total)
        return len(g.inputs) > 5
    return False


_BUS_BIT_RE = re.compile(r"^(?P<base>.+)\[(?P<idx>\d+)\]$")


def _detect_buses(
    signals: list,
) -> tuple[dict[str, list[tuple[int, str]]], set[str]]:
    """Group signals of the form 'base[i]' into per-base buses.

    Returns (buses, members) where:
      buses[base]   -> sorted (index, original_name) pairs, low index first
      members       -> set of original signal names that became bus members

    Accepts any contiguous index range (`[0..N-1]`, `[1..N]`, descending
    `[N-1..0]`, etc.) so long as the indices form a gap-free interval and
    there are at least two of them. Bus members must be 1-bit signals.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for s in signals:
        m = _BUS_BIT_RE.match(s.name)
        if not m:
            continue
        if s.width != 1:
            continue
        base = m.group("base")
        idx = int(m.group("idx"))
        groups.setdefault(base, []).append((idx, s.name))

    buses: dict[str, list[tuple[int, str]]] = {}
    members: set[str] = set()
    for base, items in groups.items():
        items.sort()
        indices = [i for i, _ in items]
        if len(indices) < 2:
            continue
        lo, hi = indices[0], indices[-1]
        if indices == list(range(lo, hi + 1)):
            buses[base] = items
            for _, nm in items:
                members.add(nm)
    return buses, members


def _group_by_prefix(
    decls: list[tuple[str, str]], depth: int = 2,
) -> list[tuple[str, list[str]]]:
    """Group (signal_name, decl_text) pairs by their leading dotted segments.

    Uses the first ``depth`` dot-separated segments of each name as the
    group label. Signals with fewer than ``depth`` segments are grouped
    under their own full name; signals with no dots fall under the empty
    prefix ''. Order of first-appearance is preserved.
    """
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for name, text in decls:
        parts = name.split(".")
        if len(parts) <= 1:
            prefix = ""
        else:
            prefix = ".".join(parts[:depth])
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append(text)
    return [(p, groups[p]) for p in order]


def emit_module(graph: GateGraph, target: str = "verilog",
                pack_buses: bool = False,
                group_ports: bool = False) -> str:
    """Emit the full Verilog text for a GateGraph.

    target:
      "verilog" (default) — Verilog-2001 / 2005 with `wire`/`reg`/`always`
      "sv" or "systemverilog" — SystemVerilog with `logic`,
        `always_comb`, `always_ff`. Same logical output, friendlier
        for SV-aware simulators and synthesis tools.

    Gates must be topologically sorted: every input of every gate must
    be either a constant sentinel ("#0", "#1"), an external input, or
    the output of an earlier gate. Module name and signal names are
    sanitized; collisions are resolved by appending '_u<N>'.
    """
    if target in ("sv", "systemverilog"):
        text = _emit_module_internal(
            graph, pack_buses=pack_buses, group_ports=group_ports
        )
        return _sv_translate(text)
    if target != "verilog":
        raise ValueError(
            f"unknown target '{target}'; valid: 'verilog', 'sv'"
        )
    return _emit_module_internal(
        graph, pack_buses=pack_buses, group_ports=group_ports
    )


def _sv_translate(verilog_text: str) -> str:
    """Rewrite Verilog-2001 emit to SystemVerilog idioms."""
    out = verilog_text
    # logic for wires and regs (declarations only, not bit-vector indexing)
    out = re.sub(r"\bwire signed", "logic signed", out)
    out = re.sub(r"\bwire ", "logic ", out)
    out = re.sub(r"\breg signed", "logic signed", out)
    out = re.sub(r"\breg ", "logic ", out)
    # always blocks
    out = re.sub(r"\balways @\(\*\)", "always_comb", out)
    out = re.sub(r"\balways @\(posedge", "always_ff @(posedge", out)
    return out


def _emit_module_internal(graph: GateGraph, pack_buses: bool = False,
                          group_ports: bool = False) -> str:
    _validate_module_name(graph.top)

    used: set[str] = set()
    sigmap: dict[str, str] = {}

    # Detect bus families before flat-name sanitization so that bus
    # members (e.g. '$a[0]'..'$a[7]') get mapped to a Verilog bit-select
    # like 'a[0]' rather than each becoming its own scalar port.
    input_buses: dict[str, list[tuple[int, str]]] = {}
    input_bus_members: set[str] = set()
    if pack_buses:
        input_buses, input_bus_members = _detect_buses(graph.inputs)
        for base, items in input_buses.items():
            # Pick a clean Verilog name for the bus (sanitized, possibly
            # stripped of the leading `$` that marks external ports).
            bus_id = _sanitize(
                base[1:] if base.startswith("$") else base, used
            )
            for idx, original_name in items:
                sigmap[original_name] = f"{bus_id}[{idx}]"
            sigmap[("__bus__", base)] = bus_id   # type: ignore[index]

    for s in graph.inputs:
        if s.name in sigmap:
            continue
        sigmap[s.name] = _sanitize(s.name, used)
    for s in graph.outputs:
        if s.name not in sigmap:
            sigmap[s.name] = _sanitize(s.name, used)
    for g in graph.gates:
        if g.name not in sigmap:
            sigmap[g.name] = _sanitize(g.name, used)

    widths: dict[str, int] = {}
    signed: dict[str, bool] = {}
    for s in list(graph.inputs) + list(graph.outputs):
        widths[s.name] = max(1, s.width)
        signed[s.name] = s.signed
    for g in graph.gates:
        widths[g.name] = max(1, g.output_width)
        signed[g.name] = g.output_signed

    # Register outputs are pre-declared: a `register` gate's D input is a
    # *sequential* edge (sampled on a clock), not a combinational dependency,
    # so it doesn't need to be earlier in the topo order. This lets users
    # build counters and accumulators (output -> +1 -> register -> output)
    # without the topo sort rejecting the feedback as a cycle.
    register_outputs = {g.name for g in graph.gates if g.kind == "register"}
    parameter_outputs = {g.name for g in graph.gates if g.kind == "parameter"}
    declared: set[str] = (
        {s.name for s in graph.inputs}
        | {"#0", "#1"}
        | register_outputs
        | parameter_outputs
    )

    all_signals = (
        {s.name for s in graph.inputs}
        | {"#0", "#1"}
        | {g.name for g in graph.gates}
    )

    for g in graph.gates:
        if g.kind not in _lowerings:
            raise ValueError(
                f"no backend lowering registered for kind '{g.kind}' "
                f"(gate '{g.name}'). Registered: {registered_kinds()}"
            )
        if g.kind == "register":
            # D input only needs to be produced *somewhere* in the graph.
            for src in g.inputs:
                if src not in all_signals:
                    raise ValueError(
                        f"register gate '{g.name}' D input '{src}' is not "
                        f"produced by any gate or external input"
                    )
            continue
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

    lines: list[str] = []
    lines.append("// Generated by safetensors2verilog.")
    lines.append("`default_nettype none")
    lines.append("")

    port_decls: list[str] = []
    port_decl_pairs: list[tuple[str, str]] = []   # (name, decl) for grouping
    explicit_input_names = {s.name for s in graph.inputs}
    if has_register and "clk" not in explicit_input_names:
        port_decls.append("  input wire clk")
        port_decl_pairs.append(("clk", "  input wire clk"))
    if has_reset and "rst" not in explicit_input_names:
        port_decls.append("  input wire rst")
        port_decl_pairs.append(("rst", "  input wire rst"))

    # Module parameter declarations come before the port list as #(...).
    parameter_signals = [s for s in graph.inputs if s.is_parameter]
    if parameter_signals:
        param_decls = []
        for s in parameter_signals:
            sign_letter = "s" if s.signed else ""
            width = max(1, s.width)
            param_decls.append(
                f"  parameter [{width-1}:0] {sigmap[s.name]} = "
                f"{width}'{sign_letter}d{s.parameter_value & ((1 << width) - 1)}"
            )
        lines.append(f"module {graph.top} #(")
        lines.append(",\n".join(param_decls))
        lines.append(") (")
    else:
        lines.append(f"module {graph.top} (")

    emitted_bus_bases: set[str] = set()
    for s in graph.inputs:
        if s.is_parameter:
            continue
        if s.name in input_bus_members:
            # find which bus this member belongs to
            m = _BUS_BIT_RE.match(s.name)
            base = m.group("base") if m else ""
            if base and base in input_buses and base not in emitted_bus_bases:
                items = input_buses[base]
                lo = items[0][0]
                hi = items[-1][0]
                bus_id = sigmap[("__bus__", base)]   # type: ignore[index]
                decl = f"  input wire [{hi}:{lo}] {bus_id}"
                port_decls.append(decl)
                port_decl_pairs.append((base, decl))
                emitted_bus_bases.add(base)
            continue
        if s.direction == "inout":
            decl = _signal_decl("inout", "wire", sigmap[s.name],
                                max(1, s.width), s.signed)
        else:
            decl = _signal_decl("input", "wire", sigmap[s.name],
                                max(1, s.width), s.signed)
        port_decls.append(decl)
        port_decl_pairs.append((s.name, decl))
    # Track gates whose lowering needs `reg` declarations
    reg_gates = {g.name for g in graph.gates if _gate_needs_reg(g)}
    for s in graph.outputs:
        if s.direction == "inout":
            decl = _signal_decl("inout", "wire", sigmap[s.name],
                                max(1, s.width), s.signed)
        else:
            kw = "reg" if s.name in reg_gates else "wire"
            decl = _signal_decl("output", kw, sigmap[s.name],
                                max(1, s.width), s.signed)
        port_decls.append(decl)
        port_decl_pairs.append((s.name, decl))

    if group_ports and len(port_decl_pairs) > 8:
        groups = _group_by_prefix(port_decl_pairs)
        rendered: list[str] = []
        for prefix, decls in groups:
            if prefix:
                rendered.append(f"  // -- {prefix} --")
            rendered.extend(decls)
        # commas separate ports across the whole list, not group lines
        out_parts: list[str] = []
        first = True
        for line in rendered:
            if line.startswith("  //"):
                if not first:
                    out_parts.append(",")
                out_parts.append("\n" + line + "\n")
                first = True
            else:
                if not first:
                    out_parts.append(",\n")
                out_parts.append(line)
                first = False
        # Strip a trailing comma if any (won't happen but be defensive)
        lines.append("".join(out_parts))
    else:
        lines.append(",\n".join(port_decls))
    lines.append(");")
    lines.append("")

    output_names = {s.name for s in graph.outputs}
    internals = [g for g in graph.gates if g.name not in output_names]
    if internals:
        lines.append("  // internal nets")
        for g in internals:
            if g.kind == "parameter":
                # parameter gates emit their own `localparam` line in the
                # gate-evaluation section; skip wire/reg declaration.
                continue
            kw = "reg" if g.name in reg_gates else "wire"
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


# ---- Multi-clock validation -----------------------------------------------


def collect_clocks(graph: GateGraph) -> set[str]:
    """Return every distinct clock signal used by register gates in `graph`.

    Useful for frontends that want to surface multi-clock-domain warnings
    or to ensure each clock is in graph.inputs.
    """
    return {
        g.attrs.get("clk", "clk")
        for g in graph.gates if g.kind == "register"
    }


def collect_resets(graph: GateGraph) -> set[str]:
    """Return every distinct reset signal used by register gates in `graph`."""
    out: set[str] = set()
    for g in graph.gates:
        if g.kind == "register":
            rst = g.attrs.get("rst")
            if rst:
                out.add(str(rst))
    return out


# ---- Top-level wrapper helper (for memory carve-out) -----------------------


def emit_top_wrapper(
    core_module: str,
    bram_module: str,
    addr_bits: int = 16,
    data_bits: int = 8,
    wrapper_name: str = "top_wrapper",
    *,
    core_port_map: dict[str, str] | None = None,
    bram_port_map: dict[str, str] | None = None,
) -> str:
    """Emit a wrapper module that instantiates the carved-out core and the
    BRAM template, closing the memory loop carved out by ``--skip-memory``.

    Logical signal names are ``addr`` (read/write address), ``data_in``
    (data being written to memory), ``data_out`` (data read from memory),
    and ``we`` (write enable).

    `core_port_map` and `bram_port_map` map each logical signal to the
    actual port name on the corresponding module. Defaults match the
    BRAM template emitted by ``emit_bram_template``. Override when the
    core's external ports use different names: for instance, the
    8bit-threshold-computer carves out memory with consumers named like
    ``control.fetch.ir.$data[0]`` rather than ``data_in``; in that case
    you'd pass a port_map that maps each logical name to the corresponding
    promoted-external port (and may need a separate adapter for
    multi-bit-to-multi-port conversion).
    """
    _validate_module_name(core_module)
    _validate_module_name(bram_module)
    _validate_module_name(wrapper_name)

    default_map = {
        "clk": "clk", "rst": "rst",
        "addr": "addr", "data_in": "data_in",
        "data_out": "data_out", "we": "we",
    }
    cm = {**default_map, **(core_port_map or {})}
    bm = {**default_map, **(bram_port_map or {})}

    return (
        f"// Generated by safetensors2verilog. Wrapper around {core_module}\n"
        f"// and {bram_module}; closes the memory loop carved out by\n"
        f"// --skip-memory.\n"
        f"\n"
        f"`default_nettype none\n"
        f"\n"
        f"module {wrapper_name} (\n"
        f"  input  wire clk,\n"
        f"  input  wire rst\n"
        f");\n"
        f"\n"
        f"  wire [{addr_bits-1}:0] addr;\n"
        f"  wire [{data_bits-1}:0] data_in;\n"
        f"  wire [{data_bits-1}:0] data_out;\n"
        f"  wire we;\n"
        f"\n"
        f"  {core_module} core (\n"
        f"    .{cm['addr']}(addr), .{cm['data_in']}(data_in),\n"
        f"    .{cm['we']}(we), .{cm['data_out']}(data_out)\n"
        f"  );\n"
        f"\n"
        f"  {bram_module} bram (\n"
        f"    .{bm['clk']}(clk), .{bm['we']}(we), .{bm['addr']}(addr),\n"
        f"    .{bm['data_in']}(data_in), .{bm['data_out']}(data_out)\n"
        f"  );\n"
        f"\n"
        f"endmodule // {wrapper_name}\n"
        f"`default_nettype wire\n"
    )


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
