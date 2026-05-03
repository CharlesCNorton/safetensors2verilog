"""Command-line entry point.

  python -m safetensors2verilog input.safetensors --frontend NAME [-o out.v] [...]

Per-frontend flags are surfaced from `Frontend.options()`. To list the
registered frontends:

  python -m safetensors2verilog --list-frontends

To list a frontend's specific flags:

  python -m safetensors2verilog --list-frontend-options threshold_logic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importing this submodule has the side effect of registering frontends.
from . import frontends  # noqa: F401
from .core import FrontendOption, GateGraph, registry
from .verilog import emit_bram_template, emit_module


def _graph_to_jsonable(graph: GateGraph) -> dict:
    """Render a GateGraph as a plain dict for --emit-ir json output."""
    return {
        "top": graph.top,
        "inputs": [
            {"name": s.name, "width": s.width, "signed": s.signed}
            for s in graph.inputs
        ],
        "outputs": [
            {"name": s.name, "width": s.width, "signed": s.signed}
            for s in graph.outputs
        ],
        "gates": [
            {
                "name": g.name,
                "kind": g.kind,
                "inputs": list(g.inputs),
                "attrs": g.attrs,
                "output_width": g.output_width,
                "output_signed": g.output_signed,
            }
            for g in graph.gates
        ],
    }


def _add_frontend_options(
    parser: argparse.ArgumentParser, opts: list[FrontendOption]
) -> None:
    for opt in opts:
        flag = "--" + opt.name
        kwargs: dict = {"help": opt.help, "default": opt.default}
        if opt.type is bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = opt.type
            if opt.metavar:
                kwargs["metavar"] = opt.metavar
        parser.add_argument(flag, **kwargs)


def _format_option_default(opt: FrontendOption) -> str:
    if opt.type is bool:
        return f"flag (default: {bool(opt.default)})"
    if opt.default is None:
        return f"{opt.type.__name__} (no default)"
    return f"{opt.type.__name__} (default: {opt.default!r})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safetensors2verilog",
        description="Compile safetensors-stored networks to synthesis-ready Verilog.",
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="path to .safetensors file"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output Verilog file (default: stdout)",
    )
    parser.add_argument(
        "--frontend", type=str, default="threshold_logic",
        help="frontend module name (default: threshold_logic)",
    )
    parser.add_argument(
        "--top", type=str, default="top",
        help="top-level Verilog module name (default: top)",
    )
    parser.add_argument(
        "--list-frontends", action="store_true",
        help="print registered frontends and exit",
    )
    parser.add_argument(
        "--list-frontend-options", type=str, metavar="FRONTEND",
        help="print options accepted by a specific frontend and exit",
    )
    parser.add_argument(
        "--emit-bram-template", type=Path, default=None, metavar="PATH",
        help=(
            "alongside the main output, write a synchronous BRAM template "
            "(single-port read/write, configurable address width)."
        ),
    )
    parser.add_argument(
        "--bram-addr-bits", type=int, default=None,
        help=(
            "address width for --emit-bram-template (default: read from "
            "manifest.addr_bits when present, else 16)."
        ),
    )
    parser.add_argument(
        "--bram-data-bits", type=int, default=8,
        help="data width for --emit-bram-template (default: 8)",
    )
    parser.add_argument(
        "--emit-ir", choices=["json"], default=None,
        help=(
            "instead of (or alongside) Verilog, dump the IR as JSON. "
            "Goes to --output if set, otherwise stdout."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "parse and validate the input but emit nothing. Useful for "
            "checking that a frontend accepts a given safetensors file."
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress progress messages on stderr.",
    )

    # Two-pass parsing: pull out the frontend name first, then add its
    # options and reparse so per-frontend flags appear in --help and bind
    # cleanly when the user runs e.g.
    #   safetensors2verilog ... --frontend threshold_logic --skip-memory
    args, _ = parser.parse_known_args(argv)

    if args.list_frontends:
        for name, desc in registry.names():
            print(f"  {name:<24}  {desc}")
        return 0

    if args.list_frontend_options:
        try:
            cls = registry.get(args.list_frontend_options)
        except KeyError as e:
            parser.error(str(e))
            return 2
        opts = cls.options()
        if not opts:
            print(f"frontend '{cls.name}' has no options.")
            return 0
        print(f"options for frontend '{cls.name}':")
        for o in opts:
            print(f"  --{o.name:<22}  {_format_option_default(o)}")
            if o.help:
                print(f"      {o.help}")
        return 0

    if args.input is None:
        parser.error("input safetensors path is required (or use --list-frontends)")
    if not args.input.exists():
        parser.error(f"file not found: {args.input}")

    try:
        frontend_cls = registry.get(args.frontend)
    except KeyError as e:
        parser.error(str(e))
        return 2

    _add_frontend_options(parser, frontend_cls.options())
    args = parser.parse_args(argv)

    fe_kwargs = {}
    for opt in frontend_cls.options():
        attr = opt.name.replace("-", "_")
        fe_kwargs[attr] = getattr(args, attr)

    frontend = frontend_cls()
    graph = frontend.parse(args.input, top=args.top, **fe_kwargs)

    def _info(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    if args.dry_run:
        _info(
            f"dry-run: {len(graph.gates)} gates, {len(graph.inputs)} inputs, "
            f"{len(graph.outputs)} outputs"
        )
        return 0

    if args.emit_ir == "json":
        text = json.dumps(_graph_to_jsonable(graph), indent=2)
    else:
        text = emit_module(graph)

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        _info(
            f"wrote {args.output} ({len(graph.gates)} gates, "
            f"{len(graph.inputs)} inputs, {len(graph.outputs)} outputs)"
        )

    if args.emit_bram_template is not None:
        addr_bits = args.bram_addr_bits
        if addr_bits is None:
            addr_bits = 16
            try:
                from safetensors import safe_open
                with safe_open(str(args.input), framework="pt") as f:
                    if "manifest.addr_bits" in f.keys():
                        addr_bits = int(f.get_tensor("manifest.addr_bits").item())
            except Exception as e:  # pragma: no cover - non-critical fallback
                _info(
                    f"warning: could not read manifest.addr_bits "
                    f"({type(e).__name__}: {e}); using default 16"
                )
        bram_text = emit_bram_template(
            addr_bits=addr_bits, data_bits=args.bram_data_bits
        )
        args.emit_bram_template.parent.mkdir(parents=True, exist_ok=True)
        args.emit_bram_template.write_text(bram_text, encoding="utf-8")
        _info(
            f"wrote {args.emit_bram_template} (BRAM template, "
            f"{addr_bits}-bit addr, {args.bram_data_bits}-bit data)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
