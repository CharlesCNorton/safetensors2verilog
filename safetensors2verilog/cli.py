"""Command-line entry point.

    python -m safetensors2verilog input.safetensors --frontend threshold_logic -o output.v
    python -m safetensors2verilog --list-frontends
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import registry
from .verilog import emit_module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="safetensors2verilog",
        description="Compile safetensors-stored networks to synthesis-ready Verilog.",
    )
    parser.add_argument("input", nargs="?", type=Path, help="path to .safetensors file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output Verilog file (default: stdout)",
    )
    parser.add_argument(
        "--frontend", type=str, default="threshold_logic",
        help="frontend module class (default: threshold_logic)",
    )
    parser.add_argument(
        "--top", type=str, default="top",
        help="top-level Verilog module name (default: top)",
    )
    parser.add_argument(
        "--list-frontends", action="store_true",
        help="print registered frontends and exit",
    )
    args = parser.parse_args(argv)

    if args.list_frontends:
        for name, desc in registry.names():
            print(f"  {name:<24}  {desc}")
        return 0

    if args.input is None:
        parser.error("input safetensors path is required (or use --list-frontends)")
    if not args.input.exists():
        parser.error(f"file not found: {args.input}")

    frontend_cls = registry.get(args.frontend)
    frontend = frontend_cls()
    graph = frontend.parse(args.input, top=args.top)
    text = emit_module(graph)

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(
            f"wrote {args.output} ({len(graph.gates)} gates, "
            f"{len(graph.inputs)} inputs, {len(graph.outputs)} outputs)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
