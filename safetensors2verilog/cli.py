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
    parser.add_argument(
        "--skip-memory", "--bram", dest="skip_memory", action="store_true",
        help="drop memory.* gates from the network so they can be served by "
             "a vendor BRAM block. Read-side signals are promoted to external "
             "inputs; address/data/write-enable signals appear as external "
             "outputs on the resulting CPU core.",
    )
    parser.add_argument(
        "--emit-bram-template", type=Path, default=None, metavar="PATH",
        help="alongside the main Verilog output, write a synchronous BRAM "
             "template module (single-port read/write, 8-bit data, "
             "configurable address width). Implies --skip-memory.",
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

    skip_memory = args.skip_memory or args.emit_bram_template is not None
    frontend_cls = registry.get(args.frontend)
    frontend = frontend_cls()
    graph = frontend.parse(args.input, top=args.top, skip_memory=skip_memory)
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

    if args.emit_bram_template is not None:
        from .verilog import emit_bram_template
        # Derive address width from the variant's manifest (memory_bytes)
        # if available; fall back to 16 (64 KB) when missing.
        addr_bits = 16
        try:
            from safetensors import safe_open
            with safe_open(str(args.input), framework="pt") as f:
                if "manifest.addr_bits" in f.keys():
                    addr_bits = int(f.get_tensor("manifest.addr_bits").item())
        except Exception:
            pass
        bram_text = emit_bram_template(addr_bits=addr_bits)
        args.emit_bram_template.parent.mkdir(parents=True, exist_ok=True)
        args.emit_bram_template.write_text(bram_text, encoding="utf-8")
        print(
            f"wrote {args.emit_bram_template} (BRAM template, "
            f"{addr_bits}-bit addr, 8-bit data)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
