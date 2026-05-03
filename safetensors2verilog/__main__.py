"""Allow `python -m safetensors2verilog ...`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
