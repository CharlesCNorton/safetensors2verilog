"""Frontends register themselves with the core registry on import.

To add a new frontend:
  1. Create safetensors2verilog/frontends/my_frontend.py
  2. Subclass `Frontend` and decorate with `@registry.register(...)`
  3. Add `from . import my_frontend` to this file.
"""

from . import threshold_logic  # noqa: F401
from . import bitnet_linear  # noqa: F401
