"""Allow `python -m safetensors2verilog ...`.

Importing the package triggers `__init__.py` which registers every
built-in frontend. We also import `frontends` explicitly here so the
registry is populated even on Python paths that turn the package into
a namespace package (e.g. when a sibling directory of the same name
shadows the installed `__init__.py`).
"""

from . import frontends  # noqa: F401  (frontend registration)
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
