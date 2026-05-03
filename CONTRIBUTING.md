# Contributing

## Development setup

```bash
git clone https://github.com/CharlesCNorton/safetensors2verilog.git
cd safetensors2verilog
pip install -e .[dev]
```

The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, `mypy`, `numpy`, and `onnx`. `iverilog` is a system dependency for the simulation tests; on Debian/Ubuntu: `sudo apt install iverilog`. Yosys (`apt install yosys`) is optional but useful if you want to verify synthesis locally.

## Running checks

```bash
pytest -v                               # 79 tests on a clean install
pytest --cov=safetensors2verilog        # with coverage
ruff check safetensors2verilog tests    # lint
mypy safetensors2verilog                # type check
python examples/threshold_alu/run.py    # iverilog round-trip on the half-adder
python examples/bitnet_linear/run.py    # iverilog round-trip on a tiny BitNet
```

CI runs the same commands across Python 3.10 / 3.11 / 3.12 plus a Yosys synthesis check on the half-adder output. See `.github/workflows/ci.yml`.

## Adding a new frontend

1. Create `safetensors2verilog/frontends/<name>.py`. Subclass `Frontend`, decorate the class with `@registry.register("<name>", description=..., metadata_namespace="<name>")`, and implement `parse(self, path, top, **options) -> GateGraph`.
2. Surface CLI flags via `Frontend.options()` returning a list of `FrontendOption`. The CLI two-pass parser will plumb them automatically.
3. Add `from . import <name>` to `safetensors2verilog/frontends/__init__.py`.
4. Drop a test file under `tests/test_<name>.py`. Use `safetensors.torch.save_file` to construct fixture inputs in temporary directories. Skip iverilog-dependent tests with `pytest.mark.skipif(not _have_iverilog())`.
5. Optionally add an `examples/<name>/run.py` that builds a fixture, converts it, simulates with iverilog, and cross-checks against a Python reference. CI runs every script under `examples/*/run.py` end-to-end.

The shared backend handles signal sanitization, port emission, topological-order validation, kind dispatch, and per-kind lowerings. You should not need to touch it unless you're adding a new operation primitive.

## Adding a new gate kind

If your frontend needs an operation the built-in lowerings don't cover, register one with `@lowering(kind)`:

```python
from safetensors2verilog.verilog import lowering


@lowering("my_op")
def lower_my_op(ctx, gate):
    """One per-kind contract: read gate.attrs and gate.inputs, return list[str] of Verilog lines."""
    return [f"  assign {ctx.name(gate.name)} = ...;"]
```

`ctx.name(sig)` resolves a signal name to its sanitized identifier or to a Verilog literal for `#0`/`#1`. `ctx.width(sig)` and `ctx.is_signed(sig)` look up the operand's width and sign.

For multi-bit kinds, validate that `gate.output_width` and `gate.output_signed` are sane for your operation, and reject impossible combinations (`slice` rejects `output_width != hi - lo + 1`, `concat` rejects width mismatch, etc.). Hard errors here catch frontend bugs early.

## Code style

`ruff` is the source of truth. The repo's lint config is in `pyproject.toml` under `[tool.ruff]`. Run `ruff check --fix` before submitting.

`mypy` is run with `ignore_missing_imports = true` and `check_untyped_defs = true`. Type annotations are encouraged but not required on internal helpers.

## Pull requests

Small focused PRs get reviewed faster. If you're adding a new frontend, the natural shape is one PR that ships the frontend file, the test file, an example script, and a README mention.
