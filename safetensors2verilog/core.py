"""Frontend interface, IR, and backend dispatch.

The IR is a dataflow graph of `Gate` operations. Each Gate has a `kind`
that selects a backend lowering rule. Built-in kinds include:

  threshold     Σ wᵢ·xᵢ + bias ≥ 0   (1-bit output)
  add, sub, mul multibit signed/unsigned arithmetic
  and, or, xor, not  bitwise logic
  shift_left, shift_right       constant shifts
  concat, slice                 width plumbing
  mux                           N-way multiplexer
  constant                      multibit integer constant
  rom                           parameter ROM with init
  register                      synchronous flip-flop
  relu, clamp                   activation primitives

Frontends emit Gates whose kind is registered with the Verilog backend.
Custom kinds can be added via the @lowering(kind) decorator in
`safetensors2verilog.verilog`.

Signal widths and sign-ness travel as `Signal(name, width, signed)` on
the GateGraph's external ports and as `output_width` / `output_signed`
on each Gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Signal:
    """An external port of the design.

    name:        free-form symbolic name; the backend sanitises for Verilog
    width:       bit width (1 = single bit, >1 = bus). When ``shape`` is set,
                 ``width`` is the per-element bit width and the total port
                 width is ``width * prod(shape)``.
    signed:      if True, declared 'signed' and arithmetic uses $signed
    direction:   "in", "out", or "inout"; "inout" enables tristate emission
    is_parameter: if True, declared as a Verilog parameter rather than wire/reg
                  (used for compile-time constants exposed at the module port)
    parameter_value: integer default for parameter ports

    Multi-dimensional layout:
      shape — optional tuple of dimension extents. Empty tuple (default)
        means a 1-D scalar signal of ``width`` bits. A non-empty shape
        flattens to ``prod(shape)`` elements of ``width`` bits each, packed
        LSB-first along the leading axis (the same layout the matmul,
        embedding, and attention blocks already use for ``x_packed`` /
        ``y_packed`` ports). A 2-D shape ``(seq, hidden)`` represents a
        ``[seq, hidden]`` tensor; element ``[i, j]`` lives at bits
        ``[(i * hidden + j + 1) * width - 1 : (i * hidden + j) * width]``.
        Shape is metadata: the backend emits a flat packed bus regardless,
        but downstream tooling (frontends emitting Conv / Attention /
        windowed access patterns, equivalence harnesses, golden references)
        consult the shape to compute strides and slice bounds.

    Q-format annotation (informational; hardware operates on raw bits):
      q_int_bits, q_frac_bits — fractional fixed-point split. The bit
        pattern represents the value ``raw / 2**q_frac_bits``; total
        ``width = (1 if signed else 0) + q_int_bits + q_frac_bits`` for a
        canonical Q-format signal, but this is not enforced because some
        carriers (matmul accumulators, packed buses) intentionally have
        slack bits or pack heterogeneous elements. When both Q fields are
        zero the signal is interpreted as a plain integer.
      scale — dequantisation multiplier (a float64) recording the mapping
        from the integer bit pattern to the original real-valued tensor.
        Hardware does not consume ``scale``; downstream tooling, golden
        models, and frontend authors use it to track quantisation error.
    """
    name: str
    width: int = 1
    signed: bool = False
    direction: str = "auto"
    is_parameter: bool = False
    parameter_value: int = 0
    q_int_bits: int = 0
    q_frac_bits: int = 0
    scale: float = 1.0
    shape: tuple[int, ...] = ()

    def element_count(self) -> int:
        """Number of elements in the signal (1 for a scalar, prod(shape) otherwise)."""
        if not self.shape:
            return 1
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def total_bits(self) -> int:
        """Total bit width of the signal accounting for shape."""
        return max(1, int(self.width)) * self.element_count()

    def element_bit_range(self, *indices: int) -> tuple[int, int]:
        """Return (high_bit, low_bit) of the element at the given multi-D
        indices. Indices are interpreted as row-major (C order); the
        leading axis varies slowest.

        For a 1-D signal, ``indices`` must be a single integer in
        ``[0, shape[0])``; for a scalar (no shape), no indices.
        """
        if not self.shape:
            if indices:
                raise ValueError(
                    f"signal {self.name!r}: scalar has no axes to index"
                )
            return (self.width - 1, 0)
        if len(indices) != len(self.shape):
            raise ValueError(
                f"signal {self.name!r}: shape={self.shape}, "
                f"got {len(indices)} indices"
            )
        flat = 0
        stride = 1
        for axis_idx in range(len(self.shape) - 1, -1, -1):
            i = int(indices[axis_idx])
            if not 0 <= i < int(self.shape[axis_idx]):
                raise ValueError(
                    f"signal {self.name!r}: index {i} out of range "
                    f"for axis {axis_idx} of length {self.shape[axis_idx]}"
                )
            flat += i * stride
            stride *= int(self.shape[axis_idx])
        lo = flat * self.width
        hi = lo + self.width - 1
        return (hi, lo)


@dataclass
class Gate:
    """A dataflow node.

    name:           unique identifier within the design (also the produced signal)
    kind:           dispatch tag for the backend lowering rule
    inputs:         signal names this node consumes
    attrs:          kind-specific data (weights, slice indices, init, etc.)
    output_width:   bit width of the produced signal (per-element when
                    ``output_shape`` is non-empty)
    output_signed:  if True, the produced signal is two's-complement
    output_shape:   optional tuple of dimension extents matching
                    ``Signal.shape``. Empty (default) means the produced
                    signal is a 1-D scalar of ``output_width`` bits. A
                    non-empty shape declares a packed multidimensional
                    output: total bus width = ``output_width *
                    prod(output_shape)``. Used by Conv2D, batched matmul,
                    attention, and other shape-aware lowerings; ignored by
                    1-D gate kinds.

    For a threshold gate, `attrs` carries:
        weights : list[int], same length as `inputs`
        bias    : int

    Frontends construct Gates directly. The frontend is responsible for
    knowing which kinds the backend (or its own registered lowerings)
    supports.
    """
    name: str
    kind: str = "threshold"
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    output_width: int = 1
    output_signed: bool = False
    output_shape: tuple[int, ...] = ()


@dataclass
class GateGraph:
    """The frontend-produced IR the backend lowers to Verilog.

    inputs:     external input ports (Signal: name + width + sign)
    outputs:    external output ports
    gates:      dataflow nodes, must be topologically sorted
    top:        module name in the generated Verilog
    submodules: additional GateGraphs to emit as separate Verilog modules
                in the same output file. Each submodule's `top` becomes a
                module name that the parent's gates may reference via the
                ``instance`` kind. Used for hierarchical compilation
                (matmul blocks, RMSNorm units, attention heads, etc.) so
                that the parent module stays small while large repeated
                primitives become reusable parameterized modules. Order
                matters only insofar as nested ``instance`` references
                must point at modules that appear earlier in the depth-
                first traversal; the backend emits leaves first.
    """
    inputs: list[Signal]
    outputs: list[Signal]
    gates: list[Gate]
    top: str = "top"
    submodules: list["GateGraph | RawSubmodule"] = field(default_factory=list)


@dataclass
class RawSubmodule:
    """A pre-written Verilog module included verbatim alongside the parent.

    Use this for parameterized templates (matmul block, RMSNorm unit,
    attention head, etc.) whose IR-graph representation would be unwieldy
    or whose synthesis behavior depends on direct vendor pragma placement.
    The backend emits ``text`` verbatim before any module that instantiates
    it via the ``instance`` IR kind.

    top:           module name (must match the ``module`` keyword in ``text``)
    text:          full Verilog source for the module, including ``module``
                   / ``endmodule`` and any necessary `default_nettype
                   directives.
    sidecar_files: filename -> file contents map for any external files the
                   module references (e.g. weight ROMs loaded via
                   ``$readmemh``). Callers writing the emitted Verilog to
                   disk should also write each sidecar file in the same
                   directory; ``collect_sidecar_files(graph)`` walks a
                   GateGraph tree and returns the merged map.
    """
    top: str
    text: str
    sidecar_files: dict[str, str] = field(default_factory=dict)


def collect_sidecar_files(
    graph: "GateGraph",
) -> dict[str, str]:
    """Walk ``graph.submodules`` recursively and merge every ``RawSubmodule``'s
    ``sidecar_files`` into a single dict for the caller to write to disk.

    Duplicate filenames across different RawSubmodules raise ValueError so
    the caller doesn't silently lose ROM contents. Filenames are intended
    to be unique per (module_name, role) combination; see the matmul block
    factory for the canonical naming convention.
    """
    out: dict[str, str] = {}

    def walk(g: "GateGraph") -> None:
        for sub in g.submodules:
            if isinstance(sub, RawSubmodule):
                for fn, contents in sub.sidecar_files.items():
                    if fn in out and out[fn] != contents:
                        raise ValueError(
                            f"sidecar filename collision: '{fn}' "
                            f"(submodule '{sub.top}')"
                        )
                    out[fn] = contents
            else:
                walk(sub)

    walk(graph)
    return out


def rewrite_readmemh_paths(
    verilog_text: str,
    path_map: dict[str, str],
) -> str:
    """Rewrite ``$readmemh("OLD", ...)`` calls in ``verilog_text`` per
    ``path_map`` (``{bare_filename: new_path}``).

    Used by the CLI when ``--sidecar-layout subdirs`` is selected: each
    sidecar hex moves under ``output_dir/<module_name>/<filename>`` and
    the corresponding ``$readmemh`` needs the same prefix for the Verilog
    to still resolve the file at simulation / synthesis time. Filenames
    not in ``path_map`` are left untouched.
    """
    import re as _re

    def repl(match: "_re.Match[str]") -> str:
        old = match.group("path")
        new = path_map.get(old, old)
        return f'$readmemh("{new}"'

    return _re.sub(
        r'\$readmemh\("(?P<path>[^"]+)"',
        repl,
        verilog_text,
    )


def write_sidecar_files(
    graph: "GateGraph",
    output_dir: "Path",
    *,
    layout: str = "subdirs",
    write_manifest: bool = True,
    tarball_path: "Path | None" = None,
) -> dict[str, int]:
    """Materialise every ``RawSubmodule.sidecar_files`` to disk under
    ``output_dir`` with one of three layouts:

      "flat"     all hex files in ``output_dir`` (the legacy behaviour;
                 fine for tens of files, slow for tens of thousands).
      "subdirs"  per-module subdirectory: a sidecar for module M lands in
                 ``output_dir / M / <basename>``. The emitted Verilog uses
                 ``$readmemh("<filename>", ...)`` with the bare filename,
                 and Verilog/Verilator search-path semantics resolve it
                 relative to the simulator's working directory; for synth
                 the user adds ``output_dir / M`` to the include path.
                 This is the default.
      "tarball"  bundle every sidecar into a single tar archive at
                 ``tarball_path`` (required when this layout is chosen).
                 No files are extracted; the user's flow is responsible for
                 unpacking. Returns the per-module count without writing
                 individual files.

    When ``write_manifest`` is True (default) a ``manifest.json`` is also
    written into ``output_dir`` mapping submodule name -> list of
    (filename, byte_size) pairs. The manifest survives across both flat
    and subdirs layouts and gives downstream tooling a structured view of
    what was emitted.

    Returns ``{"files": N, "modules": M, "bytes": B}`` summary stats.
    """
    import json as _json
    import tarfile as _tarfile

    if layout not in ("flat", "subdirs", "tarball"):
        raise ValueError(
            f"layout must be one of flat / subdirs / tarball, got {layout!r}"
        )
    if layout == "tarball" and tarball_path is None:
        raise ValueError(
            "layout='tarball' requires tarball_path to be set"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    per_module: dict[str, list[tuple[str, int]]] = {}
    total_bytes = 0

    def visit(g: "GateGraph") -> None:
        nonlocal total_bytes
        for sub in g.submodules:
            if isinstance(sub, RawSubmodule):
                files = list(sub.sidecar_files.items())
                if not files:
                    continue
                per_module.setdefault(sub.top, [])
                for fn, contents in files:
                    contents_b = contents.encode("utf-8")
                    per_module[sub.top].append((fn, len(contents_b)))
                    total_bytes += len(contents_b)
                    if layout == "flat":
                        (output_dir / fn).write_bytes(contents_b)
                    elif layout == "subdirs":
                        sub_dir = output_dir / sub.top
                        sub_dir.mkdir(parents=True, exist_ok=True)
                        (sub_dir / fn).write_bytes(contents_b)
                    # "tarball" layout writes nothing here; deferred below.
            else:
                visit(sub)

    visit(graph)

    if layout == "tarball":
        # Re-walk to feed every sidecar into the archive in a single pass.
        tarball_path.parent.mkdir(parents=True, exist_ok=True)
        with _tarfile.open(tarball_path, "w") as tar:
            def write_to_tar(g: "GateGraph") -> None:
                for sub in g.submodules:
                    if isinstance(sub, RawSubmodule):
                        for fn, contents in sub.sidecar_files.items():
                            data = contents.encode("utf-8")
                            info = _tarfile.TarInfo(name=f"{sub.top}/{fn}")
                            info.size = len(data)
                            import io as _io
                            tar.addfile(info, _io.BytesIO(data))
                    else:
                        write_to_tar(sub)
            write_to_tar(graph)

    if write_manifest and per_module:
        manifest = {
            "version": 1,
            "layout": layout,
            "modules": {
                module_name: [
                    {"file": fn, "bytes": n} for fn, n in files
                ]
                for module_name, files in sorted(per_module.items())
            },
        }
        if layout == "tarball":
            manifest["tarball"] = str(tarball_path)
        (output_dir / "manifest.json").write_text(
            _json.dumps(manifest, indent=2), encoding="utf-8",
        )

    file_count = sum(len(v) for v in per_module.values())
    return {
        "files": file_count,
        "modules": len(per_module),
        "bytes": total_bytes,
    }


# ---- Frontend abstraction ---------------------------------------------------


@dataclass
class FrontendOption:
    """One per-frontend CLI option, surfaced by the CLI driver.

    name:    flag name (without leading dashes); becomes --<name>
    type:    str / int / float, or `bool` for store_true flags
    default: value when flag is absent
    help:    short --help description
    metavar: optional argparse metavar override
    """
    name: str
    type: type = str
    default: Any = None
    help: str = ""
    metavar: str | None = None


class Frontend:
    """Subclass and implement parse() to add support for a model class.

    Class attributes (set by the @registry.register decorator):
      name                public CLI name
      description         short blurb
      metadata_namespace  reserved metadata key prefix; avoids collisions
                          when multiple frontends share a safetensors file
    """

    name: str = "base"
    description: str = ""
    metadata_namespace: str = ""

    @classmethod
    def options(cls) -> list[FrontendOption]:
        """Per-frontend CLI options. Override to expose flags."""
        return []

    def parse(self, path: Path, top: str = "top", **options) -> GateGraph:
        raise NotImplementedError("subclasses must implement parse()")

    def parse_multi(
        self, path: Path, top: str = "top", **options,
    ) -> list[GateGraph]:
        """Multi-output parse: return one or more independent ``GateGraph``
        objects, each becoming its own top-level Verilog module.

        The default implementation wraps a single ``parse(path)`` call so
        every existing frontend works without modification. Frontends that
        naturally produce multiple top modules (e.g. one per circuit in a
        multi-circuit safetensors, or one per export head in a tied-weight
        model) override this method to return the list directly.

        The CLI's ``--emit-multi DIR`` flag dispatches through
        ``parse_multi`` and writes one ``<top>.v`` file per returned graph
        under DIR. Sidecar files are emitted per the active sidecar layout.
        """
        return [self.parse(path, top=top, **options)]


# ---- Frontend registry ------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self._frontends: dict[str, type[Frontend]] = {}

    def register(
        self,
        name: str,
        description: str = "",
        metadata_namespace: str = "",
    ) -> Callable[[type[Frontend]], type[Frontend]]:
        def deco(cls: type[Frontend]) -> type[Frontend]:
            cls.name = name
            cls.description = description
            cls.metadata_namespace = metadata_namespace or name
            self._frontends[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[Frontend]:
        if name not in self._frontends:
            raise KeyError(
                f"unknown frontend '{name}'. Registered: {sorted(self._frontends)}"
            )
        return self._frontends[name]

    def names(self) -> list[tuple[str, str]]:
        return [
            (name, cls.description)
            for name, cls in sorted(self._frontends.items())
        ]


registry = _Registry()


# ---- Metadata-namespace enforcement ---------------------------------------


# Reserved metadata keys that the safetensors2verilog tool itself uses,
# regardless of frontend. Frontends must not write to these.
_RESERVED_METADATA_KEYS = frozenset({
    "format", "encoding", "version", "schema_version",
})

# Schema version for the threshold-logic / signal_registry metadata
# convention. Frontends that read signal_registry consult this so future
# format changes can be detected and rejected with a clear message rather
# than silently misinterpreted.
#
#   1   - original layout (8bit-threshold-computer through May 2026)
#         signal_registry is a JSON object mapping str(int) -> name.
#         External inputs prefixed "$"; constants "#0" / "#1".
#         Gate tensors: <gate>.weight (1-D or [N,K] packed),
#                       <gate>.bias (length-1 or length-N packed),
#                       <gate>.inputs (1-D int signal IDs).
SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED: tuple[int, ...] = (1,)
SIGNAL_REGISTRY_SCHEMA_VERSION_LATEST: int = 1


def check_schema_version(metadata: dict[str, Any], frontend_name: str) -> int:
    """Validate ``schema_version`` in safetensors metadata.

    Returns the parsed integer schema version. Files without the key are
    treated as version 1 for backward compatibility (the value the
    8bit-threshold-computer family uses today). Files whose declared
    version is not in :data:`SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED`
    raise :class:`ValueError` so the frontend never silently misreads a
    future format.
    """
    raw = metadata.get("schema_version")
    if raw is None:
        return 1
    try:
        version = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"frontend '{frontend_name}': metadata schema_version "
            f"{raw!r} is not an integer"
        ) from e
    if version not in SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED:
        raise ValueError(
            f"frontend '{frontend_name}': unsupported schema_version "
            f"{version}; supported versions: "
            f"{list(SIGNAL_REGISTRY_SCHEMA_VERSIONS_SUPPORTED)}"
        )
    return version


def validate_metadata_namespace(
    frontend: type[Frontend], metadata: dict[str, Any]
) -> None:
    """Check that every metadata key either lives under the frontend's
    namespace, is a reserved global key, or is a special shared key
    (currently: ``signal_registry``, used by both threshold_logic and
    any future frontend that wants symbolic-name lookup).

    Frontends are encouraged to use only keys with the prefix
    ``"<metadata_namespace>."``; everything else triggers a warning.
    Set ``metadata_namespace`` via the @registry.register decorator.
    """
    import warnings

    ns = frontend.metadata_namespace
    shared = {"signal_registry"}
    for key in metadata:
        if key in _RESERVED_METADATA_KEYS or key in shared:
            continue
        if not ns:
            warnings.warn(
                f"frontend '{frontend.name}' has no metadata_namespace and "
                f"is reading metadata key {key!r}; collisions with other "
                f"frontends are possible.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not (key == ns or key.startswith(ns + ".")):
            warnings.warn(
                f"frontend '{frontend.name}' (namespace '{ns}') is reading "
                f"metadata key {key!r} that lives outside its namespace; "
                f"prefer keys named '{ns}.<sub>'.",
                UserWarning,
                stacklevel=2,
            )
