"""Source-level chunking of oversized weight ROMs into BRAM-friendly banks.

The Xilinx 7-series and Versal AI Edge primitive BRAM is 36 Kbit; the
Lattice ECP5 EBR is 18 Kbit; the Intel Stratix 10 M20K is 20 Kbit. None
of these will hold the SmolLM2 49152 x 576 x 8-bit embedding (226 Mbits)
in one mapping unit; ``synth_xilinx`` rejects an oversized ``reg`` array
with "no valid mapping found" because the array would map to thousands of
non-trivially-cascaded BRAMs.

This module rewrites a logically large ROM into N parallel BRAM-sized
banks. Two axes of chunking compose:

  Row chunking:    split a depth-D ROM into ``ceil(D / chunk_depth)``
                   banks; each bank holds ``chunk_depth`` consecutive
                   rows. The read addr's high bits select which bank to
                   read from; a small mux at the output picks the right
                   bank's data.

  Column chunking: split a width-W entry into ``ceil(W / chunk_bits)``
                   parallel column banks; each bank holds the full depth
                   but only ``chunk_bits`` of each row. Banks are read in
                   parallel and their outputs concatenated to reconstruct
                   the full W-bit entry.

The two strategies compose: first column-chunk to get each bank under
``chunk_bits``, then row-chunk so each bank is under ``chunk_depth``
entries. The product gives a layout where every individual storage unit
is at most ``chunk_depth * chunk_bits`` bits, matching the target BRAM
primitive.

Public API:

  pick_chunking(depth, width, *, max_bram_bits=36864, max_bram_depth=2048)
      Returns ``(chunk_depth, chunk_bits, n_row_banks, n_col_banks)``
      computed from the BRAM constraints.

  emit_chunked_rom(name, init, *, depth, width, chunk_depth, chunk_bits)
      Returns Verilog text for an asynchronous-read ROM with the chunked
      layout: row banks selected via a high-address mux, column banks
      concatenated. Vendor synth infers each individual ROM as one BRAM.
"""
from __future__ import annotations

import math
from textwrap import dedent


def pick_chunking(
    depth: int,
    width: int,
    *,
    max_bram_bits: int = 36864,
    max_bram_depth: int = 2048,
) -> tuple[int, int, int, int]:
    """Compute (chunk_depth, chunk_bits, n_row_banks, n_col_banks) so that
    every individual bank fits within ``max_bram_bits`` total and
    ``max_bram_depth`` entries.

    Defaults match the Xilinx 7-series 36 Kbit BRAM with a 2048-deep
    16-bit-wide configuration (the most flexible address shape).
    """
    if depth <= 0 or width <= 0:
        raise ValueError(
            f"depth and width must be positive, got {depth}, {width}"
        )
    if max_bram_bits <= 0 or max_bram_depth <= 0:
        raise ValueError(
            f"max_bram_bits and max_bram_depth must be positive"
        )
    # Column chunk: largest chunk_bits such that
    # chunk_bits * min(depth, max_bram_depth) <= max_bram_bits.
    bram_depth = min(depth, max_bram_depth)
    chunk_bits = max(1, max_bram_bits // bram_depth)
    chunk_bits = min(chunk_bits, width)
    n_col_banks = math.ceil(width / chunk_bits)
    # Row chunk: chunk_depth = max_bram_depth (or depth if smaller).
    chunk_depth = min(depth, max_bram_depth)
    n_row_banks = math.ceil(depth / chunk_depth)
    return chunk_depth, chunk_bits, n_row_banks, n_col_banks


def emit_chunked_rom(
    name: str,
    init: list[int],
    *,
    depth: int,
    width: int,
    chunk_depth: int | None = None,
    chunk_bits: int | None = None,
    addr_signal: str = "addr",
    use_block_ram: bool = True,
) -> str:
    """Emit a chunked ROM as a series of small ROMs plus address mux +
    column concat.

    The generated Verilog is a snippet (not a complete module): the user
    paste it into a parent module, providing the ``addr_signal`` and
    consuming the wire ``<name>``.

    init:    list of ``depth`` integers; entry i is the row at address i.
             Each entry is masked to ``width`` bits.
    """
    if chunk_depth is None or chunk_bits is None:
        cd, cb, _, _ = pick_chunking(depth, width)
        chunk_depth = chunk_depth or cd
        chunk_bits = chunk_bits or cb
    if chunk_depth > depth:
        chunk_depth = depth
    if chunk_bits > width:
        chunk_bits = width

    n_row = math.ceil(depth / chunk_depth)
    n_col = math.ceil(width / chunk_bits)
    addr_w = max(1, (depth - 1).bit_length())
    bank_addr_w = max(1, (chunk_depth - 1).bit_length())
    bank_select_w = max(1, (n_row - 1).bit_length()) if n_row > 1 else 0

    ram_attr = '(* ram_style = "block" *) ' if use_block_ram else ""

    # Per-(row_bank, col_bank) ROM declarations.
    rom_decls: list[str] = []
    for ri in range(n_row):
        for ci in range(n_col):
            col_lo = ci * chunk_bits
            col_hi = min(col_lo + chunk_bits, width)
            chunk_w = col_hi - col_lo
            row_lo = ri * chunk_depth
            row_hi = min(row_lo + chunk_depth, depth)
            chunk_d = row_hi - row_lo
            mem = f"{name}_r{ri}_c{ci}"
            rom_decls.append(
                f"{ram_attr}reg [{chunk_w-1}:0] {mem} [0:{chunk_d-1}];"
            )
            init_lines = []
            for k in range(chunk_d):
                row_idx = row_lo + k
                row_val = init[row_idx] if row_idx < len(init) else 0
                slice_val = (row_val >> col_lo) & ((1 << chunk_w) - 1)
                init_lines.append(
                    f"  {mem}[{k}] = {chunk_w}'h{slice_val:x};"
                )
            rom_decls.append("initial begin")
            rom_decls.extend(init_lines)
            rom_decls.append("end")

    # Per-row-bank read; per-column-bank concat; final mux on bank select.
    body: list[str] = []
    if n_row == 1:
        # Single row bank: just concat the column reads.
        col_reads = []
        for ci in range(n_col):
            col_lo = ci * chunk_bits
            col_hi = min(col_lo + chunk_bits, width)
            chunk_w = col_hi - col_lo
            col_reads.append(f"{name}_r0_c{ci}[{addr_signal}]")
        if n_col == 1:
            body.append(
                f"wire [{width-1}:0] {name} = {col_reads[0]};"
            )
        else:
            # MSB-first concat.
            body.append(
                f"wire [{width-1}:0] {name} = {{"
                + ", ".join(reversed(col_reads)) + "};"
            )
    else:
        # Row-banked: split addr into (bank_select, bank_addr).
        body.append(
            f"wire [{bank_select_w-1}:0] {name}_bank_sel = "
            f"{addr_signal}[{addr_w-1}:{addr_w - bank_select_w}];"
        )
        body.append(
            f"wire [{bank_addr_w-1}:0] {name}_bank_addr = "
            f"{addr_signal}[{bank_addr_w-1}:0];"
        )
        # Per-row-bank concat.
        for ri in range(n_row):
            col_reads = [
                f"{name}_r{ri}_c{ci}[{name}_bank_addr]"
                for ci in range(n_col)
            ]
            if n_col == 1:
                body.append(
                    f"wire [{width-1}:0] {name}_bank{ri} = {col_reads[0]};"
                )
            else:
                body.append(
                    f"wire [{width-1}:0] {name}_bank{ri} = {{"
                    + ", ".join(reversed(col_reads)) + "};"
                )
        # Mux the banks by bank_sel.
        body.append(f"reg [{width-1}:0] {name};")
        body.append("always @(*) begin")
        body.append(f"  case ({name}_bank_sel)")
        for ri in range(n_row):
            body.append(f"    {bank_select_w}'d{ri}: {name} = {name}_bank{ri};")
        body.append(f"    default: {name} = {width}'h0;")
        body.append("  endcase")
        body.append("end")

    return "\n".join(rom_decls + [""] + body) + "\n"
