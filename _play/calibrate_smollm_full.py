"""Items 15 + 16 of TODO.md.

Item 15: Calibrate against real text on full 30-layer SmolLM2 and emit
the per-layer JSON.

Item 16: Run calibrate_iteratively_damped on full SmolLM2 with damping
in {0.3, 0.5, 0.7} and capture the per-site convergence trajectory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from safetensors2verilog.calibration import (
    REQUANTIZE_SITES,
    calibrate_iteratively_damped,
    collect_activation_stats,
    derive_requantize_params,
    saturation_summary,
)

SMOLLM_DIR = Path(
    r"D:\huggingface\hub\models--HuggingFaceTB--SmolLM2-135M-Instruct"
    r"\snapshots\12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
OUT = Path(__file__).resolve().parent / "smollm_calib_full"
OUT.mkdir(exist_ok=True)


def load_full(n_layers: int = 30) -> tuple[dict, dict[str, torch.Tensor]]:
    cfg = json.loads((SMOLLM_DIR / "config.json").read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = n_layers
    sd: dict[str, torch.Tensor] = {}
    keep_layer_prefixes = tuple(
        f"model.layers.{i}." for i in range(n_layers)
    )
    with safe_open(str(SMOLLM_DIR / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.layers.") and not k.startswith(keep_layer_prefixes):
                continue
            sd[k] = f.get_tensor(k).clone()
    return cfg, sd


def get_text_token_sequences(
    cfg: dict, n_seqs: int = 8, tokens_per_seq: int = 32,
) -> list[list[int]]:
    """Use the HF tokenizer + a small built-in text corpus to produce
    realistic token sequences. Falls back to deterministic random ids if
    transformers isn't available.
    """
    text_seeds = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a land far away there lived a king.",
        "Machine learning is the study of statistical patterns.",
        "She sells seashells by the seashore on a Sunday afternoon.",
        "The capital of France is Paris and it sits on the Seine.",
        "Photosynthesis converts carbon dioxide and water into sugar.",
        "Bach composed the Brandenburg concertos for a German prince.",
        "The Pacific Ocean is the largest body of water on Earth.",
    ]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(SMOLLM_DIR))
        seqs: list[list[int]] = []
        for txt in text_seeds[:n_seqs]:
            ids = tok.encode(txt, add_special_tokens=False)[:tokens_per_seq]
            if len(ids) < tokens_per_seq:
                # Pad with the same ids cycled.
                ids = (ids * ((tokens_per_seq + len(ids) - 1) // len(ids)))[:tokens_per_seq]
            seqs.append(ids)
        return seqs
    except Exception as e:
        print(f"transformers unavailable ({e}); falling back to random ids")
        torch.manual_seed(0)
        return [
            torch.randint(0, cfg["vocab_size"], (tokens_per_seq,)).tolist()
            for _ in range(n_seqs)
        ]


def main(n_layers: int = 30) -> int:
    print(f"=== item 15: calibration on real text, {n_layers} layers ===")
    cfg, sd = load_full(n_layers)
    print(f"  loaded {len(sd)} tensors; "
          f"hidden={cfg['hidden_size']}, INTER={cfg['intermediate_size']}, "
          f"layers={n_layers}")
    sequences = get_text_token_sequences(cfg, n_seqs=4, tokens_per_seq=16)
    n_tokens = sum(len(s) for s in sequences)
    print(f"  {len(sequences)} sequences x {len(sequences[0])} tokens = "
          f"{n_tokens} tokens")

    # Single-pass calibration first.
    t0 = time.time()
    stats = collect_activation_stats(
        config=cfg, state_dict=sd,
        token_sequences=sequences, abits=8, weight_bits=8,
    )
    elapsed = time.time() - t0
    print(f"  stats collected in {elapsed:.1f}s "
          f"({n_tokens * n_layers / elapsed:.0f} tok-layer/s)")
    params = derive_requantize_params(stats, target_max=80, mul_bits=8,
                                       use_p995=False)
    out_path = OUT / f"smollm_l{n_layers}_calibration_realtext.json"
    out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"  wrote {out_path} "
          f"({out_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Per-site abs_max summary across layers.
    print(f"\n  per-site abs_max max across layers:")
    print(f"  {'site':<6} " + " ".join(f"L{i:02d}" for i in range(min(5, n_layers))) + " ... " + " ".join(f"L{i:02d}" for i in range(max(0, n_layers-5), n_layers)))
    for site in REQUANTIZE_SITES:
        per_layer = []
        for layer_calib in stats.layers:
            ss = layer_calib.sites.get(site)
            per_layer.append(max(ss.abs_max) if ss else 0)
        head = " ".join(f"{v:>4d}" for v in per_layer[:5])
        tail = " ".join(f"{v:>4d}" for v in per_layer[-5:])
        print(f"  {site:<6} {head} ... {tail}")

    # Item 16: damped iterative calibration with three damping values.
    print(f"\n=== item 16: damped iteration, "
          f"damping in {{0.3, 0.5, 0.7}} ===")
    trajectory: dict[float, dict] = {}
    for damping in (0.3, 0.5, 0.7):
        print(f"\n  damping = {damping}")
        # Run 3 rounds and capture per-site abs_max each round to show
        # the convergence trajectory. We piggyback on the existing damped
        # function by manually iterating to capture intermediate states.
        from safetensors2verilog.calibration import (
            collect_activation_stats as _collect,
            derive_requantize_params as _derive,
        )
        params_iter = None
        rounds = []
        for it in range(3):
            stats_it = _collect(
                config=cfg, state_dict=sd,
                token_sequences=sequences, abits=8, weight_bits=8,
                prev_params=params_iter,
            )
            row = {}
            for site in REQUANTIZE_SITES:
                ss = stats_it.layers[0].sites[site]
                row[site] = max(ss.abs_max)
            rounds.append(row)
            raw_params = _derive(
                stats_it, target_max=80, mul_bits=8, use_p995=False,
            )
            if params_iter is None:
                params_iter = raw_params
            else:
                blended = []
                for layer_old, layer_raw in zip(params_iter, raw_params):
                    layer_b = {}
                    for site in REQUANTIZE_SITES:
                        old = layer_old[site]; raw = layer_raw[site]
                        layer_b[site] = {
                            "muls": [
                                round(damping * rm + (1 - damping) * om)
                                for rm, om in zip(raw["muls"], old["muls"])
                            ],
                            "shifts": [
                                round(damping * rs + (1 - damping) * os_)
                                for rs, os_ in zip(raw["shifts"], old["shifts"])
                            ],
                        }
                    blended.append(layer_b)
                params_iter = blended
            print(f"    iter {it}: " + "  ".join(
                f"{site}={row[site]:>5d}" for site in REQUANTIZE_SITES
            ))
        trajectory[damping] = {"rounds": rounds}
        # Write the final calibrated JSON for this damping value.
        out_d = OUT / f"smollm_l{n_layers}_calibration_damped_{damping}.json"
        out_d.write_text(json.dumps(params_iter, indent=2), encoding="utf-8")
        print(f"    wrote {out_d}")

    # Save the trajectory.
    traj_path = OUT / f"smollm_l{n_layers}_damped_trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"\n  wrote trajectory to {traj_path}")
    return 0


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    raise SystemExit(main(n))
