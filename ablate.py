"""
ablate.py — grid search over model and data configs.

Usage:
    python ablate.py            # runs the full ablation grid
    python ablate.py --dry_run  # prints configs without training

Results are written to checkpoints/<run_name>/best.pt and summarised at the end.

ABLATION GRID (edit as needed):

  Model ablations (Member B's track):
    - head_depth:  [1, 2, 3]
    - lora_rank:   [0, 4, 8, 16]   (0 = linear head, no LoRA)

  Data ablations (Member A's track):
    - neg_strategy: ["inbatch", "clip_mined", "all"]
    - data fraction: [0.2, 0.5, 1.0]
"""

import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from config import DEFAULT_CONFIG, CKPT_DIR


ABLATION_GRID = {
    "model.head_depth":      [1, 2, 3],
    "model.lora_rank":       [0, 8, 16],
    "data.neg_strategy":     ["inbatch", "clip_mined"],
}

# Subset grid for a quick overnight run
QUICK_GRID = {
    "model.head_depth":      [1, 2],
    "model.lora_rank":       [8],
    "data.neg_strategy":     ["inbatch", "clip_mined"],
}


def run_config(run_name: str, overrides: dict, dry_run: bool = False):
    """Launch train.py as a subprocess with the given overrides."""
    cmd = [sys.executable, "train.py", "--run_name", run_name]
    for key, val in overrides.items():
        # Map dotted keys to CLI args: model.lora_rank → --lora_rank
        short_key = key.split(".")[-1]
        cmd += [f"--{short_key}", str(val)]

    print(f"\n>>> {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def collect_results() -> list:
    """Read best.pt checkpoints and extract metrics for comparison."""
    results = []
    for ckpt_path in sorted(CKPT_DIR.glob("*/best.pt")):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        results.append({
            "run":     ckpt_path.parent.name,
            "epoch":   ckpt.get("epoch"),
            "metrics": ckpt.get("metrics", {}),
        })
    results.sort(key=lambda r: r["metrics"].get("acc@0.5", 0), reverse=True)
    return results


def main(quick: bool = False, dry_run: bool = False):
    grid = QUICK_GRID if quick else ABLATION_GRID

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"Total configurations: {len(combos)}")

    for combo in combos:
        overrides = dict(zip(keys, combo))
        run_name  = "abl__" + "__".join(f"{k.split('.')[-1]}={v}"
                                        for k, v in overrides.items())
        run_config(run_name, overrides, dry_run=dry_run)

    if not dry_run:
        print("\n\n=== Ablation results (sorted by Acc@0.5) ===")
        for r in collect_results():
            m = r["metrics"]
            print(f"  {r['run']:60s}  "
                  f"acc@0.5={m.get('acc@0.5','?')}  "
                  f"miou={m.get('mean_iou','?')}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick",   action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    main(quick=args.quick, dry_run=args.dry_run)