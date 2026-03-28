"""
evaluate.py — run evaluation on a saved checkpoint.

Usage:
    python evaluate.py --ckpt checkpoints/baseline/best.pt --split test
    python evaluate.py --ckpt checkpoints/baseline/best.pt --split val --visualize 20
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from config import DEFAULT_CONFIG
from data import Flickr30kGroundingDataset, collate_fn
from eval import GroundingEvaluator
from eval.metrics import iou as compute_iou
from eval.visualize import draw_grounding_result
from models import GroundingModel


@torch.no_grad()
def run_eval(cfg, ckpt_path: Path, split: str, n_visualize: int = 0):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.clip_model)

    ds     = Flickr30kGroundingDataset(cfg, split=split, tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    model = GroundingModel(cfg).to(device)
    ckpt  = model.load(ckpt_path)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    if ckpt.get("metrics"):
        print(f"Saved val metrics : {ckpt['metrics']}")

    evaluator = GroundingEvaluator(cfg)
    viz_count = 0
    viz_dir   = ckpt_path.parent / "visualizations"
    if n_visualize > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for batch in loader:
        out   = model(batch, neg_mining=None)
        preds = out["preds"].cpu()

        evaluator.update_from_indices(
            pred_idx=preds,
            proposals=batch["proposals"],
            gt_boxes=batch["gt_box"],
            entity_types=batch["entity_type"],
        )

        # Save visualizations for the first n_visualize samples
        if viz_count < n_visualize:
            B = preds.size(0)
            for i in range(min(B, n_visualize - viz_count)):
                pred_box = batch["proposals"][i, preds[i]].cpu()
                gt_box   = batch["gt_box"][i].cpu()
                score    = compute_iou(pred_box, gt_box)

                # Use the PIL image stored in the dataset directly
                img_id  = batch["image_id"][i]
                hf_row  = ds._hf_rows.get(img_id)
                pil_img = hf_row["image"] if hf_row else None

                draw_grounding_result(
                    phrase=batch["phrase"][i],
                    pred_box=pred_box,
                    image=pil_img,          # PIL image from HF dataset
                    gt_box=gt_box,
                    iou_score=score,
                    save_path=viz_dir / f"{viz_count:04d}_{img_id}.png",
                )
                viz_count += 1

    metrics = evaluator.compute()

    print(f"\n=== Results on {split} split ===")
    print(f"  Acc@0.5        : {metrics['acc@0.5']:.4f}")
    print(f"  Acc@0.25       : {metrics['acc@0.25']:.4f}")
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"  N samples      : {metrics['n_samples']}")
    if metrics.get("baseline_delta") is not None:
        print(f"  Delta vs CLIP  : {metrics['baseline_delta']:+.4f}")
    print(f"  By entity type :")
    for t, acc in metrics.get("acc_by_type", {}).items():
        print(f"    {t:20s}: {acc:.4f}")

    if n_visualize > 0:
        print(f"\n  Saved {viz_count} visualizations to {viz_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      required=True, type=Path)
    parser.add_argument("--split",     default="val")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of examples to save as annotated images")
    args = parser.parse_args()

    cfg            = DEFAULT_CONFIG
    cfg.eval.split = args.split
    run_eval(cfg, args.ckpt, args.split, args.visualize)