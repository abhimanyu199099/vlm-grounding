"""
train.py — main training loop.

All three members will read this file but no one needs to heavily modify it.
It wires together data/, models/, and eval/ using config.py.

Run:
    python train.py
    python train.py --run_name lora_r16 --model.lora_rank 16
    python train.py --debug   # fast smoke-test with 200 samples
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPTokenizer

from config import Config, DEFAULT_CONFIG, CKPT_DIR, RUNS_DIR
from data import Flickr30kGroundingDataset, collate_fn
from data.negatives import NegativeMiner
from models import GroundingModel
from eval import GroundingEvaluator


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, miner, evaluator, cfg, epoch):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        with autocast(enabled=cfg.train.mixed_precision):
            # First forward pass without hard negatives to get embeddings
            out = model(batch, neg_mining=None)

            # Mine hard negatives using the embeddings from this forward pass
            neg_mining = None
            if miner is not None:
                neg_indices, cross_image = miner.mine(
                    batch,
                    phrase_embeds=out["phrase_embeds"].detach(),
                    region_embeds=out["region_embeds"].detach(),
                )
                neg_mining = (neg_indices, cross_image)
                # Re-score with hard negatives informing the loss
                out = model(batch, neg_mining=neg_mining)

            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.trainable_parameters, cfg.train.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % cfg.train.log_every == 0:
            avg = total_loss / (step + 1)
            print(f"  Epoch {epoch} | step {step}/{len(loader)} | loss {avg:.4f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, evaluator, cfg):
    model.eval()
    evaluator.reset()

    for batch in loader:
        out      = model(batch, neg_mining=None)
        preds    = out["preds"]                    # (B,)
        proposals = batch["proposals"]             # (B, N, 4)
        gt_boxes  = batch["gt_box"]               # (B, 4)
        entity_types = batch["entity_type"]       # list[str]

        evaluator.update_from_indices(
            pred_idx=preds,
            proposals=proposals,
            gt_boxes=gt_boxes,
            entity_types=entity_types,
        )

    return evaluator.compute()


# ---------------------------------------------------------------------------
# CLIP baseline (run once before training to set the floor)
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_baseline(model, loader, evaluator, cfg):
    """
    Score proposals using frozen CLIP similarity only (no grounding head).
    Establishes the floor that the trained model must beat.
    """
    model.eval()
    evaluator.reset()

    for batch in loader:
        device = next(model.parameters()).device
        phrase_tokens  = batch["phrase_tokens"].to(device)
        proposal_crops = batch["proposal_crops"].to(device)
        attn_mask      = (phrase_tokens != 0).to(device)

        phrase_embeds = model.encoder.encode_phrase(phrase_tokens, attn_mask)  # (B, D)
        region_embeds = model.encoder.encode_region(proposal_crops)            # (B, N, D)

        # Cosine similarity: (B, N)
        scores = torch.einsum("bd,bnd->bn", phrase_embeds, region_embeds)

        if "proposal_mask" in batch:
            mask = batch["proposal_mask"].to(device)
            scores = scores.masked_fill(~mask, float("-inf"))

        preds = scores.argmax(dim=1)

        evaluator.update_from_indices(
            pred_idx=preds,
            proposals=batch["proposals"],
            gt_boxes=batch["gt_box"],
            entity_types=batch["entity_type"],
        )

    return evaluator.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.clip_model)

    # Datasets
    train_ds = Flickr30kGroundingDataset(cfg, split="train",
                                         tokenizer=tokenizer, debug=cfg.debug)
    val_ds   = Flickr30kGroundingDataset(cfg, split="val",
                                         tokenizer=tokenizer, debug=cfg.debug)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers,
                              pin_memory=cfg.data.pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers)

    # Model
    model = GroundingModel(cfg).to(device)
    print(f"Trainable params : {model.trainable_param_count():,}")
    print(f"Total params     : {model.total_param_count():,}")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )
    scaler    = GradScaler(enabled=cfg.train.mixed_precision)
    miner     = NegativeMiner(cfg, clip_model=model.encoder.clip)
    evaluator = GroundingEvaluator(cfg)

    # CLIP baseline (run once)
    print("\n--- CLIP baseline (no training) ---")
    baseline = clip_baseline(model, val_loader, evaluator, cfg)
    print(f"  Acc@0.5: {baseline['acc@0.5']:.4f}  |  mean IoU: {baseline['mean_iou']:.4f}")

    # Training loop
    best_acc = 0.0
    run_dir  = CKPT_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.train.epochs} ===")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, miner, evaluator, cfg, epoch
        )
        scheduler.step()

        if epoch % cfg.train.eval_every == 0:
            metrics = evaluate(model, val_loader, evaluator, cfg)
            acc = metrics["acc@0.5"]
            print(f"  Val Acc@0.5: {acc:.4f}  |  mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Breakdown: {metrics['acc_by_type']}")

            if acc > best_acc:
                best_acc = acc
                model.save(run_dir / "best.pt", epoch, optimizer, metrics)
                print(f"  ** New best: {best_acc:.4f} — saved to {run_dir}/best.pt")

        if epoch % cfg.train.save_every == 0:
            model.save(run_dir / f"epoch_{epoch:03d}.pt", epoch)

    print(f"\nTraining complete. Best Acc@0.5: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--head_depth", type=int, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.run_name = args.run_name
    cfg.debug    = args.debug
    if args.lora_rank  is not None: cfg.model.lora_rank  = args.lora_rank
    if args.head_depth is not None: cfg.model.head_depth = args.head_depth

    main(cfg)