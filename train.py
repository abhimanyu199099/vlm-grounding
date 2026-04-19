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
import math
import datetime
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.amp import GradScaler, autocast
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

def train_one_epoch(model, raw_model, loader, optimizer, scaler, scheduler, miner, evaluator, cfg, epoch):
    model.train()
    total_loss       = 0.0
    total_g_loss     = 0.0
    total_c_loss     = 0.0
    total_e_loss     = 0.0
    total_loc_loss   = 0.0

    ddp     = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    device  = next(raw_model.parameters()).device
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main, leave=True)

    for step, batch in enumerate(pbar):
        is_accum_boundary = (step + 1) % accum == 0 or (step + 1) == len(loader)
        sync_ctx = contextlib.nullcontext() if (not ddp or is_accum_boundary) else model.no_sync()

        # Mine hard negatives using frozen CLIP embeddings — no head needed,
        # so run outside autocast with no_grad to avoid wasting compute.
        neg_mining = None
        if miner is not None:
            with torch.no_grad():
                phrase_tokens = batch["phrase_tokens"].to(device, non_blocking=True)
                attn_mask     = batch["phrase_attn_mask"].to(device, non_blocking=True)
                if "phrase_embed" in batch and "region_embeds" in batch:
                    phrase_embeds = batch["phrase_embed"].to(device, non_blocking=True)
                    region_embeds = batch["region_embeds"].to(device, non_blocking=True)
                else:
                    proposal_crops = batch["proposal_crops"].to(device, non_blocking=True)
                    phrase_embeds  = raw_model.encoder.encode_phrase(phrase_tokens, attn_mask)
                    region_embeds  = raw_model.encoder.encode_region(proposal_crops)
            neg_indices, cross_image = miner.mine(batch, phrase_embeds, region_embeds)
            neg_mining = (neg_indices, cross_image)

        with autocast('cuda', enabled=cfg.train.mixed_precision):
            out  = model(batch, neg_mining=neg_mining)
            loss = out["loss"]

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            raw_model.trainable_parameters, cfg.train.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss     += loss.item()
        total_g_loss   += out["grounding_loss"].item()
        total_c_loss   += out["contrastive_loss"].item()
        total_e_loss   += out["entropy_loss"].item()
        total_loc_loss += out["loc_loss"].item()

        if is_accum_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.trainable_parameters, cfg.train.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % cfg.train.log_every == 0:
            tw = out["token_weights"]
            entropy = -(tw * torch.log(tw.clamp(1e-8))).sum(-1).mean().item()
            grad_norm = sum(p.grad.norm().item() for p in raw_model.head.parameters() if p.grad is not None)
            log_temp  = raw_model.head.log_temp.item()
            pos_idx_b = batch["pos_idx"]
            print(f"entropy={entropy:.3f}  grad_norm={grad_norm:.4f}  log_temp={log_temp:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"pos_idx_mean={pos_idx_b.float().mean():.1f}  pos_idx_max={pos_idx_b.max().item()}")
            n = step + 1
            pbar.set_postfix({
                "loss":        f"{total_loss     / n:.4f}",
                "grounding":   f"{total_g_loss   / n:.4f}",
                "contrastive": f"{total_c_loss   / n:.4f}",
                "entropy":     f"{total_e_loss   / n:.4f}",
                "loc":         f"{total_loc_loss / n:.4f}",
            })

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, evaluator, cfg):
    model.eval()
    evaluator.reset()

    for batch in loader:
        out          = model(batch, neg_mining=None)
        preds        = out["preds"]                              # (B,)
        gt_boxes     = batch["gt_box"]                           # (B, 4) xyxy pixel
        entity_types = batch["entity_type"]

        # Use RPN proposals from model output if available, else batch proposals
        proposals = out.get("proposals", batch.get("proposals"))

        if proposals is not None:
            evaluator.update_from_indices(
                pred_idx=preds,
                proposals=proposals,
                gt_boxes=gt_boxes,
                entity_types=entity_types,
                scores=out["scores"].detach(),
            )
        else:
            evaluator.update_from_indices(
                pred_idx=preds,
                proposals=batch["proposals"],
                gt_boxes=gt_boxes,
                entity_types=entity_types,
                scores=out["scores"].detach(),
            )

        # Also evaluate direct box prediction if available
        if "pred_boxes" in out and "gt_box_norm" in batch:
            evaluator.update_direct_boxes(
                pred_boxes_norm=out["pred_boxes"],
                gt_boxes_norm=batch["gt_box_norm"],
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
    device = next(model.parameters()).device

    for batch in loader:
        phrase_tokens = batch["phrase_tokens"].to(device, non_blocking=True)
        attn_mask     = batch["phrase_attn_mask"].to(device, non_blocking=True)

        if "region_embeds" in batch:
            region_embeds = batch["region_embeds"].to(device, non_blocking=True)  # (B, N, D)
            phrase_embeds = batch["phrase_embed"].to(device, non_blocking=True)   # (B, D)
        else:
            proposal_crops = batch["proposal_crops"].to(device, non_blocking=True)
            phrase_embeds = model.encoder.encode_phrase(phrase_tokens, attn_mask)
            region_embeds = model.encoder.encode_region(proposal_crops)

        scores = torch.einsum("bd,bnd->bn", phrase_embeds, region_embeds)

        if "proposal_mask" in batch:
            mask = batch["proposal_mask"].to(device, non_blocking=True)
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
# Oracle recall check
# ---------------------------------------------------------------------------

@torch.no_grad()
def oracle_recall(model, loader):
    """
    Check what fraction of GT boxes are covered by at least one RPN proposal at IoU≥0.5.
    Target ≥0.75; if below 0.65 consider increasing max_proposals.
    """
    from eval.metrics import iou as compute_iou
    hits, total = 0, 0
    hits5 = 0

    raw_model = model.module if hasattr(model, 'module') else model
    for batch in loader:
        device    = next(raw_model.parameters()).device
        images    = batch["images"].to(device)
        gt_boxes  = batch["gt_box"]                            # (B, 4) xyxy pixel

        _, boxes_xyxy, mask = raw_model.rpn_encoder(images, batch["image_sizes"])
        boxes_xyxy = boxes_xyxy.cpu()
        mask       = mask.cpu()

        for i in range(gt_boxes.size(0)):
            gt  = gt_boxes[i]
            n   = mask[i].sum().item()
            props = boxes_xyxy[i, :n]
            ious  = [compute_iou(props[j], gt) for j in range(n)]
            hits  += any(v >= 0.5 for v in ious)
            hits5 += any(v >= 0.5 for v in sorted(ious, reverse=True)[:5])
            total += 1

    recall    = hits  / total if total else 0.0
    recall_r5 = hits5 / total if total else 0.0
    print(f"  Oracle recall (any proposal IoU≥0.5): {recall:.4f}  |  R@5: {recall_r5:.4f}")
    if recall < 0.65:
        print("  WARNING: oracle recall < 0.65 — consider increasing max_proposals")
    return recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config):
    # --- DDP setup ---
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(hours=2))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    set_seed(cfg.train.seed + local_rank)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.clip_model)

    # Datasets
    train_ds = Flickr30kGroundingDataset(cfg, split="train",
                                         tokenizer=tokenizer, debug=cfg.debug)
    val_ds   = Flickr30kGroundingDataset(cfg, split="val",
                                         tokenizer=tokenizer, debug=cfg.debug)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    persistent    = cfg.data.num_workers > 0
    train_loader  = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                               shuffle=(train_sampler is None),
                               sampler=train_sampler,
                               collate_fn=collate_fn,
                               num_workers=cfg.data.num_workers,
                               pin_memory=cfg.data.pin_memory,
                               persistent_workers=persistent)
    val_loader    = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=cfg.data.num_workers,
                               pin_memory=cfg.data.pin_memory,
                               persistent_workers=persistent)

    # Model
    raw_model = GroundingModel(cfg).to(device)

    if is_main:
        print(f"Trainable params : {raw_model.trainable_param_count():,}")
        print(f"Total params     : {raw_model.total_param_count():,}")

    # CLIP baseline — run before DDP wrap so no collective ops are active.
    # Requires cached mode (batch needs region_embeds / proposals).
    evaluator = GroundingEvaluator(cfg)
    if is_main and not cfg.skip_baseline and cfg.data.use_cache:
        print("\n--- CLIP baseline (no training) ---")
        baseline = clip_baseline(raw_model, val_loader, evaluator, cfg)
        print(f"  Acc@0.5: {baseline['acc@0.5']:.4f}  |  mean IoU: {baseline['mean_iou']:.4f}")
    elif is_main and not cfg.skip_baseline and not cfg.data.use_cache:
        print("\n--- CLIP baseline skipped (requires cached embeddings) ---")

    # Oracle recall check (only in uncached / RPN mode)
    if is_main and not cfg.data.use_cache and not cfg.skip_baseline:
        print("\n--- Oracle RPN recall check ---")
        oracle_recall(raw_model, val_loader)

    if ddp:
        dist.barrier()  # all ranks wait for rank 0 to finish baseline

    # Wrap in DDP after baseline is done
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    # Optimizer — per-module learning rates
    head = raw_model.head
    optimizer = torch.optim.AdamW([
        {"params": head.text_proj.parameters(),      "lr": 1e-4},
        {"params": head.token_weighter.parameters(), "lr": 1e-4},
        {"params": head.region_proj.parameters(),    "lr": 2e-4},
        {"params": head.layers.parameters(),         "lr": 2e-4},
        {"params": head.scorer.parameters(),         "lr": 1e-4},
        {"params": head.box_head.parameters(),       "lr": 3e-4},
        {"params": raw_model.box_pos_enc.parameters(), "lr": 3e-4},
    ], weight_decay=cfg.train.weight_decay)
    scaler = GradScaler('cuda', enabled=cfg.train.mixed_precision)
    miner  = NegativeMiner(cfg, clip_model=raw_model.encoder.clip)

    # Training loop
    best_acc    = 0.0
    start_epoch = 1
    run_dir     = CKPT_DIR / cfg.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg.train.epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        progress = step / max(1, total_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if cfg.resume:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ckpt = raw_model.load(cfg.resume, optimizer=optimizer)
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt.get("metrics", {}).get("acc@0.5", 0.0)
        if is_main:
            print(f"Resumed from {cfg.resume} (epoch {ckpt['epoch']}, best acc {best_acc:.4f})")

    for epoch in tqdm(range(start_epoch, cfg.train.epochs + 1), desc="Training", disable=not is_main):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\n=== Epoch {epoch}/{cfg.train.epochs} ===")
        train_loss = train_one_epoch(
            model, raw_model, train_loader, optimizer, scaler, scheduler, miner, evaluator, cfg, epoch
        )

        if epoch % cfg.train.eval_every == 0:
            # All ranks evaluate — avoids rank 0 blocking others during a long eval window
            metrics = evaluate(raw_model, val_loader, evaluator, cfg)
            if is_main:
                acc = metrics["acc@0.5"]
                print(f"  Val Acc@0.5: {acc:.4f}  |  Recall@5: {metrics.get('recall@5', '?')}  |  mAP50: {metrics.get('mAP50', '?')}  |  mean IoU: {metrics['mean_iou']:.4f}")
                print(f"  Breakdown: {metrics['acc_by_type']}")
                if acc > best_acc:
                    best_acc = acc
                    raw_model.save(run_dir / "best.pt", epoch, optimizer, scheduler, metrics)
                    print(f"  ** New best: {best_acc:.4f} — saved to {run_dir}/best.pt")

        if epoch % cfg.train.save_every == 0 and is_main:
            raw_model.save(run_dir / f"epoch_{epoch:03d}.pt", epoch, scheduler=scheduler)

        # Unfreeze RPN head after epoch 10
        if epoch == 10:
            for p in raw_model.rpn_encoder.detector.rpn.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": [p for p in raw_model.rpn_encoder.detector.rpn.parameters()
                           if p.requires_grad],
                "lr": 1e-5,
            })
            if is_main:
                print("  ** Unfroze RPN head (lr=1e-5)")

        if ddp:
            dist.barrier()  # wait for rank 0 to finish eval/checkpoint before next epoch

    if ddp:
        dist.destroy_process_group()

    print(f"\nTraining complete. Best Acc@0.5: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--head_depth", type=int, default=None)
    cache_grp = parser.add_mutually_exclusive_group()
    cache_grp.add_argument("--use_cache", dest="use_cache", action="store_true",  default=None,
                           help="load precomputed CLIP embeddings from cache (default: on)")
    cache_grp.add_argument("--no_cache",  dest="use_cache", action="store_false",
                           help="disable cache; run full CLIP forward pass every step")
    parser.add_argument("--data_fraction", type=float, default=None,
                        help="random fraction of training data to use, e.g. 0.4")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="skip CLIP baseline eval before training")
    parser.add_argument("--resume", default=None,
                        help="path to checkpoint to resume training from")
    parser.add_argument("--accum_steps", type=int, default=None,
                        help="gradient accumulation steps (default: 1)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.run_name = args.run_name
    cfg.debug    = args.debug
    if args.lora_rank     is not None: cfg.model.lora_rank    = args.lora_rank
    if args.head_depth    is not None: cfg.model.head_depth   = args.head_depth
    if args.use_cache     is not None: cfg.data.use_cache     = args.use_cache
    if args.data_fraction is not None: cfg.data.data_fraction = args.data_fraction
    if args.skip_baseline: cfg.skip_baseline          = args.skip_baseline
    if args.resume:        cfg.resume                  = args.resume
    if args.accum_steps is not None: cfg.train.accum_steps = args.accum_steps

    main(cfg)