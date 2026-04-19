"""
models/grounding_model.py — top-level model: FrozenCLIPEncoder + GroundingHead.

MEMBER B owns this file.

This is the single object instantiated in train.py, evaluate.py, and demo/.
It owns the encode → score → loss pipeline, and handles device placement,
checkpoint save/load, and the interface to NegativeMiner.

Key wiring decisions:
  - GroundingHead is constructed AFTER FrozenCLIPEncoder so that
    text_hidden_dim and region_proj_dim can be read off the encoder's
    config rather than hardcoded.
  - forward() accepts an optional (neg_indices, cross_image) tuple returned
    by NegativeMiner.mine(). When absent, plain cross-entropy is used.
  - Only head weights are saved in checkpoints; encoder weights are always
    re-loaded from HuggingFace.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

import torch.nn.functional as F

from config import Config
from .encoder import FrozenCLIPEncoder
from .head import GroundingHead
from .rpn_encoder import RPNRegionEncoder
from .box_encoding import BoxPositionalEncoding, xyxy_pixel_to_cxcywh_norm
from .losses import (
    grounding_loss,
    inbatch_contrastive_loss,
    token_entropy_loss,
    localization_loss,
)


def _compute_pos_idx(proposals: torch.Tensor,   # (B, N, 4) xyxy pixel
                     gt_boxes:  torch.Tensor,   # (B, 4)   xyxy pixel
                     mask:      torch.Tensor,   # (B, N) bool
                     ) -> torch.Tensor:         # (B,) long
    """Find the proposal with highest IoU to gt_box for each image."""
    B, N, _ = proposals.shape
    # Use vectorised IoU: broadcast (B, N, 4) vs (B, 1, 4)
    gt = gt_boxes.unsqueeze(1)                                    # (B, 1, 4)
    inter_x1 = torch.max(proposals[..., 0], gt[..., 0])
    inter_y1 = torch.max(proposals[..., 1], gt[..., 1])
    inter_x2 = torch.min(proposals[..., 2], gt[..., 2])
    inter_y2 = torch.min(proposals[..., 3], gt[..., 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)  # (B, N)
    area_p   = ((proposals[..., 2] - proposals[..., 0]) *
                (proposals[..., 3] - proposals[..., 1])).clamp(0)
    area_g   = ((gt_boxes[..., 2] - gt_boxes[..., 0]) *
                (gt_boxes[..., 3] - gt_boxes[..., 1])).clamp(0).unsqueeze(1)
    union    = area_p + area_g - inter
    iou      = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    iou      = iou.masked_fill(~mask, -1.0)
    return iou.argmax(dim=1)                                      # (B,)


class GroundingModel(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.cfg     = config

        # Build encoder first — its config attributes tell us the right dims
        self.encoder = FrozenCLIPEncoder(config)

        # Build head using dims read from the encoder, not from config.model
        self.head = GroundingHead(
            config=config,
            text_hidden_dim=self.encoder.text_hidden_dim,   # 512 for ViT-B/32
            region_proj_dim=self.encoder.projection_dim,    # 512 for ViT-B/32
        )

        self.rpn_encoder = RPNRegionEncoder(
            out_dim=self.encoder.projection_dim,
            max_proposals=config.model.max_proposals,
            frozen=True,
        )
        self.box_pos_enc = BoxPositionalEncoding(d_model=config.model.embed_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict from data.collate_fn (uncached/RPN path only):
                phrase_tokens : (B, 77)
                images        : (B, 3, 224, 224)
                image_sizes   : list of (H, W) tuples
                proposals     : (B, N, 4)   xyxy pixel — for eval
                proposal_mask : (B, N) bool
                pos_idx       : (B,)
                gt_box        : (B, 4)      xyxy pixel
                gt_box_norm   : (B, 4)      normalized cxcywh
                entity_type   : list[str]

        Returns dict:
            scores           : (B, N)
            loss             : scalar
            grounding_loss   : scalar (detached, for logging)
            contrastive_loss : scalar (detached, for logging)
            entropy_loss     : scalar (detached, for logging)
            loc_loss         : scalar (detached, for logging)
            pred_boxes       : (B, 4)
            preds            : (B,)
            proposals        : (B, N, 4)  RPN boxes (for eval)
        """
        device = next(self.head.parameters()).device

        phrase_tokens = batch["phrase_tokens"].to(device, non_blocking=True)   # (B, 77)
        pos_idx       = batch["pos_idx"].to(device, non_blocking=True)         # (B,)

        proposal_mask = batch.get("proposal_mask")
        if proposal_mask is not None:
            proposal_mask = proposal_mask.to(device, non_blocking=True)        # (B, N) bool

        proposals = batch.get("proposals")
        if proposals is not None:
            proposals = proposals.to(device, non_blocking=True)                # (B, N, 4)

        # Use tokenizer's attention_mask directly — pad_token_id=49407 (not 0),
        # so (phrase_tokens != 0) would incorrectly mark all positions as real.
        attn_mask = batch["phrase_attn_mask"].to(device, non_blocking=True)    # (B, 77)

        # ---- Encode — skip if pre-computed embeddings are in the batch ----
        if "text_hidden" in batch and "region_embeds" in batch:
            text_hidden   = batch["text_hidden"].to(device, non_blocking=True)    # (B, L, D_text)
            region_embeds = batch["region_embeds"].to(device, non_blocking=True)  # (B, N, D_proj)
            phrase_embeds = batch["phrase_embed"].to(device, non_blocking=True)   # (B, D_proj)
        else:
            proposal_crops = batch["proposal_crops"].to(device, non_blocking=True)  # (B, N, 3, H, W)
            text_hidden    = self.encoder.encode_text(phrase_tokens, attn_mask)
            region_embeds  = self.encoder.encode_region(proposal_crops)
            phrase_embeds  = self.encoder.encode_phrase(phrase_tokens, attn_mask)

        text_hidden   = self.encoder.encode_text(phrase_tokens, attn_mask)
        phrase_embeds = self.encoder.encode_phrase(phrase_tokens, attn_mask)

        # ---- Head ----
        scores, token_weights, query, pred_boxes = self.head(
            text_hidden=text_hidden,
            region_embeds=region_embeds,
            text_mask=attn_mask,
            proposal_mask=proposal_mask,
            proposals=proposals,
        )                                                           # (B,N), (B,L), (B,d)

        cfg_m = self.cfg.model

        # ---- Loss 1: grounding CE ----
        g_loss = grounding_loss(scores, pos_idx)

        # ---- Loss 2: hard-negative contrastive loss ----
        # Use the head's query (learned weighted-text projection, dim=proj_dim) so that
        # the contrastive loss actually trains the head's parameters.
        # region_embeds are already L2-normalised from encode_region().
        phrase_q    = F.normalize(query.float(), dim=-1).to(query.dtype)
        phrase_q    = torch.nan_to_num(phrase_q, nan=0.0)
        rp          = self.head.region_proj(region_embeds)
        region_norm = F.normalize(rp.float(), dim=-1).to(rp.dtype)
        region_norm = torch.nan_to_num(region_norm, nan=0.0)
        cfg_m         = self.cfg.model
        c_loss = hard_negative_contrastive_loss(
            phrase_embeds=phrase_q,
            region_embeds=region_norm,
            pos_idx=pos_idx,
            proposal_mask=rpn_mask,
            temperature=cfg_m.contrastive_temperature,
        )

        # ---- Loss 3: token entropy regularisation ----
        e_loss = token_entropy_loss(
            token_weights, attn_mask.bool(), target=cfg_m.entropy_target
        )

        # ---- Loss 4: localization ----
        gt_box_norm = batch["gt_box_norm"].to(device)             # (B, 4) normalized cxcywh
        loc_loss    = localization_loss(pred_boxes, gt_box_norm)

        total = (
            g_loss
            + cfg_m.contrastive_loss_weight  * c_loss
            + cfg_m.entropy_loss_weight      * e_loss
            + cfg_m.localization_loss_weight * loc_loss
        )

        return {
            "scores":           scores,
            "loss":             total,
            "grounding_loss":   g_loss.detach(),
            "contrastive_loss": c_loss.detach(),
            "entropy_loss":     e_loss.detach(),
            "loc_loss":         loc_loss.detach(),
            "pred_boxes":       pred_boxes.detach(),
            "preds":            scores.argmax(dim=1),
            "proposals":        rpn_boxes_xyxy.detach(),
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: Path, epoch: int,
             optimizer=None, scheduler=None, metrics: dict = None):
        ckpt = {
            "epoch":         epoch,
            "head_state":    self.head.state_dict(),
            "rpn_state":     self.rpn_encoder.state_dict(),
            "box_enc_state": self.box_pos_enc.state_dict(),
            "config":        self.cfg,
            "metrics":       metrics or {},
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        torch.save(ckpt, path)
        return ckpt

    def load(self, path: Path, optimizer=None) -> dict:
        """
        Load weights from a checkpoint.
        Returns the full checkpoint dict (caller can inspect epoch, metrics, etc.).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.head.load_state_dict(ckpt["head_state"], strict=False)
        if "rpn_state" in ckpt:
            self.rpn_encoder.load_state_dict(ckpt["rpn_state"], strict=False)
        if "box_enc_state" in ckpt:
            self.box_pos_enc.load_state_dict(ckpt["box_enc_state"], strict=False)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @property
    def trainable_parameters(self):
        """Parameters to pass to the optimizer — head + box_pos_enc."""
        return (
            [p for p in self.head.parameters() if p.requires_grad]
            + list(self.box_pos_enc.parameters())
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())