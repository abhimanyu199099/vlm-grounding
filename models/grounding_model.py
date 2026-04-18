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
from .losses import (
    grounding_loss,
    hard_negative_contrastive_loss,
    token_entropy_loss,
)


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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                batch:       dict,
                neg_mining:  Optional[Tuple[torch.Tensor, bool]] = None,
                ) -> dict:
        """
        Args:
            batch: dict produced by data.collate_fn with keys:
                phrase_tokens   : (B, 77)
                proposal_crops  : (B, N, 3, H, W)
                proposals       : (B, N, 4)
                proposal_mask   : (B, N) bool
                pos_idx         : (B,)
                gt_box          : (B, 4)
                image_id        : list[str]
                phrase          : list[str]
                entity_type     : list[str]

            neg_mining: optional tuple (neg_indices, cross_image) from
                NegativeMiner.mine(). Pass None to use plain cross-entropy.
                    neg_indices  : (B, K) LongTensor
                    cross_image  : bool

        Returns dict:
            scores           : (B, N)  — raw grounding logits (padding → -inf)
            loss             : scalar  — weighted total loss
            grounding_loss   : scalar  — CE component (detached, for logging)
            contrastive_loss : scalar  — hard-neg InfoNCE component (detached)
            entropy_loss     : scalar  — token entropy component (detached)
            preds            : (B,)    — argmax predicted proposal index
            token_weights    : (B, L)  — per-token importance weights (detached)
            phrase_embeds    : (B, D)  — L2-normed phrase embeddings (for miner)
            region_embeds    : (B, N, D) — L2-normed region embeddings (for miner)
        """
        device = next(self.head.parameters()).device

        phrase_tokens = batch["phrase_tokens"].to(device)         # (B, 77)
        pos_idx       = batch["pos_idx"].to(device)               # (B,)

        proposal_mask = batch.get("proposal_mask")
        if proposal_mask is not None:
            proposal_mask = proposal_mask.to(device)              # (B, N) bool

        # Attention mask: 1 for real tokens, 0 for padding (CLIP pads with 0)
        attn_mask = (phrase_tokens != 0).to(device)               # (B, 77)

        # ---- Encode — skip if pre-computed embeddings are in the batch ----
        if "text_hidden" in batch and "region_embeds" in batch:
            text_hidden   = batch["text_hidden"].to(device)       # (B, L, D_text)
            region_embeds = batch["region_embeds"].to(device)     # (B, N, D_proj)
            phrase_embeds = batch["phrase_embed"].to(device)      # (B, D_proj)
        else:
            proposal_crops = batch["proposal_crops"].to(device)   # (B, N, 3, H, W)
            text_hidden    = self.encoder.encode_text(phrase_tokens, attn_mask)
            region_embeds  = self.encoder.encode_region(proposal_crops)
            phrase_embeds  = self.encoder.encode_phrase(phrase_tokens, attn_mask)

        # ---- Score ----
        # head now returns (scores, token_weights, query)
        scores, token_weights, query = self.head(
            text_hidden=text_hidden,
            region_embeds=region_embeds,
            text_mask=attn_mask,
            proposal_mask=proposal_mask,
        )                                                           # (B,N), (B,L), (B,D)

        # ---- Loss 1: grounding loss (CE over proposals) ----
        neg_indices = None
        cross_image = False
        if neg_mining is not None:
            neg_indices, cross_image = neg_mining
            if neg_indices is not None:
                neg_indices = neg_indices.to(device)

        g_loss = grounding_loss(scores, pos_idx, neg_indices, cross_image)

        # ---- Loss 2: hard-negative contrastive loss ----
        # Use the frozen CLIP phrase embedding (already L2-normalised) so the
        # contrastive loss does not conflict with the grounding head's own scorer.
        # region_embeds are already L2-normalised from encode_region().
        phrase_q      = F.normalize(phrase_embeds, dim=-1)          # (B, D)
        region_norm   = region_embeds                               # (B, N, D) already normed
        cfg_m         = self.cfg.model
        c_loss = hard_negative_contrastive_loss(
            phrase_embeds=phrase_q,
            region_embeds=region_norm,
            pos_idx=pos_idx,
            proposal_mask=proposal_mask if proposal_mask is not None
                          else torch.ones(scores.shape, dtype=torch.bool, device=device),
            k=cfg_m.hard_neg_k,
            temperature=cfg_m.contrastive_temperature,
            penalty_factor=cfg_m.hard_neg_penalty,
        )

        # ---- Loss 3: token entropy regularisation ----
        e_loss = token_entropy_loss(token_weights, attn_mask.bool())

        # ---- Combine ----
        total = (
            g_loss
            + cfg_m.contrastive_loss_weight * c_loss
            + cfg_m.entropy_loss_weight     * e_loss
        )

        preds = scores.argmax(dim=1)                                # (B,)

        return {
            "scores":           scores,
            "loss":             total,
            "grounding_loss":   g_loss.detach(),
            "contrastive_loss": c_loss.detach(),
            "entropy_loss":     e_loss.detach(),
            "preds":            preds,
            "token_weights":    token_weights.detach(),
            "phrase_embeds":    phrase_embeds,
            "region_embeds":    region_embeds,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: Path, epoch: int,
             optimizer=None, scheduler=None, metrics: dict = None):
        ckpt = {
            "epoch":      epoch,
            "head_state": self.head.state_dict(),
            "config":     self.cfg,
            "metrics":    metrics or {},
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        torch.save(ckpt, path)
        return ckpt

    def load(self, path: Path, optimizer=None) -> dict:
        """
        Load head weights from a checkpoint.
        Returns the full checkpoint dict (caller can inspect epoch, metrics, etc.).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # strict=True by default — will error if architecture changed
        self.head.load_state_dict(ckpt["head_state"], strict=True)
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @property
    def trainable_parameters(self):
        """Parameters to pass to the optimizer — head only."""
        return self.head.trainable_parameters()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())