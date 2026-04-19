"""
models/losses.py — all loss functions for the grounding model.

  grounding_loss()        — cross-entropy over proposal scores (plain CE).
  inbatch_contrastive_loss() — cross-batch InfoNCE: phrase vs all regions in batch.
  token_entropy_loss()    — entropy minimisation on token weights.
  localization_loss()     — L1 + GIoU on direct box prediction.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


# ---------------------------------------------------------------------------
# Grounding loss (moved from head.py — unchanged)
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) cxcywh → (..., 4) xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def localization_loss(pred_boxes:   torch.Tensor,   # (B, 4) normalized cxcywh
                      target_boxes:  torch.Tensor,   # (B, 4) normalized cxcywh
                      lambda_l1:    float = 5.0,
                      lambda_giou:  float = 2.0,
                      ) -> torch.Tensor:
    """
    L1 + GIoU localization loss on normalized cxcywh boxes.
    Clamps predicted w/h to ≥1e-4 to avoid zero-area boxes in GIoU.
    """
    # Clamp w and h to avoid degenerate boxes
    cx, cy, w, h = pred_boxes.unbind(-1)
    pred_boxes = torch.stack([cx, cy, w.clamp(min=1e-4), h.clamp(min=1e-4)], dim=-1)

    loss_l1 = F.l1_loss(pred_boxes, target_boxes, reduction='mean')

    pred_xyxy   = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    giou_matrix = generalized_box_iou(pred_xyxy, target_xyxy)   # (B, B)
    loss_giou   = (1 - giou_matrix.diagonal()).mean()

    return lambda_l1 * loss_l1 + lambda_giou * loss_giou


def grounding_loss(scores:  torch.Tensor,   # (B, N)
                   pos_idx: torch.Tensor,   # (B,)
                   ) -> torch.Tensor:
    """Plain cross-entropy over all N proposal scores."""
    B, N = scores.shape
    return F.cross_entropy(scores, pos_idx.clamp(0, N - 1))


# ---------------------------------------------------------------------------
# In-batch contrastive loss
# ---------------------------------------------------------------------------

def inbatch_contrastive_loss(
    phrase_embeds:  torch.Tensor,   # (B, D)  L2-normalised
    region_embeds:  torch.Tensor,   # (B, N, D)  L2-normalised
    pos_idx:        torch.Tensor,   # (B,)  int64
    proposal_mask:  torch.Tensor,   # (B, N)  bool, True = valid region
    temperature:    float = 0.07,
) -> torch.Tensor:
    """
    Cross-batch InfoNCE loss.

    For phrase i the positive is region_embeds[i, pos_idx[i]].
    Negatives are every valid proposal from every *other* image in the batch,
    giving B*(N-1) negatives per phrase instead of just N-1 within one image.

    Implementation:
      1. Flatten regions to (B*N, D) and compute (B, B*N) similarity matrix.
      2. Mark positive slot for each phrase: flat index = i*N + pos_idx[i].
      3. Mask own-image slots and padding to -inf (but keep the positive).
      4. cross_entropy with label = positive flat index.

    If B==1 there are no cross-image negatives; returns a graph-connected zero.
    """
    B, N, D = region_embeds.shape
    device  = phrase_embeds.device

    if B == 1:
        return (phrase_embeds.sum() + region_embeds.sum()) * 0.0

    flat_regions = region_embeds.view(B * N, D)                        # (B*N, D)
    sim = torch.mm(phrase_embeds, flat_regions.t()) / temperature      # (B, B*N)

    # Flat index of each phrase's positive proposal
    batch_idx = torch.arange(B, device=device)
    pos_flat  = batch_idx * N + pos_idx.clamp(0, N - 1)               # (B,)

    # Build validity mask: True = kept in denominator
    # Start valid, then mask own-image slots, then mask padding
    valid = torch.ones(B, B * N, dtype=torch.bool, device=device)
    for i in range(B):
        valid[i, i * N : i * N + N] = False
    flat_mask = proposal_mask.reshape(B * N)                           # (B*N,)
    valid &= flat_mask.unsqueeze(0)                                    # (B, B*N)
    # Always keep positive slot (it belongs to own image but is our target)
    valid[batch_idx, pos_flat] = True

    sim = sim.masked_fill(~valid, float("-inf"))
    return F.cross_entropy(sim, pos_flat)


# ---------------------------------------------------------------------------
# Token-focused entropy loss
# ---------------------------------------------------------------------------

def token_entropy_loss(token_weights, token_mask, eps=1e-8, target=1.0):
    # Mask out padding tokens
    w = token_weights.masked_fill(~token_mask, 0.0)

    # Re-normalize so probabilities sum to 1
    w = w / (w.sum(dim=-1, keepdim=True) + eps)

    # Compute entropy
    per_token = -w * torch.log(w + eps)
    seq_entropy = per_token.sum(dim=-1)

    Padding positions are explicitly zeroed to avoid log(0+eps) noise, even
    though the MLP's masked softmax already makes them ~0.
    """
    # Explicit zero-out for numerical safety
    w = token_weights * token_mask.float()                            # (B, L)
    per_token = -w * torch.log(w.clamp(min=eps))                     # (B, L)
    seq_entropy = per_token.sum(dim=-1)                               # (B,)
    return seq_entropy.mean()                                         # scalar
