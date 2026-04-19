"""
models/losses.py — all loss functions for the grounding model.

Three losses:

  grounding_loss()
      Original cross-entropy / hard-negative CE over proposal scores.
      Supports three modes (plain CE, per-image hard negs, cross-batch hard negs).

  hard_negative_contrastive_loss()
      InfoNCE-style loss that explicitly penalises the top-k wrong-but-high-scoring
      regions (hard negatives) for each phrase.
      Applied on L2-normalised phrase embeddings vs region embeddings.

  token_entropy_loss()
      Entropy minimisation on token weights from TokenWeightingMLP.
      Encourages the model to focus on important words rather than spreading weight
      uniformly across all tokens.
"""

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grounding loss (moved from head.py — unchanged)
# ---------------------------------------------------------------------------

def grounding_loss(scores:       torch.Tensor,                   # (B, N)
                   pos_idx:      torch.Tensor,                   # (B,)
                   neg_indices:  Optional[torch.Tensor] = None,  # (B, K)
                   cross_image:  bool = False,
                   ) -> torch.Tensor:
    """
    Grounding loss supporting three modes:

    Mode A — neg_indices is None:
        Plain cross-entropy treating all N proposals as the softmax class space.

    Mode B — neg_indices provided, cross_image=False:
        Per-image hard negatives. Build [pos_score | neg_scores] and apply CE
        with label 0 (positive always first).

    Mode C — neg_indices provided, cross_image=True:
        Cross-batch hard negatives. Flat indices into (B*N,) space; converted to
        per-image indices internally before gathering scores.
    """
    B, N = scores.shape
    device = scores.device

    if neg_indices is None:
        return F.cross_entropy(scores, pos_idx)

    if not cross_image:
        batch_idx  = torch.arange(B, device=device)
        pos_scores = scores[batch_idx, pos_idx].unsqueeze(1)              # (B, 1)
        neg_indices = neg_indices.clamp(0, N - 1)
        neg_scores  = scores[
            batch_idx.unsqueeze(1).expand_as(neg_indices),
            neg_indices,
        ]                                                                   # (B, K)
        logits = torch.cat([pos_scores, neg_scores], dim=1)                # (B, 1+K)
        labels = torch.zeros(B, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    # Mode C — cross-batch: map flat indices back to (image, proposal) pairs
    neg_prop_idx = neg_indices % N                                          # (B, K)
    batch_idx    = torch.arange(B, device=device)
    pos_scores   = scores[batch_idx, pos_idx].unsqueeze(1)                 # (B, 1)
    neg_scores   = scores[
        batch_idx.unsqueeze(1).expand_as(neg_prop_idx),
        neg_prop_idx.clamp(0, N - 1),
    ]                                                                       # (B, K)
    logits = torch.cat([pos_scores, neg_scores], dim=1)                    # (B, 1+K)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Hard-negative contrastive loss
# ---------------------------------------------------------------------------

def hard_negative_contrastive_loss(
    phrase_embeds:  torch.Tensor,           # (B, D)  L2-normalised
    region_embeds:  torch.Tensor,           # (B, N, D)  L2-normalised
    pos_idx:        torch.Tensor,           # (B,)  int64
    proposal_mask:  torch.Tensor,           # (B, N)  bool, True = valid region
    k:              int   = 4,
    temperature:    float = 0.07,
    penalty_factor: float = 1.5,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss that penalises hard negatives more.

    For each phrase in the batch:
      1. Compute cosine similarity to every region in the same image.
      2. Mask out the ground-truth positive and any padding proposals.
      3. Select the top-k highest-scoring wrong regions (hard negatives).
      4. Scale hard-negative logits by penalty_factor (>1 → harder problem).
      5. Apply cross-entropy with the positive at index 0.

    If fewer than k valid negatives exist for an item (e.g., only 1 region),
    the surplus logit slots are -inf → exp(-inf)=0, which is safe for CE.
    """
    if k == 0:
        return phrase_embeds.new_tensor(0.0)

    B, N, D = region_embeds.shape
    device  = phrase_embeds.device

    # Cosine similarity: (B, D) × (B, N, D) → (B, N)
    # phrase_embeds and region_embeds are already L2-normalised
    sim = torch.einsum("bd,bnd->bn", phrase_embeds, region_embeds)   # (B, N)

    # Positive scores
    batch_idx  = torch.arange(B, device=device)
    pos_idx_   = pos_idx.clamp(0, N - 1)
    pos_scores = sim.gather(1, pos_idx_.unsqueeze(1)).squeeze(1)      # (B,)

    # Hard-negative mask: valid region AND not the positive
    pos_one_hot = torch.zeros(B, N, dtype=torch.bool, device=device)
    pos_one_hot.scatter_(1, pos_idx_.unsqueeze(1), True)
    neg_mask = proposal_mask & ~pos_one_hot                           # (B, N)

    # Set invalid / positive positions to -inf, then take top-k
    sim_neg = sim.masked_fill(~neg_mask, float("-inf"))               # (B, N)
    actual_k = min(k, N - 1)
    topk_scores, _ = sim_neg.topk(actual_k, dim=1)                   # (B, k)

    # Apply temperature and penalty factor to hard negatives
    pos_logit  = pos_scores / temperature                             # (B,)
    neg_logits = topk_scores * penalty_factor / temperature           # (B, k)

    # [pos | hard_negs] → (B, 1+k), label=0 (positive always first)
    all_logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
    labels     = torch.zeros(B, dtype=torch.long, device=device)
    return F.cross_entropy(all_logits, labels)


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

    # Hinge: only penalize if entropy too low
    loss = torch.clamp(target - seq_entropy, min=0.0)

    return loss.mean()                                   # scalar
