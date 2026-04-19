"""
models/head.py — trainable grounding head on top of frozen CLIP features.

MEMBER B owns this file.

Architecture:
  Input:
    text_hidden  : (B, L, D_text)   — token-level features from encode_text()
                                       D_text = text_config.hidden_size (512 for ViT-B/32)
    region_embeds: (B, N, D_proj)   — CLS features from encode_region()
                                       D_proj = projection_dim (512 for ViT-B/32)
    text_mask    : (B, L) bool      — True for real tokens
    proposal_mask: (B, N) bool      — True for valid (non-padding) proposals

  Processing:
    1. text_proj:    (B, L, D_text) → (B, L, D)   learned linear
       region_proj:  (B, N, D_proj) → (B, N, D)   learned linear
    2. head_depth × TextOverRegionAttention layers:
         Q = text tokens, K = V = region features
    3. TokenWeightingMLP: (B, L, D_text) → (B, L) scalar weights, masked softmax
       Weighted sum of text_proj output replaces mean-pool → query (B, D)
    4. scorer linear → query (B, D)
    5. Dot product with region_proj output → scores (B, N)

  Returns: (scores, token_weights, query)
    scores        : (B, N)  — un-normalised logits per proposal
    token_weights : (B, L)  — per-token importance weights (sum=1 over real tokens)
    query         : (B, D)  — weighted phrase embedding (used by contrastive loss)

  Loss functions live in models/losses.py (grounding_loss, hard_negative_contrastive_loss,
  token_entropy_loss). grounding_loss is re-exported here for backwards compatibility.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .losses import grounding_loss  # re-exported; full loss suite is in losses.py


# ---------------------------------------------------------------------------
# Single cross-attention layer — text queries attend over region keys/values
# ---------------------------------------------------------------------------

class TextOverRegionAttention(nn.Module):
    """
    Multi-head cross-attention: Q from text tokens, K/V from region features.

    Pre-norm on both text (Q) and regions (K) before projection.
    Residual connection on the output.

    NaN guard: if all region slots are masked (padding-only batch item),
    softmax over all-(-inf) produces NaN. We replace those with zeros before
    the weighted sum.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, \
            f"embed_dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                text:        torch.Tensor,                    # (B, L, D)
                regions:     torch.Tensor,                    # (B, N, D)
                region_mask: Optional[torch.Tensor] = None,  # (B, N) bool
                ) -> torch.Tensor:
        """Returns text features updated by cross-attention: (B, L, D)."""
        B, L, D = text.shape
        N  = regions.size(1)
        H  = self.num_heads
        Hd = self.head_dim

        Q = self.q_proj(self.norm_q(text))      # (B, L, D)
        K = self.k_proj(self.norm_kv(regions))  # (B, N, D)
        V = self.v_proj(regions)                # (B, N, D)  — no norm on V

        # Reshape to multi-head form
        Q = Q.view(B, L, H, Hd).transpose(1, 2)   # (B, H, L, Hd)
        K = K.view(B, N, H, Hd).transpose(1, 2)   # (B, H, N, Hd)
        V = V.view(B, N, H, Hd).transpose(1, 2)   # (B, H, N, Hd)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, N)

        # Mask padding proposals — set their logits to -inf before softmax
        if region_mask is not None:
            # region_mask: (B, N) True=valid → expand to (B, H, L, N)
            bad = ~region_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(bad, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # NaN guard: rows where ALL keys were masked produce NaN after softmax.
        # Replace with zero so those text tokens are unaffected by cross-attention.
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)
        out  = torch.matmul(attn, V)                          # (B, H, L, Hd)
        out  = out.transpose(1, 2).contiguous().view(B, L, D) # (B, L, D)
        return text + self.out_proj(out)                       # residual


# ---------------------------------------------------------------------------
# Token-weighting MLP
# ---------------------------------------------------------------------------

class TokenWeightingMLP(nn.Module):
    """
    Predicts a scalar importance weight for each text token.

    Architecture: Linear(D_text, hidden) → ReLU → Linear(hidden, 1) → squeeze
    The raw scalar logits are masked (padding → -inf) and then softmax-normalised
    so weights sum to 1.0 over real tokens per sequence.

    These weights replace mean-pooling in GroundingHead: the query vector is
    a learned weighted sum of per-token projected features rather than a simple
    average. This lets the model upweight semantically important words
    (e.g. "blue", "shirt") and downweight function words ("a", "the").

    Trainable — parameters are discovered by model.parameters() and updated
    by the AdamW optimizer alongside the rest of the head.
    """

    def __init__(self, d_text: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # → (B, L, 1)
        )

    def forward(
        self,
        text_hidden: torch.Tensor,   # (B, L, D_text)
        token_mask:  torch.Tensor,   # (B, L) bool, True = real token
    ) -> torch.Tensor:               # (B, L) softmax weights
        logits = self.net(text_hidden).squeeze(-1)            # (B, L)
        logits = logits.masked_fill(~token_mask, float("-inf"))
        return torch.softmax(logits, dim=-1)                  # (B, L)


# ---------------------------------------------------------------------------
# Full grounding head
# ---------------------------------------------------------------------------

class GroundingHead(nn.Module):
    """
    Projects CLIP features into a shared space, runs cross-attention, scores regions.

    Trainable parameters:
        text_proj   — maps text_hidden_dim → D
        region_proj — maps projection_dim  → D  (learned re-scaling)
        Q, K, V, out_proj in each TextOverRegionAttention layer
        norm, scorer in each layer / the head itself
    """

    def __init__(self, config: Config,
                 text_hidden_dim: int,
                 region_proj_dim: int):
        """
        Args:
            config          : full Config object
            text_hidden_dim : D_text from encoder (text_config.hidden_size)
            region_proj_dim : D_proj from encoder (projection_dim)

        Both dims are passed in from GroundingModel which reads them off the
        encoder after construction — no hardcoding here.
        """
        super().__init__()
        cfg = config.model
        # projected dim for lightweight interaction head
        self.d = 256

        # projections to compact space
        self.text_proj   = nn.Linear(text_hidden_dim, self.d)
        self.region_proj = nn.Linear(region_proj_dim, self.d)

        # token scoring (d -> 1)
        self.token_score = nn.Linear(self.d, 1)

        # small spatial MLP for optional proposal geometry (4 -> d)
        self.spatial_proj = nn.Linear(4, self.d)

        # keep a simple layernorm for stability
        self.norm = nn.LayerNorm(self.d)

    def forward(self,
                text_hidden:   torch.Tensor,                   # (B, L, D_text)
                region_embeds: torch.Tensor,                   # (B, N, D_proj)
                text_mask:     Optional[torch.Tensor] = None,  # (B, L) bool
                proposal_mask: Optional[torch.Tensor] = None,  # (B, N) bool
                proposals:     Optional[torch.Tensor] = None,  # (B, N, 4) optional spatial coords
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token–region interaction head (lightweight).

        Implements:
          text_proj: (B,L,d)
          region_proj: (B,N,d)
          token scoring -> token_weights (B,L)
          sim = einsum("bld,bnd->bln")
          scores = sum_l token_weights * sim -> (B,N)

        Also performs a single cross-attention-style enhancement (cheap) and
        optionally incorporates spatial features from `proposals` if provided.
        """
        B, L, _ = text_hidden.shape
        N = region_embeds.size(1)
        device = text_hidden.device

        # Project to compact space
        text_p = self.text_proj(text_hidden)           # (B, L, d)
        region_p = self.region_proj(region_embeds)     # (B, N, d)

        # Normalize projected features
        text_p = F.normalize(text_p, dim=-1)
        region_p = F.normalize(region_p, dim=-1)

        # Token scoring -> token_weights
        token_logits = self.token_score(text_p).squeeze(-1)  # (B, L)
        if text_mask is not None:
            token_logits = token_logits.masked_fill(~text_mask, float("-inf"))
        token_weights = torch.softmax(token_logits, dim=-1)  # (B, L)

        # Token–region similarity: (B, L, N)
        sim = torch.einsum("bld,bnd->bln", text_p, region_p)

        # Lightweight cross-attention enhancement (optional but enabled):
        # attn over regions per token, produce context and enhance text_p
        scale = (self.d ** 0.5)
        attn = torch.softmax(sim / scale, dim=-1)                      # (B, L, N)
        context = torch.einsum("bln,bnd->bld", attn, region_p)       # (B, L, d)
        enhanced_text = text_p + context                               # (B, L, d)

        # Recompute similarity using enhanced text (keeps token_weights from text_p)
        enhanced_text = self.norm(enhanced_text)
        sim2 = torch.einsum("bld,bnd->bln", enhanced_text, region_p)  # (B, L, N)

        # Aggregate scores across tokens
        scores = (token_weights.unsqueeze(-1) * sim2).sum(dim=1)       # (B, N)

        # If `proposals` provided, incorporate geometry features (B, N, 4)
        if proposals is not None:
            prop = proposals.to(device)
            x1, y1, x2, y2 = prop.unbind(-1)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1)
            h = (y2 - y1)
            # normalize by image-wise max extent to keep features in [0,1]
            max_x = prop[..., 2].amax(dim=1).unsqueeze(-1)            # (B,1)
            max_y = prop[..., 3].amax(dim=1).unsqueeze(-1)
            denom = torch.maximum(max_x, max_y).unsqueeze(-1) + 1e-6
            geom = torch.stack([cx.unsqueeze(-1), cy.unsqueeze(-1), w.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)
            geom = geom / denom.unsqueeze(1)  # (B, N, 4) normalized
            geom_feat = self.spatial_proj(geom)  # (B, N, d)
            region_p = F.normalize(region_p + geom_feat, dim=-1)
            # recompute sim with spatial-enhanced regions
            sim2 = torch.einsum("bld,bnd->bln", enhanced_text, region_p)
            scores = (token_weights.unsqueeze(-1) * sim2).sum(dim=1)

        # Mask invalid proposals
        if proposal_mask is not None:
            scores = scores.masked_fill(~proposal_mask, float("-inf"))

        # Query: pooled enhanced text (weighted by token_weights)
        query = (token_weights.unsqueeze(-1) * enhanced_text).sum(dim=1)  # (B, d)

        return scores, token_weights, query

    def trainable_parameters(self):
        """Return only the parameters that require gradients (for the optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


# grounding_loss is imported from losses.py at the top of this file and
# re-exported so that any existing code importing it from head.py continues to work.