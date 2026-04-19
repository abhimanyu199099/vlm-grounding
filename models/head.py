"""
models/head.py — trainable grounding head on top of frozen CLIP features.

MEMBER B owns this file.

Architecture:
  Input:
    text_hidden  : (B, L, D_text)   — token-level features from encode_text()
    region_embeds: (B, N, D_proj)   — CLS features from encode_region()
    text_mask    : (B, L) bool      — True for real tokens
    proposal_mask: (B, N) bool      — True for valid (non-padding) proposals
    proposals    : (B, N, 4)        — boxes (x1,y1,x2,y2) normalised [0,1], optional

  Processing:
    1. Project text_hidden → (B, L, d) and region_embeds → (B, N, d), d=proj_dim.
       Add spatial encoding (cx,cy,w,h MLP) to region projections if proposals given.
       L2-normalise both.
    2. Compute token–region similarity: sim = einsum("bld,bnd->bln") → (B, L, N).
    3. Lightweight cross-attention: each text token attends over regions,
       producing context vectors that are added back to text (enhanced_text).
    4. Recompute sim with L2-normalised enhanced_text.
    5. Token weights: Linear(d→1)(enhanced_text) → masked softmax → (B, L).
    6. Aggregate: weighted sum of sim over tokens → scores (B, N).
    7. Apply proposal_mask, scale by learned temperature.

  Returns: (scores, token_weights, query)
    scores        : (B, N)  — un-normalised logits per proposal (padding → -inf)
    token_weights : (B, L)  — per-token importance weights (sum=1 over real tokens)
    query         : (B, D)  — weighted enhanced text embedding (for contrastive loss)

  Loss functions live in models/losses.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .losses import grounding_loss  # re-exported for backwards compatibility


# ---------------------------------------------------------------------------
# Spatial feature encoder
# ---------------------------------------------------------------------------

def _encode_spatial(proposals: torch.Tensor) -> torch.Tensor:
    """Convert (x1,y1,x2,y2) pixel boxes to normalised (cx,cy,w,h) in [0,1].

    Proposals arrive in original image pixel coordinates (variable image sizes).
    We normalise per-image by the observed coordinate range so the MLP always
    receives values in [0, 1] regardless of image resolution.
    """
    # per-image scale: max x2 and max y2 give a safe upper bound
    scale_x = proposals[..., 2].amax(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
    scale_y = proposals[..., 3].amax(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)

    cx = ((proposals[..., 0] + proposals[..., 2]) / 2) / scale_x
    cy = ((proposals[..., 1] + proposals[..., 3]) / 2) / scale_y
    w  =  (proposals[..., 2] - proposals[..., 0])      / scale_x
    h  =  (proposals[..., 3] - proposals[..., 1])      / scale_y
    return torch.stack([cx, cy, w, h], dim=-1)   # (B, N, 4)


# ---------------------------------------------------------------------------
# Full grounding head
# ---------------------------------------------------------------------------

class GroundingHead(nn.Module):
    """
    Token–region interaction grounding head.

    Each word token interacts with every region candidate, then tokens vote on
    regions weighted by learned importance. Spatial features (cx,cy,w,h) let the
    model learn positional relations ("under" → lower regions, etc.).
    """

    def __init__(self, config: Config,
                 text_hidden_dim: int,
                 region_proj_dim: int):
        super().__init__()
        cfg = config.model
        d   = cfg.proj_dim   # shared projection dim (256)

        self.text_proj   = nn.Linear(text_hidden_dim, d)
        self.region_proj = nn.Linear(region_proj_dim, d)

        # Spatial MLP: (cx, cy, w, h) → d
        self.spatial_mlp = nn.Sequential(
            nn.Linear(4, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Token scorer: one scalar per token, computed from enhanced features
        self.token_scorer = nn.Linear(d, 1, bias=False)

        # Learned temperature (CLIP convention init)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temperature=1.0 at init; learned up

        self._d = d

    def forward(self,
                text_hidden:   torch.Tensor,                    # (B, L, D_text)
                region_embeds: torch.Tensor,                    # (B, N, D_proj)
                text_mask:     Optional[torch.Tensor] = None,   # (B, L) bool
                proposal_mask: Optional[torch.Tensor] = None,   # (B, N) bool
                proposals:     Optional[torch.Tensor] = None,   # (B, N, 4) x1y1x2y2
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = text_hidden.shape
        N        = region_embeds.size(1)
        d        = self._d
        device   = text_hidden.device

        if text_mask is None:
            text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        if proposal_mask is None:
            proposal_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # ---- 1. Project + optional spatial + normalise ----
        # Promote to fp32 for normalize — fp16 can underflow to 0 norm, causing NaN.
        text    = F.normalize(self.text_proj(text_hidden).float(), dim=-1).to(text_hidden.dtype)
        regions = self.region_proj(region_embeds)                               # (B, N, d)

        if proposals is not None:
            spatial = self.spatial_mlp(_encode_spatial(proposals))    # (B, N, d)
            regions = regions + spatial

        regions = F.normalize(regions.float(), dim=-1).to(region_embeds.dtype) # (B, N, d)

        # ---- 2. Token–region similarity ----
        sim = torch.einsum("bld,bnd->bln", text, regions)             # (B, L, N)

        # ---- 3. Lightweight cross-attention: text attends over regions ----
        # Mask padding regions before softmax
        attn = sim / math.sqrt(d)                                      # (B, L, N)
        bad_regions = ~proposal_mask.unsqueeze(1)                      # (B, 1, N)
        attn = attn.masked_fill(bad_regions, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)                        # guard all-masked rows

        context  = torch.einsum("bln,bnd->bld", attn, regions)        # (B, L, d)
        residual = text + context
        enhanced = F.normalize(residual.float(), dim=-1).to(text.dtype)  # (B, L, d)
        enhanced = torch.nan_to_num(enhanced, nan=0.0)                 # guard zero-norm residuals

        # ---- 4. Recompute sim with enhanced text ----
        sim = torch.einsum("bld,bnd->bln", enhanced, regions)         # (B, L, N)

        # ---- 5. Token weights from enhanced text ----
        token_logits  = self.token_scorer(enhanced).squeeze(-1)       # (B, L)
        token_logits  = token_logits.masked_fill(~text_mask, float("-inf"))
        token_weights = F.softmax(token_logits, dim=-1)               # (B, L)
        token_weights = torch.nan_to_num(token_weights, nan=0.0)      # guard all-masked softmax

        # ---- 6. Aggregate over tokens ----
        scores = (token_weights.unsqueeze(-1) * sim).sum(dim=1)       # (B, N)

        # ---- 7. Mask + temperature ----
        scores = scores.masked_fill(~proposal_mask, float("-inf"))
        # Clamp log_temp to [-4, 4] → temperature in [0.018, 55] — prevents explosion
        temp   = self.log_temp.clamp(-4.0, 4.0).exp()
        scores = scores * temp

        # query: weighted sum of enhanced text features (for contrastive loss)
        query = (token_weights.unsqueeze(-1) * enhanced).sum(dim=1)   # (B, d)

        return scores, token_weights, query

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# grounding_loss re-exported so existing imports from head.py continue to work.
