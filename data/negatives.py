"""
data/negatives.py — automatic hard negative generation strategies.

MEMBER A owns this file.

Three strategies, all annotation-free:
  1. inbatch     : top-K highest-similarity proposals from *other* images in the batch.
  2. clip_mined  : top-K highest-similarity *wrong* proposals within the same image.
  3. cross_image : ground-truth proposals from same-entity-type images in the batch.
  4. all         : union of inbatch + clip_mined.

NegativeMiner.mine() returns neg_indices: (B, K) LongTensor.
Each row i holds K proposal indices (into that item's own proposal list) that
the loss should treat as negatives for phrase i.

The inbatch strategy is special: it returns indices into a *cross-image* flattened
proposal bank of shape (B*N,). The loss must handle this differently from the
per-image indices returned by clip_mined and cross_image — see mine() docstring.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from config import Config


class NegativeMiner:
    """
    Usage in train.py:
        miner = NegativeMiner(config, clip_model=model.encoder.clip)

        # After encoding, before the grounding head forward pass:
        phrase_embeds = model.encoder.encode_phrase(batch["phrase_tokens"], attn_mask)
        region_embeds = model.encoder.encode_region(batch["proposal_crops"])
        neg_indices, cross_image = miner.mine(batch, phrase_embeds, region_embeds)

        # Pass neg_indices and cross_image flag into grounding_loss().
    """

    def __init__(self, config: Config, clip_model=None):
        self.cfg      = config
        self.strategy = config.data.neg_strategy

    @torch.no_grad()
    def mine(self,
             batch:         dict,
             phrase_embeds: torch.Tensor,   # (B, D)  L2-normalised
             region_embeds: torch.Tensor,   # (B, N, D) L2-normalised
             ) -> tuple:
        """
        Returns:
            neg_indices  : (B, K) LongTensor
            cross_image  : bool — True when indices point into the flattened
                           cross-batch region bank (B*N,) rather than per-image (N,)

        The loss function must check cross_image to interpret the indices correctly:
          - cross_image=False  → indices are into proposals[i]  shape (N,)
          - cross_image=True   → indices are into proposals.view(B*N, 4), and
                                 the loss should gather them from the flattened bank
        """
        if self.strategy == "inbatch":
            return self._inbatch(batch, phrase_embeds, region_embeds), True
        elif self.strategy == "clip_mined":
            return self._clip_mined(phrase_embeds, region_embeds,
                                    batch["pos_idx"],
                                    batch.get("proposal_mask")), False
        elif self.strategy == "cross_image":
            return self._cross_image(batch, phrase_embeds, region_embeds), False
        elif self.strategy == "all":
            ib = self._inbatch(batch, phrase_embeds, region_embeds)   # cross-image indices
            cm = self._clip_mined(phrase_embeds, region_embeds,
                                  batch["pos_idx"],
                                  batch.get("proposal_mask"))          # per-image indices
            # Normalise cm to cross-image index space so both can be concatenated:
            # item i's per-image index j → flat index i*N + j
            B, N = region_embeds.shape[:2]
            offsets = torch.arange(B, device=cm.device).unsqueeze(1) * N
            cm_flat = cm + offsets                                     # (B, K)
            return torch.cat([ib, cm_flat], dim=1), True
        else:
            raise ValueError(f"Unknown neg_strategy: '{self.strategy}'. "
                             f"Choose from: inbatch, clip_mined, cross_image, all")

    # -----------------------------------------------------------------------
    # Strategy 1 — In-batch negatives
    # -----------------------------------------------------------------------

    def _inbatch(self,
                 batch:         dict,
                 phrase_embeds: torch.Tensor,   # (B, D)
                 region_embeds: torch.Tensor,   # (B, N, D)
                 ) -> torch.Tensor:
        """
        For each phrase i, find the K most similar proposals from all *other*
        images in the batch. These are hard because CLIP already considers them
        plausible matches, yet they belong to different images.

        Steps:
          1. Flatten all proposals to a (B*N, D) bank.
          2. Compute (B, B*N) cosine similarity matrix.
          3. Mask out the B*N positions that belong to phrase i's own image
             (indices [i*N .. i*N+N-1]) — they could be true positives.
          4. Also mask padding proposals via proposal_mask.
          5. Return top-K indices per row from the remaining positions.

        Returns: (B, K) — flat indices into the (B*N,) bank.
        """
        K = self.cfg.data.clip_mine_topk
        B, N, D = region_embeds.shape
        device   = region_embeds.device

        # (B*N, D) flat bank of all region embeddings
        flat_regions = region_embeds.view(B * N, D)               # (B*N, D)

        # (B, B*N) similarity — phrase_embeds already L2-normed, flat_regions too
        sim = torch.mm(phrase_embeds, flat_regions.t())           # (B, B*N)

        # Build mask: True = valid negative (from a different image, not padding)
        valid = torch.ones(B, B * N, dtype=torch.bool, device=device)

        # Mask own-image proposals
        for i in range(B):
            valid[i, i * N : i * N + N] = False

        # Mask padding proposals across the whole bank
        if "proposal_mask" in batch and batch["proposal_mask"] is not None:
            # proposal_mask: (B, N) Bool — True = real proposal
            flat_mask = batch["proposal_mask"].to(device).view(B * N)  # (B*N,)
            valid &= flat_mask.unsqueeze(0)                        # (B, B*N)

        # Set invalid positions to -inf before top-K
        sim = sim.masked_fill(~valid, float("-inf"))

        K = min(K, valid.sum(dim=1).min().item())
        K = max(K, 1)
        neg_indices = sim.topk(K, dim=1).indices                  # (B, K)
        return neg_indices

    # -----------------------------------------------------------------------
    # Strategy 2 — CLIP-mined hard negatives (within the same image)
    # -----------------------------------------------------------------------

    def _clip_mined(self,
                    phrase_embeds:  torch.Tensor,            # (B, D)
                    region_embeds:  torch.Tensor,            # (B, N, D)
                    pos_idx:        torch.Tensor,            # (B,)
                    proposal_mask:  Optional[torch.Tensor],  # (B, N) or None
                    ) -> torch.Tensor:
        """
        For each phrase, find the K highest-similarity *wrong* proposals
        within its own image. These are the hardest negatives for the grounding
        head because CLIP's frozen encoder already assigns them high scores.

        Steps:
          1. Compute (B, N) per-image similarity scores.
          2. Mask out pos_idx[i] for each row i (the ground-truth proposal).
          3. Mask out padding proposals.
          4. Return top-K indices per row.

        Returns: (B, K) — per-image proposal indices.
        """
        K      = self.cfg.data.clip_mine_topk
        B, N, D = region_embeds.shape
        device  = region_embeds.device

        # (B, N) cosine similarity — both inputs are already L2-normalised
        sim = torch.einsum("bd,bnd->bn", phrase_embeds, region_embeds)

        # Mask the ground-truth proposal for each item using advanced indexing
        batch_idx = torch.arange(B, device=device)
        sim[batch_idx, pos_idx] = float("-inf")

        # Mask padding proposals
        if proposal_mask is not None:
            sim = sim.masked_fill(~proposal_mask, float("-inf"))

        K = min(K, N - 1)
        K = max(K, 1)
        neg_indices = sim.topk(K, dim=1).indices                  # (B, K)
        return neg_indices

    # -----------------------------------------------------------------------
    # Strategy 3 — Cross-image entity-type swap
    # -----------------------------------------------------------------------

    def _cross_image(self,
                     batch:         dict,
                     phrase_embeds: torch.Tensor,   # (B, D)
                     region_embeds: torch.Tensor,   # (B, N, D)
                     ) -> torch.Tensor:
        """
        For each phrase i with entity_type t, find the K most similar
        ground-truth proposals from other images j in the batch that also
        have entity_type t.

        Rationale: "the man in the red jacket" should not match the bounding
        box of *another* man — even though both regions are of entity type
        'people'. Using same-type GT boxes as negatives exploits the entity
        annotations without any extra labelling effort.

        Steps:
          1. Group batch indices by entity_type.
          2. For each item i, identify candidates: same-type items j where j != i.
          3. If ≥1 candidate exists: score phrase i against each candidate's
             GT-proposal region embedding, return top-K by similarity.
          4. If no same-type candidate exists (rare in large batches):
             fall back to clip_mined within the same image.

        Returns: (B, K) — per-image proposal indices (into item i's own proposals).
        Note: The returned index for item i points to whichever of item i's own
        proposals most resembles the same-type GT region from another image.
        This keeps the index space consistent (per-image) with clip_mined.
        """
        K      = self.cfg.data.clip_mine_topk
        B, N, D = region_embeds.shape
        device  = region_embeds.device

        entity_types: List[str] = batch["entity_type"]   # list of B strings
        pos_idx: torch.Tensor   = batch["pos_idx"]        # (B,)

        # --- Group indices by entity type ---
        type_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(entity_types):
            type_to_indices[t].append(i)

        # --- For each item, collect GT region embeddings from same-type peers ---
        # gt_region[i] = region_embeds[i, pos_idx[i], :]  — the GT proposal embedding
        batch_idx  = torch.arange(B, device=device)
        gt_regions = region_embeds[batch_idx, pos_idx]    # (B, D) — GT embed per item

        neg_indices = torch.zeros(B, K, dtype=torch.long, device=device)

        for i, t in enumerate(entity_types):
            same_type = [j for j in type_to_indices[t] if j != i]

            if not same_type:
                # Fallback: clip_mined within this item's own image
                sim_i  = torch.einsum("d,nd->n", phrase_embeds[i], region_embeds[i])
                sim_i[pos_idx[i]] = float("-inf")
                if batch.get("proposal_mask") is not None:
                    sim_i = sim_i.masked_fill(~batch["proposal_mask"][i], float("-inf"))
                k_i = min(K, N - 1)
                neg_indices[i, :k_i] = sim_i.topk(k_i).indices
                continue

            # Score phrase i against GT-region embeddings of same-type peers
            # peer_gt: (P, D) where P = len(same_type)
            peer_gt  = gt_regions[same_type]                          # (P, D)
            peer_sim = torch.mv(peer_gt, phrase_embeds[i])            # (P,) dot products

            # Pick the K highest-scoring peers
            k_peers = min(K, len(same_type))
            top_peer_local = peer_sim.topk(k_peers).indices           # (k_peers,) into same_type
            top_peer_idx   = [same_type[p] for p in top_peer_local.tolist()]

            # Now find which of item i's *own* proposals most resembles
            # each peer's GT region. This keeps index space per-image.
            for slot, j in enumerate(top_peer_idx):
                peer_gt_embed = gt_regions[j]                          # (D,)
                # Similarity of peer GT against all of item i's proposals
                sim_ij = torch.mv(region_embeds[i], peer_gt_embed)    # (N,)
                # Exclude item i's own GT proposal
                sim_ij[pos_idx[i]] = float("-inf")
                if batch.get("proposal_mask") is not None:
                    sim_ij = sim_ij.masked_fill(~batch["proposal_mask"][i], float("-inf"))
                neg_indices[i, slot] = sim_ij.argmax()

            # If k_peers < K, fill remaining slots with clip_mined fallback
            if k_peers < K:
                sim_i = torch.einsum("d,nd->n", phrase_embeds[i], region_embeds[i])
                sim_i[pos_idx[i]] = float("-inf")
                if batch.get("proposal_mask") is not None:
                    sim_i = sim_i.masked_fill(~batch["proposal_mask"][i], float("-inf"))
                # Zero out slots already filled by cross-image to avoid re-selecting them
                for slot in range(k_peers):
                    sim_i[neg_indices[i, slot]] = float("-inf")
                remaining = min(K - k_peers, (sim_i > float("-inf")).sum().item())
                if remaining > 0:
                    neg_indices[i, k_peers:k_peers + remaining] = sim_i.topk(remaining).indices

        return neg_indices   # (B, K)