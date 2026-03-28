"""
models/encoder.py — frozen CLIP encoder exposing both global and patch-level features.

MEMBER B owns this file.

Key design decisions:
  - All CLIP weights are frozen (requires_grad = False)
  - Visual encoder returns patch-level features (B, N_patches+1, D_vision) so the
    grounding head could attend over spatial patches if extended in future.
    For region encoding we use only the CLS token (index 0) per crop.
  - Text encoder returns per-token features (B, L, D_text) so the grounding head
    can attend over individual phrase tokens, not just a pooled embedding.
  - D_vision and D_text are read from the CLIP config at runtime — never hardcoded.
  - LoRA adapters are injected into the grounding head, NOT into this encoder.

CLIP ViT-B/32 dimensions (for reference):
  vision_config.hidden_size  = 768   (ViT patch transformer hidden dim)
  text_config.hidden_size    = 512   (text transformer hidden dim)
  projection_dim             = 512   (shared embedding space D)
  visual_projection          : Linear(768, 512, bias=False)  — broadcasts over seq dim
  text_projection            : Linear(512, 512, bias=False)  — EOS-token only in CLIP's
                                own forward(); we do NOT use it for token-level features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from config import Config


class FrozenCLIPEncoder(nn.Module):
    """
    Wraps HuggingFace CLIPModel with all weights frozen.

    Public attributes set after __init__:
        vision_hidden_dim : int  — hidden_size of the ViT (768 for ViT-B/32)
        text_hidden_dim   : int  — hidden_size of the text transformer (512)
        projection_dim    : int  — shared embedding space (512)

    These are exposed so GroundingHead can build correctly-sized projections
    without hardcoding numbers.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config

        self.clip      = CLIPModel.from_pretrained(config.model.clip_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.model.clip_model)

        # Freeze all CLIP parameters — only the grounding head is trained
        for param in self.clip.parameters():
            param.requires_grad = False

        # Expose dims so downstream modules don't hardcode them
        self.vision_hidden_dim = self.clip.config.vision_config.hidden_size
        self.text_hidden_dim   = self.clip.config.text_config.hidden_size
        self.projection_dim    = self.clip.config.projection_dim

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen ViT on a batch of images and return projected patch features.

        Args:
            pixel_values: (B, 3, H, W)

        Returns:
            features: (B, N_patches+1, projection_dim)
                      Index 0 is the CLS token; indices 1.. are spatial patches.

        Implementation note:
            vision_model.last_hidden_state is (B, N+1, vision_hidden_dim).
            visual_projection is Linear(vision_hidden_dim, projection_dim) and
            broadcasts correctly over the sequence dimension, so we can apply it
            to the full sequence in one call.
        """
        hidden   = self.clip.vision_model(pixel_values=pixel_values).last_hidden_state
        # (B, N+1, vision_hidden_dim) → (B, N+1, projection_dim)
        features = self.clip.visual_projection(hidden)
        return features

    @torch.no_grad()
    def encode_region(self, proposal_crops: torch.Tensor) -> torch.Tensor:
        """
        Encode a padded batch of cropped region proposals.

        Args:
            proposal_crops: (B, N, 3, H, W)

        Returns:
            region_embeds: (B, N, projection_dim) — CLS embedding per crop, L2-normalised

        We use only the CLS token because each crop is already spatially isolated —
        the CLS token aggregates the full crop's content. The patch tokens would
        add compute with minimal benefit at this granularity.
        """
        B, N, C, H, W = proposal_crops.shape

        # Merge batch and proposal dims so we can run one forward pass
        flat     = proposal_crops.view(B * N, C, H, W)
        features = self.encode_image(flat)          # (B*N, N_patches+1, D_proj)
        cls      = features[:, 0, :]               # (B*N, D_proj) — CLS token only
        cls      = cls.view(B, N, -1)              # (B, N, D_proj)
        return F.normalize(cls, dim=-1)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Return token-level features from the frozen text transformer.

        Args:
            input_ids:      (B, L)  — output of CLIPTokenizer, padded to L=77
            attention_mask: (B, L)  — 1 for real tokens, 0 for padding

        Returns:
            token_hidden: (B, L, text_hidden_dim)
                          Raw last_hidden_state — NOT passed through text_projection.
                          CLIP's text_projection is designed to map only the EOS token
                          into the shared embedding space; applying it token-wise would
                          be incorrect. The grounding head's text_proj learns its own
                          mapping from text_hidden_dim → projection_dim.
        """
        return self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state   # (B, L, text_hidden_dim)

    @torch.no_grad()
    def encode_phrase(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Return a single L2-normalised phrase embedding via the EOS token.

        Used by NegativeMiner for fast similarity-based negative mining.
        Mirrors what CLIP's own get_text_features() does internally.

        Returns: (B, projection_dim)
        """
        hidden  = self.encode_text(input_ids, attention_mask)  # (B, L, text_hidden_dim)

        # EOS token sits at the last non-padding position
        eos_pos        = attention_mask.sum(dim=1) - 1          # (B,)
        eos_hidden     = hidden[torch.arange(hidden.size(0)), eos_pos]  # (B, text_hidden_dim)

        # Apply text_projection to reach the shared embedding space — correct here
        # because we want a single pooled vector comparable to region_embeds
        phrase_embeds  = self.clip.text_projection(eos_hidden)  # (B, projection_dim)
        return F.normalize(phrase_embeds, dim=-1)