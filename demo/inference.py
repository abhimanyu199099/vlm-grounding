"""
demo/inference.py — single-image inference for the demo.

Given a PIL Image and a text query, returns a predicted bounding box
and a confidence score. Called by demo/app.py.

Design notes:
  - get_proposals() requires an image_id for caching. For demo images
    (uploaded at runtime, no stable ID) we use a content hash of the
    image bytes so repeated calls on the same image hit the cache.
  - Padding proposals have score=-inf; softmax over them would produce
    incorrect probabilities. We mask them out before softmax.
  - The batch dict contains a dummy pos_idx=0 (required by the model
    signature) which is ignored at inference time since we don't compute
    loss in eval mode.
"""

import hashlib
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer

from config import DEFAULT_CONFIG
from data.dataset import get_proposals, _crop_proposals
from models import GroundingModel


class Grounder:
    """
    Stateful inference object — load once, call .predict() repeatedly.

    Usage:
        grounder = Grounder(ckpt_path="checkpoints/baseline/best.pt")
        box, conf = grounder.predict(image, "the woman in a red dress")
    """

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.cfg    = DEFAULT_CONFIG
        self.device = torch.device(device)

        self.model = GroundingModel(self.cfg).to(self.device)
        self.model.load(Path(ckpt_path))
        self.model.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.model.clip_model)
        self.transform  = transforms.Compose([
            transforms.Resize((self.cfg.data.image_size, self.cfg.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self,
                image: Image.Image,
                query: str,
                ) -> Tuple[list, float]:
        """
        Ground a text query to a bounding box in the image.

        Args:
            image : PIL Image (any size, any mode — converted to RGB internally)
            query : referring expression, e.g. "the woman in a red dress"

        Returns:
            box  : [x1, y1, x2, y2] in pixel coordinates (floats)
            conf : softmax probability of the predicted box (0–1)
        """
        image = image.convert("RGB")

        # Stable cache key derived from image content — avoids re-running
        # selective search for the same image across multiple queries
        img_hash = _image_hash(image)

        # --- Generate proposals ---
        proposals = get_proposals(
            image,
            image_id=img_hash,
            method=self.cfg.data.proposal_method,
            max_proposals=self.cfg.data.max_proposals,
        )   # (N, 4)

        N = proposals.size(0)

        # --- Crop and transform proposals ---
        proposal_crops = _crop_proposals(image, proposals, self.transform)  # (N, 3, H, W)

        # --- Tokenize query ---
        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids   # (1, 77)

        # --- Build single-item batch ---
        # proposal_mask is all-True (no padding in single-image inference)
        batch = {
            "phrase_tokens":  tokens.to(self.device),                          # (1, 77)
            "proposal_crops": proposal_crops.unsqueeze(0).to(self.device),     # (1, N, 3, H, W)
            "proposals":      proposals.unsqueeze(0).to(self.device),          # (1, N, 4)
            "proposal_mask":  torch.ones(1, N, dtype=torch.bool).to(self.device), # (1, N)
            "pos_idx":        torch.zeros(1, dtype=torch.long).to(self.device),                # dummy
        }

        # --- Forward (no neg_mining at inference) ---
        out    = self.model(batch, neg_mining=None)
        scores = out["scores"][0]   # (N,) — padding already -inf via proposal_mask

        # Softmax only over valid (finite) positions
        finite_mask = scores.isfinite()
        probs       = torch.full_like(scores, 0.0)
        if finite_mask.any():
            probs[finite_mask] = F.softmax(scores[finite_mask], dim=0)

        pred = scores.argmax().item()
        box  = proposals[pred].tolist()
        conf = probs[pred].item()

        return box, conf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _image_hash(image: Image.Image) -> str:
    """Return a short MD5 hex string of the raw image bytes — used as cache key."""
    return hashlib.md5(image.tobytes()).hexdigest()[:12]