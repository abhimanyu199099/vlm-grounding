"""
models/box_encoding.py — positional encoding for bounding boxes.
"""

import torch
import torch.nn as nn


class BoxPositionalEncoding(nn.Module):
    """
    Maps normalized cxcywh boxes → d_model-dimensional positional embeddings.

    Args:
        d_model : output embedding dimension (should match head embed_dim)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes : (B, N, 4) normalized cxcywh, values in [0, 1]
        Returns:
            (B, N, d_model)
        """
        return self.mlp(boxes)


def xyxy_pixel_to_cxcywh_norm(boxes: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Convert bounding boxes from xyxy pixel coordinates to normalized cxcywh.

    Args:
        boxes      : (..., 4) in xyxy pixel format
        image_size : side length of the (square) image in pixels

    Returns:
        (..., 4) normalized cxcywh, values clamped to [0, 1]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    out = torch.stack([cx, cy, w, h], dim=-1) / image_size
    return out.clamp(0.0, 1.0)
