"""
models/rpn_encoder.py — ResNet-18 FPN backbone + RPN for region proposal generation.

Replaces the CLIP-crop pipeline in uncached mode. The RPN proposes regions,
RoI-aligns their features, and projects them into projection_dim space for
the grounding head.

All backbone + RPN weights are frozen by default (frozen=True).
"""

import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.image_list import ImageList


class RPNRegionEncoder(nn.Module):
    """
    Extracts region features using a ResNet-18 FPN backbone + Faster-RCNN RPN.

    forward() returns:
        region_feats : (B, max_proposals, out_dim)  — projected region features
        boxes        : (B, max_proposals, 4)         — xyxy pixel-coord proposals
        mask         : (B, max_proposals) bool        — True = valid (not padding)
    """

    def __init__(self, out_dim: int = 512, max_proposals: int = 64, frozen: bool = True):
        super().__init__()
        self.max_proposals = max_proposals

        backbone = resnet_fpn_backbone('resnet18', weights='DEFAULT', trainable_layers=0)

        anchor_generator = AnchorGenerator(
            sizes=((16, 32), (32, 64), (64, 128), (128, 224), (224,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        self.anchor_generator = anchor_generator

        self.detector = FasterRCNN(
            backbone,
            num_classes=2,
            min_size=224,
            max_size=224,
            rpn_anchor_generator=anchor_generator,
        )

        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2,
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, out_dim),
            nn.LayerNorm(out_dim),
        )

        if frozen:
            for p in self.detector.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor, image_sizes):
        """
        Args:
            images      : (B, 3, 224, 224) ImageNet-normalised
            image_sizes : list of B (H, W) tuples

        Returns:
            region_feats : (B, max_proposals, out_dim)
            boxes        : (B, max_proposals, 4)  xyxy pixel coords
            mask         : (B, max_proposals) bool
        """
        image_list = ImageList(images, image_sizes)
        features   = self.detector.backbone(images)

        # RPN returns a list of B proposal tensors, each (N_i, 4)
        proposals, _ = self.detector.rpn(image_list, features)
        proposals    = [p[:self.max_proposals] for p in proposals]

        # RoI align: (sum_N, 256, 7, 7)
        roi_feats = self.roi_align(features, proposals, image_sizes)
        roi_feats = self.proj(roi_feats)   # (sum_N, out_dim)

        return self._pad(roi_feats, proposals, images.device)

    def _pad(self, feats, proposals, device):
        B   = len(proposals)
        N   = self.max_proposals
        D   = feats.shape[-1]
        out  = torch.zeros(B, N, D, device=device)
        boxes = torch.zeros(B, N, 4, device=device)
        mask  = torch.zeros(B, N, dtype=torch.bool, device=device)

        offset = 0
        for i, props in enumerate(proposals):
            n = props.shape[0]
            out[i,   :n] = feats[offset:offset + n]
            boxes[i, :n] = props.to(device)
            mask[i,  :n] = True
            offset += n

        return out, boxes, mask
