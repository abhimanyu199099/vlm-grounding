"""
eval/metrics.py — grounding evaluation metrics.

MEMBER C owns this file.

Primary metric : Acc@0.5  — fraction of phrases where predicted proposal
                            has IoU >= 0.5 with ground-truth box.
                            Standard benchmark number on Flickr30k Entities.

Secondary      : Acc@0.25 — same at a looser threshold (useful for ablations).
               : mean_iou — average IoU across all phrases.
               : acc_by_type — Acc@0.5 broken down by Flickr30k entity_type
                               (people, animals, vehicles, clothing, instruments,
                                scene, other).

Usage:
    evaluator = GroundingEvaluator(config)
    # inside eval loop:
    evaluator.update_from_indices(preds, proposals, gt_boxes, entity_types)
    results = evaluator.compute()
    evaluator.reset()

    # track CLIP baseline for delta reporting:
    evaluator.set_baseline(baseline_acc)
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch

from config import Config


# ---------------------------------------------------------------------------
# Standalone IoU helper (imported by other modules too)
# ---------------------------------------------------------------------------

def iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    Compute IoU between two axis-aligned boxes [x1, y1, x2, y2].

    Handles degenerate boxes (zero area) gracefully — returns 0.0.
    Both tensors must be 1-D with 4 elements.
    """
    xa = max(box_a[0].item(), box_b[0].item())
    ya = max(box_a[1].item(), box_b[1].item())
    xb = min(box_a[2].item(), box_b[2].item())
    yb = min(box_a[3].item(), box_b[3].item())

    inter    = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a   = max(0.0, float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])))
    area_b   = max(0.0, float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1])))
    union    = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GroundingEvaluator:
    """
    Stateful accumulator for grounding metrics.

    Call update_from_indices() (or update()) for each batch,
    then compute() at the end of the epoch, then reset() before the next.
    """

    def __init__(self, config: Config):
        self.cfg            = config
        self.threshold_high = config.eval.iou_threshold    # 0.5
        self.threshold_low  = 0.25                         # secondary threshold
        self._baseline_acc: Optional[float] = None
        self.reset()

    def reset(self):
        """Clear all accumulated state. Call before each eval epoch."""
        self._ious:         List[float] = []
        self._correct_50:   List[bool]  = []
        self._correct_25:   List[bool]  = []
        self._entity_types: List[str]   = []
        self._ap_25:        List[float] = []   # per-phrase AP at IoU≥0.25
        self._ap_50:        List[float] = []   # per-phrase AP at IoU≥0.50
        self._ap_75:        List[float] = []   # per-phrase AP at IoU≥0.75
        self._recall5:      List[bool]  = []   # any of top-5 correct at IoU≥0.5
        # Direct box prediction metrics (normalized cxcywh)
        self._direct_ious:      List[float] = []
        self._direct_correct_50: List[bool] = []

    def set_baseline(self, baseline_acc: float):
        """
        Store the vanilla-CLIP Acc@0.5 so compute() can report delta.
        Called once after the CLIP baseline run in train.py.
        """
        self._baseline_acc = baseline_acc

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update(self,
               pred_boxes:   torch.Tensor,   # (B, 4)
               gt_boxes:     torch.Tensor,   # (B, 4)
               entity_types: List[str],
               ):
        """
        Accumulate one batch of box-level predictions.

        Args:
            pred_boxes   : (B, 4) predicted [x1,y1,x2,y2] for each phrase
            gt_boxes     : (B, 4) ground-truth boxes
            entity_types : list[str] of length B
        """
        for i in range(pred_boxes.size(0)):
            score = iou(pred_boxes[i], gt_boxes[i])
            self._ious.append(score)
            self._correct_50.append(score >= self.threshold_high)
            self._correct_25.append(score >= self.threshold_low)
            self._entity_types.append(entity_types[i])

    def update_from_indices(self,
                            pred_idx:     torch.Tensor,            # (B,)
                            proposals:    torch.Tensor,            # (B, N, 4)
                            gt_boxes:     torch.Tensor,            # (B, 4)
                            entity_types: List[str],
                            scores:       Optional[torch.Tensor] = None,  # (B, N)
                            ):
        """
        Convert predicted proposal index → box, then call update().

        If scores is provided, also computes mAP50 and Recall@5 from the
        full ranked proposal list. Otherwise those metrics are omitted.
        """
        B         = pred_idx.size(0)
        pred_idx  = pred_idx.cpu()
        proposals = proposals.cpu()
        gt_boxes  = gt_boxes.cpu()

        pred_boxes = proposals[torch.arange(B), pred_idx]   # (B, 4)
        self.update(pred_boxes, gt_boxes, entity_types)

        if scores is not None:
            scores = scores.cpu()
            for i in range(B):
                ranked = scores[i].argsort(descending=True)
                gt     = gt_boxes[i]
                ap25 = ap50 = ap75 = 0.0
                r5   = False
                for rank, idx in enumerate(ranked):
                    iou_val = iou(proposals[i, idx], gt)
                    if ap25 == 0.0 and iou_val >= 0.25:
                        ap25 = 1.0 / (rank + 1)
                    if ap50 == 0.0 and iou_val >= 0.50:
                        ap50 = 1.0 / (rank + 1)
                        r5   = rank < 5
                    if ap75 == 0.0 and iou_val >= 0.75:
                        ap75 = 1.0 / (rank + 1)
                    if ap25 and ap50 and ap75:
                        break
                self._ap_25.append(ap25)
                self._ap_50.append(ap50)
                self._ap_75.append(ap75)
                self._recall5.append(r5)

    def update_direct_boxes(self,
                            pred_boxes_norm: torch.Tensor,   # (B, 4) normalized cxcywh
                            gt_boxes_norm:   torch.Tensor,   # (B, 4) normalized cxcywh
                            entity_types:    List[str],
                            ):
        """Accumulate metrics for the direct box-prediction head (normalized cxcywh)."""
        pred_boxes_norm = pred_boxes_norm.cpu()
        gt_boxes_norm   = gt_boxes_norm.cpu()

        # Convert cxcywh → xyxy for IoU computation
        def to_xyxy(b):
            cx, cy, w, h = b[0].item(), b[1].item(), b[2].item(), b[3].item()
            return torch.tensor([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        for i in range(pred_boxes_norm.size(0)):
            score = iou(to_xyxy(pred_boxes_norm[i]), to_xyxy(gt_boxes_norm[i]))
            self._direct_ious.append(score)
            self._direct_correct_50.append(score >= self.threshold_high)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self) -> Dict:
        """
        Compute and return all metrics over accumulated samples.

        Returns dict with:
            acc@0.5      : float
            acc@0.25     : float
            mean_iou     : float
            n_samples    : int
            acc_by_type  : dict[str, float]  — Acc@0.5 per entity type
            baseline_delta : float | None    — gain vs CLIP baseline if set
        """
        if not self._correct_50:
            return {}

        n            = len(self._correct_50)
        overall_50   = sum(self._correct_50) / n
        overall_25   = sum(self._correct_25) / n
        mean_iou_val = sum(self._ious) / n

        # Per-entity-type Acc@0.5
        type_buckets: Dict[str, List[bool]] = defaultdict(list)
        for correct, etype in zip(self._correct_50, self._entity_types):
            type_buckets[etype].append(correct)

        acc_by_type = {
            etype: round(sum(cs) / len(cs), 4)
            for etype, cs in sorted(type_buckets.items())
        }

        # Delta vs CLIP baseline
        delta = None
        if self._baseline_acc is not None:
            delta = round(overall_50 - self._baseline_acc, 4)

        out = {
            "acc@0.5":        round(overall_50,   4),
            "acc@0.25":       round(overall_25,   4),
            "mean_iou":       round(mean_iou_val, 4),
            "recall@1":       round(overall_50,   4),   # identical to acc@0.5 by definition
            "n_samples":      n,
            "acc_by_type":    acc_by_type,
            "baseline_delta": delta,
        }
        if self._ap_50:
            out["AP@0.25"]   = round(sum(self._ap_25)  / len(self._ap_25),  4)
            out["AP@0.50"]   = round(sum(self._ap_50)  / len(self._ap_50),  4)
            out["AP@0.75"]   = round(sum(self._ap_75)  / len(self._ap_75),  4)
            out["mAP"]       = round((out["AP@0.25"] + out["AP@0.50"] + out["AP@0.75"]) / 3, 4)
            out["recall@5"]  = round(sum(self._recall5) / len(self._recall5), 4)
        return out