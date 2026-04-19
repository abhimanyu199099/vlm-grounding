"""
data/refcoco.py — RefCOCO+ dataset for zero-shot evaluation.

Loads from HuggingFace: lmms-lab/RefCOCO+
Schema expected per row:
  image       : PIL.Image
  image_id    : int
  ann_id      : int
  split       : str   — "train" | "val" | "testA" | "testB"
  bbox        : [x, y, w, h]  (COCO format, pixel coords)
  sentences   : list of {"sent": str, ...}

Each sample = one (image, sentence, gt_box) triple.
Multiple sentences per annotation are flattened to separate samples.

Returns the same keys as Flickr30kGroundingDataset so collate_fn works unchanged.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import Config, CACHE_DIR
from data.dataset import get_proposals, _find_best_proposal, _crop_proposals

HF_REFCOCO_PLUS = "lmms-lab/RefCOCO+"


class RefCOCOPlusDataset(Dataset):
    """
    One sample = one (image, referring expression, gt_box) triple.

    Compatible with collate_fn from data/dataset.py.
    """

    def __init__(self, config: Config, split: str = "testA",
                 tokenizer=None):
        assert split in ("val", "testA", "testB"), \
            f"split must be val/testA/testB, got {split!r}"

        self.cfg       = config
        self.split     = split
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        print(f"[refcoco+] Loading '{HF_REFCOCO_PLUS}' split='{split}' ...")
        from datasets import load_dataset
        hf = load_dataset(HF_REFCOCO_PLUS, split="test")
        hf = hf.filter(lambda r: r["split"] == split)

        # Store dataset object for lazy image access; index by position
        self._hf_ds  = hf
        self._hf_idx: Dict[int, int] = {i: i for i in range(len(hf))}

        # Flatten: one sample per (annotation, sentence) pair
        self.samples: List[Tuple[int, str, list]] = []  # (row_idx, sent, bbox)
        for i, row in enumerate(hf):
            bbox = row["bbox"]          # [x, y, w, h]
            # Convert COCO bbox [x,y,w,h] → [x1,y1,x2,y2]
            box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            for sent_dict in row["sentences"]:
                sent = sent_dict["sent"] if isinstance(sent_dict, dict) else sent_dict
                self.samples.append((i, sent, box))

        print(f"[refcoco+] {split}: {len(self.samples)} samples ({len(hf)} annotations)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row_idx, sentence, gt_box_list = self.samples[idx]
        row    = self._hf_ds[row_idx]
        image  = row["image"].convert("RGB")
        img_id = str(row["image_id"])
        method = self.cfg.data.proposal_method
        gt_box = torch.tensor(gt_box_list, dtype=torch.float32)

        proposals = get_proposals(image, f"rc+_{img_id}", method,
                                  self.cfg.data.max_proposals)
        pos_idx        = _find_best_proposal(proposals, gt_box)
        proposal_crops = _crop_proposals(image, proposals, self.transform)

        phrase_tokens = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids.squeeze(0)

        return {
            "image_id":       img_id,
            "phrase":         sentence,
            "phrase_tokens":  phrase_tokens,
            "proposals":      proposals,
            "proposal_crops": proposal_crops,
            "pos_idx":        pos_idx,
            "gt_box":         gt_box,
            "entity_type":    "refcoco+",
        }
