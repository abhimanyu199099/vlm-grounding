"""
data/dataset.py — Flickr30k grounding dataset.

MEMBER A owns this file.

Data sources
------------
Images + splits : HuggingFace  nlphuji/flickr30k
                  Loaded with datasets.load_dataset() — no manual image download.
                  Each HF row: {image (PIL), caption (list[str]), split (str),
                                img_id (str), filename (str), sentids (list[str])}
                  NOTE: the HF dataset has a single HF split called "test" containing
                  all 31k images. The train/val/test membership is stored in the
                  row-level "split" field — filter on that, not the HF split name.

Phrase boxes    : Flickr30k Entities XMLs in ENTITIES_ANNO_DIR.
                  Each XML is named <img_id>.xml and provides phrase text,
                  entity type, and bounding boxes.
                  Download: git clone https://github.com/BryanPlummer/flickr30k_entities
                  then: cp -r flickr30k_entities/annotations data/flickr30k_entities/Annotations

Joining strategy
----------------
  1. Load HF dataset, filter rows to the requested split.
  2. For each row, parse ENTITIES_ANNO_DIR/<img_id>.xml → phrase dicts.
  3. Flatten to one sample per (image, phrase) pair that has ≥1 bounding box.

Tensor contract (what models/ and train.py expect per batch key):
  phrase_tokens   : LongTensor  (B, 77)
  proposal_crops  : FloatTensor (B, N, 3, 224, 224)
  proposals       : FloatTensor (B, N, 4)   [x1,y1,x2,y2] pixel coords
  proposal_mask   : BoolTensor  (B, N)      True = valid (not padding)
  pos_idx         : LongTensor  (B,)        index of GT proposal (highest IoU)
  gt_box          : FloatTensor (B, 4)
  entity_type     : list[str]   length B
  image_id        : list[str]   length B
  phrase          : list[str]   length B
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import Config, HF_DATASET_NAME, ENTITIES_ANNO_DIR, CACHE_DIR


# ---------------------------------------------------------------------------
# Entities XML parser
# ---------------------------------------------------------------------------

def parse_entities_xml(xml_path: Path) -> List[dict]:
    """
    Parse one Flickr30k Entities image annotation + sentence file pair.

    Actual format on disk:
      Annotations/<img_id>.xml  — bounding boxes, objects keyed by <name> (entity ID)
      Sentences/<img_id>.txt    — one sentence per line with inline phrase markup:
                                  [/EN#<id>/<type> phrase text]

    Returns list of phrase dicts:
        {"phrase": str, "phrase_id": str, "entity_type": str,
         "boxes": [[x1,y1,x2,y2], ...]}
    Only phrases that have at least one bounding box are returned.
    """
    import re

    if not xml_path.exists():
        return []

    # --- Pass 1: entity_id → boxes from XML ---
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes_by_id: Dict[str, List[List[int]]] = {}
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            x1 = int(bndbox.find("xmin").text)
            y1 = int(bndbox.find("ymin").text)
            x2 = int(bndbox.find("xmax").text)
            y2 = int(bndbox.find("ymax").text)
        except (AttributeError, ValueError, TypeError):
            continue
        for name_el in obj.findall("name"):
            eid = name_el.text.strip()
            boxes_by_id.setdefault(eid, []).append([x1, y1, x2, y2])

    # --- Pass 2: parse phrase text + type from the matching .txt sentences file ---
    sentences_path = xml_path.parent.parent / "Sentences" / xml_path.name.replace(".xml", ".txt")
    if not sentences_path.exists():
        return []

    # Pattern: [/EN#<id>/<type> phrase text]
    phrase_re = re.compile(r'\[/EN#(\d+)/(\w+)\s+([^\]]+)\]')

    seen: Dict[str, dict] = {}   # phrase_id → dict (deduplicate across sentences)
    for line in sentences_path.read_text(encoding="utf-8").splitlines():
        for m in phrase_re.finditer(line):
            pid, ptype, text = m.group(1), m.group(2), m.group(3).strip()
            if pid in boxes_by_id and pid not in seen:
                seen[pid] = {
                    "phrase":      text,
                    "phrase_id":   pid,
                    "entity_type": ptype,
                    "boxes":       boxes_by_id[pid],
                }

    return list(seen.values())


# ---------------------------------------------------------------------------
# Proposal generation / caching
# ---------------------------------------------------------------------------

def get_proposals(image: Image.Image,
                  image_id: str,
                  method: str = "selective_search",
                  max_proposals: int = 100) -> torch.Tensor:
    """
    Generate or load cached region proposals for an image.

    Returns: (N, 4) FloatTensor [x1, y1, x2, y2] pixel coords.

    Caches results to CACHE_DIR/<image_id>_<method>.pt.

    selective_search requires: pip install opencv-contrib-python
    grid fallback requires nothing extra — useful for smoke-tests.
    """
    cache_path = CACHE_DIR / f"{image_id}_{method}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    W, H = image.size
    proposals = []

    if method == "selective_search":
        try:
            import cv2
            import numpy as np
            img_np = np.array(image)[:, :, ::-1]   # RGB → BGR for cv2
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(img_np)
            ss.switchToSelectiveSearchFast()
            for (x, y, w, h) in ss.process():
                if w < 10 or h < 10:
                    continue
                x1 = max(0, x);       y1 = max(0, y)
                x2 = min(W, x + w);   y2 = min(H, y + h)
                proposals.append([float(x1), float(y1), float(x2), float(y2)])
                if len(proposals) >= max_proposals:
                    break
        except ImportError:
            method = "grid"   # fall back silently

    if method == "grid":
        # Multi-scale sliding windows: scales × strides give proposals that can
        # achieve IoU ≥ 0.5 with GT boxes of varied sizes.
        for scale in [0.4, 0.6, 0.8, 1.0]:
            bw, bh = W * scale, H * scale
            stride_x = max(bw * 0.5, 1)
            stride_y = max(bh * 0.5, 1)
            x = 0.0
            while x + bw <= W + stride_x:
                y = 0.0
                while y + bh <= H + stride_y:
                    x1 = max(0.0, x);       y1 = max(0.0, y)
                    x2 = min(W, x + bw);    y2 = min(H, y + bh)
                    proposals.append([x1, y1, x2, y2])
                    if len(proposals) >= max_proposals:
                        break
                    y += stride_y
                if len(proposals) >= max_proposals:
                    break
                x += stride_x
            if len(proposals) >= max_proposals:
                break

    result = torch.tensor(proposals, dtype=torch.float32)
    torch.save(result, cache_path)
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Flickr30kGroundingDataset(Dataset):
    """
    One sample = one (image, phrase, gt_box) grounding triple.

    PIL images are served directly from the HuggingFace dataset object,
    so no manual image download or local image directory is needed.
    """

    def __init__(self, config: Config, split: str = "train",
                 tokenizer=None, debug: bool = False):
        self.cfg       = config
        self.split     = split
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        # --- Load HuggingFace dataset and filter to requested split ---
        print(f"[dataset] Loading '{HF_DATASET_NAME}' (row-level split='{split}') ...")
        from datasets import load_dataset

        # nlphuji/flickr30k has a loading script (flickr30k.py) on its main
        # branch. datasets >= 4.0 refuses to run loading scripts and
        # trust_remote_code is no longer accepted as an override.
        # HuggingFace auto-converts script-based datasets to parquet stored
        # on the refs/convert/parquet branch. Loading from that revision
        # bypasses the script entirely and works with all datasets versions.
        # Schema is identical: image, caption, split, img_id, filename, sentids.
        hf_full = load_dataset(
            HF_DATASET_NAME,
            split="test",
            revision="refs/convert/parquet",
        )
        hf = hf_full.filter(lambda r: r["split"] == split)

        # Store the dataset object and a flickr_id → row-index map for O(1)
        # lazy access. Keeping full rows (with image bytes) in a dict would
        # exhaust CPU RAM when 4+ DDP workers each hold a copy.
        # img_id in the parquet revision is a sequential integer ('0','1',...),
        # NOT the original Flickr image ID. The XML files are named after the
        # Flickr ID (e.g. 1000092795.xml), which is the filename stem.
        # We use filename (e.g. "1000092795.jpg") → strip ".jpg" → flickr_id.
        self._hf_ds = hf
        self._hf_rows: Dict[str, int] = {
            r["filename"].replace(".jpg", ""): i for i, r in enumerate(hf)
        }

        # --- Join with Entities annotations and flatten to samples ---
        self.samples: List[Tuple[str, dict]] = []
        missing = 0
        for flickr_id in self._hf_rows:
            xml_path = ENTITIES_ANNO_DIR / f"{flickr_id}.xml"
            phrases  = parse_entities_xml(xml_path)
            if not phrases:
                missing += 1
                continue
            for pd in phrases:
                self.samples.append((flickr_id, pd))

        if missing:
            print(f"[dataset] Warning: {missing}/{len(self._hf_rows)} images "
                  f"have no Entities XML. Check ENTITIES_ANNO_DIR:\n"
                  f"  {ENTITIES_ANNO_DIR}")

        if debug:
            self.samples = self.samples[:200]
        elif config.data.data_fraction < 1.0:
            k = int(len(self.samples) * config.data.data_fraction)
            self.samples = random.sample(self.samples, k)

        print(f"[dataset] {split}: {len(self.samples)} samples "
              f"({len(self._hf_rows)} images) | use_cache={self.cfg.data.use_cache}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_id, phrase_dict = self.samples[idx]
        hf_row  = self._hf_ds[self._hf_rows[img_id]]
        method  = self.cfg.data.proposal_method
        gt_box  = torch.tensor(phrase_dict["boxes"][0], dtype=torch.float32)

        # ---- Try to load pre-computed CLIP embeddings ----
        region_cache_path = CACHE_DIR / f"{img_id}_{method}_clip_regions.pt"
        phrase_cache_path = CACHE_DIR / f"{img_id}_clip_phrases.pt"
        use_cache = (self.cfg.data.use_cache
                     and region_cache_path.exists()
                     and phrase_cache_path.exists())

        if use_cache:
            phrase_cache = torch.load(phrase_cache_path, weights_only=True)
            pid = phrase_dict["phrase_id"]

            if pid in phrase_cache:
                region_embeds = torch.load(region_cache_path, weights_only=True).float()  # (N, D)
                text_hidden   = phrase_cache[pid]["text_hidden"].float()    # (77, D_text)
                phrase_embed  = phrase_cache[pid]["phrase_embed"].float()   # (D_proj)

                # Still need proposals for IoU / box decoding; load from proposal cache
                image     = hf_row["image"].convert("RGB")
                proposals = get_proposals(image, img_id, method,
                                          self.cfg.data.max_proposals)
                pos_idx   = _find_best_proposal(proposals, gt_box)

                # phrase_tokens + attention_mask needed for grounding_model
                encoding = self.tokenizer(
                    phrase_dict["phrase"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                )
                phrase_tokens = encoding.input_ids.squeeze(0)
                phrase_attn_mask = encoding.attention_mask.squeeze(0).bool()

                return {
                    "image_id":        img_id,
                    "phrase":          phrase_dict["phrase"],
                    "phrase_tokens":   phrase_tokens,
                    "phrase_attn_mask": phrase_attn_mask,
                    "proposals":       proposals,
                    "pos_idx":         pos_idx,
                    "gt_box":          gt_box,
                    "entity_type":     phrase_dict["entity_type"],
                    # pre-computed embeddings — no proposal_crops
                    "text_hidden":     text_hidden,
                    "region_embeds":   region_embeds,
                    "phrase_embed":    phrase_embed,
                }

        # ---- Fallback: compute on the fly ----
        image  = hf_row["image"].convert("RGB")
        proposals = get_proposals(image, img_id, method,
                                  self.cfg.data.max_proposals)
        pos_idx        = _find_best_proposal(proposals, gt_box)
        proposal_crops = _crop_proposals(image, proposals, self.transform)

        encoding = self.tokenizer(
            phrase_dict["phrase"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        phrase_tokens    = encoding.input_ids.squeeze(0)
        phrase_attn_mask = encoding.attention_mask.squeeze(0).bool()

        return {
            "image_id":         img_id,
            "phrase":           phrase_dict["phrase"],
            "phrase_tokens":    phrase_tokens,
            "phrase_attn_mask": phrase_attn_mask,
            "proposals":        proposals,
            "proposal_crops":   proposal_crops,
            "pos_idx":          pos_idx,
            "gt_box":           gt_box,
            "entity_type":      phrase_dict["entity_type"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    xa = max(box_a[0].item(), box_b[0].item())
    ya = max(box_a[1].item(), box_b[1].item())
    xb = min(box_a[2].item(), box_b[2].item())
    yb = min(box_a[3].item(), box_b[3].item())
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = float((box_a[2]-box_a[0]) * (box_a[3]-box_a[1]))
    area_b = float((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_best_proposal(proposals: torch.Tensor,
                        gt_box: torch.Tensor) -> int:
    ious = [_iou(proposals[i], gt_box) for i in range(len(proposals))]
    return int(torch.tensor(ious).argmax())


def _crop_proposals(image: Image.Image,
                    proposals: torch.Tensor,
                    transform) -> torch.Tensor:
    crops = []
    for box in proposals:
        x1, y1, x2, y2 = box.tolist()
        x2 = max(x2, x1 + 4)   # guard against degenerate crops
        y2 = max(y2, y1 + 4)
        crops.append(transform(image.crop((x1, y1, x2, y2))))
    return torch.stack(crops)   # (N, 3, H, W)


def collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length proposal lists to the per-batch maximum N.
    Adds proposal_mask: (B, N_max) BoolTensor — True = valid, False = padding.

    Handles two modes transparently:
      Cached   : items have "text_hidden", "region_embeds", "phrase_embed" —
                 no "proposal_crops". Embeddings are padded along the N dim.
      Uncached : items have "proposal_crops" — full crops for the encoder.
    """
    max_n     = max(item["proposals"].shape[0] for item in batch)
    use_cache = "text_hidden" in batch[0]

    padded_proposals = []
    masks            = []
    padded_crops     = []          # only populated in uncached mode
    padded_regions   = []          # only populated in cached mode

    for item in batch:
        n   = item["proposals"].shape[0]
        pad = max_n - n
        padded_proposals.append(F.pad(item["proposals"], (0, 0, 0, pad)))
        m = torch.zeros(max_n, dtype=torch.bool)
        m[:n] = True
        masks.append(m)

        if use_cache:
            # region_embeds: (N, D) → pad to (max_n, D)
            padded_regions.append(F.pad(item["region_embeds"], (0, 0, 0, pad)))
        else:
            padded_crops.append(F.pad(item["proposal_crops"], (0, 0, 0, 0, 0, 0, 0, pad)))

    out = {
        "image_id":      [item["image_id"]    for item in batch],
        "phrase":        [item["phrase"]       for item in batch],
        "entity_type":   [item["entity_type"]  for item in batch],
        "phrase_tokens":    torch.stack([item["phrase_tokens"]    for item in batch]),
        "phrase_attn_mask": torch.stack([item["phrase_attn_mask"] for item in batch]),
        "proposals":     torch.stack(padded_proposals),
        "proposal_mask": torch.stack(masks),
        "pos_idx":       torch.tensor([item["pos_idx"] for item in batch], dtype=torch.long),
        "gt_box":        torch.stack([item["gt_box"]   for item in batch]),
    }

    if use_cache:
        out["text_hidden"]   = torch.stack([item["text_hidden"]   for item in batch])
        out["region_embeds"] = torch.stack(padded_regions)
        out["phrase_embed"]  = torch.stack([item["phrase_embed"]  for item in batch])
    else:
        out["proposal_crops"] = torch.stack(padded_crops)

    return out