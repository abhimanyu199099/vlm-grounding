"""
config.py — single source of truth for all paths, hyperparameters, and flags.
All other modules import from here. Change values here, not scattered across files.

Data setup
----------
This project uses TWO separate sources that must be combined:

1. Images + captions + splits — HuggingFace dataset:
       nlphuji/flickr30k   (https://huggingface.co/datasets/nlphuji/flickr30k)
   Loaded via datasets.load_dataset() — no manual download needed.
   Provides: image (PIL), caption (list[str]), split (str), img_id (str), filename (str)

2. Phrase-level bounding box annotations — Flickr30k Entities:
       https://github.com/BryanPlummer/flickr30k_entities
   Manual download required (the repo provides XML files, one per image).
   Place the extracted Annotations/ folder at:
       data/flickr30k_entities/Annotations/<image_id>.xml

   To download:
       git clone https://github.com/BryanPlummer/flickr30k_entities
       cp -r flickr30k_entities/annotations data/flickr30k_entities/Annotations

   These XML files provide: phrase text, entity type, and bounding boxes.
   The split membership (train/val/test) is read from the HuggingFace dataset,
   not from the Entities repo, since they use identical image IDs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).parent

# HuggingFace dataset identifier (no local path needed — loaded via datasets lib)
HF_DATASET_NAME = "nlphuji/flickr30k"

# Flickr30k Entities XML annotations (manual download — see docstring above)
ENTITIES_ANNO_DIR = ROOT / "data" / "flickr30k_entities" / "Annotations" / "Annotations"

# Cache: pre-computed proposals and region embeddings (large, gitignored)
CACHE_DIR = ROOT / "cache"

CKPT_DIR  = ROOT / "checkpoints"
RUNS_DIR  = ROOT / "runs"   # tensorboard / wandb logs

for d in [CACHE_DIR, CKPT_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    clip_model:      str   = "openai/clip-vit-base-patch32"  # HuggingFace model id
    freeze_clip:     bool  = True                             # always True for direction 1
    embed_dim:       int   = 512                              # CLIP ViT-B/32 output dim
    num_heads:       int   = 8                                # cross-attention heads
    head_depth:      int   = 1                                # number of cross-attn layers
    lora_rank:       int   = 8                                # LoRA rank; 0 = disable LoRA
    lora_alpha:      float = 16.0
    dropout:         float = 0.1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    max_proposals:      int   = 100    # max region proposals per image
    proposal_method:    Literal["selective_search", "grid"] = "selective_search"
    neg_strategy:       Literal["inbatch", "clip_mined", "cross_image", "all"] = "inbatch"
    clip_mine_topk:     int   = 5      # how many hard negatives to mine per phrase via CLIP
    cross_image_pool:   int   = 50     # images to sample cross-image negatives from
    image_size:         int   = 224
    num_workers:        int   = 4
    pin_memory:         bool  = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size:     int   = 32
    epochs:         int   = 20
    lr:             float = 1e-4
    weight_decay:   float = 1e-2
    warmup_steps:   int   = 500
    grad_clip:      float = 1.0
    log_every:      int   = 50         # steps
    eval_every:     int   = 1          # epochs
    save_every:     int   = 5          # epochs
    device:         str   = "cuda"     # "cpu" for debugging
    mixed_precision: bool = True
    seed:           int   = 42


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    iou_threshold:  float = 0.5        # Acc@0.5 threshold
    max_preds:      int   = 1          # top-1 prediction per phrase
    split:          str   = "val"      # "val" or "test"


# ---------------------------------------------------------------------------
# Top-level config (what you pass around)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)
    run_name: str      = "baseline"
    debug:    bool     = False         # small dataset, fast iterations


# Convenience: one default config instance
DEFAULT_CONFIG = Config()