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
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).parent

# HuggingFace dataset identifier (no local path needed — loaded via datasets lib)
HF_DATASET_NAME = "nlphuji/flickr30k"

# Flickr30k Entities XML annotations (manual download — see docstring above)
ENTITIES_ANNO_DIR = ROOT / "flickr30k_entities" / "annotations" /"Annotations"

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
    proj_dim:        int   = 256                              # shared text/region projection dim
    num_heads:       int   = 8                                # cross-attention heads (unused by new head)
    head_depth:      int   = 1                                # cross-attn layers (unused by new head)
    lora_rank:       int   = 8                                # LoRA rank; 0 = disable LoRA
    lora_alpha:      float = 32.0
    dropout:         float = 0.1

    # Hard-negative contrastive loss
    hard_neg_k:              int   = 4     # top-k wrong regions to use as hard negatives
    hard_neg_penalty:        float = 1.5  # logit scale for hard negatives (>1 = harder)
    contrastive_temperature: float = 0.07 # InfoNCE temperature
    contrastive_loss_weight: float = 0.0  # disabled for isolation test

    # Hinge entropy loss
    entropy_loss_weight: float = 0.0   # disabled for isolation test
    entropy_target:      float = 2.2   # minimum acceptable token entropy before penalising


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    max_proposals:      int   = 50     # max region proposals per image
    proposal_method:    Literal["selective_search", "grid"] = "grid"
    neg_strategy:       Literal["inbatch", "clip_mined", "cross_image", "all"] = "inbatch"
    clip_mine_topk:     int   = 5      # how many hard negatives to mine per phrase via CLIP
    cross_image_pool:   int   = 50     # images to sample cross-image negatives from
    image_size:         int   = 224
    num_workers:        int   = 1
    pin_memory:         bool  = True
    use_cache:          bool  = True  # load precomputed CLIP embeddings from CACHE_DIR if available
    data_fraction:      float = 1.0   # random subset of training data (1.0 = full dataset)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size:     int   = 16
    epochs:         int   = 10
    lr:             float = 5e-4
    weight_decay:   float = 1e-2
    warmup_steps:   int   = 500
    grad_clip:      float = 1.0
    log_every:      int   = 10         # steps
    eval_every:     int   = 1          # epochs
    save_every:     int   = 1          # epochs
    device:         str   = "cuda"     # "cpu" for debugging
    mixed_precision: bool = True
    seed:           int   = 42


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    iou_threshold:  float = 0.25        # Acc@0.25 threshold
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
    run_name:      str        = "baseline"
    debug:         bool       = False        # small dataset, fast iterations
    skip_baseline: bool       = False        # skip CLIP baseline eval before training
    resume:        Optional[str] = None      # path to checkpoint to resume from


# Convenience: one default config instance
DEFAULT_CONFIG = Config()