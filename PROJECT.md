# Visual Grounding with Frozen CLIP

## What it does

This project trains a lightweight grounding head on top of a frozen CLIP model to solve **phrase grounding**: given an image and a natural-language phrase (e.g. *"a man in a blue shirt"*), predict which region of the image it refers to.

At inference time the model:
1. Encodes the phrase into per-token features using CLIP's text transformer
2. Encodes a set of region proposals (image crops) using CLIP's vision transformer
3. Runs cross-attention between the phrase tokens and the region features
4. Scores each region and returns the highest-scoring one as the prediction

The predicted region is evaluated by checking whether it overlaps the ground-truth bounding box at IoU ≥ 0.5 (Acc@0.5), the standard benchmark metric on Flickr30k Entities.

---

## Architecture

```
Image → region proposals → CLIP ViT → region embeddings (B, N, D)
                                                      ↓
Phrase → CLIP text transformer → token features (B, L, D)
                                                      ↓
                          TextOverRegionAttention (cross-attn)
                                                      ↓
                          TokenWeightingMLP → weighted phrase pool
                                                      ↓
                          scorer linear → dot product with regions
                                                      ↓
                                      scores (B, N) → argmax → predicted region
```

**Key components:**

| Component | File | Role |
|---|---|---|
| `FrozenCLIPEncoder` | `models/encoder.py` | Frozen ViT-B/32; exposes token-level text features and region CLS embeddings |
| `TextOverRegionAttention` | `models/head.py` | Multi-head cross-attention; text tokens (Q) attend over region features (K, V) |
| `TokenWeightingMLP` | `models/head.py` | Learns a scalar importance weight per token; replaces mean-pooling with a learned weighted sum |
| `GroundingHead` | `models/head.py` | Combines the above into a full scoring pipeline |
| `GroundingModel` | `models/grounding_model.py` | Top-level module; wires encoder + head, owns loss computation and checkpointing |

LoRA adapters are injected into the grounding head (not the frozen CLIP encoder). LoRA rank is configurable; setting `lora_rank=0` disables LoRA and uses standard linear layers.

---

## Training

CLIP is fully frozen. Only the grounding head (~1.9M parameters out of ~153M total) is trained with AdamW + cosine LR schedule.

Three losses are combined:

1. **Grounding loss** — cross-entropy over proposal scores, with optional hard-negative proposals mixed in
2. **Contrastive loss** — InfoNCE loss that explicitly penalises the top-k wrong-but-high-scoring regions per phrase
3. **Token entropy loss** — entropy minimisation on token weights, encouraging the model to focus on semantically important words (*"blue"*, *"shirt"*) rather than function words (*"a"*, *"the"*)

Hard negatives are mined online each step using `NegativeMiner` (`data/negatives.py`), which finds high-CLIP-similarity wrong regions either within the batch (`inbatch`), via CLIP similarity mining (`clip_mined`), or across images (`cross_image`).

**Multi-GPU training** uses PyTorch DDP (`torchrun --nproc_per_node=4`). Each GPU handles a shard of the training data via `DistributedSampler`. Evaluation and checkpointing run only on rank 0.

**Caching** — since CLIP is frozen, region and phrase embeddings are identical every epoch. `precompute.py` pre-computes and saves them to `cache/` as float16 tensors. At training time the dataset loads these directly, skipping the CLIP forward pass entirely. Use `--no_cache` to disable.

---

## Dataset

**Flickr30k Entities** — 31,783 images with 244,035 phrase-to-bounding-box annotations across seven entity types: *people, animals, vehicles, clothing, instruments, scene, other*.

- Images and splits are loaded from HuggingFace (`nlphuji/flickr30k`)
- Bounding box annotations come from the Flickr30k Entities repo (XML files, one per image)
- Region proposals are generated with selective search or a multi-scale grid (configurable via `proposal_method` in `config.py`)
- Proposal boxes are cached to `cache/` after first generation

**RefCOCO+** — used for zero-shot transfer evaluation only (no training). Loaded from HuggingFace (`lmms-lab/RefCOCO+`). Evaluated on `testA` (people-heavy) and `testB` (object-heavy) splits via `ablate.py --eval_refcoco` or `ablate.py --ckpt <path>`.

---

## Evaluation

Primary metric: **Acc@0.5** — the fraction of phrases where the predicted region has IoU ≥ 0.5 with the ground-truth box.

Additional metrics reported per run:
- Mean IoU across all phrases
- Acc@0.5 broken down by entity type
- Delta vs. frozen CLIP cosine-similarity baseline (run once before training starts)

For RefCOCO+ zero-shot transfer, Acc@0.5 is reported separately for testA and testB.

---

## Running

```bash
# Quick smoke-test (200 samples)
python train.py --debug

# Full single-GPU training
python train.py --run_name my_run

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --run_name my_run

# Pre-compute embeddings before training (recommended)
python precompute.py

# Evaluate a checkpoint
python evaluate.py --ckpt checkpoints/my_run/best.pt --split val

# Zero-shot eval on RefCOCO+ testA/testB
python ablate.py --ckpt checkpoints/my_run/best.pt

# Ablation grid
python ablate.py --quick --eval_refcoco
```

Key config knobs in `config.py`: `head_depth`, `lora_rank`, `num_heads`, `batch_size`, `lr`, `proposal_method`, `max_proposals`, `neg_strategy`.
