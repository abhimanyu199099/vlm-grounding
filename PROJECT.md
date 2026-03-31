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
| `GroundingModel` | `models/grounding_model.py` | Top-level module; wires encoder + head, owns the loss computation and checkpointing |

---

## Training

CLIP is fully frozen. Only the grounding head (~1.9M parameters out of ~153M total) is trained with AdamW.

Three losses are combined:

1. **Grounding loss** — cross-entropy over proposal scores, with optional hard-negative proposals mixed in
2. **Contrastive loss** — InfoNCE loss that explicitly penalises the top-k wrong-but-high-scoring regions per phrase (hard negatives)
3. **Token entropy loss** — entropy minimisation on the token weights, encouraging the model to focus on semantically important words (*"blue"*, *"shirt"*) rather than spreading weight over function words (*"a"*, *"the"*)

Hard negatives are mined online each step using `NegativeMiner` (`data/negatives.py`), which finds high-CLIP-similarity wrong regions either within the batch or across images.

---

## Dataset

**Flickr30k Entities** — 31,783 images with 244,035 phrase-to-bounding-box annotations across seven entity types: *people, animals, vehicles, clothing, instruments, scene, other*.

- Images and splits are loaded from HuggingFace (`nlphuji/flickr30k`)
- Bounding box annotations come from the Flickr30k Entities repo (XML + sentence files)
- Region proposals are generated with selective search (or a multi-scale grid fallback)

---

## Evaluation

Primary metric: **Acc@0.5** — the fraction of phrases where the predicted region has IoU ≥ 0.5 with the ground-truth box.

Additional metrics reported per run:
- Acc@0.25 (looser threshold, useful for ablations)
- Mean IoU across all phrases
- Acc@0.5 broken down by entity type
- Delta vs. the frozen CLIP cosine-similarity baseline (run before training starts)

---

## Running

```bash
# Quick smoke-test (200 samples, all epochs)
python train.py --debug

# Full training run
python train.py --run_name my_run

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/my_run/best.pt
```

Config defaults live in `config.py`. Key knobs: `head_depth`, `num_heads`, `embed_dim`, `batch_size`, `lr`.
