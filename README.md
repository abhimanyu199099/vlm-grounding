# VLM Grounding

Phrase grounding on Flickr30k Entities using a trainable cross-attention head on top of a frozen CLIP backbone.

Given an image and a natural-language phrase, the model predicts which region of the image the phrase refers to.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/abhimanyu199099/vlm-grounding.git
cd vlm_grounding
pip install -r requirements.txt
```

**2. Download Flickr30k Entities annotations**

Images are loaded automatically from HuggingFace. You only need to download the bounding box annotations:

```bash
git clone https://github.com/BryanPlummer/flickr30k_entities
```

Place the cloned repo at `flickr30k_entities/` in the project root. The XML bounding box annotations must be at `flickr30k_entities/Annotations/`.

---

## Usage

**Debug run** — 200 samples, fast smoke-test:

```bash
python train.py --debug
```

**Single-GPU training:**

```bash
python train.py --run_name my_run
```

**Multi-GPU training (DDP):**

```bash
torchrun --nproc_per_node=4 train.py --run_name my_run
```

**Training without precomputed cache:**

```bash
torchrun --nproc_per_node=4 train.py --run_name my_run --no_cache
```

**Pre-compute CLIP embeddings** (strongly recommended before training — skips the frozen CLIP forward pass each step):

```bash
python precompute.py                        # train + val splits (default)
python precompute.py --split val            # val only
```

**Evaluate a saved checkpoint:**

```bash
python evaluate.py --ckpt checkpoints/my_run/best.pt --split val
python evaluate.py --ckpt checkpoints/my_run/best.pt --split test --visualize 20
```

**Run ablations:**

```bash
python ablate.py --quick             # subset grid, faster overnight run
python ablate.py                     # full grid
python ablate.py --eval_refcoco      # + zero-shot RefCOCO+ testA/testB eval after each run
python ablate.py --dry_run           # preview commands without training
```

**Evaluate a single checkpoint zero-shot on RefCOCO+ testA/testB:**

```bash
python ablate.py --ckpt checkpoints/my_run/best.pt
```

**Interactive demo:**

```bash
python demo/app.py
```

---

## Key CLI arguments (`train.py`)

| Argument | Default | Description |
|---|---|---|
| `--run_name` | `baseline` | Experiment name; checkpoint saved to `checkpoints/<run_name>/` |
| `--debug` | off | Use 200 samples for a fast smoke-test |
| `--use_cache` / `--no_cache` | cache on | Toggle precomputed CLIP embedding loading |
| `--data_fraction` | `1.0` | Random subset of training data, e.g. `0.3` |
| `--skip_baseline` | off | Skip the frozen-CLIP baseline eval before training |
| `--lora_rank` | `8` | LoRA rank for the grounding head; `0` disables LoRA |
| `--head_depth` | `1` | Number of cross-attention layers in the grounding head |

Config defaults (batch size, learning rate, proposal method, etc.) live in `config.py`.

---

## How it works

CLIP (ViT-B/32) is kept fully frozen. A small grounding head (~1.9M parameters) is trained on top:

1. The phrase is encoded into per-token features by CLIP's text transformer
2. Each region proposal (image crop) is encoded into an embedding by CLIP's vision transformer
3. A cross-attention layer lets phrase tokens attend over region embeddings
4. A token-weighting MLP learns which words matter most and produces a weighted phrase embedding
5. The weighted phrase embedding is scored against each region; the highest-scoring region is the prediction

Training uses three losses: cross-entropy over proposals, a hard-negative contrastive loss (InfoNCE), and a token entropy regulariser that encourages the model to focus on content words rather than function words.

See [PROJECT.md](PROJECT.md) for a detailed architecture description.

---

## Results

Evaluated on Flickr30k Entities validation set, primary metric **Acc@0.5** (predicted region IoU ≥ 0.5 with ground truth).

| Model | Acc@0.5 |
|---|---|
| Frozen CLIP baseline | — |
| + Grounding head (this work) | — |

---

## Project structure

```
vlm_grounding/
├── config.py                  # All paths and hyperparameters
├── train.py                   # Training loop (single-GPU and 4-GPU DDP)
├── precompute.py              # Pre-compute and cache CLIP embeddings per split
├── evaluate.py                # Standalone checkpoint evaluation
├── ablate.py                  # Ablation grid search + RefCOCO+ zero-shot eval
├── models/
│   ├── encoder.py             # Frozen CLIP wrapper (text + vision)
│   ├── head.py                # Cross-attention grounding head
│   ├── losses.py              # Grounding, contrastive, and entropy losses
│   └── grounding_model.py     # Top-level model; owns loss + checkpointing
├── data/
│   ├── dataset.py             # Flickr30k Entities dataset + proposal generation
│   ├── refcoco.py             # RefCOCO+ dataset for zero-shot evaluation
│   └── negatives.py           # Online hard-negative mining
├── eval/
│   ├── metrics.py             # Acc@0.5, mean IoU, per-entity-type breakdown
│   └── visualize.py           # Prediction visualisation
├── demo/
│   ├── app.py                 # Gradio web demo
│   └── inference.py           # Single-image inference helper
├── flickr30k_entities/        # Cloned annotation repo (not committed)
├── cache/                     # Cached proposals + CLIP embeddings (not committed)
└── checkpoints/               # Saved model weights (not committed)
```

---

## Requirements

- Python 3.8+
- PyTorch 2.4+
- `transformers`, `datasets`, `torchvision`, `Pillow`, `tqdm`
- `opencv-contrib-python` for selective search proposals (optional — falls back to multi-scale grid)
- `gradio` for the demo
