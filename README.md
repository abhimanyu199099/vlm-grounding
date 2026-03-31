# VLM Grounding

Phrase grounding on Flickr30k Entities using a trainable cross-attention head on top of a frozen CLIP backbone.

Given an image and a natural-language phrase, the model predicts which region of the image the phrase refers to.

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd vlm_grounding
pip install -r requirements.txt
```

**2. Download Flickr30k Entities annotations**

Images are loaded automatically from HuggingFace. You only need to download the bounding box annotations:

```bash
git clone https://github.com/BryanPlummer/flickr30k_entities
```

The cloned repo must be placed (or already exist) at `flickr30k_entities/` in the project root. The relevant files inside are `annotations/Annotations/` (XML bounding boxes), `annotations/Sentences/` (phrase text), and `train.txt` / `val.txt` / `test.txt` (split lists).

---

## Usage

**Debug run** — 200 samples, fast smoke-test:

```bash
python train.py --debug
```

**Full training run:**

```bash
python train.py --run_name my_run
```

**Evaluate a saved checkpoint:**

```bash
python evaluate.py --checkpoint checkpoints/my_run/best.pt
```

**Interactive demo:**

```bash
python demo/app.py
```

Config defaults (batch size, learning rate, model depth, etc.) are in `config.py`.

---

## How it works

CLIP (ViT-B/32) is kept fully frozen. A small grounding head (~1.9M parameters) is trained on top:

1. The phrase is encoded into per-token features by CLIP's text transformer
2. Each region proposal (image crop) is encoded into an embedding by CLIP's vision transformer
3. A cross-attention layer lets phrase tokens attend over region embeddings
4. A token-weighting MLP learns which words matter most and produces a weighted phrase embedding
5. The phrase embedding is scored against each region; the highest-scoring region is the prediction

Training uses three losses: cross-entropy over proposals, a hard-negative contrastive loss (InfoNCE), and a token entropy regulariser that encourages the model to focus on content words.

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
├── train.py                   # Training loop
├── evaluate.py                # Standalone evaluation
├── models/
│   ├── encoder.py             # Frozen CLIP wrapper
│   ├── head.py                # Cross-attention grounding head
│   ├── losses.py              # Grounding, contrastive, and entropy losses
│   └── grounding_model.py     # Top-level model
├── data/
│   ├── dataset.py             # Flickr30k Entities dataset + proposal generation
│   └── negatives.py           # Online hard negative mining
├── eval/
│   ├── metrics.py             # Acc@0.5, mean IoU, per-type breakdown
│   └── visualize.py           # Prediction visualisation
├── demo/
│   ├── app.py                 # Gradio web demo
│   └── inference.py           # Single-image inference helper
├── flickr30k_entities/        # Cloned annotation repo (not committed)
├── cache/                     # Cached region proposals (not committed)
└── checkpoints/               # Saved model weights (not committed)
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `transformers`, `datasets`, `torchvision`, `Pillow`
- `opencv-contrib-python` for selective search proposals (optional — falls back to a multi-scale grid)
- `gradio` for the demo
