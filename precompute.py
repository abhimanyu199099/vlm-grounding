"""
precompute.py — pre-compute and cache CLIP embeddings for all dataset splits.

Single GPU:
    python precompute.py
    python precompute.py --split val
    python precompute.py --split train val test

Multi-GPU (one process per GPU, work sharded automatically):
    CUDA_VISIBLE_DEVICES=0 python precompute.py --shard 0 4 &
    CUDA_VISIBLE_DEVICES=1 python precompute.py --shard 1 4 &
    CUDA_VISIBLE_DEVICES=2 python precompute.py --shard 2 4 &
    CUDA_VISIBLE_DEVICES=3 python precompute.py --shard 3 4 &
    wait

--shard <rank> <world>  — this process handles images[rank::world]

Saves two files per image to cache/:
    {image_id}_{method}_clip_regions.pt
        Tensor(N, D_proj) float16 — L2-normalised region embeddings

    {image_id}_clip_phrases.pt
        dict[phrase_id -> {"text_hidden":  Tensor(77, D_text) float16,
                           "phrase_embed": Tensor(D_proj) float16}]

Training detects these files and skips the encoder forward pass entirely.
"""

import argparse
from collections import defaultdict

import torch
from transformers import CLIPTokenizer
from tqdm import tqdm

from config import DEFAULT_CONFIG, CACHE_DIR
from data import Flickr30kGroundingDataset
from models.encoder import FrozenCLIPEncoder


# ---------------------------------------------------------------------------
# Per-image region embedding cache
# ---------------------------------------------------------------------------

def precompute_region_embeds(encoder: FrozenCLIPEncoder,
                             dataset: Flickr30kGroundingDataset,
                             method: str,
                             device: torch.device,
                             batch_size: int = 16,
                             shard_rank: int = 0,
                             shard_world: int = 1):
    """
    Encode region proposals for every unique image and save to cache.

    Images are processed in batches of `batch_size` for GPU efficiency.
    With shard_rank/shard_world, each process handles a disjoint subset.
    """
    # Collect unique images: one representative sample index per image
    seen = {}
    for idx in range(len(dataset)):
        img_id, _ = dataset.samples[idx]
        if img_id not in seen:
            seen[img_id] = idx

    all_images = list(seen.items())                          # [(img_id, idx), ...]
    my_images  = all_images[shard_rank::shard_world]        # this shard's slice

    to_process = [
        (img_id, idx) for img_id, idx in my_images
        if not (CACHE_DIR / f"{img_id}_{method}_clip_regions.pt").exists()
    ]
    already_done = len(my_images) - len(to_process)
    if already_done:
        print(f"  [regions] {already_done} already cached, {len(to_process)} to compute")
    if not to_process:
        return

    encoder.eval()

    for start in tqdm(range(0, len(to_process), batch_size), desc="  region embeds"):
        batch_items = to_process[start : start + batch_size]

        # Load proposal crops for this mini-batch of images
        # Each image may have a different N; pad to max_n within the mini-batch
        samples      = [dataset[idx] for _, idx in batch_items]
        max_n        = max(s["proposal_crops"].shape[0] for s in samples)
        batch_crops  = []
        actual_ns    = []
        for s in samples:
            crops = s["proposal_crops"]   # (N, 3, H, W)
            n     = crops.shape[0]
            actual_ns.append(n)
            if n < max_n:
                pad   = torch.zeros(max_n - n, *crops.shape[1:])
                crops = torch.cat([crops, pad], dim=0)
            batch_crops.append(crops)

        crops_gpu = torch.stack(batch_crops).to(device)   # (B, max_n, 3, H, W)
        embeds    = encoder.encode_region(crops_gpu)       # (B, max_n, D)

        for i, (img_id, _) in enumerate(batch_items):
            n      = actual_ns[i]
            embed  = embeds[i, :n].cpu().half()            # (N, D) float16
            path   = CACHE_DIR / f"{img_id}_{method}_clip_regions.pt"
            torch.save(embed, path)


# ---------------------------------------------------------------------------
# Per-image phrase embedding cache
# ---------------------------------------------------------------------------

def precompute_phrase_embeds(encoder: FrozenCLIPEncoder,
                             dataset: Flickr30kGroundingDataset,
                             device: torch.device,
                             batch_size: int = 128,
                             shard_rank: int = 0,
                             shard_world: int = 1):
    """
    Encode all phrases for every unique image and save in one dict per image.

    Phrases are encoded in large batches across the whole split for efficiency.
    """
    # Group samples by image_id
    image_phrases = defaultdict(list)
    for idx in range(len(dataset)):
        img_id, phrase_dict = dataset.samples[idx]
        image_phrases[img_id].append((phrase_dict["phrase_id"], idx))

    all_images = list(image_phrases.keys())
    my_images  = set(all_images[shard_rank::shard_world])

    to_process   = {
        img_id: pl for img_id, pl in image_phrases.items()
        if img_id in my_images
        and not (CACHE_DIR / f"{img_id}_clip_phrases.pt").exists()
    }
    already_done = len(my_images) - len(to_process)
    if already_done:
        print(f"  [phrases] {already_done} images already cached, {len(to_process)} to compute")
    if not to_process:
        return

    encoder.eval()
    tokenizer = encoder.tokenizer

    # Flatten to a list of (img_id, phrase_id, phrase_text) for batch encoding
    flat = []
    for img_id, phrase_list in to_process.items():
        for phrase_id, sample_idx in phrase_list:
            _, phrase_dict = dataset.samples[sample_idx]
            flat.append((img_id, phrase_id, phrase_dict["phrase"]))

    # Encode in batches, accumulate results keyed by (img_id, phrase_id)
    results = {}

    for start in tqdm(range(0, len(flat), batch_size), desc="  phrase embeds"):
        batch_items = flat[start : start + batch_size]
        texts       = [item[2] for item in batch_items]

        enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=77)
        input_ids = enc.input_ids.to(device)
        attn_mask = (input_ids != 0).to(device)

        text_hidden   = encoder.encode_text(input_ids, attn_mask).cpu().half()    # (B, 77, D_text)
        phrase_embeds = encoder.encode_phrase(input_ids, attn_mask).cpu().half()  # (B, D_proj)

        for i, (img_id, phrase_id, _) in enumerate(batch_items):
            results[(img_id, phrase_id)] = {
                "text_hidden":  text_hidden[i],
                "phrase_embed": phrase_embeds[i],
            }

    # Write one file per image
    image_results = defaultdict(dict)
    for (img_id, phrase_id), data in results.items():
        image_results[img_id][phrase_id] = data

    for img_id, phrase_data in image_results.items():
        torch.save(dict(phrase_data), CACHE_DIR / f"{img_id}_clip_phrases.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--region_batch_size", type=int, default=16,
                        help="Images per batch for region encoding")
    parser.add_argument("--phrase_batch_size", type=int, default=128,
                        help="Phrases per batch for text encoding")
    parser.add_argument("--shard", nargs=2, type=int, default=[0, 1],
                        metavar=("RANK", "WORLD"),
                        help="Shard index and total shards for multi-GPU use")
    args = parser.parse_args()

    shard_rank, shard_world = args.shard

    cfg    = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  shard {shard_rank}/{shard_world}")

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.clip_model)
    encoder   = FrozenCLIPEncoder(cfg).to(device)
    encoder.eval()

    method = cfg.data.proposal_method

    for split in args.split:
        print(f"\n=== {split} ===")
        cfg.data.use_cache = False   # always load raw crops; we are building the cache
        dataset = Flickr30kGroundingDataset(cfg, split=split, tokenizer=tokenizer)
        precompute_region_embeds(encoder, dataset, method, device,
                                 batch_size=args.region_batch_size,
                                 shard_rank=shard_rank, shard_world=shard_world)
        precompute_phrase_embeds(encoder, dataset, device,
                                 batch_size=args.phrase_batch_size,
                                 shard_rank=shard_rank, shard_world=shard_world)

    print(f"\nShard {shard_rank}/{shard_world} done.")


if __name__ == "__main__":
    main()
