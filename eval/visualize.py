"""
eval/visualize.py — draw grounding results on images for qualitative analysis.

MEMBER C owns this file.

Used for:
  - Paper qualitative figures (make_results_grid)
  - Per-sample debug images written during evaluate.py
  - Demo overlay (draw_grounding_result accepts a PIL image directly)

draw_grounding_result() accepts EITHER:
  - image_id (str) + the HuggingFace dataset object to look up the image, OR
  - image (PIL.Image) directly — used by the demo and evaluate.py

make_results_grid() tiles a list of annotated images into a grid suitable
for inclusion as a paper figure.
"""

import math
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PRED_COLOUR  = "#22c55e"   # green-500
_GT_COLOUR    = "#3b82f6"   # blue-500
_TOPK_COLOUR  = "#d1d5db"   # gray-300
_BANNER_BG    = (245, 245, 245)
_BANNER_FG    = (30, 30, 30)
_BOX_WIDTH    = 3
_TOPK_WIDTH   = 1
_BANNER_H     = 28
_FONT_SIZE    = 13


def _try_font(size: int) -> ImageFont.ImageFont:
    """Load a truetype font if available, fall back to the PIL default."""
    for face in ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(face, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# draw_grounding_result
# ---------------------------------------------------------------------------

def draw_grounding_result(
    phrase:      str,
    pred_box:    torch.Tensor,                           # (4,) [x1, y1, x2, y2]
    image:       Optional[Image.Image] = None,           # PIL image (preferred)
    image_id:    Optional[str]         = None,           # HF img_id (fallback)
    hf_dataset   = None,                                 # HF dataset object for lookup
    gt_box:      Optional[torch.Tensor] = None,          # (4,)
    top_k_boxes: Optional[torch.Tensor] = None,          # (K, 4)
    iou_score:   Optional[float]        = None,
    save_path:   Optional[Path]         = None,
) -> Image.Image:
    """
    Annotate an image with predicted box, ground-truth box, and top-K proposals.

    Image source priority:
        1. `image` argument (PIL Image) — used by demo and evaluate.py
        2. `hf_dataset` + `image_id` — looks up the row and reads .image field
        3. raises ValueError if neither is provided

    Drawing order (back → front):
        1. Top-K proposals in light gray  (background context)
        2. GT box in blue
        3. Predicted box in green with label

    Args:
        phrase      : referring expression text (shown in banner)
        pred_box    : (4,) predicted [x1,y1,x2,y2]
        image       : PIL Image, RGB
        image_id    : Flickr30k img_id string (used with hf_dataset)
        hf_dataset  : HuggingFace dataset object with .filter() support
        gt_box      : (4,) ground-truth box (optional — omit for demo)
        top_k_boxes : (K, 4) additional proposals to draw lightly (optional)
        iou_score   : IoU to display next to predicted box label
        save_path   : if provided, save the annotated image here

    Returns:
        Annotated PIL Image (RGB) with a phrase banner at the top.
    """
    # --- Resolve image ---
    if image is not None:
        img = image.convert("RGB").copy()
    elif image_id is not None and hf_dataset is not None:
        rows = hf_dataset.filter(lambda r: r["img_id"] == image_id)
        if len(rows) == 0:
            raise ValueError(f"image_id '{image_id}' not found in hf_dataset")
        img = rows[0]["image"].convert("RGB").copy()
    else:
        raise ValueError(
            "Provide either `image` (PIL Image) or both `image_id` and `hf_dataset`."
        )

    draw = ImageDraw.Draw(img)
    font = _try_font(_FONT_SIZE)

    # --- Top-K proposals (drawn first so they appear behind everything else) ---
    if top_k_boxes is not None:
        for box in top_k_boxes:
            coords = [float(v) for v in box.tolist()]
            draw.rectangle(coords, outline=_TOPK_COLOUR, width=_TOPK_WIDTH)

    # --- Ground-truth box (blue) ---
    if gt_box is not None:
        coords = [float(v) for v in gt_box.tolist()]
        draw.rectangle(coords, outline=_GT_COLOUR, width=_BOX_WIDTH)
        _draw_label(draw, coords[0], coords[1], "GT", _GT_COLOUR, font)

    # --- Predicted box (green) ---
    pred_coords = [float(v) for v in pred_box.tolist()]
    draw.rectangle(pred_coords, outline=_PRED_COLOUR, width=_BOX_WIDTH)
    pred_label  = f"pred  IoU={iou_score:.2f}" if iou_score is not None else "pred"
    _draw_label(draw, pred_coords[0], pred_coords[1], pred_label, _PRED_COLOUR, font)

    # --- Phrase banner at top ---
    annotated = _add_banner(img, phrase, font)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(save_path)

    return annotated


# ---------------------------------------------------------------------------
# make_results_grid
# ---------------------------------------------------------------------------

def make_results_grid(
    images:    List[Image.Image],
    n_cols:    int           = 4,
    cell_h:    int           = 300,    # height each image is resized to
    gap:       int           = 4,      # pixel gap between cells
    save_path: Optional[Path] = None,
) -> Image.Image:
    """
    Tile a list of annotated images into an N-column grid.

    Each image is proportionally resized to `cell_h` pixels tall.
    Columns are aligned to the widest cell in each column.
    The grid is padded with a white background.

    Args:
        images    : list of PIL Images (typically outputs of draw_grounding_result)
        n_cols    : number of columns in the grid
        cell_h    : target height for each cell in pixels
        gap       : gap between cells in pixels
        save_path : if provided, save the grid here

    Returns:
        Grid image (RGB).
    """
    if not images:
        raise ValueError("images list is empty")

    n_rows = math.ceil(len(images) / n_cols)

    # Resize each image to cell_h height, preserving aspect ratio
    resized: List[Image.Image] = []
    for img in images:
        w, h   = img.size
        new_w  = max(1, int(w * cell_h / h))
        resized.append(img.resize((new_w, cell_h), Image.LANCZOS))

    # Compute column widths: max width of images in each column
    col_widths = []
    for col in range(n_cols):
        indices   = range(col, len(resized), n_cols)
        col_widths.append(max((resized[i].size[0] for i in indices), default=0))

    total_w = sum(col_widths) + gap * (n_cols - 1)
    total_h = cell_h * n_rows + gap * (n_rows - 1)

    grid = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    for idx, img in enumerate(resized):
        row   = idx // n_cols
        col   = idx %  n_cols
        x_off = sum(col_widths[:col]) + gap * col
        y_off = row * (cell_h + gap)
        grid.paste(img, (x_off, y_off))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(save_path)

    return grid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_label(draw:   ImageDraw.ImageDraw,
                x:      float,
                y:      float,
                text:   str,
                colour: str,
                font:   ImageFont.ImageFont,
                ) -> None:
    """
    Draw a small filled-background label above a box corner.
    Falls back gracefully if the label would extend above the image top.
    """
    try:
        bbox    = draw.textbbox((0, 0), text, font=font)
        tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Pillow < 9.2 fallback
        tw, th  = draw.textsize(text, font=font)  # type: ignore[attr-defined]

    pad    = 2
    lx     = int(x)
    ly     = max(0, int(y) - th - 2 * pad)

    # Filled rectangle background for readability
    draw.rectangle(
        [lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad],
        fill=colour,
    )
    draw.text((lx + pad, ly + pad), text, fill="white", font=font)


def _add_banner(image:  Image.Image,
                phrase: str,
                font:   ImageFont.ImageFont,
                ) -> Image.Image:
    """Prepend a light-gray banner with the phrase text above the image."""
    banner   = Image.new("RGB", (image.width, _BANNER_H), color=_BANNER_BG)
    bdraw    = ImageDraw.Draw(banner)
    # Truncate phrase if too long to fit
    max_chars = max(1, image.width // 7)
    display   = phrase if len(phrase) <= max_chars else phrase[:max_chars - 1] + "…"
    bdraw.text((6, (_BANNER_H - _FONT_SIZE) // 2), display,
               fill=_BANNER_FG, font=font)

    combined = Image.new("RGB", (image.width, image.height + _BANNER_H))
    combined.paste(banner, (0, 0))
    combined.paste(image,  (0, _BANNER_H))
    return combined