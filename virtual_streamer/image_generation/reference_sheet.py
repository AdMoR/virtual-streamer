"""
Reference-sheet compositor for LTX IC-LoRA (ingredients) conditioning.

Builds a single grid image ("reference sheet") out of entity images —
locations (Setting), character identity images (Character), visual details
(Props/Logo/...) — plus the matching prose description in the format the
IC-LoRA was trained on:

    ### Reference Sheet Description
    **Top Row Left (Setting):** <prose>
    **Top Row Right (Character):** <prose>
    ...

Layout rules:
  - The canvas is exactly the video resolution (WxH), so the sheet keeps the
    video's aspect ratio by construction (IC-LoRA reference downscale factor
    is 1 — the sheet is fed at output resolution).
  - Cells are packed into up to 3 rows of up to 3 cells (max 9 cells).
  - Source images can have any aspect ratio: each is scaled to fit inside its
    cell (letterboxed, never cropped or stretched) and centered on a neutral
    background — empty space is expected and fine.
  - The compositor and the description generator share the same placement
    computation, so position labels always match the pixels.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_CELLS = 9
_MAX_COLS = 3

#: Neutral cell background — mid gray reads as "empty" to the model.
_BACKGROUND = (128, 128, 128)
#: Gutter between cells, in pixels, drawn in white so panels read as separate.
_GUTTER = 8

#: Cell category ordering on the sheet (Setting first, Logo last).
_CATEGORY_ORDER = {"Setting": 0, "Character": 1, "Props": 2, "Logo": 9}


class SheetCell(BaseModel):
    """One panel of the reference sheet."""

    image_path: str  # local filesystem path
    category: str    # "Setting", "Character", "Props", "Logo", ...
    description: str  # prose for this cell (caption or entity description)


class ReferenceSheet(BaseModel):
    """Composited sheet image + its prose description."""

    image_path: str
    description: str
    cell_count: int


def _pack_rows(n: int) -> List[int]:
    """Distribute *n* cells into up to 3 rows of up to 3, top rows first.

    7 cells -> [3, 2, 2] (the layout of the FreshMart example sheet).
    """
    n_rows = min(3, (n + _MAX_COLS - 1) // _MAX_COLS)
    base, extra = divmod(n, n_rows)
    return [base + (1 if r < extra else 0) for r in range(n_rows)]


def _position_labels(row_sizes: List[int]) -> List[str]:
    """Position label for each cell, matching the packing order."""
    row_names = {
        1: ["Top Row"],
        2: ["Top Row", "Bottom Row"],
        3: ["Top Row", "Middle Row", "Bottom Row"],
    }[len(row_sizes)]
    col_names = {1: [""], 2: [" Left", " Right"], 3: [" Left", " Middle", " Right"]}

    labels: List[str] = []
    for row_name, size in zip(row_names, row_sizes):
        for col in col_names[size]:
            labels.append(f"{row_name}{col}")
    return labels


def _sort_cells(cells: List[SheetCell]) -> List[SheetCell]:
    """Order cells by category convention: Setting, Character, Props..., Logo."""
    return sorted(
        cells, key=lambda c: _CATEGORY_ORDER.get(c.category, _CATEGORY_ORDER["Props"])
    )


def _paste_letterboxed(canvas: Image.Image, image_path: str, box: Tuple[int, int, int, int]) -> None:
    """Scale the image to fit inside *box* (aspect preserved) and center it."""
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        scale = min(box_w / img.width, box_h / img.height)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        offset = (x0 + (box_w - new_size[0]) // 2, y0 + (box_h - new_size[1]) // 2)
        canvas.paste(img, offset)


def build_reference_sheet(
    cells: List[SheetCell],
    resolution: str,
    output_dir: str,
    filename: str = "reference_sheet.png",
) -> Optional[ReferenceSheet]:
    """Composite *cells* into a grid sheet at *resolution* ("WxH").

    Returns None when there are no usable cells. Cells beyond MAX_CELLS are
    dropped (lowest-priority categories last, so they are the ones cut).
    """
    cells = [c for c in _sort_cells(cells) if os.path.exists(c.image_path)]
    if not cells:
        logger.warning("Reference sheet requested but no cell image exists on disk")
        return None
    if len(cells) > MAX_CELLS:
        dropped = [c.category for c in cells[MAX_CELLS:]]
        logger.warning(f"Reference sheet capped at {MAX_CELLS} cells — dropping {dropped}")
        cells = cells[:MAX_CELLS]

    width, height = (int(v) for v in resolution.lower().split("x"))
    row_sizes = _pack_rows(len(cells))
    labels = _position_labels(row_sizes)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    row_h = (height - _GUTTER * (len(row_sizes) - 1)) // len(row_sizes)

    idx = 0
    for row, size in enumerate(row_sizes):
        y0 = row * (row_h + _GUTTER)
        cell_w = (width - _GUTTER * (size - 1)) // size
        for col in range(size):
            x0 = col * (cell_w + _GUTTER)
            box = (x0, y0, x0 + cell_w, y0 + row_h)
            # Fill the cell background, then letterbox the image inside it
            canvas.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), _BACKGROUND), box[:2])
            _paste_letterboxed(canvas, cells[idx].image_path, box)
            idx += 1

    os.makedirs(output_dir, exist_ok=True)
    sheet_path = os.path.join(output_dir, filename)
    canvas.save(sheet_path)

    parts = ["### Reference Sheet Description"]
    for label, cell in zip(labels, cells):
        parts.append(f"**{label} ({cell.category}):** {cell.description.strip()}")
    description = " ".join(parts)

    logger.info(f"Reference sheet built: {sheet_path} ({len(cells)} cells, {resolution})")
    return ReferenceSheet(image_path=sheet_path, description=description, cell_count=len(cells))
