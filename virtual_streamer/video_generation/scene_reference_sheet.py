"""
Scene-level reference-sheet assembly.

Bridges DB entities (location, characters, visual details) and the pure
compositor in `image_generation/reference_sheet.py`: downloads the entity
images from MinIO, picks up to 3 character views (preferring view diversity
from their LabeledImage tags), builds the sheet, and attaches
`reference_sheet_path` / `reference_sheet_description` to the SceneInput so
the ReferenceSheetStrategy picks the scene up.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, List, Optional

from virtual_streamer.image_generation.reference_sheet import (
    SheetCell,
    build_reference_sheet,
)
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.video_generation.scene_input import SceneInput
from virtual_streamer.video_server.models import LabeledImage

logger = logging.getLogger(__name__)

MAX_CHARACTER_VIEWS = 3


def select_character_views(character: dict, max_views: int = MAX_CHARACTER_VIEWS) -> List[LabeledImage]:
    """Pick up to *max_views* identity images, preferring view diversity.

    Uses the LabeledImage tags when present: distinct angles first, then
    distinct framings. Unlabeled images (plain identity_images paths) fall
    back to first-N order.
    """
    labels = [LabeledImage.model_validate(l) for l in (character.get("labeled_images") or [])]
    labeled_paths = {l.image_path for l in labels}
    # Identity images without a label entry still count, as unlabeled
    labels += [
        LabeledImage(image_path=p)
        for p in (character.get("identity_images") or [])
        if p not in labeled_paths
    ]
    if len(labels) <= max_views:
        return labels

    selected: List[LabeledImage] = []
    seen_angles, seen_framings = set(), set()
    # First pass: one image per distinct angle
    for label in labels:
        if len(selected) >= max_views:
            return selected
        if label.angle and label.angle not in seen_angles:
            selected.append(label)
            seen_angles.add(label.angle)
    # Second pass: distinct framings not yet covered
    for label in labels:
        if len(selected) >= max_views:
            return selected
        if label not in selected and label.framing and label.framing not in seen_framings:
            selected.append(label)
            seen_framings.add(label.framing)
    # Fill with whatever remains
    for label in labels:
        if len(selected) >= max_views:
            break
        if label not in selected:
            selected.append(label)
    return selected


def _character_cell_description(character: dict, label: LabeledImage) -> str:
    """Cell prose: the image caption when available, else the character
    description with a view clause built from the labels (never invented)."""
    if label.caption:
        return label.caption
    base = character.get("description") or character.get("name") or "The character"
    clauses = []
    if label.is_multi_view:
        clauses.append("shown from multiple angles")
    elif label.angle:
        clauses.append(f"shown from the {label.angle.value.replace('_', '-')}")
    if label.framing:
        clauses.append(f"({label.framing.value.replace('_', ' ')} view)")
    return f"{base}. {' '.join(clauses).capitalize()}." if clauses else base


async def attach_reference_sheet(
    scene_input: SceneInput,
    location: Optional[dict],
    character_dicts: List[dict],
    visual_details: List[dict],
    resolution: str,
    output_dir: str,
) -> SceneInput:
    """Build the reference sheet for one scene and attach it to the SceneInput.

    Returns the (copied) SceneInput with reference_sheet_path/description set,
    or the original unchanged when no sheet could be built. Best effort.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        storage = get_storage_client()
        _local_cache: Dict[str, str] = {}

        async def _download(minio_key: str) -> Optional[str]:
            if minio_key in _local_cache:
                return _local_cache[minio_key]
            local = os.path.join(output_dir, f"ref_{uuid.uuid4().hex[:8]}_{os.path.basename(minio_key)}")
            try:
                await storage.download_file(minio_key, local)
            except Exception as exc:
                logger.warning(f"[sheet] could not download {minio_key}: {exc}")
                return None
            _local_cache[minio_key] = local
            return local

        cells: List[SheetCell] = []

        if location and location.get("image_path"):
            local = await _download(location["image_path"])
            if local:
                cells.append(SheetCell(
                    image_path=local, category="Setting", description=location["description"],
                ))

        for character in character_dicts:
            for label in select_character_views(character):
                local = await _download(label.image_path)
                if local:
                    cells.append(SheetCell(
                        image_path=local,
                        category="Character",
                        description=_character_cell_description(character, label),
                    ))

        for detail in visual_details:
            if not detail.get("image_path"):
                continue
            local = await _download(detail["image_path"])
            if local:
                label = detail.get("label") or {}
                cells.append(SheetCell(
                    image_path=local,
                    category=detail.get("category") or "Props",
                    description=label.get("caption") or detail["description"],
                ))

        sheet = build_reference_sheet(cells, resolution=resolution, output_dir=output_dir)
        if sheet is None:
            return scene_input

        logger.info(
            f"[scene {scene_input.scene_index}] reference sheet attached "
            f"({sheet.cell_count} cells): {sheet.image_path}"
        )
        return scene_input.model_copy(update={
            "reference_sheet_path": sheet.image_path,
            "reference_sheet_description": sheet.description,
        })
    except Exception as exc:
        logger.warning(f"[scene {scene_input.scene_index}] reference sheet build failed: {exc}", exc_info=True)
        return scene_input
