"""
LLM-based image tagger for entity images.

Produces a `LabeledImage` (view labels + caption) for an image, so uploaded
character/detail images integrate automatically into the reference-sheet
pipeline. Two behaviors, matching how the image came to exist:

  - `tag_image(...)` — the image was **uploaded** (no known provenance): a
    vision LLM classifies framing/angle/variation/is_multi_view and writes a
    caption. The structured output schema *is* the stored `LabeledImage`
    schema (minus image_path), so tagging and persistence cannot diverge.

  - `label_from_prompt(...)` — the image was **generated** from a prompt
    (e.g. visual details created during story creation): the prompt already
    describes the image, so it becomes the caption directly — no LLM call.

Tagging is best-effort by design: callers should treat a plain
`LabeledImage(image_path=...)` fallback as acceptable.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Optional

from pydantic import BaseModel

from virtual_streamer.video_server.models import ImageAngle, ImageFraming, LabeledImage
from virtual_streamer.video_generation.config import LLMConfig
from virtual_streamer.video_generation.implementations import create_llm
from virtual_streamer.video_generation.interfaces import LLMInterface

logger = logging.getLogger(__name__)


class ImageTags(BaseModel):
    """Structured tagger output — LabeledImage minus the storage path."""

    framing: Optional[ImageFraming] = None
    angle: Optional[ImageAngle] = None
    variation: Optional[str] = None
    is_multi_view: bool = False
    caption: str = ""


_TAGGING_PROMPT = """You are labeling a reference image of an entity for a video generation pipeline.
{entity_context}
Analyze the image and reply with ONLY a JSON object (no markdown fence, no commentary) with these fields:
- "framing": one of "face", "bust", "full_body", or null if not a character image
- "angle": one of "front", "back", "left", "right", "three_quarter", or null if unclear
- "variation": short free text for any notable variation (outfit, expression, accessory), or null
- "is_multi_view": true if the image is a collage/sheet showing the subject from multiple angles
- "caption": 1-3 sentences of rich prose describing exactly what is visible (subject, colors, clothing, style, background), suitable for use in an image-generation prompt
"""


def _extract_json(text: str) -> dict:
    """Parse the model reply into a dict, tolerating markdown fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in tagger reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


async def tag_image(
    local_image_path: str,
    storage_path: str,
    entity_name: Optional[str] = None,
    entity_description: Optional[str] = None,
    llm: Optional[LLMInterface] = None,
) -> LabeledImage:
    """Tag an **uploaded** image with a vision LLM. Best-effort.

    Returns a LabeledImage; on any failure returns an unlabeled one so the
    upload flow never breaks because of tagging.
    """
    try:
        llm = llm or create_llm(LLMConfig())

        context = ""
        if entity_name or entity_description:
            context = (
                f"The image is a reference for: {entity_name or 'an entity'}"
                + (f" — {entity_description}" if entity_description else "")
                + "\n"
            )
        prompt = _TAGGING_PROMPT.format(entity_context=context)

        with open(local_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()

        reply = await llm.complete_with_vision(prompt, image_base64)
        try:
            tags = ImageTags.model_validate(_extract_json(reply))
        except Exception:
            # One retry — vision models occasionally garble the JSON/enums
            reply = await llm.complete_with_vision(prompt, image_base64)
            tags = ImageTags.model_validate(_extract_json(reply))

        return LabeledImage(image_path=storage_path, **tags.model_dump())
    except Exception as exc:
        logger.warning(f"Image tagging failed for {storage_path}: {exc}")
        return LabeledImage(image_path=storage_path)


def label_from_prompt(storage_path: str, prompt: str, is_multi_view: bool = False) -> LabeledImage:
    """Label a **generated** image from its generation prompt — no LLM call.

    The prompt already describes the image, so it serves as the caption.
    """
    return LabeledImage(
        image_path=storage_path,
        caption=prompt.strip(),
        is_multi_view=is_multi_view,
    )
