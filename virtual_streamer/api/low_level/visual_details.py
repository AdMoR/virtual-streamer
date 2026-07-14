"""
Low-level API: Visual Detail Management

CRUD for visual details (props, logos, vehicles, ...) scoped to story
templates. Visual details carry an identity image that is embedded into
IC-LoRA reference sheets so the generated video keeps their appearance.

Image provenance sets the label:
  - generated from the description -> the prompt itself becomes the caption
    (no LLM call needed)
  - uploaded by the user            -> the image-tagger vision LLM captions it
"""

import base64
import logging
import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from typing import List, Optional

from virtual_streamer.video_server.models import VisualDetail
from virtual_streamer.image_generation.image_tagger import label_from_prompt, tag_image
from virtual_streamer.utils.entity_repository import get_entity_repository
from virtual_streamer.utils.minio_client import get_storage_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visual-details", tags=["Visual Details"])

PREFIX_VISUAL_DETAILS = "visual_details/"


def _dict_to_model(data: dict) -> VisualDetail:
    return VisualDetail(
        detail_id=data["detail_id"],
        name=data["name"],
        description=data["description"],
        category=data["category"],
        story_template_id=data["story_template_id"],
        image_path=data.get("image_path"),
        label=data.get("label"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.post("", response_model=VisualDetail, status_code=status.HTTP_201_CREATED)
async def create_visual_detail(
    name: str = Form(..., description="Display name (e.g. 'FreshMart tote bag')"),
    description: str = Form(..., description="Prose/diffusion prompt describing the detail"),
    story_template_id: str = Form(..., description="Story template this detail belongs to"),
    category: str = Form("Props", description="Reference-sheet label: 'Props', 'Logo', ..."),
):
    """Creates a new VisualDetail. The `detail_id` is a slug derived from the name."""
    repo = get_entity_repository()

    detail_id = name.lower().replace(" ", "-")

    template = await repo.get_story_template(story_template_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Story template '{story_template_id}' not found",
        )
    if await repo.get_visual_detail(detail_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Visual detail '{detail_id}' already exists",
        )

    data = await repo.create_visual_detail(
        detail_id=detail_id,
        name=name,
        description=description,
        story_template_id=story_template_id,
        category=category,
    )
    return _dict_to_model(data)


@router.get("", response_model=List[VisualDetail])
async def list_visual_details(story_template_id: str):
    """Lists all visual details of a story template."""
    repo = get_entity_repository()
    details = await repo.list_visual_details_by_template(story_template_id)
    return [_dict_to_model(d) for d in details]


@router.get("/{detail_id}", response_model=VisualDetail)
async def get_visual_detail(detail_id: str):
    repo = get_entity_repository()
    data = await repo.get_visual_detail(detail_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Visual detail '{detail_id}' not found")
    return _dict_to_model(data)


@router.delete("/{detail_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_visual_detail(detail_id: str):
    repo = get_entity_repository()
    if not await repo.delete_visual_detail(detail_id):
        raise HTTPException(status_code=404, detail=f"Visual detail '{detail_id}' not found")


# =============================================================================
# Image: upload (tagged by vision LLM) or generate (labeled from prompt)
# =============================================================================


@router.post("/{detail_id}/upload-image", response_model=VisualDetail)
async def upload_visual_detail_image(
    detail_id: str,
    image_file: UploadFile = File(..., description="Image file (PNG, JPEG, or WebP)"),
):
    """Upload a custom image for a visual detail.

    The image is captioned/labeled by the image-tagger vision LLM (best
    effort) since its content is not known from a generation prompt.
    """
    repo = get_entity_repository()
    detail = await repo.get_visual_detail(detail_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Visual detail '{detail_id}' not found")

    content = await image_file.read()
    fname = (image_file.filename or "").lower()
    if fname.endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif fname.endswith(".webp"):
        content_type = "image/webp"
    else:
        content_type = "image/png"

    minio_key = f"{PREFIX_VISUAL_DETAILS}{detail_id}/identity.png"
    storage = get_storage_client()
    await storage.put_object(minio_key, content, content_type=content_type)

    # Tag the uploaded image (best effort — never blocks the upload)
    tmp_dir = os.path.join(os.environ.get("TEMP_DIR", "./temp"), f"vd_tag_{uuid.uuid4().hex[:8]}")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        local_path = os.path.join(tmp_dir, "image")
        with open(local_path, "wb") as f:
            f.write(content)
        label = await tag_image(
            local_path,
            storage_path=minio_key,
            entity_name=detail["name"],
            entity_description=detail["description"],
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    data = await repo.update_visual_detail_image(detail_id, minio_key, label.model_dump(mode="json"))
    return _dict_to_model(data)


class GenerateDetailImageRequest(BaseModel):
    sd_server_url: str = Field("http://gx10-cbc5:1234", description="Stable Diffusion server URL")
    prompt_override: Optional[str] = Field(
        None, description="Custom generation prompt (defaults to the detail description)"
    )


@router.post("/{detail_id}/generate-image", response_model=VisualDetail)
async def generate_visual_detail_image(detail_id: str, request: GenerateDetailImageRequest):
    """Generate the detail image from its description with Stable Diffusion.

    The generation prompt is stored as the image label/caption directly — no
    LLM tagging pass is needed since the prompt already describes the image.
    """
    from virtual_streamer.image_generation.stable_cpp_client import (
        StableDiffusionCppClient,
        StableDiffusionCppConfig,
        Txt2ImageParams,
    )

    repo = get_entity_repository()
    detail = await repo.get_visual_detail(detail_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Visual detail '{detail_id}' not found")

    prompt = request.prompt_override or detail["description"]
    tmp_dir = os.path.join(os.environ.get("TEMP_DIR", "./temp"), f"vd_gen_{uuid.uuid4().hex[:8]}")
    try:
        config = StableDiffusionCppConfig(server_url=request.sd_server_url)
        async with StableDiffusionCppClient(config) as client:
            result = await client.txt2image(
                Txt2ImageParams(
                    prompt=prompt,
                    negative_prompt="text, watermark, blurry, distorted",
                    width=1280,
                    height=720,
                ),
                output_dir=tmp_dir,
            )
        if not result.image_path or not os.path.exists(result.image_path):
            raise HTTPException(status_code=502, detail="Image generation failed")

        minio_key = f"{PREFIX_VISUAL_DETAILS}{detail_id}/identity.png"
        storage = get_storage_client()
        with open(result.image_path, "rb") as f:
            await storage.put_object(minio_key, f.read(), content_type="image/png")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    label = label_from_prompt(minio_key, prompt)
    data = await repo.update_visual_detail_image(detail_id, minio_key, label.model_dump(mode="json"))
    return _dict_to_model(data)
