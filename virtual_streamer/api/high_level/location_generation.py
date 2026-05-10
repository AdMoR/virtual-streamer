"""
High-level API: Location Generation

Orchestrates the full workflow:
  1. LocationBuilderAgent (ADK) — location_writer → location_formatter
     Input : location name + story_template_id
     Output: LocationDescriptionOutput in state (description)
  2. Format into Location data model
  3. Persist via EntityRepository
  4. Return the created Location
"""

import base64
import logging
import os
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from virtual_streamer.agents.common.state_keys import (
    LOCATION_NAME,
    STORY_TEMPLATE_ID,
    LOCATION_OUTPUT,
)
from virtual_streamer.agents.location_builder import get_location_builder
from virtual_streamer.agents.location_builder.schema import LocationDescriptionOutput
from virtual_streamer.utils.entity_repository import get_entity_repository
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.video_server.models import Location

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/location-generation", tags=["Location Generation"])

APP_NAME = "location_generation"


# =============================================================================
# Request / Response Models
# =============================================================================


class LocationGenerationRequest(BaseModel):
    """Request to generate and register a new location."""

    location_name: str = Field(
        ...,
        description=(
            "The name of the location to generate a description for. "
            "Example: 'Medieval Castle' or 'Tokyo Street Market'."
        ),
    )
    story_template_id: str = Field(
        ...,
        description="ID of the story template this location will be scoped to.",
    )
    sd_server_url: Optional[str] = Field(
        None,
        description=(
            "If provided, immediately generate an identity image via this "
            "Stable Diffusion server URL (e.g. 'http://gx10-cbc5:1234') and "
            "store it with the location."
        ),
    )


class LocationGenerationResponse(BaseModel):
    """Response after generating and registering a location."""

    location: Location
    agent_output: LocationDescriptionOutput


# =============================================================================
# ADK Runner Helper
# =============================================================================


async def _run_location_builder(
    location_name: str,
    story_template_id: str,
) -> LocationDescriptionOutput:
    """
    Run LocationBuilderAgent for the given location name and template.

    Sets LOCATION_NAME and STORY_TEMPLATE_ID in session state, runs the
    sequential pipeline, and extracts LOCATION_OUTPUT (description) from state.

    Raises:
        RuntimeError: If the agent completed without producing output.
    """
    agent = get_location_builder()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"loc_{uuid.uuid4().hex[:8]}"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            LOCATION_NAME: location_name,
            STORY_TEMPLATE_ID: story_template_id,
        },
    )

    logger.info(
        f"Running LocationBuilderAgent for '{location_name}' "
        f"(template: {story_template_id})..."
    )

    content = types.Content(role="user", parts=[types.Part(text=location_name)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            logger.debug(f"Final response from {event.author}")

    # Read updated session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    raw = session.state.get(LOCATION_OUTPUT)
    if not raw:
        raise RuntimeError(
            "LocationBuilderAgent completed but LOCATION_OUTPUT is missing from state."
        )

    if isinstance(raw, LocationDescriptionOutput):
        return raw
    return LocationDescriptionOutput.model_validate(raw)


# =============================================================================
# API Endpoint
# =============================================================================


@router.post("/generate", response_model=LocationGenerationResponse)
async def generate_location(request: LocationGenerationRequest):
    """
    Generate a new location description using the LocationBuilderAgent and register it.

    The agent uses the story template context to write a diffusion-model prompt
    describing the location environment (no characters, only setting).

    The `location_id` is derived from the name by lowercasing and replacing
    spaces with hyphens (e.g. 'Medieval Castle' → 'medieval-castle').

    **Workflow**:
    1. Verifies the story template exists.
    2. Checks for duplicate location_id (returns 409 if already registered).
    3. Runs `LocationBuilderAgent` (writer → formatter).
    4. Persists the Location in the database.
    5. Returns the created `Location` alongside the raw agent output.
    """
    repo = get_entity_repository()

    # Verify the story template exists
    template = await repo.get_story_template(request.story_template_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Story template '{request.story_template_id}' not found",
        )

    location_id = request.location_name.lower().replace(" ", "-")

    # Check for duplicate
    existing = await repo.get_location(location_id)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Location '{location_id}' already exists for this template",
        )

    # Run the agent pipeline
    try:
        agent_output = await _run_location_builder(
            location_name=request.location_name,
            story_template_id=request.story_template_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"Agent produced location description for '{request.location_name}' "
        f"({len(agent_output.description)} chars)"
    )

    # Persist via repository
    try:
        location_data = await repo.create_location(
            location_id=location_id,
            name=request.location_name,
            description=agent_output.description,
            story_template_id=request.story_template_id,
        )
    except Exception as e:
        logger.error(f"Failed to persist location: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Location generated successfully but could not be saved: {e}",
        )

    # Optionally generate and store the identity image
    if request.sd_server_url:
        temp_dir = os.path.join(
            os.environ.get("TEMP_DIR", "./temp"),
            f"loc_img_{uuid.uuid4().hex[:8]}",
        )
        try:
            from virtual_streamer.video_generation.story_to_video import generate_location_image

            image_path = await generate_location_image(
                location=location_data,
                character={},
                output_dir=temp_dir,
                sd_server_url=request.sd_server_url,
            )

            if image_path and os.path.exists(image_path):
                minio_key = f"locations/{location_id}/identity.png"
                storage = get_storage_client()
                await storage.upload_file(image_path, minio_key)
                location_data = await repo.update_location_image(location_id, minio_key)
                logger.info(f"Stored identity image for '{location_id}' at '{minio_key}'")
            else:
                logger.warning(
                    f"Image generation skipped or failed for '{location_id}' "
                    "(SD server may be unavailable)"
                )
        except Exception as img_err:
            logger.warning(
                f"Identity image generation failed for '{location_id}': {img_err}",
                exc_info=True,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    location = Location(
        location_id=location_data["location_id"],
        name=location_data["name"],
        description=location_data["description"],
        story_template_id=location_data["story_template_id"],
        image_path=location_data.get("image_path"),
        created_at=location_data["created_at"],
        updated_at=location_data["updated_at"],
    )

    logger.info(f"Registered location '{location.location_id}'")

    return LocationGenerationResponse(
        location=location,
        agent_output=agent_output,
    )


# =============================================================================
# Location Image Generation
# =============================================================================


class LocationImageRequest(BaseModel):
    location_id: str = Field(..., description="ID of the location to render")
    character_id: Optional[str] = Field(
        None, description="Optional character ID to include in the scene"
    )
    sd_server_url: str = Field(
        "http://gx10-cbc5:1234",
        description="Stable Diffusion server URL",
    )


class LocationImageResponse(BaseModel):
    image_data: str = Field(..., description="Base64-encoded PNG image")
    location_id: str
    location_name: str
    character_id: Optional[str]
    character_name: Optional[str]


@router.post("/generate-image", response_model=LocationImageResponse)
async def generate_location_image_endpoint(request: LocationImageRequest):
    """
    Generate a conditioning image for a location using Stable Diffusion.

    Fetches the location (and optional character) from the database, then
    calls the SD server to produce a 1280×720 PNG.  The image is returned
    as a base64-encoded string so the caller can embed it directly.
    """
    from virtual_streamer.video_generation.story_to_video import generate_location_image

    repo = get_entity_repository()

    location_data = await repo.get_location(request.location_id)
    if location_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{request.location_id}' not found",
        )

    character_data: dict = {}
    if request.character_id:
        character_data = await repo.get_character(request.character_id) or {}
        if not character_data:
            raise HTTPException(
                status_code=404,
                detail=f"Character '{request.character_id}' not found",
            )

    temp_dir = os.path.join(
        os.environ.get("TEMP_DIR", "./temp"),
        f"loc_img_{uuid.uuid4().hex[:8]}",
    )
    try:
        image_path = await generate_location_image(
            location=location_data,
            character=character_data,
            output_dir=temp_dir,
            sd_server_url=request.sd_server_url,
        )

        if image_path is None or not os.path.exists(image_path):
            raise HTTPException(
                status_code=502,
                detail="Image generation failed — SD server may be unavailable",
            )

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return LocationImageResponse(
        image_data=image_b64,
        location_id=location_data["location_id"],
        location_name=location_data["name"],
        character_id=character_data.get("character_id") or None,
        character_name=character_data.get("name") or None,
    )


# =============================================================================
# Regenerate & persist location image
# =============================================================================


class LocationRegenerateImageRequest(BaseModel):
    sd_server_url: str = Field(
        "http://gx10-cbc5:1234",
        description="Stable Diffusion server URL",
    )
    prompt_override: Optional[str] = Field(
        None,
        description=(
            "Custom prompt for this generation run. "
            "When omitted the location description is used."
        ),
    )


class LocationRegenerateImageResponse(BaseModel):
    image_data: str = Field(..., description="Base64-encoded PNG image")
    location: Location


@router.post("/{location_id}/regenerate-image", response_model=LocationRegenerateImageResponse)
async def regenerate_location_image(
    location_id: str,
    request: LocationRegenerateImageRequest,
):
    """
    Generate (or regenerate) the identity image for an existing location.

    Generates a 1280×720 PNG via Stable Diffusion using either the stored
    description or a custom ``prompt_override``, uploads it to MinIO at
    ``locations/{location_id}/identity.png``, persists the path in the database,
    and returns the image as a base64-encoded PNG alongside the updated Location.
    """
    from virtual_streamer.video_generation.story_to_video import generate_location_image

    repo = get_entity_repository()

    location_data = await repo.get_location(location_id)
    if location_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location_id}' not found",
        )

    # Allow the caller to override the prompt for this run without persisting it
    gen_data = dict(location_data)
    if request.prompt_override:
        gen_data["description"] = request.prompt_override

    temp_dir = os.path.join(
        os.environ.get("TEMP_DIR", "./temp"),
        f"loc_regen_{uuid.uuid4().hex[:8]}",
    )
    try:
        image_path = await generate_location_image(
            location=gen_data,
            character={},
            output_dir=temp_dir,
            sd_server_url=request.sd_server_url,
        )

        if image_path is None or not os.path.exists(image_path):
            raise HTTPException(
                status_code=502,
                detail="Image generation failed — SD server may be unavailable",
            )

        minio_key = f"locations/{location_id}/identity.png"
        storage = get_storage_client()
        await storage.upload_file(image_path, minio_key)
        location_data = await repo.update_location_image(location_id, minio_key)

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    location = Location(
        location_id=location_data["location_id"],
        name=location_data["name"],
        description=location_data["description"],
        story_template_id=location_data["story_template_id"],
        image_path=location_data.get("image_path"),
        created_at=location_data["created_at"],
        updated_at=location_data["updated_at"],
    )

    logger.info(f"Regenerated identity image for '{location_id}'")

    return LocationRegenerateImageResponse(image_data=image_b64, location=location)


# =============================================================================
# Upload a custom location image
# =============================================================================


@router.post("/{location_id}/upload-image", response_model=LocationRegenerateImageResponse)
async def upload_location_image(
    location_id: str,
    image_file: UploadFile = File(..., description="Image file (PNG, JPEG, or WebP)"),
):
    """
    Upload a custom image for a location, replacing any existing image.

    Stores the file in MinIO at ``locations/{location_id}/identity.png`` and
    updates the database record, then returns the image as base64 alongside the
    updated Location.
    """
    repo = get_entity_repository()

    location_data = await repo.get_location(location_id)
    if location_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location_id}' not found",
        )

    content = await image_file.read()

    fname = (image_file.filename or "").lower()
    if fname.endswith(".jpg") or fname.endswith(".jpeg"):
        content_type = "image/jpeg"
    elif fname.endswith(".webp"):
        content_type = "image/webp"
    else:
        content_type = "image/png"

    minio_key = f"locations/{location_id}/identity.png"
    storage = get_storage_client()
    await storage.put_object(minio_key, content, content_type=content_type)

    location_data = await repo.update_location_image(location_id, minio_key)

    image_b64 = base64.b64encode(content).decode()

    location = Location(
        location_id=location_data["location_id"],
        name=location_data["name"],
        description=location_data["description"],
        story_template_id=location_data["story_template_id"],
        image_path=location_data.get("image_path"),
        created_at=location_data["created_at"],
        updated_at=location_data["updated_at"],
    )

    logger.info(f"Uploaded custom identity image for '{location_id}'")

    return LocationRegenerateImageResponse(image_data=image_b64, location=location)


# =============================================================================
# Retrieve stored location image as base64
# =============================================================================


@router.get("/{location_id}/image", response_model=LocationImageResponse)
async def get_location_stored_image(location_id: str):
    """
    Retrieve the stored identity image for a location as a base64-encoded PNG.

    Returns 404 when no image has been generated for this location yet.
    """
    repo = get_entity_repository()

    location_data = await repo.get_location(location_id)
    if location_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location_id}' not found",
        )

    image_path = location_data.get("image_path")
    if not image_path:
        raise HTTPException(
            status_code=404,
            detail="No image stored for this location",
        )

    temp_dir = os.path.join(
        os.environ.get("TEMP_DIR", "./temp"),
        f"loc_dl_{uuid.uuid4().hex[:8]}",
    )
    os.makedirs(temp_dir, exist_ok=True)
    try:
        local_path = os.path.join(temp_dir, "identity.png")
        storage = get_storage_client()
        await storage.download_file(image_path, local_path)

        with open(local_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return LocationImageResponse(
        image_data=image_b64,
        location_id=location_data["location_id"],
        location_name=location_data["name"],
        character_id=None,
        character_name=None,
    )
