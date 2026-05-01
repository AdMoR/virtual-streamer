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

import logging
import uuid

from fastapi import APIRouter, HTTPException
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

    location = Location(
        location_id=location_data["location_id"],
        name=location_data["name"],
        description=location_data["description"],
        story_template_id=location_data["story_template_id"],
        created_at=location_data["created_at"],
        updated_at=location_data["updated_at"],
    )

    logger.info(f"Registered location '{location.location_id}'")

    return LocationGenerationResponse(
        location=location,
        agent_output=agent_output,
    )
