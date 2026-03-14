"""
High-level API: Story Template Generation

Orchestrates the full workflow:
  1. StoryTemplateBuilderAgent (ADK) — guardrail → template_writer → template_formatter
     Input : story concept (title)
     Output: StoryTemplateOutput in state (name, prompt, target_lines)
  2. Format into StoryTemplate data model
  3. Persist via EntityRepository (same as /story-templates POST)
  4. Return the created StoryTemplate
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from virtual_streamer.agents.common.state_keys import TITLE, TEMPLATE_OUTPUT
from virtual_streamer.agents.story_template_builder import get_story_template_builder
from virtual_streamer.agents.story_template_builder.schema import StoryTemplateOutput
from virtual_streamer.utils.entity_repository import get_entity_repository
from virtual_streamer.video_server.models import StoryTemplate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/story-template-generation", tags=["Story Template Generation"])

APP_NAME = "story_template_generation"


# =============================================================================
# Request / Response Models
# =============================================================================


class StoryTemplateGenerationRequest(BaseModel):
    """Request to generate and register a new story template."""

    story_concept: str = Field(
        ...,
        description=(
            "The creative idea for the story template. "
            "Describe the tone, characters, scenario type, and comedic style you want. "
            "Example: 'A parody of C\\'est pas Sorcier where Fred launches a startup.'"
        ),
    )
    collection: str = Field(
        ...,
        description="Qdrant collection name for video search (e.g. 'cps_videos').",
    )
    character_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Character IDs to associate with this template. "
            "The agent will suggest characters in the prompt text, "
            "but only IDs listed here are wired to the template for TTS/video lookup."
        ),
    )


class StoryTemplateGenerationResponse(BaseModel):
    """Response after generating and registering a story template."""

    template: StoryTemplate
    agent_output: StoryTemplateOutput


# =============================================================================
# ADK Runner Helper
# =============================================================================


async def _run_template_builder(story_concept: str) -> StoryTemplateOutput:
    """
    Run StoryTemplateBuilderAgent for the given concept.

    Sets TITLE in session state, runs the sequential pipeline, and extracts
    TEMPLATE_OUTPUT (name, prompt, target_lines) from state.

    Args:
        story_concept: The user's creative story idea.

    Returns:
        StoryTemplateOutput parsed from session state.

    Raises:
        RuntimeError: If the agent completed without producing output.
    """
    agent = get_story_template_builder()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"tmpl_{uuid.uuid4().hex[:8]}"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={TITLE: story_concept},
    )

    logger.info(f"Running StoryTemplateBuilderAgent for concept: {story_concept[:80]}...")

    content = types.Content(role="user", parts=[types.Part(text=story_concept)])

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

    raw = session.state.get(TEMPLATE_OUTPUT)
    if not raw:
        raise RuntimeError(
            "StoryTemplateBuilderAgent completed but TEMPLATE_OUTPUT is missing from state."
        )

    if isinstance(raw, StoryTemplateOutput):
        return raw
    return StoryTemplateOutput.model_validate(raw)


# =============================================================================
# API Endpoint
# =============================================================================


@router.post("/generate", response_model=StoryTemplateGenerationResponse)
async def generate_story_template(request: StoryTemplateGenerationRequest):
    """
    Generate a new story template from a creative concept and register it.

    **Workflow**:
    1. Runs `StoryTemplateBuilderAgent` (guardrail → writer → formatter).
       The writer fetches available characters from the API and injects them
       into the prompt to guide character selection.
    2. The agent produces `name`, `prompt`, and `target_lines`.
    3. These are combined with the `collection` and `character_ids` from the
       request and persisted via the same repository as `POST /story-templates`.
    4. The created `StoryTemplate` is returned alongside the raw agent output.

    **Note**: `character_ids` must reference characters that already exist.
    The agent will describe characters in the prompt text, but only IDs
    you provide here are wired to the template for TTS and video lookup.
    """
    # Step 1: verify character_ids exist before running the (expensive) agent
    if request.character_ids:
        repo = get_entity_repository()
        for char_id in request.character_ids:
            char = await repo.get_character(char_id)
            if char is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Character '{char_id}' not found. Create it first.",
                )

    # Step 2: run the agent pipeline
    try:
        agent_output = await _run_template_builder(request.story_concept)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"Agent produced template '{agent_output.name}' "
        f"({agent_output.target_lines} lines)"
    )

    # Step 3: derive template_id from name (same logic as low-level endpoint)
    template_id = agent_output.name.lower().replace(" ", "_").replace("-", "_")

    # Step 4: persist via repository
    repo = get_entity_repository()
    try:
        template_data = await repo.create_story_template(
            template_id=template_id,
            name=agent_output.name,
            prompt=agent_output.prompt,
            collection=request.collection,
            target_lines=agent_output.target_lines,
            character_ids=request.character_ids,
        )
    except Exception as e:
        logger.error(f"Failed to persist story template: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Template generated successfully but could not be saved: {e}",
        )

    template = StoryTemplate(
        template_id=template_data["template_id"],
        name=template_data["name"],
        prompt=template_data["prompt"],
        collection=template_data["collection"],
        target_lines=template_data["target_lines"],
        character_ids=template_data.get("character_ids", []),
        created_at=template_data["created_at"],
        updated_at=template_data["updated_at"],
    )

    logger.info(f"Registered story template '{template.template_id}'")

    return StoryTemplateGenerationResponse(
        template=template,
        agent_output=agent_output,
    )