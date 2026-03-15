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


FIXED_COLLECTION = "random"
FIXED_CHARACTER_ID = "narrator"


class StoryTemplateGenerationRequest(BaseModel):
    """Request to generate and register a new story template."""

    story_concept: str = Field(
        ...,
        description=(
            "The creative idea for the story template. "
            "Describe the tone, scenario type, and comedic style you want. "
            "Example: 'A parody of C\\'est pas Sorcier where Fred launches a startup.'"
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

    **Fixed parameters** (not configurable via this endpoint):
    - `collection`: always `"random"` — videos are drawn from the generic random pool,
      not from a character- or show-specific collection.
    - `character_ids`: always `["narrator"]` — the narrator is an off-screen voice
      with no on-screen character asset. No other characters are wired to this template.

    **Workflow**:
    1. Runs `StoryTemplateBuilderAgent` (guardrail → writer → formatter).
       The writer generates a narrator-driven prompt suited for off-screen delivery.
    2. The agent produces `name`, `prompt`, and `target_lines`.
    3. These are persisted with the fixed `collection="random"` and
       `character_ids=["narrator"]` via the same repository as `POST /story-templates`.
    4. The created `StoryTemplate` is returned alongside the raw agent output.
    """
    # Run the agent pipeline
    try:
        agent_output = await _run_template_builder(request.story_concept)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"Agent produced template '{agent_output.name}' "
        f"({agent_output.target_lines} lines)"
    )

    # Derive template_id from name (same logic as low-level endpoint)
    template_id = agent_output.name.lower().replace(" ", "_").replace("-", "_")

    # Persist via repository with fixed collection and narrator character
    repo = get_entity_repository()
    try:
        template_data = await repo.create_story_template(
            template_id=template_id,
            name=agent_output.name,
            prompt=agent_output.prompt,
            collection=FIXED_COLLECTION,
            target_lines=agent_output.target_lines,
            character_ids=[FIXED_CHARACTER_ID],
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