"""
High-level API: Story Pipeline Agent Testing

Exposes the StoryPipelineAgent as a testable REST endpoint.

  Input : title (str), optional news_context (str)
  Output: raw_story_text, recurrent_locations, detailed_scenes
"""

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from virtual_streamer.agents.common.state_keys import (
    TITLE,
    STORY_TEMPLATE_ID,
    NEWS_CONTEXT,
    RAW_STORY_TEXT,
    RECURRENT_LOCATIONS,
    DETAILED_SCENES,
    SECURITY_FLAG,
)
from virtual_streamer.agents.guardrails_agent.schema import GuardrailFlag
from virtual_streamer.agents.story_pipeline.agent import get_story_pipeline
from virtual_streamer.agents.story_pipeline.schema import (
    RecurrentLocationsOutput,
    DetailedScenesOutput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/story-pipeline", tags=["Story Pipeline"])

APP_NAME = "story_pipeline_test"


# =============================================================================
# Request / Response Models
# =============================================================================


class StoryPipelineRequest(BaseModel):
    title: str = Field(
        ...,
        description="Story title — the seed for the full pipeline.",
        examples=["Fred se lance dans l'IA"],
    )
    story_template_id: Optional[str] = Field(
        None,
        description="Story template ID to customise generation style and character roster.",
    )
    news_context: Optional[str] = Field(
        None,
        description=(
            "Optional news context injected into the prompt. "
            "Format: 'Titre: ...\\nRésumé: ...\\nSource: ...\\nDate: ...'"
        ),
    )


class StoryPipelineResponse(BaseModel):
    title: str
    raw_story_text: str
    recurrent_locations: Any
    detailed_scenes: Any


# =============================================================================
# ADK Runner Helper
# =============================================================================


async def _run_story_pipeline(
    title: str,
    news_context: Optional[str],
    story_template_id: Optional[str] = None,
) -> StoryPipelineResponse:
    logger.info(
        f"[story-pipeline] start  title={title!r}  "
        f"story_template_id={story_template_id!r}  "
        f"news_context={'yes' if news_context else 'no'}"
    )

    agent = get_story_pipeline()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"sp_{uuid.uuid4().hex[:8]}"

    initial_state: dict = {TITLE: title}
    if story_template_id:
        initial_state[STORY_TEMPLATE_ID] = story_template_id
    if news_context:
        initial_state[NEWS_CONTEXT] = news_context

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    logger.info(f"[story-pipeline] session created  session_id={session_id}")

    content = types.Content(role="user", parts=[types.Part(text=title)])

    event_count = 0
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        event_count += 1
        if event.is_final_response():
            logger.info(f"[story-pipeline] final response from agent={event.author!r}")

    logger.info(f"[story-pipeline] pipeline finished  events={event_count}")

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    state = session.state
    state_keys = list(state.keys())
    logger.info(f"[story-pipeline] state keys present: {state_keys}")

    security_flag = state.get(SECURITY_FLAG)
    if security_flag:
        # GuardrailsOutput stores {flag: GuardrailFlag} — only block on MALICIOUS
        raw_flag = security_flag
        if isinstance(raw_flag, dict):
            flag_value = raw_flag.get("flag")
        else:
            flag_value = getattr(raw_flag, "flag", raw_flag)

        is_malicious = (
            flag_value == GuardrailFlag.MALICIOUS
            or flag_value == "MALICIOUS"
        )
        logger.info(f"[story-pipeline] guardrail flag={flag_value!r}  blocked={is_malicious}")
        if is_malicious:
            raise RuntimeError(f"Content blocked by guardrails: {flag_value}")

    raw_story = state.get(RAW_STORY_TEXT)
    if not raw_story:
        logger.error(
            f"[story-pipeline] RAW_STORY_TEXT missing from state. "
            f"Present keys: {state_keys}"
        )
        raise RuntimeError(
            f"StoryPipelineAgent completed but RAW_STORY_TEXT is missing. "
            f"State keys: {state_keys}"
        )

    logger.info(f"[story-pipeline] raw_story  chars={len(raw_story)}")

    raw_locations = state.get(RECURRENT_LOCATIONS)
    raw_scenes = state.get(DETAILED_SCENES)
    logger.info(
        f"[story-pipeline] recurrent_locations present={raw_locations is not None}  "
        f"detailed_scenes present={raw_scenes is not None}"
    )

    locations_dict = None
    if raw_locations:
        try:
            if isinstance(raw_locations, str):
                locations_dict = RecurrentLocationsOutput.model_validate_json(raw_locations).model_dump()
            else:
                locations_dict = RecurrentLocationsOutput.model_validate(raw_locations).model_dump()
            logger.info(f"[story-pipeline] parsed {len(locations_dict.get('locations', []))} location(s)")
        except Exception as exc:
            logger.error(f"[story-pipeline] failed to parse recurrent_locations: {exc}", exc_info=True)
            logger.debug(f"[story-pipeline] raw_locations value: {str(raw_locations)[:500]}")
            raise RuntimeError(f"Failed to parse recurrent_locations: {exc}") from exc

    scenes_dict = None
    if raw_scenes:
        try:
            if isinstance(raw_scenes, str):
                scenes_dict = DetailedScenesOutput.model_validate_json(raw_scenes).model_dump()
            else:
                scenes_dict = DetailedScenesOutput.model_validate(raw_scenes).model_dump()
            logger.info(f"[story-pipeline] parsed {len(scenes_dict.get('scenes', []))} scene(s)")
        except Exception as exc:
            logger.error(f"[story-pipeline] failed to parse detailed_scenes: {exc}", exc_info=True)
            logger.debug(f"[story-pipeline] raw_scenes value: {str(raw_scenes)[:500]}")
            raise RuntimeError(f"Failed to parse detailed_scenes: {exc}") from exc

    logger.info("[story-pipeline] done — returning response")
    return StoryPipelineResponse(
        title=title,
        raw_story_text=raw_story,
        recurrent_locations=locations_dict,
        detailed_scenes=scenes_dict,
    )


# =============================================================================
# API Endpoint
# =============================================================================


@router.post("/run", response_model=StoryPipelineResponse)
async def run_story_pipeline(request: StoryPipelineRequest):
    """
    Run the full StoryPipelineAgent and return all intermediate outputs.

    Executes the three-step pipeline:
    1. **story_writer** — generates free-text narrative from the title
    2. **recurrent_location_builder** — extracts recurring locations with Flux prompts
    3. **detailed_scene_builder** — produces one DetailedScene per scene

    Nothing is persisted; this endpoint is intended for testing and inspection.
    """
    try:
        return await _run_story_pipeline(request.title, request.news_context, request.story_template_id)
    except RuntimeError as e:
        logger.error(f"[story-pipeline] RuntimeError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[story-pipeline] unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
