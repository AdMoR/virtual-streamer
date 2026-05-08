"""
Scene Enricher Pipeline.

A two-step SequentialAgent that analyses reference video frames and enriches
a scene description (ltx_prompt) with precise movement and action details.

State flow:
    Input:  enrichment_video_path (str) — path to reference video
            enrichment_scene_text (str) — original scene text to enrich
    Step 1: VideoDescriptionAgent  → enrichment_video_description (str)
    Step 2: SceneEnrichmentAgent   → enriched_scene (str)
"""

import logging
import uuid

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.scene_enricher.callback import (
    InjectFramesCallback,
    StoreDescriptionCallback,
    StoreEnrichedSceneCallback,
)
from virtual_streamer.agents.scene_enricher.prompt import (
    DESCRIBE_PROMPT,
    SceneEnrichmentInstructionProvider,
)
from virtual_streamer.agents.common.state_keys import (
    ENRICHMENT_VIDEO_PATH,
    ENRICHMENT_SCENE_TEXT,
    ENRICHED_SCENE,
)

logger = logging.getLogger(__name__)

_APP_NAME = "scene_enricher"


class VideoDescriptionAgent(BaseLlmAgent):
    """
    Step 1: injects N video frames and asks the model to describe the action.

    Output stored in state["enrichment_video_description"].
    """

    def __init__(self):
        super().__init__(
            name="video_description_agent",
            instruction=DESCRIBE_PROMPT,
            output_schema=None,
            before_model_callback=[InjectFramesCallback()],
            after_model_callback=[StoreDescriptionCallback()],
        )


class SceneEnrichmentAgent(BaseLlmAgent):
    """
    Step 2: injects N video frames again + uses the description from step 1
    to enrich the scene text.

    Output stored in state["enriched_scene"].
    """

    def __init__(self):
        super().__init__(
            name="scene_enrichment_agent",
            instruction=SceneEnrichmentInstructionProvider(),
            output_schema=None,
            before_model_callback=[InjectFramesCallback()],
            after_model_callback=[StoreEnrichedSceneCallback()],
        )


class SceneEnricherPipeline(SequentialAgent):
    """
    Sequential pipeline: VideoDescriptionAgent → SceneEnrichmentAgent.

    Requires state keys:
        enrichment_video_path  — path to the reference video file
        enrichment_scene_text  — the original ltx_prompt / scene description
    """

    def __init__(self):
        super().__init__(
            name="scene_enricher_pipeline",
            sub_agents=[
                VideoDescriptionAgent(),
                SceneEnrichmentAgent(),
            ],
        )


def get_scene_enricher_pipeline() -> SceneEnricherPipeline:
    """Factory function returning a configured SceneEnricherPipeline."""
    return SceneEnricherPipeline()


async def run_scene_enricher(video_path: str, scene_text: str) -> str:
    """
    Run the SceneEnricherPipeline programmatically.

    Args:
        video_path: Absolute path to the reference video file.
        scene_text: The original scene text (ltx_prompt) to enrich.

    Returns:
        Enriched scene text, or the original scene_text if the pipeline fails.
    """
    pipeline = get_scene_enricher_pipeline()
    session_service = InMemorySessionService()

    user_id = f"enricher_{uuid.uuid4().hex[:8]}"
    session_id = f"run_{uuid.uuid4().hex[:8]}"

    initial_state = {
        ENRICHMENT_VIDEO_PATH: video_path,
        ENRICHMENT_SCENE_TEXT: scene_text,
    }

    session = await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    runner = Runner(
        agent=pipeline,
        app_name=_APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=scene_text)])

    try:
        async for _ in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            pass
    except Exception as exc:
        logger.error(f"[run_scene_enricher] Pipeline failed: {exc}")
        return scene_text

    final_session = await session_service.get_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    enriched = final_session.state.get(ENRICHED_SCENE)
    if not enriched:
        logger.warning("[run_scene_enricher] No enriched scene in final state; returning original")
        return scene_text

    return enriched


root_agent = get_scene_enricher_pipeline()
