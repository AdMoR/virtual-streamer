"""
Callbacks for SceneEnricherPipeline.

InjectFramesCallback  — extracts N evenly-spaced frames from the reference video
                        and appends them to the LLM request (shared by both sub-agents).
StoreDescriptionCallback — stores the video description (Agent 1 output) in state.
StoreEnrichedSceneCallback — stores the enriched scene text (Agent 2 output) in state.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from virtual_streamer.lib.agents import AfterModelCallback, BeforeModelCallback
from virtual_streamer.lib.agents.callbacks import extract_llm_response_text
from virtual_streamer.agents.common.state_keys import (
    ENRICHMENT_VIDEO_PATH,
    ENRICHMENT_VIDEO_DESCRIPTION,
    ENRICHED_SCENE,
)
from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames

logger = logging.getLogger(__name__)


class InjectFramesCallback(BeforeModelCallback):
    """
    Extracts N evenly-spaced JPEG frames from the reference video in state
    and injects them into the LLM request as vision parts.

    Used as before_model_callback by both VideoDescriptionAgent and SceneEnrichmentAgent,
    following the same pattern as InjectVisionFrameCallback in video_matcher/callback.py.
    """

    def __init__(self, n_frames: int = 4):
        self.n_frames = n_frames

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        video_path = callback_context.state.get(ENRICHMENT_VIDEO_PATH)
        if not video_path:
            logger.warning("[InjectFramesCallback] No video path in state; skipping frame injection")
            return None

        frames = extract_evenly_spaced_frames(video_path, self.n_frames)
        if not frames:
            logger.warning(f"[InjectFramesCallback] No frames extracted from {video_path!r}")
            return None

        for frame_bytes in frames:
            llm_request.contents[0].parts.append(
                types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
            )

        logger.info(f"[InjectFramesCallback] Injected {len(frames)} frame(s) from {video_path!r}")
        return None


class StoreDescriptionCallback(AfterModelCallback):
    """Stores the video description text (Agent 1 output) in state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        if not text:
            logger.warning("[StoreDescriptionCallback] Empty description from VideoDescriptionAgent")
        callback_context.state[ENRICHMENT_VIDEO_DESCRIPTION] = text
        logger.info(f"[StoreDescriptionCallback] Stored video description ({len(text)} chars)")


class StoreEnrichedSceneCallback(AfterModelCallback):
    """Stores the enriched scene text (Agent 2 output) in state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        if not text:
            logger.warning("[StoreEnrichedSceneCallback] Empty output from SceneEnrichmentAgent")
        callback_context.state[ENRICHED_SCENE] = text
        logger.info(f"[StoreEnrichedSceneCallback] Stored enriched scene ({len(text)} chars)")
