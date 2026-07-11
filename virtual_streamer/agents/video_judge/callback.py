"""
Callbacks for VideoJudgeAgent.

InjectCandidateFramesCallback — samples N frames from the candidate video and
                                appends them to the LLM request as vision parts.
StoreVerdictCallback          — parses the JSON verdict into state.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from virtual_streamer.lib.agents import AfterModelCallback, BeforeModelCallback
from virtual_streamer.lib.agents.callbacks import extract_llm_response_json
from virtual_streamer.agents.common.utils import extract_evenly_spaced_frames
from virtual_streamer.agents.video_judge.schema import (
    JUDGE_VIDEO_PATH,
    JUDGE_VERDICT,
    JudgeVerdict,
)

logger = logging.getLogger(__name__)


class InjectCandidateFramesCallback(BeforeModelCallback):
    """Injects N evenly-spaced JPEG frames from the candidate video into the request."""

    def __init__(self, n_frames: int = 8):
        self.n_frames = n_frames

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        video_path = callback_context.state.get(JUDGE_VIDEO_PATH)
        if not video_path:
            logger.warning("[InjectCandidateFramesCallback] No video path in state")
            return None

        frames = extract_evenly_spaced_frames(video_path, self.n_frames)
        if not frames:
            logger.warning(f"[InjectCandidateFramesCallback] No frames extracted from {video_path!r}")
            return None

        for frame_bytes in frames:
            llm_request.contents[0].parts.append(
                types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
            )
        logger.info(
            f"[InjectCandidateFramesCallback] Injected {len(frames)} frame(s) from {video_path!r}"
        )
        return None


class StoreVerdictCallback(AfterModelCallback):
    """Parses the judge JSON output and stores a JudgeVerdict dict in state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        verdict = extract_llm_response_json(llm_response, JudgeVerdict)
        if verdict is None:
            logger.warning("[StoreVerdictCallback] Could not parse judge output as JudgeVerdict")
            verdict = JudgeVerdict.permissive_default("unparseable judge output")
        callback_context.state[JUDGE_VERDICT] = verdict.model_dump()
        logger.info(
            f"[StoreVerdictCallback] verdict: passed={verdict.passed} score={verdict.score} "
            f"artifacts={len(verdict.artifacts)}"
        )
