"""
Video Judge Agent.

Judges one generated video segment for generation artifacts (unrealistic
movements, impossible bodies/settings, identity drift, glitches) using the
local vision LLM configured in configs/agents/video_judge.yaml.

Usage:
    from virtual_streamer.agents.video_judge.agent import run_video_judge

    verdict = await run_video_judge(
        video_path="/path/to/candidate.mp4",
        scene_description="Fred walks toward the camera in a workshop...",
    )
    if not verdict.passed:
        ...  # try another seed
"""

import logging
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.video_judge.callback import (
    InjectCandidateFramesCallback,
    StoreVerdictCallback,
)
from virtual_streamer.agents.video_judge.prompt import JudgeInstructionProvider
from virtual_streamer.agents.video_judge.schema import (
    JUDGE_SCENE_DESCRIPTION,
    JUDGE_VERDICT,
    JUDGE_VIDEO_PATH,
    JudgeVerdict,
)

logger = logging.getLogger(__name__)

_APP_NAME = "video_judge"


class VideoJudgeAgent(BaseLlmAgent):
    """Single-step vision agent producing a JudgeVerdict for a candidate video."""

    def __init__(self, n_frames: int = 8):
        super().__init__(
            name="video_judge",
            instruction=JudgeInstructionProvider(),
            output_schema=None,  # verdict parsed by StoreVerdictCallback (local models are unreliable with native structured output)
            before_model_callback=[InjectCandidateFramesCallback(n_frames=n_frames)],
            after_model_callback=[StoreVerdictCallback()],
        )


def get_video_judge_agent() -> VideoJudgeAgent:
    """Factory function returning a configured VideoJudgeAgent."""
    return VideoJudgeAgent()


async def run_video_judge(video_path: str, scene_description: str) -> JudgeVerdict:
    """
    Judge one candidate video segment.

    Never raises — on any failure returns JudgeVerdict.permissive_default so
    that a broken judge can never block video generation.
    """
    try:
        agent = get_video_judge_agent()
    except Exception as exc:
        logger.error(f"[run_video_judge] Could not build judge agent: {exc}")
        return JudgeVerdict.permissive_default(f"agent init failed: {exc}")

    session_service = InMemorySessionService()
    user_id = f"judge_{uuid.uuid4().hex[:8]}"
    session_id = f"run_{uuid.uuid4().hex[:8]}"

    try:
        session = await session_service.create_session(
            app_name=_APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state={
                JUDGE_VIDEO_PATH: video_path,
                JUDGE_SCENE_DESCRIPTION: scene_description,
            },
        )
        runner = Runner(agent=agent, app_name=_APP_NAME, session_service=session_service)
        content = types.Content(
            role="user",
            parts=[types.Part(text="Judge the following video frames against the scene description.")],
        )
        async for _ in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            pass

        final_session = await session_service.get_session(
            app_name=_APP_NAME, user_id=user_id, session_id=session_id
        )
        raw = final_session.state.get(JUDGE_VERDICT)
        if not raw:
            return JudgeVerdict.permissive_default("no verdict in final state")
        return JudgeVerdict.model_validate(raw)
    except Exception as exc:
        logger.error(f"[run_video_judge] Judge run failed: {exc}", exc_info=True)
        return JudgeVerdict.permissive_default(str(exc))


root_agent = get_video_judge_agent()
