"""
Atari Action Agent — vision agent that chooses the next game action
from a screenshot of the current frame.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types
from pydantic import BaseModel

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.atari_action_agent.prompt import ATARI_ACTION_PROMPT

logger = logging.getLogger(__name__)


class AtariActionOutput(BaseModel):
    action: str       # must match a legal action name exactly
    reasoning: str


class InjectGameFrameCallback:
    """BeforeModelCallback that injects the current frame as an image/jpeg Part."""

    def __init__(self):
        self._frame: Optional[bytes] = None

    def set_frame(self, jpeg: bytes) -> None:
        self._frame = jpeg

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        if self._frame is None:
            logger.warning("InjectGameFrameCallback: no frame set, skipping injection")
            return None
        llm_request.contents[0].parts.append(
            types.Part.from_bytes(data=self._frame, mime_type="image/jpeg")
        )
        return None


class AtariActionAgent(BaseLlmAgent):

    def __init__(self):
        self._cb = InjectGameFrameCallback()
        super().__init__(
            name="atari_action_agent",
            instruction=ATARI_ACTION_PROMPT,
            output_schema=AtariActionOutput,
            before_model_callback=self._cb,
        )

    def set_frame(self, jpeg: bytes) -> None:
        self._cb.set_frame(jpeg)


def get_atari_action_agent() -> AtariActionAgent:
    return AtariActionAgent()
