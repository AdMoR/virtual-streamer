"""
Atari Planning Agent — vision agent that plans the next 10 moves.

Analyzes the current game frame and outputs a 10-move action plan stored in
ADK session state under the key "atari_plan".

Execution is gated by the session variable PLANNING_NEEDED: when that key is
absent or falsy the before_agent_callback short-circuits the agent entirely,
so the game loop only pays the LLM cost when a new plan is actually required.
"""

import logging
from typing import List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from pydantic import BaseModel, Field

from virtual_streamer.lib.agents import (
    AgentCallback,
    AfterModelCallback,
    BaseLlmAgent,
    BeforeModelCallback,
    extract_llm_response_json,
)
from virtual_streamer.agents.atari_planning_agent.prompt import ATARI_PLANNING_PROMPT

logger = logging.getLogger(__name__)

SESSION_KEY_PLAN = "atari_plan"
SESSION_KEY_PLANNING_NEEDED = "PLANNING_NEEDED"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class AtariPlan(BaseModel):
    moves: List[str] = Field(
        description="Exactly 10 action names chosen from the legal actions list"
    )
    reasoning: str = Field(
        description="Brief tactical explanation for the planned sequence"
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PlanningGateCallback(AgentCallback):
    """Skip the agent when PLANNING_NEEDED is absent or falsy in session state."""

    async def __call__(
        self, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if callback_context.state.get(SESSION_KEY_PLANNING_NEEDED, False):
            return None  # proceed to LLM
        logger.debug("AtariPlanningAgent skipped: PLANNING_NEEDED is not set")
        return types.Content(
            role="model",
            parts=[types.Part(text="Planning skipped: PLANNING_NEEDED is not set.")],
        )


class InjectPlanningFrameCallback(BeforeModelCallback):
    """Inject the current game frame as an image into the LLM request."""

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
            logger.warning("InjectPlanningFrameCallback: no frame set, skipping injection")
            return None
        llm_request.contents[0].parts.append(
            types.Part.from_bytes(data=self._frame, mime_type="image/jpeg")
        )
        return None


class StorePlanCallback(AfterModelCallback):
    """Parse the 10-move plan and store it in ADK session state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        plan = extract_llm_response_json(llm_response, AtariPlan)
        if plan is None:
            logger.warning("StorePlanCallback: failed to parse AtariPlan from response")
            return None
        callback_context.state[SESSION_KEY_PLAN] = plan.model_dump_json()
        # Clear the flag so subsequent moves in the same session skip planning
        callback_context.state[SESSION_KEY_PLANNING_NEEDED] = False
        logger.info(
            f"Stored plan at '{SESSION_KEY_PLAN}': "
            f"moves={plan.moves}, reasoning={plan.reasoning[:80]!r}"
        )
        return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AtariPlanningAgent(BaseLlmAgent):

    def __init__(self):
        _frame_cb = InjectPlanningFrameCallback()
        super().__init__(
            name="atari_planning_agent",
            instruction=ATARI_PLANNING_PROMPT,
            output_schema=AtariPlan,
            before_agent_callback=PlanningGateCallback(),
            before_model_callback=_frame_cb,
            after_model_callback=StorePlanCallback(),
        )

    def set_frame(self, jpeg: bytes) -> None:
        """Update the frame that will be injected on the next LLM call."""
        self.before_model_callback.set_frame(jpeg)


def get_atari_planning_agent() -> AtariPlanningAgent:
    return AtariPlanningAgent()


root_agent = get_atari_planning_agent()