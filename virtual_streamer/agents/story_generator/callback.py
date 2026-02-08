import json
import logging

from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

from virtual_streamer.lib.agents import AfterModelCallback, AgentCallback
from virtual_streamer.agents.common.state_keys import STORY_OUTPUT, SENTENCES
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.agents.common.state_keys import SECURITY_FLAG
from virtual_streamer.agents.guardrails_agent.schema import GuardrailsOutput, GuardrailFlag

logger = logging.getLogger(__name__)


class SafetyFlagCheckerCallback(AgentCallback):

    async def __call__(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        security_flag_json = callback_context.state.get(SECURITY_FLAG)

        if not security_flag_json:
            return None
        else:
            print(type(security_flag_json), security_flag_json)
            security_output = GuardrailsOutput.model_validate(security_flag_json)
            security_flag = security_output.flag
            if security_flag == GuardrailFlag.MALICIOUS:
                return types.Content(parts=[])
            else:
                return None



class SplitStoryCallback(AfterModelCallback):
    """
    Callback that parses the LLM response and stores dialog lines.

    This callback:
    1. Extracts the structured StoryOutput from the LLM response
    2. Stores the full story output and DialogLines in state
    
    The consumer agent (SentenceVideoMatcher) handles unpacking
    DialogLines into individual DialogLine objects.
    """

    async def __call__(
            self,
            callback_context: CallbackContext,
            llm_response: LlmResponse,
    ) -> None:
        """
        Parse story and store DialogLines in state.

        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the LLM
        """
        from virtual_streamer.lib.agents.callbacks import extract_llm_response_json

        # Parse the structured output into StoryOutput model
        story = extract_llm_response_json(llm_response, StoryOutput)

        if not story:
            raise Exception("Could not extract StoryOutput from response")

        # Store the full story output (as dict for state serialization)
        callback_context.state[STORY_OUTPUT] = story.model_dump()

        # Store DialogLines as serialized dict (consumer agent will parse back)
        callback_context.state[SENTENCES] = story.model_dump()["dialog"]

        logger.info(
            f"Generated story '{story.title}' with {len(story.dialog)} dialog lines"
        )
