import json
import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import AfterModelCallback
from virtual_streamer.agents.common.state_keys import STORY_OUTPUT, SENTENCES
from virtual_streamer.agents.story_generator.schema import StoryOutput, DialogLines

logger = logging.getLogger(__name__)


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
