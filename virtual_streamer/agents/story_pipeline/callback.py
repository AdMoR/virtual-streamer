"""
Story Pipeline Callbacks.

StoreRawStoryCallback stores the free-text output of story_writer into state
so that story_formatter can read and format it into StoryOutput.
"""

import logging
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import AfterModelCallback
from virtual_streamer.lib.agents.callbacks import extract_llm_response_text
from virtual_streamer.agents.common.state_keys import RAW_STORY_TEXT

logger = logging.getLogger(__name__)


class StoreRawStoryCallback(AfterModelCallback):
    """
    Stores the raw free-text story from story_writer into state.

    The text is stored under RAW_STORY_TEXT so that story_formatter's
    instruction provider can inject it into the formatting prompt.
    """

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        if not text:
            logger.warning("story_writer returned an empty response")
        callback_context.state[RAW_STORY_TEXT] = text
        logger.info(f"Stored raw story text ({len(text)} chars) in state")