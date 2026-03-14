"""
Story Template Builder Callbacks.

StoreRawTemplateCallback stores the free-text output of template_writer into
state so that template_formatter can read and extract the structured fields.

StoreTemplateOutputCallback stores the final StoryTemplateOutput in state
after the formatter has run.
"""

import logging
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import AfterModelCallback
from virtual_streamer.lib.agents.callbacks import (
    extract_llm_response_text,
    extract_llm_response_json,
)
from virtual_streamer.agents.common.state_keys import RAW_TEMPLATE_TEXT, TEMPLATE_OUTPUT
from virtual_streamer.agents.story_template_builder.schema import StoryTemplateOutput

logger = logging.getLogger(__name__)


class StoreRawTemplateCallback(AfterModelCallback):
    """
    Stores the raw free-text template from template_writer into state.

    The text is stored under RAW_TEMPLATE_TEXT so that template_formatter's
    instruction provider can inject it into the formatting prompt.
    """

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        if not text:
            logger.warning("template_writer returned an empty response")
        callback_context.state[RAW_TEMPLATE_TEXT] = text
        logger.info(f"Stored raw template text ({len(text)} chars) in state")


class StoreTemplateOutputCallback(AfterModelCallback):
    """
    Parses the StoryTemplateOutput from template_formatter and stores it in state.
    """

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        template = extract_llm_response_json(llm_response, StoryTemplateOutput)

        if not template:
            raise Exception("Could not extract StoryTemplateOutput from template_formatter response")

        callback_context.state[TEMPLATE_OUTPUT] = template.model_dump()
        logger.info(f"Stored StoryTemplateOutput '{template.name}' in state")