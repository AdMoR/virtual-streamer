"""
Location Builder Callbacks.

StoreRawLocationCallback stores the free-text output of location_writer into
state so that location_formatter can read and extract the structured field.

StoreLocationOutputCallback stores the final LocationDescriptionOutput in state
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
from virtual_streamer.agents.common.state_keys import RAW_LOCATION_TEXT, LOCATION_OUTPUT
from virtual_streamer.agents.location_builder.schema import LocationDescriptionOutput

logger = logging.getLogger(__name__)


class StoreRawLocationCallback(AfterModelCallback):
    """Stores the raw free-text location description from location_writer into state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        if not text:
            logger.warning("location_writer returned an empty response")
        callback_context.state[RAW_LOCATION_TEXT] = text
        logger.info(f"Stored raw location text ({len(text)} chars) in state")


class StoreLocationOutputCallback(AfterModelCallback):
    """Parses the LocationDescriptionOutput from location_formatter and stores it in state."""

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        output = extract_llm_response_json(llm_response, LocationDescriptionOutput)

        if not output:
            raise Exception(
                "Could not extract LocationDescriptionOutput from location_formatter response"
            )

        callback_context.state[LOCATION_OUTPUT] = output.model_dump()
        logger.info(f"Stored LocationDescriptionOutput ({len(output.description)} chars) in state")