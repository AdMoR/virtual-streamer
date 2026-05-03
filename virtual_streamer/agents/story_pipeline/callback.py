"""
Story Pipeline Callbacks.

StoreRawStoryCallback         — stores free-text output of story_writer (unchanged)
StoreRecurrentLocationsCallback — stores RecurrentLocationsOutput after location builder
StoreDetailedScenesCallback     — stores DetailedScenesOutput after scene builder
"""

import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import AfterModelCallback
from virtual_streamer.lib.agents.callbacks import extract_llm_response_text, extract_llm_response_json
from virtual_streamer.agents.common.state_keys import (
    RAW_STORY_TEXT,
    RECURRENT_LOCATIONS,
    DETAILED_SCENES,
)
from virtual_streamer.agents.story_pipeline.schema import (
    RecurrentLocationsOutput,
    DetailedScenesOutput,
)

logger = logging.getLogger(__name__)


class StoreRawStoryCallback(AfterModelCallback):
    """
    Stores the raw free-text story from story_writer into state.

    The text is stored under RAW_STORY_TEXT so that downstream agents'
    instruction providers can inject it into their prompts.
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


class StoreRecurrentLocationsCallback(AfterModelCallback):
    """
    Parses RecurrentLocationsOutput from the location builder and stores it in state as JSON.

    Stored as a JSON string under RECURRENT_LOCATIONS so that
    DetailedSceneBuilderInstructionProvider can read and inject it into the prompt,
    and the API can deserialize it into a typed object via get_recurrent_locations_from_state().
    """

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        output = extract_llm_response_json(llm_response, RecurrentLocationsOutput)
        if not output:
            raise Exception(
                "Could not extract RecurrentLocationsOutput from recurrent_location_builder response"
            )
        callback_context.state[RECURRENT_LOCATIONS] = output.model_dump_json()
        logger.info(
            f"Stored {len(output.locations)} recurrent location(s) in state"
        )


class StoreDetailedScenesCallback(AfterModelCallback):
    """
    Parses DetailedScenesOutput from the scene builder and stores it in state as JSON.

    Stored as a JSON string under DETAILED_SCENES.
    Use get_detailed_scenes_from_state() to deserialize in the API.
    """

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        output = extract_llm_response_json(llm_response, DetailedScenesOutput)
        if not output:
            raise Exception(
                "Could not extract DetailedScenesOutput from detailed_scene_builder response"
            )
        callback_context.state[DETAILED_SCENES] = output.model_dump_json()
        logger.info(
            f"Stored {len(output.scenes)} detailed scene(s) in state"
        )
