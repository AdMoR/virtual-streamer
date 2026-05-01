"""
Location Builder Agent.

A two-step sequential pipeline:
1. location_writer   — generates a rich free-text diffusion description for the
                       location, using the story template as context.
2. location_formatter — extracts the structured LocationDescriptionOutput field.

State flow:
    Input:  LOCATION_NAME (str) + STORY_TEMPLATE_ID (str)
    Step 1: location_writer    → RAW_LOCATION_TEXT (str)
    Step 2: location_formatter → LOCATION_OUTPUT (dict with description)
"""

import logging

from google.adk.agents import SequentialAgent

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.location_builder.schema import LocationDescriptionOutput
from virtual_streamer.agents.location_builder.prompt import (
    LocationWriterInstructionProvider,
    LocationFormatterInstructionProvider,
)
from virtual_streamer.agents.location_builder.callback import (
    StoreRawLocationCallback,
    StoreLocationOutputCallback,
)

logger = logging.getLogger(__name__)


class LocationWriterAgent(BaseLlmAgent):
    """
    Step 1: generates a rich free-text diffusion description for the location.

    Reads LOCATION_NAME and STORY_TEMPLATE_ID from state, loads the template
    context from DB, and writes the description to RAW_LOCATION_TEXT.
    """

    def __init__(self):
        super().__init__(
            name="location_writer",
            instruction=LocationWriterInstructionProvider(),
            output_schema=None,
            after_model_callback=[StoreRawLocationCallback()],
        )


class LocationFormatterAgent(BaseLlmAgent):
    """
    Step 2: extracts the single description field from the raw location text.

    Reads RAW_LOCATION_TEXT from state and formats it into LocationDescriptionOutput.
    StoreLocationOutputCallback persists the result under LOCATION_OUTPUT.
    """

    def __init__(self):
        super().__init__(
            name="location_formatter",
            instruction=LocationFormatterInstructionProvider(),
            output_schema=LocationDescriptionOutput,
            after_model_callback=[StoreLocationOutputCallback()],
        )


class LocationBuilderAgent(SequentialAgent):
    """
    Sequential agent: location_writer → location_formatter.

    State flow:
        Input:  LOCATION_NAME + STORY_TEMPLATE_ID in session state
        Step 1: location_writer    → RAW_LOCATION_TEXT
        Step 2: location_formatter → LOCATION_OUTPUT
    """

    def __init__(self):
        super().__init__(
            name="location_builder",
            sub_agents=[
                LocationWriterAgent(),
                LocationFormatterAgent(),
            ],
        )


def get_location_builder() -> LocationBuilderAgent:
    """Factory function returning a configured LocationBuilderAgent."""
    return LocationBuilderAgent()


root_agent = get_location_builder()