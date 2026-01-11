"""
Story Generator Agent.

Generates structured story from a title using LLM.
This is a standard BaseLlmAgent with:
- InstructionProvider that reads TITLE from state
- AfterModelCallback that splits dialog into sentences
"""

import logging
from functools import lru_cache

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.agents.common.state_keys import TITLE
from virtual_streamer.agents.rubric_builder_agent.prompt import StoryInstructionProvider
from virtual_streamer.agents.rubric_builder_agent.schema import MapPhaseOutput
from virtual_streamer.lib.agents import BaseLlmAgent

logger = logging.getLogger(__name__)


class RubricBuilderAgent(BaseLlmAgent):
    """
    Agent that generates a story from a title.

    Uses:
    - StoryInstructionProvider to read title from state and format prompt
    - StoryOutput schema for structured output
    - SplitStoryCallback to parse response and split into sentences
    """

    def __init__(self):
        super().__init__(
            name="rubric_builder",
            instruction=StoryInstructionProvider(),
            output_schema=MapPhaseOutput,
            after_model_callback=[],
        )


def get_rubric_builder() -> RubricBuilderAgent:
    """
    Factory function to get the StoryGeneratorAgent singleton.

    Returns:
        Configured StoryGeneratorAgent instance
    """
    return RubricBuilderAgent()


root_agent = get_rubric_builder()
