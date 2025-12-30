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
from virtual_streamer.agents.story_generator.callback import SplitStoryCallback
from virtual_streamer.agents.story_generator.prompt import format_story_prompt, StoryInstructionProvider
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.lib.agents import BaseLlmAgent


logger = logging.getLogger(__name__)


class StoryGeneratorAgent(BaseLlmAgent):
    """
    Agent that generates a story from a title.
    
    Uses:
    - StoryInstructionProvider to read title from state and format prompt
    - StoryOutput schema for structured output
    - SplitStoryCallback to parse response and split into sentences
    """

    def __init__(self):
        super().__init__(
            name="story_generator",
            instruction=StoryInstructionProvider(),
            output_schema=StoryOutput,
            after_model_callback=[SplitStoryCallback()],
        )


def get_story_generator() -> StoryGeneratorAgent:
    """
    Factory function to get the StoryGeneratorAgent singleton.
    
    Returns:
        Configured StoryGeneratorAgent instance
    """
    return StoryGeneratorAgent()


root_agent = get_story_generator()
