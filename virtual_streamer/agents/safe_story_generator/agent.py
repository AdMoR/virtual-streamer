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
from google.adk.agents.sequential_agent import SequentialAgent

from virtual_streamer.agents.common.state_keys import TITLE
from virtual_streamer.agents.story_generator.agent import get_story_generator
from virtual_streamer.agents.guardrails_agent.agent import get_story_generation_guardrail_agent

logger = logging.getLogger(__name__)


class SafeStoryGeneratorAgent(SequentialAgent):
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
            sub_agents=[
                get_story_generation_guardrail_agent(), get_story_generator()
            ]
        )


def get_safe_story_generator() -> SafeStoryGeneratorAgent:
    """
    Factory function to get the StoryGeneratorAgent singleton.

    Returns:
        Configured StoryGeneratorAgent instance
    """
    return SafeStoryGeneratorAgent()


root_agent = get_safe_story_generator()
