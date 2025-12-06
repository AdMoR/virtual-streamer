"""
Story Generator Agent.

Generates structured story from a title using LLM.
"""

from virtual_streamer.agents.story_generator.agent import (
    StoryGeneratorAgent,
    get_story_generator,
)

__all__ = ["StoryGeneratorAgent", "get_story_generator"]

