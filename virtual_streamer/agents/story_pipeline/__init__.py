"""
Story Pipeline Agent.

Two-step sequential pipeline:
1. StoryWriterAgent   — generates free-text story (same prompt as StoryGeneratorAgent)
2. StoryFormatterAgent — formats raw text into structured StoryOutput
"""

from virtual_streamer.agents.story_pipeline.agent import (
    StoryPipelineAgent,
    StoryWriterAgent,
    StoryFormatterAgent,
    get_story_pipeline,
)

__all__ = [
    "StoryPipelineAgent",
    "StoryWriterAgent",
    "StoryFormatterAgent",
    "get_story_pipeline",
]