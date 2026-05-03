"""
Story Pipeline Agent.

Three-step sequential pipeline:
1. StoryWriterAgent               — generates free-text story
2. RecurrentLocationBuilderAgent  — extracts recurring locations with FluxPrompts
3. DetailedSceneBuilderAgent      — produces one DetailedScene per scene

StoryFormatterAgent is kept for the legacy Wav2Lip pipeline.
"""

from virtual_streamer.agents.story_pipeline.agent import (
    StoryPipelineAgent,
    StoryWriterAgent,
    StoryFormatterAgent,
    RecurrentLocationBuilderAgent,
    DetailedSceneBuilderAgent,
    get_story_pipeline,
)

__all__ = [
    "StoryPipelineAgent",
    "StoryWriterAgent",
    "StoryFormatterAgent",
    "RecurrentLocationBuilderAgent",
    "DetailedSceneBuilderAgent",
    "get_story_pipeline",
]
