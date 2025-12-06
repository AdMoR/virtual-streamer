"""
ADK Agents for Video Generation.

This module provides Google ADK agents for the video generation workflow:
- StoryGeneratorAgent: Generates story from title using LLM
- VideoMatcherAgent: Judges video-dialogue match using vision LLM
- KeywordGeneratorAgent: Generates search keywords
- SentenceProcessorAgent: Orchestrates sentence-level processing
- VideoGenerationOrchestrator: Main pipeline orchestrator
"""

from virtual_streamer.agents.orchestrator.agent import (
    get_video_generation_orchestrator,
)
from virtual_streamer.agents.story_generator.agent import (
    StoryGeneratorAgent,
    get_story_generator,
)
from virtual_streamer.agents.video_matcher.agent import (
    get_video_matcher,
)
from virtual_streamer.agents.keyword_generator.agent import (
    get_keyword_generator,
)
from virtual_streamer.agents.sentence_processor.agent import (
    SentenceProcessorAgent,
)
from virtual_streamer.agents.common.state_keys import (
    TITLE,
    CONFIG,
    STORY_OUTPUT,
    SENTENCES,
    VIDEO_MATCHES,
    AUDIO_FILES,
    SUBTITLE_FILES,
    VIDEO_SEGMENTS,
    FINAL_VIDEO_PATH,
)

__all__ = [
    # Orchestrator
    "get_video_generation_orchestrator",
    # Agents
    "StoryGeneratorAgent",
    "get_story_generator",
    "get_video_matcher",
    "get_keyword_generator",
    "SentenceProcessorAgent",
    # State keys
    "TITLE",
    "CONFIG",
    "STORY_OUTPUT",
    "SENTENCES",
    "VIDEO_MATCHES",
    "AUDIO_FILES",
    "SUBTITLE_FILES",
    "VIDEO_SEGMENTS",
    "FINAL_VIDEO_PATH",
]

