"""
Video Generation Orchestrator.

Main SequentialAgent that coordinates the entire video generation pipeline:
1. StoryGenerator - generates story with DialogLines from title
2. SentenceVideoMatcher - matches each dialog line to a video

Output: List[DialogLineMatch] with (character, dialog, video) pairs.
"""

from virtual_streamer.agents.orchestrator.agent import (
    get_video_generation_orchestrator,
    create_root_agent,
)

__all__ = [
    "get_video_generation_orchestrator",
    "create_root_agent",
]

