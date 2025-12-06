"""
Video Generation Orchestrator.

Main SequentialAgent that coordinates the entire video generation pipeline.
"""

from virtual_streamer.agents.orchestrator.agent import (
    get_video_generation_orchestrator,
    create_root_agent,
    get_root_agent,
)

__all__ = [
    "get_video_generation_orchestrator",
    "create_root_agent",
    "get_root_agent",
]

