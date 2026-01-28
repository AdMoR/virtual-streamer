"""
Virtual Streamer Agent.

ADK agent that controls a Twitch streaming channel by:
- Monitoring chat messages and responding to viewers
- Creating videos based on viewer requests or proactively
- Managing the video queue to ensure fresh content

This agent uses tools defined in the tools/ submodule and builds
context from the context/ submodule.
"""

from virtual_streamer.agents.virtual_streamer.agent import (
    VirtualStreamerAgent,
    get_virtual_streamer_agent,
)

__all__ = [
    "VirtualStreamerAgent",
    "get_virtual_streamer_agent",
]
