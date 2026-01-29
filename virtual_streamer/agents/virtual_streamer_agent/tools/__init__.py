"""
Tools for the Virtual Streamer Agent.

This module provides:
- Tool implementations (create_video, send_message)
- ToolFactory for config-based tool loading
- MockToolFactory for testing
"""

from virtual_streamer.agents.virtual_streamer_agent.tools.factory import ToolFactory
from virtual_streamer.agents.virtual_streamer_agent.tools.create_video import create_video
from virtual_streamer.agents.virtual_streamer_agent.tools.send_message import send_twitch_message
from virtual_streamer.agents.virtual_streamer_agent.tools.mock import (
    MockToolFactory,
    MockToolFactoryConfig,
    MockToolConfig,
    ToolCallLog,
    ToolCall,
)

__all__ = [
    "ToolFactory",
    "create_video",
    "send_twitch_message",
    # Mock tools for testing
    "MockToolFactory",
    "MockToolFactoryConfig",
    "MockToolConfig",
    "ToolCallLog",
    "ToolCall",
]
