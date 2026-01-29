"""
Callbacks for the Virtual Streamer Agent.

This module provides callback implementations for:
- Context injection as tool responses
"""

from virtual_streamer.agents.virtual_streamer_agent.callbacks.context_injector import (
    InjectContextCallback,
)

__all__ = [
    "InjectContextCallback",
]
