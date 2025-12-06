"""
Virtual Streamer Library - Base classes for Google ADK agents.

This module provides base classes and utilities for building ADK agents
following best practices for configuration, callbacks, and instruction providers.
"""

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.lib.agents.callbacks import (
    AfterModelCallback,
    BeforeModelCallback,
    AgentCallback,
)
from virtual_streamer.lib.config import AgentConfig, get_agent_configuration
from virtual_streamer.lib.providers import InstructionProvider

__all__ = [
    "BaseLlmAgent",
    "AfterModelCallback",
    "BeforeModelCallback",
    "AgentCallback",
    "AgentConfig",
    "get_agent_configuration",
    "InstructionProvider",
]

