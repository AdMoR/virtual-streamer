"""
Context management for the Virtual Streamer Agent.

This module provides:
- ContextBuilder: Assembles context from various sources
- ConversationManager: Manages chat history with configurable strategies
- Providers: Fetch queue info, system status, etc.
"""

from virtual_streamer.agents.virtual_streamer.context.builder import ContextBuilder
from virtual_streamer.agents.virtual_streamer.context.conversation import (
    ConversationManagerStrategy,
    KeepLastN,
)
from virtual_streamer.agents.virtual_streamer.context.providers import (
    QueueInfoProvider,
    WorkloadProvider,
)

__all__ = [
    "ContextBuilder",
    "ConversationManagerStrategy",
    "KeepLastN",
    "QueueInfoProvider",
    "WorkloadProvider",
]
