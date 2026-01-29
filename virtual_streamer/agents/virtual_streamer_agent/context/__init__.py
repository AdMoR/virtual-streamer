"""
Context management for the Virtual Streamer Agent.

This module provides:
- ContextBuilder: Assembles context from various sources
- ConversationManager: Manages chat history with configurable strategies
- Providers: Fetch queue info, system status, etc.
- Mock providers for testing
"""

from virtual_streamer.agents.virtual_streamer_agent.context.builder import ContextBuilder
from virtual_streamer.agents.virtual_streamer_agent.context.conversation import (
    ConversationManagerStrategy,
    KeepLastN,
)
from virtual_streamer.agents.virtual_streamer_agent.context.providers import (
    QueueInfoProvider,
    WorkloadProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.context.mock_providers import (
    MockContextProviders,
    MockQueueInfoProvider,
    MockWorkloadProvider,
    MockChatStore,
    MockQueueConfig,
    MockWorkloadConfig,
    MockChatConfig,
    MockChatMessage,
)

__all__ = [
    "ContextBuilder",
    "ConversationManagerStrategy",
    "KeepLastN",
    "QueueInfoProvider",
    "WorkloadProvider",
    # Mock providers for testing
    "MockContextProviders",
    "MockQueueInfoProvider",
    "MockWorkloadProvider",
    "MockChatStore",
    "MockQueueConfig",
    "MockWorkloadConfig",
    "MockChatConfig",
    "MockChatMessage",
]
