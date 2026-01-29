"""
Context management for the Virtual Streamer Agent.

This module provides:
- ContextProviderProtocol: Interface for composable context providers
- Mock providers for testing (queue, system, chat)
- Legacy providers (ContextBuilder, ConversationManager)
"""

# Protocol
from virtual_streamer.agents.virtual_streamer_agent.context.protocol import (
    ContextProviderProtocol,
)

# Composable Mock Providers (new pattern)
from virtual_streamer.agents.virtual_streamer_agent.context.queue_provider import (
    MockProcessingQueueContextProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.context.system_provider import (
    MockSystemStatusContextProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.context.chat_provider import (
    MockChatMessageContextProvider,
)

# Legacy imports (for backward compatibility)
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
    # Protocol
    "ContextProviderProtocol",
    
    # Composable Mock Providers (new pattern)
    "MockProcessingQueueContextProvider",
    "MockSystemStatusContextProvider",
    "MockChatMessageContextProvider",
    
    # Legacy - ContextBuilder
    "ContextBuilder",
    "ConversationManagerStrategy",
    "KeepLastN",
    "QueueInfoProvider",
    "WorkloadProvider",
    
    # Legacy - Mock providers
    "MockContextProviders",
    "MockQueueInfoProvider",
    "MockWorkloadProvider",
    "MockChatStore",
    "MockQueueConfig",
    "MockWorkloadConfig",
    "MockChatConfig",
    "MockChatMessage",
]
