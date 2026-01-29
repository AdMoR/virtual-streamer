"""
Mock context providers for testing the Virtual Streamer Agent.

This module provides mock implementations of context providers that can be
configured to simulate various scenarios for testing agent behavior.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from virtual_streamer.agents.virtual_streamer_agent.schema import (
    ChatMessage,
    QueueInfo,
    SystemStatus,
    WorkloadStatus,
)
from virtual_streamer.agents.common.state_keys import (
    STATE_CHAT_MESSAGES,
    STATE_QUEUE_INFO,
    STATE_SYSTEM_STATUS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Configuration Data Classes
# =============================================================================

@dataclass
class MockQueueConfig:
    """Configuration for mock queue info."""
    
    pending_count: int = 5
    played_count: int = 10
    next_videos: List[str] = field(default_factory=lambda: [
        "Fred se lance dans l'IA",
        "Pourquoi les chats retombent sur leurs pattes",
        "Les mystères du fromage français",
    ])
    is_replaying: bool = False
    active_jobs: int = 0


@dataclass
class MockWorkloadConfig:
    """Configuration for mock system status."""
    
    workload: WorkloadStatus = WorkloadStatus.LOW
    active_jobs: int = 0
    queue_pending: int = 5


@dataclass
class MockChatMessage:
    """A mock chat message with relative timestamp support."""
    
    username: str
    message: str
    is_mention: bool = False
    minutes_ago: float = 0.0  # How many minutes ago this message was sent


@dataclass
class MockChatConfig:
    """Configuration for mock chat messages."""
    
    messages: List[MockChatMessage] = field(default_factory=list)
    time_offset_minutes: float = 0.0  # Global time offset for all messages
    
    @classmethod
    def with_default_conversation(cls) -> "MockChatConfig":
        """Create a config with a default sample conversation."""
        return cls(messages=[
            MockChatMessage(
                username="viewer42",
                message="Salut le stream ! 👋",
                minutes_ago=10.0,
            ),
            MockChatMessage(
                username="science_fan",
                message="J'adore vos vidéos sur l'espace !",
                minutes_ago=8.0,
            ),
            MockChatMessage(
                username="curious_bob",
                message="@virtualstreamer tu pourrais faire une vidéo sur les dinosaures ?",
                is_mention=True,
                minutes_ago=5.0,
            ),
            MockChatMessage(
                username="regular_viewer",
                message="Trop bien la dernière vidéo !",
                minutes_ago=3.0,
            ),
            MockChatMessage(
                username="new_viewer",
                message="C'est quoi cette chaîne ?",
                minutes_ago=1.0,
            ),
        ])


# =============================================================================
# Mock Providers
# =============================================================================

class MockQueueInfoProvider:
    """
    Mock provider for queue information.
    
    Returns configurable QueueInfo data without making API calls.
    """
    
    def __init__(self, config: Optional[MockQueueConfig] = None):
        """
        Initialize the provider.
        
        Args:
            config: Optional configuration for queue info
        """
        self.config = config or MockQueueConfig()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    async def get_queue_info(self, programmation_id: str = "mock") -> QueueInfo:
        """
        Get mock queue information.
        
        Args:
            programmation_id: Ignored in mock, but kept for API compatibility
            
        Returns:
            Configured QueueInfo
        """
        return QueueInfo(
            pending_count=self.config.pending_count,
            played_count=self.config.played_count,
            next_videos=self.config.next_videos.copy(),
            is_replaying=self.config.is_replaying,
            active_jobs=self.config.active_jobs,
        )


class MockWorkloadProvider:
    """
    Mock provider for system workload status.
    
    Returns configurable SystemStatus data.
    """
    
    def __init__(self, config: Optional[MockWorkloadConfig] = None):
        """
        Initialize the provider.
        
        Args:
            config: Optional configuration for workload
        """
        self.config = config or MockWorkloadConfig()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    async def get_system_status(self) -> SystemStatus:
        """
        Get mock system status.
        
        Returns:
            Configured SystemStatus
        """
        return SystemStatus(
            workload=self.config.workload,
            active_jobs=self.config.active_jobs,
            queue_pending=self.config.queue_pending,
        )


class MockChatStore:
    """
    Mock chat store with configurable conversation history.
    
    Supports:
    - Static conversation with relative timestamps
    - Adding new messages dynamically
    - Time offset to simulate "stale" conversations
    """
    
    def __init__(self, config: Optional[MockChatConfig] = None):
        """
        Initialize the chat store.
        
        Args:
            config: Optional configuration for chat messages
        """
        self.config = config or MockChatConfig.with_default_conversation()
        self._dynamic_messages: List[ChatMessage] = []
    
    def set_time_offset(self, minutes: float) -> None:
        """
        Set global time offset for all messages.
        
        Args:
            minutes: How many minutes to offset timestamps into the past
        """
        self.config.time_offset_minutes = minutes
    
    def add_message(
        self,
        username: str,
        message: str,
        is_mention: bool = False,
    ) -> ChatMessage:
        """
        Add a new message to the chat (at current time).
        
        Args:
            username: Username of the sender
            message: Message content
            is_mention: Whether this mentions the bot
            
        Returns:
            The created ChatMessage
        """
        chat_msg = ChatMessage(
            timestamp=datetime.now().isoformat(),
            username=username,
            message=message,
            is_mention=is_mention,
        )
        self._dynamic_messages.append(chat_msg)
        logger.info(f"Added chat message from @{username}: {message[:50]}...")
        return chat_msg
    
    def get_messages(self) -> List[ChatMessage]:
        """
        Get all chat messages with computed timestamps.
        
        Returns:
            List of ChatMessage objects with proper timestamps
        """
        messages = []
        now = datetime.now()
        base_offset = timedelta(minutes=self.config.time_offset_minutes)
        
        # Add configured static messages
        for mock_msg in self.config.messages:
            msg_time = now - base_offset - timedelta(minutes=mock_msg.minutes_ago)
            messages.append(ChatMessage(
                timestamp=msg_time.isoformat(),
                username=mock_msg.username,
                message=mock_msg.message,
                is_mention=mock_msg.is_mention,
            ))
        
        # Add dynamic messages (already have timestamps)
        messages.extend(self._dynamic_messages)
        
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        
        return messages
    
    def clear_dynamic_messages(self) -> None:
        """Clear only dynamically added messages."""
        self._dynamic_messages.clear()
    
    def clear_all(self) -> None:
        """Clear all messages including static ones."""
        self.config.messages.clear()
        self._dynamic_messages.clear()
    
    def set_static_messages(self, messages: List[MockChatMessage]) -> None:
        """Replace static messages with new ones."""
        self.config.messages = messages


# =============================================================================
# Combined Mock Provider
# =============================================================================

class MockContextProviders:
    """
    Container for all mock context providers.
    
    Provides a unified interface to manage all mock context data.
    
    Usage:
        providers = MockContextProviders()
        
        # Configure queue
        providers.set_queue_pending(2)
        
        # Configure workload
        providers.set_workload(WorkloadStatus.HIGH)
        
        # Add chat message
        providers.add_chat_message("user123", "Hello!", is_mention=True)
        
        # Build context for agent
        context = await providers.build_context()
    """
    
    def __init__(
        self,
        queue_config: Optional[MockQueueConfig] = None,
        workload_config: Optional[MockWorkloadConfig] = None,
        chat_config: Optional[MockChatConfig] = None,
    ):
        """
        Initialize all mock providers.
        
        Args:
            queue_config: Optional queue configuration
            workload_config: Optional workload configuration
            chat_config: Optional chat configuration
        """
        self.queue_provider = MockQueueInfoProvider(queue_config)
        self.workload_provider = MockWorkloadProvider(workload_config)
        self.chat_store = MockChatStore(chat_config)
        
        logger.info("MockContextProviders initialized")
    
    # -------------------------------------------------------------------------
    # Queue Configuration
    # -------------------------------------------------------------------------
    
    def set_queue_pending(self, count: int) -> None:
        """Set the pending video count."""
        self.queue_provider.update_config(pending_count=count)
        self.workload_provider.update_config(queue_pending=count)
    
    def set_queue_played(self, count: int) -> None:
        """Set the played video count."""
        self.queue_provider.update_config(played_count=count)
    
    def set_next_videos(self, videos: List[str]) -> None:
        """Set the next videos list."""
        self.queue_provider.update_config(next_videos=videos)
    
    def set_replay_mode(self, is_replaying: bool) -> None:
        """Set whether in replay mode."""
        self.queue_provider.update_config(is_replaying=is_replaying)
    
    def set_active_jobs(self, count: int) -> None:
        """Set active job count."""
        self.queue_provider.update_config(active_jobs=count)
        self.workload_provider.update_config(active_jobs=count)
    
    # -------------------------------------------------------------------------
    # Workload Configuration
    # -------------------------------------------------------------------------
    
    def set_workload(self, status: WorkloadStatus) -> None:
        """Set the workload status."""
        self.workload_provider.update_config(workload=status)
    
    # -------------------------------------------------------------------------
    # Chat Configuration
    # -------------------------------------------------------------------------
    
    def add_chat_message(
        self,
        username: str,
        message: str,
        is_mention: bool = False,
    ) -> ChatMessage:
        """Add a new chat message."""
        return self.chat_store.add_message(username, message, is_mention)
    
    def set_chat_time_offset(self, minutes: float) -> None:
        """Set time offset for chat messages."""
        self.chat_store.set_time_offset(minutes)
    
    def clear_chat_history(self) -> None:
        """Clear dynamic chat messages."""
        self.chat_store.clear_dynamic_messages()
    
    def get_chat_messages(self) -> List[ChatMessage]:
        """Get all chat messages."""
        return self.chat_store.get_messages()
    
    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------
    
    async def get_queue_info(self) -> QueueInfo:
        """Get current queue info."""
        return await self.queue_provider.get_queue_info()
    
    async def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        return await self.workload_provider.get_system_status()
    
    async def build_context(self) -> Dict[str, Any]:
        """
        Build the complete context dictionary for the agent.
        
        Returns:
            Dictionary with queue_info, system_status, and chat_messages
        """
        queue_info = await self.get_queue_info()
        system_status = await self.get_system_status()
        chat_messages = self.get_chat_messages()
        
        # Sync values between queue and system status
        system_status.queue_pending = queue_info.pending_count
        queue_info.active_jobs = system_status.active_jobs
        
        logger.debug(
            f"Built mock context: queue_pending={queue_info.pending_count}, "
            f"workload={system_status.workload.value}, "
            f"chat_messages={len(chat_messages)}"
        )
        
        return {
            STATE_QUEUE_INFO: queue_info,
            STATE_SYSTEM_STATUS: system_status,
            STATE_CHAT_MESSAGES: chat_messages,
        }


# =============================================================================
# Factory Functions for Common Scenarios
# =============================================================================

def create_empty_queue_providers() -> MockContextProviders:
    """Create providers simulating an empty queue (should trigger video creation)."""
    return MockContextProviders(
        queue_config=MockQueueConfig(
            pending_count=0,
            played_count=5,
            next_videos=[],
            is_replaying=True,
            active_jobs=0,
        ),
        workload_config=MockWorkloadConfig(
            workload=WorkloadStatus.LOW,
            active_jobs=0,
            queue_pending=0,
        ),
    )


def create_busy_system_providers() -> MockContextProviders:
    """Create providers simulating a busy system (should avoid video creation)."""
    return MockContextProviders(
        queue_config=MockQueueConfig(
            pending_count=2,
            played_count=10,
            next_videos=["Video en cours"],
            is_replaying=False,
            active_jobs=5,
        ),
        workload_config=MockWorkloadConfig(
            workload=WorkloadStatus.CRITICAL,
            active_jobs=5,
            queue_pending=2,
        ),
    )


def create_stale_conversation_providers(minutes_ago: float = 10.0) -> MockContextProviders:
    """Create providers simulating a stale conversation (no recent activity)."""
    providers = MockContextProviders(
        chat_config=MockChatConfig.with_default_conversation(),
    )
    providers.set_chat_time_offset(minutes_ago)
    return providers
