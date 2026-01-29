"""
Conversation management for the Virtual Streamer Agent.

This module provides strategies for managing chat history and
controlling the context window size.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from virtual_streamer.agents.virtual_streamer_agent.schema import ChatMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Interface
# =============================================================================

class ConversationManagerStrategy(ABC):
    """
    Abstract base class for conversation management strategies.
    
    Different strategies can be implemented to control which messages
    are included in the agent's context window.
    """
    
    @abstractmethod
    def select_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Select which messages to include in the context.
        
        Args:
            messages: All available chat messages
            
        Returns:
            Filtered/selected list of messages for context
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...


# =============================================================================
# Strategy Implementations
# =============================================================================

class KeepLastN(ConversationManagerStrategy):
    """
    Keep only the last N messages.
    
    This is the simplest strategy - just take the most recent messages.
    """
    
    def __init__(self, n: int = 100):
        """
        Initialize the strategy.
        
        Args:
            n: Maximum number of messages to keep (default: 100)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        self._n = n
    
    @property
    def n(self) -> int:
        """Number of messages to keep."""
        return self._n
    
    @property
    def name(self) -> str:
        return f"KeepLastN(n={self._n})"
    
    def select_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Keep only the last N messages."""
        if len(messages) <= self._n:
            return messages
        return messages[-self._n:]


class KeepMentionsAndRecent(ConversationManagerStrategy):
    """
    Keep all messages that mention the bot, plus the most recent messages.
    
    This strategy ensures the agent sees all messages addressed to it,
    plus recent context.
    """
    
    def __init__(self, recent_count: int = 50, max_total: int = 150):
        """
        Initialize the strategy.
        
        Args:
            recent_count: Number of recent messages to always keep
            max_total: Maximum total messages to return
        """
        self._recent_count = recent_count
        self._max_total = max_total
    
    @property
    def name(self) -> str:
        return f"KeepMentionsAndRecent(recent={self._recent_count}, max={self._max_total})"
    
    def select_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Keep mentions and recent messages."""
        if not messages:
            return []
        
        # Get all mentions
        mentions = [m for m in messages if m.is_mention]
        
        # Get recent messages
        recent = messages[-self._recent_count:]
        
        # Combine, removing duplicates while preserving order
        seen = set()
        result = []
        
        # First add mentions (older ones first)
        for msg in mentions:
            key = (msg.timestamp, msg.username, msg.message)
            if key not in seen:
                seen.add(key)
                result.append(msg)
        
        # Then add recent that aren't already included
        for msg in recent:
            key = (msg.timestamp, msg.username, msg.message)
            if key not in seen:
                seen.add(key)
                result.append(msg)
        
        # Sort by timestamp and limit
        result.sort(key=lambda m: m.timestamp)
        
        if len(result) > self._max_total:
            # Keep most recent if over limit
            result = result[-self._max_total:]
        
        return result


class KeepAll(ConversationManagerStrategy):
    """
    Keep all messages (no filtering).
    
    Use with caution - can lead to very large context windows.
    """
    
    @property
    def name(self) -> str:
        return "KeepAll"
    
    def select_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Keep all messages."""
        return messages


# =============================================================================
# Conversation Manager
# =============================================================================

class ConversationManager:
    """
    Manages chat message history using a configurable strategy.
    
    This class stores messages and applies a strategy to select
    which messages to include when building context.
    """
    
    def __init__(
        self,
        strategy: ConversationManagerStrategy = None,
        max_storage: int = 1000,
    ):
        """
        Initialize the conversation manager.
        
        Args:
            strategy: Strategy for selecting messages (default: KeepLastN(100))
            max_storage: Maximum messages to store before pruning old ones
        """
        self._strategy = strategy or KeepLastN(100)
        self._max_storage = max_storage
        self._messages: List[ChatMessage] = []
        
        logger.info(f"ConversationManager initialized with strategy: {self._strategy.name}")
    
    @property
    def strategy(self) -> ConversationManagerStrategy:
        """Current selection strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, value: ConversationManagerStrategy) -> None:
        """Change the selection strategy."""
        self._strategy = value
        logger.info(f"ConversationManager strategy changed to: {value.name}")
    
    def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the history.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
        
        # Prune if over storage limit
        if len(self._messages) > self._max_storage:
            # Keep only the most recent max_storage messages
            excess = len(self._messages) - self._max_storage
            self._messages = self._messages[excess:]
            logger.debug(f"Pruned {excess} old messages from storage")
    
    def add_messages(self, messages: List[ChatMessage]) -> None:
        """
        Add multiple messages to the history.
        
        Args:
            messages: Messages to add
        """
        for msg in messages:
            self.add_message(msg)
    
    def get_messages_for_context(self) -> List[ChatMessage]:
        """
        Get messages for the agent context using the current strategy.
        
        Returns:
            Selected messages based on the strategy
        """
        return self._strategy.select_messages(self._messages)
    
    def get_all_messages(self) -> List[ChatMessage]:
        """Get all stored messages (regardless of strategy)."""
        return list(self._messages)
    
    def clear(self) -> None:
        """Clear all stored messages."""
        self._messages = []
        logger.info("ConversationManager cleared")
    
    @property
    def message_count(self) -> int:
        """Number of messages currently stored."""
        return len(self._messages)
