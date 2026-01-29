"""
Chat message storage for the Virtual Streamer Agent.

This module provides in-memory storage for chat messages with
thread-safe access and automatic pruning.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

from virtual_streamer.agents.virtual_streamer_agent.schema import ChatMessage

logger = logging.getLogger(__name__)


class ChatStore:
    """
    Thread-safe in-memory storage for chat messages.
    
    Uses a deque with a maximum size to automatically prune old messages.
    Provides async-safe methods for adding and retrieving messages.
    
    Usage:
        store = ChatStore(max_messages=1000)
        
        # Add messages
        store.add_message(ChatMessage(...))
        
        # Get messages for context
        messages = store.get_messages(limit=100)
    """
    
    def __init__(self, max_messages: int = 1000):
        """
        Initialize the chat store.
        
        Args:
            max_messages: Maximum messages to store (oldest are pruned)
        """
        self._max_messages = max_messages
        self._messages: Deque[ChatMessage] = deque(maxlen=max_messages)
        self._lock = asyncio.Lock()
        
        logger.info(f"ChatStore initialized with max_messages={max_messages}")
    
    async def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the store (async-safe).
        
        Args:
            message: Message to add
        """
        async with self._lock:
            self._messages.append(message)
    
    def add_message_sync(self, message: ChatMessage) -> None:
        """
        Add a message to the store (sync version).
        
        Use this when not in an async context (e.g., from callbacks).
        Note: This is not thread-safe without external synchronization.
        
        Args:
            message: Message to add
        """
        self._messages.append(message)
    
    async def add_messages(self, messages: List[ChatMessage]) -> None:
        """
        Add multiple messages to the store.
        
        Args:
            messages: Messages to add
        """
        async with self._lock:
            for msg in messages:
                self._messages.append(msg)
    
    async def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get messages from the store.
        
        Args:
            limit: Maximum messages to return (None = all stored)
            
        Returns:
            List of messages (oldest first)
        """
        async with self._lock:
            messages = list(self._messages)
        
        if limit is not None and len(messages) > limit:
            return messages[-limit:]
        return messages
    
    def get_messages_sync(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get messages from the store (sync version).
        
        Args:
            limit: Maximum messages to return
            
        Returns:
            List of messages (oldest first)
        """
        messages = list(self._messages)
        if limit is not None and len(messages) > limit:
            return messages[-limit:]
        return messages
    
    async def clear(self) -> None:
        """Clear all stored messages."""
        async with self._lock:
            self._messages.clear()
        logger.info("ChatStore cleared")
    
    def clear_sync(self) -> None:
        """Clear all stored messages (sync version)."""
        self._messages.clear()
    
    @property
    def count(self) -> int:
        """Number of messages currently stored."""
        return len(self._messages)
    
    @property
    def max_messages(self) -> int:
        """Maximum messages this store can hold."""
        return self._max_messages


def create_chat_message(
    username: str,
    message: str,
    is_mention: bool = False,
    timestamp: Optional[str] = None,
) -> ChatMessage:
    """
    Helper function to create a ChatMessage.
    
    Args:
        username: Twitch username
        message: Message content
        is_mention: Whether the message mentions the bot
        timestamp: ISO timestamp (auto-generated if not provided)
        
    Returns:
        ChatMessage instance
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
    
    return ChatMessage(
        timestamp=timestamp,
        username=username,
        message=message,
        is_mention=is_mention,
    )
