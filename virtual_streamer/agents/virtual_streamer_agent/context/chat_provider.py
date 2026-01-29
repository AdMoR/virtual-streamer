"""
Chat Message Context Provider for the Virtual Streamer Agent.

Provides Twitch chat messages for the agent's prompt.
"""

import logging
from datetime import datetime, timedelta
from typing import List

from virtual_streamer.agents.virtual_streamer_agent.schema import ChatMessage

logger = logging.getLogger(__name__)


class MockChatMessageContextProvider:
    """
    Mock provider for Twitch chat messages.
    
    Maintains an internal message store for testing. A real implementation
    would read from a Twitch WebSocket connection or chat store.
    
    Supports:
    - Adding messages dynamically
    - Time offset to simulate "old" conversations
    - Static preset messages for consistent test scenarios
    
    Usage:
        provider = MockChatMessageContextProvider()
        provider.load_default_conversation()  # Load sample messages
        provider.add_message("user", "@bot make a video", is_mention=True)
        section = await provider.render()  # Get formatted prompt section
    """
    
    # Default sample conversation for testing
    DEFAULT_MESSAGES = [
        {
            "username": "viewer42",
            "message": "Salut le stream ! 👋",
            "is_mention": False,
            "minutes_ago": 10.0,
        },
        {
            "username": "science_fan",
            "message": "J'adore vos vidéos sur l'espace !",
            "is_mention": False,
            "minutes_ago": 8.0,
        },
        {
            "username": "curious_bob",
            "message": "@virtualstreamer tu pourrais faire une vidéo sur les dinosaures ?",
            "is_mention": True,
            "minutes_ago": 5.0,
        },
        {
            "username": "regular_viewer",
            "message": "Trop bien la dernière vidéo !",
            "is_mention": False,
            "minutes_ago": 3.0,
        },
        {
            "username": "new_viewer",
            "message": "C'est quoi cette chaîne ?",
            "is_mention": False,
            "minutes_ago": 1.0,
        },
    ]
    
    def __init__(self, max_messages: int = 50):
        """
        Initialize the chat provider.
        
        Args:
            max_messages: Maximum number of messages to include in rendered output
        """
        self._max_messages = max_messages
        self._time_offset_minutes = 0.0
        self._messages: List[ChatMessage] = []
        self._static_messages: List[dict] = []
    
    @property
    def name(self) -> str:
        """Provider name for logging/debugging."""
        return "chat_messages"
    
    async def render(self) -> str:
        """
        Fetch messages and render the chat section.
        
        Returns:
            Formatted chat messages string for the prompt
        """
        messages = await self._fetch()
        return self._format(messages)
    
    async def _fetch(self) -> List[ChatMessage]:
        """
        Fetch chat messages.
        
        Mock: returns internal message store + static messages with time offset
        Real: would read from Twitch WebSocket or chat store
        
        Returns:
            List of ChatMessage objects
        """
        all_messages = []
        now = datetime.now()
        
        # Add static messages with computed timestamps
        for static in self._static_messages:
            msg_time = now - timedelta(
                minutes=self._time_offset_minutes + static.get("minutes_ago", 0)
            )
            all_messages.append(ChatMessage(
                timestamp=msg_time.isoformat(),
                username=static["username"],
                message=static["message"],
                is_mention=static.get("is_mention", False),
            ))
        
        # Add dynamic messages
        all_messages.extend(self._messages)
        
        # Sort by timestamp and return recent
        all_messages.sort(key=lambda m: m.timestamp)
        return all_messages[-self._max_messages:]
    
    def _format(self, messages: List[ChatMessage]) -> str:
        """
        Format messages into a prompt section.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            Formatted markdown string
        """
        if not messages:
            return "## Recent Chat Messages\n\n*No recent messages*"
        
        lines = ["## Recent Chat Messages", ""]
        
        for msg in messages:
            mention_marker = " [MENTION]" if msg.is_mention else ""
            lines.append(f"[{msg.timestamp}] @{msg.username}{mention_marker}: {msg.message}")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Configuration methods for testing
    # -------------------------------------------------------------------------
    
    def add_message(
        self,
        username: str,
        message: str,
        is_mention: bool = False,
    ) -> ChatMessage:
        """
        Add a new message at current time.
        
        Args:
            username: Username of the message sender
            message: Message content
            is_mention: Whether this message mentions the bot
            
        Returns:
            The created ChatMessage
        """
        msg = ChatMessage(
            timestamp=datetime.now().isoformat(),
            username=username,
            message=message,
            is_mention=is_mention,
        )
        self._messages.append(msg)
        logger.debug(f"Added chat message from @{username}: {message[:50]}...")
        return msg
    
    def set_time_offset(self, minutes: float) -> None:
        """
        Set time offset for static messages (simulates old conversation).
        
        Args:
            minutes: Number of minutes to offset into the past
        """
        self._time_offset_minutes = minutes
        logger.debug(f"Chat time offset set to {minutes} minutes")
    
    def set_static_messages(self, messages: List[dict]) -> None:
        """
        Set static messages for consistent test scenarios.
        
        Args:
            messages: List of message dicts with keys:
                      username, message, is_mention, minutes_ago
        """
        self._static_messages = messages
    
    def load_default_conversation(self) -> None:
        """Load the default sample conversation."""
        self._static_messages = [m.copy() for m in self.DEFAULT_MESSAGES]
        logger.debug("Loaded default conversation")
    
    def get_messages(self) -> List[ChatMessage]:
        """
        Get all dynamic messages (sync, for state injection).
        
        Returns:
            Copy of the dynamic messages list
        """
        return self._messages.copy()
    
    def has_mentions(self) -> bool:
        """
        Check if there are any mention messages.
        
        Returns:
            True if any message has is_mention=True
        """
        for msg in self._messages:
            if msg.is_mention:
                return True
        for static in self._static_messages:
            if static.get("is_mention", False):
                return True
        return False
    
    def clear(self) -> None:
        """Clear all dynamic messages."""
        self._messages.clear()
        logger.debug("Dynamic messages cleared")
    
    def clear_all(self) -> None:
        """Clear all messages including static ones."""
        self._messages.clear()
        self._static_messages.clear()
        logger.debug("All messages cleared")
    
    def reset_to_defaults(self) -> None:
        """Reset to default state with default conversation."""
        self._messages.clear()
        self._time_offset_minutes = 0.0
        self.load_default_conversation()
        logger.debug("Chat provider reset to defaults")
