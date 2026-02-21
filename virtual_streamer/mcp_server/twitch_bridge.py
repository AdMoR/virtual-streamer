"""
Twitch bridge for the MCP server.

Manages TwitchClient lifecycle and ChatStore message buffering.
The bridge connects to Twitch IRC, receives all chat messages,
and stores them for the MCP tools/resources to read.
"""

import asyncio
import logging
from typing import List, Optional

from virtual_streamer.mcp_server.config import MCPConfig
from virtual_streamer.streaming.agent_loop.chat_store import ChatStore, create_chat_message
from virtual_streamer.streaming.twitch.chat_reader import TwitchClient

logger = logging.getLogger(__name__)

# Bot mention patterns used to flag messages as mentions
BOT_MENTION_PATTERNS = [
    "allo",
    "@virtualstreamer",
    "@virtual_streamer",
    "virtual streamer",
]


def _is_mention(message: str) -> bool:
    """Check if a message mentions the bot."""
    lower = message.lower()
    return any(pattern in lower for pattern in BOT_MENTION_PATTERNS)


class TwitchBridge:
    """
    Manages the TwitchClient and ChatStore for the MCP server.

    Starts the Twitch WebSocket connection in the background,
    buffers incoming messages, and provides send capability.
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.chat_store = ChatStore(max_messages=1000)
        self._twitch_client: Optional[TwitchClient] = None
        self._connection_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return self._twitch_client is not None and self._twitch_client.is_connected

    async def start(self) -> None:
        """Start the Twitch connection in the background."""
        if not self.config.has_twitch_credentials:
            logger.warning(
                "Twitch credentials not configured — chat tools will be unavailable"
            )
            return

        self._twitch_client = TwitchClient(
            client_id=self.config.twitch_client_id,
            client_secret=self.config.twitch_client_secret,
            refresh_token=self.config.twitch_refresh_token,
            channel_name=self.config.twitch_channel,
            bot_username=self.config.twitch_bot_username,
        )

        # Register message callback to buffer into ChatStore
        self._twitch_client.set_on_new_message_callback(self._on_message)

        # Run the connection loop in the background
        self._connection_task = asyncio.create_task(self._run_connection_loop())
        logger.info(f"Twitch bridge started for #{self.config.twitch_channel}")

    async def stop(self) -> None:
        """Stop the Twitch connection."""
        if self._connection_task is not None:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None
        self._twitch_client = None
        logger.info("Twitch bridge stopped")

    async def send_message(self, message: str) -> None:
        """
        Send a message to Twitch chat.

        Raises:
            RuntimeError: If not connected to Twitch
        """
        if self._twitch_client is None:
            raise RuntimeError("Twitch client not initialized")
        await self._twitch_client.send_chat_message(message)

    async def get_messages(
        self, limit: int = 50, mentions_only: bool = False
    ) -> List[dict]:
        """
        Get recent chat messages.

        Args:
            limit: Maximum messages to return
            mentions_only: If True, only return messages that mention the bot

        Returns:
            List of message dicts with timestamp, username, message, is_mention
        """
        messages = await self.chat_store.get_messages(limit=limit if not mentions_only else None)

        if mentions_only:
            messages = [m for m in messages if m.is_mention]
            messages = messages[-limit:]

        return [m.model_dump() for m in messages]

    def _on_message(self, username: str, message: str) -> None:
        """Callback invoked by TwitchClient for each chat message (sync)."""
        chat_msg = create_chat_message(
            username=username,
            message=message,
            is_mention=_is_mention(message),
        )
        self.chat_store.add_message_sync(chat_msg)

    async def _run_connection_loop(self) -> None:
        """Run the Twitch connection with automatic reconnection."""
        backoff = 1
        max_backoff = 300

        while True:
            try:
                logger.info(f"Connecting to Twitch chat for #{self.config.twitch_channel}")
                await self._twitch_client.connect_to_chat()
                backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Twitch connection error: {e}")

            logger.info(f"Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
