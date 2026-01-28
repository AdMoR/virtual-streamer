"""
Twitch message tool for the Virtual Streamer Agent.

This tool allows the agent to send messages to the Twitch chat.
"""

import logging
from typing import Optional, Callable, Awaitable

from virtual_streamer.agents.virtual_streamer.tools.base import register_tool

logger = logging.getLogger(__name__)


# Maximum message length for Twitch chat
MAX_MESSAGE_LENGTH = 500

# Global reference to the message sender function
# This is set by the runner when initializing the agent
_message_sender: Optional[Callable[[str], Awaitable[None]]] = None


def set_message_sender(sender: Callable[[str], Awaitable[None]]) -> None:
    """
    Set the message sender function.
    
    This should be called by the runner to inject the TwitchClient's
    send_message method.
    
    Args:
        sender: Async function that sends a message to Twitch chat
    """
    global _message_sender
    _message_sender = sender
    logger.info("Twitch message sender configured")


def get_message_sender() -> Optional[Callable[[str], Awaitable[None]]]:
    """Get the currently configured message sender."""
    return _message_sender


@register_tool("send_twitch_message")
async def send_twitch_message(message: str) -> dict:
    """
    Send a message to the Twitch chat.
    
    The message will be truncated to 500 characters if longer.
    
    Args:
        message: The message to send to the chat
        
    Returns:
        dict with:
            - success: Whether the message was sent
            - message: The actual message sent (possibly truncated)
            - error: Error message if sending failed
    """
    global _message_sender
    
    if _message_sender is None:
        logger.error("No message sender configured - cannot send Twitch message")
        return {
            "success": False,
            "error": "Message sender not configured",
            "message": None,
        }
    
    # Truncate message if too long
    truncated_message = message[:MAX_MESSAGE_LENGTH]
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(
            f"Message truncated from {len(message)} to {MAX_MESSAGE_LENGTH} chars"
        )
    
    try:
        await _message_sender(truncated_message)
        logger.info(f"Sent Twitch message: {truncated_message[:50]}...")
        return {
            "success": True,
            "message": truncated_message,
        }
    except Exception as e:
        logger.error(f"Failed to send Twitch message: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": truncated_message,
        }
