"""
Main entrypoint for the Virtual Streamer Agent.

This module provides the CLI entrypoint for running the Virtual Streamer
Agent as a standalone service.

Usage:
    python -m virtual_streamer.streaming.agent_loop.main
    
Environment variables:
    - API_URL: Virtual Streamer API URL (default: http://localhost:8000)
    - STREAM_ID: Stream ID for video generation (default: default)
    - PROGRAMMATION_ID: Programmation ID for queue queries (required)
    - TOOLS_CONFIG: Path to tools configuration YAML
    - LOOP_INTERVAL: Seconds between agent iterations (default: 5)
    - MAX_CHAT_MESSAGES: Maximum chat messages in context (default: 100)
    - TWITCH_CLIENT_ID: Twitch API client ID
    - TWITCH_CLIENT_SECRET: Twitch API client secret
    - TWITCH_REFRESH_TOKEN: Twitch refresh token
    - TWITCH_CHANNEL: Twitch channel to connect to
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from virtual_streamer.streaming.agent_loop.runner import VirtualStreamerRunner
from virtual_streamer.streaming.twitch.chat_reader import TwitchClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get environment variable with optional default and required check."""
    value = os.environ.get(name, default)
    if required and value is None:
        logger.error(f"Required environment variable {name} is not set")
        sys.exit(1)
    return value


async def run_with_twitch():
    """
    Run the Virtual Streamer Agent with Twitch integration.
    
    Sets up the Twitch client and connects the runner to it.
    """
    # Load configuration from environment
    api_url = get_env("API_URL", "http://localhost:8000")
    stream_id = get_env("STREAM_ID", "default")
    programmation_id = get_env("PROGRAMMATION_ID", required=True)
    tools_config = get_env("TOOLS_CONFIG")
    loop_interval = float(get_env("LOOP_INTERVAL", "5"))
    max_chat_messages = int(get_env("MAX_CHAT_MESSAGES", "100"))
    
    # Twitch credentials
    twitch_client_id = get_env("TWITCH_CLIENT_ID", required=True)
    twitch_client_secret = get_env("TWITCH_CLIENT_SECRET", required=True)
    twitch_refresh_token = get_env("TWITCH_REFRESH_TOKEN", required=True)
    twitch_channel = get_env("TWITCH_CHANNEL", required=True)
    
    logger.info("=" * 60)
    logger.info("Virtual Streamer Agent Starting")
    logger.info("=" * 60)
    logger.info(f"API URL: {api_url}")
    logger.info(f"Stream ID: {stream_id}")
    logger.info(f"Programmation ID: {programmation_id}")
    logger.info(f"Twitch Channel: {twitch_channel}")
    logger.info(f"Loop Interval: {loop_interval}s")
    logger.info("=" * 60)
    
    # Create the runner
    runner = VirtualStreamerRunner(
        api_url=api_url,
        stream_id=stream_id,
        programmation_id=programmation_id,
        tools_config_path=tools_config,
        loop_interval=loop_interval,
        max_chat_messages=max_chat_messages,
    )
    
    # Create Twitch client
    twitch_client = TwitchClient(
        client_id=twitch_client_id,
        client_secret=twitch_client_secret,
        refresh_token=twitch_refresh_token,
        channel_name=twitch_channel,
    )
    
    # Set up message sender
    async def send_message(message: str):
        """Send a message via Twitch client."""
        # Note: TwitchClient.send_message expects websocket, we need to adapt
        # For now, we'll log and skip if no active connection
        logger.info(f"[OUTGOING] {message}")
        # In a full implementation, we'd track the active websocket
        # and send through it
    
    runner.setup_message_sender(send_message)
    
    # Set up the runner
    await runner.setup()
    
    # Create a custom message handler that feeds into the runner
    original_handle_privmsg = twitch_client.handle_privmsg
    
    async def custom_handle_privmsg(websocket, message: str):
        """Custom handler that also feeds messages to the runner."""
        # Parse the message
        try:
            parts = message.split(":", 2)
            if len(parts) > 2:
                user_info, chat_message = parts[1], parts[2]
                username = user_info.split("!")[0]
                
                # Add to runner
                runner.add_chat_message(username=username, message=chat_message)
        except Exception as e:
            logger.warning(f"Failed to parse message for runner: {e}")
        
        # Call original handler
        await original_handle_privmsg(websocket, message)
    
    # Patch the handler
    twitch_client.handle_privmsg = custom_handle_privmsg
    
    # Run both the Twitch client and the agent loop concurrently
    async def run_twitch():
        """Run Twitch client with reconnection."""
        backoff_time = 1
        max_backoff = 300
        
        while runner.running:
            try:
                logger.info(f"Connecting to Twitch chat for channel: {twitch_channel}")
                await twitch_client.connect_to_chat()
                backoff_time = 1
            except Exception as e:
                logger.error(f"Twitch connection error: {e}")
            
            if runner.running:
                logger.info(f"Reconnecting in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff)
    
    # Run both tasks
    try:
        await asyncio.gather(
            runner.run(),
            run_twitch(),
        )
    except asyncio.CancelledError:
        logger.info("Tasks cancelled")
    finally:
        runner.shutdown()
        logger.info("Virtual Streamer Agent stopped")


async def run_standalone():
    """
    Run the Virtual Streamer Agent without Twitch (for testing).
    
    In this mode, no chat messages are received but the agent
    can still manage the queue proactively.
    """
    # Load configuration from environment
    api_url = get_env("API_URL", "http://localhost:8000")
    stream_id = get_env("STREAM_ID", "default")
    programmation_id = get_env("PROGRAMMATION_ID", required=True)
    tools_config = get_env("TOOLS_CONFIG")
    loop_interval = float(get_env("LOOP_INTERVAL", "5"))
    max_chat_messages = int(get_env("MAX_CHAT_MESSAGES", "100"))
    
    logger.info("=" * 60)
    logger.info("Virtual Streamer Agent Starting (STANDALONE MODE)")
    logger.info("=" * 60)
    logger.info(f"API URL: {api_url}")
    logger.info(f"Stream ID: {stream_id}")
    logger.info(f"Programmation ID: {programmation_id}")
    logger.info(f"Loop Interval: {loop_interval}s")
    logger.info("=" * 60)
    
    # Create the runner
    runner = VirtualStreamerRunner(
        api_url=api_url,
        stream_id=stream_id,
        programmation_id=programmation_id,
        tools_config_path=tools_config,
        loop_interval=loop_interval,
        max_chat_messages=max_chat_messages,
    )
    
    # Set up a dummy message sender
    async def dummy_sender(message: str):
        logger.info(f"[WOULD SEND] {message}")
    
    runner.setup_message_sender(dummy_sender)
    
    # Set up and run
    await runner.setup()
    await runner.run()


def main():
    """Main entrypoint."""
    # Check if we should run with Twitch or standalone
    twitch_enabled = all([
        os.environ.get("TWITCH_CLIENT_ID"),
        os.environ.get("TWITCH_CLIENT_SECRET"),
        os.environ.get("TWITCH_REFRESH_TOKEN"),
        os.environ.get("TWITCH_CHANNEL"),
    ])
    
    if twitch_enabled:
        logger.info("Twitch credentials found - running with Twitch integration")
        asyncio.run(run_with_twitch())
    else:
        logger.info("Twitch credentials not found - running in standalone mode")
        asyncio.run(run_standalone())


if __name__ == "__main__":
    main()
