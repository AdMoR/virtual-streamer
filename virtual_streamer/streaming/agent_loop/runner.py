"""
Virtual Streamer Runner.

This module provides the main application runner that orchestrates
the Virtual Streamer Agent, Twitch connection, and context building.
"""

import asyncio
import logging
import os
import signal
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.agents.virtual_streamer import (
    VirtualStreamerAgent,
    get_virtual_streamer_agent,
)
from virtual_streamer.agents.virtual_streamer.schema import ChatMessage
from virtual_streamer.agents.virtual_streamer.context import ContextBuilder
from virtual_streamer.agents.virtual_streamer.tools import ToolFactory
from virtual_streamer.agents.virtual_streamer.tools.send_message import set_message_sender
from virtual_streamer.streaming.agent_loop.chat_store import ChatStore, create_chat_message

logger = logging.getLogger(__name__)


class VirtualStreamerRunner:
    """
    Main application runner for the Virtual Streamer Agent.
    
    Responsibilities:
    - Maintain Twitch WebSocket connection
    - Build context for each agent iteration
    - Run agent and execute tool calls
    - Handle errors and reconnection
    
    Usage:
        runner = VirtualStreamerRunner(
            api_url="http://localhost:8000",
            stream_id="main",
            programmation_id="my-prog",
        )
        
        # Set up Twitch connection externally and pass message handler
        runner.setup_message_sender(twitch_client.send_message)
        
        # Run the agent loop
        await runner.run()
    """
    
    def __init__(
        self,
        api_url: str,
        stream_id: str,
        programmation_id: str,
        tools_config_path: Optional[str] = None,
        loop_interval: float = 5.0,
        max_chat_messages: int = 100,
        bot_mention_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the runner.
        
        Args:
            api_url: Base URL of the Virtual Streamer API
            stream_id: Stream ID for video generation
            programmation_id: Programmation ID for queue queries
            tools_config_path: Path to tools configuration YAML
            loop_interval: Seconds between agent iterations
            max_chat_messages: Maximum chat messages to include in context
            bot_mention_patterns: Patterns that indicate a mention (default: ["allo", "@virtualstreamer"])
        """
        self.api_url = api_url
        self.stream_id = stream_id
        self.programmation_id = programmation_id
        self.loop_interval = loop_interval
        self.max_chat_messages = max_chat_messages
        
        # Bot mention detection
        self.bot_mention_patterns = bot_mention_patterns or [
            "allo",
            "@virtualstreamer",
            "@virtual_streamer",
            "virtual streamer",
        ]
        
        # Components
        self.chat_store = ChatStore(max_messages=1000)
        self.tool_factory = ToolFactory(config_path=tools_config_path)
        self.context_builder = ContextBuilder(
            programmation_id=programmation_id,
            api_url=api_url,
            max_chat_messages=max_chat_messages,
        )
        
        # Agent and runner (created in setup)
        self.agent: Optional[VirtualStreamerAgent] = None
        self.adk_runner: Optional[Runner] = None
        self.session_service: Optional[InMemorySessionService] = None
        
        # Control flags
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Message sender (set externally)
        self._message_sender: Optional[Callable] = None
        
        logger.info(
            f"VirtualStreamerRunner initialized: "
            f"api_url={api_url}, stream_id={stream_id}, "
            f"programmation_id={programmation_id}, "
            f"loop_interval={loop_interval}s"
        )
    
    def setup_message_sender(self, sender: Callable) -> None:
        """
        Set up the Twitch message sender.
        
        This should be called with the TwitchClient's send_message method
        before starting the runner.
        
        Args:
            sender: Async function to send messages to Twitch
        """
        self._message_sender = sender
        set_message_sender(sender)
        logger.info("Message sender configured")
    
    async def setup(self) -> None:
        """
        Set up the agent and ADK runner.
        
        Call this after configuring the message sender.
        """
        # Get available tools
        tools = self.tool_factory.get_available_tools()
        logger.info(f"Loaded {len(tools)} tools")
        
        # Create agent
        self.agent = get_virtual_streamer_agent(
            tools=tools,
            max_chat_messages=self.max_chat_messages,
        )
        
        # Create session service and runner
        self.session_service = InMemorySessionService()
        self.adk_runner = Runner(
            agent=self.agent,
            app_name="virtual_streamer",
            session_service=self.session_service,
        )
        
        logger.info("Agent and runner set up")
    
    def add_chat_message(
        self,
        username: str,
        message: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Add a chat message from Twitch.
        
        Call this from the Twitch message handler.
        
        Args:
            username: Twitch username
            message: Message content
            timestamp: ISO timestamp (auto-generated if not provided)
        """
        # Detect if message mentions the bot
        is_mention = self._is_mention(message)
        
        # Create and store message
        chat_msg = create_chat_message(
            username=username,
            message=message,
            is_mention=is_mention,
            timestamp=timestamp,
        )
        self.chat_store.add_message_sync(chat_msg)
        
        # Also add to context builder
        self.context_builder.add_chat_message(chat_msg)
        
        if is_mention:
            logger.info(f"Bot mentioned by @{username}: {message[:50]}...")
    
    def _is_mention(self, message: str) -> bool:
        """Check if a message mentions the bot."""
        message_lower = message.lower()
        return any(
            pattern.lower() in message_lower
            for pattern in self.bot_mention_patterns
        )
    
    async def run_iteration(self) -> None:
        """
        Run a single agent iteration.
        
        Builds context, runs the agent, and processes any tool calls.
        """
        if self.adk_runner is None:
            logger.error("Runner not set up - call setup() first")
            return
        
        try:
            # Build context
            context = await self.context_builder.build()
            
            # Create or get session
            session_id = "virtual_streamer_session"
            user_id = "system"
            
            # Check if we have any mentions or need to proactively act
            chat_messages = context.get("chat_messages", [])
            queue_info = context.get("queue_info")
            
            has_mentions = any(m.is_mention for m in chat_messages if isinstance(m, ChatMessage))
            queue_low = queue_info and queue_info.pending_count < 3
            
            # Only run agent if there's something to do
            if not has_mentions and not queue_low:
                logger.debug("No mentions and queue healthy - skipping iteration")
                return
            
            logger.info(
                f"Running agent iteration: mentions={has_mentions}, "
                f"queue_pending={queue_info.pending_count if queue_info else 'unknown'}"
            )
            
            # Prepare the user message (summary of current state)
            user_message = self._build_user_message(context, has_mentions)
            
            # Run the agent
            async for event in self.adk_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    parts=[types.Part(text=user_message)],
                    role="user",
                ),
                state=context,
            ):
                # Log events for debugging
                if hasattr(event, 'content'):
                    logger.debug(f"Agent event: {event}")
            
            logger.info("Agent iteration completed")
            
        except Exception as e:
            logger.error(f"Agent iteration failed: {e}", exc_info=True)
    
    def _build_user_message(
        self,
        context: Dict[str, Any],
        has_mentions: bool,
    ) -> str:
        """Build the user message to send to the agent."""
        parts = []
        
        if has_mentions:
            # Find recent mentions
            messages = context.get("chat_messages", [])
            mentions = [m for m in messages if isinstance(m, ChatMessage) and m.is_mention]
            recent_mentions = mentions[-3:]  # Last 3 mentions
            
            parts.append("Nouveaux messages qui te mentionnent:")
            for m in recent_mentions:
                parts.append(f"- @{m.username}: {m.message}")
        
        queue_info = context.get("queue_info")
        if queue_info and queue_info.pending_count < 3:
            parts.append(f"\n⚠️ La queue est presque vide ({queue_info.pending_count} vidéos pending)")
        
        if not parts:
            parts.append("Vérifie l'état du stream et du chat.")
        
        return "\n".join(parts)
    
    async def run(self) -> None:
        """
        Run the main agent loop.
        
        Runs until shutdown() is called or a signal is received.
        """
        if self.agent is None:
            await self.setup()
        
        self.running = True
        logger.info(f"Starting agent loop (interval: {self.loop_interval}s)")
        
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
        
        try:
            while self.running:
                await self.run_iteration()
                
                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.loop_interval,
                    )
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout - continue loop
                    pass
                    
        except asyncio.CancelledError:
            logger.info("Agent loop cancelled")
        finally:
            self.running = False
            logger.info("Agent loop stopped")
    
    def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self.shutdown()
    
    def shutdown(self) -> None:
        """Request graceful shutdown of the agent loop."""
        self.running = False
        self._shutdown_event.set()
        logger.info("Shutdown requested")
    
    async def reload_tools(self) -> None:
        """Reload tools from configuration file."""
        logger.info("Reloading tools configuration")
        self.tool_factory.reload()
        tools = self.tool_factory.get_available_tools()
        
        # Recreate agent with new tools
        self.agent = get_virtual_streamer_agent(
            tools=tools,
            max_chat_messages=self.max_chat_messages,
        )
        
        # Update runner
        self.adk_runner = Runner(
            agent=self.agent,
            app_name="virtual_streamer",
            session_service=self.session_service,
        )
        
        logger.info(f"Tools reloaded: {len(tools)} tools available")
