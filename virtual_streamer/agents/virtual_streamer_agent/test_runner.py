"""
Test Runner for the Virtual Streamer Agent.

This module provides a test harness that uses mock tools and composable
context providers to test agent behavior without requiring real infrastructure.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.agents.virtual_streamer_agent.schema import (
    ChatMessage,
    WorkloadStatus,
)
from virtual_streamer.agents.virtual_streamer_agent.tools.mock import (
    MockToolFactory,
    MockToolFactoryConfig,
    ToolCall,
)
from virtual_streamer.agents.virtual_streamer_agent.context.protocol import (
    ContextProviderProtocol,
)
from virtual_streamer.agents.virtual_streamer_agent.context.queue_provider import (
    MockProcessingQueueContextProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.context.system_provider import (
    MockSystemStatusContextProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.context.chat_provider import (
    MockChatMessageContextProvider,
)
from virtual_streamer.agents.virtual_streamer_agent.prompt import (
    VirtualStreamerInstructionProvider,
)
from virtual_streamer.lib.agents import BaseLlmAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Event Log
# =============================================================================

@dataclass
class AgentEvent:
    """Record of an agent event/response."""
    
    timestamp: str
    event_type: str
    content: Optional[str] = None
    raw_event: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "content": self.content,
        }


class AgentEventLog:
    """Collects and stores agent events for inspection."""
    
    def __init__(self):
        self._events: List[AgentEvent] = []
    
    def log(self, event_type: str, content: Optional[str] = None, raw_event: Any = None) -> None:
        """Log an agent event."""
        event = AgentEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            content=content,
            raw_event=raw_event,
        )
        self._events.append(event)
        logger.debug(f"Agent event: {event_type} - {content[:100] if content else 'N/A'}...")
    
    def get_events(self) -> List[AgentEvent]:
        """Get all logged events."""
        return self._events.copy()
    
    def get_recent_events(self, n: int = 20) -> List[AgentEvent]:
        """Get the N most recent events."""
        return self._events[-n:]
    
    def clear(self) -> None:
        """Clear all logged events."""
        self._events.clear()


# =============================================================================
# Test Runner Configuration
# =============================================================================

@dataclass
class TestRunnerConfig:
    """Configuration for the test runner."""
    
    # Agent configuration
    max_chat_messages: int = 100
    
    # Session configuration
    app_name: str = "virtual_streamer_test"
    user_id: str = "test_user"
    session_id: str = "test_session"
    
    # Mock tool configuration
    tool_factory_config: MockToolFactoryConfig = field(default_factory=MockToolFactoryConfig)
    
    # Whether to load default conversation on init
    load_default_conversation: bool = True


# =============================================================================
# Test Runner
# =============================================================================

class VirtualStreamerTestRunner:
    """
    Test harness for the Virtual Streamer Agent.
    
    This runner uses composable mock context providers that each handle
    fetching and rendering their own section of the prompt.
    
    Usage:
        runner = VirtualStreamerTestRunner()
        await runner.setup()
        
        # Configure environment through individual providers
        runner.set_workload(WorkloadStatus.LOW)
        runner.set_queue_pending(2)
        
        # Add a chat message
        runner.add_chat_message("user123", "@virtualstreamer make a video about cats", is_mention=True)
        
        # Run agent iteration
        response = await runner.run_iteration("New mention from user123")
        
        # Inspect results
        tool_calls = runner.get_tool_calls()
        sent_messages = runner.get_sent_messages()
    """
    
    def __init__(self, config: Optional[TestRunnerConfig] = None):
        """
        Initialize the test runner.
        
        Args:
            config: Optional configuration
        """
        self.config = config or TestRunnerConfig()
        
        # Mock tool factory
        self.tool_factory = MockToolFactory(self.config.tool_factory_config)
        
        # Create composable mock context providers
        self.queue_provider = MockProcessingQueueContextProvider()
        self.system_provider = MockSystemStatusContextProvider()
        self.chat_provider = MockChatMessageContextProvider(
            max_messages=self.config.max_chat_messages
        )
        
        # Load default conversation for realistic testing
        if self.config.load_default_conversation:
            self.chat_provider.load_default_conversation()
        
        # Collect providers in order they should appear in prompt
        self.context_providers: List[ContextProviderProtocol] = [
            self.queue_provider,
            self.system_provider,
            self.chat_provider,
        ]
        
        # Event logging
        self.event_log = AgentEventLog()
        
        # Agent and runner (created in setup)
        self.agent: Optional[BaseLlmAgent] = None
        self.adk_runner: Optional[Runner] = None
        self.session_service: Optional[InMemorySessionService] = None
        self.instruction_provider: Optional[VirtualStreamerInstructionProvider] = None
        
        # Last prompt sent to agent (for debugging)
        self._last_prompt: Optional[str] = None
        self._last_user_message: Optional[str] = None
        
        logger.info("VirtualStreamerTestRunner initialized with composable providers")
    
    async def setup(self) -> None:
        """
        Set up the agent and ADK runner.
        
        Call this before running iterations.
        """
        # Get mock tools
        tools = self.tool_factory.get_tools()
        logger.info(f"Loaded {len(tools)} mock tools")
        
        # Create instruction provider with mock providers
        self.instruction_provider = VirtualStreamerInstructionProvider(
            context_providers=self.context_providers,
            tools=tools,
        )
        
        # Create agent with the instruction provider
        self.agent = BaseLlmAgent(
            name="virtual_streamer",
            instruction=self.instruction_provider,
            tools=tools,
            output_schema=None,
        )
        
        # Create session service and runner
        self.session_service = InMemorySessionService()
        self.adk_runner = Runner(
            agent=self.agent,
            app_name=self.config.app_name,
            session_service=self.session_service,
        )
        await self.session_service.create_session(
            app_name=self.config.app_name,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
        )
        
        logger.info("Test runner set up complete")
    
    # -------------------------------------------------------------------------
    # Context Configuration - Convenience methods that delegate to providers
    # -------------------------------------------------------------------------
    
    def set_workload(self, status: WorkloadStatus) -> None:
        """Set the workload status."""
        self.system_provider.set_workload(status)
    
    def set_queue_pending(self, count: int) -> None:
        """Set the pending video count (updates both queue and system providers)."""
        self.queue_provider.set_pending_count(count)
        self.system_provider.set_queue_pending(count)
    
    def set_queue_played(self, count: int) -> None:
        """Set the played video count."""
        self.queue_provider.set_played_count(count)
    
    def set_next_videos(self, videos: List[str]) -> None:
        """Set the next videos list."""
        self.queue_provider.set_next_videos(videos)
    
    def set_replay_mode(self, is_replaying: bool) -> None:
        """Set whether in replay mode."""
        self.queue_provider.set_replay_mode(is_replaying)
    
    def set_active_jobs(self, count: int) -> None:
        """Set active job count (updates both queue and system providers)."""
        self.queue_provider.set_active_jobs(count)
        self.system_provider.set_active_jobs(count)
    
    def set_chat_time_offset(self, minutes: float) -> None:
        """Set time offset for chat messages (simulates old conversation)."""
        self.chat_provider.set_time_offset(minutes)
    
    # -------------------------------------------------------------------------
    # Chat Management
    # -------------------------------------------------------------------------
    
    def add_chat_message(
        self,
        username: str,
        message: str,
        is_mention: bool = False,
    ) -> ChatMessage:
        """
        Add a new chat message.
        
        Args:
            username: Username of the sender
            message: Message content
            is_mention: Whether this mentions the bot
            
        Returns:
            The created ChatMessage
        """
        return self.chat_provider.add_message(username, message, is_mention)
    
    def get_chat_messages(self) -> List[ChatMessage]:
        """Get all dynamic chat messages."""
        return self.chat_provider.get_messages()
    
    def clear_chat_history(self) -> None:
        """Clear dynamic chat messages."""
        self.chat_provider.clear()
    
    def has_mentions(self) -> bool:
        """Check if there are any mention messages."""
        return self.chat_provider.has_mentions()
    
    # -------------------------------------------------------------------------
    # Running the Agent
    # -------------------------------------------------------------------------
    
    async def run_iteration(
        self,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single agent iteration.
        
        Args:
            user_message: Optional custom user message. If None, builds one
                         automatically based on context.
                         
        Returns:
            Dictionary with:
            - response: Agent's text response (if any)
            - tool_calls: List of tool calls made
            - events: List of events during this iteration
            - user_message: The user message that was sent
            - prompt: The full prompt that was generated
        """
        if self.adk_runner is None:
            await self.setup()
        
        # Build user message if not provided
        if user_message is None:
            user_message = self._build_user_message()
        self._last_user_message = user_message
        
        logger.info(f"Running iteration with message: {user_message[:100]}...")
        
        # Track events during this iteration
        iteration_events: List[AgentEvent] = []
        response_text: Optional[str] = None
        initial_tool_count = len(self.tool_factory.get_tool_calls())
        
        try:
            # Run the agent (context providers will render prompt dynamically)
            async for event in self.adk_runner.run_async(
                user_id=self.config.user_id,
                session_id=self.config.session_id,
                new_message=types.Content(
                    parts=[types.Part(text=user_message)],
                    role="user",
                ),
            ):
                # Log and process events
                event_type = type(event).__name__
                content = None
                
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        parts = event.content.parts
                        if parts and hasattr(parts[0], 'text'):
                            content = parts[0].text
                            response_text = content
                
                self.event_log.log(event_type, content, event)
                iteration_events.append(AgentEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type=event_type,
                    content=content,
                    raw_event=event,
                ))
            
            logger.info("Agent iteration completed")
            
        except Exception as e:
            logger.error(f"Agent iteration failed: {e}", exc_info=True)
            self.event_log.log("Error", str(e))
            iteration_events.append(AgentEvent(
                timestamp=datetime.now().isoformat(),
                event_type="Error",
                content=str(e),
            ))
        
        # Get tool calls made during this iteration
        all_tool_calls = self.tool_factory.get_tool_calls()
        new_tool_calls = all_tool_calls[initial_tool_count:]
        
        return {
            "response": response_text,
            "tool_calls": [tc.to_dict() for tc in new_tool_calls],
            "events": [e.to_dict() for e in iteration_events],
            "user_message": user_message,
        }
    
    def _build_user_message(self) -> str:
        """Build user message based on current provider state."""
        parts = []
        
        # Check for mentions
        if self.chat_provider.has_mentions():
            messages = self.chat_provider.get_messages()
            mentions = [m for m in messages if m.is_mention]
            recent_mentions = mentions[-3:]
            
            parts.append("Nouveaux messages qui te mentionnent:")
            for m in recent_mentions:
                parts.append(f"- @{m.username}: {m.message}")
        
        # Check queue status
        pending = self.queue_provider.get_pending_count()
        if pending < 3:
            parts.append(f"\n⚠️ La queue est presque vide ({pending} vidéos pending)")
        
        if not parts:
            parts.append("Vérifie l'état du stream et du chat.")
        
        return "\n".join(parts)
    
    async def get_current_prompt(self) -> str:
        """
        Get the current full prompt that would be sent to the agent.
        
        Useful for debugging and inspection.
        
        Returns:
            The rendered prompt string
        """
        if self.instruction_provider is None:
            await self.setup()
        
        # Create a dummy context since our providers don't use it
        class DummyContext:
            state = {}
        
        prompt = await self.instruction_provider(DummyContext())
        self._last_prompt = prompt
        return prompt
    
    # -------------------------------------------------------------------------
    # Inspection Methods
    # -------------------------------------------------------------------------
    
    def get_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls made by the agent."""
        return self.tool_factory.get_tool_calls()
    
    def get_recent_tool_calls(self, n: int = 10) -> List[ToolCall]:
        """Get the N most recent tool calls."""
        return self.tool_factory.get_recent_tool_calls(n)
    
    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """Get all messages sent by the agent to Twitch."""
        return self.tool_factory.get_sent_messages()
    
    def get_events(self) -> List[AgentEvent]:
        """Get all agent events."""
        return self.event_log.get_events()
    
    def get_recent_events(self, n: int = 20) -> List[AgentEvent]:
        """Get the N most recent events."""
        return self.event_log.get_recent_events(n)
    
    def get_last_prompt(self) -> Optional[str]:
        """Get the last prompt that was rendered."""
        return self._last_prompt
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message sent to the agent."""
        return self._last_user_message
    
    # -------------------------------------------------------------------------
    # Reset Methods
    # -------------------------------------------------------------------------
    
    def clear_history(self) -> None:
        """Clear all tool call and event history."""
        self.tool_factory.clear_history()
        self.event_log.clear()
    
    def reset_providers(self) -> None:
        """Reset all providers to their default state."""
        self.queue_provider.reset_to_defaults()
        self.system_provider.reset_to_defaults()
        self.chat_provider.reset_to_defaults()
        logger.info("All providers reset to defaults")
    
    def reset_session(self) -> None:
        """Reset the session (creates new session ID)."""
        self.config.session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        logger.info(f"Session reset to: {self.config.session_id}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_test(
    user_message: str,
    workload: WorkloadStatus = WorkloadStatus.LOW,
    queue_pending: int = 5,
    with_mention: bool = False,
) -> Dict[str, Any]:
    """
    Quick test helper for simple scenarios.
    
    Args:
        user_message: Message to send to the agent
        workload: System workload level
        queue_pending: Number of pending videos
        with_mention: Whether to add a mention message
        
    Returns:
        Result dictionary from run_iteration
    """
    runner = VirtualStreamerTestRunner()
    await runner.setup()
    
    runner.set_workload(workload)
    runner.set_queue_pending(queue_pending)
    
    if with_mention:
        runner.add_chat_message(
            "test_user",
            "@virtualstreamer " + user_message,
            is_mention=True,
        )
    
    return await runner.run_iteration(user_message)


async def test_stale_conversation(minutes_old: float = 10.0) -> Dict[str, Any]:
    """
    Test agent behavior with a stale conversation.
    
    Args:
        minutes_old: How many minutes old the conversation should be
        
    Returns:
        Result dictionary from run_iteration
    """
    runner = VirtualStreamerTestRunner()
    await runner.setup()
    
    runner.set_chat_time_offset(minutes_old)
    
    return await runner.run_iteration()


async def test_empty_queue() -> Dict[str, Any]:
    """
    Test agent behavior when queue is empty.
    
    Returns:
        Result dictionary from run_iteration
    """
    runner = VirtualStreamerTestRunner()
    await runner.setup()
    
    runner.set_queue_pending(0)
    runner.set_replay_mode(True)
    runner.set_workload(WorkloadStatus.LOW)
    
    return await runner.run_iteration()
