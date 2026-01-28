"""
Virtual Streamer Agent.

ADK agent that controls a Twitch streaming channel through tools
for video creation and chat interaction.
"""

import logging
from functools import lru_cache
from typing import Any, List, Optional

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.virtual_streamer.prompt import (
    VIRTUAL_STREAMER_SYSTEM_PROMPT,
    build_full_context,
)
from virtual_streamer.agents.virtual_streamer.schema import (
    QueueInfo,
    SystemStatus,
    ChatMessage,
    WorkloadStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# State Keys
# =============================================================================

# Keys for reading context from state
STATE_QUEUE_INFO = "queue_info"
STATE_SYSTEM_STATUS = "system_status"
STATE_CHAT_MESSAGES = "chat_messages"


# =============================================================================
# Instruction Provider
# =============================================================================

class VirtualStreamerInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that builds context-aware prompts.
    
    Reads queue info, system status, and chat messages from state
    and formats them into the agent's instruction.
    """
    
    def __init__(self, max_chat_messages: int = 100):
        """
        Initialize the instruction provider.
        
        Args:
            max_chat_messages: Maximum number of chat messages to include in context
        """
        self.max_chat_messages = max_chat_messages
    
    async def __call__(self, context: ReadonlyContext) -> str:
        """Generate the instruction with current context."""
        # Extract context from state
        queue_info = self._get_queue_info(context)
        system_status = self._get_system_status(context)
        messages = self._get_chat_messages(context)
        
        # Build dynamic context section
        dynamic_context = build_full_context(
            queue_info=queue_info,
            system_status=system_status,
            messages=messages,
            max_chat_messages=self.max_chat_messages,
        )
        
        # Combine system prompt with dynamic context
        full_prompt = f"{VIRTUAL_STREAMER_SYSTEM_PROMPT}\n\n---\n\n{dynamic_context}"
        
        return full_prompt
    
    def _get_queue_info(self, context: ReadonlyContext) -> QueueInfo:
        """Extract queue info from state or return defaults."""
        raw = context.state.get(STATE_QUEUE_INFO)
        if raw is None:
            return QueueInfo(
                pending_count=0,
                played_count=0,
                next_videos=[],
                is_replaying=False,
                active_jobs=0,
            )
        if isinstance(raw, QueueInfo):
            return raw
        if isinstance(raw, dict):
            return QueueInfo(**raw)
        return QueueInfo(
            pending_count=0,
            played_count=0,
            next_videos=[],
            is_replaying=False,
            active_jobs=0,
        )
    
    def _get_system_status(self, context: ReadonlyContext) -> SystemStatus:
        """Extract system status from state or return defaults."""
        raw = context.state.get(STATE_SYSTEM_STATUS)
        if raw is None:
            return SystemStatus(
                workload=WorkloadStatus.UNKNOWN,
                active_jobs=0,
                queue_pending=0,
            )
        if isinstance(raw, SystemStatus):
            return raw
        if isinstance(raw, dict):
            return SystemStatus(**raw)
        return SystemStatus(
            workload=WorkloadStatus.UNKNOWN,
            active_jobs=0,
            queue_pending=0,
        )
    
    def _get_chat_messages(self, context: ReadonlyContext) -> List[ChatMessage]:
        """Extract chat messages from state or return empty list."""
        raw = context.state.get(STATE_CHAT_MESSAGES)
        if raw is None:
            return []
        if isinstance(raw, list):
            messages = []
            for item in raw:
                if isinstance(item, ChatMessage):
                    messages.append(item)
                elif isinstance(item, dict):
                    messages.append(ChatMessage(**item))
            return messages
        return []


# =============================================================================
# Agent Class
# =============================================================================

class VirtualStreamerAgent(BaseLlmAgent):
    """
    Virtual Streamer Agent that controls a Twitch channel.
    
    This agent:
    - Monitors Twitch chat and responds to viewers
    - Creates videos based on requests or proactively
    - Manages the video queue to ensure fresh content
    
    Tools are injected at construction time via the ToolFactory.
    """
    
    def __init__(
        self,
        tools: List[Any],
        max_chat_messages: int = 100,
    ):
        """
        Initialize the Virtual Streamer Agent.
        
        Args:
            tools: List of tool functions available to the agent
            max_chat_messages: Maximum chat messages to include in context
        """
        instruction_provider = VirtualStreamerInstructionProvider(
            max_chat_messages=max_chat_messages
        )
        
        super().__init__(
            name="virtual_streamer",
            instruction=instruction_provider,
            tools=tools,
            # No output_schema - agent uses tools directly
            output_schema=None,
        )
        
        logger.info(
            f"VirtualStreamerAgent initialized with {len(tools)} tools: "
            f"{[getattr(t, '__name__', str(t)) for t in tools]}"
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_virtual_streamer_agent(
    tools: Optional[List[Any]] = None,
    max_chat_messages: int = 100,
) -> VirtualStreamerAgent:
    """
    Factory function to create a Virtual Streamer Agent.
    
    Args:
        tools: List of tools to provide to the agent. If None, an empty list is used.
        max_chat_messages: Maximum chat messages to include in context
        
    Returns:
        Configured VirtualStreamerAgent instance
    """
    if tools is None:
        tools = []
        logger.warning("Creating VirtualStreamerAgent with no tools")
    
    return VirtualStreamerAgent(
        tools=tools,
        max_chat_messages=max_chat_messages,
    )
