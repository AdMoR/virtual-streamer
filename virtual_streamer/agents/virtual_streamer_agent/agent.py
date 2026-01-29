"""
Virtual Streamer Agent.

ADK agent that controls a Twitch streaming channel through tools
for video creation and chat interaction.
"""

import logging
from typing import Any, List, Optional

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.virtual_streamer_agent.prompt import (
    VirtualStreamerInstructionProvider
)
from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)


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
        context_providers: List[Any],
    ):
        """
        Initialize the Virtual Streamer Agent.
        
        Args:
            tools: List of tool functions available to the agent
            max_chat_messages: Maximum chat messages to include in context
        """
        instruction_provider = VirtualStreamerInstructionProvider(
            context_providers=context_providers,
            tools=tools,
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
        context_providers=[]
    )


def get_virtual_streamer_agent_dummy_tools(
) -> VirtualStreamerAgent:
    """
    Factory function to create a Virtual Streamer Agent.

    Args:
        tools: List of tools to provide to the agent. If None, an empty list is used.
        max_chat_messages: Maximum chat messages to include in context

    Returns:
        Configured VirtualStreamerAgent instance
    """
    def create_video(title: str):
        return None

    def answer_chat_message(msg: str):
        return None

    tools = [
        create_video, answer_chat_message,
    ]

    return VirtualStreamerAgent(
        tools=tools,
        context_providers=[]
    )


root_agent = get_virtual_streamer_agent_dummy_tools()
