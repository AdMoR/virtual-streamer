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
        max_chat_messages: int = 100,
    ):
        """
        Initialize the Virtual Streamer Agent.
        
        Args:
            tools: List of tool functions available to the agent
            max_chat_messages: Maximum chat messages to include in context
        """
        instruction_provider = VirtualStreamerInstructionProvider(
            tools=tools,
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
        max_chat_messages=100,
    )


#root_agent = get_virtual_streamer_agent_dummy_tools()


def get_stock_price(symbol: str):
    """
    Retrieves the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price, or None if an error occurs.
    """
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


APP_NAME = "stock_app"
USER_ID = "1234"
SESSION_ID = "session1234"

def get_stock_price(symbol: str):
    """
    Retrieves the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "GOOG").

    Returns:
        float: The current stock price, or None if an error occurs.
    """
    return 10


def get_temperature(city: str, day: Optional[str] = None) -> float:
    """
    Retrieves the current temperature in a city on a given day

    Args:
        city (str): the city name
        day (str | None): the day (optional)

    Returns:
        float: The temp at the location
    """
    return 10


root_agent = Agent(
    model= LiteLlm(model="openai/gpt-oss-120b",api_base="http://100.114.182.89:8000/v1"),
    name='stock_agent',
    instruction= 'You are an agent who retrieves stock prices. If a ticker symbol is provided, fetch the current price. If only a company name is given, first perform a Google search to find the correct ticker symbol before retrieving the stock price. If the provided ticker symbol is invalid or data cannot be retrieved, inform the user that the stock price could not be found.',
    description='This agent specializes in retrieving real-time stock prices and temperature of the city. Given a stock ticker symbol (e.g., AAPL, GOOG, MSFT) or the stock name, use the tools and reliable data sources to provide the most up-to-date price.',
    tools=[get_stock_price, get_temperature], # You can add Python functions directly to the tools list; they will be automatically wrapped as FunctionTools.
)


