"""
Mock tools for testing the Virtual Streamer Agent.

This module provides mock implementations of all agent tools that can be
configured to simulate various scenarios without requiring real infrastructure.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Call Log
# =============================================================================

@dataclass
class ToolCall:
    """Record of a tool call made by the agent."""
    
    timestamp: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
        }


class ToolCallLog:
    """Collects and stores tool calls for inspection."""
    
    def __init__(self):
        self._calls: List[ToolCall] = []
    
    def log(self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log a tool call."""
        call = ToolCall(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        self._calls.append(call)
        logger.info(f"Tool call logged: {tool_name}({arguments}) -> {result}")
    
    def get_calls(self) -> List[ToolCall]:
        """Get all logged calls."""
        return self._calls.copy()
    
    def get_recent_calls(self, n: int = 10) -> List[ToolCall]:
        """Get the N most recent calls."""
        return self._calls[-n:]
    
    def clear(self) -> None:
        """Clear all logged calls."""
        self._calls.clear()


# =============================================================================
# Mock Tool Configuration
# =============================================================================

@dataclass
class MockToolConfig:
    """Configuration for mock tool behavior."""
    
    # Success rate (0.0 to 1.0)
    success_rate: float = 1.0
    
    # Custom response data
    custom_response: Optional[Dict[str, Any]] = None
    
    # Whether to simulate delay (not actually sleeping)
    simulate_delay: bool = False
    delay_seconds: float = 0.0


# =============================================================================
# Mock Tool Implementations
# =============================================================================

def create_mock_create_video(
    tool_log: ToolCallLog,
    config: Optional[MockToolConfig] = None,
) -> Callable:
    """
    Create a mock create_video tool.
    
    Args:
        tool_log: Log to record tool calls
        config: Optional configuration for behavior
        
    Returns:
        Mock tool function
    """
    cfg = config or MockToolConfig()
    
    async def mock_create_video(
        title: str,
        story_template_id: str = "cest_pas_sorcier",
        stream_id: Optional[str] = None,
    ) -> dict:
        """
        Create a new video and add it to the streaming queue.
        
        This is a MOCK implementation for testing. The video will not
        actually be created, but the call will be logged.
        
        Args:
            title: The topic/title for the video
            story_template_id: Which story template to use for generation
            stream_id: Stream ID to add the video to
            
        Returns:
            dict with job_id, status, success, and message
        """
        arguments = {
            "title": title,
            "story_template_id": story_template_id,
            "stream_id": stream_id,
        }
        
        # Check success rate
        import random
        if random.random() > cfg.success_rate:
            result = {
                "success": False,
                "error": "Simulated failure",
                "message": "Erreur simulée lors de la création de la vidéo",
            }
        elif cfg.custom_response:
            result = cfg.custom_response
        else:
            job_id = f"mock-job-{uuid.uuid4().hex[:8]}"
            result = {
                "success": True,
                "job_id": job_id,
                "status": "pending",
                "message": f"[MOCK] Vidéo '{title}' en cours de création !",
            }
        
        tool_log.log("create_video", arguments, result)
        return result
    
    # Preserve function metadata for ADK
    mock_create_video.__name__ = "create_video"
    mock_create_video.__doc__ = """
        Create a new video and add it to the streaming queue.
        
        This tool submits a video generation job. The video will be
        generated asynchronously and automatically added to the playlist when complete.
        
        Args:
            title: The topic/title for the video (e.g., "Pourquoi les chats retombent toujours sur leurs pattes")
            story_template_id: Which story template to use for generation (default: cest_pas_sorcier)
            stream_id: Stream ID to add the video to
            
        Returns:
            dict with job_id, status, success, and message
    """
    
    return mock_create_video


def create_mock_send_message(
    tool_log: ToolCallLog,
    sent_messages: Optional[List[Dict[str, Any]]] = None,
    config: Optional[MockToolConfig] = None,
) -> Callable:
    """
    Create a mock send_twitch_message tool.
    
    Args:
        tool_log: Log to record tool calls
        sent_messages: Optional list to collect sent messages
        config: Optional configuration for behavior
        
    Returns:
        Mock tool function
    """
    cfg = config or MockToolConfig()
    messages_store = sent_messages if sent_messages is not None else []
    
    async def mock_send_twitch_message(message: str) -> dict:
        """
        Send a message to the Twitch chat.
        
        This is a MOCK implementation for testing. The message will not
        actually be sent to Twitch, but will be logged and stored.
        
        Args:
            message: The message to send to the chat
            
        Returns:
            dict with success, message, and optional error
        """
        # Truncate message if too long (same as real implementation)
        max_length = 500
        truncated_message = message[:max_length]
        
        arguments = {"message": message}
        
        # Check success rate
        import random
        if random.random() > cfg.success_rate:
            result = {
                "success": False,
                "error": "Simulated failure",
                "message": truncated_message,
            }
        elif cfg.custom_response:
            result = cfg.custom_response
        else:
            result = {
                "success": True,
                "message": truncated_message,
            }
            
            # Store the sent message
            messages_store.append({
                "timestamp": datetime.now().isoformat(),
                "message": truncated_message,
                "source": "agent",
            })
        
        tool_log.log("send_twitch_message", arguments, result)
        return result
    
    # Preserve function metadata for ADK
    mock_send_twitch_message.__name__ = "send_twitch_message"
    mock_send_twitch_message.__doc__ = """
        Send a message to the Twitch chat.
        
        The message will be truncated to 500 characters if longer.
        
        Args:
            message: The message to send to the chat
            
        Returns:
            dict with success, message, and optional error
    """
    
    return mock_send_twitch_message


# =============================================================================
# Mock Tool Factory
# =============================================================================

@dataclass
class MockToolFactoryConfig:
    """Configuration for the MockToolFactory."""
    
    # Individual tool configs
    create_video_config: MockToolConfig = field(default_factory=MockToolConfig)
    send_message_config: MockToolConfig = field(default_factory=MockToolConfig)
    
    # Whether to include each tool
    include_create_video: bool = True
    include_send_message: bool = True


class MockToolFactory:
    """
    Factory that produces mock tools for testing.
    
    Usage:
        factory = MockToolFactory()
        tools = factory.get_tools()
        
        # Get tool call history
        calls = factory.get_tool_calls()
        
        # Get messages sent by agent
        messages = factory.get_sent_messages()
    """
    
    def __init__(self, config: Optional[MockToolFactoryConfig] = None):
        """
        Initialize the factory.
        
        Args:
            config: Optional configuration for tool behavior
        """
        self.config = config or MockToolFactoryConfig()
        self.tool_log = ToolCallLog()
        self.sent_messages: List[Dict[str, Any]] = []
        
        self._tools: Optional[List[Callable]] = None
    
    def get_tools(self) -> List[Callable]:
        """
        Get the list of mock tools.
        
        Returns:
            List of mock tool functions
        """
        if self._tools is None:
            self._tools = self._build_tools()
        return self._tools
    
    def _build_tools(self) -> List[Callable]:
        """Build the tool list based on configuration."""
        tools = []
        
        if self.config.include_create_video:
            tools.append(create_mock_create_video(
                tool_log=self.tool_log,
                config=self.config.create_video_config,
            ))
        
        if self.config.include_send_message:
            tools.append(create_mock_send_message(
                tool_log=self.tool_log,
                sent_messages=self.sent_messages,
                config=self.config.send_message_config,
            ))
        
        logger.info(f"MockToolFactory built {len(tools)} tools")
        return tools
    
    def get_tool_calls(self) -> List[ToolCall]:
        """Get all tool calls made."""
        return self.tool_log.get_calls()
    
    def get_recent_tool_calls(self, n: int = 10) -> List[ToolCall]:
        """Get the N most recent tool calls."""
        return self.tool_log.get_recent_calls(n)
    
    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """Get all messages sent by the agent."""
        return self.sent_messages.copy()
    
    def clear_history(self) -> None:
        """Clear all tool call and message history."""
        self.tool_log.clear()
        self.sent_messages.clear()
    
    def rebuild_tools(self) -> List[Callable]:
        """Rebuild tools with current configuration."""
        self._tools = self._build_tools()
        return self._tools
