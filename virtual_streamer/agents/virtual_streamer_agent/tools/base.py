"""
Base definitions for Virtual Streamer tools.

This module provides:
- Tool registry for available tools
- Base types and interfaces
- Shared configuration
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# Configuration from Environment
# =============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:8000")
STREAM_ID = os.environ.get("STREAM_ID", "default")


# =============================================================================
# Tool Configuration Types
# =============================================================================

@dataclass
class ToolConfig:
    """Configuration for a single tool."""
    
    name: str
    enabled: bool
    module: str
    function: str
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    availability: Optional[Dict[str, Any]] = None


@dataclass
class ToolsSettings:
    """Global settings for tools."""
    
    max_tools: int = 10
    availability_check_interval: int = 60


# =============================================================================
# Tool Registry
# =============================================================================

# Registry of all available tool implementations
# Maps tool function name to the actual function
_TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str):
    """
    Decorator to register a tool function.
    
    Usage:
        @register_tool("create_video")
        async def create_video(title: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        _TOOL_REGISTRY[name] = func
        return func
    return decorator


def get_registered_tool(name: str) -> Optional[Callable]:
    """Get a tool function from the registry by name."""
    return _TOOL_REGISTRY.get(name)


def list_registered_tools() -> List[str]:
    """List all registered tool names."""
    return list(_TOOL_REGISTRY.keys())
