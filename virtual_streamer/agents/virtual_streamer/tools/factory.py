"""
Tool Factory for the Virtual Streamer Agent.

This module provides the ToolFactory class that dynamically builds
the tool list from YAML configuration.
"""

import importlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ToolAvailability:
    """Time-based availability rules for a tool."""
    enabled_hours: List[int] = field(default_factory=lambda: list(range(24)))
    timezone: str = "UTC"


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
    
    def get_availability(self) -> Optional[ToolAvailability]:
        """Parse availability config into ToolAvailability object."""
        if self.availability is None:
            return None
        return ToolAvailability(
            enabled_hours=self.availability.get("enabled_hours", list(range(24))),
            timezone=self.availability.get("timezone", "UTC"),
        )


@dataclass
class ToolsSettings:
    """Global settings for tools from config."""
    max_tools: int = 10
    availability_check_interval: int = 60


# =============================================================================
# Tool Factory
# =============================================================================

class ToolFactory:
    """
    Factory that builds tool list from YAML configuration.
    
    Features:
    - Dynamic tool loading from config file
    - Pre-filled parameters for scenario variants
    - Time-based tool availability
    - Hot-reload capability (call reload() to refresh config)
    
    Usage:
        factory = ToolFactory("configs/virtual_streamer_tools.yaml")
        tools = factory.get_available_tools()
        
        # Pass tools to agent
        agent = VirtualStreamerAgent(tools=tools)
    """
    
    DEFAULT_CONFIG_PATH = "configs/virtual_streamer_tools.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ToolFactory.
        
        Args:
            config_path: Path to the YAML config file. If None, uses
                        TOOLS_CONFIG env var or default path.
        """
        self.config_path = config_path or os.environ.get(
            "TOOLS_CONFIG", self.DEFAULT_CONFIG_PATH
        )
        self.config: Dict[str, Any] = {}
        self.tool_configs: List[ToolConfig] = []
        self.settings = ToolsSettings()
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and parse the configuration file."""
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(
                f"Config file not found: {self.config_path}. Using empty config."
            )
            self.config = {"tools": [], "settings": {}}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            self.config = {"tools": [], "settings": {}}
        
        # Parse tool configs
        raw_tools = self.config.get("tools", [])
        self.tool_configs = []
        for t in raw_tools:
            try:
                self.tool_configs.append(ToolConfig(
                    name=t["name"],
                    enabled=t.get("enabled", True),
                    module=t["module"],
                    function=t["function"],
                    description=t.get("description", ""),
                    defaults=t.get("defaults", {}),
                    availability=t.get("availability"),
                ))
            except KeyError as e:
                logger.warning(f"Skipping tool with missing field: {e}")
        
        # Parse settings
        raw_settings = self.config.get("settings", {})
        self.settings = ToolsSettings(
            max_tools=raw_settings.get("max_tools", 10),
            availability_check_interval=raw_settings.get(
                "availability_check_interval", 60
            ),
        )
        
        logger.info(
            f"Loaded {len(self.tool_configs)} tool configs from {self.config_path}"
        )
    
    def reload(self) -> None:
        """Reload configuration from file."""
        logger.info(f"Reloading tool config from {self.config_path}")
        self._load_config()
    
    def get_available_tools(self) -> List[Callable]:
        """
        Build list of currently available tools.
        
        Filters by:
        - enabled flag
        - time-based availability rules
        
        Returns:
            List of tool functions ready to be used by the agent
        """
        tools = []
        now = datetime.now()
        
        for tc in self.tool_configs:
            if not tc.enabled:
                logger.debug(f"Tool {tc.name} is disabled")
                continue
            
            if not self._check_availability(tc, now):
                logger.debug(f"Tool {tc.name} is not available at current time")
                continue
            
            try:
                tool_fn = self._build_tool(tc)
                tools.append(tool_fn)
                logger.debug(f"Tool {tc.name} added to available tools")
            except Exception as e:
                logger.error(f"Failed to build tool {tc.name}: {e}")
        
        # Limit to max_tools
        if len(tools) > self.settings.max_tools:
            logger.warning(
                f"Limiting tools from {len(tools)} to {self.settings.max_tools}"
            )
            tools = tools[:self.settings.max_tools]
        
        logger.info(f"Returning {len(tools)} available tools")
        return tools
    
    def _check_availability(self, tc: ToolConfig, now: datetime) -> bool:
        """Check if tool is available based on time rules."""
        availability = tc.get_availability()
        if availability is None:
            return True
        
        try:
            tz = ZoneInfo(availability.timezone)
            local_now = now.astimezone(tz)
            return local_now.hour in availability.enabled_hours
        except Exception as e:
            logger.warning(
                f"Failed to check availability for {tc.name}: {e}. Assuming available."
            )
            return True
    
    def _build_tool(self, tc: ToolConfig) -> Callable:
        """
        Import and wrap tool function with defaults.
        
        Args:
            tc: Tool configuration
            
        Returns:
            Tool function ready for use with ADK
        """
        # Import the module and get the function
        module = importlib.import_module(tc.module)
        fn = getattr(module, tc.function)
        
        # Apply defaults as partial if present
        if tc.defaults:
            original_fn = fn
            fn = partial(fn, **tc.defaults)
            
            # Preserve function signature for introspection
            # ADK needs to see the remaining parameters
            @wraps(original_fn)
            async def wrapped_fn(*args, **kwargs):
                # Merge defaults with provided kwargs
                merged_kwargs = {**tc.defaults, **kwargs}
                return await original_fn(*args, **merged_kwargs)
            
            fn = wrapped_fn
        
        # Update function metadata for ADK
        fn.__name__ = tc.name
        fn.__doc__ = tc.description
        
        return fn
    
    def get_tool_names(self) -> List[str]:
        """Get list of all configured tool names (regardless of availability)."""
        return [tc.name for tc in self.tool_configs]
    
    def get_enabled_tool_names(self) -> List[str]:
        """Get list of enabled tool names (regardless of time availability)."""
        return [tc.name for tc in self.tool_configs if tc.enabled]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_factory: Optional[ToolFactory] = None


def get_default_tool_factory() -> ToolFactory:
    """Get or create the default ToolFactory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ToolFactory()
    return _default_factory


def get_tools_from_config(config_path: Optional[str] = None) -> List[Callable]:
    """
    Convenience function to get tools from config.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        List of available tool functions
    """
    if config_path:
        factory = ToolFactory(config_path)
    else:
        factory = get_default_tool_factory()
    return factory.get_available_tools()
