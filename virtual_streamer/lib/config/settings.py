"""
Configuration management for ADK agents using Pydantic Settings.

This module provides configuration classes for agent settings, following
the patterns established in the codebase (see video_generation/config.py).
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for the LLM model used by an agent."""

    provider: str = Field(
        default="google",
        description="LLM provider: google (Gemini), anthropic, openai, or litellm",
    )
    model: str = Field(
        default="gemini-2.0-flash",
        description="Model identifier (e.g., gemini-2.0-flash, claude-sonnet-4-5-20250929)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate (None for model default)",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Top-k sampling parameter",
    )


class AgentMetadata(BaseModel):
    """Metadata for an agent (for tracking and documentation)."""

    version: str = Field(
        default="1.0.0",
        description="Agent version following semantic versioning",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the agent's purpose",
    )
    owners: List[str] = Field(
        default_factory=list,
        description="List of owner identifiers (emails or usernames)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and discovery",
    )


class TransferPolicy(BaseModel):
    """Policy for agent transfer behavior in multi-agent systems."""

    allow_transfer: bool = Field(
        default=True,
        description="Whether this agent can transfer control to other agents",
    )
    allowed_targets: List[str] = Field(
        default_factory=list,
        description="List of agent names this agent can transfer to (empty = all)",
    )


class AgentConfig(BaseModel):
    """
    Complete configuration for an ADK agent.

    This configuration can be loaded from YAML files, environment variables,
    or constructed programmatically. It provides all settings needed to
    instantiate and configure an agent.

    Example YAML (configs/agents/my_agent.yaml):
        name: my_agent
        model:
          provider: google
          model: gemini-2.0-flash
          temperature: 0.7
        metadata:
          version: "1.0.0"
          description: "Agent for processing user queries"
          owners: ["team@example.com"]
    """

    name: str = Field(
        description="Unique agent name (must match across code and config)",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="LLM model configuration",
    )
    metadata: AgentMetadata = Field(
        default_factory=AgentMetadata,
        description="Agent metadata for tracking",
    )
    transfer_policy: TransferPolicy = Field(
        default_factory=TransferPolicy,
        description="Transfer behavior in multi-agent systems",
    )
    include_contents: bool = Field(
        default=True,
        description="Whether to include full content in responses",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AgentConfig":
        """Load agent configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            AgentConfig instance with settings from the file.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        import yaml

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {yaml_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config: expected dict, got {type(data)}")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(mode="json")

    def save_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            yaml_path: Path where the YAML file will be written.
        """
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class AgentConfigRegistry(BaseSettings):
    """
    Registry for loading agent configurations.

    This settings class provides environment-based configuration for
    the agent configuration system itself.
    """

    model_config = SettingsConfigDict(
        env_prefix="ADK_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=(".env", ".env.public"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_dir: str = Field(
        default="configs/agents",
        description="Directory containing agent YAML configurations",
    )
    default_provider: str = Field(
        default="google",
        description="Default LLM provider for agents",
    )
    default_model: str = Field(
        default="gemini-2.0-flash",
        description="Default model for agents",
    )

    def get_config_path(self, agent_name: str) -> Path:
        """Get the expected config file path for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Path to the agent's YAML config file.
        """
        return Path(self.config_dir) / f"{agent_name}.yaml"

    def load_agent_config(self, agent_name: str) -> AgentConfig:
        """Load configuration for a specific agent.

        Attempts to load from YAML file first, falls back to defaults
        if no config file exists.

        Args:
            agent_name: Name of the agent to load config for.

        Returns:
            AgentConfig for the specified agent.
        """
        config_path = self.get_config_path(agent_name)

        if config_path.exists():
            return AgentConfig.from_yaml(config_path)

        # Return default config if no file exists
        return AgentConfig(
            name=agent_name,
            model=ModelConfig(
                provider=self.default_provider,
                model=self.default_model,
            ),
        )


# Cached singleton instances
_registry: Optional[AgentConfigRegistry] = None
_agent_configs: Dict[str, AgentConfig] = {}


@lru_cache
def get_agent_configuration() -> AgentConfigRegistry:
    """Get the agent configuration registry (singleton).

    Returns:
        AgentConfigRegistry instance for loading agent configurations.
    """
    return AgentConfigRegistry()


def get_config_for_agent(agent_name: str) -> AgentConfig:
    """Get configuration for a specific agent (cached).

    Args:
        agent_name: Name of the agent.

    Returns:
        AgentConfig for the specified agent.
    """
    if agent_name not in _agent_configs:
        registry = get_agent_configuration()
        _agent_configs[agent_name] = registry.load_agent_config(agent_name)
    return _agent_configs[agent_name]


def clear_config_cache() -> None:
    """Clear all cached configurations.

    Useful for testing or when configuration files have changed.
    """
    global _agent_configs
    _agent_configs = {}
    get_agent_configuration.cache_clear()

