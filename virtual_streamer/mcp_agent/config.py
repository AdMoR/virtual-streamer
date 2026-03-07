"""
MCP Agent configuration.

All settings are loaded from environment variables.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the MCP-based agentic loop."""

    # MCP server command (override for testing with mock server)
    mcp_server_command: list[str] = field(
        default_factory=lambda: _load_mcp_server_command()
    )

    # LLM endpoint (OpenAI-compatible)
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", "not-needed")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o")
    )

    # Loop tuning
    loop_interval: float = field(
        default_factory=lambda: float(os.environ.get("LOOP_INTERVAL_SECONDS", "5.0"))
    )
    max_history: int = field(
        default_factory=lambda: int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))
    )

    # REST API (forwarded to MCP subprocess)
    api_url: str = field(
        default_factory=lambda: os.environ.get("API_URL", "http://localhost:8000")
    )
    stream_id: str = field(
        default_factory=lambda: os.environ.get("STREAM_ID", "default")
    )
    programmation_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("PROGRAMMATION_ID")
    )

    # Twitch credentials (forwarded to MCP subprocess)
    twitch_client_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("TWITCH_CLIENT_ID")
    )
    twitch_client_secret: Optional[str] = field(
        default_factory=lambda: os.environ.get("TWITCH_CLIENT_SECRET")
    )
    twitch_refresh_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("TWITCH_REFRESH_TOKEN")
    )
    twitch_channel: Optional[str] = field(
        default_factory=lambda: os.environ.get("TWITCH_CHANNEL")
    )
    twitch_bot_username: str = field(
        default_factory=lambda: os.environ.get("TWITCH_BOT_USERNAME", "virtualstreamerbot")
    )

    def to_mcp_env(self) -> dict:
        """Return env vars to forward to the MCP server subprocess."""
        env = {
            "API_URL": self.api_url,
            "STREAM_ID": self.stream_id,
        }
        if self.programmation_id:
            env["PROGRAMMATION_ID"] = self.programmation_id
        if self.twitch_client_id:
            env["TWITCH_CLIENT_ID"] = self.twitch_client_id
        if self.twitch_client_secret:
            env["TWITCH_CLIENT_SECRET"] = self.twitch_client_secret
        if self.twitch_refresh_token:
            env["TWITCH_REFRESH_TOKEN"] = self.twitch_refresh_token
        if self.twitch_channel:
            env["TWITCH_CHANNEL"] = self.twitch_channel
        env["TWITCH_BOT_USERNAME"] = self.twitch_bot_username
        return env


def _load_mcp_server_command() -> list[str]:
    """Load MCP server command from MCP_SERVER_COMMAND env var or use default.

    The env var is parsed as a JSON array if it starts with '[', otherwise
    split on colons (e.g. "python:-m:virtual_streamer.mcp_agent.mock_server").
    """
    raw = os.environ.get("MCP_SERVER_COMMAND")
    if not raw:
        return ["python", "-m", "virtual_streamer.mcp_server"]
    if raw.startswith("["):
        return json.loads(raw)
    return raw.split(":")
