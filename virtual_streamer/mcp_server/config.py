"""
MCP Server configuration.

All settings are loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""

    # REST API connection
    api_url: str = field(
        default_factory=lambda: os.environ.get("API_URL", "http://localhost:8000")
    )
    stream_id: str = field(
        default_factory=lambda: os.environ.get("STREAM_ID", "default")
    )
    programmation_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("PROGRAMMATION_ID")
    )

    # Twitch credentials
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

    @property
    def has_twitch_credentials(self) -> bool:
        """Whether all required Twitch credentials are set."""
        return all([
            self.twitch_client_id,
            self.twitch_client_secret,
            self.twitch_refresh_token,
            self.twitch_channel,
        ])
