"""
ADK Agent Application Factory.

Creates a FastAPI application that serves ADK agents using Google's ADK framework.
This app can be mounted onto the main Virtual Streamer API for unified access.
"""

import logging
import pathlib

from google.adk.cli.fast_api import get_fast_api_app

logger = logging.getLogger(__name__)

# Path to the agents directory (relative to this file)
AGENTS_DIR = pathlib.Path(__file__).parent.parent / "agents"


def create_adk_app(
    agents_dir: str | pathlib.Path | None = None,
    web: bool = True,
) -> "FastAPI":
    """Create a FastAPI application serving ADK agents.

    This uses Google ADK's get_fast_api_app to create an application that
    auto-discovers and serves all agents with `root_agent` exposed at module level.

    The ADK app provides endpoints like:
    - GET /list-apps - List available agents
    - POST /run/{agent_name} - Run an agent
    - WebSocket /run_sse/{agent_name} - Stream agent execution

    Args:
        agents_dir: Path to the directory containing ADK agents.
                   Defaults to virtual_streamer/agents/
        web: Whether to include the web UI (ADK dev UI)

    Returns:
        FastAPI application configured to serve ADK agents

    Example:
        ```python
        from virtual_streamer.api.adk_app import create_adk_app
        
        adk_app = create_adk_app()
        # Mount onto main app
        main_app.mount("/adk", adk_app)
        ```
    """
    if agents_dir is None:
        agents_dir = AGENTS_DIR

    agents_path = pathlib.Path(agents_dir)
    
    if not agents_path.exists():
        raise FileNotFoundError(f"Agents directory not found: {agents_path}")

    logger.info(f"Creating ADK app with agents from: {agents_path}")

    # Use Google ADK's built-in FastAPI app factory
    # This auto-discovers agents that expose `root_agent` at module level
    adk_app = get_fast_api_app(
        agents_dir=str(agents_path),
        web=web,
    )

    logger.info("ADK app created successfully")

    return adk_app

