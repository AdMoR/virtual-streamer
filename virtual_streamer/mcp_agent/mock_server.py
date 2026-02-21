"""
Mock MCP server for testing the mcp_agent.

Exposes the same tool surface as the real server but with static, configurable
responses — no REST API or Twitch connection required.

State is controlled via env vars at startup:
  MOCK_CHAT_JSON    JSON array of chat message dicts
  MOCK_QUEUE_JSON   JSON dict with queue status fields

Run as subprocess:
  python -m virtual_streamer.mcp_agent.mock_server
"""

import json
import os

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Default mock data
# ---------------------------------------------------------------------------

DEFAULT_CHAT: list[dict] = [
    {
        "username": "viewer1",
        "message": "Bonjour tout le monde!",
        "timestamp": "2026-02-21T10:00:00",
        "is_mention": False,
    },
    {
        "username": "viewer2",
        "message": "@allojesuschrist peux-tu faire une vidéo sur l'IA ?",
        "timestamp": "2026-02-21T10:00:05",
        "is_mention": True,
    },
]

DEFAULT_QUEUE: dict = {
    "programmation_id": "mock-prog-1",
    "pending_count": 2,
    "played_count": 5,
    "is_replaying": False,
}

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(name="mock-virtual-streamer")


def _get_chat() -> list[dict]:
    raw = os.environ.get("MOCK_CHAT_JSON")
    if raw:
        return json.loads(raw)
    return DEFAULT_CHAT


def _get_queue() -> dict:
    raw = os.environ.get("MOCK_QUEUE_JSON")
    if raw:
        return json.loads(raw)
    return DEFAULT_QUEUE


# ---------------------------------------------------------------------------
# Tools — Chat
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_chat_messages(limit: int = 50, mentions_only: bool = False) -> list[dict]:
    """Get recent Twitch chat messages."""
    messages = _get_chat()
    if mentions_only:
        messages = [m for m in messages if m.get("is_mention")]
    return messages[:limit]


@mcp.tool()
async def send_twitch_message(message: str) -> dict:
    """Send a message to Twitch chat."""
    return {"success": True, "message": message[:500]}


# ---------------------------------------------------------------------------
# Tools — Queue / Programmation
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_queue_status() -> dict:
    """Get current video queue status."""
    return _get_queue()


@mcp.tool()
async def get_active_programmation() -> dict:
    """Get the currently active programmation."""
    return {"programmation_id": "mock-prog-1", "name": "Mock Stream"}


@mcp.tool()
async def list_programmations() -> dict:
    """List all programmations."""
    return {"programmations": [{"programmation_id": "mock-prog-1", "name": "Mock Stream"}]}


@mcp.tool()
async def get_playlist(programmation_id: str, status_filter: str | None = None) -> dict:
    """Get playlist entries for a programmation."""
    return {"entries": [], "total": 0}


@mcp.tool()
async def get_next_video() -> dict:
    """Get the next video to play."""
    return {"entry_id": "mock-entry-001", "title": "Mock Video", "status": "pending"}


@mcp.tool()
async def mark_video_played(entry_id: str) -> dict:
    """Mark a playlist entry as played."""
    return {"success": True, "entry_id": entry_id}


# ---------------------------------------------------------------------------
# Tools — Video Generation
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_video(title: str, story_template_id: str = "cest_pas_sorcier") -> dict:
    """Submit a video generation job."""
    return {"job_id": "mock-job-001", "title": title, "status": "queued"}


@mcp.tool()
async def create_video_from_broadcast(title: str, user: str = "mcp_agent") -> dict:
    """Generate a video using the active broadcast's story template."""
    return {"job_id": "mock-job-002", "title": title, "status": "queued"}


@mcp.tool()
async def get_job_status(job_id: str) -> dict:
    """Check the status of a video generation job."""
    return {"job_id": job_id, "status": "processing", "progress": 0.5}


@mcp.tool()
async def list_jobs(limit: int = 20) -> dict:
    """List recent video generation jobs."""
    return {"jobs": [], "total": 0}


@mcp.tool()
async def submit_feedback(entry_id: str, user: str, feedback: str) -> dict:
    """Store user feedback for a played video."""
    return {"success": True, "entry_id": entry_id}


@mcp.tool()
async def greet_viewer(user_name: str, character_id: str = "jesus_short") -> dict:
    """Generate a greeting video for a viewer."""
    return {"job_id": "mock-greet-001", "user_name": user_name, "status": "queued"}


@mcp.tool()
async def answer_viewer_question(
    question: str, user_name: str, character_id: str = "jesus_short"
) -> dict:
    """Generate a video answering a viewer's question."""
    return {"job_id": "mock-answer-001", "question": question, "status": "queued"}


# ---------------------------------------------------------------------------
# Tools — Content / Knowledge
# ---------------------------------------------------------------------------


@mcp.tool()
async def fetch_news() -> list[dict]:
    """Fetch latest news articles."""
    return [
        {
            "title": "L'IA révolutionne tout",
            "summary": "Les modèles de langage continuent de progresser.",
            "source": "lemonde.fr",
            "link": "",
        },
        {
            "title": "Le réchauffement climatique s'accélère",
            "summary": "Nouvelles données alarmantes.",
            "source": "liberation.fr",
            "link": "",
        },
    ]


@mcp.tool()
async def list_story_templates(limit: int = 100) -> dict:
    """List available story templates."""
    return {
        "templates": [{"id": "cest_pas_sorcier", "name": "C'est Pas Sorcier"}],
        "total": 1,
    }


@mcp.tool()
async def get_story_template(template_id: str) -> dict:
    """Get details of a specific story template."""
    return {"id": template_id, "name": "C'est Pas Sorcier", "character_ids": ["jesus"]}


@mcp.tool()
async def list_characters() -> dict:
    """List available characters."""
    return {"characters": [{"id": "jesus", "name": "Jesus Christ"}]}


# ---------------------------------------------------------------------------
# Tools — System
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_system_status() -> dict:
    """Get system health and workload information."""
    return {
        "workload": "LOW",
        "active_jobs": 0,
        "queue_pending": _get_queue().get("pending_count", 0),
    }


@mcp.tool()
async def get_stream_config() -> dict:
    """Get the stream configuration."""
    return {"stream_id": "mock-stream", "name": "Mock Stream"}


@mcp.tool()
async def health_check() -> dict:
    """Global system health check."""
    return {"status": "ok"}


@mcp.tool()
async def check_content_safety(text: str) -> dict:
    """Check if content is safe (always returns NORMAL in mock)."""
    return {"classification": "NORMAL", "justification": "Mock: always safe"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
