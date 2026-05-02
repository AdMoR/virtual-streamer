"""
MCP Server for Virtual Streamer.

Exposes the Virtual Streamer capabilities (Twitch chat, video generation,
stream management, content tools) through the Model Context Protocol.

Usage:
    python -m virtual_streamer.mcp_server.server

Environment variables:
    See config.py for the full list.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

from virtual_streamer.mcp_server.config import MCPConfig
from virtual_streamer.mcp_server.twitch_bridge import TwitchBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state shared between lifespan and tools
_config: Optional[MCPConfig] = None
_bridge: Optional[TwitchBridge] = None
_http_client = None  # lazy httpx.AsyncClient


def _get_config() -> MCPConfig:
    if _config is None:
        raise RuntimeError("MCP server not initialized")
    return _config


def _get_bridge() -> TwitchBridge:
    if _bridge is None:
        raise RuntimeError("MCP server not initialized")
    return _bridge


async def _api_get(path: str, params: dict | None = None) -> dict:
    """Make a GET request to the REST API."""
    import httpx

    cfg = _get_config()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{cfg.api_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


async def _api_post(path: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    """Make a POST request to the REST API."""
    import httpx

    cfg = _get_config()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{cfg.api_url}{path}", json=body)
        resp.raise_for_status()
        return resp.json()


async def _api_post_multipart(
    path: str,
    fields: dict,
    file_paths: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> dict:
    """Make a multipart/form-data POST request, optionally attaching local files.

    Args:
        path: API path
        fields: Form fields (str values)
        file_paths: Mapping of form field name → local file path to attach
    """
    import httpx

    cfg = _get_config()
    data = {k: str(v) for k, v in fields.items()}
    files: list[tuple] = []
    opened: list = []
    try:
        for field_name, fpath in (file_paths or {}).items():
            fh = open(fpath, "rb")
            opened.append(fh)
            files.append((field_name, (os.path.basename(fpath), fh)))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{cfg.api_url}{path}",
                data=data,
                files=files if files else None,
            )
            resp.raise_for_status()
            return resp.json()
    finally:
        for fh in opened:
            fh.close()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Start/stop the Twitch bridge alongside the MCP server."""
    global _config, _bridge

    _config = MCPConfig()
    _bridge = TwitchBridge(_config)

    logger.info(
        f"MCP server starting: api_url={_config.api_url}, "
        f"stream_id={_config.stream_id}, "
        f"twitch={'enabled' if _config.has_twitch_credentials else 'disabled'}"
    )

    await _bridge.start()
    try:
        yield {}
    finally:
        await _bridge.stop()
        logger.info("MCP server stopped")


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="virtual-streamer",
    instructions=(
        "Virtual Streamer MCP server. Provides tools to host a Twitch stream: "
        "read/send chat messages, generate videos, manage playlists, "
        "monitor system status, and interact with viewers."
    ),
    lifespan=server_lifespan,
)


# ===================================================================
# TOOLS — Category A: Twitch Chat
# ===================================================================

@mcp.tool()
async def get_chat_messages(limit: int = 50, mentions_only: bool = False) -> list[dict]:
    """Get recent Twitch chat messages.

    Args:
        limit: Maximum number of messages to return (default 50)
        mentions_only: If true, only return messages that mention the bot
    """
    return await _get_bridge().get_messages(limit=limit, mentions_only=mentions_only)


@mcp.tool()
async def send_twitch_message(message: str) -> dict:
    """Send a message to the Twitch chat channel.

    The message is automatically truncated to 500 characters.

    Args:
        message: The message to send
    """
    bridge = _get_bridge()
    truncated = message[:500]
    try:
        await bridge.send_message(truncated)
        return {"success": True, "message": truncated}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}


# ===================================================================
# TOOLS — Category B: Video Generation
# ===================================================================

@mcp.tool()
async def create_video(title: str, story_template_id: str) -> dict:
    """Submit a video generation job with a title and story template.

    Args:
        title: The video title/topic
        story_template_id: Story template to use (default: cest_pas_sorcier)
    """
    cfg = _get_config()
    return await _api_post(
        "/api/v1/video-generation/submit",
        {
            "title": title,
            "stream_id": cfg.stream_id,
            "story_template_id": story_template_id,
            "metadata": {"source": "mcp_server"},
        },
    )


@mcp.tool()
async def create_video_from_broadcast(title: str, user: str = "mcp_agent") -> dict:
    """Generate a video using the active broadcast's story template.

    Resolves the active programmation, enforces queue limits (max 5 pending),
    and auto-adds the video to the playlist.

    Args:
        title: The video title/topic
        user: Username to attribute the generation to
    """
    cfg = _get_config()
    return await _api_post(
        "/api/v1/video-generation/generate-from-broadcast",
        {"stream_id": cfg.stream_id, "title": title, "user": user},
    )


@mcp.tool()
async def create_video_ltx(
    story_template_id: str,
    title: str | None = None,
    story_text: str | None = None,
    ltx_server_url: str = "http://gx10-cbc5:8082",
    video_width: int = 1280,
    video_height: int = 720,
    video_duration_seconds: float = 5.0,
) -> dict:
    """Generate a video from a story template using LTX-2 text-to-video.

    Uses WanGP (Gradio API) + LTX-2 to generate video with synchronized audio.
    Returns a job_id that can be tracked with get_job_status.

    Args:
        story_template_id: Required. Story template defining characters and prompt.
        title: Video title/topic (used to generate the story). Mutually exclusive with story_text.
        story_text: Pre-written story text. Mutually exclusive with title.
        ltx_server_url: URL of the WanGP server running LTX-2.
        video_width: Output video width in pixels (default 1280).
        video_height: Output video height in pixels (default 720).
        video_duration_seconds: Duration per video segment in seconds (default 5.0).
    """
    body: dict = {
        "story_template_id": story_template_id,
        "ltx_server_url": ltx_server_url,
        "video_width": video_width,
        "video_height": video_height,
        "video_duration_seconds": video_duration_seconds,
    }
    if title is not None:
        body["title"] = title
    if story_text is not None:
        body["story_text"] = story_text
    return await _api_post("/api/v1/video-generation/generate-ltx", body, timeout=7200.0)


@mcp.tool()
async def create_video_ltx_i2v(
    prompt: str,
    image_path: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    ltx_server_url: str = "http://gx10-cbc5:8082",
    resolution: str = "1280x720",
    duration_seconds: float = 4.0,
    fps: int = 24,
    steps: int = 8,
    guidance_scale: float = 3.0,
    flow_shift: float = 3.0,
    seed: int = -1,
) -> dict:
    """Generate a video from a conditioning image and text prompt using LTX-2 image-to-video.

    The image acts as the first frame; the model animates it according to the prompt.
    Returns a job_id that can be tracked with get_job_status. The completed job result
    contains a base64-encoded MP4 (video_b64) plus resolution/duration metadata.

    Args:
        prompt: Text description of the desired motion and scene.
        image_path: Local path to the conditioning image (PNG/JPG).
        negative_prompt: What to avoid in the generated video.
        ltx_server_url: URL of the WanGP server running LTX-2.
        resolution: Output resolution as WxH (default 1280x720).
        duration_seconds: Clip duration in seconds (default 4.0).
        fps: Frames per second (default 24).
        steps: Denoising steps — lower is faster, higher is higher quality (default 8).
        guidance_scale: Classifier-free guidance scale (default 3.0).
        flow_shift: Flow shift parameter (default 3.0).
        seed: Random seed, -1 for random (default -1).
    """
    return await _api_post_multipart(
        "/api/v1/video-generation/single-clip",
        fields={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "wangp_url": ltx_server_url,
            "model_type": "ltx2_22B_distilled",
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
        },
        file_paths={"image": image_path},
    )


@mcp.tool()
async def create_video_ltx_audio_i2v(
    prompt: str,
    image_path: str,
    audio_path: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    ltx_server_url: str = "http://gx10-cbc5:8082",
    resolution: str = "1280x720",
    duration_seconds: float = 4.0,
    fps: int = 24,
    steps: int = 8,
    guidance_scale: float = 3.0,
    flow_shift: float = 3.0,
    seed: int = -1,
    audio_scale: float = 1.0,
    audio_guidance: float = 4.5,
) -> dict:
    """Generate a video from a conditioning image guided by an audio track using LTX-2.

    The image acts as the first frame; the audio drives the motion and expression.
    Returns a job_id trackable with get_job_status. The completed result contains
    video_b64 (base64-encoded MP4) plus resolution/duration metadata.

    Args:
        prompt: Text description of the desired scene and motion.
        image_path: Local path to the conditioning image (PNG/JPG).
        audio_path: Local path to the audio guide track (WAV).
        negative_prompt: What to avoid in the generated video.
        ltx_server_url: URL of the WanGP server running LTX-2.
        resolution: Output resolution as WxH (default 1280x720).
        duration_seconds: Clip duration in seconds (default 4.0).
        fps: Frames per second (default 24).
        steps: Denoising steps — lower is faster, higher is higher quality (default 8).
        guidance_scale: Classifier-free guidance scale (default 3.0).
        flow_shift: Flow shift parameter (default 3.0).
        seed: Random seed, -1 for random (default -1).
        audio_scale: How strongly the audio conditions the video (0.0–1.0, default 1.0).
        audio_guidance: Audio guidance strength (default 4.5).
    """
    return await _api_post_multipart(
        "/api/v1/video-generation/single-clip",
        fields={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "wangp_url": ltx_server_url,
            "model_type": "ltx2_22B_distilled",
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "audio_scale": audio_scale,
            "audio_guidance": audio_guidance,
        },
        file_paths={"image": image_path, "audio": audio_path},
    )


@mcp.tool()
async def get_job_status(job_id: str) -> dict:
    """Check the status of a video generation job.

    Args:
        job_id: The job ID returned by create_video or create_video_from_broadcast
    """
    return await _api_get(f"/api/v1/video-generation/jobs/{job_id}")


@mcp.tool()
async def list_jobs(limit: int = 20) -> dict:
    """List recent video generation jobs.

    Args:
        limit: Maximum number of jobs to return
    """
    return await _api_get("/api/v1/video-generation/jobs", params={"limit": limit})


@mcp.tool()
async def submit_feedback(entry_id: str, user: str, feedback: str) -> dict:
    """Store user feedback for a played video.

    Args:
        entry_id: The playlist entry ID of the video
        user: Username providing the feedback
        feedback: The feedback text (e.g. '+' or '-')
    """
    return await _api_post(
        "/api/v1/video-generation/feedback",
        {"entry_id": entry_id, "user": user, "feedback": feedback},
    )


@mcp.tool()
async def greet_viewer(user_name: str, character_id: str = "jesus_short") -> dict:
    """Generate a personalized greeting video for a viewer.
    Use this tool when a use enters the channel for the first time

    The agent generates a greeting analyzing the viewer's username,
    then creates a TTS + lip-sync video.

    Args:
        user_name: The Twitch username to greet
        character_id: Character to use for the greeting
    """
    cfg = _get_config()
    return await _api_post(
        "/api/v1/jesus-agents/greeting/submit",
        {
            "user_name": user_name,
            "character_id": character_id,
            "stream_id": cfg.stream_id,
        },
    )


@mcp.tool()
async def answer_viewer_question(
    question: str, user_name: str, character_id: str = "jesus_short"
) -> dict:
    """Generate a video answering a viewer's question.

    The agent generates a sarcastic answer, then creates a TTS + lip-sync video.

    Args:
        question: The viewer's question
        user_name: The Twitch username asking
        character_id: Character to use for the answer
    """
    cfg = _get_config()
    return await _api_post(
        "/api/v1/jesus-agents/answering/submit",
        {
            "question": question,
            "user_name": user_name,
            "character_id": character_id,
            "stream_id": cfg.stream_id,
        },
    )


# ===================================================================
# TOOLS — Category C: Stream & Playlist Management
# ===================================================================

@mcp.tool()
async def get_queue_status() -> dict:
    """Get current video queue status (pending count, played count, active jobs).

    Returns queue info for the configured programmation.
    """
    cfg = _get_config()
    prog_id = cfg.programmation_id

    if not prog_id:
        # Try to resolve from active programmation
        try:
            prog = await _api_get(
                f"/api/v1/streams/{cfg.stream_id}/programmations/active"
            )
            prog_id = prog.get("programmation_id")
        except Exception:
            return {"error": "No programmation_id configured and no active programmation found"}

    if not prog_id:
        return {"error": "No active programmation"}

    pending = await _api_get(
        f"/api/v1/programmations/{prog_id}/playlist",
        params={"status": "pending"},
    )
    played = await _api_get(
        f"/api/v1/programmations/{prog_id}/playlist",
        params={"status": "played"},
    )

    pending_list = pending if isinstance(pending, list) else pending.get("entries", [])
    played_list = played if isinstance(played, list) else played.get("entries", [])

    return {
        "programmation_id": prog_id,
        "pending_count": len(pending_list),
        "played_count": len(played_list),
        "is_replaying": len(pending_list) == 0 and len(played_list) > 0,
    }


@mcp.tool()
async def get_playlist(programmation_id: str, status_filter: str | None = None) -> dict:
    """Get all playlist entries for a programmation.

    Args:
        programmation_id: The programmation ID
        status_filter: Optional filter by status (pending, playing, played, skipped)
    """
    params = {}
    if status_filter:
        params["status"] = status_filter
    return await _api_get(
        f"/api/v1/programmations/{programmation_id}/playlist", params=params
    )


@mcp.tool()
async def get_next_video() -> dict:
    """Get the next video to play for the stream.

    Returns pending videos first, then random from played videos as fallback.
    """
    cfg = _get_config()
    return await _api_get(f"/api/v1/streams/{cfg.stream_id}/next-video")


@mcp.tool()
async def mark_video_played(entry_id: str) -> dict:
    """Mark a playlist entry as played.

    Args:
        entry_id: The playlist entry ID to mark as played
    """
    return await _api_post(f"/api/v1/playlist/{entry_id}/played")


@mcp.tool()
async def get_active_programmation() -> dict:
    """Get the currently active programmation for the stream.

    Returns the programmation matching the current time slot.
    """
    cfg = _get_config()
    return await _api_get(f"/api/v1/streams/{cfg.stream_id}/programmations/active")


@mcp.tool()
async def list_programmations() -> dict:
    """List all programmations for the stream."""
    cfg = _get_config()
    return await _api_get(f"/api/v1/streams/{cfg.stream_id}/programmations")


# ===================================================================
# TOOLS — Category D: Content & Knowledge
# ===================================================================

@mcp.tool()
async def list_story_templates(limit: int = 100) -> dict:
    """List available story templates.

    Story templates define characters, prompts, and video collections
    used for video generation.

    Args:
        limit: Maximum number of templates to return
    """
    return await _api_get("/api/v1/story-templates", params={"limit": limit})


@mcp.tool()
async def get_story_template(template_id: str) -> dict:
    """Get details of a specific story template.

    Args:
        template_id: The story template ID
    """
    return await _api_get(f"/api/v1/story-templates/{template_id}")


@mcp.tool()
async def create_story_template(story_concept: str) -> dict:
    """Generate and register a new story template from a creative concept.

    Runs an AI agent pipeline (guardrail → writer → formatter) to produce a
    fully-formed story template, then persists it in the database.

    **Fixed parameters** (not configurable via this tool):
    - `collection`: always `"random"` — videos are drawn from the generic random
      pool, not from a character- or show-specific collection.
    - `character`: always `"narrator"` — an off-screen voice with no on-screen
      character asset. The generated content is designed for narrator-only delivery.

    The returned template can immediately be used with `create_video` as the
    `story_template_id`. The `template_id` is derived from the generated name
    (lowercased, spaces replaced with underscores).

    ⚠ This call is slow (~30–60 s) because it runs a multi-step LLM pipeline.
    Do not call it during time-sensitive interactions.

    Args:
        story_concept: Creative description of the story idea — tone, scenario
            type, comedic style, and any thematic constraints. Example:
            "A parody documentary where an AI tries to understand why humans
            love eating cereal at midnight."
    """
    return await _api_post(
        "/api/v1/story-template-generation/generate",
        {"story_concept": story_concept},
        timeout=120.0,
    )


@mcp.tool()
async def list_locations(story_template_id: str) -> list[dict]:
    """List all locations registered for a story template.

    Locations define the scene backgrounds used for conditioned image-to-video
    generation. Each location has a diffusion-model description that is used to
    generate a conditioning image before the LTX video segment is produced.

    Args:
        story_template_id: The story template whose locations to list
    """
    return await _api_get("/api/v1/locations", params={"story_template_id": story_template_id})


@mcp.tool()
async def create_location(
    location_name: str,
    story_template_id: str,
    sd_server_url: Optional[str] = None,
) -> dict:
    """Generate and register a new location for a story template.

    Runs an AI agent pipeline (writer → formatter) to produce a detailed
    diffusion-model description for the location environment, then persists
    it in the database scoped to the given story template.

    The `location_id` is derived from the name (lowercased, spaces → hyphens).
    Example: "Medieval Castle" → location_id "medieval-castle".

    Locations must be created before running video generation with a story
    template — the pipeline validates that all referenced location IDs exist.

    ⚠ This call is slow (~20–40 s) because it runs an LLM pipeline.
    If sd_server_url is provided, image generation adds extra time (~30–60 s).
    Do not call it during time-sensitive interactions.

    Args:
        location_name: Human-readable name of the location (e.g. "Medieval Castle")
        story_template_id: The story template this location belongs to
        sd_server_url: Optional Stable Diffusion server URL to immediately generate
            an identity image for this location (e.g. "http://gx10-cbc5:1234")
    """
    payload: dict = {
        "location_name": location_name,
        "story_template_id": story_template_id,
    }
    if sd_server_url is not None:
        payload["sd_server_url"] = sd_server_url
    return await _api_post(
        "/api/v1/location-generation/generate",
        payload,
        timeout=180.0,
    )


@mcp.tool()
async def fetch_news() -> list[dict]:
    """Fetch latest news articles from French RSS sources.

    Returns recent articles that can be used as inspiration for
    video content generation.
    """
    from virtual_streamer.news.fetcher import RSSFetcher

    fetcher = RSSFetcher()
    articles = await fetcher.fetch_all()
    return [
        {
            "title": meta.title,
            "summary": content.summary if content else "",
            "source": meta.source,
            "link": content.link if content else "",
        }
        for meta, content in articles
    ]


@mcp.tool()
async def list_characters() -> dict:
    """List available characters with their voice samples and video clips."""
    return await _api_get("/api/v1/characters")


# ===================================================================
# TOOLS — Category E: System Monitoring
# ===================================================================

@mcp.tool()
async def get_system_status() -> dict:
    """Get system health and workload information.

    Returns active jobs, queue pending count, and overall workload level.
    """
    return await _api_get("/api/v1/video-generation/health")


@mcp.tool()
async def get_stream_config() -> dict:
    """Get the stream configuration details."""
    cfg = _get_config()
    return await _api_get(f"/api/v1/streams/{cfg.stream_id}")


@mcp.tool()
async def health_check() -> dict:
    """Global system health check."""
    return await _api_get("/health")


# ===================================================================
# TOOLS — Category F: Content Safety
# ===================================================================

@mcp.tool()
async def check_content_safety(text: str) -> dict:
    """Check if user-submitted content is safe.

    Returns NORMAL or MALICIOUS classification with justification.

    Args:
        text: The text to check for safety
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        from virtual_streamer.agents.guardrails_agent.agent import GuardrailAgent

        agent = GuardrailAgent(
            agent_name="ContentSafety",
            agent_context="Check if user-submitted content is safe for the stream.",
        )
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="mcp_safety", session_service=session_service)
        session = await session_service.create_session(app_name="mcp_safety", user_id="mcp")

        result_text = ""
        async for event in runner.run_async(
            session_id=session.id,
            user_id="mcp",
            new_message=types.Content(
                role="user", parts=[types.Part(text=text)]
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        result_text += part.text

        try:
            parsed = json.loads(result_text)
            return parsed
        except json.JSONDecodeError:
            return {"classification": "UNKNOWN", "raw_response": result_text}

    except ImportError:
        return {"error": "Guardrail agent dependencies not available"}
    except Exception as e:
        return {"error": f"Safety check failed: {e}"}


# ===================================================================
# RESOURCES
# ===================================================================

@mcp.resource("virtual-streamer://chat/recent")
async def chat_recent() -> str:
    """Last 50 Twitch chat messages."""
    messages = await _get_bridge().get_messages(limit=50)
    return json.dumps(messages, ensure_ascii=False, indent=2)


@mcp.resource("virtual-streamer://queue/status")
async def queue_status_resource() -> str:
    """Current video queue status."""
    try:
        status = await get_queue_status()
        return json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("virtual-streamer://system/status")
async def system_status_resource() -> str:
    """System health and workload status."""
    try:
        status = await _api_get("/api/v1/video-generation/health")
        return json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("virtual-streamer://stream/config")
async def stream_config_resource() -> str:
    """Current stream configuration and active programmation."""
    cfg = _get_config()
    try:
        stream = await _api_get(f"/api/v1/streams/{cfg.stream_id}")
        try:
            prog = await _api_get(
                f"/api/v1/streams/{cfg.stream_id}/programmations/active"
            )
        except Exception:
            prog = None
        return json.dumps(
            {"stream": stream, "active_programmation": prog},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("virtual-streamer://templates/list")
async def templates_list_resource() -> str:
    """Available story templates."""
    try:
        templates = await _api_get("/api/v1/story-templates")
        return json.dumps(templates, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
