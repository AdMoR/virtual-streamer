# MCP Server — Twitch Stream Hosting Agent

This document describes the MCP (Model Context Protocol) server that exposes Virtual Streamer capabilities to an external agent (e.g., Claude via Claude Desktop) for autonomous Twitch stream hosting.

## Overview

The MCP server replaces the tightly-coupled `VirtualStreamerRunner` agent loop with a protocol-based approach. Instead of an internal ADK agent making decisions in a fixed loop, an **external agent** reads chat, generates content, and manages the stream through MCP tools.

```
Claude Desktop / External Agent
        |
        | MCP protocol (stdio)
        v
+---------------------------+
| MCP Server (Python)       |
|  +- Tools (22)            |
|  +- Resources (5)         |
|  +- TwitchClient          |---- wss://irc-ws.chat.twitch.tv
|  +- ChatStore             |
+--------+------------------+
         | httpx
         v
   FastAPI REST API (:8000)
         |
    +----+----+
    v    v    v
  MySQL MinIO ML Services
```

**Key design decisions:**

1. **Hybrid approach** — REST API proxy for most tools (video generation, playlist, streams), direct Python calls for Twitch chat I/O, news fetching, and guardrail checks
2. **Embedded TwitchClient** — The MCP server owns the WebSocket connection and buffers messages in a `ChatStore`
3. **No authentication** — MCP runs locally via stdio; Twitch OAuth is handled internally by `TwitchClient`
4. **stdio transport** — Standard MCP transport for Claude Desktop integration

## MCP Tools

### Category A: Twitch Chat (Critical)

| Tool | Status | Description |
|------|--------|-------------|
| `get_chat_messages` | NEW | Read recent chat messages from the buffered ChatStore |
| `send_twitch_message` | EXISTING (wired) | Send a message to Twitch chat via the active WebSocket |

**`get_chat_messages(limit: int = 50, mentions_only: bool = false)`**
- Returns: `[{timestamp, username, message, is_mention}]`
- Source: `ChatStore` at `streaming/agent_loop/chat_store.py`

**`send_twitch_message(message: str)`**
- Sends via the active Twitch WebSocket (max 500 chars, auto-truncated)
- Existing pattern: `agents/virtual_streamer_agent/tools/send_message.py`

### Category B: Video Generation (High Priority)

| Tool | Status | REST Endpoint |
|------|--------|---------------|
| `create_video` | EXISTING | `POST /api/v1/video-generation/submit` |
| `create_video_from_broadcast` | EXISTING | `POST /api/v1/video-generation/generate-from-broadcast` |
| `get_job_status` | EXISTING | `GET /api/v1/video-generation/jobs/{job_id}` |
| `list_jobs` | EXISTING | `GET /api/v1/video-generation/jobs` |
| `submit_feedback` | EXISTING | `POST /api/v1/video-generation/feedback` |
| `greet_viewer` | EXISTING | `POST /api/v1/jesus-agents/greeting/submit` |
| `answer_viewer_question` | EXISTING | `POST /api/v1/jesus-agents/answering/submit` |

### Category C: Stream & Playlist Management (High Priority)

| Tool | Status | REST Endpoint |
|------|--------|---------------|
| `get_queue_status` | EXISTING | Via `QueueInfoProvider` (playlist API) |
| `get_playlist` | EXISTING | `GET /api/v1/programmations/{id}/playlist` |
| `get_next_video` | EXISTING | `GET /api/v1/streams/{stream_id}/next-video` |
| `mark_video_played` | EXISTING | `POST /api/v1/playlist/{entry_id}/played` |
| `get_active_programmation` | EXISTING | `GET /api/v1/streams/{stream_id}/programmations/active` |
| `list_programmations` | EXISTING | `GET /api/v1/streams/{stream_id}/programmations` |

### Category D: Content & Knowledge (Medium Priority)

| Tool | Status | Source |
|------|--------|--------|
| `list_story_templates` | EXISTING | `GET /api/v1/story-templates` |
| `get_story_template` | EXISTING | `GET /api/v1/story-templates/{id}` |
| `fetch_news` | EXISTING (no REST) | Direct call to `news/fetcher.py` `RSSFetcher` |
| `list_characters` | EXISTING | `GET /api/v1/characters` |

### Category E: System Monitoring (Medium Priority)

| Tool | Status | Source |
|------|--------|--------|
| `get_system_status` | EXISTING | `WorkloadProvider` + `/api/v1/video-generation/health` |
| `get_stream_config` | EXISTING | `GET /api/v1/streams/{stream_id}` |
| `health_check` | EXISTING | `GET /health` |

### Category F: Content Safety (Low Priority)

| Tool | Status | Source |
|------|--------|--------|
| `check_content_safety` | EXISTING (no REST) | Direct call to `GuardrailAgent` |

**Total: 22 tools — 20 wrapping existing functionality, 1 new (`get_chat_messages`), 1 existing needing wiring (`send_twitch_message`)**

## MCP Resources

Resources provide read-only context that the agent can reference without explicit tool calls.

| URI | Description | Source |
|-----|-------------|--------|
| `virtual-streamer://chat/recent` | Last 50 chat messages | `ChatStore` |
| `virtual-streamer://queue/status` | Pending/played video counts | `QueueInfoProvider` |
| `virtual-streamer://system/status` | Workload & health | `WorkloadProvider` |
| `virtual-streamer://stream/config` | Stream + active programmation | REST API |
| `virtual-streamer://templates/list` | Available story templates | REST API |

> **Note:** The original design used `virtual-streamer://stream/{id}/config` with a path parameter. The implementation uses `virtual-streamer://stream/config` instead, since `stream_id` is always available from server config — no per-request ID needed.

## Module Structure

```
virtual_streamer/
  mcp_server/                     # NEW module
    __init__.py
    server.py                     # Entry point (FastMCP), lifespan manages Twitch
    config.py                     # Env-based config (API_URL, STREAM_ID, Twitch creds)
    twitch_bridge.py              # TwitchClient lifecycle + ChatStore management
    tools/
      __init__.py
      chat.py                     # get_chat_messages, send_twitch_message
      video.py                    # create_video, create_video_from_broadcast, jobs, feedback
      stream.py                   # queue_status, playlist, programmations
      content.py                  # fetch_news, story_templates, characters
      agents.py                   # greet_viewer, answer_viewer_question, check_safety
      system.py                   # system_status, health, stream_config
    resources/
      __init__.py
      chat.py                     # chat/recent resource
      queue.py                    # queue/status resource
      system.py                   # system/status resource
```

## Data Flow

### Agent Hosting a Twitch Stream

```
1. Agent reads chat via get_chat_messages() or chat/recent resource
2. Agent decides how to respond:
   - Greeting: call greet_viewer() for new viewers
   - Q&A: call answer_viewer_question() for questions
   - Video request: call create_video_from_broadcast() for !generate commands
   - Chat response: call send_twitch_message() for casual interaction
3. Agent monitors queue via get_queue_status()
   - If queue is low, proactively calls create_video() with trending topics
   - If system is overloaded, waits before generating more
4. Generated videos automatically enter the playlist
5. OBS Video Server plays them via GET /streams/{id}/next-video
6. Agent can track job progress via get_job_status()
```

### Comparison with Previous Architecture

| Aspect | Old (VirtualStreamerRunner) | New (MCP Server) |
|--------|---------------------------|-------------------|
| Agent | Internal ADK agent in fixed loop | External agent (Claude) via MCP |
| Chat | ChatStore + ContextBuilder | MCP tools + resources |
| Tools | ADK tool registry | MCP tool protocol |
| Loop | Fixed interval (5s) | Agent-driven, on-demand |
| Flexibility | Hardcoded behavior | Agent decides strategy |
| Debugging | Logs only | Claude Desktop conversation |

## Configuration

### Environment Variables

```bash
# REST API connection
API_URL=http://localhost:8000
STREAM_ID=default
PROGRAMMATION_ID=my-prog

# Twitch credentials
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
TWITCH_REFRESH_TOKEN=your_refresh_token
TWITCH_CHANNEL=allojesuschrist
```

### Claude Desktop Config

```json
{
  "mcpServers": {
    "virtual-streamer": {
      "command": "python",
      "args": ["-m", "virtual_streamer.mcp_server.server"],
      "env": {
        "API_URL": "http://localhost:8000",
        "STREAM_ID": "default",
        "TWITCH_CLIENT_ID": "...",
        "TWITCH_CLIENT_SECRET": "...",
        "TWITCH_REFRESH_TOKEN": "...",
        "TWITCH_CHANNEL": "allojesuschrist"
      }
    }
  }
}
```

## Required Changes to Existing Code

### 1. TwitchClient: Add `send_chat_message()` method

**File:** `streaming/twitch/chat_reader.py`

The `TwitchClient` currently sends messages within handler methods using a local `websocket` parameter. For the MCP server, the active websocket must be stored as an instance attribute and exposed as a public async method:

```python
class TwitchClient:
    def __init__(self, ...):
        ...
        self._active_websocket = None  # NEW: store active connection

    async def send_chat_message(self, message: str) -> None:
        """Send a message to the connected Twitch channel."""
        if self._active_websocket is None:
            raise RuntimeError("Not connected to Twitch")
        await self._active_websocket.send(
            f"PRIVMSG #{self.channel_name} :{message}"
        )
```

### 2. Add `mcp[cli]` dependency

**File:** `pyproject.toml`

```toml
dependencies = [
    ...
    "mcp[cli]>=1.0.0",
    ...
]
```

## Key Files Referenced

| File | Role |
|------|------|
| `streaming/twitch/chat_reader.py` | Twitch WebSocket connection (to be refactored) |
| `streaming/agent_loop/chat_store.py` | Thread-safe chat message buffer (reused) |
| `streaming/agent_loop/runner.py` | Architecture pattern reference |
| `agents/virtual_streamer_agent/tools/send_message.py` | Message sending pattern reference |
| `agents/virtual_streamer_agent/context/providers.py` | `QueueInfoProvider`, `WorkloadProvider` (reused) |
| `news/fetcher.py` | `RSSFetcher` for news tool (reused) |
| `agents/guardrails_agent/agent.py` | `GuardrailAgent` for safety tool (reused) |
| `api/high_level/video_generation.py` | Video generation REST endpoints (proxied) |
| `api/high_level/jesus_agents.py` | Jesus agent REST endpoints (proxied) |
