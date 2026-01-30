# Virtual Streamer - Design Documentation

This folder contains the design documentation for the Virtual Streamer system architecture.

## Design Documents

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | High-level system architecture and principles |
| [Package Structure](./package-structure.md) | Detailed folder and package organization |
| [Data Models](./data-models.md) | Core entity models: Character, VideoClip, Collection |
| [Remote Services](./remote-services.md) | External services: Video Search, MinIO, Qdrant |
| [API Services](./api-services.md) | Unified API with ML services: TTS, STT, Wav2Lip |
| [Agents](./agents.md) | Google ADK agent architecture and patterns |
| [Streaming](./streaming.md) | OBS streaming infrastructure and playlist management |
| [Migration Plan](./migration-plan.md) | Step-by-step migration roadmap |

## Key Principles

1. **Unified API Server** - All services integrated into single FastAPI application at `virtual_streamer.api.main`
2. **Shared agents, not per-app** - Agents are reusable across projects; apps are configurations
3. **Clean data models** - Single `Character` model with `video_search_tag` and `identity_images` for remote integration
4. **Remote-first storage** - All files stored in MinIO; video search via external Video Search Server
5. **ADK structure** - Agents follow Google ADK conventions: `agent.py`, `prompt.py`, `schema.py`
6. **Layered API** - Low-level (entities), Medium-level (ML services), High-level (applications)

## Quick Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED API SERVER                            │
│                  (virtual_streamer.api.main)                     │
│                      Port: 8000                                  │
├─────────────────────────────────────────────────────────────────┤
│  HIGH-LEVEL       │  MEDIUM-LEVEL     │  LOW-LEVEL             │
│  video-generation │  tts              │  characters            │
│  jesus-agents     │  stt              │  clips                 │
│                   │  wav2lip          │  streams               │
│                   │                   │  programmations        │
│                   │                   │  playlist              │
├─────────────────────────────────────────────────────────────────┤
│  ADK AGENTS (mounted at /adk)                                    │
│  story_generator, video_matcher, orchestrator                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ADK AGENTS                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              virtual_streamer_agent                      │    │
│  │                         │                                │    │
│  │    ┌────────────────────┼────────────────────┐          │    │
│  │    ▼                    ▼                    ▼          │    │
│  │ story_generator   video_matcher        orchestrator     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Fish TTS    │   │    MinIO      │   │ Video Search  │
│   :8003       │   │    :9000      │   │    Server     │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Streaming Stack

For OBS streaming, see [streaming.md](./streaming.md):

```
┌───────────────────────────────────────────────────────────────┐
│                    STREAMING STACK                             │
│                                                                │
│  Twitch Chat ──▶ Main API ──▶ MySQL ◀── Video Server ──▶ OBS  │
│                      │                      ▲                  │
│                      └───▶ MinIO ──────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

## Current Implementation Status

### ✅ Implemented

| Component | Location | Status |
|-----------|----------|--------|
| Unified API | `virtual_streamer/api/main.py` | Complete |
| Agent Factory | `virtual_streamer/agents/factory.py` | Complete |
| Jesus Agents | `greeting_jesus_agent/`, `answering_jesus_agent/` | Complete |
| Story Generator | `story_generator/` | Complete |
| Video Matcher | `video_matcher/`, `sentence_video_matcher/` | Complete |
| Rubric Builder | `rubric_builder_agent/`, `rubric_builder_map_reduce/` | Complete |
| Streaming Models | `virtual_streamer/streaming/models.py` | Complete |
| Video Server | `virtual_streamer/streaming/video_server/` | Complete |
| Twitch Integration | `virtual_streamer/streaming/twitch/` | Complete |
| Agent Base Classes | `virtual_streamer/lib/agents/` | Complete |

### 🔄 In Progress

| Component | Notes |
|-----------|-------|
| Virtual Streamer Agent | Main streaming agent with tools |
| End-to-end streaming | Final integration testing |

## Getting Started

1. Read [Architecture](./architecture.md) for the big picture
2. Review [Package Structure](./package-structure.md) for code organization
3. Check [API Services](./api-services.md) for service contracts
4. See [Agents](./agents.md) for agent development patterns
5. Review [Migration Plan](./migration-plan.md) for current status

## Running the System

### Development

```bash
# Install dependencies
uv sync

# Start the API server
uvicorn virtual_streamer.api.main:app --host 0.0.0.0 --port 8000 --reload

# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Production

```bash
# Start with Docker Compose
docker compose up -d

# Start streaming stack
docker compose -f compose_streaming.yml up -d
```

### Testing Agents

```bash
# List available agents
python -c "from virtual_streamer.agents.factory import list_agents; print(list_agents())"

# Test an agent
python -c "
from virtual_streamer.agents.factory import get_agent
agent = get_agent('greeting_jesus_agent')
result = agent.run({'user_name': 'TestUser'})
print(result)
"
```