# Architecture Overview

## Core Principles

### 1. All ML Processing via HTTP APIs

Every ML model (Wav2Lip, Face Detection, TTS, STT) is accessed through HTTP APIs:

```
Agent → API Client → HTTP → ML Service (Docker/Internal)
```

**Benefits:**
- Portability: Switch ML backends without changing agent code
- Scalability: Scale services independently
- Testability: Mock API clients for unit tests
- Language agnostic: Services could be in any language

### 2. Unified API Server

All services are integrated into a single FastAPI application with layered architecture:

```
❌ Old: Multiple separate services
✅ Current: Single unified API at virtual_streamer.api.main
```

**Why?**
- Single GPU model instance (efficient resource usage)
- Simplified deployment
- Consistent API patterns
- Centralized health monitoring

### 3. Agents are Shared, Not Per-App

Agents are **shared resources** that multiple applications can use:

```
✅ Correct: virtual_streamer/agents/qa_responder/agent.py  (used by AI Jesus)
✅ Correct: virtual_streamer/agents/story_generator/agent.py  (used by Fred & Jamy)
```

**Why?**
- Complex agents reuse sub-agents
- Reduces code duplication
- Easier maintenance
- Applications become configurations, not code

### 4. Clean Data Models

One model per entity, no redundancy:

```python
# ✅ Correct: Single unified model in virtual_streamer/video_server/models.py
class Character:
    character_id: str
    name: str
    video_clip_path: str           # For Wav2Lip
    voice_samples: List[VoiceSample]  # For TTS
    video_search_tag: Optional[str]   # For video search filtering
    identity_images: List[str]        # For face detection
```

### 5. ADK Agent Structure

All agents follow Google ADK conventions:

```
agents/
├── agent_name/
│   ├── __init__.py
│   ├── agent.py      # Agent definition (REQUIRED)
│   ├── prompt.py     # Prompt templates
│   ├── schema.py     # Pydantic models (optional)
│   └── callback.py   # Event callbacks (optional)
```

### 6. Shared Utilities

Common functionality is centralized to avoid duplication:

```python
# Character loading (used by TTS, Wav2Lip, jesus_agents, legacy_qa)
from virtual_streamer.utils.character_loader import load_character
character = await load_character("jesus")

# Agent factory (loads agents by name)
from virtual_streamer.agents.factory import get_agent
agent = get_agent("greeting_jesus_agent")

# Storage operations
from virtual_streamer.utils.minio_client import get_storage_client
storage = get_storage_client()
await storage.upload_file(local_path, minio_key)
```

---

## Layer Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 4: APPLICATION INTERFACES                                      │
│                                                                      │
│  apps/                                                               │
│  ├── agent_test_interface.py   → Test agents interactively          │
│  ├── video_generation_ui.py    → Gradio UI for video generation     │
│  └── creation_interface.py     → Content creation interface          │
│                                                                      │
│  streaming/                                                          │
│  └── compose_streaming.yml     → OBS streaming stack                │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ HTTP API calls
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 3: UNIFIED API (virtual_streamer.api.main)                     │
│                                                                      │
│  High-level (Applications):                                          │
│  ├── /api/v1/video-generation/*   → Full video pipeline             │
│  ├── /api/v1/jesus-agents/*       → AI Jesus video responses        │
│  └── /process                     → Legacy Q&A (deprecated)          │
│                                                                      │
│  Medium-level (Services):                                            │
│  ├── /api/v1/tts/*               → Text-to-speech                   │
│  ├── /api/v1/stt/*               → Speech-to-text                   │
│  └── /api/v1/wav2lip/*           → Lip synchronization              │
│                                                                      │
│  Low-level (Entities):                                               │
│  ├── /api/v1/characters/*        → Character CRUD                   │
│  ├── /api/v1/clips/*             → Video clip management            │
│  ├── /api/v1/streams/*           → Stream configuration             │
│  ├── /api/v1/programmations/*    → Scheduling                       │
│  └── /api/v1/playlist/*          → Playlist management              │
│                                                                      │
│  ADK Agents (mounted at /adk):                                       │
│  ├── story_generator             → Generate stories                 │
│  ├── video_matcher               → Match videos to dialogue         │
│  └── orchestrator                → Full pipeline orchestration      │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ uses
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2: ADK AGENTS (virtual_streamer.agents)                        │
│                                                                      │
│  Character Agents:                                                   │
│  ├── greeting_jesus_agent/       → Greet Twitch viewers             │
│  └── answering_jesus_agent/      → Answer viewer questions          │
│                                                                      │
│  Content Generation:                                                 │
│  ├── story_generator/            → Generate parody stories          │
│  ├── keyword_generator/          → Extract keywords                 │
│  └── rubric_builder_agent/       → Build video rubrics              │
│                                                                      │
│  Video Processing:                                                   │
│  ├── video_matcher/              → Match videos to dialogue         │
│  ├── sentence_video_matcher/     → Per-sentence matching            │
│  └── rubric_builder_map_reduce/  → Parallel rubric building         │
│                                                                      │
│  Orchestration:                                                      │
│  ├── orchestrator/               → Pipeline coordination            │
│  └── virtual_streamer_agent/     → Main streaming agent             │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ uses
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1: CORE LIBRARY (virtual_streamer.lib)                         │
│                                                                      │
│  Agent Base Classes:                                                 │
│  ├── BaseLlmAgent               → Simple LLM agent                  │
│  ├── StatefulLlmAgent           → Agent with state management       │
│  ├── MapReduceAgent             → Parallel processing agent         │
│  ├── MapperAgent                → Split input into items            │
│  └── AggregatorAgent            → Combine results                   │
│                                                                      │
│  Callbacks:                                                          │
│  ├── BeforeModelCallback        → Pre-LLM processing                │
│  ├── AfterModelCallback         → Post-LLM processing               │
│  ├── StateInputCallback         → Inject state before agent         │
│  └── StateOutputCallback        → Save state after agent            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Streaming Stack (compose_streaming.yml)

```
┌──────────────────────────────────────────────────────────────────────┐
│ STREAMING INFRASTRUCTURE                                             │
│                                                                      │
│  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────────┐ │
│  │Twitch Reader│───▶│   Main API          │───▶│  MySQL (Playlists│ │
│  │(Chat Input) │    │  /api/v1/streams/*  │    │  StreamConfig,   │ │
│  └─────────────┘    └──────────┬──────────┘    │  Programmations) │ │
│                                │               └────────┬─────────┘ │
│                                ▼                        │           │
│                    ┌─────────────────────┐              │           │
│                    │   MinIO (Videos)    │◀─────────────┤           │
│                    └──────────┬──────────┘              │           │
│                               │                         │           │
│                               ▼                         ▼           │
│                    ┌─────────────────────┐    ┌──────────────────┐ │
│                    │   Video Server      │◀───│ Playlist API     │ │
│                    │   (Proxy :5000)     │    │ (next-video)     │ │
│                    └──────────┬──────────┘    └──────────────────┘ │
│                               │                                     │
│                               ▼                                     │
│                    ┌─────────────────────┐                         │
│                    │   OBS Container     │────▶ Twitch/RTMP       │
│                    │   (Browser Source)  │                         │
│                    └─────────────────────┘                         │
│                                                                      │
│  Ports:                                                              │
│  - 5000: Video Server (HTML player)                                 │
│  - 5901: OBS VNC                                                    │
│  - 6901: OBS noVNC (web)                                            │
│  - 4455: OBS WebSocket                                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Examples

### Example 1: AI Jesus Answering a Question (via Jesus Agents API)

```
1. Client → POST /api/v1/jesus-agents/answering/submit
   {"question": "What is the meaning of life?", "user_name": "Dany"}

2. Background job starts:
   ├── Agent Factory loads answering_jesus_agent
   └── Agent generates sarcastic response text

3. Video pipeline executes:
   ├── load_character("jesus") → Character with voice samples
   ├── TTS → Fish TTS API → audio.wav
   ├── Wav2Lip → Lip-synced video.mp4
   ├── STT → Subtitles (SRT)
   └── FFmpeg → final.mp4 with burned subtitles

4. Upload to MinIO:
   └── generated_videos/answering_jesus_agent/{job_id}.mp4

5. Client polls /video-generation/jobs/{job_id}
   └── Returns video_url when complete
```

### Example 2: Fred & Jamy Video Generation

```
1. User provides title: "Fred et Jamy découvrent TikTok"

2. ADK Server invokes story_generator agent
   ├── Uses LLM to generate parody dialogue
   └── Returns structured dialogue JSON

3. video_matcher agent (for each dialogue line):
   ├── Search video database for matching clips
   ├── Score and rank results
   └── Return best matching video per line

4. Video generation pipeline:
   ├── TTS → Generate audio for each line
   ├── Wav2Lip → Generate lip-synced clips
   └── FFmpeg → Concatenate clips

5. Output: Complete Fred & Jamy video
```

### Example 3: OBS Streaming with Playlist

```
1. Video Server requests next video
   GET /api/v1/streams/ai_jesus/next-video

2. API determines active programmation
   ├── Checks current time against MediaProgrammation schedule
   └── Returns programmation_id with highest priority

3. API queries playlist for next video
   ├── First: Get pending videos (ordered by play_order, created_at)
   ├── Fallback: Random selection from played videos
   └── Returns video_storage_key and entry_id

4. Video Server fetches video from MinIO
   GET /api/v1/files/stream?key={video_storage_key}

5. HTML Player plays video in browser source

6. On video end, mark as played
   POST /api/v1/playlist/{entry_id}/played

7. Loop back to step 1
```

---

## Service Communication

All services communicate via the unified API:

```yaml
# compose.yaml
services:
  virtual_streamer_api:
    build: ./docker/docker_unified_api
    ports: ["8000:8000"]
    environment:
      - FISH_TTS_HOST=fish_tts
      - FISH_TTS_PORT=8003
      - MINIO_ENDPOINT=minio:9000
    volumes:
      - data:/data

  fish_tts:
    image: fishaudio/fish-speech:latest
    ports: ["8003:8003"]

  minio:
    image: minio/minio
    ports: ["9000:9000"]

volumes:
  data:
```

### Shared Volume Strategy

Since services need to exchange files (audio, video), they share a volume:

```
/data/
├── audio/          # TTS outputs
├── video/          # Video files
├── cache/          # Face detection cache
├── output/         # Final generated videos
└── characters/     # Character JSON definitions
```

All API requests reference paths within `/data/`:
```json
{
  "video_path": "/data/video/jesus_reference.mp4",
  "audio_path": "/data/audio/response_123.wav",
  "output_path": "/data/output/result_123.mp4"
}
```

---

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
| Streaming Infrastructure | `virtual_streamer/streaming/` | Complete |
| Character Loader | `virtual_streamer/utils/character_loader.py` | Complete |
| MinIO Client | `virtual_streamer/utils/minio_client.py` | Complete |
| Agent Base Classes | `virtual_streamer/lib/agents/` | Complete |

### 🔄 In Progress

| Component | Notes |
|-----------|-------|
| Virtual Streamer Agent | Main streaming agent with tools |
| Orchestrator Agent | Pipeline coordination |

### 📋 Planned

| Component | Notes |
|-----------|-------|
| Sub-agents (TTS, Lip-sync, Video Composer) | Will be extracted from current pipeline |

---

## Scalability Considerations

### Horizontal Scaling

Each service can be scaled independently:

```yaml
services:
  virtual_streamer_api:
    deploy:
      replicas: 3  # Multiple API instances
```

### GPU Allocation

Different operations require different GPU resources:

| Operation | GPU Memory | Can Share GPU? |
|-----------|------------|----------------|
| Wav2Lip | ~4GB | Yes |
| Face Detection | ~2GB | Yes |
| Fish TTS | ~4GB | Yes |
| Whisper STT | ~2-8GB | Yes |

### Queue-Based Processing (Future)

For high load, add a message queue:

```
Agent → RabbitMQ → Worker Pool → Service
```

This is not implemented in v1 but the API-based architecture makes it easy to add.
