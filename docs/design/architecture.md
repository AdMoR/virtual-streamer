# Architecture Overview

## Core Principles

### 1. All ML Processing via HTTP APIs

Every ML model (Wav2Lip, Face Detection, TTS, STT) is accessed through HTTP APIs:

```
Agent → API Client → HTTP → ML Service (Docker)
```

**Benefits:**
- Portability: Switch ML backends without changing agent code
- Scalability: Scale services independently
- Testability: Mock API clients for unit tests
- Language agnostic: Services could be in any language

### 2. Agents are Shared, Not Per-App

Instead of each application having its own agents, agents are **shared resources** that multiple applications can use:

```
❌ Wrong: ai_jesus/agents/qa_agent.py
❌ Wrong: fred_et_jamy/agents/story_agent.py

✅ Correct: agents/qa_responder/agent.py  (used by AI Jesus config)
✅ Correct: agents/story_generator/agent.py  (used by Fred & Jamy config)
```

**Why?**
- Complex agents reuse sub-agents
- Reduces code duplication
- Easier maintenance
- Applications become configurations, not code

### 3. Clean Data Models

One model per entity, no redundancy:

```python
# ❌ Wrong: Redundant models
class CharacterReference:
    face_embeddings: List
    
class Character:
    face_embeddings: List  # Duplicated!

# ✅ Correct: Single unified model
class Character:
    # Contains everything needed across all modalities
    voice_samples: List[VoiceSample]
    representative_video_path: str
    face_reference_paths: List[str]
```

### 4. ADK Agent Structure

All agents follow Google ADK conventions:

```
agents/
├── agent_name/
│   ├── agent.py      # LlmAgent definition
│   ├── prompt.py     # Prompt templates
│   └── callback.py   # Event callbacks
```

### 5. Docker for Services Only

| Component | Docker? | Runtime |
|-----------|---------|---------|
| Wav2Lip Service | ✅ Yes | Container with GPU |
| Face Detection Service | ✅ Yes | Container with GPU |
| Fish TTS | ✅ Yes | Container with GPU |
| Entity API | ✅ Yes | Container |
| ADK Agents | ❌ No | ADK Server |
| Video Indexer | ❌ No | CLI tool |

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
│ LAYER 4: APPLICATION CONFIGURATIONS                                 │
│                                                                      │
│  apps/                                                               │
│  ├── ai_jesus/config.yaml      → Uses: qa_responder, video_creator  │
│  ├── fred_et_jamy/config.yaml  → Uses: story_generator, video_creator│
│  └── streaming/                → compose_streaming.yml (OBS infra)  │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ configure
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 3: ADK AGENTS (Run via ADK Server)                            │
│                                                                      │
│  agents/                                                             │
│  ├── sub_agents/           ← Reusable building blocks               │
│  │   ├── tts_agent/                                                 │
│  │   ├── lip_sync_agent/                                            │
│  │   └── video_composer_agent/                                      │
│  │                                                                   │
│  ├── qa_responder/         ← Answers questions (uses sub_agents)    │
│  ├── story_generator/      ← Generates stories                      │
│  └── video_creator/        ← Orchestrates full video creation       │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ uses
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2: API CLIENTS (Python library)                               │
│                                                                      │
│  packages/vs-core/src/vs_core/                                       │
│  ├── api_clients/                                                    │
│  │   ├── tts.py            → FishTTSClient, TTSClient               │
│  │   ├── stt.py            → WhisperSTTClient, STTClient            │
│  │   ├── wav2lip.py        → Wav2LipClient                          │
│  │   └── face_detection.py → FaceDetectionClient                    │
│  │                                                                   │
│  ├── api_models/           ← Request/Response Pydantic models       │
│  └── models/               ← Data entities (Character, VideoClip)   │
│                                                                      │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ HTTP calls
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1: ML SERVICES (Docker Containers)                            │
│                                                                      │
│  services/                                                           │
│  ├── wav2lip_service/      → POST /generate, GET /health            │
│  ├── face_detection_service/ → POST /detect, POST /preprocess       │
│  └── entity_service/       → CRUD for Character, VideoClip          │
│                                                                      │
│  External:                                                           │
│  ├── Fish TTS (fishaudio/fish-speech)                               │
│  └── Whisper STT                                                     │
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

3. video_creator agent (for each dialogue line):
   ├── tts_agent → Generate audio for line
   ├── lip_sync_agent → Generate lip-synced clip
   └── video_composer_agent → Stitch clips together

4. Scorer evaluates parody quality (optional)

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

All services communicate via REST APIs over a shared Docker network:

```yaml
# infra/compose.yml
services:
  wav2lip:
    ports: ["8001:8001"]
    
  face_detection:
    ports: ["8005:8005"]
    
  entity_api:
    ports: ["8002:8002"]
    
  fish_tts:
    ports: ["8003:8003"]

# All share:
volumes:
  - data:/data  # Shared file storage
```

### Shared Volume Strategy

Since services need to exchange files (audio, video), they share a volume:

```
/data/
├── audio/          # TTS outputs
├── video/          # Video files
├── cache/          # Face detection cache
└── output/         # Final generated videos
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

## Scalability Considerations

### Horizontal Scaling

Each service can be scaled independently:

```yaml
services:
  wav2lip:
    deploy:
      replicas: 3  # 3 GPU workers
```

### GPU Allocation

Different services may need different GPU resources:

| Service | GPU Memory | Can Share GPU? |
|---------|------------|----------------|
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

