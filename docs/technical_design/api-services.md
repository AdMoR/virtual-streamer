# API Services

All services are exposed through a unified HTTP API at `virtual_streamer.api.main`. This document defines the service contracts.

## Architecture

The API follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED API SERVER                            │
│                  (virtual_streamer.api.main)                     │
│                      Port: 8000                                  │
├─────────────────────────────────────────────────────────────────┤
│  HIGH-LEVEL       │  MEDIUM-LEVEL     │  LOW-LEVEL             │
│  /api/v1/         │  /api/v1/         │  /api/v1/              │
│  video-generation │  tts              │  characters            │
│  jesus-agents     │  stt              │  clips                 │
│                   │  wav2lip          │  streams               │
│                   │                   │  programmations        │
│                   │                   │  playlist              │
│                   │                   │  story-templates       │
│                   │                   │  articles              │
├─────────────────────────────────────────────────────────────────┤
│  ADK AGENTS (mounted at /adk)                                    │
│  story_generator, video_matcher, orchestrator                    │
└─────────────────────────────────────────────────────────────────┘
```

## Service Overview

| Layer | Endpoint Prefix | Purpose |
|-------|-----------------|---------|
| Low-level | `/api/v1/characters` | Character CRUD |
| Low-level | `/api/v1/clips` | Video clip management |
| Low-level | `/api/v1/streams` | Stream configuration |
| Low-level | `/api/v1/programmations` | Scheduling |
| Low-level | `/api/v1/playlist` | Playlist management |
| Low-level | `/api/v1/story-templates` | Story templates |
| Low-level | `/api/v1/articles` | Article management |
| Medium-level | `/api/v1/tts` | Text-to-speech |
| Medium-level | `/api/v1/stt` | Speech-to-text |
| Medium-level | `/api/v1/wav2lip` | Lip synchronization |
| High-level | `/api/v1/video-generation` | Full video pipeline |
| High-level | `/api/v1/jesus-agents` | AI Jesus video responses |
| Legacy | `/process` | Q&A video generation (deprecated) |

---

## Low-Level APIs (Entity Management)

### Characters API

```
GET    /api/v1/characters              List all characters
POST   /api/v1/characters              Create character (multipart)
GET    /api/v1/characters/{id}         Get character by ID
PUT    /api/v1/characters/{id}         Update character
DELETE /api/v1/characters/{id}         Delete character
```

**Create Character (Multipart Form):**
```bash
curl -X POST http://localhost:8000/api/v1/characters \
  -F "name=fred" \
  -F "description=Host of C'est pas Sorcier" \
  -F "voice_files=@fred_sample1.wav" \
  -F "voice_files=@fred_sample2.wav" \
  -F "transcripts=Eh bien dis donc Jamy" \
  -F "transcripts=C'est pas sorcier" \
  -F "video_file=@fred_talking.mp4" \
  -F "video_search_tag=person:fred"
```

**Response:**
```json
{
  "character_id": "fred",
  "name": "fred",
  "description": "Host of C'est pas Sorcier",
  "video_clip_path": "clips/fred.mp4",
  "voice_samples": [
    {
      "sample_storage_path": "audio/fred_sample1.wav",
      "transcript": "Eh bien dis donc Jamy"
    }
  ],
  "video_search_tag": "person:fred",
  "identity_images": []
}
```

### Streams API

```
GET    /api/v1/streams                 List all streams
POST   /api/v1/streams                 Create stream
GET    /api/v1/streams/{id}            Get stream by ID
PUT    /api/v1/streams/{id}            Update stream
DELETE /api/v1/streams/{id}            Delete stream
GET    /api/v1/streams/{id}/next-video Get next video to play
```

### Programmations API

```
GET    /api/v1/streams/{id}/programmations       List programmations
POST   /api/v1/streams/{id}/programmations       Create programmation
GET    /api/v1/streams/{id}/programmations/active Get active programmation
GET    /api/v1/programmations/{id}               Get programmation
PUT    /api/v1/programmations/{id}               Update programmation
DELETE /api/v1/programmations/{id}               Delete programmation
```

### Playlist API

```
GET    /api/v1/programmations/{id}/playlist      List playlist entries
POST   /api/v1/programmations/{id}/playlist      Add video to playlist
POST   /api/v1/playlist/{entry_id}/played        Mark video as played
```

---

## Medium-Level APIs (ML Services)

### TTS Service

**Location:** `virtual_streamer/api/medium_level/tts.py`

#### Endpoint: `POST /api/v1/tts/generate`

Generate speech audio from text using a character's voice.

**Request:**
```python
class TTSRequest(BaseModel):
    text: str                    # Text to synthesize
    character_id: str            # Character whose voice to use
    output_format: str = "wav"   # Output format
```

**Response:**
Returns audio file as binary stream with:
- `Content-Type: audio/wav`
- `X-Duration` header with audio duration

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour mes petits pécheurs", "character_id": "jesus"}' \
  --output response.wav
```

### STT Service

**Location:** `virtual_streamer/api/medium_level/stt.py`

#### Endpoint: `POST /api/v1/stt/transcribe`

Transcribe audio to text with timestamps.

**Request:**
```python
class STTRequest(BaseModel):
    audio_path: str              # Path to audio file
    language: str = "fr"         # Language code
```

**Response:**
```python
class TranscriptionSegment(BaseModel):
    start: float                 # Start time in seconds
    end: float                   # End time in seconds
    text: str                    # Transcribed text

class STTResponse(BaseModel):
    text: str                    # Full transcription
    segments: List[TranscriptionSegment]
    language: str
    duration_seconds: float
```

### Wav2Lip Service

**Location:** `virtual_streamer/api/medium_level/wav2lip.py`

#### Endpoint: `POST /api/v1/wav2lip/generate`

Generate lip-synced video from source video and audio.

**Request:**
```python
class Wav2LipRequest(BaseModel):
    video_path: str        # Path to source video
    audio_path: str        # Path to audio file
    fps: int = 24
    face_padding: int = 10
```

**Response:**
Returns the generated video file directly as `FileResponse` with:
- `Content-Type: video/mp4`
- `X-Processing-Time` header with processing duration

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/wav2lip/generate \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/video/jesus_ref.mp4",
    "audio_path": "/data/audio/response.wav"
  }' --output output.mp4
```

---

## High-Level APIs (Application Workflows)

### Jesus Agents API

**Location:** `virtual_streamer/api/high_level/jesus_agents.py`

High-level API endpoints that wrap ADK agents with video generation pipeline.

#### Agent Factory

Agents are loaded via a factory registry pattern:

```python
from virtual_streamer.agents.factory import get_agent, list_agents

# Get available agents
available = list_agents()  # ["greeting_jesus_agent", "answering_jesus_agent"]

# Load an agent by name
agent = get_agent("greeting_jesus_agent")
```

#### Endpoint: `POST /api/v1/jesus-agents/greeting/submit`

Generate a greeting video for a user.

**Request:**
```python
class GreetingJesusRequest(BaseModel):
    user_name: str                              # Username to greet
    character_id: str = "jesus"                 # Character for TTS/Wav2Lip
    agent_name: str = "greeting_jesus_agent"    # Agent to generate greeting
```

**Response:**
```python
class JesusAgentResponse(BaseModel):
    job_id: str      # UUID for tracking
    status: str      # "pending"
    message: str     # Confirmation message
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/jesus-agents/greeting/submit" \
  -H "Content-Type: application/json" \
  -d '{"user_name": "DrPetitesFesses"}'
```

#### Endpoint: `POST /api/v1/jesus-agents/answering/submit`

Generate a Q&A video response.

**Request:**
```python
class AnsweringJesusRequest(BaseModel):
    question: str                                 # Question to answer
    user_name: str                                # User asking the question
    character_id: str = "jesus"                   # Character for TTS/Wav2Lip
    agent_name: str = "answering_jesus_agent"     # Agent to generate answer
```

**Response:** Same as greeting endpoint.

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/jesus-agents/answering/submit" \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment fait-on les hosties?", "user_name": "Dany"}'
```

### Video Generation API

**Location:** `virtual_streamer/api/high_level/video_generation.py`

#### Endpoint: `GET /api/v1/video-generation/jobs/{job_id}`

Check job status.

**Response:**
```json
{
  "job_id": "abc-123",
  "status": "completed",
  "result": {
    "video_url": "https://minio.../generated_videos/greeting_jesus_agent/abc-123.mp4",
    "minio_video_key": "generated_videos/greeting_jesus_agent/abc-123.mp4",
    "agent_response": "Salut DrPetitesFesses, que la paix soit avec tes...",
    "user_name": "DrPetitesFesses",
    "timestamp": "2026-01-30T10:30:00.000Z"
  }
}
```

### Pipeline Flow

Both Jesus Agents endpoints follow the same pipeline:

```
Agent → TTS → Wav2Lip → Combine → STT → Subtitles → MinIO Upload
```

1. **Agent**: Generate text response using ADK agent
2. **TTS**: Convert text to audio using character's voice
3. **Wav2Lip**: Generate lip-synced video from character's reference video
4. **Combine**: Merge video and audio tracks
5. **STT**: Transcribe audio to SRT subtitles
6. **Subtitles**: Burn subtitles into video
7. **Upload**: Store final video in MinIO

---

## ADK Agents API

ADK agents are mounted at `/adk` and provide access to Google ADK agent functionality.

**Available agents:**
- `story_generator` - Generate stories from titles
- `video_matcher` - Match videos to dialogue
- `orchestrator` - Full video generation pipeline

See the ADK documentation for endpoint details.

---

## Shared Utilities

The implementation uses centralized utilities:

```python
# Character loading (used by TTS, Wav2Lip, legacy_qa, jesus_agents)
from virtual_streamer.utils.character_loader import load_character
character = await load_character("jesus")

# Agent factory (used by jesus_agents)
from virtual_streamer.agents.factory import get_agent
agent = get_agent("greeting_jesus_agent")

# Storage client (used by all layers)
from virtual_streamer.utils.minio_client import get_storage_client
storage = get_storage_client()
```

---

## Health Check

### Endpoint: `GET /health`

```python
class HealthResponse(BaseModel):
    status: str          # "healthy" or "unhealthy"
    service: str         # "virtual-streamer-api"
    device: str          # "cuda" or "cpu"
    data_dir: str        # Data directory path
    temp_dir: str        # Temp directory path
```

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "virtual-streamer-api",
  "device": "cuda",
  "data_dir": "/data",
  "temp_dir": "./temp"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `DATA_DIR` | `/data` | Data directory |
| `TEMP_DIR` | `./temp` | Temporary files directory |
| `FISH_TTS_HOST` | `localhost` | Fish TTS service host |
| `FISH_TTS_PORT` | `8003` | Fish TTS service port |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `virtual-streamer` | MinIO bucket name |
| `VIDEO_SEARCH_SERVER_URL` | `http://localhost:8003` | Video search service URL |

---

## Docker Compose

```yaml
# compose.yaml
services:
  virtual_streamer_api:
    build:
      context: .
      dockerfile: docker/docker_unified_api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATA_DIR=/data
      - TEMP_DIR=/temp
      - FISH_TTS_HOST=fish_tts
      - FISH_TTS_PORT=8003
      - MINIO_ENDPOINT=minio:9000
    volumes:
      - data:/data
      - temp:/temp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  fish_tts:
    image: fishaudio/fish-speech:latest
    ports:
      - "8003:8003"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data

volumes:
  data:
  temp:
  minio-data:
```

---

## Running the API

### Development

```bash
# Start the API server in development mode
uvicorn virtual_streamer.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production

```bash
# Using Docker Compose
docker compose up -d

# Check health
curl http://localhost:8000/health
```

### API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
