# API Services

All ML processing is exposed through HTTP APIs. This document defines the service contracts.

## Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| Wav2Lip | 8001 | Lip-sync video generation |
| Entity API | 8002 | Character/Clip/Collection CRUD |
| Fish TTS | 8003 | Text-to-speech |
| Whisper STT | 8004 | Speech-to-text |
| Face Detection | 8005 | Face detection and preprocessing |

---

## Wav2Lip Service

### Endpoint: `POST /generate`

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

```python
# Implementation uses:
return FileResponse(
    path=output_path,
    media_type="video/mp4",
    filename="lipsync_output.mp4",
    headers={"X-Processing-Time": str(processing_time)}
)
```

**Example:**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/data/video/jesus_ref.mp4",
    "audio_path": "/data/audio/response.wav"
  }'
```

### Endpoint: `GET /health`

```python
class Wav2LipHealthResponse(BaseModel):
    status: str          # "healthy" or "unhealthy"
    device: str          # "cuda" or "cpu"
    model_loaded: bool
```

---

## Face Detection Service

### Endpoint: `POST /detect`

Detect faces in video or image.

**Request:**
```python
class FaceDetectionRequest(BaseModel):
    source_path: str       # Path to video or image
    sample_rate: int = 1   # Process every N-th frame
    min_confidence: float = 0.5
```

**Response:**
```python
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class DetectedFace(BaseModel):
    frame_index: int
    bounding_box: BoundingBox

class FaceDetectionResponse(BaseModel):
    source_path: str
    total_frames: int
    faces: List[DetectedFace]
    processing_time_seconds: float
```

### Endpoint: `POST /preprocess`

Preprocess and cache face detection for Wav2Lip.

**Request:**
```python
class PreprocessedFacesRequest(BaseModel):
    video_path: str
    character_id: str
```

**Response:**
```python
class PreprocessedFacesResponse(BaseModel):
    character_id: str
    video_path: str
    cache_path: str       # Path to cached data
    frame_count: int
    status: str
```

Note the /preprocess endpoint should store the preprocessing result in a noSQL format to avoid recomputing it every time the same video is queried.
Storing the data on S3, is a good cheap and efficient solution.


---

## TTS Service (Fish Audio)

Uses Fish Audio's API format.

### Endpoint: `POST /v1/tts`

**Request:**
```python
class TTSRequest(BaseModel):
    text: str
    reference_id: str = None    # Character voice ID
    reference_audio: str = None  # Path to reference audio
    format: str = "wav"
```

**Response:**
Binary audio file (wav/mp3).

### Client Abstraction

```python
# packages/vs-core/src/vs_core/api_clients/tts.py

class BaseTTSClient(ABC):
    @abstractmethod
    def synthesize(
        self, 
        text: str, 
        character_id: str,
        output_path: str
    ) -> str:
        """Generate audio and return output path."""
        pass

class FishTTSClient(BaseTTSClient):
    def __init__(self, host: str = "localhost", port: int = 8003):
        self.base_url = f"http://{host}:{port}"
    
    def synthesize(
        self, 
        text: str, 
        character_id: str,
        output_path: str
    ) -> str:
        response = httpx.post(
            f"{self.base_url}/v1/tts",
            json={
                "text": text,
                "reference_id": character_id
            },
            timeout=60.0
        )
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return output_path
```

---

## STT Service (Whisper)

### Endpoint: `POST /transcribe`

**Request:**
```python
class STTRequest(BaseModel):
    audio_path: str
    language: str = "auto"
    task: str = "transcribe"  # or "translate"
```

**Response:**
```python
class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str

class STTResponse(BaseModel):
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration_seconds: float
```

---

## Entity Service

CRUD operations for Characters, VideoClips, and Collections.

### Characters

```
GET    /characters              List all characters
POST   /characters              Create character
GET    /characters/{id}         Get character by ID
PUT    /characters/{id}         Update character
DELETE /characters/{id}         Delete character
```

### Video Clips

```
GET    /clips                   List clips (with filters)
POST   /clips                   Create clip
GET    /clips/{id}              Get clip by ID
PUT    /clips/{id}              Update clip
DELETE /clips/{id}              Delete clip
GET    /clips/{id}/metadata     Get clip metadata
PUT    /clips/{id}/metadata     Update clip metadata
```

### Collections

```
GET    /collections             List collections
POST   /collections             Create collection
GET    /collections/{id}        Get collection by ID
PUT    /collections/{id}        Update collection
DELETE /collections/{id}        Delete collection
GET    /collections/{id}/clips  List clips in collection
```

---

## Client Implementations

### Base Client

```python
# packages/vs-core/src/vs_core/api_clients/base.py

class BaseAPIClient(ABC):
    """Base class for all ML service API clients."""
    
    def __init__(self, host: str, port: int, timeout: float = 300.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def _post(self, endpoint: str, data: dict) -> dict:
        response = self._client.post(
            f"{self.base_url}{endpoint}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def _get(self, endpoint: str) -> dict:
        response = self._client.get(f"{self.base_url}{endpoint}")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        return self._get("/health")
    
    def close(self):
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

### Wav2Lip Client

```python
# packages/vs-core/src/vs_core/api_clients/wav2lip.py

from vs_core.api_clients.base import BaseAPIClient
from vs_core.api_models.wav2lip import Wav2LipRequest, Wav2LipResponse

class Wav2LipClient(BaseAPIClient):
    def __init__(self, host: str = "localhost", port: int = 8001):
        super().__init__(host, port, timeout=300.0)
    
    def generate(self, request: Wav2LipRequest) -> Wav2LipResponse:
        response_data = self._post("/generate", request.model_dump())
        return Wav2LipResponse(**response_data)
    
    def health(self) -> dict:
        return self._get("/health")


def create_wav2lip_client() -> Wav2LipClient:
    """Create client with defaults from environment."""
    import os
    return Wav2LipClient(
        host=os.environ.get("WAV2LIP_HOST", "localhost"),
        port=int(os.environ.get("WAV2LIP_PORT", "8001"))
    )
```

### Face Detection Client

```python
# packages/vs-core/src/vs_core/api_clients/face_detection.py

class FaceDetectionClient(BaseAPIClient):
    def __init__(self, host: str = "localhost", port: int = 8005):
        super().__init__(host, port, timeout=120.0)
    
    def detect(self, request: FaceDetectionRequest) -> FaceDetectionResponse:
        response_data = self._post("/detect", request.model_dump())
        return FaceDetectionResponse(**response_data)
    
    def preprocess_for_wav2lip(
        self, 
        request: PreprocessedFacesRequest
    ) -> PreprocessedFacesResponse:
        response_data = self._post("/preprocess", request.model_dump())
        return PreprocessedFacesResponse(**response_data)
```

---

## Docker Compose

```yaml
# infra/compose.yml
version: '3.8'

services:
  wav2lip:
    build:
      context: ../services/wav2lip_service
    ports:
      - "8001:8001"
    environment:
      - CHECKPOINT_PATH=/models/Wav2Lip.pth
    volumes:
      - ../data:/data
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  face_detection:
    build:
      context: ../services/face_detection_service
    ports:
      - "8005:8005"
    volumes:
      - ../data:/data
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

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

  entity_api:
    build:
      context: ../services/entity_service
    ports:
      - "8002:8002"
    volumes:
      - ../data:/data
    environment:
      - DATA_DIR=/data

volumes:
  models:
  data:
```

---

---

## Jesus Agents API

High-level API endpoints that wrap ADK agents with video generation pipeline.

### Agent Factory

Agents are loaded via a factory registry pattern (`agents/factory.py`):

```python
from virtual_streamer.agents.factory import get_agent, list_agents

# Get available agents
available = list_agents()  # ["greeting_jesus_agent", "answering_jesus_agent"]

# Load an agent by name
agent = get_agent("greeting_jesus_agent")
```

### Endpoint: `POST /api/v1/jesus-agents/greeting/submit`

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

### Endpoint: `POST /api/v1/jesus-agents/answering/submit`

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

### Job Status

Check job status using the shared video-generation jobs endpoint:

```bash
curl "http://localhost:8000/api/v1/video-generation/jobs/{job_id}"
```

**Completed Job Result:**
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

### Pipeline

Both endpoints follow the same pipeline:

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

### Shared Utilities

The implementation uses centralized utilities:

```python
# Character loading (used by TTS, Wav2Lip, legacy_qa, jesus_agents)
from virtual_streamer.utils.character_loader import load_character
character = await load_character("jesus")

# Agent factory (used by jesus_agents)
from virtual_streamer.agents.factory import get_agent
agent = get_agent("greeting_jesus_agent")
```

---

## Environment Variables

Each client reads from environment:

| Variable | Default | Used By |
|----------|---------|---------|
| `WAV2LIP_HOST` | localhost | Wav2LipClient |
| `WAV2LIP_PORT` | 8001 | Wav2LipClient |
| `FACE_DETECTION_HOST` | localhost | FaceDetectionClient |
| `FACE_DETECTION_PORT` | 8005 | FaceDetectionClient |
| `TTS_HOST` | localhost | TTSClient |
| `TTS_PORT` | 8003 | TTSClient |
| `STT_HOST` | localhost | STTClient |
| `STT_PORT` | 8004 | STTClient |
| `ENTITY_API_HOST` | localhost | EntityClient |
| `ENTITY_API_PORT` | 8002 | EntityClient |

