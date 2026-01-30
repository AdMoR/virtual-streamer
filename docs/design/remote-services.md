# Remote Services

This document describes the external/remote services that Virtual Streamer depends on for video search, file storage, and other capabilities.

## Overview

Virtual Streamer follows a **remote-first architecture** where heavy processing and storage are delegated to specialized services:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VIRTUAL STREAMER APPLICATION                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐       ┌───────────────┐
│ Video Search  │      │    MinIO      │       │    Qdrant     │
│    Server     │      │  (Storage)    │       │  (Vectors)    │
│  :8003        │      │  :9000        │       │  :6333        │
└───────────────┘      └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                        ┌───────────────┐
                        │  Video Files  │
                        │  (on disk)    │
                        └───────────────┘
```

---

## Video Search Service

The Video Search Service provides semantic search over video clips using VideoPrism embeddings stored in Qdrant.

### Purpose

- **Semantic video search**: Find video clips matching natural language queries
- **Tag-based filtering**: Filter results by character presence (e.g., `person:fred`)
- **Multi-modal retrieval**: Combines visual and text understanding

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO SEARCH SERVER                          │
│                        (Port 8003)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  REST API   │────▶│  VideoPrism │────▶│   Qdrant    │       │
│  │  /search    │     │  Embedding  │     │  Vector DB  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Client Usage

```python
from virtual_streamer.video_search.client import VideoSearchClient, VideoSearchResult

# Initialize client
client = VideoSearchClient(server_url="http://localhost:8003")

# Search for videos
results: list[VideoSearchResult] = client.search(
    query="person explaining science in a workshop",
    collection="cest_pas_sorcier",
    top_k=10,
    tags=["person:fred"],  # Filter by character tag
    tag_mode="all"         # "all" (AND) or "any" (OR)
)

# Use results
for result in results:
    print(f"Video: {result.path}")
    print(f"Similarity: {result.similarity:.4f}")
    print(f"Duration: {result.duration}s")
    print(f"Tags: {[t.name for t in result.tags]}")
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check with model and Qdrant status |
| `/collections` | GET | List available Qdrant collections |
| `/tags/{collection}` | GET | List unique tags in a collection |
| `/search` | POST | Search for video segments by text similarity |

### Search Request

```json
POST /search
{
    "query": "person dancing happily",
    "collection": "my_videos",
    "top_k": 5,
    "prompt_template": "a video of {}.",
    "tags": ["person:fred"],
    "tag_mode": "all"
}
```

### Search Response

```json
{
    "results": [
        {
            "segment_id": "vid_001_seg_003",
            "video_id": "vid_001",
            "segment_index": 3,
            "duration": 5.2,
            "path": "/data/videos/fred_workshop_003.mp4",
            "tags": [
                {"name": "person:fred", "start": 0.0, "end": 5.2}
            ],
            "similarity": 0.8542
        }
    ]
}
```

### Data Models

```python
@dataclass
class TagInfo:
    """Tag information associated with a video segment."""
    name: str      # Tag name (e.g., "person:fred")
    start: float   # Start time in segment
    end: float     # End time in segment


@dataclass
class VideoSearchResult:
    """Result from a video search query."""
    segment_id: str           # Unique segment identifier
    video_id: str             # Parent video identifier
    segment_index: int        # Index within the video
    duration: float           # Segment duration in seconds
    path: str                 # File path to the video segment
    tags: list[TagInfo]       # Associated tags with timing
    similarity: float         # Cosine similarity score (0-1)
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VIDEO_SEARCH_SERVER_URL` | `http://localhost:8003` | Video Search Server base URL |

### Collections

Collections in Qdrant organize video embeddings by source:

| Collection | Description | Tags |
|------------|-------------|------|
| `cest_pas_sorcier` | C'est pas Sorcier episodes | `person:fred`, `person:jamy`, `person:sabine` |
| `ai_jesus_clips` | AI Jesus reference clips | `person:jesus` |

---

## MinIO Object Storage

MinIO provides S3-compatible object storage for all media files (audio, video, images).

### Purpose

- **Centralized storage**: Single source for all media files
- **S3 compatibility**: Standard API for file operations
- **Shared access**: Multiple services can access the same files

### Storage Structure

```
minio-bucket/
├── audio/                    # TTS-generated audio files
│   ├── tts_output_001.wav
│   └── voice_sample_fred_1.wav
│
├── clips/                    # Video clips for Wav2Lip
│   ├── fred_reference.mp4
│   └── jamy_reference.mp4
│
├── identity_images/          # Character reference images
│   ├── fred_face_1.jpg
│   ├── fred_face_2.jpg
│   └── jamy_face_1.jpg
│
├── output/                   # Generated output videos
│   └── final_video_001.mp4
│
└── characters/               # Character JSON definitions
    ├── fred.json
    └── jamy.json
```

### Client Usage

```python
from virtual_streamer.utils.storage_client import StorageClient

# Initialize client
storage = StorageClient(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    bucket="virtual-streamer"
)

# Upload a file
await storage.put_object(
    key="audio/tts_output_001.wav",
    data=audio_bytes,
    content_type="audio/wav"
)

# Download a file
audio_data = await storage.get_object("audio/tts_output_001.wav")

# Upload JSON data
await storage.put_json(
    key="characters/fred.json",
    data=character.model_dump()
)

# Get JSON data
character_data = await storage.get_json("characters/fred.json")
```

### Storage Paths by Model

| Model | Storage Prefix | Content Type |
|-------|---------------|--------------|
| Character voice samples | `audio/` | `audio/wav` |
| Character video clips | `clips/` | `video/mp4` |
| Character identity images | `identity_images/` | `image/jpeg`, `image/png` |
| Character definitions | `characters/` | `application/json` |
| TTS output | `audio/` | `audio/wav` |
| Lip-sync output | `output/` | `video/mp4` |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | Access key |
| `MINIO_SECRET_KEY` | `minioadmin` | Secret key |
| `MINIO_BUCKET` | `virtual-streamer` | Default bucket name |
| `MINIO_SECURE` | `false` | Use HTTPS |

---

## Qdrant Vector Database

Qdrant stores video embeddings for semantic search.

### Purpose

- **Vector storage**: Store VideoPrism embeddings for video segments
- **Similarity search**: Fast nearest-neighbor search for retrieval
- **Metadata filtering**: Filter by tags, timestamps, etc.

### Architecture

The Video Search Server acts as an intermediary:

```
Virtual Streamer → Video Search Server → Qdrant
                      (embedding)         (storage/search)
```

### Collection Schema

Each Qdrant collection stores:

```json
{
    "id": "segment_unique_id",
    "vector": [0.1, 0.2, ...],  // VideoPrism embedding (1408-dim)
    "payload": {
        "video_id": "video_001",
        "segment_index": 3,
        "path": "/data/videos/video_001_seg_003.mp4",
        "duration": 5.2,
        "tags": [
            {"name": "person:fred", "start": 0.0, "end": 5.2}
        ]
    }
}
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

---

## Service Dependencies

### Required for Video Generation

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| MinIO | 9000 | File storage | ✅ Yes |
| Video Search Server | 8003 | Video retrieval | ✅ Yes |
| Qdrant | 6333 | Vector storage | ✅ Yes (via Video Search) |
| TTS Service | 8003 | Text-to-speech | ✅ Yes |
| Wav2Lip Service | 8001 | Lip synchronization | ✅ Yes |

### Docker Compose Example

```yaml
version: '3.8'

services:
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

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage

  video-search:
    image: video-embedding-server:latest
    ports:
      - "8003:8003"
    environment:
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
    depends_on:
      - qdrant

volumes:
  minio-data:
  qdrant-data:
```

---

## Integration with Character Model

> **Note**: See [`virtual_streamer/video_server/models.py`](../../virtual_streamer/video_server/models.py) for the canonical Character model definition.

The Character model integrates with remote services:

```python
class Character(BaseModel):
    # ... other fields (see canonical definition) ...
    
    # Stored in MinIO
    video_clip_path: str          # clips/fred_reference.mp4
    voice_samples: List[VoiceSample]  # audio/fred_sample1.wav
    identity_images: List[str]    # identity_images/fred_1.jpg
    
    # Used for Video Search filtering
    video_search_tag: str         # person:fred
```

### Data Flow: Video Generation

```
1. User requests video for character "fred"
   
2. System loads Character from MinIO
   → GET characters/fred.json
   
3. System searches for relevant clips
   → VideoSearchClient.search(
       query="explaining science",
       tags=["person:fred"]  # From character.video_search_tag
     )
   
4. System generates TTS audio
   → Voice samples from: audio/fred_sample*.wav
   
5. System runs Wav2Lip
   → Reference video from: clips/fred_reference.mp4
   
6. System saves output
   → PUT output/final_video.mp4
```

---

## Health Checks

### Video Search Server

```bash
curl http://localhost:8003/health
```

Response:
```json
{
    "status": "healthy",
    "model": "videoprism",
    "qdrant_host": "localhost:6333",
    "collections": ["cest_pas_sorcier", "ai_jesus_clips"]
}
```

### MinIO

```bash
curl http://localhost:9000/minio/health/live
```

### Qdrant

```bash
curl http://localhost:6333/health
```

---

## Error Handling

### Video Search Errors

| Error | HTTP Code | Cause |
|-------|-----------|-------|
| Collection not found | 404 | Invalid collection name |
| Invalid tag_mode | 400 | tag_mode must be "all" or "any" |
| Server unavailable | 503 | Qdrant or model not ready |

### MinIO Errors

| Error | Cause |
|-------|-------|
| `NoSuchBucket` | Bucket doesn't exist |
| `NoSuchKey` | File not found |
| `AccessDenied` | Invalid credentials |

### Best Practices

1. **Always check service health** before starting video generation
2. **Use retries** with exponential backoff for transient failures
3. **Validate file paths** exist in MinIO before processing
4. **Log all API calls** for debugging

