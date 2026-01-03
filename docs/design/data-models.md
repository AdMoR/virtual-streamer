# Data Models

This document defines the core entity models for the Virtual Streamer system.

## Design Principles

1. **Single source of truth** - One model per entity, no redundancy
2. **Modality-agnostic** - Models work for voice, video, and detection use cases
3. **Minimal but complete** - Include only necessary fields, but all of them
4. **Pydantic-based** - Use Pydantic v2 for validation and serialization
5. **Remote-first** - Models support remote storage (MinIO) and remote services (Video Search)

---

## Entity Relationship

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  Collection │──────<│  VideoClip  │>──────│  Character  │
└─────────────┘ 1:N   └─────────────┘ N:M   └─────────────┘
      │                      │                     │
      │ contains             │ features            │ appears in
      ▼                      ▼                     ▼
   clip_ids            character_presences   representative_video
                                                   │
                                                   ▼
                                           video_search_tag
                                           identity_images
```

---

## Character Model

A character that can appear in generated videos. Contains all information needed across modalities.

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `character_id` | `str` | ✅ | Unique identifier (often same as name) |
| `name` | `str` | ✅ | Display name of the character |
| `description` | `str` | ❌ | Optional character description |
| `video_clip_path` | `str` | ✅ | Storage path to representative video for Wav2Lip |
| `voice_samples` | `List[VoiceSample]` | ✅ | Audio samples for voice cloning |
| `video_search_tag` | `str` | ❌ | Tag for filtering videos in remote search (e.g., `person:fred`) |
| `identity_images` | `List[str]` | ❌ | Storage paths to identity/reference images |
| `created_at` | `datetime` | Auto | Creation timestamp |
| `updated_at` | `datetime` | Auto | Last update timestamp |

### Model Definition

```python
# virtual_streamer/video_server/models.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class VoiceSample(BaseModel):
    """Audio sample for voice cloning."""
    sample_storage_path: str = Field(..., description="Path/key to the speaker WAV file (e.g., MinIO key)")
    transcript: str = Field(..., description="Accurate transcript of the WAV file")


class CharacterBase(BaseModel):
    """Base character fields for creation."""
    name: str = Field(..., description="Display name of the character/voice")
    description: Optional[str] = Field(None, description="Optional description")
    video_clip_path: str = Field(
        ..., description="Storage path or URL to the representative video clip"
    )
    voice_samples: List[VoiceSample] = Field(
        ..., min_items=0, description="Samples used to define/clone the voice"
    )
    video_search_tag: Optional[str] = Field(
        None, description="Tag for filtering videos in search (e.g., 'person:fred')"
    )
    identity_images: List[str] = Field(
        default_factory=list,
        description="Storage paths to identity/reference images for the character",
    )


class Character(CharacterBase):
    """
    A character that can appear in generated videos.
    
    Contains all information needed across modalities:
    - Voice: samples for TTS voice cloning
    - Video: reference video for Wav2Lip lip-sync
    - Search: tag for filtering in Video Search service
    - Identity: reference images for face detection/recognition
    
    Examples:
        - Fred (C'est pas Sorcier host)
        - Jamy (C'est pas Sorcier scientist)
        - Jesus (AI Jesus project)
    """
    character_id: str = Field(..., description="Unique identifier for the character/voice")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True
        json_schema_extra = {
            "example": {
                "character_id": "fred",
                "name": "Fred",
                "description": "Bombastic host of C'est pas Sorcier",
                "video_clip_path": "clips/fred_reference.mp4",
                "voice_samples": [
                    {
                        "sample_storage_path": "audio/fred_sample1.wav",
                        "transcript": "Eh bien dis donc Jamy..."
                    }
                ],
                "video_search_tag": "person:fred",
                "identity_images": [
                    "identity_images/fred_face_1.jpg",
                    "identity_images/fred_face_2.jpg"
                ]
            }
        }
```

### Video Search Tag

The `video_search_tag` field enables character-specific filtering when searching for video clips:

- **Format**: Typically `person:<character_name>` (e.g., `person:fred`, `person:jamy`)
- **Usage**: Passed to the Video Search Service to filter results by character presence
- **Integration**: When generating videos, the system uses this tag to find clips where the character appears

```python
# Example: Search for clips featuring Fred
video_results = video_search_client.search(
    query="explaining science",
    collection="cest_pas_sorcier",
    tags=["person:fred"],  # Uses character's video_search_tag
    tag_mode="all"
)
```

### Identity Images

The `identity_images` field stores paths to reference images for the character:

- **Purpose**: Used for face detection, recognition, and video filtering
- **Storage**: Paths are relative to MinIO bucket (e.g., `identity_images/fred_1.jpg`)
- **Upload**: Added during character registration via the Character API

### Character Registration

Characters are registered via the `/api/v1/characters` endpoint:

```bash
# Register a new character with all fields
python scripts/register_character.py \
    --name "fred" \
    --description "Host of C'est pas Sorcier" \
    --audio-dir ./samples/fred_voice/ \
    --video ./videos/fred_talking.mp4 \
    --identity-images ./faces/fred_1.jpg ./faces/fred_2.jpg \
    --video-search-tag "person:fred"
```

The API accepts multipart form data:
- `name`: Character name (used as character_id)
- `description`: Optional description
- `voice_files`: Audio files for voice cloning
- `transcripts`: Transcripts for each audio file
- `video_file`: Representative video for Wav2Lip
- `identity_files`: Identity/reference images
- `video_search_tag`: Tag for video search filtering

---

## VideoClip Model

A segment of video with associated metadata.

```python
# packages/vs-core/src/vs_core/models/video_clip.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class CharacterPresence(BaseModel):
    """Records when a character appears in a clip."""
    character_id: str = Field(..., description="Reference to Character")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")


class VideoClip(BaseModel):
    """
    A segment of video with associated metadata.
    
    Can be:
    - A source clip extracted from a larger video (Collection)
    - A generated output clip
    
    Metadata can be populated by the video_indexer tool.
    """
    
    # === Identity ===
    clip_id: str = Field(..., description="Unique identifier")
    storage_path: str = Field(..., description="Path to video file")
    
    # === Collection Reference ===
    collection_id: Optional[str] = Field(
        None,
        description="Parent collection ID (if from a collection)"
    )
    
    # === Content Metadata ===
    duration: Optional[float] = Field(None, gt=0, description="Duration in seconds")
    transcription: Optional[str] = Field(None, description="Speech transcription")
    visual_description: Optional[str] = Field(
        None, 
        description="Visual scene description (from Florence or similar)"
    )
    keywords: List[str] = Field(default_factory=list, description="Scene keywords")
    
    # === Character Information ===
    character_presences: List[CharacterPresence] = Field(
        default_factory=list,
        description="Characters detected in this clip"
    )
    
    # === Source Information ===
    source_video_path: Optional[str] = Field(
        None,
        description="Original video this was extracted from"
    )
    start_time_in_source: Optional[float] = Field(
        None, ge=0,
        description="Start time in source video"
    )
    end_time_in_source: Optional[float] = Field(
        None, gt=0,
        description="End time in source video"
    )
    
    # === Embeddings (for retrieval) ===
    text_embedding: Optional[List[float]] = Field(
        None,
        description="Text embedding of transcription + description"
    )
    visual_embedding: Optional[List[float]] = Field(
        None,
        description="Visual embedding (e.g., CLIP)"
    )
    
    # === Metadata ===
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def primary_character(self) -> Optional[str]:
        """Returns the character with most screen time."""
        if not self.character_presences:
            return None
        return max(
            self.character_presences,
            key=lambda p: p.end_time - p.start_time
        ).character_id
    
    class Config:
        json_schema_extra = {
            "example": {
                "clip_id": "cps_s01e05_scene_042",
                "storage_path": "/data/clips/cps_s01e05_scene_042.mp4",
                "collection_id": "cest_pas_sorcier_s01",
                "duration": 12.5,
                "transcription": "Fred explique le fonctionnement du moteur...",
                "visual_description": "Fred in workshop pointing at engine diagram",
                "keywords": ["workshop", "engine", "explanation"],
                "character_presences": [
                    {
                        "character_id": "fred",
                        "start_time": 0.0,
                        "end_time": 12.5,
                        "confidence": 0.95
                    }
                ],
                "source_video_path": "/data/sources/cps_s01e05.mp4",
                "start_time_in_source": 342.0,
                "end_time_in_source": 354.5
            }
        }
```

---

## Collection Model

A collection of video clips from a common source.

```python
# packages/vs-core/src/vs_core/models/collection.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class Collection(BaseModel):
    """
    A collection of video clips from a common source.
    
    Examples:
        - "C'est pas Sorcier Season 1"
        - "Friends Season 1"
        - "AI Jesus Response Archive"
    
    Collections group related clips for:
    - Organized storage
    - Efficient retrieval
    - Training data management
    """
    
    # === Identity ===
    collection_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: Optional[str] = None
    
    # === Source Information ===
    source_show: Optional[str] = Field(
        None,
        description="Name of source show/content"
    )
    source_path: Optional[str] = Field(
        None,
        description="Path to original source material"
    )
    
    # === Statistics (computed) ===
    clip_count: int = Field(0, description="Number of clips in collection")
    total_duration: float = Field(0.0, description="Total duration in seconds")
    characters: List[str] = Field(
        default_factory=list,
        description="Character IDs found in this collection"
    )
    
    # === Processing Status ===
    is_indexed: bool = Field(False, description="Whether clips have been indexed")
    indexing_progress: Optional[float] = Field(
        None, ge=0, le=1,
        description="Indexing progress (0-1)"
    )
    
    # === Metadata ===
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "collection_id": "cest_pas_sorcier_s01",
                "name": "C'est pas Sorcier - Season 1",
                "description": "First season of the classic French science show",
                "source_show": "C'est pas Sorcier",
                "source_path": "/data/sources/cps_season_1/",
                "clip_count": 342,
                "total_duration": 14523.5,
                "characters": ["fred", "jamy", "sabine"],
                "is_indexed": True
            }
        }
```

---

## Model Exports

```python
# packages/vs-core/src/vs_core/models/__init__.py

from .character import Character, VoiceSample
from .video_clip import VideoClip, CharacterPresence
from .collection import Collection

__all__ = [
    "Character",
    "VoiceSample",
    "VideoClip",
    "CharacterPresence",
    "Collection",
]
```

---

## Storage Considerations

These models can be persisted in various ways:

### Option 1: JSON Files (Simple)
```python
# Storage as JSON files
/data/entities/
├── characters/
│   ├── fred.json
│   ├── jamy.json
│   └── jesus.json
├── clips/
│   ├── clip_001.json
│   └── clip_002.json
└── collections/
    └── cps_s01.json
```

### Option 2: SQLite (Indexed)
```python
# For larger datasets with search requirements
from sqlmodel import SQLModel, Field as SQLField

class CharacterDB(SQLModel, table=True):
    character_id: str = SQLField(primary_key=True)
    name: str
    # ... other fields as columns
```

### Option 3: Entity Service API
```python
# Through the entity_service HTTP API
client = EntityClient(host="localhost", port=8002)
character = client.get_character("fred")
```

The current design uses **Option 3** (Entity Service API) for production and **Option 1** (JSON files) for development/testing.

