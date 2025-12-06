# Data Models

This document defines the core entity models for the Virtual Streamer system.

## Design Principles

1. **Single source of truth** - One model per entity, no redundancy
2. **Modality-agnostic** - Models work for voice, video, and detection use cases
3. **Minimal but complete** - Include only necessary fields, but all of them
4. **Pydantic-based** - Use Pydantic v2 for validation and serialization

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
```

---

## Character Model

A character that can appear in generated videos. Contains all information needed across modalities.

```python
# packages/vs-core/src/vs_core/models/character.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class VoiceSample(BaseModel):
    """Audio sample for voice cloning."""
    storage_path: str = Field(..., description="Path to audio file (wav/mp3)")
    transcript: str = Field(..., description="Exact transcript of the audio")
    duration_seconds: Optional[float] = None


class Character(BaseModel):
    """
    A character that can appear in generated videos.
    
    Contains all information needed across modalities:
    - Voice: samples for TTS voice cloning
    - Video: reference video for Wav2Lip
    - Detection: face images for recognition
    
    Examples:
        - Jesus (AI Jesus project)
        - Fred (C'est pas Sorcier)
        - Jamy (C'est pas Sorcier)
    """
    
    # === Identity ===
    character_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Character description")
    
    # === Voice Configuration ===
    voice_samples: List[VoiceSample] = Field(
        default_factory=list,
        description="Audio samples for voice cloning"
    )
    tts_config: Optional[Dict[str, Any]] = Field(
        None,
        description="TTS provider-specific configuration"
    )
    
    # === Video Configuration ===
    representative_video_path: Optional[str] = Field(
        None,
        description="Path to reference video for Wav2Lip"
    )
    
    # === Face Detection ===
    face_reference_paths: List[str] = Field(
        default_factory=list,
        description="Paths to face reference images for detection"
    )
    
    # === Metadata ===
    source_show: Optional[str] = Field(
        None,
        description="Original show/source (e.g., 'C'est pas Sorcier')"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "character_id": "fred",
                "name": "Fred",
                "description": "Bombastic host of C'est pas Sorcier",
                "voice_samples": [
                    {
                        "storage_path": "/data/voices/fred_sample1.wav",
                        "transcript": "Eh bien dis donc Jamy...",
                        "duration_seconds": 5.2
                    }
                ],
                "representative_video_path": "/data/video/fred_reference.mp4",
                "face_reference_paths": [
                    "/data/faces/fred_1.jpg",
                    "/data/faces/fred_2.jpg"
                ],
                "source_show": "C'est pas Sorcier"
            }
        }
```

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

