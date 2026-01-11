"""
Streaming data models.

These models define the core entities for the streaming infrastructure:
- StreamConfig: Generic stream configuration
- MediaProgrammation: Time-based schedule linking to a StoryTemplate
- PlaylistEntry: Video in a programmation's playlist
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, time
from enum import Enum


class PlaylistStatus(str, Enum):
    """Status of a playlist entry."""
    PENDING = "pending"
    PLAYING = "playing"
    PLAYED = "played"
    SKIPPED = "skipped"


class StreamConfigBase(BaseModel):
    """Base model for stream configuration (used for creation)."""
    name: str = Field(..., description="Display name of the stream")
    description: Optional[str] = Field(None, description="Optional description")
    is_active: bool = Field(default=True, description="Whether the stream is active")


class StreamConfig(StreamConfigBase):
    """Generic stream configuration."""
    stream_id: str = Field(..., description="Unique stream identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class MediaProgrammationBase(BaseModel):
    """Base model for media programmation (used for creation)."""
    stream_id: str = Field(..., description="Which stream this belongs to")
    story_template_id: str = Field(..., description="StoryTemplate used for generation")
    name: str = Field(..., description="Display name, e.g., 'News Hour'")
    start_time: time = Field(..., description="Daily start time (e.g., 12:00)")
    end_time: time = Field(..., description="Daily end time (e.g., 13:00)")
    priority: int = Field(default=0, description="Higher priority wins on overlap")
    is_active: bool = Field(default=True, description="Whether the programmation is active")


class MediaProgrammation(MediaProgrammationBase):
    """Time-based schedule linking to a StoryTemplate."""
    programmation_id: str = Field(..., description="Unique programmation identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class PlaylistEntryBase(BaseModel):
    """Base model for playlist entry (used for creation)."""
    video_storage_key: str = Field(..., description="MinIO storage key for the video")
    play_order: int = Field(default=0, description="Order within playlist (lower = earlier)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PlaylistEntry(PlaylistEntryBase):
    """Video in a programmation's playlist."""
    entry_id: str = Field(..., description="Unique entry identifier")
    programmation_id: str = Field(..., description="Which programmation this belongs to")
    status: PlaylistStatus = Field(default=PlaylistStatus.PENDING, description="Playback status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    played_at: Optional[datetime] = Field(None, description="When the video was last played")

    class Config:
        from_attributes = True


# Response models for API

class NextVideoResponse(BaseModel):
    """Response for the next-video endpoint."""
    video: Optional[Dict[str, Any]] = Field(
        None, 
        description="Video information (entry_id, url, is_replay) or None if no video"
    )
    programmation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Active programmation info (id, name)"
    )
    reason: Optional[str] = Field(
        None, 
        description="Reason if no video (no_active_programmation, playlist_empty)"
    )


class PlaylistAddRequest(BaseModel):
    """Request to add a video to a playlist."""
    video_storage_key: str = Field(..., description="MinIO storage key for the video")
    play_order: Optional[int] = Field(None, description="Order within playlist")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
