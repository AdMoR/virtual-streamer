"""
Streaming module for OBS video streaming infrastructure.

This module provides:
- Data models for stream configuration and scheduling
- Database storage for playlists and programmation
- Video server proxy for OBS browser source
"""

from virtual_streamer.streaming.models import (
    PlaylistStatus,
    StreamConfig,
    StreamConfigBase,
    MediaProgrammation,
    MediaProgrammationBase,
    PlaylistEntry,
    PlaylistEntryBase,
    NextVideoResponse,
    PlaylistAddRequest,
)
from virtual_streamer.streaming.store import (
    StreamingStoreInterface,
    MySQLStreamingStore,
    get_streaming_store,
    reset_streaming_store,
)

__all__ = [
    # Models
    "PlaylistStatus",
    "StreamConfig",
    "StreamConfigBase",
    "MediaProgrammation",
    "MediaProgrammationBase",
    "PlaylistEntry",
    "PlaylistEntryBase",
    "NextVideoResponse",
    "PlaylistAddRequest",
    # Store
    "StreamingStoreInterface",
    "MySQLStreamingStore",
    "get_streaming_store",
    "reset_streaming_store",
]
