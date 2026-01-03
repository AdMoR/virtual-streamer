"""
Low-level API: Video Clip Management

Handles CRUD operations for video clip entities and their metadata.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
import uuid
import os
from datetime import datetime

from virtual_streamer.video_server.models import (
    VideoClip,
    VideoClipCreate,
    VideoClipMetadataInput,
)
from virtual_streamer.utils.minio_client import get_storage_client

# Router setup
router = APIRouter(prefix="/clips", tags=["Video Clips"])

# Storage configuration
PREFIX_CLIPS = "clips/"


@router.post("", response_model=VideoClip, status_code=status.HTTP_201_CREATED)
async def create_video_clip(clip_data: VideoClipCreate):
    """Creates a new Video Clip record."""
    storage = get_storage_client()

    clip_id = str(uuid.uuid4())
    now = datetime.utcnow()

    clip = VideoClip(
        clip_id=clip_id,
        storage_path=clip_data.storage_path,
        collection_ids=clip_data.collection_ids,
        metadata=None,  # Metadata added via PUT endpoint
        created_at=now,
        updated_at=now,
    )

    key = f"{PREFIX_CLIPS}{clip_id}.json"
    await storage.put_json(key, clip.dict())

    return clip


@router.get("/{clip_id}", response_model=VideoClip)
async def get_video_clip(clip_id: str):
    """Retrieves a specific Video Clip by its ID."""
    storage = get_storage_client()
    key = f"{PREFIX_CLIPS}{clip_id}.json"
    data = await storage.get_json(key)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    return VideoClip(**data)


@router.put("/{clip_id}/metadata", response_model=VideoClip)
async def update_video_clip_metadata(clip_id: str, metadata: VideoClipMetadataInput):
    """Adds or replaces the metadata for a specific Video Clip."""
    storage = get_storage_client()
    key = f"{PREFIX_CLIPS}{clip_id}.json"
    data = await storage.get_json(key)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    clip = VideoClip(**data)
    clip.metadata = metadata
    clip.updated_at = datetime.utcnow()

    await storage.put_json(key, clip.dict())
    return clip


@router.get("", response_model=List[VideoClip])
async def list_video_clips(
    limit: int = Query(100, ge=1, le=1000), prefix: Optional[str] = None
):
    """Lists Video Clips (metadata only). Limited results."""
    storage = get_storage_client()
    target_prefix = f"{PREFIX_CLIPS}{prefix if prefix else ''}"
    keys = await storage.list_objects(target_prefix)

    clips = []
    count = 0
    for key in keys:
        if key.endswith(".json"):
            data = await storage.get_json(key)
            if data:
                clips.append(VideoClip(**data))
                count += 1
                if count >= limit:
                    break

    return clips


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_clip(clip_id: str):
    """Deletes the metadata record of a Video Clip."""
    storage = get_storage_client()
    key = f"{PREFIX_CLIPS}{clip_id}.json"
    await storage.delete_object(key)
    return None
