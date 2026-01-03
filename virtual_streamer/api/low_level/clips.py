"""
Low-level API: Video Clip Management

Handles CRUD operations for video clip entities and their metadata.
Uses MySQL for metadata storage.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional

from virtual_streamer.video_server.models import (
    VideoClip,
    VideoClipCreate,
    VideoClipMetadataInput,
    CharacterPresence,
)
from virtual_streamer.utils.entity_repository import get_entity_repository

# Router setup
router = APIRouter(prefix="/clips", tags=["Video Clips"])


@router.post("", response_model=VideoClip, status_code=status.HTTP_201_CREATED)
async def create_video_clip(clip_data: VideoClipCreate):
    """Creates a new Video Clip record."""
    repo = get_entity_repository()

    import uuid
    clip_id = str(uuid.uuid4())

    clip_dict = await repo.create_video_clip(
        clip_id=clip_id,
        storage_path=clip_data.storage_path,
        collection_ids=clip_data.collection_ids,
    )

    return _dict_to_video_clip(clip_dict)


@router.get("/{clip_id}", response_model=VideoClip)
async def get_video_clip(clip_id: str):
    """Retrieves a specific Video Clip by its ID."""
    repo = get_entity_repository()
    clip_dict = await repo.get_video_clip(clip_id)

    if clip_dict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    return _dict_to_video_clip(clip_dict)


@router.put("/{clip_id}/metadata", response_model=VideoClip)
async def update_video_clip_metadata(clip_id: str, metadata: VideoClipMetadataInput):
    """Adds or replaces the metadata for a specific Video Clip."""
    repo = get_entity_repository()

    # Check if clip exists
    existing = await repo.get_video_clip(clip_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    # Update metadata
    clip_dict = await repo.update_video_clip_metadata(
        clip_id=clip_id,
        duration=metadata.duration,
        scene_description_text=metadata.scene_description_text,
        scene_keywords=metadata.scene_keywords,
        character_presences=[
            {
                "character_id": p.character_id,
                "start_time": p.start_time,
                "end_time": p.end_time,
            }
            for p in metadata.character_presences
        ] if metadata.character_presences else None,
        source_show_name=metadata.source_show_name,
        source_episode_name=metadata.source_episode_name,
        start_time_in_source=metadata.start_time_in_source,
        end_time_in_source=metadata.end_time_in_source,
    )

    return _dict_to_video_clip(clip_dict)


@router.get("", response_model=List[VideoClip])
async def list_video_clips(
    limit: int = Query(100, ge=1, le=1000), prefix: Optional[str] = None
):
    """Lists Video Clips (metadata only). Limited results."""
    repo = get_entity_repository()
    clips_data = await repo.list_video_clips(limit)

    return [_dict_to_video_clip(c) for c in clips_data]


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_clip(clip_id: str):
    """Deletes the metadata record of a Video Clip."""
    repo = get_entity_repository()
    deleted = await repo.delete_video_clip(clip_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    return None


def _dict_to_video_clip(clip_dict: dict) -> VideoClip:
    """Convert repository dict to Pydantic VideoClip model."""
    metadata = None
    if clip_dict.get("metadata"):
        m = clip_dict["metadata"]
        metadata = VideoClipMetadataInput(
            duration=m.get("duration", 0.0),
            scene_description_text=m.get("scene_description_text"),
            scene_keywords=m.get("scene_keywords", []),
            character_presences=[
                CharacterPresence(
                    character_id=p["character_id"],
                    start_time=p["start_time"],
                    end_time=p["end_time"],
                )
                for p in m.get("character_presences", [])
            ],
            source_show_name=m.get("source_show_name"),
            source_episode_name=m.get("source_episode_name"),
            start_time_in_source=m.get("start_time_in_source"),
            end_time_in_source=m.get("end_time_in_source"),
        )

    return VideoClip(
        clip_id=clip_dict["clip_id"],
        storage_path=clip_dict["storage_path"],
        collection_ids=clip_dict.get("collection_ids", []),
        metadata=metadata,
        created_at=clip_dict.get("created_at"),
        updated_at=clip_dict.get("updated_at"),
    )
