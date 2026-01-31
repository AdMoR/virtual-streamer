"""
Low-level API: Playlist Management

Handles playlist operations for programmations, including:
- Adding videos to playlists
- Getting the next video to play (with fallback logic)
- Marking videos as played
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from virtual_streamer.streaming.models import (
    PlaylistEntry,
    PlaylistStatus,
    NextVideoResponse,
    PlaylistAddRequest,
)
from virtual_streamer.streaming.store import get_streaming_store
from virtual_streamer.utils.minio_client import get_storage_client

# Router setup
router = APIRouter(tags=["Playlist"])


class PlaylistEntryResponse(PlaylistEntry):
    """Playlist entry with optional presigned URL."""
    url: Optional[str] = Field(None, description="Presigned URL for video access")


class MarkPlayedResponse(BaseModel):
    """Response for mark-played endpoint."""
    status: str = "ok"
    entry_id: str
    played_at: datetime


# ========== Programmation Playlist Endpoints ==========

@router.get("/programmations/{programmation_id}/playlist", response_model=List[PlaylistEntry])
async def get_playlist(
    programmation_id: str,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (pending, playing, played, skipped)")
):
    """
    Get all playlist entries for a programmation.
    
    Optionally filter by status. Returns entries ordered by play_order.
    """
    store = await get_streaming_store()
    
    # Verify programmation exists
    programmation = await store.get_programmation(programmation_id)
    if programmation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Programmation '{programmation_id}' not found"
        )
    
    # Validate status filter if provided
    if status_filter and status_filter not in [s.value for s in PlaylistStatus]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {[s.value for s in PlaylistStatus]}"
        )
    
    entries = await store.get_playlist(programmation_id, status_filter)
    return entries


@router.post("/programmations/{programmation_id}/playlist", response_model=PlaylistEntry, status_code=status.HTTP_201_CREATED)
async def add_to_playlist(programmation_id: str, request: PlaylistAddRequest):
    """
    Add a video to a programmation's playlist.
    
    The video will be added with status 'pending' and assigned the next play_order.
    """
    store = await get_streaming_store()
    
    # Verify programmation exists
    programmation = await store.get_programmation(programmation_id)
    if programmation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Programmation '{programmation_id}' not found"
        )
    
    entry = await store.add_to_playlist(
        prog_id=programmation_id,
        video_key=request.video_storage_key,
        metadata=request.metadata,
        play_once=request.play_once,
    )
    
    return entry


# ========== Stream-level Next Video Endpoint ==========

@router.get("/streams/{stream_id}/next-video", response_model=NextVideoResponse)
async def get_next_video_for_stream(stream_id: str):
    """
    Get the next video to play for a stream.
    
    This is the main endpoint for the video server. It:
    1. Finds the active programmation for the current time
    2. Gets the next pending video from the playlist
    3. Falls back to a random played video if no pending videos
    4. Returns null if no videos are available
    
    When a pending video is returned, it's automatically marked as 'playing'.
    """
    store = await get_streaming_store()
    storage = get_storage_client()
    current_time = datetime.now().time()
    
    # Verify stream exists
    stream = await store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    # Find active programmation
    programmation = await store.get_active_programmation(stream_id, current_time)
    if programmation is None:
        return NextVideoResponse(
            video=None,
            programmation=None,
            reason="no_active_programmation"
        )
    
    # Get next video
    entry = await store.get_next_video(programmation.programmation_id)
    if entry is None:
        return NextVideoResponse(
            video=None,
            programmation={
                "id": programmation.programmation_id,
                "name": programmation.name,
            },
            reason="playlist_empty"
        )
    
    # Mark as playing if it was pending
    is_replay = entry.status == PlaylistStatus.PLAYED
    if entry.status == PlaylistStatus.PENDING:
        await store.mark_as_playing(entry.entry_id)
    
    # Generate presigned URL
    try:
        video_url = storage.get_url(entry.video_storage_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate video URL: {str(e)}"
        )
    
    return NextVideoResponse(
        video={
            "entry_id": entry.entry_id,
            "url": video_url,
            "is_replay": is_replay,
            "storage_key": entry.video_storage_key,
            "metadata": entry.metadata,
        },
        programmation={
            "id": programmation.programmation_id,
            "name": programmation.name,
        }
    )


# ========== Playlist Entry Endpoints ==========

@router.get("/playlist/{entry_id}", response_model=PlaylistEntryResponse)
async def get_playlist_entry(
    entry_id: str,
    include_url: bool = Query(False, description="Include presigned URL in response")
):
    """
    Get a specific playlist entry by ID.
    
    Optionally include a presigned URL for video access.
    """
    store = await get_streaming_store()
    entry = await store.get_playlist_entry(entry_id)
    
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playlist entry '{entry_id}' not found"
        )
    
    response = PlaylistEntryResponse(**entry.model_dump())
    
    if include_url:
        storage = get_storage_client()
        try:
            response.url = storage.get_url(entry.video_storage_key)
        except Exception:
            pass  # URL generation failed, leave as None
    
    return response


@router.post("/playlist/{entry_id}/played", response_model=MarkPlayedResponse)
async def mark_video_played(entry_id: str):
    """
    Mark a playlist entry as played.
    
    Called by the video server when a video finishes playing.
    This allows the video to be selected as a fallback (replay) later.
    """
    store = await get_streaming_store()
    
    # Verify entry exists
    entry = await store.get_playlist_entry(entry_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playlist entry '{entry_id}' not found"
        )
    
    await store.mark_as_played(entry_id)
    
    return MarkPlayedResponse(
        status="ok",
        entry_id=entry_id,
        played_at=datetime.utcnow()
    )


@router.put("/playlist/{entry_id}/status")
async def update_entry_status(
    entry_id: str,
    new_status: PlaylistStatus = Query(..., description="New status for the entry")
):
    """
    Update a playlist entry's status.
    
    Available statuses: pending, playing, played, skipped
    """
    store = await get_streaming_store()
    
    # Verify entry exists
    entry = await store.get_playlist_entry(entry_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Playlist entry '{entry_id}' not found"
        )
    
    if new_status == PlaylistStatus.PLAYING:
        await store.mark_as_playing(entry_id)
    elif new_status == PlaylistStatus.PLAYED:
        await store.mark_as_played(entry_id)
    else:
        # For pending and skipped, we need to update directly
        pool = await store._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE playlist_entries SET status = %s WHERE entry_id = %s",
                    (new_status.value, entry_id)
                )
    
    return {"status": "ok", "entry_id": entry_id, "new_status": new_status.value}


@router.get("/streams/{stream_id}/played-since", response_model=List[PlaylistEntry])
async def get_entries_played_since(
    stream_id: str,
    since: datetime = Query(..., description="ISO timestamp to filter played_at > since")
):
    """
    Get playlist entries that were played since a given timestamp.
    Used by the feedback monitor to detect newly played videos.
    
    Returns entries where:
    - status = 'played'
    - played_at > since
    - Entry belongs to a programmation of this stream
    """
    store = await get_streaming_store()
    
    # Verify stream exists
    stream = await store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    entries = await store.get_entries_played_since(stream_id, since)
    return entries
