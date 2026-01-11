"""
Low-level API: Stream Configuration Management

Handles CRUD operations for stream configurations.
Streams are the top-level entity for video streaming infrastructure.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from virtual_streamer.streaming.models import StreamConfig, StreamConfigBase
from virtual_streamer.streaming.store import get_streaming_store

# Router setup
router = APIRouter(prefix="/streams", tags=["Streams"])


class StreamCreateRequest(StreamConfigBase):
    """Request body for creating a stream."""
    stream_id: Optional[str] = Field(None, description="Optional custom stream ID (auto-generated if not provided)")


class StreamUpdateRequest(BaseModel):
    """Request body for updating a stream."""
    name: Optional[str] = Field(None, description="New display name")
    description: Optional[str] = Field(None, description="New description")
    is_active: Optional[bool] = Field(None, description="Whether the stream is active")


@router.post("", response_model=StreamConfig, status_code=status.HTTP_201_CREATED)
async def create_stream(request: StreamCreateRequest):
    """
    Create a new stream configuration.
    
    A stream represents a single streaming instance (e.g., a Twitch channel or OBS output).
    Streams contain programmations which define the content schedule.
    """
    store = await get_streaming_store()
    
    data = request.model_dump()
    if data.get("stream_id"):
        # Check if stream already exists
        existing = await store.get_stream(data["stream_id"])
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Stream with ID '{data['stream_id']}' already exists"
            )
    
    stream = await store.create_stream(data)
    return stream


@router.get("", response_model=List[StreamConfig])
async def list_streams():
    """
    List all stream configurations.
    
    Returns all streams ordered by creation date (newest first).
    """
    store = await get_streaming_store()
    streams = await store.list_streams()
    return streams


@router.get("/{stream_id}", response_model=StreamConfig)
async def get_stream(stream_id: str):
    """
    Get a specific stream configuration by ID.
    """
    store = await get_streaming_store()
    stream = await store.get_stream(stream_id)
    
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    return stream


@router.put("/{stream_id}", response_model=StreamConfig)
async def update_stream(stream_id: str, request: StreamUpdateRequest):
    """
    Update a stream configuration.
    
    Only provided fields are updated. Omit fields to keep existing values.
    """
    store = await get_streaming_store()
    
    # Check if stream exists
    existing = await store.get_stream(stream_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    # Filter out None values
    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not update_data:
        return existing
    
    stream = await store.update_stream(stream_id, update_data)
    return stream


@router.delete("/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stream(stream_id: str):
    """
    Delete a stream configuration.
    
    WARNING: This also deletes all associated programmations and playlist entries (cascading delete).
    """
    store = await get_streaming_store()
    deleted = await store.delete_stream(stream_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    return None
