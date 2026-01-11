"""
Low-level API: Media Programmation Management

Handles CRUD operations for media programmations.
Programmations define time-based schedules linking StoryTemplates to playlists.
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime, time
from pydantic import BaseModel, Field

from virtual_streamer.streaming.models import MediaProgrammation, MediaProgrammationBase
from virtual_streamer.streaming.store import get_streaming_store

# Router setup
router = APIRouter(tags=["Programmations"])


class ProgrammationCreateRequest(BaseModel):
    """Request body for creating a programmation."""
    programmation_id: Optional[str] = Field(None, description="Optional custom ID (auto-generated if not provided)")
    story_template_id: str = Field(..., description="StoryTemplate used for generation")
    name: str = Field(..., description="Display name, e.g., 'News Hour'")
    start_time: time = Field(..., description="Daily start time")
    end_time: time = Field(..., description="Daily end time")
    priority: int = Field(default=0, description="Higher priority wins on overlap")
    is_active: bool = Field(default=True, description="Whether the programmation is active")


class ProgrammationUpdateRequest(BaseModel):
    """Request body for updating a programmation."""
    story_template_id: Optional[str] = Field(None, description="New story template ID")
    name: Optional[str] = Field(None, description="New display name")
    start_time: Optional[time] = Field(None, description="New start time")
    end_time: Optional[time] = Field(None, description="New end time")
    priority: Optional[int] = Field(None, description="New priority")
    is_active: Optional[bool] = Field(None, description="New active status")


# Nested under streams
@router.get("/streams/{stream_id}/programmations", response_model=List[MediaProgrammation])
async def list_programmations(stream_id: str):
    """
    List all programmations for a stream.
    
    Returns programmations ordered by priority (highest first), then start time.
    """
    store = await get_streaming_store()
    
    # Verify stream exists
    stream = await store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    programmations = await store.list_programmations(stream_id)
    return programmations


@router.post("/streams/{stream_id}/programmations", response_model=MediaProgrammation, status_code=status.HTTP_201_CREATED)
async def create_programmation(stream_id: str, request: ProgrammationCreateRequest):
    """
    Create a new programmation for a stream.
    
    Programmations define when specific content (via StoryTemplate) should be scheduled.
    If multiple programmations overlap in time, the one with higher priority wins.
    """
    store = await get_streaming_store()
    
    # Verify stream exists
    stream = await store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    # Validate time range
    if request.start_time >= request.end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be before end_time"
        )
    
    data = request.model_dump()
    data["stream_id"] = stream_id
    
    if data.get("programmation_id"):
        # Check if programmation already exists
        existing = await store.get_programmation(data["programmation_id"])
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Programmation with ID '{data['programmation_id']}' already exists"
            )
    
    programmation = await store.create_programmation(data)
    return programmation


@router.get("/streams/{stream_id}/programmations/active", response_model=Optional[MediaProgrammation])
async def get_active_programmation(
    stream_id: str,
    at_time: Optional[time] = Query(None, description="Time to check (defaults to current time)")
):
    """
    Get the currently active programmation for a stream.
    
    If multiple programmations are active at the given time, returns the one with highest priority.
    Returns null if no programmation is active.
    """
    store = await get_streaming_store()
    
    # Verify stream exists
    stream = await store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream '{stream_id}' not found"
        )
    
    # Use current time if not specified
    check_time = at_time or datetime.now().time()
    
    programmation = await store.get_active_programmation(stream_id, check_time)
    return programmation


# Direct programmation endpoints
@router.get("/programmations/{programmation_id}", response_model=MediaProgrammation)
async def get_programmation(programmation_id: str):
    """
    Get a specific programmation by ID.
    """
    store = await get_streaming_store()
    programmation = await store.get_programmation(programmation_id)
    
    if programmation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Programmation '{programmation_id}' not found"
        )
    
    return programmation


@router.put("/programmations/{programmation_id}", response_model=MediaProgrammation)
async def update_programmation(programmation_id: str, request: ProgrammationUpdateRequest):
    """
    Update a programmation.
    
    Only provided fields are updated. Omit fields to keep existing values.
    """
    store = await get_streaming_store()
    
    # Check if programmation exists
    existing = await store.get_programmation(programmation_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Programmation '{programmation_id}' not found"
        )
    
    # Filter out None values
    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not update_data:
        return existing
    
    # Validate time range if both times are being updated
    new_start = update_data.get("start_time", existing.start_time)
    new_end = update_data.get("end_time", existing.end_time)
    if new_start >= new_end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be before end_time"
        )
    
    programmation = await store.update_programmation(programmation_id, update_data)
    return programmation


@router.delete("/programmations/{programmation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_programmation(programmation_id: str):
    """
    Delete a programmation.
    
    WARNING: This also deletes all associated playlist entries (cascading delete).
    """
    store = await get_streaming_store()
    deleted = await store.delete_programmation(programmation_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Programmation '{programmation_id}' not found"
        )
    
    return None
