from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    Body,
    Query,
    Form,
    File,
    UploadFile,
    Response,
)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import os
import json

# Assuming models.py is in the same directory or accessible via PYTHONPATH
from virtual_streamer.video_server.models import (
    VideoClip,
    VideoClipCreate,
    VideoClipMetadataInput,
    VideoClipBase,
    Character,
    CharacterBase,
    VoiceSample,
    DialogueEntry,
    VideoOptions,
    JobStatus,
    ValidationStatus,
    ValidationIssue,
    ProjectValidationResult,
    CharacterPresence,  # Ensure all needed models are imported
)
from virtual_streamer.utils.minio_client import get_storage_client, MinIOClient


# --- Configuration ---
storage_client = get_storage_client()

PREFIX_CLIPS = "clips/"
PREFIX_AUDIO = "audios/"
PREFIX_CHARACTERS = "characters/"
PREFIX_PROJECTS = "projects/"
PREFIX_IDENTITY_IMAGES = "identity_images/"


# --- FastAPI App ---
app = FastAPI(
    title="Entity Management Service",
    description="API for managing Video Clips, Collections, Characters, and Projects using MinIO storage.",
    version="0.2.0",
)


# --- Health Check ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "storage": "minio"}


# --- File Retrieval Endpoints ---


@app.get("/files/{path:path}", tags=["Files"])
async def get_file(path: str):
    """
    Stream file from MinIO storage.
    
    Args:
        path: Storage key path (e.g., "clips/video.mp4" or "audios/sample.wav")
        
    Returns:
        File content with appropriate content type
    """
    data = await storage_client.get_object(path)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {path}"
        )
    content_type = MinIOClient._guess_content_type(path) or "application/octet-stream"
    return Response(content=data, media_type=content_type)


@app.get("/files/{path:path}/url", tags=["Files"])
async def get_file_url(path: str, expiry: int = Query(3600, ge=60, le=86400)):
    """
    Get a presigned URL for direct MinIO access.
    
    Args:
        path: Storage key path (e.g., "clips/video.mp4")
        expiry: URL expiry time in seconds (default: 3600, max: 86400)
        
    Returns:
        Object with presigned URL
    """
    if not await storage_client.object_exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {path}"
        )
    url = storage_client.get_url(path)
    return {"path": path, "url": url, "expiry_seconds": expiry}


# --- Video Clip Endpoints ---


@app.post(
    "/clips",
    response_model=VideoClip,
    status_code=status.HTTP_201_CREATED,
    tags=["Video Clips"],
)
async def create_video_clip(clip_data: VideoClipCreate):
    """Creates a new Video Clip record."""
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
    await storage_client.put_json(key, clip.dict())
    # Potential: Update collections if clip_data.collection_ids is not empty
    return clip


@app.get("/clips/{clip_id}", response_model=VideoClip, tags=["Video Clips"])
async def get_video_clip(clip_id: int):
    """Retrieves a specific Video Clip by its ID."""
    key = f"{PREFIX_CLIPS}{clip_id}.json"
    data = await storage_client.get_json(key)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )
    return VideoClip(**data)


@app.put("/clips/{clip_id}/metadata", response_model=VideoClip, tags=["Video Clips"])
async def update_video_clip_metadata(clip_id: str, metadata: VideoClipMetadataInput):
    """Adds or replaces the metadata for a specific Video Clip."""
    key = f"{PREFIX_CLIPS}{clip_id}.json"
    data = await storage_client.get_json(key)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found"
        )

    clip = VideoClip(**data)
    clip.metadata = metadata
    clip.updated_at = datetime.utcnow()

    await storage_client.put_json(key, clip.dict())
    return clip


@app.get("/clips", response_model=List[VideoClip], tags=["Video Clips"])
async def list_video_clips(
    limit: int = Query(100, ge=1, le=1000), prefix: Optional[str] = None
):
    """Lists Video Clips (metadata only). Limited results."""
    # Note: This lists keys and fetches each JSON individually. Can be slow/costly.
    # Consider alternative listing/indexing for large scale.
    target_prefix = f"{PREFIX_CLIPS}{prefix if prefix else ''}"
    keys = await storage_client.list_objects(target_prefix)
    clips = []
    count = 0
    for key in keys:
        if key.endswith(".json"):  # Basic check
            # Optimization: Could use list_objects_v2 metadata if sufficient
            data = await storage_client.get_json(key)
            if data:
                clips.append(VideoClip(**data))
                count += 1
                if count >= limit:
                    break
    return clips


@app.delete(
    "/clips/{clip_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Video Clips"]
)
async def delete_video_clip(clip_id: str):
    """Deletes the metadata record of a Video Clip. Does NOT delete the video file."""
    key = f"{PREFIX_CLIPS}/{clip_id}.json"
    # Check if exists first? Optional, delete is idempotent.
    await storage_client.delete_object(key)
    # Potential: Remove clip_id from associated collections
    return None


# --- Character Endpoints ---


@app.post(
    "/characters",
    response_model=Character,
    status_code=status.HTTP_201_CREATED,
    tags=["Characters"],
)
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    video_search_tag: str = Form(None),
    voice_files: List[UploadFile] = File(...),
    transcripts: List[str] = Form(...),
    video_file: UploadFile = File(...),
    identity_files: List[UploadFile] = File(default=[]),
):
    """Creates a new Character definition with voice samples, video, and identity images."""
    character_id = name
    now = datetime.utcnow()
    # Save uploaded voice sample files and build VoiceSample objects
    voice_samples_list = []
    for vf, tr in zip(voice_files, transcripts):
        # Read file content and upload to storage
        file_content = await vf.read()
        storage_key = f"{PREFIX_AUDIO}{vf.filename}"
        await storage_client.put_object(storage_key, file_content, content_type="audio/wav")
        voice_samples_list.append(
            VoiceSample(sample_storage_path=storage_key, transcript=tr)
        )

    video_path = None
    if video_file:
        # Read file content and upload to storage
        file_content = await video_file.read()
        storage_key = f"{PREFIX_CLIPS}{video_file.filename}"
        await storage_client.put_object(storage_key, file_content, content_type="video/mp4")
        video_path = storage_key

    # Save uploaded identity images
    identity_image_paths = []
    for img_file in identity_files:
        file_content = await img_file.read()
        # Determine content type from filename
        content_type = "image/jpeg"
        if img_file.filename.lower().endswith(".png"):
            content_type = "image/png"
        elif img_file.filename.lower().endswith(".webp"):
            content_type = "image/webp"
        storage_key = f"{PREFIX_IDENTITY_IMAGES}{character_id}/{img_file.filename}"
        await storage_client.put_object(storage_key, file_content, content_type=content_type)
        identity_image_paths.append(storage_key)

    character = Character(
        character_id=character_id,
        name=name,
        description=description,
        voice_samples=voice_samples_list,
        video_clip_path=video_path,
        video_search_tag=video_search_tag,
        identity_images=identity_image_paths,
        created_at=now,
        updated_at=now,
    )
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    await storage_client.put_json(key, character.model_dump())
    return character


@app.get("/characters/{character_id}", response_model=Character, tags=["Characters"])
async def get_character(character_id: str):
    """Retrieves a specific Character by ID."""
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    data = await storage_client.get_json(key)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )
    return Character(**data)


@app.get("/characters", response_model=List[Character], tags=["Characters"])
async def list_characters(limit: int = Query(100, ge=1, le=1000)):
    """Lists Characters. Limited results."""
    keys = await storage_client.list_objects(PREFIX_CHARACTERS)
    characters = []
    count = 0
    for key in keys:
        if key.endswith(".json"):
            data = await storage_client.get_json(key)
            if data:
                print(data)
                data["video_clip_path"] = data.get("video_clip_path", "")
                characters.append(Character(**data))
                count += 1
                if count >= limit:
                    break
    return characters


@app.delete(
    "/characters/{character_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Characters"],
)
async def delete_character(character_id: str):
    """Deletes a Character definition."""
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    await storage_client.delete_object(key)
    return None
