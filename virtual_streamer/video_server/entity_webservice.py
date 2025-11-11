import boto3
from fastapi import FastAPI, HTTPException, Depends, status, Body, Query, Form, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import os
import json

# Assuming models.py is in the same directory or accessible via PYTHONPATH
from virtual_streamer.video_server.models import (
        VideoClip, VideoClipCreate, VideoClipMetadataInput, VideoClipBase,
        Character, CharacterBase, VoiceSample,DialogueEntry, VideoOptions,
        JobStatus, ValidationStatus, ValidationIssue, ProjectValidationResult,
        CharacterPresence # Ensure all needed models are imported
    )
from virtual_streamer.utils.s3_client import AsyncS3Client
from virtual_streamer.utils.local_fs_client import LocalFSClient

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("ENTITY_S3_BUCKET", "your-entity-bucket-name") # Use a dedicated bucket or prefix
#s3_cli = AsyncS3Client(S3_BUCKET_NAME)
s3_cli = LocalFSClient("/data")

S3_PREFIX_CLIPS = "clips/"
S3_PREFIX_AUDIO = "audios/"
S3_PREFIX_CHARACTERS = "characters/"
S3_PREFIX_PROJECTS = "projects/"


# --- FastAPI App ---
app = FastAPI(
    title="Entity Management Service",
    description="API for managing Video Clips, Collections, Characters, and Projects using S3 backend.",
    version="0.1.0"
)

# --- Health Check ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check."""
    # Could add S3 connectivity check here if needed
    return {"status": "healthy", "s3_bucket": S3_BUCKET_NAME}

# --- Video Clip Endpoints ---

@app.post("/clips", response_model=VideoClip, status_code=status.HTTP_201_CREATED, tags=["Video Clips"])
async def create_video_clip(clip_data: VideoClipCreate):
    """Creates a new Video Clip record."""
    clip_id = str(uuid.uuid4())
    now = datetime.utcnow()
    clip = VideoClip(
        clip_id=clip_id,
        storage_path=clip_data.storage_path,
        collection_ids=clip_data.collection_ids,
        metadata=None, # Metadata added via PUT endpoint
        created_at=now,
        updated_at=now
    )
    s3_key = f"{S3_PREFIX_CLIPS}{clip_id}.json"
    await s3_cli.s3_put_json(s3_key, clip.dict())
    # Potential: Update collections if clip_data.collection_ids is not empty
    return clip

@app.get("/clips/{clip_id}", response_model=VideoClip, tags=["Video Clips"])
async def get_video_clip(clip_id: int):
    """Retrieves a specific Video Clip by its ID."""
    s3_key = f"{S3_PREFIX_CLIPS}{clip_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found")
    return VideoClip(**data)

@app.put("/clips/{clip_id}/metadata", response_model=VideoClip, tags=["Video Clips"])
async def update_video_clip_metadata(clip_id: str, metadata: VideoClipMetadataInput):
    """Adds or replaces the metadata for a specific Video Clip."""
    s3_key = f"{S3_PREFIX_CLIPS}{clip_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Clip not found")

    clip = VideoClip(**data)
    clip.metadata = metadata
    clip.updated_at = datetime.utcnow()

    await s3_cli.s3_put_json(s3_key, clip.dict())
    return clip

@app.get("/clips", response_model=List[VideoClip], tags=["Video Clips"])
async def list_video_clips(limit: int = Query(100, ge=1, le=1000), prefix: Optional[str] = None):
    """Lists Video Clips (metadata only). Limited results."""
    # Note: This lists keys and fetches each JSON individually. Can be slow/costly.
    # Consider alternative listing/indexing for large scale.
    target_prefix = f"{S3_PREFIX_CLIPS}{prefix if prefix else ''}"
    keys = await s3_cli.s3_list_keys(target_prefix)
    clips = []
    count = 0
    for key in keys:
        if key.endswith('.json'): # Basic check
             # Optimization: Could use list_objects_v2 metadata if sufficient
            data = await s3_cli.s3_get_json(key)
            if data:
                clips.append(VideoClip(**data))
                count += 1
                if count >= limit:
                    break
    return clips

@app.delete("/clips/{clip_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Video Clips"])
async def delete_video_clip(clip_id: str):
    """Deletes the metadata record of a Video Clip. Does NOT delete the video file."""
    s3_key = f"{S3_PREFIX_CLIPS}/{clip_id}.json"
    # Check if exists first? Optional, delete is idempotent.
    await s3_cli.s3_delete_object(s3_key)
    # Potential: Remove clip_id from associated collections
    return None


# --- Character Endpoints ---

@app.post("/characters", response_model=Character, status_code=status.HTTP_201_CREATED, tags=["Characters"])
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    voice_files: List[UploadFile] = File(...),
    transcripts: List[str] = Form(...),
    tts_model_config: Optional[str] = Form(None),
    video_file: UploadFile = File(...)
):
    """Creates a new Character definition with optional representative video upload."""
    character_id = name
    now = datetime.utcnow()
    # Save uploaded voice sample files and build VoiceSample objects
    voice_samples_list = []
    for vf, tr in zip(voice_files, transcripts):
        if not os.path.exists(vf.filename):
            with open(vf.filename, 'wb') as file:
                file.write(await vf.read())
        print(">>>>> ", len(vf.file.read()), vf.filename, os.listdir())
        s3_path = await s3_cli.s3_put_file(vf.filename, s3_prefix=S3_PREFIX_AUDIO)
        voice_samples_list.append(VoiceSample(sample_storage_path=s3_path, transcript=tr))

    video_path = None
    if video_file:
        if not os.path.exists(video_file.filename):
            with open(video_file.filename, 'wb') as file:
                file.write(await video_file.read())
        video_path = await s3_cli.s3_put_file(video_file.filename, s3_prefix=S3_PREFIX_CLIPS)

    character = Character(
        character_id=character_id,
        name=name,
        description=description,
        voice_samples=voice_samples_list,
        tts_model_config=None,
        video_clip_path=video_path,
        created_at=now,
        updated_at=now
    )
    s3_key = os.path.join(S3_PREFIX_CHARACTERS, f"{character_id}.json")
    await s3_cli.s3_put_json(s3_key, character.model_dump())
    return character

@app.get("/characters/{character_id}", response_model=Character, tags=["Characters"])
async def get_character(character_id: str):
    """Retrieves a specific Character by ID."""
    s3_key = f"{S3_PREFIX_CHARACTERS}/{character_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")
    return Character(**data)

@app.get("/characters", response_model=List[Character], tags=["Characters"])
async def list_characters(limit: int = Query(100, ge=1, le=1000)):
    """Lists Characters. Limited results."""
    keys = await s3_cli.s3_list_keys(S3_PREFIX_CHARACTERS)
    characters = []
    count = 0
    for key in keys:
        if key.endswith('.json'):
            data = await s3_cli.s3_get_json(key)
            if data:
                print(data)
                data["video_clip_path"] = data.get("video_clip_path", "")
                characters.append(Character(**data))
                count += 1
                if count >= limit:
                    break
    return characters

@app.delete("/characters/{character_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Characters"])
async def delete_character(character_id: str):
    """Deletes a Character definition."""
    s3_key = f"{S3_PREFIX_CHARACTERS}/{character_id}.json"
    await s3_cli.s3_delete_object(s3_key)
    return None
