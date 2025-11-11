"""
Low-level API: Character Management

Handles CRUD operations for character entities with voice samples and video clips.
"""

from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from typing import List, Optional
import uuid
import os
from datetime import datetime

from virtual_streamer.video_server.models import Character, VoiceSample
from virtual_streamer.utils.local_fs_client import LocalFSClient

# Router setup
router = APIRouter(prefix="/characters", tags=["Characters"])

# Storage configuration
S3_PREFIX_AUDIO = "audios/"
S3_PREFIX_CLIPS = "clips/"
S3_PREFIX_CHARACTERS = "characters/"


def get_storage_client() -> LocalFSClient:
    """Dependency to get storage client."""
    data_dir = os.environ.get("DATA_DIR", "/data")
    return LocalFSClient(data_dir)


@router.post("", response_model=Character, status_code=status.HTTP_201_CREATED)
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    voice_files: List[UploadFile] = File(...),
    transcripts: List[str] = Form(...),
    tts_model_config: Optional[str] = Form(None),
    video_file: UploadFile = File(...)
):
    """Creates a new Character definition with voice samples and representative video."""
    storage = get_storage_client()
    
    character_id = name
    now = datetime.utcnow()
    
    # Save uploaded voice sample files
    voice_samples_list = []
    for vf, tr in zip(voice_files, transcripts):
        if not os.path.exists(vf.filename):
            with open(vf.filename, 'wb') as file:
                file.write(await vf.read())
        
        s3_path = await storage.s3_put_file(vf.filename, s3_prefix=S3_PREFIX_AUDIO)
        voice_samples_list.append(VoiceSample(
            sample_storage_path=s3_path,
            transcript=tr
        ))
    
    # Save video file
    video_path = None
    if video_file:
        if not os.path.exists(video_file.filename):
            with open(video_file.filename, 'wb') as file:
                file.write(await video_file.read())
        video_path = await storage.s3_put_file(video_file.filename, s3_prefix=S3_PREFIX_CLIPS)
    
    # Create character entity
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
    
    # Save to storage
    s3_key = os.path.join(S3_PREFIX_CHARACTERS, f"{character_id}.json")
    await storage.s3_put_json(s3_key, character.model_dump())
    
    return character


@router.get("/{character_id}", response_model=Character)
async def get_character(character_id: str):
    """Retrieves a specific Character by ID."""
    storage = get_storage_client()
    s3_key = f"{S3_PREFIX_CHARACTERS}/{character_id}.json"
    data = await storage.s3_get_json(s3_key)
    
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    return Character(**data)


@router.get("", response_model=List[Character])
async def list_characters(limit: int = 100):
    """Lists all Characters with optional limit."""
    storage = get_storage_client()
    keys = await storage.s3_list_keys(S3_PREFIX_CHARACTERS)
    
    characters = []
    count = 0
    for key in keys:
        if key.endswith('.json'):
            data = await storage.s3_get_json(key)
            if data:
                # Ensure backward compatibility
                data["video_clip_path"] = data.get("video_clip_path", "")
                characters.append(Character(**data))
                count += 1
                if count >= limit:
                    break
    
    return characters


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(character_id: str):
    """Deletes a Character definition."""
    storage = get_storage_client()
    s3_key = f"{S3_PREFIX_CHARACTERS}/{character_id}.json"
    await storage.s3_delete_object(s3_key)
    return None

