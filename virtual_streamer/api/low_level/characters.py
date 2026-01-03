"""
Low-level API: Character Management

Handles CRUD operations for character entities with voice samples and video clips.
"""

from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from typing import List
import uuid
import os
from datetime import datetime

from virtual_streamer.video_server.models import Character, VoiceSample
from virtual_streamer.utils.minio_client import get_storage_client

# Router setup
router = APIRouter(prefix="/characters", tags=["Characters"])

# Storage configuration
PREFIX_AUDIO = "audios/"
PREFIX_CLIPS = "clips/"
PREFIX_CHARACTERS = "characters/"


@router.post("", response_model=Character, status_code=status.HTTP_201_CREATED)
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    voice_files: List[UploadFile] = File(...),
    transcripts: List[str] = Form(...),
    video_file: UploadFile = File(...),
):
    """Creates a new Character definition with voice samples and representative video."""
    storage = get_storage_client()

    character_id = name
    now = datetime.utcnow()

    # Save uploaded voice sample files
    voice_samples_list = []
    for vf, tr in zip(voice_files, transcripts):
        file_content = await vf.read()
        storage_key = f"{PREFIX_AUDIO}{vf.filename}"
        await storage.put_object(storage_key, file_content, content_type="audio/wav")
        voice_samples_list.append(
            VoiceSample(sample_storage_path=storage_key, transcript=tr)
        )

    # Save video file
    video_path = None
    if video_file:
        file_content = await video_file.read()
        storage_key = f"{PREFIX_CLIPS}{video_file.filename}"
        await storage.put_object(storage_key, file_content, content_type="video/mp4")
        video_path = storage_key

    # Create character entity
    character = Character(
        character_id=character_id,
        name=name,
        description=description,
        voice_samples=voice_samples_list,
        video_clip_path=video_path,
        created_at=now,
        updated_at=now,
    )

    # Save to storage
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    await storage.put_json(key, character.model_dump())

    return character


@router.get("/{character_id}", response_model=Character)
async def get_character(character_id: str):
    """Retrieves a specific Character by ID."""
    storage = get_storage_client()
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    data = await storage.get_json(key)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )

    return Character(**data)


@router.get("", response_model=List[Character])
async def list_characters(limit: int = 100):
    """Lists all Characters with optional limit."""
    storage = get_storage_client()
    keys = await storage.list_objects(PREFIX_CHARACTERS)

    characters = []
    count = 0
    for key in keys:
        if key.endswith(".json"):
            data = await storage.get_json(key)
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
    key = f"{PREFIX_CHARACTERS}{character_id}.json"
    await storage.delete_object(key)
    return None
