"""
Low-level API: Character Management

Handles CRUD operations for character entities with voice samples and video clips.
Uses MySQL for metadata storage and MinIO for binary files (audio/video).
"""

from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from typing import List

from virtual_streamer.video_server.models import Character, VoiceSample
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.entity_repository import get_entity_repository

# Router setup
router = APIRouter(prefix="/characters", tags=["Characters"])

# Storage prefixes for MinIO
PREFIX_AUDIO = "audios/"
PREFIX_CLIPS = "clips/"
PREFIX_IDENTITY_IMAGES = "identity_images/"


@router.post("", response_model=Character, status_code=status.HTTP_201_CREATED)
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
    storage = get_storage_client()
    repo = get_entity_repository()

    character_id = name

    # Upload voice sample files to MinIO and collect metadata
    voice_samples_data = []
    for vf, tr in zip(voice_files, transcripts):
        file_content = await vf.read()
        storage_key = f"{PREFIX_AUDIO}{vf.filename}"
        await storage.put_object(storage_key, file_content, content_type="audio/wav")
        voice_samples_data.append({"storage_path": storage_key, "transcript": tr})

    # Upload video file to MinIO
    video_path = None
    if video_file:
        file_content = await video_file.read()
        storage_key = f"{PREFIX_CLIPS}{video_file.filename}"
        await storage.put_object(storage_key, file_content, content_type="video/mp4")
        video_path = storage_key

    # Upload identity images to MinIO
    identity_image_paths = []
    for img_file in identity_files:
        file_content = await img_file.read()
        content_type = "image/jpeg"
        if img_file.filename.lower().endswith(".png"):
            content_type = "image/png"
        elif img_file.filename.lower().endswith(".webp"):
            content_type = "image/webp"
        storage_key = f"{PREFIX_IDENTITY_IMAGES}{character_id}/{img_file.filename}"
        await storage.put_object(storage_key, file_content, content_type=content_type)
        identity_image_paths.append(storage_key)

    # Store metadata in MySQL
    character_data = await repo.create_character(
        character_id=character_id,
        name=name,
        description=description,
        video_clip_path=video_path,
        voice_samples=voice_samples_data,
        video_search_tag=video_search_tag,
        identity_images=identity_image_paths,
    )

    # Convert to Pydantic model
    return Character(
        character_id=character_data["character_id"],
        name=character_data["name"],
        description=character_data["description"],
        video_clip_path=character_data["video_clip_path"],
        voice_samples=[
            VoiceSample(
                sample_storage_path=s["sample_storage_path"],
                transcript=s["transcript"],
            )
            for s in character_data["voice_samples"]
        ],
        video_search_tag=character_data.get("video_search_tag"),
        identity_images=character_data.get("identity_images", []),
        created_at=character_data["created_at"],
        updated_at=character_data["updated_at"],
    )


@router.get("/{character_id}", response_model=Character)
async def get_character(character_id: str):
    """Retrieves a specific Character by ID."""
    repo = get_entity_repository()
    character_data = await repo.get_character(character_id)

    if character_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )

    return Character(
        character_id=character_data["character_id"],
        name=character_data["name"],
        description=character_data["description"],
        video_clip_path=character_data["video_clip_path"],
        voice_samples=[
            VoiceSample(
                sample_storage_path=s["sample_storage_path"],
                transcript=s["transcript"],
            )
            for s in character_data["voice_samples"]
        ],
        video_search_tag=character_data.get("video_search_tag"),
        identity_images=character_data.get("identity_images", []),
        created_at=character_data["created_at"],
        updated_at=character_data["updated_at"],
    )


@router.get("", response_model=List[Character])
async def list_characters(limit: int = 100):
    """Lists all Characters with optional limit."""
    repo = get_entity_repository()
    characters_data = await repo.list_characters(limit)

    return [
        Character(
            character_id=c["character_id"],
            name=c["name"],
            description=c["description"],
            video_clip_path=c["video_clip_path"],
            voice_samples=[
                VoiceSample(
                    sample_storage_path=s["sample_storage_path"],
                    transcript=s["transcript"],
                )
                for s in c["voice_samples"]
            ],
            video_search_tag=c.get("video_search_tag"),
            identity_images=c.get("identity_images", []),
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in characters_data
    ]


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(character_id: str):
    """Deletes a Character definition (metadata only, files remain in MinIO)."""
    repo = get_entity_repository()
    deleted = await repo.delete_character(character_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )

    return None
