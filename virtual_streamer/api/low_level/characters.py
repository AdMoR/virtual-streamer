"""
Low-level API: Character Management

Handles CRUD operations for character entities with voice samples and video clips.
Uses MySQL for metadata storage and MinIO for binary files (audio/video).
"""

import os
import tempfile

from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from typing import List, Optional, Tuple

from virtual_streamer.video_server.models import Character, VoiceSample
from virtual_streamer.image_generation.image_tagger import tag_image
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.entity_repository import get_entity_repository
from virtual_streamer.utils.transcription import transcribe_audio

# Router setup
router = APIRouter(prefix="/characters", tags=["Characters"])

# Storage prefixes for MinIO
PREFIX_AUDIO = "audios/"
PREFIX_CLIPS = "clips/"
PREFIX_IDENTITY_IMAGES = "identity_images/"


async def _tag_identity_images(
    uploads: List[Tuple[bytes, str]],
    character_name: str,
    character_description: Optional[str],
) -> List[dict]:
    """Auto-label uploaded identity images with the image-tagger vision LLM.

    Best effort: a failed tag yields an unlabeled entry (path only) and never
    fails the upload. Returns LabeledImage dicts ready for DB storage.
    """
    labels = []
    for content, storage_key in uploads:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            label = await tag_image(
                tmp_path,
                storage_path=storage_key,
                entity_name=character_name,
                entity_description=character_description,
            )
        finally:
            os.unlink(tmp_path)
        labels.append(label.model_dump(mode="json"))
    return labels


@router.post("", response_model=Character, status_code=status.HTTP_201_CREATED)
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    video_search_tag: str = Form(None),
    voice_files: List[UploadFile] = File(...),
    transcripts: Optional[List[str]] = Form(
        None,
        description="Transcripts for voice files. Omit to auto-transcribe with Whisper.",
    ),
    video_file: Optional[UploadFile] = File(
        None, description="Representative video clip (optional)"
    ),
    identity_files: List[UploadFile] = File(default=[]),
):
    """Creates a new Character definition with voice samples, optional video, and identity images.

    Transcripts are auto-generated with Whisper when not provided.
    """
    storage = get_storage_client()
    repo = get_entity_repository()

    character_id = name

    # Validate transcript count when provided manually
    if transcripts is not None and len(transcripts) != len(voice_files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Number of transcripts ({len(transcripts)}) must match number of voice files ({len(voice_files)})",
        )

    # Upload voice sample files to MinIO and collect metadata
    voice_samples_data = []
    for i, vf in enumerate(voice_files):
        file_content = await vf.read()
        storage_key = f"{PREFIX_AUDIO}{vf.filename}"
        await storage.put_object(storage_key, file_content, content_type="audio/wav")

        # Use provided transcript or auto-transcribe with Whisper
        if transcripts is not None:
            tr = transcripts[i]
        else:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(vf.filename)[1] or ".wav", delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            try:
                tr = transcribe_audio(tmp_path)
            finally:
                os.unlink(tmp_path)

        voice_samples_data.append({"storage_path": storage_key, "transcript": tr})

    # Upload video file to MinIO (optional)
    video_path = None
    if video_file is not None:
        file_content = await video_file.read()
        storage_key = f"{PREFIX_CLIPS}{video_file.filename}"
        await storage.put_object(storage_key, file_content, content_type="video/mp4")
        video_path = storage_key

    # Upload identity images to MinIO
    identity_image_paths = []
    identity_uploads = []
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
        identity_uploads.append((file_content, storage_key))

    labeled_images = await _tag_identity_images(identity_uploads, name, description)

    # Store metadata in MySQL
    character_data = await repo.create_character(
        character_id=character_id,
        name=name,
        description=description,
        video_clip_path=video_path,
        voice_samples=voice_samples_data,
        video_search_tag=video_search_tag,
        identity_images=identity_image_paths,
        labeled_images=labeled_images,
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


@router.put("/{character_id}", response_model=Character)
async def update_character(
    character_id: str,
    name: Optional[str] = Form(None, description="New display name"),
    description: Optional[str] = Form(None, description="New description"),
    video_search_tag: Optional[str] = Form(None, description="New video search tag"),
    voice_files: Optional[List[UploadFile]] = File(
        None, description="New voice sample files (replaces all existing)"
    ),
    transcripts: Optional[List[str]] = Form(
        None, description="Transcripts for new voice files"
    ),
    video_file: Optional[UploadFile] = File(
        None, description="New video file (replaces existing)"
    ),
    identity_files: Optional[List[UploadFile]] = File(
        None, description="New identity images (replaces all existing)"
    ),
):
    """
    Updates an existing Character.
    
    Only provided fields are updated. Omit fields to keep existing values.
    
    Note: If voice_files are provided, transcripts must also be provided and
    will REPLACE all existing voice samples.
    """
    storage = get_storage_client()
    repo = get_entity_repository()

    # Check if character exists
    existing = await repo.get_character(character_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )

    # Prepare update data
    update_kwargs = {}
    
    if name is not None:
        update_kwargs["name"] = name
    if description is not None:
        update_kwargs["description"] = description
    if video_search_tag is not None:
        update_kwargs["video_search_tag"] = video_search_tag

    # Handle voice files update
    if voice_files is not None:
        # Validate transcript count only when provided manually
        if transcripts is not None and len(transcripts) != len(voice_files):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Number of transcripts ({len(transcripts)}) must match number of voice files ({len(voice_files)})",
            )

        voice_samples_data = []
        for i, vf in enumerate(voice_files):
            file_content = await vf.read()
            storage_key = f"{PREFIX_AUDIO}{character_id}/{vf.filename}"
            await storage.put_object(storage_key, file_content, content_type="audio/wav")

            # Use provided transcript or auto-transcribe with Whisper
            if transcripts is not None:
                tr = transcripts[i]
            else:
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(vf.filename)[1] or ".wav", delete=False) as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    tr = transcribe_audio(tmp_path)
                finally:
                    os.unlink(tmp_path)

            voice_samples_data.append({"storage_path": storage_key, "transcript": tr})

        update_kwargs["voice_samples"] = voice_samples_data

    # Handle video file update
    if video_file is not None:
        file_content = await video_file.read()
        storage_key = f"{PREFIX_CLIPS}{character_id}/{video_file.filename}"
        await storage.put_object(storage_key, file_content, content_type="video/mp4")
        update_kwargs["video_clip_path"] = storage_key

    # Handle identity images update
    if identity_files is not None:
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
        
        update_kwargs["identity_images"] = identity_image_paths

    # Perform update
    character_data = await repo.update_character(character_id, **update_kwargs)

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


@router.post("/{character_id}/identity-images", response_model=Character)
async def add_identity_images(
    character_id: str,
    identity_files: List[UploadFile] = File(..., description="Image files to append"),
):
    """Appends new identity images to an existing Character without replacing existing ones."""
    storage = get_storage_client()
    repo = get_entity_repository()

    existing = await repo.get_character(character_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
        )

    existing_images = existing.get("identity_images", [])
    existing_labels = existing.get("labeled_images", [])

    new_paths = []
    new_uploads = []
    for img_file in identity_files:
        file_content = await img_file.read()
        content_type = "image/jpeg"
        if img_file.filename.lower().endswith(".png"):
            content_type = "image/png"
        elif img_file.filename.lower().endswith(".webp"):
            content_type = "image/webp"
        storage_key = f"{PREFIX_IDENTITY_IMAGES}{character_id}/{img_file.filename}"
        await storage.put_object(storage_key, file_content, content_type=content_type)
        new_paths.append(storage_key)
        new_uploads.append((file_content, storage_key))

    new_labels = await _tag_identity_images(
        new_uploads, existing["name"], existing.get("description")
    )

    character_data = await repo.update_character(
        character_id,
        identity_images=existing_images + new_paths,
        labeled_images=existing_labels + new_labels,
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
