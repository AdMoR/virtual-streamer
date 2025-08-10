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
        Collection, CollectionCreate, CollectionUpdate, CollectionBase,
        Character, CharacterCreate, CharacterBase, VoiceSample,
        VideoProject, VideoProjectCreate, VideoProjectUpdate, VideoProjectBase,
        Scene, SceneCreate, SceneBase, DialogueEntry, VideoOptions,
        JobStatus, ValidationStatus, ValidationIssue, ProjectValidationResult,
        CharacterPresence # Ensure all needed models are imported
    )
from virtual_streamer.utils.s3_client import AsyncS3Client
from virtual_streamer.utils.local_fs_client import LocalFSClient
import aiofiles


# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("ENTITY_S3_BUCKET", "your-entity-bucket-name") # Use a dedicated bucket or prefix
#s3_cli = AsyncS3Client(S3_BUCKET_NAME)
s3_cli = LocalFSClient("/data")

S3_PREFIX_CLIPS = "clips/"
S3_PREFIX_COLLECTIONS = "collections/"
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
async def get_video_clip(clip_id: str):
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

# --- Collection Endpoints ---

@app.post("/collections", response_model=Collection, status_code=status.HTTP_201_CREATED, tags=["Collections"])
async def create_collection(collection_data: CollectionCreate):
    """Creates a new Collection."""
    collection_id = str(uuid.uuid4())
    now = datetime.utcnow()
    collection = Collection(
        collection_id=collection_id,
        name=collection_data.name,
        description=collection_data.description,
        metadata=collection_data.metadata,
        clip_ids=collection_data.clip_ids, # Validate these clip IDs exist?
        created_at=now,
        updated_at=now
    )
    s3_key = f"{S3_PREFIX_COLLECTIONS}/{collection_id}.json"
    await s3_cli.s3_put_json(s3_key, collection.dict())
    # Potential: Update clips if collection_data.clip_ids is not empty
    return collection

@app.get("/collections/{collection_id}", response_model=Collection, tags=["Collections"])
async def get_collection(collection_id: str):
    """Retrieves a specific Collection by its ID."""
    s3_key = f"{S3_PREFIX_COLLECTIONS}/{collection_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return Collection(**data)

@app.patch("/collections/{collection_id}", response_model=Collection, tags=["Collections"])
async def update_collection(collection_id: str, update_data: CollectionUpdate):
    """Updates specific fields of a Collection."""
    s3_key = f"{S3_PREFIX_COLLECTIONS}/{collection_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")

    collection = Collection(**data)
    update_dict = update_data.dict(exclude_unset=True) # Get only fields that were provided
    updated_collection = collection.copy(update=update_dict)
    updated_collection.updated_at = datetime.utcnow()

    # Handle potential changes in clip_ids (add/remove from clips?) - Complex interaction
    # For simplicity, this PATCH only updates the collection's own record.
    # Managing relationships might need dedicated endpoints or background tasks.

    await s3_cli.s3_put_json(s3_key, updated_collection.dict())
    return updated_collection

@app.get("/collections", response_model=List[Collection], tags=["Collections"])
async def list_collections(limit: int = Query(100, ge=1, le=1000)):
    """Lists Collections. Limited results."""
    keys = await s3_cli.s3_list_keys(S3_PREFIX_COLLECTIONS)
    collections = []
    count = 0
    for key in keys:
         if key.endswith('.json'):
            data = await s3_cli.s3_get_json(key)
            if data:
                collections.append(Collection(**data))
                count += 1
                if count >= limit:
                    break
    return collections

@app.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Collections"])
async def delete_collection(collection_id: str):
    """Deletes a Collection metadata record."""
    s3_key = f"{S3_PREFIX_COLLECTIONS}/{collection_id}.json"
    await s3_cli.s3_delete_object(s3_key)
    # Potential: Remove this collection_id from associated clips
    return None

# --- Character Endpoints ---

@app.post("/characters", response_model=Character, status_code=status.HTTP_201_CREATED, tags=["Characters"])
async def create_character(
    name: str = Form(...),
    description: str = Form(None),
    voice_samples: str = Form(...),
    tts_model_config: Optional[str] = Form(None),
    video_file: UploadFile = File(None)
):
    """Creates a new Character definition with optional representative video upload."""
    character_id = name
    now = datetime.utcnow()
    # Parse voice_samples JSON string into list of VoiceSample objects
    voice_samples_list = [VoiceSample(**vs) for vs in json.loads(voice_samples)]
    tts_config = json.loads(tts_model_config) if tts_model_config else None

    video_path = None
    if video_file:
        ext = os.path.splitext(video_file.filename)[1]
        video_key = f"{S3_PREFIX_CHARACTERS}{character_id}{ext}"
        full_path = s3_cli._get_full_path(video_key)
        async with aiofiles.open(full_path, "wb") as out_file:
            content = await video_file.read()
            await out_file.write(content)
        video_path = video_key

    character = Character(
        character_id=character_id,
        name=name,
        description=description,
        voice_samples=voice_samples_list,
        tts_model_config=tts_config,
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


# --- Video Project Endpoints ---

@app.post("/projects", response_model=VideoProject, status_code=status.HTTP_201_CREATED, tags=["Video Projects"])
async def create_video_project(project_data: VideoProjectCreate):
    """Creates a new Video Project."""
    project_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Process initial scenes, giving them IDs
    processed_scenes = []
    for i, scene_create_data in enumerate(project_data.scenes):
        scene = Scene(
            scene_id=str(uuid.uuid4()),
            order=scene_create_data.order if scene_create_data.order is not None else i, # Assign order if missing
            video_clip_id=scene_create_data.video_clip_id, # Validate clip exists?
            dialogue=scene_create_data.dialogue, # Validate dialogue entries?
            options=scene_create_data.options,
            validation_status=ValidationStatus.PENDING,
            validation_issues=[],
            generated_audio_paths={},
            lipsynced_video_path=None
        )
        processed_scenes.append(scene)

    # Sort scenes by order just in case
    processed_scenes.sort(key=lambda s: s.order)

    project = VideoProject(
        project_id=project_id,
        user_id=project_data.user_id,
        title=project_data.title,
        description=project_data.description,
        scenes=processed_scenes,
        status=JobStatus.PENDING,
        final_video_path=None,
        last_validation_result=None,
        created_at=now,
        updated_at=now
    )
    s3_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    await s3_cli.s3_put_json(s3_key, project.dict())
    return project

@app.get("/projects/{project_id}", response_model=VideoProject, tags=["Video Projects"])
async def get_video_project(project_id: str):
    """Retrieves a specific Video Project by ID."""
    s3_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Project not found")
    # Ensure scenes are loaded correctly if nested
    project = VideoProject(**data)
    # Re-sort scenes just to be safe, although they should be stored sorted
    project.scenes.sort(key=lambda s: s.order)
    return project

@app.patch("/projects/{project_id}", response_model=VideoProject, tags=["Video Projects"])
async def update_video_project(project_id: str, update_data: VideoProjectUpdate):
    """Updates basic properties (title, description, user_id) of a Video Project. Does not modify scenes."""
    s3_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    data = await s3_cli.s3_get_json(s3_key)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Project not found")

    project = VideoProject(**data)
    update_dict = update_data.dict(exclude_unset=True)
    updated_project = project.copy(update=update_dict)
    updated_project.updated_at = datetime.utcnow()

    await s3_cli.s3_put_json(s3_key, updated_project.dict())
    # Re-sort scenes just to be safe
    updated_project.scenes.sort(key=lambda s: s.order)
    return updated_project

@app.get("/projects", response_model=List[VideoProject], tags=["Video Projects"])
async def list_video_projects(limit: int = Query(100, ge=1, le=1000)):
    """Lists Video Projects. Limited results."""
    keys = await s3_cli.s3_list_keys(S3_PREFIX_PROJECTS)
    projects = []
    count = 0
    for key in keys:
        if key.endswith('.json'):
            data = await s3_cli.s3_get_json(key)
            if data:
                project = VideoProject(**data)
                # Re-sort scenes just to be safe
                project.scenes.sort(key=lambda s: s.order)
                projects.append(project)
                count += 1
                if count >= limit:
                    break
    return projects

@app.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Video Projects"])
async def delete_video_project(project_id: str):
    """Deletes a Video Project metadata record."""
    s3_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    await s3_cli.s3_delete_object(s3_key)
    return None

# --- Scene Endpoints (within Projects) ---

@app.post("/projects/{project_id}/scenes", response_model=Scene, status_code=status.HTTP_201_CREATED, tags=["Scenes"])
async def add_scene_to_project(project_id: str, scene_data: SceneCreate):
    """Adds a new Scene to an existing Video Project."""
    s3_project_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    project_data = await s3_cli.s3_get_json(s3_project_key)
    if project_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Project not found")

    project = VideoProject(**project_data)

    # Determine order if not provided
    order = scene_data.order
    if order is None:
        order = len(project.scenes) # Append to the end

    scene = Scene(
        scene_id=str(uuid.uuid4()),
        order=order,
        video_clip_id=scene_data.video_clip_id, # Validate clip exists?
        dialogue=scene_data.dialogue, # Validate dialogue entries?
        options=scene_data.options,
        validation_status=ValidationStatus.PENDING,
        validation_issues=[],
        generated_audio_paths={},
        lipsynced_video_path=None
    )

    # Insert scene and re-sort
    project.scenes.append(scene)
    project.scenes.sort(key=lambda s: s.order)
    # Re-assign order based on sorted position to ensure consistency
    for i, s in enumerate(project.scenes):
        s.order = i

    project.updated_at = datetime.utcnow()
    await s3_cli.s3_put_json(s3_project_key, project.dict())

    return scene # Return the newly created scene

@app.get("/projects/{project_id}/scenes", response_model=List[Scene], tags=["Scenes"])
async def list_project_scenes(project_id: str):
    """Lists all Scenes for a specific Video Project, ordered."""
    project = await get_video_project(project_id) # Reuses the project getter
    return project.scenes # Already sorted by get_video_project

@app.get("/projects/{project_id}/scenes/{scene_id}", response_model=Scene, tags=["Scenes"])
async def get_project_scene(project_id: str, scene_id: str):
    """Retrieves a specific Scene from a Video Project."""
    project = await get_video_project(project_id)
    for scene in project.scenes:
        if scene.scene_id == scene_id:
            return scene
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found in project")

@app.put("/projects/{project_id}/scenes/{scene_id}", response_model=Scene, tags=["Scenes"])
async def update_project_scene(project_id: str, scene_id: str, scene_update_data: SceneBase):
    """Updates an existing Scene within a Video Project."""
    s3_project_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    project_data = await s3_cli.s3_get_json(s3_project_key)
    if project_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Project not found")

    project = VideoProject(**project_data)
    scene_index = -1
    for i, scene in enumerate(project.scenes):
        if scene.scene_id == scene_id:
            scene_index = i
            break

    if scene_index == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found in project")

    # Update the scene - retain existing state fields
    existing_scene = project.scenes[scene_index]
    updated_scene = Scene(
        **scene_update_data.dict(), # Base fields from request
        scene_id=existing_scene.scene_id, # Keep original ID
        # Keep existing state unless explicitly changed by another process
        validation_status=existing_scene.validation_status,
        validation_issues=existing_scene.validation_issues,
        generated_audio_paths=existing_scene.generated_audio_paths,
        lipsynced_video_path=existing_scene.lipsynced_video_path
    )
    project.scenes[scene_index] = updated_scene

    # Re-sort if order might have changed
    project.scenes.sort(key=lambda s: s.order)
    # Re-assign order based on sorted position
    for i, s in enumerate(project.scenes):
        s.order = i

    project.updated_at = datetime.utcnow()
    await s3_cli.s3_put_json(s3_project_key, project.dict())

    return updated_scene

@app.delete("/projects/{project_id}/scenes/{scene_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Scenes"])
async def delete_project_scene(project_id: str, scene_id: str):
    """Deletes a Scene from a Video Project."""
    s3_project_key = f"{S3_PREFIX_PROJECTS}/{project_id}.json"
    project_data = await s3_cli.s3_get_json(s3_project_key)
    if project_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video Project not found")

    project = VideoProject(**project_data)
    initial_length = len(project.scenes)
    project.scenes = [scene for scene in project.scenes if scene.scene_id != scene_id]

    if len(project.scenes) == initial_length:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found in project")

    # Re-sort and re-order remaining scenes
    project.scenes.sort(key=lambda s: s.order)
    for i, s in enumerate(project.scenes):
        s.order = i

    project.updated_at = datetime.utcnow()
    await s3_cli.s3_put_json(s3_project_key, project.dict())

    return None

# --- Optional: Add entrypoint for running with uvicorn directly ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) # Example port
