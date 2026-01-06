"""
Low-level API: Story Template Management

Handles CRUD operations for story templates with associated characters.
Uses MySQL for storage via EntityRepository.
"""

from fastapi import APIRouter, HTTPException, status, Form
from typing import List, Optional

from virtual_streamer.video_server.models import StoryTemplate
from virtual_streamer.utils.entity_repository import get_entity_repository

# Router setup
router = APIRouter(prefix="/story-templates", tags=["Story Templates"])


def _dict_to_model(data: dict) -> StoryTemplate:
    """Convert repository dict to Pydantic model."""
    return StoryTemplate(
        template_id=data["template_id"],
        name=data["name"],
        prompt=data["prompt"],
        collection=data["collection"],
        target_lines=data["target_lines"],
        character_ids=data.get("character_ids", []),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.post("", response_model=StoryTemplate, status_code=status.HTTP_201_CREATED)
async def create_story_template(
    name: str = Form(..., description="Display name for the story template"),
    prompt: str = Form(..., description="Full prompt text for story generation"),
    collection: str = Form(..., description="Qdrant collection name for video search"),
    target_lines: int = Form(6, description="Target number of dialogue lines"),
    character_ids: List[str] = Form(
        default=[],
        description="List of character IDs to associate with this template",
    ),
):
    """
    Creates a new Story Template.
    
    The prompt should contain story-specific instructions (tone, rules, examples).
    Variables {title}, {target_lines}, and {characters} will be auto-injected
    by the meta-prompt at generation time.
    """
    repo = get_entity_repository()

    # Use name as template_id (sanitized)
    template_id = name.lower().replace(" ", "_").replace("-", "_")

    # Verify all characters exist
    for char_id in character_ids:
        char = await repo.get_character(char_id)
        if char is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Character '{char_id}' not found",
            )

    # Create template in database
    template_data = await repo.create_story_template(
        template_id=template_id,
        name=name,
        prompt=prompt,
        collection=collection,
        target_lines=target_lines,
        character_ids=character_ids,
    )

    return _dict_to_model(template_data)


@router.get("/{template_id}", response_model=StoryTemplate)
async def get_story_template(template_id: str):
    """Retrieves a specific Story Template by ID."""
    repo = get_entity_repository()
    template_data = await repo.get_story_template(template_id)

    if template_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story template not found",
        )

    return _dict_to_model(template_data)


@router.get("", response_model=List[StoryTemplate])
async def list_story_templates(limit: int = 100):
    """Lists all Story Templates with optional limit."""
    repo = get_entity_repository()
    templates_data = await repo.list_story_templates(limit)

    return [_dict_to_model(t) for t in templates_data]


@router.put("/{template_id}", response_model=StoryTemplate)
async def update_story_template(
    template_id: str,
    name: Optional[str] = Form(None, description="New display name"),
    prompt: Optional[str] = Form(None, description="New prompt text"),
    collection: Optional[str] = Form(None, description="New Qdrant collection name"),
    target_lines: Optional[int] = Form(None, description="New target line count"),
    character_ids: Optional[List[str]] = Form(
        None,
        description="New list of character IDs (replaces existing)",
    ),
):
    """Updates an existing Story Template."""
    repo = get_entity_repository()

    # Check if template exists
    existing = await repo.get_story_template(template_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story template not found",
        )

    # Verify all new characters exist
    if character_ids is not None:
        for char_id in character_ids:
            char = await repo.get_character(char_id)
            if char is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Character '{char_id}' not found",
                )

    # Update template
    template_data = await repo.update_story_template(
        template_id=template_id,
        name=name,
        prompt=prompt,
        collection=collection,
        target_lines=target_lines,
        character_ids=character_ids,
    )

    return _dict_to_model(template_data)


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_story_template(template_id: str):
    """Deletes a Story Template."""
    repo = get_entity_repository()
    deleted = await repo.delete_story_template(template_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Story template not found",
        )

    return None


@router.post("/admin/drop-tables", status_code=status.HTTP_200_OK)
async def drop_story_template_tables():
    """
    Drops and recreates the story template tables.
    
    WARNING: This deletes all story templates permanently.
    Use this for schema migrations when the table structure has changed.
    
    Returns:
        Message confirming tables were reset
    """
    repo = get_entity_repository()
    await repo.drop_story_template_tables()
    
    return {"message": "Story template tables dropped and recreated with latest schema"}

