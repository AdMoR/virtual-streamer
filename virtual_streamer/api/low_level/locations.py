"""
Low-level API: Location Management

Handles CRUD operations for locations scoped to story templates.
Uses MySQL for storage via EntityRepository.
"""

from fastapi import APIRouter, HTTPException, status, Form
from typing import List, Optional

from virtual_streamer.video_server.models import Location
from virtual_streamer.utils.entity_repository import get_entity_repository

router = APIRouter(prefix="/locations", tags=["Locations"])


def _dict_to_model(data: dict) -> Location:
    """Convert repository dict to Pydantic model."""
    return Location(
        location_id=data["location_id"],
        name=data["name"],
        description=data["description"],
        story_template_id=data["story_template_id"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.post("", response_model=Location, status_code=status.HTTP_201_CREATED)
async def create_location(
    name: str = Form(..., description="Display name for the location (e.g. 'Medieval Castle')"),
    description: str = Form(..., description="Diffusion-model prompt for this location"),
    story_template_id: str = Form(..., description="ID of the story template this location belongs to"),
):
    """
    Creates a new Location manually (without the agent pipeline).

    The `location_id` is derived from the name by lowercasing and replacing
    spaces with hyphens (e.g. 'Medieval Castle' → 'medieval-castle').

    Use `POST /location-generation/generate` to create a location with an
    AI-generated description based on the story template context.
    """
    repo = get_entity_repository()

    location_id = name.lower().replace(" ", "-")

    # Verify the story template exists
    template = await repo.get_story_template(story_template_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Story template '{story_template_id}' not found",
        )

    # Check for duplicate
    existing = await repo.get_location(location_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Location '{location_id}' already exists",
        )

    location_data = await repo.create_location(
        location_id=location_id,
        name=name,
        description=description,
        story_template_id=story_template_id,
    )

    return _dict_to_model(location_data)


@router.get("/{location_id}", response_model=Location)
async def get_location(location_id: str):
    """Retrieves a specific Location by ID."""
    repo = get_entity_repository()
    location_data = await repo.get_location(location_id)

    if location_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Location not found",
        )

    return _dict_to_model(location_data)


@router.get("", response_model=List[Location])
async def list_locations(
    story_template_id: Optional[str] = None,
    limit: int = 100,
):
    """
    Lists Locations.

    When `story_template_id` is provided, returns only locations for that template.
    Otherwise returns all locations across all templates.
    """
    repo = get_entity_repository()

    if story_template_id:
        locations_data = await repo.list_locations_by_template(story_template_id, limit)
    else:
        locations_data = await repo.list_all_locations(limit)

    return [_dict_to_model(loc) for loc in locations_data]


@router.delete("/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_location(location_id: str):
    """Deletes a Location."""
    repo = get_entity_repository()
    deleted = await repo.delete_location(location_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Location not found",
        )

    return None