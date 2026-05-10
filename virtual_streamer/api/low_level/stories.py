"""
Low-level API: Story and Scene read-only access.

Provides read-only endpoints for analytics and replay:
  GET /stories                              — list stories
  GET /stories/{story_id}                   — get a single story
  GET /stories/{story_id}/scenes            — list scenes for a story
  GET /stories/{story_id}/scenes/{scene_id}/artifacts — conditioning image artifacts
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status

from virtual_streamer.utils.story_repository import get_story_repository

router = APIRouter(prefix="/stories", tags=["Stories"])


@router.get("", response_model=List[dict])
async def list_stories(
    story_template_id: Optional[str] = None,
    limit: int = 50,
):
    """List generated stories, optionally filtered by template."""
    repo = get_story_repository()
    return await repo.list_stories(story_template_id=story_template_id, limit=limit)


@router.get("/{story_id}", response_model=dict)
async def get_story(story_id: str):
    """Get a single story by ID, including its raw agent output for replay."""
    repo = get_story_repository()
    story = await repo.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")
    return story


@router.get("/{story_id}/scenes", response_model=List[dict])
async def list_scenes(story_id: str):
    """List all scenes for a story, ordered by scene_index."""
    repo = get_story_repository()
    story = await repo.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")
    return await repo.list_scenes_for_story(story_id)


@router.get("/{story_id}/scenes/{scene_id}/artifacts", response_model=List[dict])
async def get_scene_artifacts(story_id: str, scene_id: str):
    """Get conditioning image artifacts for a scene."""
    repo = get_story_repository()
    scene = await repo.get_scene(scene_id)
    if scene is None or scene["story_id"] != story_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    return await repo.get_artifacts_for_scene(scene_id)
