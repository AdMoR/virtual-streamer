"""
Low-level API: Segment candidates and judge feedback.

Every generated take of a scene (seed hunting) is a candidate. These endpoints
make the judge's decisions observable, allow a human to override the selected
take, and collect human preference labels to improve the judge.

  GET  /stories/{story_id}/scenes/{scene_id}/candidates — all takes + verdicts
  GET  /candidates/{candidate_id}                        — one candidate
  POST /candidates/{candidate_id}/select                 — human override of selection
  POST /candidates/{candidate_id}/feedback               — human preference label
  GET  /judge-feedback/export                            — labelled data for judge training
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from virtual_streamer.utils.story_repository import get_story_repository

router = APIRouter(tags=["Candidates"])

logger = logging.getLogger(__name__)


def _with_video_url(candidate: dict) -> dict:
    """Attach a ready-to-play presigned URL so consumers never handle raw MinIO keys."""
    if candidate.get("video_key"):
        try:
            from virtual_streamer.utils.minio_client import get_storage_client
            candidate["video_url"] = get_storage_client().get_url(candidate["video_key"])
        except Exception as exc:
            logger.warning(f"presign failed for {candidate.get('video_key')}: {exc}")
            candidate["video_url"] = None
    else:
        candidate["video_url"] = None
    return candidate


class SelectRequest(BaseModel):
    user: str = Field(default="anonymous", description="Who made the override")
    comment: Optional[str] = None


class FeedbackRequest(BaseModel):
    user: str = Field(default="anonymous")
    human_passed: Optional[bool] = Field(
        default=None, description="Human verdict: is this take usable?"
    )
    human_score: Optional[float] = Field(default=None, ge=0, le=10)
    artifact_tags: List[str] = Field(
        default_factory=list,
        description="Artifact categories the human saw (same vocabulary as the judge)",
    )
    comment: Optional[str] = None


@router.get("/stories/{story_id}/scenes/{scene_id}/candidates", response_model=List[dict])
async def list_candidates(story_id: str, scene_id: str):
    """List all generated takes for a scene, with judge verdicts and selection state."""
    repo = get_story_repository()
    scene = await repo.get_scene(scene_id)
    if scene is None or scene["story_id"] != story_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    return [_with_video_url(c) for c in await repo.list_candidates_for_scene(scene_id)]


@router.get("/candidates/{candidate_id}", response_model=dict)
async def get_candidate(candidate_id: str):
    repo = get_story_repository()
    cand = await repo.get_candidate(candidate_id)
    if cand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    return _with_video_url(cand)


@router.post("/candidates/{candidate_id}/select", response_model=dict)
async def select_candidate(candidate_id: str, request: SelectRequest):
    """
    Human override: mark this take as the selected one for its scene.

    Implicitly records a positive preference label for this candidate so the
    override also feeds judge improvement. Recompose the story afterwards with
    POST /stories/{story_id}/recompose to rebuild the final video.
    """
    repo = get_story_repository()
    cand = await repo.set_selected_candidate(candidate_id, selection_source="human")
    if cand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    await repo.create_judge_feedback(
        feedback_id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        user=request.user,
        human_passed=True,
        comment=request.comment or "human override: selected this take",
    )
    return cand


@router.post("/candidates/{candidate_id}/feedback", response_model=dict)
async def submit_feedback(candidate_id: str, request: FeedbackRequest):
    """Record a human preference label on a candidate (agrees/disagrees with the judge)."""
    repo = get_story_repository()
    cand = await repo.get_candidate(candidate_id)
    if cand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    return await repo.create_judge_feedback(
        feedback_id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        user=request.user,
        human_passed=request.human_passed,
        human_score=request.human_score,
        artifact_tags=request.artifact_tags,
        comment=request.comment,
    )


@router.get("/judge-feedback/export", response_model=List[dict])
async def export_judge_feedback(limit: int = 1000):
    """
    Export human labels joined with judge verdicts and scene context.

    Use this dataset to evaluate the judge and tune its prompt/model
    (disagreements between judge_passed and human_passed are the signal).
    """
    repo = get_story_repository()
    return await repo.export_judge_feedback(limit=limit)
