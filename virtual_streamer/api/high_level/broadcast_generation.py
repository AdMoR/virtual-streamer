"""
High-level API: Broadcast-integrated video generation.

Endpoints for generating videos from active broadcasts and collecting user feedback.
"""
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.api.high_level.models import (
    VideoGenerationRequest,
    GenerateFromBroadcastRequest,
    GenerateFromBroadcastResponse,
    FeedbackRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-generation", tags=["Broadcast Generation"])

MAX_PENDING_JOBS = 5


async def _run_broadcast_generation(
    job_id: str,
    request: VideoGenerationRequest,
    programmation_id: str,
    user: Optional[str],
):
    """Broadcast workflow: generate video, then add to playlist."""
    from virtual_streamer.api.high_level.video_generation import _run_video_generation

    await _run_video_generation(job_id, request)

    job_store = await get_global_job_store()
    job = await job_store.get_job(job_id)

    if job is None or job["status"] != "completed":
        logger.warning(f"[Broadcast {job_id}] Generation failed, skipping playlist")
        return

    result = job["result"]
    minio_video_key = result["metadata"]["minio_video_key"]

    from virtual_streamer.streaming.store import get_streaming_store

    store = await get_streaming_store()
    entry = await store.add_to_playlist(
        prog_id=programmation_id,
        video_key=minio_video_key,
        metadata={
            "job_id": job_id,
            "user": user,
            "title": request.title,
            "story_template_id": request.story_template_id,
        },
    )

    result["entry_id"] = entry.entry_id
    await job_store.update_job(job_id, result=result)
    logger.info(f"[Broadcast {job_id}] Added to playlist: {entry.entry_id}")


@router.post("/generate-from-broadcast", response_model=GenerateFromBroadcastResponse)
async def generate_from_broadcast(
    request: GenerateFromBroadcastRequest, background_tasks: BackgroundTasks
):
    """
    Generate video from title using the active broadcast's story template.

    Designed for Twitch chat integration — auto-detects the story template from
    the active programmation and enforces a per-template queue limit of 5 jobs.
    """
    from virtual_streamer.streaming.store import get_streaming_store

    store = await get_streaming_store()
    stream = await store.get_stream(request.stream_id)
    if stream is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stream '{request.stream_id}' not found",
        )

    programmation = await store.get_active_programmation(
        request.stream_id, datetime.now().time()
    )
    if programmation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active programmation for stream '{request.stream_id}' at this time",
        )

    story_template_id = programmation.story_template_id
    job_store = await get_global_job_store()

    if not request.skip_queue_limit:
        pending_count = await job_store.count_pending_jobs(story_template_id)
        if pending_count >= MAX_PENDING_JOBS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Queue full: {pending_count} jobs pending for template "
                    f"'{story_template_id}'. Max is {MAX_PENDING_JOBS}."
                ),
            )

    video_request = VideoGenerationRequest(
        title=request.title,
        story_template_id=story_template_id,
    )

    job_id = str(uuid.uuid4())
    job_data = video_request.model_dump()
    job_data["source"] = "broadcast"
    job_data["stream_id"] = request.stream_id
    job_data["programmation_id"] = programmation.programmation_id
    if request.user:
        job_data["user"] = request.user

    await job_store.create_job(job_id, job_data)

    background_tasks.add_task(
        _run_broadcast_generation,
        job_id,
        video_request,
        programmation.programmation_id,
        request.user,
    )

    return GenerateFromBroadcastResponse(
        job_id=job_id,
        status="pending",
        message=f"Video generation job submitted for '{story_template_id}'",
        story_template_id=story_template_id,
    )


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Store raw user feedback for a played video in MinIO."""
    from virtual_streamer.streaming.store import get_streaming_store

    store = await get_streaming_store()
    entry = await store.get_playlist_entry(request.entry_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Entry '{request.entry_id}' not found")

    metadata = entry.metadata
    if not metadata:
        raise HTTPException(status_code=400, detail="Entry has no metadata")

    feedback_data = {
        "entry_id": request.entry_id,
        "job_id": metadata["job_id"],
        "story_template_id": metadata["story_template_id"],
        "title": metadata["title"],
        "video_path": entry.video_storage_key,
        "user": request.user,
        "feedback": request.feedback,
        "timestamp": datetime.now().isoformat(),
    }

    storage = get_storage_client()
    feedback_key = f"feedback/{metadata['story_template_id']}/{request.entry_id}.json"
    await storage.put_json(feedback_key, feedback_data)

    logger.info(f"Feedback stored: {feedback_key}")
    return {"status": "ok", "feedback_key": feedback_key}
