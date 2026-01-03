"""
High-level API: Video Generation Application

Provides complete video generation workflow from story/title to final video.
This is a high-level application that orchestrates multiple services.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import uuid
import os
from datetime import datetime

from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_llm,
    create_tts,
    create_stt,
    create_video_retriever,
    create_prompt_provider,
    generate_story,
    generate_video_from_story,
    recreate_from_config_dump,
    GenerationResult,
    StoryOutput,
)
from virtual_streamer.utils.job_store import get_global_job_store

# Router setup
router = APIRouter(prefix="/video-generation", tags=["Video Generation"])


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""

    # Input (mutually exclusive)
    title: Optional[str] = None
    story_text: Optional[str] = None
    from_config_dump: Optional[str] = None

    # Configuration overrides
    character_name: Optional[str] = None
    llm_provider: Optional[str] = "anthropic"
    llm_model: Optional[str] = "claude-sonnet-4-5-20250929"
    tts_provider: Optional[str] = "fish"
    tts_host: Optional[str] = "127.0.0.1"
    tts_port: int = 8003
    stt_provider: Optional[str] = "whisper"
    stt_model: Optional[str] = "base"

    output_dir: Optional[str] = None
    max_parallel_llm_calls: int = 5
    verbose: bool = False


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str


class VideoGenerationResponse(BaseModel):
    """Response model for video generation submission."""

    job_id: str
    status: str
    message: str


async def _run_video_generation(job_id: str, request: VideoGenerationRequest):
    """Background task to run video generation."""
    job_store = await get_global_job_store()
    try:
        # Update job status to running
        await job_store.update_job(job_id, status="running")

        # Build configuration
        config = VideoGenerationConfig(
            title=request.title,
            story_file=None,  # We'll use story_text directly
            from_config_dump=request.from_config_dump,
            character_name=request.character_name,
            output_dir=request.output_dir or "./output",
            verbose=request.verbose,
            max_parallel_llm_calls=request.max_parallel_llm_calls,
        )

        # Override config with request parameters
        config.llm.provider = request.llm_provider
        config.llm.model = request.llm_model
        config.tts.provider = request.tts_provider
        config.tts.host = request.tts_host
        config.tts.port = request.tts_port
        config.stt.provider = request.stt_provider
        config.stt.model = request.stt_model

        # Validate inputs
        inputs = [request.title, request.story_text, request.from_config_dump]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Exactly one of title, story_text, or from_config_dump must be provided"
            )

        # Simple progress callback (logs to console only)
        class ProgressCallback:
            def __init__(self):
                self.current_step = 0
                self.total_steps = 0

            def update(self, message: str):
                print(f"[Job {job_id}] {message}")

            def set_total_steps(self, total: int):
                self.total_steps = total

            def increment_step(self, message: str):
                self.current_step += 1
                print(f"[Job {job_id}] [{self.current_step}/{self.total_steps}] {message}")

        progress = ProgressCallback()

        # Handle config dump recreation
        if request.from_config_dump:
            tts = create_tts(config.tts, character_name=config.character_name)
            stt = create_stt(config.stt)

            result = await recreate_from_config_dump(
                request.from_config_dump, tts, stt, config, progress
            )

        else:
            # Initialize components
            llm = create_llm(config.llm)
            tts = create_tts(config.tts, character_name=config.character_name)
            stt = create_stt(config.stt)
            video_retriever = create_video_retriever(config.video_retrieval)
            prompt_provider = create_prompt_provider(config.prompt)

            # Create semaphore for LLM concurrency
            llm_semaphore = asyncio.Semaphore(config.max_parallel_llm_calls)

            # Generate or use provided story
            story_output = None
            if request.title:
                story_output = await generate_story(
                    request.title, llm, prompt_provider, config, progress, llm_semaphore
                )
                story = story_output.dialog
            else:
                story = request.story_text

            # Generate video
            result = await generate_video_from_story(
                story,
                llm,
                tts,
                stt,
                video_retriever,
                config,
                progress,
                story_output=story_output,
            )

        # Job completed successfully
        result_data = {
            "video_path": result.video_path,
            "config_dump_path": result.config_dump_path,
            "metadata": result.metadata,
            "story_output": result.story_output.model_dump()
            if result.story_output
            else None,
        }
        await job_store.update_job(job_id, status="completed", result=result_data)

    except Exception as e:
        # Job failed
        await job_store.update_job(job_id, status="failed", error=str(e))

        import traceback

        print(f"Video generation job {job_id} failed:")
        traceback.print_exc()


@router.post("/submit", response_model=VideoGenerationResponse)
async def submit_video_generation(
    request: VideoGenerationRequest, background_tasks: BackgroundTasks
):
    """
    Submit a video generation job.

    The job runs asynchronously in the background. Use the job_id
    to check status and retrieve results.

    Args:
        request: VideoGenerationRequest with title, story, or config dump

    Returns:
        VideoGenerationResponse with job_id for tracking
    """
    job_store = await get_global_job_store()

    # Create job
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, request.model_dump())

    # Start background task
    background_tasks.add_task(_run_video_generation, job_id, request)

    return VideoGenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation job submitted successfully",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a video generation job.

    Args:
        job_id: Job ID returned from submit endpoint

    Returns:
        JobStatusResponse with current status and results
    """
    job_store = await get_global_job_store()
    job = await job_store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(**job)


@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 20):
    """
    List recent video generation jobs.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        List of JobStatusResponse
    """
    job_store = await get_global_job_store()
    jobs = await job_store.list_jobs(limit)

    return [JobStatusResponse(**job) for job in jobs]


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the tracking system.

    Note: This only removes the job metadata, not the generated files.
    """
    job_store = await get_global_job_store()
    deleted = await job_store.delete_job(job_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Job deleted successfully"}


@router.get("/health")
async def health():
    """Health check for video generation service."""
    job_store = await get_global_job_store()
    jobs = await job_store.list_jobs(limit=1000)  # Get recent jobs for stats

    active_count = sum(1 for j in jobs if j["status"] in ["pending", "running"])

    return {
        "status": "healthy",
        "active_jobs": active_count,
        "total_jobs": len(jobs),
        "storage_backend": os.environ.get("JOB_STORAGE_BACKEND", "memory"),
    }
