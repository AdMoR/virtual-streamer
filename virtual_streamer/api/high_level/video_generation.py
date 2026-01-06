"""
High-level API: Video Generation Application

Provides complete video generation workflow from story/title to final video.
This is a high-level application that orchestrates ADK agents and webservices.

Architecture:
    1. StoryGeneratorAgent (ADK) - generates story with DialogLines from title
    2. SentenceVideoMatcher (ADK) - matches each dialog line to a video
    3. script_to_video - TTS/Wav2Lip/STT via webservice + local video combination
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import uuid
import os
import logging
from datetime import datetime
from dataclasses import dataclass

import httpx

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Agent imports
from virtual_streamer.agents.story_generator import get_story_generator
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.agents.sentence_video_matcher import (
    create_sentence_video_matcher,
    SentenceVideoMatcherOutput,
    DialogLineMatch,
)
from virtual_streamer.agents.common.state_keys import (
    TITLE,
    STORY_TEMPLATE_ID,
    STORY_OUTPUT,
    SENTENCES,
    VIDEO_MATCHES,
    VIDEO_COLLECTION,
)

# Video generation imports
from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_video_retriever,
    GenerationResult,
)

# Local video processing utilities
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
    get_length,
)
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/video-generation", tags=["Video Generation"])


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class APIConfig:
    """Configuration for webservice API calls."""

    base_url: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
    timeout: float = 120.0


# ============================================================================
# Webservice Client
# ============================================================================


class WebserviceClient:
    """
    Async HTTP client for calling the Virtual Streamer webservice API.

    Handles TTS, Wav2Lip, and STT API calls with proper error handling.
    """

    def __init__(self, config: APIConfig, character_id: str = "fred"):
        self.config = config
        self.character_id = character_id
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def generate_tts(self, text: str, entry_id: str = "") -> str:
        """
        Call TTS API to generate audio from text.

        Args:
            text: Dialog text to synthesize
            entry_id: Optional entry ID for tracking

        Returns:
            Path to generated audio file
        """
        response = await self._client.post(
            "/api/v1/tts/generate",
            json={
                "entry_id": entry_id or f"tts_{datetime.now().timestamp()}",
                "character_id": self.character_id,
                "text": text,
                "timestamp": 0,
            },
            timeout=30*60,
        )
        response.raise_for_status()
        data = response.json()
        return data["audio_path"]

    async def generate_wav2lip(
        self,
        audio_path: str,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Call Wav2Lip API to generate lip-synced video.

        Args:
            audio_path: Path to audio file
            video_path: Path to source video
            output_dir: Optional output directory

        Returns:
            Path to generated lip-synced video
        """
        response = await self._client.post(
            "/api/v1/wav2lip/generate",
            json={
                "audio_path": audio_path,
                "video": {
                    "storage_path": video_path,
                    "collection_ids": [],
                },
                "options": {
                    "subtitles_enabled": False,
                    "subtitle_style": None,
                },
                "character_id": self.character_id,
                "output_dir": output_dir,
            },
            timeout=30*60,
        )
        response.raise_for_status()
        data = response.json()
        return data["raw_video_path"]

    async def transcribe_to_srt(self, audio_path: str) -> str:
        """
        Call STT API to generate SRT subtitles from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to generated SRT file
        """
        with open(audio_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_path), f, "audio/wav")}
            response = await self._client.post(
                "/api/v1/stt/transcribe-to-srt",
                files=files,
                timeout=30*60,
            )
        response.raise_for_status()
        data = response.json()
        return data["srt_path"]


# ============================================================================
# ADK Agent Runners
# ============================================================================


APP_NAME = "virtual_streamer"


async def run_story_generator(
    title: str, story_template_id: Optional[str] = None
) -> StoryOutput:
    """
    Run the StoryGeneratorAgent to generate a story from a title.

    Args:
        title: The title/topic for story generation
        story_template_id: Optional story template ID to customize generation

    Returns:
        StoryOutput with title, story_plan, and dialog
    """
    # Get the story generator agent
    story_generator = get_story_generator()

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=story_generator,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create session with initial state
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"story_{uuid.uuid4().hex[:8]}"
    initial_state = {TITLE: title}
    if story_template_id:
        initial_state[STORY_TEMPLATE_ID] = story_template_id
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    logger.info(f"Running StoryGeneratorAgent for title: {title}")

    # Create message content
    content = types.Content(role="user", parts=[types.Part(text=title)])

    # Run agent via runner (state_delta is applied automatically)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                logger.debug(f"Final response from {event.author}")

    # Get updated session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    # Extract story output from state
    story_output_data = session.state.get(STORY_OUTPUT)
    if not story_output_data:
        raise RuntimeError("StoryGeneratorAgent completed but no story output in state")

    # Parse story output
    if isinstance(story_output_data, StoryOutput):
        return story_output_data
    elif isinstance(story_output_data, dict):
        return StoryOutput.model_validate(story_output_data)
    else:
        raise RuntimeError(f"Unexpected story_output type: {type(story_output_data)}")


async def run_sentence_video_matcher(
    sentences: List[Any],
    collection: str,
    config: VideoGenerationConfig,
) -> SentenceVideoMatcherOutput:
    """
    Run the SentenceVideoMatcher to match dialog lines to videos.

    Args:
        sentences: List of sentences/dialog lines to match
        collection: Qdrant collection name for video search (from StoryTemplate)
        config: Video generation configuration

    Returns:
        SentenceVideoMatcherOutput with matches for each dialog line
    """
    # Create video retriever
    video_retriever = create_video_retriever(config.video_retrieval)

    # Create the sentence video matcher agent
    video_matcher = create_sentence_video_matcher(
        video_retriever=video_retriever,
        max_candidates=config.max_video_judgement_attempts,
    )

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=video_matcher,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # Create session with initial state including video collection
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"matcher_{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            SENTENCES: sentences,
            VIDEO_COLLECTION: collection,
        },
    )

    logger.info(f"Running SentenceVideoMatcher agent with collection '{collection}'")

    # Create message content
    content = types.Content(role="user", parts=[types.Part(text="Match videos to sentences")])

    # Run agent via runner (state_delta is applied automatically)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                logger.debug(f"Final response from {event.author}")

    # Get updated session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    # Extract video matches from state
    video_matches_data = session.state.get(VIDEO_MATCHES)
    if not video_matches_data:
        raise RuntimeError("SentenceVideoMatcher completed but no matches in state")

    # Parse video matches
    if isinstance(video_matches_data, SentenceVideoMatcherOutput):
        return video_matches_data
    elif isinstance(video_matches_data, str):
        return SentenceVideoMatcherOutput.model_validate_json(video_matches_data)
    elif isinstance(video_matches_data, dict):
        return SentenceVideoMatcherOutput.model_validate(video_matches_data)
    else:
        raise RuntimeError(f"Unexpected video_matches type: {type(video_matches_data)}")


# ============================================================================
# Script to Video Function
# ============================================================================


async def script_to_video(
    matches: List[DialogLineMatch],
    client: WebserviceClient,
    config: VideoGenerationConfig,
    progress_callback: Optional[callable] = None,
) -> str:
    """
    Convert matched dialog lines to final video using webservices.

    For each match:
    1. Generate audio via TTS API
    2. Generate lip-synced video via Wav2Lip API
    3. Generate subtitles via STT API
    4. Combine video + audio + subtitles locally

    Then concatenate all segments into final video.

    Args:
        matches: List of DialogLineMatch from SentenceVideoMatcher
        client: WebserviceClient for API calls
        config: Video generation configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Path to final concatenated video
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.temp_dir, exist_ok=True)

    video_segments = []

    for i, match in enumerate(matches):
        dialog = match.dialog_line.text
        video_path = match.video_path

        if progress_callback:
            progress_callback(f"Processing segment {i+1}/{len(matches)}: {dialog[:50]}...")

        logger.info(f"Processing segment {i+1}/{len(matches)}: {dialog[:50]}...")

        try:
            # Step 1: Generate audio via TTS API
            logger.info(f"  [1/4] Generating TTS audio...")
            audio_path = await client.generate_tts(
                text=dialog,
                entry_id=f"segment_{i}",
            )
            logger.info(f"  Audio generated: {audio_path}")

            # Step 2: Generate lip-synced video via Wav2Lip API
            logger.info(f"  [2/4] Generating Wav2Lip video...")
            wav2lip_output_dir = os.path.join(config.temp_dir, f"wav2lip_{i}")
            lip_synced_video = await client.generate_wav2lip(
                audio_path=audio_path,
                video_path=video_path,
                output_dir=wav2lip_output_dir,
            )
            logger.info(f"  Lip-synced video: {lip_synced_video}")

            # Step 3: Combine video and audio
            logger.info(f"  [3/4] Combining video and audio...")
            combined_path = os.path.join(config.temp_dir, f"combined_{i}.mp4")
            combine_video_and_short_audio(lip_synced_video, audio_path, combined_path)

            # Step 4: Generate subtitles and add to video
            logger.info(f"  [4/4] Adding subtitles...")
            srt_path = await client.transcribe_to_srt(audio_path)
            segment_path = os.path.join(config.temp_dir, f"segment_{i}.mp4")
            add_subtitle_from_srt(
                combined_path,
                srt_path,
                segment_path,
                fontsize=config.video_processing.fontsize,
            )

            video_segments.append(segment_path)
            logger.info(f"  Segment {i+1} complete: {segment_path}")

        except Exception as e:
            logger.error(f"  Error processing segment {i+1}: {e}", exc_info=True)
            raise

    # Concatenate all segments
    logger.info(f"Concatenating {len(video_segments)} segments...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = os.path.join(config.output_dir, f"video_{timestamp}.mp4")
    concat_file = os.path.join(config.temp_dir, "concat_list.txt")

    combine_part_in_concat_file(video_segments, concat_file, final_video_path)

    logger.info(f"Final video created: {final_video_path}")
    return final_video_path


# ============================================================================
# Request/Response Models
# ============================================================================


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""

    # Input (mutually exclusive: title or story_text)
    title: Optional[str] = None
    story_text: Optional[str] = None

    # Story template (required - defines characters, prompt, and video collection)
    story_template_id: str

    # Character configuration
    character_name: Optional[str] = "fred"

    # LLM configuration (for ADK agents)
    llm_provider: Optional[str] = "anthropic"
    llm_model: Optional[str] = "claude-sonnet-4-5-20250929"

    # Output configuration
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


# ============================================================================
# Background Task
# ============================================================================


async def _run_video_generation(job_id: str, request: VideoGenerationRequest):
    """Background task to run video generation using ADK agents."""
    job_store = await get_global_job_store()
    try:
        # Update job status to running
        await job_store.update_job(job_id, status="running")

        # Build configuration
        config = VideoGenerationConfig(
            title=request.title,
            story_file=None,
            character_name=request.character_name,
            output_dir=request.output_dir or "./output",
            verbose=request.verbose,
            max_parallel_llm_calls=request.max_parallel_llm_calls,
        )

        # Validate inputs - require title or story_text
        inputs = [request.title, request.story_text]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Exactly one of title or story_text must be provided"
            )

        # API configuration
        api_config = APIConfig()
        character_id = request.character_name or "fred"

        story_output = None
        sentences = None

        # Load StoryTemplate to get collection for video search
        from virtual_streamer.utils.entity_repository import get_entity_repository
        repo = get_entity_repository()
        story_template = await repo.get_story_template(request.story_template_id)
        if story_template is None:
            raise ValueError(f"Story template '{request.story_template_id}' not found")
        collection = story_template["collection"]
        logger.info(
            f"[Job {job_id}] Using story template: {request.story_template_id}, "
            f"collection: {collection}"
        )

        if request.title:
            # Step 1: Run StoryGeneratorAgent
            logger.info(f"[Job {job_id}] Running StoryGeneratorAgent...")
            story_output = await run_story_generator(
                request.title, story_template_id=request.story_template_id
            )
            logger.info(
                f"[Job {job_id}] Story generated: {story_output.title} "
                f"with {len(story_output.dialog)} dialog lines"
            )
            # Extract sentences from story output
            sentences = story_output.dialog

        elif request.story_text:
            # Parse story_text as DialogLines
            # For now, treat story_text as simple text to be split
            from virtual_streamer.agents.story_generator.schema import (
                DialogLine,
                DialogLines,
            )

            # Simple split by newlines - each line is a dialog from "Fred"
            lines = [
                DialogLine(character="Fred", text=line.strip())
                for line in request.story_text.split("\n")
                if line.strip()
            ]
            dialog_lines = DialogLines(lines=lines)
            sentences = dialog_lines.model_dump()

        # Step 2: Run SentenceVideoMatcher
        logger.info(f"[Job {job_id}] Running SentenceVideoMatcher...")
        video_matches = await run_sentence_video_matcher(sentences, collection, config)
        logger.info(
            f"[Job {job_id}] Video matching complete: {len(video_matches.matches)} matches"
        )

        # Step 3: Script to Video (TTS, Wav2Lip, STT via webservices)
        logger.info(f"[Job {job_id}] Running script_to_video...")

        async with WebserviceClient(api_config, character_id) as client:
            final_video_path = await script_to_video(
                matches=video_matches.matches,
                client=client,
                config=config,
                progress_callback=lambda msg: logger.info(f"[Job {job_id}] {msg}"),
            )

        # Upload to MinIO storage
        logger.info(f"[Job {job_id}] Uploading to storage...")
        storage = get_storage_client()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        minio_video_key = f"generated_videos/{timestamp}/video_{timestamp}.mp4"
        await storage.upload_file(final_video_path, minio_video_key)

        # Generate presigned URL for video access
        video_url = storage.get_url(minio_video_key)
        logger.info(f"[Job {job_id}] Video URL generated: {video_url[:80]}...")

        result = GenerationResult(
            video_path=final_video_path,
            config_dump_path=None,
            story_output=story_output,
            metadata={
                "sentence_count": len(video_matches.matches),
                "total_duration": get_length(final_video_path),
                "timestamp": datetime.now().isoformat(),
                "minio_video_key": minio_video_key,
                "video_url": video_url,
            },
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

        logger.error(f"Video generation job {job_id} failed:")
        traceback.print_exc()


# ============================================================================
# API Endpoints
# ============================================================================


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
        "api_base_url": os.environ.get("API_BASE_URL", "http://localhost:8000"),
    }
