"""
High-level API: Jesus Agent Video Generation

Provides endpoints for character agent video responses.
Pipeline: Agent → TTS → STT → Subtitles → MinIO
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Callable, Optional
import uuid
import os
import logging
from datetime import datetime

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Reused imports
from virtual_streamer.video_server.models import DialogueEntry
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.api.medium_level.tts import generate_tts
from virtual_streamer.agents.factory import get_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jesus-agents", tags=["Jesus Agents"])

APP_NAME = "jesus_agents"


# =============================================================================
# Request/Response Models
# =============================================================================

class GreetingJesusRequest(BaseModel):
    """Request for greeting a user."""
    user_name: str
    character_id: str = "jesus_short"
    agent_name: str = "greeting_jesus_agent"
    stream_id: Optional[str] = None  # If provided, video will be added to broadcast playlist (play once)


class AnsweringJesusRequest(BaseModel):
    """Request for answering a question."""
    question: str
    user_name: str
    character_id: str = "jesus_short"
    agent_name: str = "answering_jesus_agent"
    stream_id: Optional[str] = None  # If provided, video will be added to broadcast playlist (play once)


class JesusAgentResponse(BaseModel):
    """Response for job submission."""
    job_id: str
    status: str
    message: str


# =============================================================================
# Local STT Helper (to avoid HTTP round-trip within same process)
# =============================================================================

async def _transcribe_audio_to_srt(audio_path: str, output_dir: str) -> str:
    """
    Transcribe audio to SRT subtitle file.
    
    This is a local helper that directly uses the shared transcription utility,
    avoiding an HTTP call when running in the same process.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the SRT file
        
    Returns:
        Path to the generated SRT file
    """
    from virtual_streamer.utils.transcription import transcribe_to_srt
    
    # Generate SRT file path
    srt_filename = f"subtitle_{uuid.uuid4().hex[:8]}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    
    # Use the shared transcription utility with cached model
    transcribe_to_srt(audio_path, srt_path, model_name="base", use_faster=False)
    
    return srt_path


# =============================================================================
# Core Functions
# =============================================================================

async def run_adk_agent(agent_name: str, user_message: str) -> str:
    """
    Run an ADK agent and return its text response.
    
    Uses factory pattern to get agent instance.
    """
    agent = get_agent(agent_name)
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={},
    )
    
    content = types.Content(role="user", parts=[types.Part(text=user_message)])
    
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
    
    return response_text


# =============================================================================
# Factored Background Task
# =============================================================================

async def _run_agent_video_job(
    job_id: str,
    agent_name: str,
    character_id: str,
    user_message: str,
    extra_result: Optional[Dict[str, Any]] = None,
):
    """
    Common background task for agent text generation (video generation removed).

    Args:
        job_id: Job identifier
        agent_name: Name of agent to run
        character_id: Character identifier (kept for API compatibility)
        user_message: The message to send to the agent
        extra_result: Additional fields to include in result
    """
    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")

        # Run agent
        logger.info(f"[Job {job_id}] Running agent '{agent_name}' with message: {user_message[:50]}...")
        agent_text = await run_adk_agent(agent_name, user_message)
        logger.info(f"[Job {job_id}] Agent response: {agent_text[:100]}...")

        result: Dict[str, Any] = {
            "agent_response": agent_text,
            "timestamp": datetime.now().isoformat(),
        }
        if extra_result:
            result.update(extra_result)

        await job_store.update_job(job_id, status="completed", result=result)
        logger.info(f"[Job {job_id}] Completed successfully")

    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(e))


async def _run_agent_video_job_with_broadcast(
    job_id: str,
    agent_name: str,
    character_id: str,
    user_message: str,
    stream_id: str,
    extra_result: Optional[Dict[str, Any]] = None,
):
    """
    Agent video generation with broadcast playlist integration.
    
    Wraps video generation with post-processing to add the video to the
    active programmation's playlist with play_once=True.
    
    Args:
        job_id: Job identifier
        agent_name: Name of agent to run
        character_id: Character identifier (kept for API compatibility)
        user_message: The message to send to the agent
        stream_id: Stream to add the video to
        extra_result: Additional fields to include in result
    """
    # Step 1: Run video generation
    await _run_agent_video_job(
        job_id=job_id,
        agent_name=agent_name,
        character_id=character_id,
        user_message=user_message,
        extra_result=extra_result,
    )
    
    # Step 2: On success, add to playlist
    job_store = await get_global_job_store()
    job = await job_store.get_job(job_id)
    
    if job is None or job["status"] != "completed":
        logger.warning(f"[Broadcast {job_id}] Generation failed, skipping playlist")
        return
    
    result = job["result"]
    minio_video_key = result["minio_video_key"]
    
    # Get active programmation for the stream
    from virtual_streamer.streaming.store import get_streaming_store
    store = await get_streaming_store()
    
    current_time = datetime.now().time()
    programmation = await store.get_active_programmation(stream_id, current_time)
    
    if programmation is None:
        logger.warning(f"[Broadcast {job_id}] No active programmation for stream '{stream_id}', skipping playlist")
        return
    
    # Add to playlist with play_once=True
    entry = await store.add_to_playlist(
        prog_id=programmation.programmation_id,
        video_key=minio_video_key,
        metadata={
            "job_id": job_id,
            "agent_name": agent_name,
            "source": "jesus_agent",
        },
        play_once=True,
    )
    
    # Update job result with entry_id
    result["entry_id"] = entry.entry_id
    result["programmation_id"] = programmation.programmation_id
    await job_store.update_job(job_id, result=result)
    logger.info(f"[Broadcast {job_id}] Added to playlist (play_once): {entry.entry_id}")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/greeting/submit", response_model=JesusAgentResponse)
async def submit_greeting(
    request: GreetingJesusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a greeting agent job.

    The agent generates a personalized greeting for the user.

    If stream_id is provided, the result will be added to the broadcast
    playlist with play_once=True (no replay).
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, request.model_dump())
    
    extra_result = {"user_name": request.user_name}
    
    if request.stream_id:
        # Use broadcast wrapper to add to playlist
        background_tasks.add_task(
            _run_agent_video_job_with_broadcast,
            job_id=job_id,
            agent_name=request.agent_name,
            character_id=request.character_id,
            user_message=request.user_name,
            stream_id=request.stream_id,
            extra_result=extra_result,
        )
        message = "Greeting video generation job submitted (will add to broadcast)"
    else:
        # Standard generation without playlist
        background_tasks.add_task(
            _run_agent_video_job,
            job_id=job_id,
            agent_name=request.agent_name,
            character_id=request.character_id,
            user_message=request.user_name,
            extra_result=extra_result,
        )
        message = "Greeting video generation job submitted"
    
    return JesusAgentResponse(
        job_id=job_id,
        status="pending",
        message=message,
    )


@router.post("/answering/submit", response_model=JesusAgentResponse)
async def submit_answering(
    request: AnsweringJesusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a Q&A agent job.

    The agent generates an answer to the question.

    If stream_id is provided, the result will be added to the broadcast
    playlist with play_once=True (no replay).
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, request.model_dump())
    
    user_message = f"Question from {request.user_name}: {request.question}"
    extra_result = {"user_name": request.user_name, "question": request.question}
    
    if request.stream_id:
        # Use broadcast wrapper to add to playlist
        background_tasks.add_task(
            _run_agent_video_job_with_broadcast,
            job_id=job_id,
            agent_name=request.agent_name,
            character_id=request.character_id,
            user_message=user_message,
            stream_id=request.stream_id,
            extra_result=extra_result,
        )
        message = "Answering video generation job submitted (will add to broadcast)"
    else:
        # Standard generation without playlist
        background_tasks.add_task(
            _run_agent_video_job,
            job_id=job_id,
            agent_name=request.agent_name,
            character_id=request.character_id,
            user_message=user_message,
            extra_result=extra_result,
        )
        message = "Answering video generation job submitted"
    
    return JesusAgentResponse(
        job_id=job_id,
        status="pending",
        message=message,
    )
