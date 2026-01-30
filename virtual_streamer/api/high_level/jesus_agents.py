"""
High-level API: Jesus Agent Video Generation

Provides endpoints for character agent video responses.
Pipeline: Agent → TTS → Wav2Lip → STT → Subtitles → MinIO
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Callable, Optional
import uuid
import os
import shutil
import logging
import tempfile
from datetime import datetime

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Reused imports
from virtual_streamer.video_server.models import DialogueEntry, VideoClipBase, VideoOptions
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.utils import combine_video_and_audio, add_subtitle_from_srt
from virtual_streamer.api.medium_level.tts import generate_tts
from virtual_streamer.api.medium_level.wav2lip import generate_wav2lip, Wav2LipRequest
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
    character_id: str = "jesus"
    agent_name: str = "greeting_jesus_agent"


class AnsweringJesusRequest(BaseModel):
    """Request for answering a question."""
    question: str
    user_name: str
    character_id: str = "jesus"
    agent_name: str = "answering_jesus_agent"


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
    
    This is a local helper that directly uses stable_whisper,
    avoiding an HTTP call when running in the same process.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the SRT file
        
    Returns:
        Path to the generated SRT file
    """
    import stable_whisper
    
    # Load Whisper model
    model = stable_whisper.load_model("base")
    
    # Transcribe
    result = model.transcribe(audio_path)
    
    # Generate SRT file
    srt_filename = f"subtitle_{uuid.uuid4().hex[:8]}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    result.to_srt_vtt(srt_path, word_level=False)
    
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


async def text_to_video_with_subtitles(
    text: str,
    character_id: str,
    job_id: str,
    agent_name: str,
) -> Dict[str, Any]:
    """
    Convert text to video with TTS, Wav2Lip, and subtitles.
    
    Pipeline (reusing existing services like script_to_video):
    1. TTS: Text → Audio
    2. Wav2Lip: Audio + Character video → Lip-synced video
    3. Combine: Video + Audio → Combined video
    4. STT: Audio → SRT subtitles
    5. Add subtitles to video
    6. Upload to MinIO
    """
    temp_dir = os.environ.get("TEMP_DIR", "./temp")
    work_dir = os.path.join(temp_dir, f"jesus_agent_{job_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    final_path = None
    
    try:
        # Load character (using new utility)
        character = await load_character(character_id)
        
        # Step 1: TTS (reused from script_to_video pattern)
        logger.info(f"[Job {job_id}] [1/5] Generating TTS audio...")
        tts_response = await generate_tts(
            DialogueEntry(
                entry_id=str(uuid.uuid4()),
                character_id=character.character_id,
                text=text,
                timestamp=0,
            )
        )
        audio_path = tts_response.audio_path
        
        # Step 2: Wav2Lip (reused)
        logger.info(f"[Job {job_id}] [2/5] Generating Wav2Lip video...")
        wav2lip_response = await generate_wav2lip(Wav2LipRequest(
            audio_path=os.path.abspath(audio_path),
            video=VideoClipBase(
                storage_path=character.video_clip_path,
                collection_ids=[],
            ),
            character_id=character.character_id,
            output_dir=work_dir,
            options=VideoOptions(subtitles_enabled=False, subtitle_style=None),
        ))
        raw_video_path = wav2lip_response.raw_video_path
        
        # Step 3: Combine video + audio (reused)
        logger.info(f"[Job {job_id}] [3/5] Combining video and audio...")
        combined_path = os.path.join(work_dir, "combined.mp4")
        combine_video_and_audio(raw_video_path, audio_path, combined_path)
        
        # Step 4: Generate subtitles via STT (local call)
        logger.info(f"[Job {job_id}] [4/5] Generating subtitles...")
        srt_path = await _transcribe_audio_to_srt(audio_path, work_dir)
        
        # Step 5: Add subtitles to video (reused)
        logger.info(f"[Job {job_id}] [5/5] Adding subtitles...")
        final_path = os.path.join(work_dir, "final.mp4")
        add_subtitle_from_srt(combined_path, srt_path, final_path, fontsize=14)
        
        # Step 6: Upload to MinIO (reused)
        logger.info(f"[Job {job_id}] Uploading to MinIO...")
        storage = get_storage_client()
        minio_video_key = f"generated_videos/{agent_name}/{job_id}.mp4"
        await storage.upload_file(final_path, minio_video_key)
        video_url = storage.get_url(minio_video_key)
        
        return {
            "video_path": final_path,
            "minio_video_key": minio_video_key,
            "video_url": video_url,
            "agent_response": text,
        }
        
    finally:
        # Cleanup work directory (but not the final uploaded video)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


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
    Common background task for agent video generation.
    
    Args:
        job_id: Job identifier
        agent_name: Name of agent to run
        character_id: Character for TTS/Wav2Lip
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
        
        # Generate video with subtitles
        result = await text_to_video_with_subtitles(
            text=agent_text,
            character_id=character_id,
            job_id=job_id,
            agent_name=agent_name,
        )
        
        # Add timestamp and any extra fields
        result["timestamp"] = datetime.now().isoformat()
        if extra_result:
            result.update(extra_result)
        
        await job_store.update_job(job_id, status="completed", result=result)
        logger.info(f"[Job {job_id}] Completed successfully")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(e))


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/greeting/submit", response_model=JesusAgentResponse)
async def submit_greeting(
    request: GreetingJesusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a greeting video generation job.
    
    The agent generates a personalized greeting for the user,
    then TTS + Wav2Lip + subtitles creates a video.
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, request.model_dump())
    
    # Use factored background task
    background_tasks.add_task(
        _run_agent_video_job,
        job_id=job_id,
        agent_name=request.agent_name,
        character_id=request.character_id,
        user_message=request.user_name,  # Agent receives just the username
        extra_result={"user_name": request.user_name},
    )
    
    return JesusAgentResponse(
        job_id=job_id,
        status="pending",
        message="Greeting video generation job submitted",
    )


@router.post("/answering/submit", response_model=JesusAgentResponse)
async def submit_answering(
    request: AnsweringJesusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a Q&A video generation job.
    
    The agent generates an answer to the question,
    then TTS + Wav2Lip + subtitles creates a video.
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, request.model_dump())
    
    # Use factored background task
    background_tasks.add_task(
        _run_agent_video_job,
        job_id=job_id,
        agent_name=request.agent_name,
        character_id=request.character_id,
        user_message=f"Question from {request.user_name}: {request.question}",
        extra_result={"user_name": request.user_name, "question": request.question},
    )
    
    return JesusAgentResponse(
        job_id=job_id,
        status="pending",
        message="Answering video generation job submitted",
    )
