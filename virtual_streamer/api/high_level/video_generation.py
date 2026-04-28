"""
High-level API: Video Generation Application

Provides complete video generation workflow from story/title to final video.
This is a high-level application that orchestrates ADK agents and webservices.

Architecture (Traditional Pipeline):
    1. StoryPipelineAgent (ADK) - two-step story generation (writer → formatter) with guardrail
    2. SentenceVideoMatcher (ADK) - matches each dialog line to a video
    3. script_to_video - TTS/Wav2Lip/STT via webservice + local video combination

Architecture (LTX-2 Pipeline):
    1. StoryPipelineAgent (ADK) - two-step story generation (writer → formatter) with guardrail
    2. story_to_video - LTX-2 text-to-video for each dialog line (video + audio)
    3. Concatenate segments into final video
"""

import base64
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, File, Form, HTTPException, BackgroundTasks, UploadFile
# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from virtual_streamer.agents.common.state_keys import (
    TITLE,
    STORY_TEMPLATE_ID,
    STORY_OUTPUT,
    SENTENCES,
    VIDEO_MATCHES,
    VIDEO_COLLECTION,
)
from virtual_streamer.agents.sentence_video_matcher import (
    create_sentence_video_matcher,
    SentenceVideoMatcherOutput,
    DialogLineMatch
)
# Agent imports
from virtual_streamer.agents.story_pipeline import get_story_pipeline
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import MinIOClient
from virtual_streamer.utils.minio_client import get_storage_client
# Local video processing utilities
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
    get_length,
    txt_to_speech_call_fish_async,
)
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.api.dependencies import get_storage_resolver
# Video generation imports
from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_video_retriever,
    GenerationResult,
    GenerationBlueprint,
)
from virtual_streamer.video_generation.config import LTXConfig
from virtual_streamer.video_generation.ltx_client import (
    WanGPLTXClient,
    LTXVideoConfig,
    VideoGenerationParams,
)
from virtual_streamer.video_generation.ltx_prompt_builder import build_negative_prompt
from virtual_streamer.video_generation.story_to_video import (
    story_to_video,
    StoryVideoResult,
)
from virtual_streamer.video_server.models import Character, VoiceSample

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/video-generation", tags=["Video Generation"])

# ============================================================================
# Webservice Client (imported from shared module)
# ============================================================================

from virtual_streamer.api.clients.webservice_client import WebserviceClient, APIConfig

# ============================================================================
# ADK Agent Runners
# ============================================================================


APP_NAME = "virtual_streamer"


async def run_story_generator(
        title: str,
        story_template_id: Optional[str] = None,
        safe: bool = True,
) -> StoryOutput:
    """
    Run the StoryPipelineAgent to generate a story from a title.

    Uses the two-step pipeline (writer → formatter) for more robust structured output.
    Guardrails are always included.

    Args:
        title: The title/topic for story generation
        story_template_id: Optional story template ID to customize generation
        safe: kept for API compatibility (pipeline always runs with guardrails)

    Returns:
        StoryOutput with title, story_plan, and dialog
    """
    story_generator = get_story_pipeline()

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
        character_map: Dict[str, Character],
        config: VideoGenerationConfig,
) -> SentenceVideoMatcherOutput:
    """
    Run the SentenceVideoMatcher to match dialog lines to videos.

    Args:
        sentences: List of sentences/dialog lines to match
        collection: Qdrant collection name for video search (from StoryTemplate)
        character_map: Dictionary mapping characters id to full character definition
        config: Video generation configuration

    Returns:
        SentenceVideoMatcherOutput with matches for each dialog line
    """
    # Create video retriever
    video_retriever = create_video_retriever(config.video_retrieval)

    # Create the sentence video matcher agent
    video_matcher = create_sentence_video_matcher(
        video_retriever=video_retriever,
        max_candidates=config.max_video_candidates,
        character_map=character_map
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
# LTX Fallback Video Generation
# ============================================================================


async def generate_ltx_fallback_video(
        scene_description: str,
        ltx_config: LTXConfig,
        output_dir: str,
        segment_index: int,
) -> str:
    """
    Generate a video segment using LTX-2 text-to-video.

    Used as fallback when video matching returns non-CONTEXTUAL ratings.

    Args:
        scene_description: Visual description of the scene (from DialogLine)
        ltx_config: LTX configuration with server URL and video params
        output_dir: Directory to save the generated video
        segment_index: Segment index for naming the output file

    Returns:
        Path to the generated video file
    """
    # Build the LTX Video API config
    api_config = LTXVideoConfig(
        server_url=ltx_config.server_url,
        timeout=ltx_config.timeout,
    )

    # Build video generation params from scene description
    prompt = f"{scene_description} {ltx_config.style_suffix}"
    params = VideoGenerationParams(
        prompt=prompt,
        negative_prompt=build_negative_prompt(),
        width=ltx_config.width,
        height=ltx_config.height,
        duration_seconds=ltx_config.duration_seconds,
        fps=ltx_config.fps,
        steps=ltx_config.steps,
        cfg_scale=ltx_config.cfg_scale,
    )

    # Generate video
    segment_dir = os.path.join(output_dir, f"ltx_segment_{segment_index:03d}")
    os.makedirs(segment_dir, exist_ok=True)

    async with WanGPLTXClient(api_config) as ltx_client:
        result = await ltx_client.generate_video(
            params=params,
            output_dir=segment_dir,
        )

    return result.video_path


# ============================================================================
# Script to Video Function
# ============================================================================


async def script_to_video(
        matches: List[DialogLineMatch],
        client: WebserviceClient,
        config: VideoGenerationConfig,
        ltx_config: Optional[LTXConfig] = None,
        enable_ltx_fallback: bool = False,
        progress_callback: Optional[callable] = None,
        debug_upload_prefix: Optional[str] = None,
) -> str:
    """
    Convert matched dialog lines to final video using webservices.

    For each match:
    1. Generate audio via TTS API
    2. Generate lip-synced video via Wav2Lip API (or LTX-2 if fallback enabled)
    3. Generate subtitles via STT API
    4. Combine video + audio + subtitles locally

    Then concatenate all segments into final video.

    Args:
        matches: List of DialogLineMatch from SentenceVideoMatcher
        client: WebserviceClient for API calls
        config: Video generation configuration
        ltx_config: Optional LTX configuration for fallback video generation
        enable_ltx_fallback: If True, use LTX-2 for non-CONTEXTUAL matches
        progress_callback: Optional callback for progress updates
        debug_upload_prefix: Optional MinIO prefix for uploading debug artifacts
            (e.g., "debug/video-generation/template_id/job_id")

    Returns:
        Path to final concatenated video
    """
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.temp_dir, exist_ok=True)

    # Get storage client if debug upload is enabled
    storage = get_storage_client() if debug_upload_prefix else None

    video_segments = []

    for i, match in enumerate(matches):
        dialog = match.dialog_line.text

        if progress_callback:
            progress_callback(f"Processing segment {i + 1}/{len(matches)}: {dialog[:50]}...")

        logger.info(f"Processing segment {i + 1}/{len(matches)}: {dialog[:50]}...")

        try:
            # Check if we need to generate video via LTX fallback
            if match.needs_generation and enable_ltx_fallback and ltx_config:
                logger.info(
                    f"  [FALLBACK] Generating video via LTX-2 (rating: {match.rating.value})..."
                )
                video_path = await generate_ltx_fallback_video(
                    scene_description=match.dialog_line.scene_description,
                    ltx_config=ltx_config,
                    output_dir=config.temp_dir,
                    segment_index=i,
                )
                logger.info(f"  [FALLBACK] LTX video generated: {video_path}")

                # Upload LTX video if debug enabled
                if storage and debug_upload_prefix:
                    ltx_key = f"{debug_upload_prefix}/ltx_fallback/segment_{i}.mp4"
                    await storage.upload_file(video_path, ltx_key)
                    logger.info(f"  [DEBUG] Uploaded LTX fallback: {ltx_key}")
            else:
                video_path = match.video_path

            # Step 1: Generate audio via TTS API
            logger.info(f"  [1/4] Generating TTS audio...")
            audio_path = await client.generate_tts(
                text=dialog,
                character_id=match.dialog_line.character_id,
                entry_id=f"segment_{i}",
            )
            logger.info(f"  Audio generated: {audio_path}")

            # Upload TTS audio if debug enabled
            if storage and debug_upload_prefix:
                tts_key = f"{debug_upload_prefix}/tts/segment_{i}.wav"
                await storage.upload_file(audio_path, tts_key)
                logger.info(f"  [DEBUG] Uploaded TTS: {tts_key}")

            # Step 2: Generate lip-synced video via Wav2Lip API
            logger.info(f"  [2/4] Generating Wav2Lip video...")
            wav2lip_output_dir = os.path.join(config.temp_dir, f"wav2lip_{i}")
            lip_synced_video = await client.generate_wav2lip(
                audio_path=audio_path,
                video_path=video_path,
                character_id=match.dialog_line.character_id,
                output_dir=wav2lip_output_dir,
            )
            logger.info(f"  Lip-synced video: {lip_synced_video}")

            # Upload Wav2Lip video if debug enabled
            if storage and debug_upload_prefix:
                wav2lip_key = f"{debug_upload_prefix}/wav2lip/segment_{i}.mp4"
                await storage.upload_file(lip_synced_video, wav2lip_key)
                logger.info(f"  [DEBUG] Uploaded Wav2Lip: {wav2lip_key}")

            # Step 3: Combine video and audio
            logger.info(f"  [3/4] Combining video and audio...")
            combined_path = os.path.join(config.temp_dir, f"combined_{i}.mp4")
            combine_video_and_short_audio(lip_synced_video, audio_path, combined_path)

            # Upload combined video if debug enabled
            if storage and debug_upload_prefix:
                combined_key = f"{debug_upload_prefix}/combined/segment_{i}.mp4"
                await storage.upload_file(combined_path, combined_key)
                logger.info(f"  [DEBUG] Uploaded combined: {combined_key}")

            # Step 4: Generate subtitles and add to video
            logger.info(f"  [4/4] Adding subtitles...")
            srt_path = await client.transcribe_to_srt(audio_path)

            # Upload subtitles if debug enabled
            if storage and debug_upload_prefix:
                srt_key = f"{debug_upload_prefix}/subtitles/segment_{i}.srt"
                await storage.upload_file(srt_path, srt_key)
                logger.info(f"  [DEBUG] Uploaded subtitles: {srt_key}")

            segment_path = os.path.join(config.temp_dir, f"segment_{i}.mp4")
            add_subtitle_from_srt(
                combined_path,
                srt_path,
                segment_path,
                fontsize=config.video_processing.fontsize,
            )

            video_segments.append(segment_path)
            logger.info(f"  Segment {i + 1} complete: {segment_path}")

        except Exception as e:
            logger.error(f"  Error processing segment {i + 1}: {e}", exc_info=True)
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

    # LLM configuration (for ADK agents)
    llm_provider: Optional[str] = "anthropic"
    llm_model: Optional[str] = "claude-sonnet-4-5-20250929"

    # Output configuration
    output_dir: Optional[str] = None
    max_parallel_llm_calls: int = 5
    verbose: bool = False

    # Debug options
    enable_blueprint_dump: bool = False  # Upload debug artifacts to MinIO

    # LTX fallback configuration
    enable_ltx_fallback: bool = False  # Use LTX-2 for non-CONTEXTUAL matches
    ltx_server_url: str = "http://gx10-cbc5:8082"
    ltx_timeout: float = 600.0


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


class LTXVideoGenerationRequest(BaseModel):
    """
    Request model for LTX-2 audio-conditioned video generation.

    Pipeline (when enable_audio=True):
      1. StoryGeneratorAgent → StoryOutput (N dialog lines)
      2. For each dialog line:  TTS via Fish-Speech → segment_NNN.wav
      3. For each dialog line:  WanGP LTX (audio_guide=wav) → segment_NNN.mp4
      4. ffmpeg concat → final video → MinIO upload

    Host rules (Docker Compose):
      - LTX / WanGP REST server: gx10-cbc5:8082  (ltx_server_url)
      - Fish-Speech TTS:          tts:8003         (tts_host / tts_port)
        The TTS service is named "tts" in compose.  Inside the stack it is
        NEVER reachable as "localhost" — always use the service name "tts".
        The FISH_TTS_HOST env var is set to "tts" by compose automatically.
    """

    # Input (mutually exclusive: title or story_text)
    title: Optional[str] = None
    story_text: Optional[str] = None

    # Story template (required - defines characters, prompt)
    story_template_id: str

    # WanGP LTX server — port 8082 on the remote GPU host
    ltx_server_url: str = "http://gx10-cbc5:8082"
    ltx_timeout: float = 600.0

    # Video generation parameters
    video_width: int = 1280
    video_height: int = 720
    video_duration_seconds: float = 5.0
    video_fps: int = 24
    video_steps: int = 20
    video_cfg_scale: float = 4.0
    video_seed: int = -1
    enable_audio: bool = True

    # TTS — Fish-Speech service, Docker Compose service name "tts" (NOT localhost)
    # Defaults driven by FISH_TTS_HOST / FISH_TTS_PORT env vars set in compose.
    tts_host: str = os.environ.get("FISH_TTS_HOST", "tts")
    tts_port: int = int(os.environ.get("FISH_TTS_PORT", "8003"))
    adapt_duration_to_audio: bool = True

    # Output configuration
    output_dir: Optional[str] = None
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting."

    # LLM configuration (for ADK agents)
    llm_provider: Optional[str] = "anthropic"
    llm_model: Optional[str] = "claude-sonnet-4-5-20250929"


class LTXVideoGenerationResponse(BaseModel):
    """Response model for LTX-2 video generation."""

    job_id: str
    status: str
    message: str


class GenerateFromBroadcastRequest(BaseModel):
    """Request model for video generation from active broadcast."""

    stream_id: str
    title: str
    user: Optional[str] = None

    # ADMIN FLAG: Bypass queue limit for batch operations.
    # When True, ignores MAX_PENDING_JOBS limit.
    # WARNING: For admin/batch use only - do not expose to end users.
    skip_queue_limit: bool = Field(
        default=False,
        description="[ADMIN] Bypass MAX_PENDING_JOBS queue limit. "
                    "For batch operations only - do not expose to end users.",
    )


class GenerateFromBroadcastResponse(BaseModel):
    """Response model for video generation from broadcast."""

    job_id: str
    status: str
    message: str
    story_template_id: str


class FeedbackRequest(BaseModel):
    """Request model for video feedback."""

    entry_id: str
    user: str
    feedback: str  # Raw user message


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
            story_output: StoryOutput = await run_story_generator(
                request.title, story_template_id=request.story_template_id
            )
            cli = MinIOClient(bucket="debug")
            json.dump(story_output.model_dump(), open("temp.json", "w"))
            await cli.upload_file("temp.json", f"story/{request.story_template_id}/{job_id}.json")

            logger.info(
                f"[Job {job_id}] Story generated: {story_output.title} "
                f"with {len(story_output.dialog)} dialog lines"
            )
            # Extract sentences from story output
            sentences = story_output.dialog

        elif request.story_text:
            # Parse story_text as DialogLines
            # For now, treat story_text as simple text to be split
            story_output: StoryOutput = await run_story_generator(
                request.story_text, story_template_id=request.story_template_id
            )

        # Step 2: Run SentenceVideoMatcher
        characters = {cid: await repo.get_character(cid) for cid in story_output.get_character_names()}
        characters = {
            cid: Character(
                character_id=character_data["character_id"],
                name=character_data["name"],
                description=character_data.get("description"),
                video_clip_path=character_data.get("video_clip_path", ""),
                voice_samples=[
                    VoiceSample(
                        sample_storage_path=s["sample_storage_path"],
                        transcript=s["transcript"],
                    )
                    for s in character_data.get("voice_samples", [])
                ],
                video_search_tag=character_data.get("video_search_tag"),
                identity_images=character_data.get("identity_images", []),
                created_at=character_data.get("created_at"),
                updated_at=character_data.get("updated_at"),
            )
            for cid, character_data in characters.items()
        }
        logger.info(f"[Job {job_id}] Running SentenceVideoMatcher...")
        video_matches = await run_sentence_video_matcher(sentences, collection, characters, config)
        logger.info(
            f"[Job {job_id}] Video matching complete: {len(video_matches.matches)} matches"
        )

        # Determine debug upload prefix if blueprint dump is enabled
        debug_upload_prefix = None
        if request.enable_blueprint_dump:
            debug_upload_prefix = f"debug/video-generation/{request.story_template_id}/{job_id}"
            logger.info(f"[Job {job_id}] Blueprint dump enabled, prefix: {debug_upload_prefix}")

            # Create and upload generation blueprint
            planned_tts = [
                {
                    "segment_index": i,
                    "character_id": match.dialog_line.character_id,
                    "text": match.dialog_line.text,
                    "scene_description": match.dialog_line.scene_description,
                }
                for i, match in enumerate(video_matches.matches)
            ]

            blueprint = GenerationBlueprint(
                timestamp=datetime.now().isoformat(),
                job_id=job_id,
                api_endpoint="video-generation",
                story_template_id=request.story_template_id,
                story_output=story_output,
                video_matches=[match.model_dump() for match in video_matches.matches],
                planned_tts=planned_tts,
                collection=collection,
            )

            # Upload blueprint to MinIO
            storage = get_storage_client()
            blueprint_key = blueprint.get_storage_path()
            await storage.put_json(blueprint_key, blueprint.model_dump())
            logger.info(f"[Job {job_id}] Blueprint uploaded: {blueprint_key}")

        # Step 3: Script to Video (TTS, Wav2Lip, STT via webservices)
        logger.info(f"[Job {job_id}] Running script_to_video...")

        # Build LTX config for fallback if enabled
        ltx_config = None
        if request.enable_ltx_fallback:
            ltx_config = LTXConfig(
                server_url=request.ltx_server_url,
                timeout=request.ltx_timeout,
            )
            logger.info(
                f"[Job {job_id}] LTX fallback enabled, server: {request.ltx_server_url}"
            )

        async with WebserviceClient(api_config) as client:
            final_video_path = await script_to_video(
                matches=video_matches.matches,
                client=client,
                config=config,
                ltx_config=ltx_config,
                enable_ltx_fallback=request.enable_ltx_fallback,
                progress_callback=lambda msg: logger.info(f"[Job {job_id}] {msg}"),
                debug_upload_prefix=debug_upload_prefix,
            )

        # Upload to MinIO storage
        # Path structure: generated_videos/{collection}/{job_id}.mp4
        logger.info(f"[Job {job_id}] Uploading to storage...")
        storage = get_storage_client()
        minio_video_key = f"generated_videos/{collection}/{job_id}.mp4"
        await storage.upload_file(final_video_path, minio_video_key)

        # Generate presigned URL for video access
        video_url = storage.get_url(minio_video_key)
        logger.info(f"[Job {job_id}] Video URL generated: {video_url[:80]}...")

        # Build metadata
        metadata = {
            "sentence_count": len(video_matches.matches),
            "total_duration": get_length(final_video_path),
            "timestamp": datetime.now().isoformat(),
            "minio_video_key": minio_video_key,
            "video_url": video_url,
        }

        # Add debug info if blueprint dump was enabled
        if debug_upload_prefix:
            metadata["debug_upload_prefix"] = debug_upload_prefix
            metadata["blueprint_key"] = f"{debug_upload_prefix}/blueprint.json"

        result = GenerationResult(
            video_path=final_video_path,
            config_dump_path=None,
            story_output=story_output,
            metadata=metadata,
        )

        # Job completed successfully
        result_data = {
            "video_path": result.video_path,
            "config_dump_path": result.config_dump_path,
            "metadata": result.metadata,
            "story_output": result.story_output.model_dump()
            if result.story_output
            else None,
            "video_matches": [match.model_dump() for match in video_matches.matches],
        }
        await job_store.update_job(job_id, status="completed", result=result_data)

    except Exception as e:
        # Job failed
        await job_store.update_job(job_id, status="failed", error=str(e))

        import traceback

        logger.error(f"Video generation job {job_id} failed:")
        traceback.print_exc()


async def _run_broadcast_generation(
        job_id: str,
        request: VideoGenerationRequest,
        programmation_id: str,
        user: str,
):
    """
    Broadcast workflow: generate video, then add to playlist.
    Wraps video generation with broadcast-specific post-processing.
    """
    # Step 1: Run video generation (reuse existing, unmodified function)
    await _run_video_generation(job_id, request)

    # Step 2: On success, add to playlist
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
        }
    )

    # Update job result with entry_id
    result["entry_id"] = entry.entry_id
    await job_store.update_job(job_id, result=result)
    logger.info(f"[Broadcast {job_id}] Added to playlist: {entry.entry_id}")


async def _run_ltx_video_generation(job_id: str, request: LTXVideoGenerationRequest):
    """Background task to run LTX-2 video generation."""
    job_store = await get_global_job_store()
    try:
        # Update job status to running
        await job_store.update_job(job_id, status="running")

        # Validate inputs - require title or story_text
        inputs = [request.title, request.story_text]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Exactly one of title or story_text must be provided"
            )

        # Build LTX Video API configuration
        ltx_config = LTXVideoConfig(
            server_url=request.ltx_server_url,
            timeout=request.ltx_timeout,
        )

        # Build video generation parameters
        video_params = VideoGenerationParams(
            prompt="",  # Will be set per segment
            width=request.video_width,
            height=request.video_height,
            duration_seconds=request.video_duration_seconds,
            fps=request.video_fps,
            steps=request.video_steps,
            cfg_scale=request.video_cfg_scale,
            seed=request.video_seed,
            enable_audio=request.enable_audio,
        )

        output_dir = request.output_dir or f"./output/ltx_{job_id}"

        story_output = None

        if request.title:
            # Step 1: Run StoryGeneratorAgent
            logger.info(f"[LTX Job {job_id}] Running StoryGeneratorAgent...")
            story_output = await run_story_generator(
                request.title, story_template_id=request.story_template_id
            )
            logger.info(
                f"[LTX Job {job_id}] Story generated: {story_output.title} "
                f"with {len(story_output.dialog)} dialog lines"
            )

        elif request.story_text:
            # Parse story_text via story generator
            story_output = await run_story_generator(
                request.story_text, story_template_id=request.story_template_id
            )

        # Step 2: Generate TTS audio per segment (when enable_audio=True)
        segment_audio_paths: Dict[int, str] = {}
        if request.enable_audio:
            logger.info(
                f"[LTX Job {job_id}] Generating TTS audio for "
                f"{len(story_output.dialog)} segments "
                f"(host={request.tts_host}:{request.tts_port})…"
            )
            tts_dir = os.path.join(output_dir, "tts")
            os.makedirs(tts_dir, exist_ok=True)
            storage_resolver = get_storage_resolver()

            for i, dialog_line in enumerate(story_output.dialog):
                try:
                    character = await load_character(dialog_line.character_id)
                    reference_audio: Optional[str] = None
                    reference_text: Optional[str] = None
                    if character.voice_samples:
                        sample = character.voice_samples[0]
                        reference_audio = await storage_resolver.resolve_file(
                            sample.sample_storage_path
                        )
                        reference_text = sample.transcript

                    audio_out = os.path.join(tts_dir, f"segment_{i:03d}.wav")
                    await txt_to_speech_call_fish_async(
                        speech_lines=dialog_line.text,
                        reference_audio=reference_audio,
                        reference_text=reference_text,
                        outpath=audio_out,
                        format="wav",
                        host=request.tts_host,
                        port=request.tts_port,
                    )
                    segment_audio_paths[i] = audio_out
                    logger.info(
                        f"[LTX Job {job_id}] TTS segment {i} "
                        f"({dialog_line.character_id}): {audio_out}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[LTX Job {job_id}] TTS failed for segment {i} "
                        f"({dialog_line.character_id}): {exc} — continuing without audio"
                    )

        # Step 3: Run story_to_video (LTX-2 audio-conditioned pipeline)
        logger.info(f"[LTX Job {job_id}] Running story_to_video with LTX-2…")

        def progress_callback(current: int, total: int, message: str):
            logger.info(f"[LTX Job {job_id}] Progress: {current}/{total} - {message}")

        result: StoryVideoResult = await story_to_video(
            story_output=story_output,
            ltx_config=ltx_config,
            video_params=video_params,
            output_dir=output_dir,
            progress_callback=progress_callback,
            style_suffix=request.style_suffix,
            segment_audio_paths=segment_audio_paths or None,
        )

        logger.info(
            f"[LTX Job {job_id}] Video generation complete: {result.final_video_path}"
        )

        # Upload to MinIO storage
        logger.info(f"[LTX Job {job_id}] Uploading to storage...")
        storage = get_storage_client()
        minio_video_key = f"generated_videos/ltx/{job_id}.mp4"
        await storage.upload_file(result.final_video_path, minio_video_key)

        # Generate presigned URL for video access
        video_url = storage.get_url(minio_video_key)
        logger.info(f"[LTX Job {job_id}] Video URL generated: {video_url[:80]}...")

        # Build result data
        result_data = {
            "video_path": result.final_video_path,
            "story_title": result.story_title,
            "total_duration_seconds": result.total_duration_seconds,
            "segment_count": len(result.segments),
            "segments": [
                {
                    "index": seg.index,
                    "video_path": seg.video_path,
                    "duration_seconds": seg.duration_seconds,
                    "character_id": seg.dialog_line.character_id,
                    "text": seg.dialog_line.text,
                    "scene_description": seg.dialog_line.scene_description,
                }
                for seg in result.segments
            ],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "minio_video_key": minio_video_key,
                "video_url": video_url,
                "pipeline": "ltx-2",
            },
            "story_output": story_output.model_dump() if story_output else None,
        }

        # Job completed successfully
        await job_store.update_job(job_id, status="completed", result=result_data)

    except Exception as e:
        # Job failed
        await job_store.update_job(job_id, status="failed", error=str(e))

        import traceback

        logger.error(f"LTX video generation job {job_id} failed:")
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


@router.post("/generate-ltx", response_model=LTXVideoGenerationResponse)
async def generate_video_ltx(
        request: LTXVideoGenerationRequest, background_tasks: BackgroundTasks
):
    """
    Generate video from title using LTX-2 for video+audio.

    This endpoint uses the LTX-2 text-to-video model via WanGP (Gradio API) to generate
    video segments with synchronized audio. The pipeline:

    1. StoryGeneratorAgent generates a story with DialogLines from the title
    2. For each DialogLine, LTX-2 generates a video segment with audio
    3. All segments are concatenated into the final video

    This is an alternative to the traditional pipeline that uses TTS + Wav2Lip.
    LTX-2 generates both video and audio together, resulting in more natural
    lip-sync and scene coherence.

    Args:
        request: LTXVideoGenerationRequest with title and configuration

    Returns:
        LTXVideoGenerationResponse with job_id for tracking
    """
    job_store = await get_global_job_store()

    # Create job
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {**request.model_dump(), "pipeline": "ltx-2"})

    # Start background task
    background_tasks.add_task(_run_ltx_video_generation, job_id, request)

    return LTXVideoGenerationResponse(
        job_id=job_id,
        status="pending",
        message="LTX-2 video generation job submitted successfully",
    )


# Maximum number of pending jobs allowed per story template
MAX_PENDING_JOBS = 5


@router.post("/generate-from-broadcast", response_model=GenerateFromBroadcastResponse)
async def generate_from_broadcast(
        request: GenerateFromBroadcastRequest, background_tasks: BackgroundTasks
):
    """
    Generate video from title using the active broadcast's story template.

    This endpoint is designed for Twitch chat integration. It:
    1. Gets the active programmation for the stream
    2. Uses the programmation's story_template_id for generation
    3. Enforces a queue limit of 5 pending jobs per story template

    Args:
        request: GenerateFromBroadcastRequest with stream_id, title, and optional user

    Returns:
        GenerateFromBroadcastResponse with job_id for tracking

    Raises:
        404: If stream not found or no active programmation
        429: If queue is full (>= 5 pending jobs)
    """
    from virtual_streamer.streaming.store import get_streaming_store

    # Get active programmation for the stream
    store = await get_streaming_store()
    stream = await store.get_stream(request.stream_id)
    if stream is None:
        raise HTTPException(
            status_code=404,
            detail=f"Stream '{request.stream_id}' not found"
        )

    # Get the active programmation
    from datetime import datetime
    current_time = datetime.now().time()
    programmation = await store.get_active_programmation(request.stream_id, current_time)

    if programmation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active programmation for stream '{request.stream_id}' at this time"
        )

    story_template_id = programmation.story_template_id

    # Check pending job count - UNLESS admin flag is set
    # NOTE: skip_queue_limit is for admin/batch use only.
    # It allows batch scripts to submit many jobs without hitting the queue limit.
    if not request.skip_queue_limit:
        job_store = await get_global_job_store()
        pending_count = await job_store.count_pending_jobs(story_template_id)

        if pending_count >= MAX_PENDING_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"Queue full: {pending_count} jobs pending for template '{story_template_id}'. Max is {MAX_PENDING_JOBS}.",
            )

    # Create the video generation request
    video_request = VideoGenerationRequest(
        title=request.title,
        story_template_id=story_template_id,
    )

    # Create job with metadata
    job_id = str(uuid.uuid4())
    job_data = video_request.model_dump()
    job_data["source"] = "broadcast"
    job_data["stream_id"] = request.stream_id
    job_data["programmation_id"] = programmation.programmation_id
    if request.user:
        job_data["user"] = request.user

    await job_store.create_job(job_id, job_data)

    # Start broadcast-specific background task (wraps video gen + playlist insert)
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
    """
    Store raw user feedback for a played video.
    Saves to MinIO at: feedback/{story_template_id}/{entry_id}.json
    """
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


# ============================================================================
# Single-Clip Generation (image + audio → video)
# ============================================================================

class SingleClipResponse(BaseModel):
    job_id: str
    status: str
    message: str


# Internal dataclass passed from the endpoint to the background task.
# Not a FastAPI model — just a plain container for already-read file bytes.
class _SingleClipJob:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        image_bytes: Optional[bytes],
        audio_bytes: Optional[bytes],
        wangp_url: str,
        wangp_timeout: float,
        model_type: str,
        resolution: str,
        duration_seconds: float,
        fps: int,
        steps: int,
        guidance_scale: float,
        flow_shift: float,
        seed: int,
        audio_scale: float,
        audio_guidance: float,
    ):
        self.prompt           = prompt
        self.negative_prompt  = negative_prompt
        self.image_bytes      = image_bytes
        self.audio_bytes      = audio_bytes
        self.wangp_url        = wangp_url
        self.wangp_timeout    = wangp_timeout
        self.model_type       = model_type
        self.resolution       = resolution
        self.duration_seconds = duration_seconds
        self.fps              = fps
        self.steps            = steps
        self.guidance_scale   = guidance_scale
        self.flow_shift       = flow_shift
        self.seed             = seed
        self.audio_scale      = audio_scale
        self.audio_guidance   = audio_guidance


async def _run_single_clip(job_id: str, job: _SingleClipJob) -> None:
    """Background task: write uploaded files to temp dir, generate video, store b64 result."""
    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path: Optional[str] = None
            audio_path: Optional[str] = None

            if job.image_bytes:
                image_path = os.path.join(tmpdir, "input_image.png")
                with open(image_path, "wb") as fh:
                    fh.write(job.image_bytes)

            if job.audio_bytes:
                audio_path = os.path.join(tmpdir, "input_audio.wav")
                with open(audio_path, "wb") as fh:
                    fh.write(job.audio_bytes)

            config = LTXVideoConfig(
                server_url=job.wangp_url,
                timeout=job.wangp_timeout,
            )
            params = VideoGenerationParams(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                image_path=image_path,
                audio_path=audio_path,
                model_type=job.model_type,
                resolution=job.resolution,
                duration_seconds=job.duration_seconds,
                fps=job.fps,
                steps=job.steps,
                guidance_scale=job.guidance_scale,
                flow_shift=job.flow_shift,
                seed=job.seed,
                audio_scale=job.audio_scale,
                audio_guidance=job.audio_guidance,
            )

            output_dir = os.path.join(tmpdir, "output")
            async with WanGPLTXClient(config) as client:
                result = await client.generate_video(params, output_dir=output_dir)

            with open(result.video_path, "rb") as fh:
                video_b64 = base64.b64encode(fh.read()).decode()

        await job_store.update_job(
            job_id,
            status="completed",
            result={
                "video_b64": video_b64,
                "duration_seconds": result.duration_seconds,
                "width": result.width,
                "height": result.height,
                "fps": result.fps,
                "prompt_id": result.prompt_id,
            },
        )

    except Exception as exc:
        logger.error(f"single-clip job {job_id} failed: {exc}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post("/single-clip", response_model=SingleClipResponse)
async def generate_single_clip(
    background_tasks: BackgroundTasks,
    prompt:           str          = Form(...),
    negative_prompt:  str          = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    wangp_url:        str          = Form("http://gx10-cbc5:8082"),
    wangp_timeout:    float        = Form(600.0),
    model_type:       str          = Form("ltx2_22B_distilled"),
    resolution:       str          = Form("1280x720"),
    duration_seconds: float        = Form(4.0),
    fps:              int          = Form(24),
    steps:            int          = Form(8),
    guidance_scale:   float        = Form(3.0),
    flow_shift:       float        = Form(3.0),
    seed:             int          = Form(-1),
    audio_scale:      float        = Form(1.0),
    audio_guidance:   float        = Form(4.5),
    image:            Optional[UploadFile] = File(default=None),
    audio:            Optional[UploadFile] = File(default=None),
):
    """
    Generate a single video clip from an optional conditioning image and/or audio.

    Accepts ``multipart/form-data`` — no base64 encoding required, no body-size issues.

    Modes:
    - Text-to-video:          prompt only, no image, no audio
    - Image-to-video (i2v):   prompt + image file
    - Audio-conditioned i2v:  prompt + image file + audio file

    The job runs asynchronously. Poll ``GET /api/v1/video-generation/jobs/{job_id}``
    until ``status == "completed"``.  The result contains ``video_b64``
    (base64-encoded MP4) plus duration/resolution metadata.
    """
    image_bytes = await image.read() if image else None
    audio_bytes = await audio.read() if audio else None

    job = _SingleClipJob(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        wangp_url=wangp_url,
        wangp_timeout=wangp_timeout,
        model_type=model_type,
        resolution=resolution,
        duration_seconds=duration_seconds,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
        seed=seed,
        audio_scale=audio_scale,
        audio_guidance=audio_guidance,
    )

    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {
        "prompt": prompt, "resolution": resolution,
        "duration_seconds": duration_seconds, "model_type": model_type,
    })
    background_tasks.add_task(_run_single_clip, job_id, job)
    return SingleClipResponse(
        job_id=job_id,
        status="pending",
        message="Single-clip generation job submitted",
    )
