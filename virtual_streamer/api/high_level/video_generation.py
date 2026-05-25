"""
High-level API: Video Generation Application

Orchestrates ADK agents and the LTX-2 text-to-video model to produce complete
videos from a story title or text.

Architecture:
    1. StoryPipelineAgent (ADK) — 3-step: writer → recurrent_location_builder → detailed_scene_builder
    2. scenes_to_video — LTX-2 text-to-video per scene with audio conditioning
    3. Optional subtitle step — Whisper transcription → SRT → burned into segments
    4. ffmpeg concat → final video → MinIO upload
"""

import base64
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, List

from fastapi import APIRouter, File, Form, BackgroundTasks, UploadFile
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from virtual_streamer.agents.common.state_keys import (
    TITLE,
    STORY_TEMPLATE_ID,
    RECURRENT_LOCATIONS,
    DETAILED_SCENES,
    RAW_STORY_TEXT,
)
from virtual_streamer.agents.story_pipeline import get_story_pipeline
from virtual_streamer.agents.story_pipeline.schema import (
    get_recurrent_locations_from_state,
    get_detailed_scenes_from_state,
)
from virtual_streamer.api.high_level.models import (
    StoryPipelineResult,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoFromScriptRequest,
)
from virtual_streamer.image_generation.stable_cpp_client import (
    StableDiffusionCppClient,
    StableDiffusionCppConfig,
    Txt2ImageParams,
)
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.utils import add_subtitle_from_srt
from virtual_streamer.video_generation.ltx_client import (
    WanGPLTXClient,
    LTXVideoConfig,
    VideoGenerationParams,
    VIDEO_PRESETS,
    DEFAULT_NEGATIVE_PROMPT,
)
from virtual_streamer.video_generation.story_to_video import (
    scenes_to_video,
    StoryVideoResult,
    concatenate_videos,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-generation", tags=["Video Generation"])

APP_NAME = "virtual_streamer"

# Server-side LTX configuration — override via environment variables
_LTX_SERVER_URL = os.environ.get("LTX_SERVER_URL", "http://gx10-cbc5:8082")
_LTX_TIMEOUT = float(os.environ.get("LTX_TIMEOUT", "3600.0"))
_ENABLE_DEBUG = os.environ.get("ENABLE_DEBUG_DUMP", "true").lower() == "true"
SD_URL = os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")

# ============================================================================
# Story Pipeline Runner
# ============================================================================


async def run_story_pipeline(
    title: str,
    story_template_id: Optional[str] = None,
) -> StoryPipelineResult:
    """
    Run the 3-step StoryPipelineAgent and return typed outputs.

    State flow:
        story_writer → RAW_STORY_TEXT
        recurrent_location_builder → RECURRENT_LOCATIONS (JSON str)
        detailed_scene_builder → DETAILED_SCENES (JSON str)
    """
    story_generator = get_story_pipeline()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=story_generator,
        app_name=APP_NAME,
        session_service=session_service,
    )

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

    logger.info(f"Running StoryPipelineAgent (3-step) for title: {title!r}")
    content = types.Content(role="user", parts=[types.Part(text=title)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            logger.debug(f"Final response from {event.author}")

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    if not session.state.get(RECURRENT_LOCATIONS):
        raise RuntimeError("StoryPipelineAgent completed but RECURRENT_LOCATIONS not in state")
    if not session.state.get(DETAILED_SCENES):
        raise RuntimeError("StoryPipelineAgent completed but DETAILED_SCENES not in state")

    recurrent_locations = get_recurrent_locations_from_state(session.state)
    detailed_scenes = get_detailed_scenes_from_state(session.state)

    return StoryPipelineResult(
        recurrent_locations=recurrent_locations,
        detailed_scenes=detailed_scenes,
        title=detailed_scenes.title,
        raw_story_text=session.state.get(RAW_STORY_TEXT),
    )


# ============================================================================
# Subtitle Helper
# ============================================================================


def _apply_subtitles(
    result: StoryVideoResult,
    output_dir: str,
    fontsize: int,
) -> str:
    """
    Burn subtitles into each segment that has TTS audio, then re-concatenate.
    Returns the path to the new final video.
    """
    from virtual_streamer.utils.transcription import transcribe_to_srt

    sub_dir = os.path.join(output_dir, "subtitles")
    temp_dir = os.path.join(sub_dir, "temp")
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    subtitled: List[str] = []

    for seg in result.segments:
        if seg.audio_path and os.path.exists(seg.audio_path):
            srt = os.path.join(sub_dir, f"seg_{seg.index:03d}.srt")
            out = os.path.join(sub_dir, f"seg_{seg.index:03d}.mp4")
            transcribe_to_srt(seg.audio_path, srt)
            add_subtitle_from_srt(seg.video_path, srt, out, fontsize=fontsize)
            subtitled.append(out)
        else:
            subtitled.append(seg.video_path)

    sub_final = os.path.join(output_dir, "final_with_subtitles.mp4")
    concatenate_videos(subtitled, sub_final, temp_dir)
    return sub_final


# ============================================================================
# Background Tasks
# ============================================================================


async def _run_video_generation(job_id: str, request: VideoGenerationRequest):
    """Background task: LTX-2 video generation from title or story text."""
    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")

        inputs = [request.title, request.story_text]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError("Exactly one of title or story_text must be provided")

        ltx_config = LTXVideoConfig(server_url=_LTX_SERVER_URL, timeout=_LTX_TIMEOUT)
        video_params = request.to_video_params()
        output_dir = request.output_dir or f"./output/ltx_{job_id}"
        input_title = request.title or request.story_text

        # Step 1: 3-step story pipeline
        logger.info(f"[Job {job_id}] Running 3-step StoryPipelineAgent...")
        pipeline_result: StoryPipelineResult = await run_story_pipeline(
            title=input_title,
            story_template_id=request.story_template_id,
        )
        story_title = pipeline_result.title or input_title or "story"
        scenes = pipeline_result.detailed_scenes.scenes
        locations = pipeline_result.recurrent_locations.locations
        logger.info(
            f"[Job {job_id}] Pipeline complete: title={story_title!r} "
            f"scenes={len(scenes)} locations={len(locations)}"
        )

        # Step 2: Generate location base images (Flux + MinIO upload) for non-existing locations
        if locations:
            logger.info(f"[Job {job_id}] Generating base images for {len(locations)} location(s)...")
            from virtual_streamer.utils.entity_repository import get_entity_repository

            repo = get_entity_repository()
            storage = get_storage_client()
            sd_config = StableDiffusionCppConfig(server_url=SD_URL)
            loc_image_dir = os.path.join(output_dir, "location_images")
            os.makedirs(loc_image_dir, exist_ok=True)

            for loc in locations:
                try:
                    existing = await repo.get_location(loc.location_id)
                    if not existing:
                        prompt = loc.flux_prompt.to_prompt()
                        async with StableDiffusionCppClient(sd_config) as sd_client:
                            img_result = await sd_client.txt2image(
                                Txt2ImageParams(
                                    prompt=prompt + ", no people, cinematic composition, photorealistic, high quality",
                                    negative_prompt="text, watermark, blurry, distorted, people, persons, characters",
                                    width=VideoGenerationRequest.video_width,
                                    height=VideoGenerationRequest.video_height,
                                ),
                                output_dir=loc_image_dir,
                            )
                        minio_key = f"locations/{request.story_template_id}/{loc.location_id}.png"
                        await storage.upload_file(img_result.image_path, minio_key)
                        await repo.create_location(
                            location_id=loc.location_id,
                            name=loc.name,
                            description=loc.flux_prompt.to_prompt(),
                            story_template_id=request.story_template_id,
                            image_path=minio_key,
                        )
                        await repo.update_location_image(loc.location_id, minio_key)
                    else:
                        logger.info(f"[Job {job_id}] Location '{loc.location_id}' already exists")
                except Exception as loc_exc:
                    logger.warning(f"[Job {job_id}] Failed to create location '{loc.location_id}': {loc_exc}")

        # Step 3: scenes_to_video — audio sourced automatically from character voice samples
        logger.info(f"[Job {job_id}] Running scenes_to_video with LTX-2 (talking-head A1O)...")

        debug_prefix = None
        if _ENABLE_DEBUG:
            ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
            debug_prefix = f"ltx/{request.story_template_id}/{ts}-{job_id}"
            logger.info(f"[Job {job_id}] Debug dump enabled → debug/{debug_prefix}/")

        from virtual_streamer.utils.story_repository import get_story_repository

        _story_repo = None
        _db_story_id = None
        if request.story_template_id:
            _story_repo = get_story_repository()
            _db_story_id = str(uuid.uuid4())
            try:
                await _story_repo.create_story(
                    story_id=_db_story_id,
                    story_template_id=request.story_template_id,
                    title=story_title,
                    story_plan=pipeline_result.raw_story_text or "",
                    raw_agent_output={
                        "recurrent_locations": [l.model_dump() for l in locations],
                        "detailed_scenes": [s.model_dump() for s in scenes],
                        "title": story_title,
                    },
                    status="GENERATING",
                )
                logger.info(f"[Job {job_id}] Story row created: {_db_story_id}")
            except Exception as db_exc:
                logger.warning(f"[Job {job_id}] Failed to persist Story to DB: {db_exc}")
                _db_story_id = None

        def progress_callback(current: int, total: int, message: str):
            logger.info(f"[Job {job_id}] Progress: {current}/{total} - {message}")

        result: StoryVideoResult = await scenes_to_video(
            scenes=scenes,
            story_title=story_title,
            ltx_config=ltx_config,
            video_params=video_params,
            output_dir=output_dir,
            story_template_id=request.story_template_id,
            sd_server_url=SD_URL,
            progress_callback=progress_callback,
            debug_minio_prefix=debug_prefix,
            story_repo=_story_repo,
            db_story_id=_db_story_id,
        )

        # Step 4 (optional): Burn subtitles per segment, then re-concatenate
        final_video_path = result.final_video_path
        if request.enable_subtitles and result.segments:
            logger.info(f"[Job {job_id}] Adding subtitles...")
            final_video_path = _apply_subtitles(result, output_dir, request.subtitle_fontsize)

        logger.info(f"[Job {job_id}] Video generation complete: {final_video_path}")

        # Upload to MinIO
        storage = get_storage_client()
        minio_video_key = f"generated_videos/ltx/{job_id}.mp4"
        await storage.upload_file(final_video_path, minio_video_key)
        video_url = storage.get_url(minio_video_key)
        logger.info(f"[Job {job_id}] Video URL: {video_url[:80]}...")

        result_data = {
            "video_path": final_video_path,
            "story_title": result.story_title,
            "total_duration_seconds": result.total_duration_seconds,
            "segment_count": len(result.segments),
            "segments": [
                {
                    "index": seg.index,
                    "video_path": seg.video_path,
                    "duration_seconds": seg.duration_seconds,
                    "speaker_id": seg.scene_input.speaker_id if seg.scene_input else None,
                    "spoken_line": seg.scene_input.spoken_line if seg.scene_input else None,
                    "location": seg.scene_input.location_id if seg.scene_input else None,
                    "ltx_prompt": seg.scene_input.ltx_prompt if seg.scene_input else None,
                    "minio_video_key": seg.minio_video_key,
                    "minio_audio_key": seg.minio_audio_key,
                    "minio_image_key": seg.minio_image_key,
                }
                for seg in result.segments
            ],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "minio_video_key": minio_video_key,
                "video_url": video_url,
                "pipeline": "ltx-2-scenes",
                "debug_minio_prefix": result.debug_minio_prefix,
                "minio_manifest_key": result.minio_manifest_key,
                "minio_debug_final_key": result.minio_final_video_key,
            },
            "recurrent_locations": [loc.model_dump() for loc in locations] if pipeline_result else None,
            "detailed_scenes": [s.model_dump() for s in scenes] if pipeline_result else None,
        }

        await job_store.update_job(job_id, status="completed", result=result_data)

    except Exception as e:
        await job_store.update_job(job_id, status="failed", error=str(e))
        import traceback
        logger.error(f"Video generation job {job_id} failed:")
        traceback.print_exc()


async def _run_from_script(job_id: str, request: VideoFromScriptRequest):
    """
    Background task: generate LTX video from a pre-built (user-edited) script.

    Skips the story pipeline — scenes and locations come directly from the request.
    Steps: location images → TTS → scenes_to_video → optional subtitles → MinIO upload.
    """
    from virtual_streamer.agents.story_pipeline.schema import DetailedScene, RecurrentLocation

    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")

        scenes = [DetailedScene.model_validate(s) for s in request.scenes]
        locations = [RecurrentLocation.model_validate(loc) for loc in request.locations]
        story_title = request.story_title

        ltx_config = LTXVideoConfig(server_url=_LTX_SERVER_URL, timeout=_LTX_TIMEOUT)
        video_params = request.to_video_params()
        output_dir = request.output_dir or f"./output/ltx_{job_id}"
        _sd_url = SD_URL

        logger.info(
            f"[FromScript {job_id}] title={story_title!r} "
            f"scenes={len(scenes)} locations={len(locations)}"
        )

        # Step 1: Upsert location entities with generated base images
        if locations:
            logger.info(f"[FromScript {job_id}] Generating base images for {len(locations)} location(s)...")
            from virtual_streamer.utils.entity_repository import get_entity_repository

            repo = get_entity_repository()
            storage = get_storage_client()
            sd_config = StableDiffusionCppConfig(server_url=_sd_url)
            loc_image_dir = os.path.join(output_dir, "location_images")
            os.makedirs(loc_image_dir, exist_ok=True)

            for loc in locations:
                try:
                    prompt = loc.flux_prompt.to_prompt()
                    async with StableDiffusionCppClient(sd_config) as sd_client:
                        img_result = await sd_client.txt2image(
                            Txt2ImageParams(
                                prompt=prompt + ", no people, cinematic composition, photorealistic, high quality",
                                negative_prompt="text, watermark, blurry, distorted, people, persons, characters",
                                width=video_params.width,
                                height=video_params.height,
                            ),
                            output_dir=loc_image_dir,
                        )
                    minio_key = f"locations/{request.story_template_id}/{loc.location_id}.png"
                    await storage.upload_file(img_result.image_path, minio_key)
                    existing = await repo.get_location(loc.location_id)
                    if existing:
                        await repo.update_location_image(loc.location_id, minio_key)
                    else:
                        await repo.create_location(
                            location_id=loc.location_id,
                            name=loc.name,
                            description=loc.flux_prompt.to_prompt(),
                            story_template_id=request.story_template_id,
                            image_path=minio_key,
                        )
                    logger.info(f"[FromScript {job_id}] Location '{loc.location_id}' → {minio_key}")
                except Exception as loc_exc:
                    logger.warning(f"[FromScript {job_id}] Location '{loc.location_id}' failed: {loc_exc}")

        # Step 2: scenes_to_video — audio sourced automatically from character voice samples
        debug_prefix = None
        if _ENABLE_DEBUG:
            ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
            debug_prefix = f"ltx/{request.story_template_id}/{ts}-{job_id}"

        from virtual_streamer.utils.story_repository import get_story_repository

        _story_repo = None
        _db_story_id = None
        if request.story_template_id:
            _story_repo = get_story_repository()
            _db_story_id = str(uuid.uuid4())
            try:
                await _story_repo.create_story(
                    story_id=_db_story_id,
                    story_template_id=request.story_template_id,
                    title=story_title,
                    story_plan="",
                    raw_agent_output={
                        "scenes": [s.model_dump() for s in scenes],
                        "locations": [l.model_dump() for l in locations],
                    },
                    status="GENERATING",
                )
                logger.info(f"[FromScript {job_id}] Story row created: {_db_story_id}")
            except Exception as db_exc:
                logger.warning(f"[FromScript {job_id}] Failed to persist Story to DB: {db_exc}")
                _db_story_id = None

        def progress_callback(current: int, total: int, message: str):
            logger.info(f"[FromScript {job_id}] {current}/{total} — {message}")

        result: StoryVideoResult = await scenes_to_video(
            scenes=scenes,
            story_title=story_title,
            ltx_config=ltx_config,
            video_params=video_params,
            output_dir=output_dir,
            story_template_id=request.story_template_id,
            sd_server_url=_sd_url,
            progress_callback=progress_callback,
            debug_minio_prefix=debug_prefix,
            story_repo=_story_repo,
            db_story_id=_db_story_id,
        )

        # Step 3 (optional): Burn subtitles per segment, then re-concatenate
        final_video_path = result.final_video_path
        if request.enable_subtitles and result.segments:
            logger.info(f"[FromScript {job_id}] Adding subtitles...")
            final_video_path = _apply_subtitles(result, output_dir, request.subtitle_fontsize)

        # Upload final video
        storage = get_storage_client()
        minio_video_key = f"generated_videos/ltx/{job_id}.mp4"
        await storage.upload_file(final_video_path, minio_video_key)
        video_url = storage.get_url(minio_video_key)

        result_data = {
            "video_path": final_video_path,
            "story_title": result.story_title,
            "total_duration_seconds": result.total_duration_seconds,
            "segment_count": len(result.segments),
            "segments": [
                {
                    "index": seg.index,
                    "video_path": seg.video_path,
                    "duration_seconds": seg.duration_seconds,
                    "speaker_id": seg.scene_input.speaker_id if seg.scene_input else None,
                    "spoken_line": seg.scene_input.spoken_line if seg.scene_input else None,
                    "location": seg.scene_input.location_id if seg.scene_input else None,
                    "ltx_prompt": seg.scene_input.ltx_prompt if seg.scene_input else None,
                    "minio_video_key": seg.minio_video_key,
                    "minio_audio_key": seg.minio_audio_key,
                    "minio_image_key": seg.minio_image_key,
                }
                for seg in result.segments
            ],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "minio_video_key": minio_video_key,
                "video_url": video_url,
                "pipeline": "ltx-2-from-script",
                "debug_minio_prefix": result.debug_minio_prefix,
                "minio_manifest_key": result.minio_manifest_key,
            },
            "recurrent_locations": [loc.model_dump() for loc in locations],
            "detailed_scenes": [s.model_dump() for s in scenes],
        }

        await job_store.update_job(job_id, status="completed", result=result_data)

    except Exception as e:
        await job_store.update_job(job_id, status="failed", error=str(e))
        import traceback
        logger.error(f"LTX-from-script job {job_id} failed:")
        traceback.print_exc()


# ============================================================================
# Single-Clip Generation (image + audio → video)
# ============================================================================


class SingleClipResponse(BaseModel):
    job_id: str
    status: str
    message: str


# Internal container passed from endpoint to background task.
# Not a FastAPI model — stores already-read file bytes.
class _SingleClipJob:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        image_bytes: Optional[bytes],
        audio_bytes: Optional[bytes],
        video_bytes: Optional[bytes],
        end_image_bytes: Optional[bytes],
        wangp_url: str,
        wangp_timeout: float,
        model_type: str,
        resolution: str,
        duration_seconds: float,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        flow_shift: float,
        seed: int,
        audio_scale: float,
        audio_guidance_scale: float,
        denoising_strength: float,
        video_prompt_type: str,
        image_prompt_type: str,
        audio_prompt_type: str,
        keep_frames_video_guide: Optional[str] = None,
        activated_loras: Optional[List[str]] = None,
        loras_multipliers: str = "",
        # Identity reference conditioning ("I" mode)
        image_ref_bytes: Optional[bytes] = None,
        remove_background_images_ref: Optional[int] = None,
        sample_solver: Optional[str] = None,
        guidance_phases: Optional[int] = None,
    ):
        self.prompt                      = prompt
        self.negative_prompt             = negative_prompt
        self.image_bytes                 = image_bytes
        self.audio_bytes                 = audio_bytes
        self.video_bytes                 = video_bytes
        self.end_image_bytes             = end_image_bytes
        self.wangp_url                   = wangp_url
        self.wangp_timeout               = wangp_timeout
        self.model_type                  = model_type
        self.resolution                  = resolution
        self.duration_seconds            = duration_seconds
        self.fps                         = fps
        self.num_inference_steps         = num_inference_steps
        self.guidance_scale              = guidance_scale
        self.flow_shift                  = flow_shift
        self.seed                        = seed
        self.audio_scale                 = audio_scale
        self.audio_guidance_scale        = audio_guidance_scale
        self.denoising_strength          = denoising_strength
        self.video_prompt_type           = video_prompt_type
        self.image_prompt_type           = image_prompt_type
        self.audio_prompt_type           = audio_prompt_type
        self.keep_frames_video_guide     = keep_frames_video_guide
        self.activated_loras             = activated_loras or []
        self.loras_multipliers           = loras_multipliers
        self.image_ref_bytes             = image_ref_bytes
        self.remove_background_images_ref = remove_background_images_ref
        self.sample_solver               = sample_solver
        self.guidance_phases             = guidance_phases


async def _run_single_clip(job_id: str, job: _SingleClipJob) -> None:
    """Background task: write uploaded files to temp dir, generate video, store b64 result."""
    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path: Optional[str] = None
            audio_path: Optional[str] = None
            video_path: Optional[str] = None
            end_image_path: Optional[str] = None
            image_ref_path: Optional[str] = None

            if job.video_bytes:
                video_path = os.path.join(tmpdir, "input_video.mp4")
                with open(video_path, "wb") as fh:
                    fh.write(job.video_bytes)

            if job.image_bytes:
                image_path = os.path.join(tmpdir, "input_image.png")
                with open(image_path, "wb") as fh:
                    fh.write(job.image_bytes)

            if job.audio_bytes:
                audio_path = os.path.join(tmpdir, "input_audio.wav")
                with open(audio_path, "wb") as fh:
                    fh.write(job.audio_bytes)

            if job.end_image_bytes:
                end_image_path = os.path.join(tmpdir, "input_end_image.png")
                with open(end_image_path, "wb") as fh:
                    fh.write(job.end_image_bytes)

            if job.image_ref_bytes:
                image_ref_path = os.path.join(tmpdir, "input_ref_image.png")
                with open(image_ref_path, "wb") as fh:
                    fh.write(job.image_ref_bytes)

            config = LTXVideoConfig(server_url=job.wangp_url, timeout=job.wangp_timeout)
            params = VideoGenerationParams(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                image_start=image_path,
                audio_guide=audio_path,
                video_guide=video_path,
                image_end=end_image_path,
                image_refs=[image_ref_path] if image_ref_path else [],
                model_type=job.model_type,
                resolution=job.resolution,
                duration_seconds=job.duration_seconds,
                fps=job.fps,
                num_inference_steps=job.num_inference_steps,
                guidance_scale=job.guidance_scale,
                flow_shift=job.flow_shift,
                seed=job.seed,
                audio_scale=job.audio_scale,
                audio_guidance_scale=job.audio_guidance_scale,
                denoising_strength=job.denoising_strength,
                video_prompt_type=job.video_prompt_type,
                image_prompt_type=job.image_prompt_type,
                audio_prompt_type=job.audio_prompt_type,
                keep_frames_video_guide=job.keep_frames_video_guide,
                activated_loras=job.activated_loras,
                loras_multipliers=job.loras_multipliers,
                remove_background_images_ref=job.remove_background_images_ref,
                sample_solver=job.sample_solver,
                guidance_phases=job.guidance_phases,
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


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate video from a title using the LTX-2 pipeline.

    Runs asynchronously — poll ``GET /api/v1/jobs/{job_id}`` for status.

    Pipeline:
    1. 3-step StoryPipelineAgent → scenes + locations
    2. Location base images via Flux (Stable Diffusion)
    3. TTS per scene (Fish-Speech)
    4. LTX-2 text-to-video per scene (audio-conditioned)
    5. Optional: Whisper transcription → SRT subtitles burned into segments
    6. ffmpeg concat → final video → MinIO upload
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {**request.model_dump(), "pipeline": "ltx-2"})
    background_tasks.add_task(_run_video_generation, job_id, request)
    return VideoGenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation job submitted successfully",
    )


@router.post("/generate-from-script", response_model=VideoGenerationResponse)
async def generate_video_from_script(
    request: VideoFromScriptRequest, background_tasks: BackgroundTasks
):
    """
    Generate LTX video from a pre-built (optionally user-edited) script.

    Skips the story pipeline — scenes and locations are taken directly from the
    request (previously generated by ``POST /story-pipeline/run``).
    """
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(
        job_id,
        {
            "story_title": request.story_title,
            "story_template_id": request.story_template_id,
            "scene_count": len(request.scenes),
            "pipeline": "ltx-2-from-script",
        },
    )
    background_tasks.add_task(_run_from_script, job_id, request)
    return VideoGenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation (from script) job submitted successfully",
    )


@router.get("/health")
async def health():
    """Health check for video generation service."""
    job_store = await get_global_job_store()
    jobs = await job_store.list_jobs(limit=1000)
    active_count = sum(1 for j in jobs if j["status"] in ["pending", "running"])
    return {
        "status": "healthy",
        "active_jobs": active_count,
        "total_jobs": len(jobs),
        "storage_backend": os.environ.get("JOB_STORAGE_BACKEND", "memory"),
        "ltx_server_url": _LTX_SERVER_URL,
    }


@router.post("/single-clip", response_model=SingleClipResponse)
async def generate_single_clip(
    background_tasks: BackgroundTasks,
    prompt:                str           = Form(...),
    negative_prompt:       str           = Form(DEFAULT_NEGATIVE_PROMPT),
    wangp_url:             str           = Form("http://gx10-cbc5:8082"),
    wangp_timeout:         float         = Form(600.0),
    quality_preset:        Optional[str] = Form(None, description="Named preset: 'fast', 'quality', 'high_quality'."),
    model_type:            str           = Form("ltx2_22B_distilled_1_1"),
    resolution:            str           = Form("1280x720"),
    duration_seconds:      float         = Form(4.0),
    fps:                   int           = Form(24),
    num_inference_steps:   int           = Form(8),
    guidance_scale:        float         = Form(1.0),
    flow_shift:            float         = Form(5.0),
    seed:                  int           = Form(-1),
    audio_scale:           float         = Form(1.0),
    audio_guidance_scale:  float         = Form(7.0),
    denoising_strength:    float         = Form(1.0),
    video_prompt_type:     str           = Form("", description="V2V / identity flags, e.g. 'DVG', 'PVG', 'I'. Empty = unused."),
    image_prompt_type:     str           = Form("", description="Image conditioning flags: 'S', 'SE', 'E'. Empty = unused."),
    audio_prompt_type:     str           = Form("", description="Audio flags: 'A', 'A1O'. Empty = unused."),
    keep_frames_video_guide: Optional[str] = Form(None, description="Blank first N guide frames, e.g. '17:-1'."),
    activated_loras:       str           = Form("", description="Comma-separated LoRA filenames."),
    loras_multipliers:     str           = Form("", description="Space-separated multipliers for activated_loras."),
    remove_background_images_ref: Optional[int] = Form(None, description="Strip background from image_refs (0=off, 1=on, None=server default)."),
    sample_solver:         Optional[str] = Form(None, description="Solver type, e.g. 'distilled_8_steps'."),
    guidance_phases:       Optional[int] = Form(None, description="Number of guidance phases (1 or 2)."),
    image:                 Optional[UploadFile] = File(default=None, description="Start/first-frame image (also set image_prompt_type='S')."),
    audio:                 Optional[UploadFile] = File(default=None, description="Audio conditioning file (also set audio_prompt_type='A')."),
    video:                 Optional[UploadFile] = File(default=None, description="Source video for V2V (also set video_prompt_type)."),
    end_image:             Optional[UploadFile] = File(default=None, description="End/last-frame conditioning image (also set image_prompt_type containing 'E')."),
    image_ref:             Optional[UploadFile] = File(default=None, description="Identity reference image (set video_prompt_type='I')."),
):
    """
    Generate a single video clip via POST /jobs/raw.

    Accepts ``multipart/form-data``.  All conditioning flags must be set explicitly:

    - Text-to-video:               prompt only
    - Image-to-video:              image + image_prompt_type="S"
    - Audio-conditioned i2v:       image + audio + image_prompt_type="S" + audio_prompt_type="A"
    - Video-to-video (depth):      video + video_prompt_type="DVG"
    - Identity reference:          image_ref + video_prompt_type="I"
    - Talking head (ID-LoRA):      image + audio + image_prompt_type="S" + audio_prompt_type="A1O"

    Quality presets (``quality_preset``): ``fast`` · ``quality`` · ``high_quality``.

    Poll ``GET /api/v1/jobs/{job_id}`` for status. The completed result contains
    ``video_b64`` (base64-encoded MP4) and duration/resolution metadata.
    """
    if quality_preset and quality_preset in VIDEO_PRESETS:
        preset = VIDEO_PRESETS[quality_preset]
        if model_type == "ltx2_22B_distilled_1_1":
            model_type = preset.get("model_type", model_type)
        if num_inference_steps == 8:
            num_inference_steps = int(preset.get("num_inference_steps", num_inference_steps))

    image_bytes     = await image.read()     if image     else None
    audio_bytes     = await audio.read()     if audio     else None
    video_bytes     = await video.read()     if video     else None
    end_image_bytes = await end_image.read() if end_image else None
    image_ref_bytes = await image_ref.read() if image_ref else None

    # Parse comma-separated LoRA list
    lora_list = [l.strip() for l in activated_loras.split(",") if l.strip()]

    job = _SingleClipJob(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        video_bytes=video_bytes,
        end_image_bytes=end_image_bytes,
        wangp_url=wangp_url,
        wangp_timeout=wangp_timeout,
        model_type=model_type,
        resolution=resolution,
        duration_seconds=duration_seconds,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
        seed=seed,
        audio_scale=audio_scale,
        audio_guidance_scale=audio_guidance_scale,
        denoising_strength=denoising_strength,
        video_prompt_type=video_prompt_type,
        image_prompt_type=image_prompt_type,
        audio_prompt_type=audio_prompt_type,
        keep_frames_video_guide=keep_frames_video_guide,
        activated_loras=lora_list,
        loras_multipliers=loras_multipliers,
        image_ref_bytes=image_ref_bytes,
        remove_background_images_ref=remove_background_images_ref,
        sample_solver=sample_solver,
        guidance_phases=guidance_phases,
    )

    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(
        job_id,
        {"prompt": prompt, "resolution": resolution, "duration_seconds": duration_seconds, "model_type": model_type},
    )
    background_tasks.add_task(_run_single_clip, job_id, job)
    return SingleClipResponse(
        job_id=job_id,
        status="pending",
        message="Single-clip generation job submitted",
    )
