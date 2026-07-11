"""
Medium-level API: Review actions — recompose a story and regenerate a scene.

These endpoints make a generated story re-composable after human review:

  POST /stories/{story_id}/recompose
      Rebuild the final video from the currently *selected* candidate of each
      scene (judge choice or human override). Background job.

  POST /stories/{story_id}/scenes/{scene_id}/regenerate
      Run a fresh seed hunt for one scene only (new takes, judged, persisted
      as candidates). Background job. Recompose afterwards to apply.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from virtual_streamer.utils.gpu_queue import enqueue_gpu_job, PRIORITY_INTERACTIVE
from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.story_repository import get_story_repository

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Review"])

_LTX_SERVER_URL = os.environ.get("LTX_SERVER_URL", "http://gx10-cbc5:8082")
_LTX_TIMEOUT = float(os.environ.get("LTX_TIMEOUT", "3600.0"))
_SD_URL = os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")


class RecomposeRequest(BaseModel):
    enable_subtitles: bool = False
    subtitle_fontsize: int = 14


class RegenerateRequest(BaseModel):
    max_candidates: int = Field(default=3, ge=1, le=10)
    accept_score: float = 7.5
    seeds: Optional[List[int]] = Field(
        default=None, description="Explicit seeds; random when null"
    )
    quality_preset: str = Field(default="fast", description="'fast', 'quality' or 'high_quality'")


class ReviewJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Recompose
# ---------------------------------------------------------------------------


async def _run_recompose(job_id: str, story_id: str, request: RecomposeRequest):
    from virtual_streamer.video_generation.story_to_video import concatenate_videos

    job_store = await get_global_job_store()
    repo = get_story_repository()
    storage = get_storage_client()
    try:
        await job_store.update_job(job_id, status="running")

        scenes = await repo.list_scenes_for_story(story_id)
        if not scenes:
            raise RuntimeError(f"Story {story_id} has no scenes")
        selected = {
            c["scene_id"]: c
            for c in await repo.get_selected_candidates_for_story(story_id)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            local_paths: List[str] = []
            composition: List[dict] = []
            for scene in scenes:
                cand = selected.get(scene["scene_id"])
                video_key = (cand or {}).get("video_key") or scene.get("video_segment_key")
                if not video_key:
                    raise RuntimeError(
                        f"Scene {scene['scene_index']} ({scene['scene_id']}) has no "
                        "candidate video nor segment key — cannot recompose"
                    )
                local = os.path.join(tmpdir, f"scene_{scene['scene_index']:03d}.mp4")
                await storage.download_file(video_key, local)
                local_paths.append(local)
                composition.append({
                    "scene_id": scene["scene_id"],
                    "scene_index": scene["scene_index"],
                    "video_key": video_key,
                    "candidate_id": (cand or {}).get("candidate_id"),
                    "seed": (cand or {}).get("seed"),
                    "selection_source": (cand or {}).get("selection_source"),
                })

            # Subtitles: each segment video carries its own generated speech
            # audio track, so transcribe the video itself and burn per segment
            # before concatenation (same flow as _apply_subtitles at generation).
            if request.enable_subtitles:
                import asyncio

                from virtual_streamer.utils.transcription import transcribe_to_srt
                from virtual_streamer.utils.utils import add_subtitle_from_srt

                subtitled: List[str] = []
                for idx, seg_path in enumerate(local_paths):
                    srt = os.path.join(tmpdir, f"seg_{idx:03d}.srt")
                    out = os.path.join(tmpdir, f"seg_{idx:03d}_sub.mp4")
                    try:
                        await asyncio.to_thread(transcribe_to_srt, seg_path, srt)
                        await asyncio.to_thread(
                            add_subtitle_from_srt, seg_path, srt, out,
                            fontsize=request.subtitle_fontsize,
                        )
                        subtitled.append(out)
                    except Exception as sub_exc:
                        logger.warning(
                            f"[recompose {job_id}] subtitle failed for segment {idx}: {sub_exc}"
                        )
                        subtitled.append(seg_path)
                local_paths = subtitled

            final_local = os.path.join(tmpdir, "recomposed.mp4")
            concatenate_videos(local_paths, final_local, tmpdir)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_key = f"generated_videos/ltx/recomposed_{story_id}_{ts}.mp4"
            await storage.upload_file(final_local, final_key)

        await repo.update_story_status(story_id, "COMPLETED", final_video_key=final_key)
        await job_store.update_job(
            job_id,
            status="completed",
            result={
                "story_id": story_id,
                "final_video_key": final_key,
                "video_url": storage.get_url(final_key),
                "composition": composition,
            },
        )
    except Exception as exc:
        logger.error(f"[recompose {job_id}] failed: {exc}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post("/stories/{story_id}/recompose", response_model=ReviewJobResponse)
async def recompose_story(
    story_id: str, request: RecomposeRequest, background_tasks: BackgroundTasks
):
    """Rebuild the final video from each scene's currently selected candidate."""
    repo = get_story_repository()
    story = await repo.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")

    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {"story_id": story_id, "pipeline": "recompose"})
    background_tasks.add_task(_run_recompose, job_id, story_id, request)
    return ReviewJobResponse(job_id=job_id, status="pending", message="Recompose job submitted")


# ---------------------------------------------------------------------------
# Regenerate one scene
# ---------------------------------------------------------------------------


async def _run_regenerate(job_id: str, story_id: str, scene_id: str, request: RegenerateRequest):
    from virtual_streamer.video_generation.ltx_client import (
        LTXVideoConfig,
        VideoGenerationParams,
        WanGPLTXClient,
    )
    from virtual_streamer.video_generation.scene_input import SceneInput
    from virtual_streamer.video_generation.seed_hunting import (
        SeedHuntConfig,
        hunt_segment,
        persist_candidates,
    )
    from virtual_streamer.video_generation.story_to_video import (
        generate_segment_from_input,
    )

    job_store = await get_global_job_store()
    repo = get_story_repository()
    storage = get_storage_client()
    try:
        await job_store.update_job(job_id, status="running")

        scene = await repo.get_scene(scene_id)
        raw = scene.get("raw_scene_data") or {}
        scene_input = SceneInput(
            scene_index=scene["scene_index"],
            ltx_prompt=scene["prompt"],
            speaker_id=scene.get("speaker_id"),
            spoken_line=scene.get("spoken_line"),
            location_id=scene.get("location_id"),
            character_ids_on_screen=raw.get("character_on_screen") or [],
            scene_visual_description=raw.get("scene_visual_description"),
            raw_scene_data=raw,
        )

        output_dir = f"./output/regen_{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Reuse the original conditioning image if one was persisted
        image_path = None
        try:
            artifacts = await repo.get_artifacts_for_scene(scene_id)
            if artifacts:
                key = artifacts[-1]["final_image_key"]
                image_path = os.path.join(output_dir, "conditioning.png")
                await storage.download_file(key, image_path)
        except Exception as exc:
            logger.warning(f"[regen {job_id}] no conditioning image reused: {exc}")
            image_path = None

        # Reuse the speaker's voice sample (talking-head mode) when available
        audio_path = None
        if scene_input.speaker_id:
            try:
                from virtual_streamer.utils.entity_repository import get_entity_repository
                char = await get_entity_repository().get_character(scene_input.speaker_id)
                samples = (char or {}).get("voice_samples") or []
                sample_key = samples[0].get("sample_storage_path") if samples else None
                if sample_key:
                    audio_path = os.path.join(output_dir, os.path.basename(sample_key))
                    await storage.download_file(sample_key, audio_path)
            except Exception as exc:
                logger.warning(f"[regen {job_id}] no voice sample reused: {exc}")
                audio_path = None

        params = VideoGenerationParams.from_preset(request.quality_preset, duration_seconds=5.0)
        config = LTXVideoConfig(server_url=_LTX_SERVER_URL, timeout=_LTX_TIMEOUT)
        hunt_config = SeedHuntConfig(
            max_candidates=request.max_candidates,
            accept_score=request.accept_score,
            seeds=request.seeds,
        )

        async with WanGPLTXClient(config) as client:

            async def _generate_take(seed: int):
                return await generate_segment_from_input(
                    client=client,
                    scene_input=scene_input,
                    output_dir=output_dir,
                    video_params=params.model_copy(update={"seed": seed}),
                    audio_path=audio_path,
                    image_path=image_path,
                )

            candidates = await hunt_segment(
                generate_fn=_generate_take,
                scene_input=scene_input,
                hunt_config=hunt_config,
            )

        await persist_candidates(
            story_repo=repo,
            storage=storage,
            db_scene_id=scene_id,
            candidates=candidates,
            minio_prefix=f"candidates/{story_id}/{scene_id}",
        )
        best = next(c for c in candidates if c.selected)
        # Make the new best the scene's selected take (unselects previous takes)
        await repo.set_selected_candidate(best.candidate_id, selection_source=best.selection_source)

        await job_store.update_job(
            job_id,
            status="completed",
            result={
                "story_id": story_id,
                "scene_id": scene_id,
                "selected_candidate_id": best.candidate_id,
                "candidates": [
                    {
                        "candidate_id": c.candidate_id,
                        "seed": c.seed,
                        "score": c.score,
                        "passed": c.passed,
                        "selected": c.selected,
                    }
                    for c in candidates
                ],
                "next_step": f"POST /api/v1/stories/{story_id}/recompose to rebuild the final video",
            },
        )
    except Exception as exc:
        logger.error(f"[regen {job_id}] failed: {exc}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post(
    "/stories/{story_id}/scenes/{scene_id}/regenerate", response_model=ReviewJobResponse
)
async def regenerate_scene(
    story_id: str,
    scene_id: str,
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
):
    """Run a fresh seed hunt for one scene; new takes are judged and persisted as candidates."""
    repo = get_story_repository()
    scene = await repo.get_scene(scene_id)
    if scene is None or scene["story_id"] != story_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")

    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(
        job_id, {"story_id": story_id, "scene_id": scene_id, "pipeline": "regenerate-scene"}
    )
    position = await enqueue_gpu_job(
        job_id,
        lambda: _run_regenerate(job_id, story_id, scene_id, request),
        priority=PRIORITY_INTERACTIVE,
    )
    return ReviewJobResponse(
        job_id=job_id, status="pending",
        message=f"Scene regeneration job queued (position {position})",
    )


# ---------------------------------------------------------------------------
# Backfill candidates for pre-seed-hunt stories
# ---------------------------------------------------------------------------


async def _run_backfill(job_id: str, story_id: str):
    """Create one candidate row (with judge verdict) per scene lacking candidates."""
    from virtual_streamer.agents.video_judge.agent import run_video_judge

    job_store = await get_global_job_store()
    repo = get_story_repository()
    storage = get_storage_client()
    try:
        await job_store.update_job(job_id, status="running")

        scenes = await repo.list_scenes_for_story(story_id)
        backfilled, skipped = [], []
        with tempfile.TemporaryDirectory() as tmpdir:
            for scene in scenes:
                if await repo.list_candidates_for_scene(scene["scene_id"]):
                    skipped.append(scene["scene_id"])
                    continue
                video_key = scene.get("video_segment_key")
                if not video_key:
                    skipped.append(scene["scene_id"])
                    continue

                local = os.path.join(tmpdir, f"{scene['scene_id']}.mp4")
                await storage.download_file(video_key, local)
                verdict = await run_video_judge(
                    local,
                    scene["prompt"]
                    + (f'\nSpoken line: "{scene["spoken_line"]}"' if scene.get("spoken_line") else ""),
                )

                candidate_id = str(uuid.uuid4())
                await repo.create_candidate(
                    candidate_id=candidate_id,
                    scene_id=scene["scene_id"],
                    seed=-1,  # original seed unknown for pre-seed-hunt segments
                    generation_params={"backfilled": True},
                    video_key=video_key,
                    judge_verdict=verdict.model_dump(),
                    duration_seconds=scene.get("duration_seconds"),
                    selected=True,
                    selection_source="fallback",
                )
                backfilled.append({
                    "scene_id": scene["scene_id"],
                    "candidate_id": candidate_id,
                    "judge_score": verdict.score,
                    "judge_passed": verdict.passed,
                })

        await job_store.update_job(
            job_id,
            status="completed",
            result={"story_id": story_id, "backfilled": backfilled, "skipped": skipped},
        )
    except Exception as exc:
        logger.error(f"[backfill {job_id}] failed: {exc}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post("/stories/{story_id}/backfill-candidates", response_model=ReviewJobResponse)
async def backfill_candidates(story_id: str, background_tasks: BackgroundTasks):
    """
    Make a pre-seed-hunt story reviewable: for each scene with a segment video
    but no candidates, create one selected candidate from the existing segment
    and judge it. Afterwards the story works in the review UI, select/feedback
    and recompose exactly like a seed-hunted one.
    """
    repo = get_story_repository()
    story = await repo.get_story(story_id)
    if story is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")

    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {"story_id": story_id, "pipeline": "backfill-candidates"})
    background_tasks.add_task(_run_backfill, job_id, story_id)
    return ReviewJobResponse(job_id=job_id, status="pending", message="Backfill job submitted")
