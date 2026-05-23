"""
Story-to-Video Pipeline

Converts a StoryOutput into a final video by:
  1. Generating one LTX video segment per DialogLine (via WanGP REST API).
  2. Concatenating all segments with ffmpeg.

When *segment_audio_paths* is supplied, each segment uses LTX audio-conditioned
generation (audio_guide + audio_prompt_type="A") and its duration is adapted to
the TTS audio length (snapped to the nearest valid 8n+1 frame count).

Usage:
    from virtual_streamer.video_generation.story_to_video import story_to_video

    result = await story_to_video(
        story_output=story,
        ltx_config=LTXVideoConfig(server_url="http://gx10-cbc5:8082"),
        segment_audio_paths={0: "/tmp/tts_000.wav", 1: "/tmp/tts_001.wav"},
        output_dir="./output",
    )
    print(result.final_video_path)       # concatenated final video
    print(len(result.segments))          # number of segments successfully generated

Important host rule:
    The WanGP REST server is on port 8082 (gx10-cbc5:8082).
    The Fish-Speech TTS service is reached via the Docker Compose service name
    "tts" on port 8003 — NOT "localhost" — when running inside the compose stack.
    Both defaults are driven by env vars FISH_TTS_HOST / FISH_TTS_PORT and
    the LTXVideoConfig.server_url field.
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from virtual_streamer.agents.scene_enricher.agent import run_scene_enricher
from virtual_streamer.video_generation.ltx_client import (
    WanGPLTXClient,
    LTXVideoConfig,
    VideoGenerationParams,
)
from virtual_streamer.video_generation.ltx_prompt_builder import (
    build_ltx_prompt,
    build_negative_prompt,
)
from virtual_streamer.video_generation.scene_input import (
    SceneInput,
    StoryInput,
    DetailedSceneInput,
    DialogLineInput,
)
from virtual_streamer.image_generation.stable_cpp_client import (
    StableDiffusionCppClient,
    StableDiffusionCppConfig,
    Txt2ImageParams,
    ImageEditParams,
)
from virtual_streamer.utils.minio_client import get_storage_client

logger = logging.getLogger(__name__)

# Quality keywords appended to every location identity image prompt.
# Tune these to steer the default aesthetic of all generated location images.
LOCATION_IMAGE_QUALITY_KEYWORDS = (
    "cinematic composition, photorealistic, high quality, detailed environment, "
    "dramatic lighting, sharp focus, 8k"
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

class SegmentResult(BaseModel):
    """Result of generating a single video segment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    video_path: str
    duration_seconds: float
    prompt_id: str
    scene_input: Optional[SceneInput] = None   # stable abstraction (new pipeline)
    dialog_line: Optional[Any] = None           # backward compat — legacy pipeline
    scene: Optional[Any] = None                 # backward compat — legacy pipeline
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    # MinIO keys set after upload
    minio_video_key: Optional[str] = None
    minio_audio_key: Optional[str] = None
    minio_image_key: Optional[str] = None
    db_scene_id: Optional[str] = None          # FK into scenes table


class StoryVideoResult(BaseModel):
    """Result of the full story-to-video pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_video_path: str
    segments: List[SegmentResult]
    story_title: str
    total_duration_seconds: float
    debug_minio_prefix: Optional[str] = None
    minio_final_video_key: Optional[str] = None
    minio_manifest_key: Optional[str] = None
    db_story_id: Optional[str] = None          # FK into stories table


# ---------------------------------------------------------------------------
# Location validation and image generation
# ---------------------------------------------------------------------------

async def sanitize_story_locations(
    story_output: Any, story_template_id: str
) -> int:
    """
    Clear location_id on any DialogLine whose ID is missing from the DB or
    belongs to a different template.  Returns the number of IDs stripped.

    Instead of failing hard, invalid IDs are set to None so the segment can
    fall back to scene_description-based image conditioning.
    """
    from virtual_streamer.utils.entity_repository import get_entity_repository

    repo = get_entity_repository()
    location_ids = {dl.location_id for dl in story_output.dialog if dl.location_id}
    if not location_ids:
        return 0

    valid: set[str] = set()
    for loc_id in location_ids:
        loc = await repo.get_location(loc_id)
        if loc and loc["story_template_id"] == story_template_id:
            valid.add(loc_id)
        else:
            logger.warning(
                f"location_id '{loc_id}' not found for template '{story_template_id}' "
                "— clearing from dialog line (will use scene_description instead)"
            )

    stripped = 0
    for dl in story_output.dialog:
        if dl.location_id and dl.location_id not in valid:
            dl.location_id = None
            stripped += 1
    return stripped


async def generate_location_image(
    location: dict,
    character: dict,
    output_dir: str,
    sd_server_url: str = "http://gx10-cbc5:1234",
) -> Optional[str]:
    """
    Generate a conditioning image for one scene using the location description
    and optionally the character's identity images as style references.

    Returns the local PNG path, or None if image generation fails (graceful
    degradation — the segment falls back to text-to-video mode).
    """

    try:
        os.makedirs(output_dir, exist_ok=True)

        char_name: str = character.get("name", "")
        char_desc: str = character.get("description", "")
        identity_images: list[str] = character.get("identity_images") or []
        has_character = bool(char_name or char_desc)

        if has_character:
            char_label = char_name or char_desc
            char_detail = f", {char_desc}" if char_desc and char_name else ""
            prompt = (
                f"{char_label}{char_detail} in {location['description']}, "
                f"{LOCATION_IMAGE_QUALITY_KEYWORDS}"
            )
            negative_prompt = "text, watermark, blurry, distorted"
        else:
            prompt = (
                f"{location['description']}, "
                f"no people, {LOCATION_IMAGE_QUALITY_KEYWORDS}"
            )
            negative_prompt = "text, watermark, blurry, distorted"

        logger.info(f"Len identity_images: {len(identity_images)}")

        config = StableDiffusionCppConfig(server_url=sd_server_url)
        async with StableDiffusionCppClient(config) as client:
            if identity_images:
                # Download MinIO paths to local files for image_edit
                local_paths: list[str] = []
                try:
                    storage = get_storage_client()
                    for minio_path in identity_images[:2]:  # limit to 2 references
                        fname = os.path.basename(minio_path)
                        local_tmp = os.path.join(output_dir, f"ref_{uuid.uuid4().hex[:8]}_{fname}")
                        await storage.download_file(minio_path, local_tmp)
                        if os.path.exists(local_tmp):
                            local_paths.append(local_tmp)
                            prompt += ", 1 person"
                except Exception as dl_err:
                    logger.warning(
                        f"Could not download identity images for conditioning: {dl_err}"
                    )

                if local_paths:
                    result = await client.image_edit(
                        ImageEditParams(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image_paths=local_paths,
                            width=1920,
                            height=1080,
                            extra_args={"denoising_strength": 0.1},
                        ),
                        output_dir=output_dir,
                    )
                    logger.info(
                        f"Conditioning image (img-edit) generated: {result.image_path}"
                    )
                    return result.image_path

            # Fallback: pure text-to-image
            if not has_character:
                negative_prompt += ", people, persons, characters"
            result = await client.txt2image(
                Txt2ImageParams(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=1280,
                    height=720,
                ),
                output_dir=output_dir,
            )
            logger.info(f"Conditioning image (txt2img) generated: {result.image_path}")
            return result.image_path

    except Exception as exc:
        logger.warning(
            f"Conditioning image generation failed (will use t2v instead): {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frames_from_duration(duration_seconds: float, fps: int) -> int:
    """Round a duration to the nearest valid LTX frame count (8n+1, min 9)."""
    raw = int(duration_seconds * fps)
    n = max(round((raw - 1) / 8), 1)
    return 8 * n + 1


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _safe_path(path: str, temp_dir: str) -> str:
    """Return a whitespace-free copy of *path* placed in *temp_dir* if needed."""
    if not any(c in path for c in (' ', '\t')):
        return path
    safe_name = os.path.basename(path).replace(' ', '_').replace('\t', '_')
    base, ext = os.path.splitext(safe_name)
    dest = os.path.join(temp_dir, f"{base}_{uuid.uuid4().hex[:8]}{ext}")
    shutil.copy2(path, dest)
    return dest


# ---------------------------------------------------------------------------
# concatenate_videos
# ---------------------------------------------------------------------------

def concatenate_videos(
    video_paths: List[str],
    output_path: str,
    temp_dir: str,
) -> str:
    """
    Concatenate *video_paths* into *output_path* using ffmpeg.

    Single-file shortcut: if only one path is supplied the file is copied
    directly (no ffmpeg needed, avoids any codec-mismatch risk).

    Strategy:
      1. Attempt stream-copy concat (-c copy) — fast, lossless.
      2. Fall back to re-encode (libx264 / aac) if stream-copy fails.

    All input files are validated (existence + non-zero size) before calling
    ffmpeg so that a missing segment surfaces as a clear error rather than a
    silent truncation of the output.
    """
    # ── Validate inputs ────────────────────────────────────────────────────
    logger.info(
        f"concatenate_videos: {len(video_paths)} input file(s) → {output_path}"
    )
    for i, p in enumerate(video_paths):
        size = _file_size(p)
        exists = os.path.exists(p)
        logger.info(f"  [{i}] {p}  exists={exists}  size={size} bytes")
        if not exists:
            raise FileNotFoundError(f"Segment file missing: {p}")
        if size == 0:
            raise ValueError(f"Segment file is empty (0 bytes): {p}")

    # ── Sanitize paths (replace whitespace so ffmpeg concat list is safe) ──
    video_paths = [_safe_path(p, temp_dir) for p in video_paths]

    # ── Single-file shortcut ───────────────────────────────────────────────
    if len(video_paths) == 1:
        logger.info("Single segment — copying directly (no ffmpeg concat needed)")
        shutil.copy2(video_paths[0], output_path)
        logger.info(
            f"Copy done: {output_path}  size={_file_size(output_path)} bytes"
        )
        return output_path

    # ── Build ffmpeg concat list ───────────────────────────────────────────
    concat_file = os.path.join(temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
    with open(concat_file, "w") as fh:
        for p in video_paths:
            fh.write(f"file '{os.path.abspath(p)}'\n")

    def _run_ffmpeg(extra_args: List[str], label: str) -> subprocess.CompletedProcess:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            *extra_args,
            output_path,
        ]
        logger.info(f"ffmpeg [{label}]: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            logger.debug(f"ffmpeg [{label}] stdout:\n{proc.stdout}")
        if proc.stderr:
            # ffmpeg writes progress + any dropped-frame warnings to stderr
            logger.info(f"ffmpeg [{label}] stderr:\n{proc.stderr}")
        return proc

    # ── Attempt 1: stream copy ─────────────────────────────────────────────
    proc = _run_ffmpeg(["-c", "copy"], "stream-copy")
    if proc.returncode != 0:
        logger.warning(
            f"ffmpeg stream-copy failed (rc={proc.returncode}), "
            f"falling back to re-encode.\n"
            f"stderr tail: {proc.stderr[-800:]}"
        )
        # ── Attempt 2: re-encode ───────────────────────────────────────────
        proc = _run_ffmpeg(
            ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast"],
            "re-encode",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg re-encode also failed (rc={proc.returncode}).\n"
                f"stderr: {proc.stderr[-1200:]}"
            )

    try:
        os.remove(concat_file)
    except OSError:
        pass

    out_size = _file_size(output_path)
    logger.info(
        f"Concatenation complete: {output_path}  size={out_size} bytes"
    )
    if out_size == 0:
        raise RuntimeError(
            f"ffmpeg produced an empty output file: {output_path}"
        )
    return output_path


# ---------------------------------------------------------------------------
# Debug upload helpers
# ---------------------------------------------------------------------------

async def _upload_debug_artifact(storage: Any, local_path: str, minio_key: str, label: str) -> Optional[str]:
    """Upload one file to MinIO. Logs but never raises — debug uploads are best-effort."""
    try:
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            logger.warning(f"[debug] skip upload {label}: file missing or empty ({local_path})")
            return None
        await storage.upload_file(local_path, minio_key)
        logger.info(f"[debug] uploaded {label} → {minio_key}")
        return minio_key
    except Exception as exc:
        logger.warning(f"[debug] upload failed for {label} ({minio_key}): {exc}")
        return None


async def _upload_debug_manifest(storage: Any, minio_key: str, data: dict) -> Optional[str]:
    """Upload the manifest JSON to MinIO. Best-effort."""
    try:
        await storage.put_json(minio_key, data)
        logger.info(f"[debug] manifest updated → {minio_key}")
        return minio_key
    except Exception as exc:
        logger.warning(f"[debug] manifest upload failed ({minio_key}): {exc}")
        return None


# ---------------------------------------------------------------------------
# generate_segment_from_input  (unified, works with SceneInput)
# ---------------------------------------------------------------------------

async def generate_segment_from_input(
    client: WanGPLTXClient,
    scene_input: SceneInput,
    output_dir: str,
    video_params: VideoGenerationParams,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> SegmentResult:
    """
    Generate one video segment from a stable SceneInput.

    Uses scene_input.ltx_prompt directly — no prompt-building step.
    Audio and image conditioning follow the same logic as generate_segment.
    """
    i = scene_input.scene_index
    mode = "audio-conditioned i2v" if audio_path else ("i2v" if image_path else "t2v")
    logger.info(
        f"[scene {i}] START  mode={mode}  "
        f"speaker={scene_input.speaker_id!r}  "
        f"line={str(scene_input.spoken_line or '')[:60]!r}"
    )

    duration = video_params.duration_seconds
    if audio_path:
        if not os.path.exists(audio_path):
            logger.warning(f"[scene {i}] Audio file not found: {audio_path} — skipping audio")
            audio_path = None
        else:
            try:
                from virtual_streamer.utils.utils import get_length
                audio_dur = get_length(audio_path)
                if audio_dur > 0:
                    duration = audio_dur + 0.5
                    logger.info(f"[scene {i}] Duration adapted to audio: {duration:.2f}s")
                else:
                    logger.warning(f"[scene {i}] get_length returned {audio_dur} — using configured duration")
            except Exception as exc:
                logger.warning(f"[scene {i}] Could not read audio length: {exc} — using configured duration")

    frames = _frames_from_duration(duration, video_params.fps)
    logger.info(
        f"[scene {i}] frames={frames}  duration={duration:.2f}s  "
        f"fps={video_params.fps}  audio_path={audio_path}"
    )

    segment_params = VideoGenerationParams(
        prompt=scene_input.ltx_prompt,
        negative_prompt=build_negative_prompt(),
        width=video_params.width,
        height=video_params.height,
        frames=frames,
        fps=video_params.fps,
        steps=video_params.steps,
        guidance_scale=video_params.guidance_scale,
        seed=video_params.seed,
        enable_audio=audio_path is not None,
        audio_path=audio_path,
        image_path=image_path,
    )

    segment_dir = os.path.join(output_dir, f"scene_{i:03d}_{uuid.uuid4().hex[:8]}")
    os.makedirs(segment_dir, exist_ok=True)

    result = await client.generate_video(params=segment_params, output_dir=segment_dir)

    logger.info(
        f"[scene {i}] DONE  video={result.video_path}  "
        f"duration={result.duration_seconds:.2f}s"
    )

    return SegmentResult(
        index=i,
        video_path=result.video_path,
        duration_seconds=result.duration_seconds,
        prompt_id=result.prompt_id,
        scene_input=scene_input,
        audio_path=audio_path,
        image_path=image_path,
    )


# ---------------------------------------------------------------------------
# generate_scene_image_from_input  (conditioning image from SceneInput)
# ---------------------------------------------------------------------------

async def generate_scene_image_from_input(
    scene_input: SceneInput,
    location: Optional[dict],
    character_dicts: List[dict],
    output_dir: str,
    video_params: VideoGenerationParams,
    sd_server_url: str = "http://gx10-cbc5:1234",
) -> Optional[str]:
    """
    Generate a conditioning image for one SceneInput.

    Uses scene_input.scene_visual_description (a FluxPrompt dict) as the prompt.
    Collects reference images from location.image_path and character identity images.
    Mirrors the logic of generate_scene_image but operates on SceneInput.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if scene_input.scene_visual_description:
            from virtual_streamer.image_generation.models import FluxPrompt
            flux_prompt = FluxPrompt.model_validate(scene_input.scene_visual_description)
            prompt = flux_prompt.to_prompt()
        else:
            prompt = scene_input.ltx_prompt  # fallback

        negative_prompt = "text, watermark, blurry, distorted"

        ref_minio_paths: List[str] = []
        if location and location.get("image_path"):
            ref_minio_paths.append(location["image_path"])
        for char in character_dicts:
            for img_path in (char.get("identity_images") or [])[:1]:
                ref_minio_paths.append(img_path)

        config = StableDiffusionCppConfig(server_url=sd_server_url)
        async with StableDiffusionCppClient(config) as client:
            if ref_minio_paths:
                local_refs: List[str] = []
                try:
                    storage = get_storage_client()
                    for minio_path in ref_minio_paths[:3]:
                        fname = os.path.basename(minio_path)
                        local_tmp = os.path.join(output_dir, f"ref_{uuid.uuid4().hex[:8]}_{fname}")
                        await storage.download_file(minio_path, local_tmp)
                        if os.path.exists(local_tmp):
                            local_refs.append(local_tmp)
                except Exception as dl_err:
                    logger.warning(f"Could not download reference images: {dl_err}")

                if local_refs:
                    result = await client.image_edit(
                        ImageEditParams(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image_paths=local_refs,
                            width=video_params.width,
                            height=video_params.height,
                            extra_args={"denoising_strength": 0.1},
                        ),
                        output_dir=output_dir,
                    )
                    logger.info(f"Scene conditioning image (img-edit): {result.image_path}")
                    return result.image_path

            result = await client.txt2image(
                Txt2ImageParams(
                    prompt=prompt,
                    negative_prompt=negative_prompt + ", people, persons, characters",
                    width=video_params.width,
                    height=video_params.height,
                ),
                output_dir=output_dir,
            )
            logger.info(f"Scene conditioning image (txt2img): {result.image_path}")
            return result.image_path

    except Exception as exc:
        logger.warning(f"Scene conditioning image generation failed: {exc}", exc_info=True)
        return None


async def _upload_conditioning_image(
    storage: Any,
    local_path: str,
    story_template_id: str,
    scene_id: str,
) -> Optional[str]:
    """Upload a conditioning image to MinIO and return its key. Best-effort."""
    try:
        key = f"conditioning_images/{story_template_id}/{scene_id}.png"
        await storage.upload_file(local_path, key)
        logger.info(f"[conditioning] uploaded → {key}")
        return key
    except Exception as exc:
        logger.warning(f"[conditioning] upload failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# story_input_to_video  (unified entry point — works with SceneInput)
# ---------------------------------------------------------------------------

async def story_input_to_video(
    story_input: StoryInput,
    ltx_config: Optional[LTXVideoConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    segment_audio_paths: Optional[Dict[int, str]] = None,
    sd_server_url: Optional[str] = None,
    debug_minio_prefix: Optional[str] = None,
    reference_videos: Optional[Dict[int, str]] = None,
    story_repo: Optional[Any] = None,
    db_story_id: Optional[str] = None,
) -> StoryVideoResult:
    """
    Convert a StoryInput into a final concatenated video.

    This is the unified core entry point. story_to_video() and scenes_to_video()
    are thin wrappers that construct a StoryInput and delegate here.

    DB persistence (best-effort, never aborts generation):
      - create_scene()                    before each segment
      - create_conditioning_image_artifact after conditioning image upload
      - update_scene_artifacts()          after each segment completes
      - update_story_status(COMPLETED)    after final video upload
      - update_story_status(FAILED)       on unrecoverable exception
    """
    config = ltx_config or LTXVideoConfig()
    params = video_params or VideoGenerationParams.from_preset("fast", duration_seconds=5.0)
    _sd_url = sd_server_url or os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / f"temp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(exist_ok=True)

    _debug_storage = None
    _debug_prefix = None
    if debug_minio_prefix:
        try:
            _debug_storage = get_storage_client()
            _debug_prefix = f"debug/{debug_minio_prefix}"
            logger.info(f"[debug] MinIO artifacts at: {_debug_prefix}/")
        except Exception as exc:
            logger.warning(f"[debug] Could not initialize MinIO client: {exc}")

    _manifest: dict = {
        "title": story_input.title,
        "story_template_id": story_input.story_template_id,
        "debug_prefix": _debug_prefix,
        "total_scenes": len(story_input.scenes),
        "segments": [],
        "failed_indices": [],
        "final_video_key": None,
    }

    # Pre-load location and character entities from DB
    location_map: Dict[str, dict] = {}
    character_map: Dict[str, dict] = {}
    template_id = story_input.story_template_id

    if template_id:
        from virtual_streamer.utils.entity_repository import get_entity_repository
        repo = get_entity_repository()

        loc_rows = await repo.list_locations_by_template(template_id)
        location_map = {loc["location_id"]: loc for loc in loc_rows}

        all_char_ids: set = set()
        for si in story_input.scenes:
            all_char_ids.update(si.character_ids_on_screen)
            if si.speaker_id:
                all_char_ids.add(si.speaker_id)

        for char_id in all_char_ids:
            char = await repo.get_character(char_id)
            if char:
                character_map[char_id] = char

        logger.info(
            f"story_input_to_video: loaded {len(location_map)} location(s) "
            f"and {len(character_map)} character(s)"
        )

    # Storage client for conditioning image uploads (only initialised if needed)
    _perm_storage = None
    if story_repo and db_story_id and template_id:
        try:
            _perm_storage = get_storage_client()
        except Exception as exc:
            logger.warning(f"[db] Could not initialise storage client for conditioning uploads: {exc}")

    audio_map = segment_audio_paths or {}
    segments: List[SegmentResult] = []
    failed_indices: List[int] = []
    total = len(story_input.scenes)

    try:
        async with WanGPLTXClient(config) as client:
            for i, scene_input in enumerate(story_input.scenes):
                if progress_callback:
                    progress_callback(i, total, f"Generating scene {i + 1}/{total}")

                # ── Optional: prompt enrichment via reference video ─────────
                if reference_videos and i in reference_videos:
                    ref_path = reference_videos[i]
                    if os.path.exists(ref_path):
                        enriched_prompt = await run_scene_enricher(ref_path, scene_input.ltx_prompt)
                        scene_input = scene_input.model_copy(update={"ltx_prompt": enriched_prompt})
                        logger.info(f"[scene {i}] ltx_prompt enriched via {ref_path!r}")
                    else:
                        logger.warning(f"[scene {i}] Reference video not found: {ref_path!r}")

                # ── DB: create scene row before generation ─────────────────
                db_scene_id: Optional[str] = None
                if story_repo and db_story_id:
                    db_scene_id = str(uuid.uuid4())
                    try:
                        await story_repo.create_scene(
                            scene_id=db_scene_id,
                            story_id=db_story_id,
                            scene_index=scene_input.scene_index,
                            prompt=scene_input.ltx_prompt,
                            raw_scene_data=scene_input.raw_scene_data,
                            speaker_id=scene_input.speaker_id,
                            spoken_line=scene_input.spoken_line,
                            location_id=scene_input.location_id,
                        )
                    except Exception as db_exc:
                        logger.warning(f"[db] create_scene failed for scene {i}: {db_exc}")
                        db_scene_id = None

                # ── Generate conditioning image ─────────────────────────────
                image_dir = str(output_path / f"images_{i:03d}_{uuid.uuid4().hex[:8]}")
                location = location_map.get(scene_input.location_id) if scene_input.location_id else None
                char_dicts = [
                    character_map[cid]
                    for cid in scene_input.character_ids_on_screen
                    if cid in character_map
                ]

                conditioning_image_path = await generate_scene_image_from_input(
                    scene_input=scene_input,
                    location=location,
                    character_dicts=char_dicts,
                    output_dir=image_dir,
                    sd_server_url=_sd_url,
                    video_params=video_params,
                )

                # ── Upload conditioning image & persist artifact ────────────
                minio_cond_key: Optional[str] = None
                if conditioning_image_path and _perm_storage and db_scene_id and template_id:
                    minio_cond_key = await _upload_conditioning_image(
                        _perm_storage, conditioning_image_path, template_id, db_scene_id
                    )
                    if minio_cond_key:
                        char_img_keys = [
                            img
                            for cid in scene_input.character_ids_on_screen
                            for img in (character_map.get(cid, {}).get("identity_images") or [])[:1]
                        ]
                        loc_img_key = location.get("image_path") if location else None
                        try:
                            await story_repo.create_conditioning_image_artifact(
                                artifact_id=str(uuid.uuid4()),
                                scene_id=db_scene_id,
                                final_image_key=minio_cond_key,
                                character_image_keys=char_img_keys,
                                flux_prompt_json=scene_input.scene_visual_description or {},
                                location_image_key=loc_img_key,
                            )
                        except Exception as db_exc:
                            logger.warning(f"[db] create_conditioning_image_artifact failed: {db_exc}")

                try:
                    segment = await generate_segment_from_input(
                        client=client,
                        scene_input=scene_input,
                        output_dir=str(output_path),
                        video_params=params,
                        audio_path=audio_map.get(i),
                        image_path=conditioning_image_path,
                    )
                    segment.db_scene_id = db_scene_id
                    segments.append(segment)

                    # ── Debug uploads ─────────────────────────────────────
                    if _debug_storage and _debug_prefix:
                        seg_key_video = f"{_debug_prefix}/segments/scene_{i:03d}.mp4"
                        seg_key_audio = f"{_debug_prefix}/audio/scene_{i:03d}.wav" if segment.audio_path else None
                        seg_key_image = f"{_debug_prefix}/images/scene_{i:03d}.png" if segment.image_path else None

                        segment.minio_video_key = await _upload_debug_artifact(
                            _debug_storage, segment.video_path, seg_key_video, f"scene_{i:03d} video"
                        )
                        if seg_key_audio and segment.audio_path:
                            segment.minio_audio_key = await _upload_debug_artifact(
                                _debug_storage, segment.audio_path, seg_key_audio, f"scene_{i:03d} audio"
                            )
                        if seg_key_image and segment.image_path:
                            segment.minio_image_key = await _upload_debug_artifact(
                                _debug_storage, segment.image_path, seg_key_image, f"scene_{i:03d} image"
                            )

                        _manifest["segments"].append({
                            "index": i,
                            "speaker_id": scene_input.speaker_id,
                            "spoken_line": scene_input.spoken_line,
                            "location_id": scene_input.location_id,
                            "ltx_prompt": scene_input.ltx_prompt,
                            "duration_seconds": segment.duration_seconds,
                            "failed": False,
                            "minio_video_key": segment.minio_video_key,
                            "minio_audio_key": segment.minio_audio_key,
                            "minio_image_key": segment.minio_image_key,
                        })
                        await _upload_debug_manifest(
                            _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
                        )

                    # ── DB: update scene artifacts ─────────────────────────
                    if story_repo and db_scene_id:
                        try:
                            await story_repo.update_scene_artifacts(
                                scene_id=db_scene_id,
                                video_segment_key=segment.minio_video_key,
                                audio_key=segment.minio_audio_key,
                                duration_seconds=segment.duration_seconds,
                            )
                        except Exception as db_exc:
                            logger.warning(f"[db] update_scene_artifacts failed for scene {i}: {db_exc}")

                except Exception as exc:
                    failed_indices.append(i)
                    logger.warning(
                        f"Scene {i + 1}/{total} FAILED "
                        f"(speaker={scene_input.speaker_id!r}): {exc}",
                        exc_info=True,
                    )
                    if _debug_storage and _debug_prefix:
                        _manifest["failed_indices"].append(i)
                        _manifest["segments"].append({
                            "index": i,
                            "speaker_id": scene_input.speaker_id,
                            "spoken_line": scene_input.spoken_line,
                            "failed": True,
                            "error": str(exc),
                        })
                        await _upload_debug_manifest(
                            _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
                        )

        if not segments:
            raise RuntimeError(
                f"All {total} scene(s) failed — cannot produce a video."
            )

        if progress_callback:
            progress_callback(total, total, "Concatenating scenes…")

        video_paths = [seg.video_path for seg in segments]
        safe_title = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in story_input.title
        )[:50].strip()
        final_filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.mp4"
        final_path = str(output_path / final_filename)

        concatenate_videos(video_paths=video_paths, output_path=final_path, temp_dir=str(temp_dir))
        total_duration = sum(seg.duration_seconds for seg in segments)

        # Clean up temporary segment dirs, image dirs, and temp dir
        for seg in segments:
            seg_dir = os.path.dirname(seg.video_path)
            shutil.rmtree(seg_dir, ignore_errors=True)
        for d in output_path.glob("images_*"):
            shutil.rmtree(str(d), ignore_errors=True)
        shutil.rmtree(str(temp_dir), ignore_errors=True)

        minio_final_key: Optional[str] = None
        minio_manifest_key: Optional[str] = None
        if _debug_storage and _debug_prefix:
            minio_final_key = await _upload_debug_artifact(
                _debug_storage, final_path, f"{_debug_prefix}/final.mp4", "final video"
            )
            _manifest["final_video_key"] = minio_final_key
            _manifest["total_duration_seconds"] = total_duration
            minio_manifest_key = await _upload_debug_manifest(
                _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
            )

        # ── DB: mark story completed ───────────────────────────────────────
        if story_repo and db_story_id:
            try:
                await story_repo.update_story_status(
                    story_id=db_story_id,
                    status="COMPLETED",
                    final_video_key=minio_final_key,
                )
            except Exception as db_exc:
                logger.warning(f"[db] update_story_status(COMPLETED) failed: {db_exc}")

        if progress_callback:
            progress_callback(total, total, "Complete!")

        logger.info(
            f"story_input_to_video DONE  final={final_path}  "
            f"scenes={len(segments)}/{total}  duration={total_duration:.1f}s"
        )

        return StoryVideoResult(
            final_video_path=final_path,
            segments=segments,
            story_title=story_input.title,
            total_duration_seconds=total_duration,
            debug_minio_prefix=_debug_prefix,
            minio_final_video_key=minio_final_key,
            minio_manifest_key=minio_manifest_key,
            db_story_id=db_story_id,
        )

    except Exception as outer_exc:
        # ── DB: mark story failed ──────────────────────────────────────────
        if story_repo and db_story_id:
            try:
                await story_repo.update_story_status(db_story_id, "FAILED")
            except Exception:
                pass
        raise


# ---------------------------------------------------------------------------
# generate_segment  (legacy — kept for backward compatibility)
# ---------------------------------------------------------------------------

async def generate_segment(
    client: WanGPLTXClient,
    dialog_line: Any,
    index: int,
    output_dir: str,
    video_params: VideoGenerationParams,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> SegmentResult:
    """
    Generate one video segment for *dialog_line*.

    Backward-compatible wrapper around generate_segment_from_input. Builds an
    LTX prompt from the DialogLine and delegates to the unified implementation.
    The dialog_line is stored on the result for backward compatibility.
    """
    prompt = build_ltx_prompt(dialog_line=dialog_line, include_dialog_audio=True)
    scene_input = DialogLineInput.from_dialog_line(dialog_line, index, prompt)
    result = await generate_segment_from_input(
        client=client,
        scene_input=scene_input,
        output_dir=output_dir,
        video_params=video_params,
        audio_path=audio_path,
        image_path=image_path,
    )
    result.dialog_line = dialog_line  # backward compat
    return result


# ---------------------------------------------------------------------------
# story_to_video  (main entry point)
# ---------------------------------------------------------------------------

async def story_to_video(
    story_output: Any,
    ltx_config: Optional[LTXVideoConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting.",
    segment_audio_paths: Optional[Dict[int, str]] = None,
    story_template_id: Optional[str] = None,
    sd_server_url: Optional[str] = None,
    debug_minio_prefix: Optional[str] = None,
) -> StoryVideoResult:
    """
    Convert a StoryOutput into a final concatenated video.

    Thin wrapper: converts StoryOutput → StoryInput and delegates to
    story_input_to_video. Public signature is unchanged for backward compat.
    """
    if story_template_id:
        stripped = await sanitize_story_locations(story_output, story_template_id)
        if stripped:
            logger.info(f"Stripped {stripped} invalid location_id(s); will use scene_description fallback")

    scene_inputs = [
        DialogLineInput.from_dialog_line(
            dl, i, build_ltx_prompt(dialog_line=dl, include_dialog_audio=True)
        )
        for i, dl in enumerate(story_output.dialog)
    ]
    story_input = StoryInput(
        title=story_output.title,
        story_plan=getattr(story_output, "story_plan", ""),
        story_template_id=story_template_id,
        raw_agent_output=story_output.model_dump(by_alias=True),
        scenes=scene_inputs,
    )
    return await story_input_to_video(
        story_input=story_input,
        ltx_config=ltx_config,
        video_params=video_params,
        output_dir=output_dir,
        progress_callback=progress_callback,
        segment_audio_paths=segment_audio_paths,
        sd_server_url=sd_server_url,
        debug_minio_prefix=debug_minio_prefix,
    )


# ---------------------------------------------------------------------------
# scenes_to_video  (new LTX pipeline using DetailedScene list)
# ---------------------------------------------------------------------------

async def generate_scene_image(
    scene: Any,
    location: Optional[dict],
    character_dicts: List[dict],
    output_dir: str,
    sd_server_url: str = "http://gx10-cbc5:1234",
) -> Optional[str]:
    """
    Generate a conditioning image for one DetailedScene.

    When a location with a pre-generated image_path is available:
      - Download the location base image from MinIO
      - Collect character identity images
      - Run ImageEdit with all references + scene_visual_description prompt

    Fallback: txt2image from scene_visual_description alone.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        prompt = scene.scene_visual_description.to_prompt()
        negative_prompt = "text, watermark, blurry, distorted"

        # Collect all reference images: location base + character identity images
        ref_minio_paths: list[str] = []

        if location and location.get("image_path"):
            ref_minio_paths.append(location["image_path"])

        for char in character_dicts:
            for img_path in (char.get("identity_images") or [])[:1]:  # 1 ref per character
                ref_minio_paths.append(img_path)

        config = StableDiffusionCppConfig(server_url=sd_server_url)
        async with StableDiffusionCppClient(config) as client:
            if ref_minio_paths:
                local_refs: list[str] = []
                try:
                    storage = get_storage_client()
                    for minio_path in ref_minio_paths[:3]:  # cap at 3 references
                        fname = os.path.basename(minio_path)
                        local_tmp = os.path.join(output_dir, f"ref_{uuid.uuid4().hex[:8]}_{fname}")
                        await storage.download_file(minio_path, local_tmp)
                        if os.path.exists(local_tmp):
                            local_refs.append(local_tmp)
                except Exception as dl_err:
                    logger.warning(f"Could not download reference images: {dl_err}")

                if local_refs:
                    result = await client.image_edit(
                        ImageEditParams(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image_paths=local_refs,
                            width=1920,
                            height=1080,
                            extra_args={"denoising_strength": 0.1},
                        ),
                        output_dir=output_dir,
                    )
                    logger.info(f"Scene conditioning image (img-edit): {result.image_path}")
                    return result.image_path

            # Fallback: txt2image
            result = await client.txt2image(
                Txt2ImageParams(
                    prompt=prompt,
                    negative_prompt=negative_prompt + ", people, persons, characters",
                    width=1920,
                    height=1080,
                ),
                output_dir=output_dir,
            )
            logger.info(f"Scene conditioning image (txt2img): {result.image_path}")
            return result.image_path

    except Exception as exc:
        logger.warning(f"Scene conditioning image generation failed: {exc}", exc_info=True)
        return None


async def generate_scene_segment(
    client: WanGPLTXClient,
    scene: Any,
    index: int,
    output_dir: str,
    video_params: VideoGenerationParams,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> SegmentResult:
    """
    Generate one video segment for a DetailedScene.

    Uses scene.ltx_prompt directly (no prompt building step).
    Audio and image conditioning follow the same logic as generate_segment.
    """
    mode = "audio-conditioned i2v" if audio_path else ("i2v" if image_path else "t2v")
    logger.info(
        f"[scene {index}] START  mode={mode}  "
        f"speaker={scene.speaker_id!r}  "
        f"line={str(scene.spoken_line or '')[:60]!r}"
    )

    duration = video_params.duration_seconds
    if audio_path:
        if not os.path.exists(audio_path):
            logger.warning(f"[scene {index}] Audio file not found: {audio_path} — skipping audio")
            audio_path = None
        else:
            try:
                from virtual_streamer.utils.utils import get_length
                audio_dur = get_length(audio_path)
                if audio_dur > 0:
                    duration = audio_dur
                    logger.info(f"[scene {index}] Duration adapted to audio: {duration:.2f}s")
            except Exception as exc:
                logger.warning(f"[scene {index}] Could not read audio length: {exc}")

    frames = _frames_from_duration(duration, video_params.fps)

    segment_params = VideoGenerationParams(
        prompt=scene.ltx_prompt,
        negative_prompt=build_negative_prompt(),
        width=video_params.width,
        height=video_params.height,
        frames=frames,
        fps=video_params.fps,
        steps=video_params.steps,
        guidance_scale=video_params.guidance_scale,
        seed=video_params.seed,
        enable_audio=audio_path is not None,
        audio_path=audio_path,
        image_path=image_path,
    )

    segment_dir = os.path.join(output_dir, f"scene_{index:03d}")
    os.makedirs(segment_dir, exist_ok=True)

    result = await client.generate_video(params=segment_params, output_dir=segment_dir)

    logger.info(
        f"[scene {index}] DONE  video={result.video_path}  "
        f"duration={result.duration_seconds:.2f}s"
    )

    return SegmentResult(
        index=index,
        video_path=result.video_path,
        duration_seconds=result.duration_seconds,
        prompt_id=result.prompt_id,
        scene=scene,
        audio_path=audio_path,
        image_path=image_path,
    )


async def scenes_to_video(
    scenes: List[Any],
    story_title: str = "story",
    ltx_config: Optional[LTXVideoConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    segment_audio_paths: Optional[Dict[int, str]] = None,
    story_template_id: Optional[str] = None,
    sd_server_url: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    debug_minio_prefix: Optional[str] = None,
    reference_videos: Optional[Dict[int, str]] = None,
    story_repo: Optional[Any] = None,
    db_story_id: Optional[str] = None,
) -> StoryVideoResult:
    """
    Convert a list of DetailedScene objects into a final concatenated video.

    Thin wrapper: converts scenes → StoryInput (via DetailedSceneInput) and
    delegates to story_input_to_video. Public signature is backward-compatible;
    story_repo and db_story_id are new optional parameters for DB persistence.
    """
    scene_inputs = [DetailedSceneInput.from_detailed_scene(s, i) for i, s in enumerate(scenes)]
    story_input = StoryInput(
        title=story_title,
        story_plan="",
        story_template_id=story_template_id,
        raw_agent_output={"scenes": [s.model_dump() for s in scenes]},
        scenes=scene_inputs,
    )
    return await story_input_to_video(
        story_input=story_input,
        ltx_config=ltx_config,
        video_params=video_params,
        output_dir=output_dir,
        progress_callback=progress_callback,
        segment_audio_paths=segment_audio_paths,
        sd_server_url=sd_server_url,
        debug_minio_prefix=debug_minio_prefix,
        reference_videos=reference_videos,
        story_repo=story_repo,
        db_story_id=db_story_id,
    )


# ---------------------------------------------------------------------------
# title_to_video  (convenience end-to-end entry point)
# ---------------------------------------------------------------------------

async def title_to_video(
    title: str,
    story_template_id: Optional[str] = None,
    ltx_config: Optional[LTXVideoConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> StoryVideoResult:
    """
    End-to-end: title → story → video.

    1. StoryGeneratorAgent generates a StoryOutput from *title*.
    2. story_to_video generates and concatenates all segments.
    """
    from virtual_streamer.api.high_level.video_generation import run_story_pipeline

    logger.info(f"title_to_video: generating story for {title!r}")

    if progress_callback:
        progress_callback(0, 1, "Generating story…")

    story_output = await run_story_pipeline(
        title=title,
        story_template_id=story_template_id,
    )
    logger.info(
        f"Story generated: {story_output.title!r} "
        f"with {len(story_output.dialog)} dialog lines"
    )

    return await story_to_video(
        story_output=story_output,
        ltx_config=ltx_config,
        video_params=video_params,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
