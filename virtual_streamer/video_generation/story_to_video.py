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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from virtual_streamer.video_generation.config import DialogLine, StoryOutput
from virtual_streamer.video_generation.ltx_client import (
    WanGPLTXClient,
    LTXVideoConfig,
    VideoGenerationParams,
)
from virtual_streamer.video_generation.ltx_prompt_builder import (
    build_ltx_prompt,
    build_negative_prompt,
)
from virtual_streamer.image_generation.stable_cpp_client import (
    StableDiffusionCppClient,
    StableDiffusionCppConfig,
    Txt2ImageParams,
    ImageEditParams,
)
from virtual_streamer.utils.minio_client import get_storage_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """Result of generating a single video segment."""
    index: int
    dialog_line: DialogLine
    video_path: str
    duration_seconds: float
    prompt_id: str
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    # MinIO keys for debug artifacts (set when debug_minio_prefix is provided)
    minio_video_key: Optional[str] = None
    minio_audio_key: Optional[str] = None
    minio_image_key: Optional[str] = None


@dataclass
class StoryVideoResult:
    """Result of the full story-to-video pipeline."""
    final_video_path: str
    segments: List[SegmentResult]
    story_title: str
    total_duration_seconds: float
    debug_minio_prefix: Optional[str] = None
    minio_final_video_key: Optional[str] = None
    minio_manifest_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Location validation and image generation
# ---------------------------------------------------------------------------

async def sanitize_story_locations(
    story_output: StoryOutput, story_template_id: str
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
                "cinematic composition, photorealistic, high quality"
            )
            negative_prompt = "text, watermark, blurry, distorted"
        else:
            prompt = (
                f"{location['description']}, "
                "cinematic composition, photorealistic, no people, high quality, "
                "detailed environment"
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
                        local_tmp = os.path.join(output_dir, f"ref_{fname}")
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
                            width=1280,
                            height=720,
                            extra_args={"denoising_strength": 0.85},
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
    dest = os.path.join(temp_dir, safe_name)
    if not os.path.exists(dest):
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
# generate_segment
# ---------------------------------------------------------------------------

async def generate_segment(
    client: WanGPLTXClient,
    dialog_line: DialogLine,
    index: int,
    output_dir: str,
    video_params: VideoGenerationParams,
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting.",
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> SegmentResult:
    """
    Generate one video segment for *dialog_line*.

    When *audio_path* is provided:
    - The segment uses LTX audio-conditioned generation
      (audio_guide sent to WanGP, audio_prompt_type="A").
    - Video duration is adapted to the TTS audio length (snapped to nearest
      valid 8n+1 frame count) so the clip is long enough to cover the speech.

    When *image_path* is provided the segment uses image-to-video (i2v) mode
    with the conditioning image as the start frame.

    When neither is provided the segment falls back to plain text-to-video.
    """
    if audio_path:
        mode = "audio-conditioned i2v"
    elif image_path:
        mode = "i2v"
    else:
        mode = "t2v"
    logger.info(
        f"[segment {index}] START  mode={mode}  "
        f"char={dialog_line.character_id!r}  "
        f"text={dialog_line.text[:60]!r}"
    )

    prompt = build_ltx_prompt(
        dialog_line=dialog_line,
        include_dialog_audio=True,
        style_suffix=style_suffix,
    )

    # Adapt duration to TTS audio length when audio is available
    duration = video_params.duration_seconds
    if audio_path:
        if not os.path.exists(audio_path):
            logger.warning(
                f"[segment {index}] Audio file not found: {audio_path} — "
                "falling back to configured duration"
            )
            audio_path = None
        else:
            try:
                from virtual_streamer.utils.utils import get_length
                audio_dur = get_length(audio_path)
                if audio_dur > 0:
                    duration = audio_dur
                    logger.info(
                        f"[segment {index}] Duration adapted to audio: "
                        f"{duration:.2f}s  (file size: {_file_size(audio_path)} bytes)"
                    )
                else:
                    logger.warning(
                        f"[segment {index}] get_length returned {audio_dur} — "
                        "using configured duration"
                    )
            except Exception as exc:
                logger.warning(
                    f"[segment {index}] Could not read audio length: {exc} — "
                    "using configured duration"
                )

    frames = _frames_from_duration(duration, video_params.fps)
    logger.info(
        f"[segment {index}] frames={frames}  duration={duration:.2f}s  "
        f"fps={video_params.fps}  audio_path={audio_path}"
    )

    segment_params = VideoGenerationParams(
        prompt=prompt,
        negative_prompt=build_negative_prompt(),
        width=video_params.width,
        height=video_params.height,
        frames=frames,
        fps=video_params.fps,
        steps=video_params.steps,
        cfg_scale=video_params.cfg_scale,
        seed=video_params.seed,
        enable_audio=audio_path is not None,
        audio_path=audio_path,
        image_path=image_path,
    )

    segment_dir = os.path.join(output_dir, f"segment_{index:03d}")
    os.makedirs(segment_dir, exist_ok=True)

    result = await client.generate_video(
        params=segment_params,
        output_dir=segment_dir,
    )

    logger.info(
        f"[segment {index}] DONE  video={result.video_path}  "
        f"size={_file_size(result.video_path)} bytes  "
        f"duration={result.duration_seconds:.2f}s"
    )

    return SegmentResult(
        index=index,
        dialog_line=dialog_line,
        video_path=result.video_path,
        duration_seconds=result.duration_seconds,
        prompt_id=result.prompt_id,
        audio_path=audio_path,
        image_path=image_path,
    )


# ---------------------------------------------------------------------------
# story_to_video  (main entry point)
# ---------------------------------------------------------------------------

async def story_to_video(
    story_output: StoryOutput,
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

    Pipeline:
      For each dialog line →
        (optional) load TTS audio from *segment_audio_paths*
        → generate_segment (LTX audio-conditioned or t2v)
      → concatenate all successful segments

    Per-segment resilience: if a segment fails (WanGP error, network issue,
    OOM, etc.) the error is logged as a WARNING and the segment is skipped.
    The pipeline continues with the remaining segments.  If NO segment
    succeeds a RuntimeError is raised.

    Args:
        story_output: StoryOutput with title and dialog list.
        ltx_config: WanGP REST server config (default: gx10-cbc5:8082).
        video_params: Base parameters (width, height, fps, steps, …).
        output_dir: Directory for output files and intermediaries.
        progress_callback: Optional callback(current, total, message).
        style_suffix: Style text appended to every LTX prompt.
        segment_audio_paths: Mapping {segment_index: local_wav_path} produced
            by the TTS step in _run_ltx_video_generation.  When a key is
            absent the segment falls back to plain text-to-video.
        debug_minio_prefix: When set, all intermediate artifacts (per-segment
            video, audio, conditioning image) and a running manifest.json are
            uploaded to MinIO under ``debug/{debug_minio_prefix}/``.  Uploads
            are best-effort and never fail the pipeline.
    """
    config = ltx_config or LTXVideoConfig()
    params = video_params or VideoGenerationParams(
        prompt="",
        duration_seconds=5.0,
        width=1280,
        height=720,
        fps=24,
        steps=8,
        cfg_scale=4.0,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Set up MinIO debug uploads (best-effort; None disables all uploads)
    _debug_storage = None
    _debug_prefix = None
    if debug_minio_prefix:
        try:
            from virtual_streamer.utils.minio_client import get_storage_client
            _debug_storage = get_storage_client()
            _debug_prefix = f"debug/{debug_minio_prefix}"
            logger.info(f"[debug] MinIO debug artifacts will be uploaded to: {_debug_prefix}/")
        except Exception as exc:
            logger.warning(f"[debug] Could not initialize MinIO storage client: {exc}")

    # Running manifest — updated after each segment
    _manifest: dict = {
        "title": story_output.title,
        "story_template_id": story_template_id,
        "debug_prefix": _debug_prefix,
        "total_dialog_lines": len(story_output.dialog),
        "segments": [],
        "failed_indices": [],
        "final_video_key": None,
    }

    total_lines = len(story_output.dialog)
    logger.info(
        f"story_to_video START  title={story_output.title!r}  "
        f"dialog_lines={total_lines}  "
        f"audio_segments_provided={len(segment_audio_paths or {})}  "
        f"ltx_server={config.server_url}"
    )

    # --- Location validation and pre-loading ---
    location_map: Dict[str, dict] = {}
    character_map: Dict[str, dict] = {}
    _sd_url = sd_server_url or os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")

    if story_template_id:
        stripped = await sanitize_story_locations(story_output, story_template_id)
        if stripped:
            logger.info(f"Stripped {stripped} invalid location_id(s); will use scene_description fallback")

        from virtual_streamer.utils.entity_repository import get_entity_repository
        repo = get_entity_repository()

        loc_rows = await repo.list_locations_by_template(story_template_id)
        location_map = {loc["location_id"]: loc for loc in loc_rows}

        for dialog_line in story_output.dialog:
            char_id = dialog_line.character_id
            if char_id not in character_map:
                char = await repo.get_character(char_id)
                if char:
                    character_map[char_id] = char

        logger.info(
            f"Loaded {len(location_map)} location(s) and "
            f"{len(character_map)} character(s) for conditioning"
        )

    audio_map = segment_audio_paths or {}
    segments: List[SegmentResult] = []
    failed_indices: List[int] = []

    async with WanGPLTXClient(config) as client:
        for i, dialog_line in enumerate(story_output.dialog):
            if progress_callback:
                progress_callback(
                    i, total_lines, f"Generating segment {i + 1}/{total_lines}"
                )

            # Generate conditioning image for this segment.
            # Priority: registered location description → scene_description fallback.
            conditioning_image_path: Optional[str] = None
            if story_template_id:
                image_dir = str(output_path / f"images_{i:03d}")
                if dialog_line.location_id and dialog_line.location_id in location_map:
                    conditioning_image_path = await generate_location_image(
                        location=location_map[dialog_line.location_id],
                        character=character_map.get(dialog_line.character_id, {}),
                        output_dir=image_dir,
                        sd_server_url=_sd_url,
                    )
                elif dialog_line.scene_description:
                    # No valid location — use the scene description as the prompt
                    conditioning_image_path = await generate_location_image(
                        location={"description": dialog_line.scene_description},
                        character=character_map.get(dialog_line.character_id, {}),
                        output_dir=image_dir,
                        sd_server_url=_sd_url,
                    )

            try:
                segment = await generate_segment(
                    client=client,
                    dialog_line=dialog_line,
                    index=i,
                    output_dir=str(output_path),
                    video_params=params,
                    style_suffix=style_suffix,
                    audio_path=audio_map.get(i),
                    image_path=conditioning_image_path,
                )
                segments.append(segment)
                logger.info(
                    f"Segment {i + 1}/{total_lines} OK — "
                    f"video={segment.video_path}"
                )

                # ── Debug uploads ──────────────────────────────────────────
                if _debug_storage and _debug_prefix:
                    seg_key_video = f"{_debug_prefix}/segments/segment_{i:03d}.mp4"
                    seg_key_audio = f"{_debug_prefix}/audio/segment_{i:03d}.wav" if segment.audio_path else None
                    seg_key_image = f"{_debug_prefix}/images/segment_{i:03d}.png" if segment.image_path else None

                    segment.minio_video_key = await _upload_debug_artifact(
                        _debug_storage, segment.video_path, seg_key_video, f"segment_{i:03d} video"
                    )
                    if seg_key_audio and segment.audio_path:
                        segment.minio_audio_key = await _upload_debug_artifact(
                            _debug_storage, segment.audio_path, seg_key_audio, f"segment_{i:03d} audio"
                        )
                    if seg_key_image and segment.image_path:
                        segment.minio_image_key = await _upload_debug_artifact(
                            _debug_storage, segment.image_path, seg_key_image, f"segment_{i:03d} image"
                        )

                    _manifest["segments"].append({
                        "index": i,
                        "character_id": dialog_line.character_id,
                        "text": dialog_line.text,
                        "location_id": dialog_line.location_id,
                        "scene_description": dialog_line.scene_description,
                        "duration_seconds": segment.duration_seconds,
                        "prompt_id": segment.prompt_id,
                        "failed": False,
                        "minio_video_key": segment.minio_video_key,
                        "minio_audio_key": segment.minio_audio_key,
                        "minio_image_key": segment.minio_image_key,
                        "story_output_dialog": "\n".join(str(x) for x in story_output.dialog),
                    })
                    await _upload_debug_manifest(
                        _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
                    )

            except Exception as exc:
                failed_indices.append(i)
                logger.warning(
                    f"Segment {i + 1}/{total_lines} FAILED (char="
                    f"{dialog_line.character_id!r}, "
                    f"text={dialog_line.text[:40]!r}): {exc}",
                    exc_info=True,
                )
                if _debug_storage and _debug_prefix:
                    _manifest["failed_indices"].append(i)
                    _manifest["segments"].append({
                        "index": i,
                        "character_id": dialog_line.character_id,
                        "text": dialog_line.text,
                        "location_id": dialog_line.location_id,
                        "scene_description": dialog_line.scene_description,
                        "failed": True,
                        "error": str(exc),
                    })
                    await _upload_debug_manifest(
                        _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
                    )

    logger.info(
        f"Generation loop complete: {len(segments)} succeeded, "
        f"{len(failed_indices)} failed"
        + (f"  failed_indices={failed_indices}" if failed_indices else "")
    )

    if not segments:
        raise RuntimeError(
            f"All {total_lines} segment(s) failed — cannot produce a video. "
            f"Check server logs for WanGP errors."
        )

    if progress_callback:
        progress_callback(total_lines, total_lines, "Concatenating segments…")

    video_paths = [seg.video_path for seg in segments]
    print(
        f"Concatenating {len(video_paths)} segment(s):\n"
        + "\n".join(f"  [{j}] {p}" for j, p in enumerate(video_paths))
    )

    safe_title = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in story_output.title
    )
    safe_title = safe_title[:50].strip()
    final_filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.mp4"
    final_path = str(output_path / final_filename)

    concatenate_videos(
        video_paths=video_paths,
        output_path=final_path,
        temp_dir=str(temp_dir),
    )

    total_duration = sum(seg.duration_seconds for seg in segments)
    print(
        f"story_to_video DONE  final={final_path}  "
        f"segments={len(segments)}/{total_lines}  "
        f"duration={total_duration:.1f}s"
    )

    # ── Upload final video + close manifest ───────────────────────────────
    minio_final_key: Optional[str] = None
    minio_manifest_key: Optional[str] = None
    if _debug_storage and _debug_prefix:
        minio_final_key = await _upload_debug_artifact(
            _debug_storage, final_path, f"{_debug_prefix}/final.mp4", "final video"
        )
        _manifest["final_video_key"] = minio_final_key
        _manifest["total_duration_seconds"] = total_duration
        _manifest["failed_indices"] = failed_indices
        minio_manifest_key = await _upload_debug_manifest(
            _debug_storage, f"{_debug_prefix}/manifest.json", _manifest
        )
        logger.info(
            f"[debug] artifacts at minio://{_debug_prefix}/  "
            f"final={minio_final_key}  manifest={minio_manifest_key}"
        )

    if progress_callback:
        progress_callback(total_lines, total_lines, "Complete!")

    return StoryVideoResult(
        final_video_path=final_path,
        segments=segments,
        story_title=story_output.title,
        total_duration_seconds=total_duration,
        debug_minio_prefix=_debug_prefix,
        minio_final_video_key=minio_final_key,
        minio_manifest_key=minio_manifest_key,
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
    from virtual_streamer.api.high_level.video_generation import run_story_generator

    logger.info(f"title_to_video: generating story for {title!r}")

    if progress_callback:
        progress_callback(0, 1, "Generating story…")

    story_output = await run_story_generator(
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
