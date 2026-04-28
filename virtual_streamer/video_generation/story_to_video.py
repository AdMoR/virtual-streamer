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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

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


@dataclass
class StoryVideoResult:
    """Result of the full story-to-video pipeline."""
    final_video_path: str
    segments: List[SegmentResult]
    story_title: str
    total_duration_seconds: float


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
            # ffmpeg writes progress to stderr even on success — always log it
            logger.debug(f"ffmpeg [{label}] stderr:\n{proc.stderr}")
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
) -> SegmentResult:
    """
    Generate one video segment for *dialog_line*.

    When *audio_path* is provided:
    - The segment uses LTX audio-conditioned generation
      (audio_guide sent to WanGP, audio_prompt_type="A").
    - Video duration is adapted to the TTS audio length (snapped to nearest
      valid 8n+1 frame count) so the clip is long enough to cover the speech.

    When *audio_path* is None the segment falls back to plain text-to-video.
    """
    mode = "audio-conditioned i2v" if audio_path else "t2v"
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

    total_lines = len(story_output.dialog)
    logger.info(
        f"story_to_video START  title={story_output.title!r}  "
        f"dialog_lines={total_lines}  "
        f"audio_segments_provided={len(segment_audio_paths or {})}  "
        f"ltx_server={config.server_url}"
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
            try:
                segment = await generate_segment(
                    client=client,
                    dialog_line=dialog_line,
                    index=i,
                    output_dir=str(output_path),
                    video_params=params,
                    style_suffix=style_suffix,
                    audio_path=audio_map.get(i),
                )
                segments.append(segment)
                logger.info(
                    f"Segment {i + 1}/{total_lines} OK — "
                    f"video={segment.video_path}"
                )
            except Exception as exc:
                failed_indices.append(i)
                logger.warning(
                    f"Segment {i + 1}/{total_lines} FAILED (char="
                    f"{dialog_line.character_id!r}, "
                    f"text={dialog_line.text[:40]!r}): {exc}",
                    exc_info=True,
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

    if progress_callback:
        progress_callback(total_lines, total_lines, "Complete!")

    return StoryVideoResult(
        final_video_path=final_path,
        segments=segments,
        story_title=story_output.title,
        total_duration_seconds=total_duration,
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
