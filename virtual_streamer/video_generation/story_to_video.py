"""
Story-to-Video Pipeline

Converts a generated story (StoryOutput) to a final video using LTX-2
for text-to-video generation. Each DialogLine becomes a video segment,
and all segments are concatenated into the final video.

Usage:
    from virtual_streamer.video_generation.story_to_video import story_to_video
    
    final_video = await story_to_video(
        story_output=story,
        comfyui_config=ComfyUIConfig(server_url="http://localhost:8188"),
        output_dir="./output",
    )
"""

import asyncio
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from virtual_streamer.video_generation.config import DialogLine, StoryOutput
from virtual_streamer.video_generation.comfyui_client import (
    ComfyUIClient,
    ComfyUIConfig,
    VideoGenerationParams,
    VideoGenerationResult,
)
from virtual_streamer.video_generation.ltx_prompt_builder import (
    build_ltx_prompt,
    build_negative_prompt,
    build_prompts_from_story,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    """Result of generating a single video segment."""
    index: int
    dialog_line: DialogLine
    video_path: str
    duration_seconds: float
    prompt_id: str


@dataclass
class StoryVideoResult:
    """Result of the full story-to-video pipeline."""
    final_video_path: str
    segments: List[SegmentResult]
    story_title: str
    total_duration_seconds: float


def concatenate_videos(
    video_paths: List[str],
    output_path: str,
    temp_dir: str,
) -> str:
    """
    Concatenate multiple video files into a single video using ffmpeg.
    
    Args:
        video_paths: List of paths to video files to concatenate
        output_path: Path for the output concatenated video
        temp_dir: Directory for temporary files
    
    Returns:
        Path to the concatenated video
    """
    # Create concat file
    concat_file = os.path.join(temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
    
    with open(concat_file, "w") as f:
        for path in video_paths:
            # Use absolute paths and escape single quotes
            abs_path = os.path.abspath(path)
            f.write(f"file '{abs_path}'\n")
    
    # Run ffmpeg concat
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",  # Copy streams without re-encoding
        output_path,
    ]
    
    logger.info(f"Concatenating {len(video_paths)} videos...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg concat failed: {result.stderr}")
        # Try with re-encoding as fallback
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path,
        ]
        result = subprocess.run(cmd_reencode, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
    
    # Clean up concat file
    try:
        os.remove(concat_file)
    except OSError:
        pass
    
    return output_path


async def generate_segment(
    client: ComfyUIClient,
    dialog_line: DialogLine,
    index: int,
    output_dir: str,
    video_params: VideoGenerationParams,
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting.",
) -> SegmentResult:
    """
    Generate a single video segment for a DialogLine.
    
    Args:
        client: ComfyUI client instance
        dialog_line: The DialogLine to generate video for
        index: Segment index (for naming)
        output_dir: Directory to save the segment
        video_params: Base video generation parameters
        style_suffix: Style instructions to append to prompt
    
    Returns:
        SegmentResult with video path and metadata
    """
    # Build prompt from dialog line
    prompt = build_ltx_prompt(
        dialog_line=dialog_line,
        include_dialog_audio=True,
        style_suffix=style_suffix,
    )
    
    # Create params for this segment
    segment_params = VideoGenerationParams(
        prompt=prompt,
        negative_prompt=build_negative_prompt(),
        width=video_params.width,
        height=video_params.height,
        duration_seconds=video_params.duration_seconds,
        fps=video_params.fps,
        steps=video_params.steps,
        cfg_scale=video_params.cfg_scale,
        seed=video_params.seed,
        enable_audio=video_params.enable_audio,
    )
    
    # Create segment-specific output directory
    segment_dir = os.path.join(output_dir, f"segment_{index:03d}")
    os.makedirs(segment_dir, exist_ok=True)
    
    logger.info(f"Generating segment {index}: {dialog_line.text[:50]}...")
    
    # Generate video
    result = await client.generate_video(
        params=segment_params,
        output_dir=segment_dir,
    )
    
    return SegmentResult(
        index=index,
        dialog_line=dialog_line,
        video_path=result.video_path,
        duration_seconds=result.duration_seconds,
        prompt_id=result.prompt_id,
    )


async def story_to_video(
    story_output: StoryOutput,
    comfyui_config: Optional[ComfyUIConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting.",
) -> StoryVideoResult:
    """
    Convert a generated story to video using LTX-2.
    
    This is the main pipeline function that:
    1. Takes a StoryOutput with DialogLines
    2. Generates a video segment for each DialogLine using LTX-2
    3. Concatenates all segments into a final video
    
    Args:
        story_output: StoryOutput containing title, story_plan, and dialog lines
        comfyui_config: ComfyUI server configuration (defaults to localhost:8188)
        video_params: Base video generation parameters for each segment
        output_dir: Directory to save output files
        progress_callback: Optional callback(current, total, message) for progress
        style_suffix: Style instructions to append to each prompt
    
    Returns:
        StoryVideoResult with final video path and segment details
    
    Example:
        story = StoryOutput(
            title="Fred discovers AI",
            story_plan="...",
            dialog=[
                DialogLine(character_id="fred", text="...", scene_description="..."),
                ...
            ]
        )
        
        result = await story_to_video(story, output_dir="./videos")
        print(f"Final video: {result.final_video_path}")
    """
    # Use defaults if not provided
    config = comfyui_config or ComfyUIConfig()
    params = video_params or VideoGenerationParams(
        prompt="",  # Will be overwritten per segment
        duration_seconds=5.0,
        width=1280,
        height=720,
        fps=24,
        steps=20,
        cfg_scale=4.0,
    )
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    segments: List[SegmentResult] = []
    total_lines = len(story_output.dialog)
    
    logger.info(f"Starting story-to-video for '{story_output.title}' with {total_lines} dialog lines")
    
    async with ComfyUIClient(config) as client:
        for i, dialog_line in enumerate(story_output.dialog):
            if progress_callback:
                progress_callback(i, total_lines, f"Generating segment {i+1}/{total_lines}")
            
            segment = await generate_segment(
                client=client,
                dialog_line=dialog_line,
                index=i,
                output_dir=str(output_path),
                video_params=params,
                style_suffix=style_suffix,
            )
            segments.append(segment)
            
            logger.info(f"Segment {i+1}/{total_lines} complete: {segment.video_path}")
    
    if progress_callback:
        progress_callback(total_lines, total_lines, "Concatenating segments...")
    
    # Concatenate all segments
    video_paths = [seg.video_path for seg in segments]
    
    # Generate output filename from story title
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in story_output.title)
    safe_title = safe_title[:50].strip()
    final_filename = f"{safe_title}_{uuid.uuid4().hex[:8]}.mp4"
    final_path = str(output_path / final_filename)
    
    concatenate_videos(
        video_paths=video_paths,
        output_path=final_path,
        temp_dir=str(temp_dir),
    )
    
    total_duration = sum(seg.duration_seconds for seg in segments)
    
    logger.info(f"Story-to-video complete: {final_path} ({total_duration:.1f}s)")
    
    if progress_callback:
        progress_callback(total_lines, total_lines, "Complete!")
    
    return StoryVideoResult(
        final_video_path=final_path,
        segments=segments,
        story_title=story_output.title,
        total_duration_seconds=total_duration,
    )


async def title_to_video(
    title: str,
    story_template_id: Optional[str] = None,
    comfyui_config: Optional[ComfyUIConfig] = None,
    video_params: Optional[VideoGenerationParams] = None,
    output_dir: str = "./output",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> StoryVideoResult:
    """
    Generate a complete video from just a title.
    
    This is the end-to-end pipeline that:
    1. Runs StoryGeneratorAgent to create a story from the title
    2. Generates video for each dialog line using LTX-2
    3. Concatenates into final video
    
    Args:
        title: Topic/title for story generation
        story_template_id: Optional story template ID to use
        comfyui_config: ComfyUI server configuration
        video_params: Video generation parameters
        output_dir: Output directory
        progress_callback: Progress callback
    
    Returns:
        StoryVideoResult with final video
    """
    # Import here to avoid circular imports
    from virtual_streamer.api.high_level.video_generation import run_story_generator
    
    logger.info(f"Generating story for title: {title}")
    
    if progress_callback:
        progress_callback(0, 1, "Generating story...")
    
    # Generate story
    story_output = await run_story_generator(
        title=title,
        story_template_id=story_template_id,
    )
    
    logger.info(f"Story generated: {story_output.title} with {len(story_output.dialog)} lines")
    
    # Convert story to video
    return await story_to_video(
        story_output=story_output,
        comfyui_config=comfyui_config,
        video_params=video_params,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
