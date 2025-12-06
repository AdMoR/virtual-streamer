"""
Utility functions for ADK agents.

This module provides reusable utility functions for video generation:
- Text processing (sentence splitting)
- Video processing (frame extraction, segment combining)

These functions are adapted from virtual_streamer.video_generation.core
to be used within the ADK agent architecture.
"""

import base64
import os
from typing import List, Optional

import cv2

from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
    get_length,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Text Processing
# ═══════════════════════════════════════════════════════════════════════════════


def separation_fn(raw_text: str, max_length: int = 35) -> List[str]:
    """
    Split text into manageable sentences for video generation.

    This function breaks down a dialog text into individual sentences,
    respecting natural sentence boundaries and maximum length constraints.

    Args:
        raw_text: Raw story/dialog text to split
        max_length: Maximum length per sentence segment

    Returns:
        List of sentence segments ready for video generation
    
    Example:
        >>> text = "Fred: Hello Jamy!\\nJamy: What's up?"
        >>> separation_fn(text)
        ['Fred: Hello Jamy!', "Jamy: What's up?"]
    """
    def split(txt: str, separator: str) -> List[str]:
        return [x.strip() for x in txt.split(separator) if len(x.replace(" ", "")) > 0]

    parts = []
    for p in split(raw_text, "\n"):
        if len(p) > max_length:
            broken_down = False
            for sep in [".", "!", "?"]:
                sub_parts = split(p, sep)
                if len(sub_parts) > 1:
                    broken_down = True
                    parts.extend(sub_parts)
                    break
            if not broken_down:
                parts.append(p)
        else:
            parts.append(p)

    return parts


# ═══════════════════════════════════════════════════════════════════════════════
# Video Processing
# ═══════════════════════════════════════════════════════════════════════════════


def extract_middle_frame(video_path: str) -> Optional[str]:
    """
    Extract the middle frame from a video and return as base64 encoded string.

    This is used for the vision LLM to judge video-dialogue matching.

    Args:
        video_path: Path to the video file

    Returns:
        Base64 encoded JPEG image string, or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2

        # Set position to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return base64_image
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return None


def combine_segment(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_path: str,
    fontsize: int = 14,
) -> str:
    """
    Combine video, audio, and subtitles into a single segment.

    Args:
        video_path: Path to the source video clip
        audio_path: Path to the generated audio file
        subtitle_path: Path to the SRT subtitle file
        output_path: Path for the output combined segment
        fontsize: Font size for subtitles

    Returns:
        Path to the combined segment file
    """
    # Create intermediate path for video+audio combination
    temp_combined = output_path.replace(".mp4", "_temp.mp4")
    
    # First combine video and audio
    combine_video_and_short_audio(video_path, audio_path, temp_combined)
    
    # Then add subtitles
    add_subtitle_from_srt(temp_combined, subtitle_path, output_path, fontsize=fontsize)
    
    # Clean up temp file
    if os.path.exists(temp_combined):
        os.remove(temp_combined)
    
    return output_path


def concatenate_videos(segment_paths: List[str], output_dir: str) -> str:
    """
    Concatenate multiple video segments into a final video.

    Args:
        segment_paths: List of paths to video segments to concatenate
        output_dir: Directory for the output video

    Returns:
        Path to the final concatenated video
    """
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    concat_file = os.path.join(output_dir, f"concat_list_{timestamp}.txt")
    
    combine_part_in_concat_file(segment_paths, concat_file, final_video_path)
    
    # Clean up concat file
    if os.path.exists(concat_file):
        os.remove(concat_file)
    
    return final_video_path


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds
    """
    return get_length(video_path)

