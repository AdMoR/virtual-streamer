"""
Common utilities for ADK agents.

This module provides:
- State key constants for shared state management
- Utility functions for video/audio processing
- Shared callbacks used across agents
"""

from virtual_streamer.agents.common.state_keys import (
    TITLE,
    CONFIG,
    STORY_OUTPUT,
    SENTENCES,
    VIDEO_MATCHES,
    AUDIO_FILES,
    SUBTITLE_FILES,
    VIDEO_SEGMENTS,
    FINAL_VIDEO_PATH,
    task_key,
    result_key,
    keyword_key,
)
from virtual_streamer.agents.common.callbacks import (
    FinalizeVideoCallback,
)
from virtual_streamer.agents.common.utils import (
    separation_fn,
    extract_middle_frame,
    combine_segment,
    concatenate_videos,
    get_video_duration,
)

__all__ = [
    # State keys
    "TITLE",
    "CONFIG",
    "STORY_OUTPUT",
    "SENTENCES",
    "VIDEO_MATCHES",
    "AUDIO_FILES",
    "SUBTITLE_FILES",
    "VIDEO_SEGMENTS",
    "FINAL_VIDEO_PATH",
    "task_key",
    "result_key",
    "keyword_key",
    # Callbacks
    "FinalizeVideoCallback",
    # Utils
    "separation_fn",
    "extract_middle_frame",
    "combine_segment",
    "concatenate_videos",
    "get_video_duration",
]

