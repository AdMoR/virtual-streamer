"""Shared frame-count math for LTX segment generation."""

from typing import Optional

#: Approximate spoken words per second (French/general conversational rate).
WORDS_PER_SECOND: float = 2.2
MIN_SPEECH_SECONDS: float = 5.0
MAX_SPEECH_SECONDS: float = 15.0


def frames_from_duration(duration_seconds: float, fps: int) -> int:
    """Round a duration to the nearest valid LTX frame count (8n+1, min 9)."""
    raw = int(duration_seconds * fps)
    n = max(round((raw - 1) / 8), 1)
    return 8 * n + 1


def video_length_from_spoken_line(spoken_line: Optional[str], fps: int) -> int:
    """
    Estimate the required video_length (8n+1 frames) from the word count of
    *spoken_line*.

    Uses a conversational speech rate of ~2.2 words/second plus a 1.5-second
    margin for lead-in/lead-out, clamped to [5, 15] seconds.
    """
    words = len(spoken_line.split()) if spoken_line else 0
    duration = max(
        MIN_SPEECH_SECONDS,
        min(MAX_SPEECH_SECONDS, words / WORDS_PER_SECOND + 1.5),
    )
    return frames_from_duration(duration, fps)
