"""
Video Matcher Agent.

Judges if a video clip matches a dialogue using vision LLM.
"""

from virtual_streamer.agents.video_matcher.agent import get_video_matcher
from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
from virtual_streamer.agents.video_matcher.schema import (
    ContextualRating,
    VideoJudgementOutput,
    VideoMatchResult,
    VideoSentenceInput,
)

__all__ = [
    "get_video_matcher",
    "BestMatchAggregator",
    "ContextualRating",
    "VideoJudgementOutput",
    "VideoMatchResult",
    "VideoSentenceInput",
]

