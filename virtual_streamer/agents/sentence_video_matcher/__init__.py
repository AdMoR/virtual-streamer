"""
Sentence Video Matcher Agent.

Finds the best matching video for each dialog line from a video database
using parallel video matching and aggregation.
"""

from virtual_streamer.agents.sentence_video_matcher.agent import (
    create_sentence_video_matcher,
)
from virtual_streamer.agents.sentence_video_matcher.schema import (
    DialogLineMatch,
    SentenceVideoMatcherOutput,
)

__all__ = [
    "create_sentence_video_matcher",
    "DialogLineMatch",
    "SentenceVideoMatcherOutput",
]

