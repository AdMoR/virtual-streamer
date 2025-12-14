"""
Sentence Video Matcher Agent.

Finds the best matching video for each sentence from a video database
using parallel video matching and aggregation.
"""

from virtual_streamer.agents.sentence_video_matcher.agent import (
    SentenceVideoMatcherAgent,
)
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherInput,
    SentenceVideoMatcherOutput,
)

__all__ = [
    "SentenceVideoMatcherAgent",
    "SentenceVideoMatcherInput",
    "SentenceVideoMatcherOutput",
]

