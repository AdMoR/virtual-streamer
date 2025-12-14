"""
Schemas for SentenceVideoMatcherAgent.

Input: List of sentences to match
Output: List of sentence-video matches with ratings
"""

from typing import List

from pydantic import BaseModel, Field

from virtual_streamer.agents.video_matcher.schema import VideoMatchResult


class SentenceVideoMatcherInput(BaseModel):
    """Input for SentenceVideoMatcherAgent."""
    
    sentences: List[str] = Field(
        description="List of sentences (dialogue lines) to find matching videos for"
    )


class SentenceVideoMatcherOutput(BaseModel):
    """Output from SentenceVideoMatcherAgent."""
    
    matches: List[VideoMatchResult] = Field(
        description="List of best video matches, one per input sentence"
    )
    
    def to_dict_by_sentence(self) -> dict:
        """Convert to dictionary keyed by sentence."""
        return {match.sentence: match for match in self.matches}

