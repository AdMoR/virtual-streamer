"""
Schemas for SentenceVideoMatcherAgent.

Input: DialogLines from story generator
Output: List of DialogLineMatch with video matches
"""

from typing import List

from pydantic import BaseModel, Field

from virtual_streamer.agents.story_generator.schema import DialogLine
from virtual_streamer.agents.video_matcher.schema import ContextualRating


class DialogLineMatch(BaseModel):
    """
    A dialog line matched to a video.
    
    Combines the original DialogLine with video match information.
    """
    
    dialog_line: DialogLine = Field(description="The original dialog line")
    video_path: str = Field(description="Path to the matched video")
    rating: ContextualRating = Field(description="Match quality rating")
    grade: int = Field(description="Numeric grade for ranking", ge=0, le=10)
    reasoning: str = Field(description="Explanation of the rating")


class SentenceVideoMatcherOutput(BaseModel):
    """Output from SentenceVideoMatcherAgent."""
    
    matches: List[DialogLineMatch] = Field(
        description="List of best video matches, one per input dialog line"
    )
    
    def to_dict_by_dialog(self) -> dict:
        """Convert to dictionary keyed by dialog text."""
        return {match.dialog_line.text: match for match in self.matches}

