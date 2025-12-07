"""
Pydantic schema for VideoMatcherAgent output.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ContextualRating(str, Enum):
    """Rating for video-dialogue contextual match."""
    CONTEXTUAL = "CONTEXTUAL"
    NEUTRAL = "NEUTRAL"
    NOT_CONTEXTUAL = "NOT_CONTEXTUAL"
    FAILURE = "FAILURE"


class VideoJudgementOutput(BaseModel):
    """
    Structured output from video-dialogue matching judgement.
    
    The vision LLM analyzes a video frame and determines how well
    it matches the dialogue line being spoken.
    """
    
    rating: ContextualRating = Field(
        description="Overall rating: CONTEXTUAL (good match), NEUTRAL (vaguely related), "
                    "or NOT_CONTEXTUAL (poor match)"
    )
    grade: int = Field(
        description="Numeric grade counting factors supporting the rating (for ranking). "
                    "Higher is better.",
        ge=0,
        le=10,
    )
    reasoning: str = Field(
        description="Brief explanation of why this rating was given, "
                    "describing what visual elements match or don't match the dialogue."
    )

class VideoSentenceInput(BaseModel):
    sentence: str
    video_path: str