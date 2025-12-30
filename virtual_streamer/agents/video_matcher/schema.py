"""
Pydantic schemas for VideoMatcherAgent.

Schemas:
- VideoSentenceInput: Input for video matching (sentence + video path)
- VideoJudgementOutput: LLM output (rating + grade + reasoning)
- VideoMatchResult: Full result combining input and output (for aggregation)
"""

from enum import Enum
from pydantic import BaseModel, Field


class ContextualRating(str, Enum):
    """Rating for video-dialogue contextual match."""
    CONTEXTUAL = "CONTEXTUAL"
    NEUTRAL = "NEUTRAL"
    NOT_CONTEXTUAL = "NOT_CONTEXTUAL"
    FAILURE = "FAILURE"


class VideoSentenceInput(BaseModel):
    """Input for video matching: character, sentence and video path."""
    character: str = Field(description="The character speaking this line")
    sentence: str = Field(description="The dialogue sentence to match")
    video_path: str = Field(description="Path to the video file to evaluate")


class VideoJudgementOutput(BaseModel):
    """
    Structured output from vision LLM judgement.
    
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


class VideoMatchResult(BaseModel):
    """
    Complete video match result combining input and LLM output.
    
    This is what gets stored in state and used by aggregators.
    Includes the video_path so we know which video was judged.
    """
    
    character: str = Field(description="The character speaking this line")
    sentence: str = Field(description="The original sentence being matched")
    video_path: str = Field(description="Path to the video that was evaluated")
    rating: ContextualRating = Field(description="Match quality rating")
    grade: int = Field(description="Numeric grade for ranking", ge=0, le=10)
    reasoning: str = Field(description="Explanation of the rating")
    
    @classmethod
    def from_input_and_output(
        cls,
        input_data: VideoSentenceInput,
        output_data: VideoJudgementOutput,
    ) -> "VideoMatchResult":
        """Create VideoMatchResult by combining input and output."""
        return cls(
            character=input_data.character,
            sentence=input_data.sentence,
            video_path=input_data.video_path,
            rating=output_data.rating,
            grade=output_data.grade,
            reasoning=output_data.reasoning,
        )