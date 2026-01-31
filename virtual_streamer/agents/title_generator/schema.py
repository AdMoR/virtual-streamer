"""Output schema for title generation."""
from pydantic import BaseModel, Field
from typing import List


class TitlesOutput(BaseModel):
    """Structured output from title generation."""

    titles: List[str] = Field(
        description="List of creative story titles. Each title should be unique, "
        "catchy, and appropriate for the story template's theme.",
        min_length=1,
    )
