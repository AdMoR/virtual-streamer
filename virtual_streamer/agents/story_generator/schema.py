"""
Pydantic schema for StoryGeneratorAgent output.
"""

from pydantic import BaseModel, Field


class DialogLine(BaseModel):
    character: str
    dialog: str


class DialogLines(BaseModel):
    lines: list[DialogLine]


class StoryOutput(BaseModel):
    """
    Structured output from story generation.
    
    The LLM generates a story with three components:
    - title: A refined, catchy title
    - story_plan: The creative reasoning and comedic arc
    - dialog: The actual spoken lines
    """
    
    title: str = Field(
        description="Refined/expanded title for the story. Should be catchy and descriptive."
    )
    story_plan: str = Field(
        description="Overall plan and reasoning used to create the dialog. "
                    "Explains the creative choices and comedic arc."
    )
    dialog: DialogLines = Field(
        description="The actual dialog lines produced by Fred and other characters. "
                    "Only spoken lines, no stage directions or descriptions."
    )
