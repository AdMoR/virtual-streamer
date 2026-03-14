"""
Pydantic schema for StoryTemplateBuilderAgent output.

Only covers the creative fields this agent is responsible for.
The API layer (collection, character_ids) is handled separately.
"""

from pydantic import BaseModel, Field


class StoryTemplateOutput(BaseModel):
    """
    Creative output of the story template builder.

    Fields:
        name: Human-readable display name for the template.
        prompt: Full template prompt text — the story-specific instructions
                (character definitions, tone, story arc, rules, examples).
                The meta-prompt will inject {title}, {target_lines}, {characters}
                at generation time; do NOT include those variables here.
        target_lines: Recommended number of dialogue lines for stories using
                      this template.
    """

    name: str = Field(
        description="Display name for the story template (e.g. 'C\\'est pas Sorcier Parody')."
    )
    prompt: str = Field(
        description=(
            "Full template-specific prompt: character personalities, story arc, "
            "tone elements, rules, and examples. "
            "Do NOT include meta-prompt variables like {title} or {characters}."
        )
    )
    target_lines: int = Field(
        description="Recommended number of dialogue lines per generated story.",
        ge=1,
    )