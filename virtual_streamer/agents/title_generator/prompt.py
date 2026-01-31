"""Instruction provider for title generation."""
import logging
from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import STORY_TEMPLATE_ID

logger = logging.getLogger(__name__)


TITLE_GENERATION_PROMPT = """You are a creative title generator for video stories.

Based on the story template below, generate {count} unique and creative titles.

**Story Template:**
- Name: {template_name}
- Theme/Prompt: {template_prompt}
- Characters: {characters}

**Requirements:**
- Each title should be catchy, intriguing, and appropriate for the theme
- Titles should be varied - cover different angles, topics, scenarios
- Keep titles concise (5-15 words)
- Make them suitable for short-form video content
- Avoid repetitive patterns

Generate exactly {count} titles as a JSON list.
"""


async def load_template_info(template_id: str) -> dict:
    """Load story template and character info from DB."""
    from virtual_streamer.utils.entity_repository import get_entity_repository

    repo = get_entity_repository()
    template = await repo.get_story_template(template_id)
    if template is None:
        raise ValueError(f"Story template '{template_id}' not found")

    # Load character names
    characters = []
    for char_id in template.get("character_ids", []):
        char = await repo.get_character(char_id)
        if char:
            characters.append(char.get("name", char_id))

    return {
        "template_name": template["name"],
        "template_prompt": template["prompt"][:500],  # Truncate for context
        "characters": ", ".join(characters) if characters else "No specific characters",
    }


class TitleInstructionProvider(InstructionProvider):
    """Dynamic instruction provider that loads story template and builds prompt."""

    def __init__(self, count: int = 50):
        self.count = count

    async def __call__(self, ctx: ReadonlyContext) -> str:
        template_id = ctx.state.get(STORY_TEMPLATE_ID)
        if not template_id:
            raise ValueError("No story_template_id in state")

        template_info = await load_template_info(template_id)

        return TITLE_GENERATION_PROMPT.format(
            count=self.count,
            **template_info,
        )
