"""
Prompts for the Location Builder pipeline.

location_writer  — generates a rich free-text diffusion description for a location,
                   with the story template context injected.
location_formatter — extracts the single LocationDescriptionOutput field.
"""

import logging

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import (
    LOCATION_NAME,
    STORY_TEMPLATE_ID,
    RAW_LOCATION_TEXT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Writer prompt
# =============================================================================

WRITER_PROMPT = """\
You are a visual environment designer for a generative AI video pipeline.

Your task: write a detailed diffusion-model image generation prompt for a location.

The location will appear as the background/setting of scenes in a video story.
It must be described with enough visual detail for a text-to-image model to
consistently reproduce the same environment across multiple scenes.

=== STORY TEMPLATE CONTEXT ===
{template_context}

=== LOCATION NAME ===
{location_name}

Write a detailed diffusion-model prompt for this location. Include:
1. Overall architectural or natural environment style
2. Lighting conditions and time of day
3. Color palette and dominant textures
4. Atmosphere and mood
5. Specific background/foreground visual elements
6. Camera/perspective framing hints (e.g. wide angle, eye level)

Do NOT describe any characters or people — only the environment.
Do NOT include story or narrative elements.
Keep it under 150 words and highly visual.
"""


# =============================================================================
# Formatter prompt
# =============================================================================

FORMATTER_PROMPT = """\
You are a structured data extractor.

You will receive a free-text description of a location environment written for \
a diffusion-model image generation pipeline.
Your sole task is to extract the description and output it as structured data.
Do NOT rewrite or alter the content.

Extract this field:
- **description**: The complete diffusion-model prompt text for this location.

Raw location description:
{raw_location}"""


# =============================================================================
# Instruction Providers
# =============================================================================


class LocationWriterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction for location_writer.

    Reads LOCATION_NAME and STORY_TEMPLATE_ID from state, loads the template
    prompt from DB as context, and formats WRITER_PROMPT.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        location_name = ctx.state.get(LOCATION_NAME, "")
        template_id = ctx.state.get(STORY_TEMPLATE_ID, "")

        if not location_name:
            logger.warning("No location_name found in state")

        template_context = ""
        if template_id:
            try:
                from virtual_streamer.utils.entity_repository import get_entity_repository
                repo = get_entity_repository()
                template = await repo.get_story_template(template_id)
                if template:
                    template_context = template.get("prompt", "")[:500]
                    logger.info(f"Loaded template context for '{template_id}'")
                else:
                    logger.warning(f"Template '{template_id}' not found")
            except Exception as e:
                logger.warning(f"Could not load template context: {e}")

        return WRITER_PROMPT.format(
            location_name=location_name,
            template_context=template_context or "No specific story context available.",
        )


class LocationFormatterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction for location_formatter.

    Reads RAW_LOCATION_TEXT from state (set by location_writer) and injects
    it into the formatting prompt.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_location = ctx.state.get(RAW_LOCATION_TEXT, "")
        if not raw_location:
            logger.warning("No raw location text found in state for formatting")
        return FORMATTER_PROMPT.format(raw_location=raw_location)