"""
Story Formatter Prompt.

Provides the instruction for story_formatter: reads the raw story text from state
and asks the LLM to format it into a structured StoryOutput.
"""

import logging
from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import RAW_STORY_TEXT

logger = logging.getLogger(__name__)

FORMATTER_PROMPT = """You are a structured data extractor.

You will receive the raw text of a story generated in the style of the French educational TV show "C'est pas Sorcier".
Your sole task is to extract and reformat this story into the required structured output — do NOT rewrite or alter the content.

Extract the following fields:
- **title**: The refined story title.
- **story_plan**: The overall creative plan and reasoning described in the story (the thinking section).
- **dialog**: The list of dialogue lines. For each line extract:
  - **character_id**: The character identifier (e.g. "fred", "jamy"). Use lowercase.
  - **text**: The spoken dialogue text (what the character says out loud).
  - **scene_description**: The visual scene description used for video matching.

Raw story:
{raw_story}"""


class StoryFormatterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider for story_formatter.

    Reads RAW_STORY_TEXT from state (set by story_writer) and injects it
    into the formatting prompt.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_story = ctx.state.get(RAW_STORY_TEXT, "")
        if not raw_story:
            logger.warning("No raw story text found in state for formatting")
        return FORMATTER_PROMPT.format(raw_story=raw_story)