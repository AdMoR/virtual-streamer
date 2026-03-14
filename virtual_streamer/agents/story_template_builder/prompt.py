"""
Prompts for the Story Template Builder pipeline.

template_writer  — builds a rich free-text template from the user's story idea,
                   with characters fetched from the API injected into the prompt.
template_formatter — takes the raw text and extracts the structured fields.
"""

import logging
from typing import List

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.api.clients.character_client import list_characters
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import TITLE, RAW_TEMPLATE_TEXT
from virtual_streamer.video_server.models import Character

logger = logging.getLogger(__name__)

# Narrator fallback used when no real characters are appropriate
NARRATOR_CHARACTER = (
    "- narrator (character_id: \"narrator\"): "
    "An off-screen narrator voice that presents information objectively, "
    "like a documentary voiceover. Use this character when no specific "
    "on-screen character fits the story."
)


# =============================================================================
# Writer prompt — guides rich template creation
# =============================================================================

WRITER_PROMPT = """\
You are a creative director designing a story template for an AI video generation system.

The system generates humorous short-form video stories. Each story template defines:
- The characters' personalities and roles
- The overall story arc and comedic structure
- Tone guidelines, rules, and stylistic constraints
- One or more concrete examples illustrating the desired style

Your task is to create a complete story template based on the user's story concept below.

=== AVAILABLE CHARACTERS ===
{characters_block}

If none of these characters naturally fit the requested story concept, use the narrator
character instead (describe the story as a documentary-style voiceover).

=== YOUR TEMPLATE MUST INCLUDE ===

1. **Template name** — a short, descriptive name for this template.

2. **Character definitions** — for each character you select from the list above,
   describe their personality, speech style, and role within the story. Be specific
   and give concrete examples of vocabulary, attitude, and recurring behaviors.

3. **Story arc** — a numbered list of story beats (typically 4–6). Each beat must be
   concrete: what happens, what escalation occurs, what the comedic tension is.

4. **Tone elements** — bullet-pointed tone guidelines. Each point must include:
   - The guideline
   - A concrete example of a line of dialogue that illustrates it

5. **Rules** — hard constraints the LLM must follow when generating a story with
   this template (length of lines, what characters can/cannot say, format, etc.).

6. **Target lines** — your recommendation for the number of dialogue lines
   (typically 6 for a punchy format). Justify briefly.

7. **Full example** — write one complete example story (in the exact format:
   "CharacterName: dialogue line") followed by a commentary explaining what works
   and what to avoid in each line.

=== QUALITY BAR ===
Study this reference carefully — your template must match this level of detail:

The template prompt is what an LLM will receive when generating a new story.
It must be self-contained: the LLM should be able to create a perfect story
using only this prompt, without any external explanation.

Keep it practical, specific, and example-rich. Avoid vague guidance like
"be funny" — instead show what funny looks like with concrete dialogue examples.

=== USER STORY CONCEPT ===
{title}

Now write the complete story template:
"""

# =============================================================================
# Formatter prompt — extracts structured fields from the free-text template
# =============================================================================

FORMATTER_PROMPT = """\
You are a structured data extractor.

You will receive a detailed story template written in free text.
Your task is to extract and format it into the required structured output.
Do NOT rewrite, summarize, or alter any content — preserve it faithfully.

Extract these fields:
- **name**: The template display name.
- **prompt**: The complete template prompt text. This must include everything:
  character definitions, story arc, tone elements, rules, examples, and commentary.
  Preserve all formatting (bullet points, numbered lists, examples, etc.).
- **target_lines**: The recommended number of dialogue lines (integer).

Raw template:
{raw_template}
"""


# =============================================================================
# Instruction Providers
# =============================================================================


def _build_characters_block(characters: List[Character]) -> str:
    """Format a list of Character objects into a prompt-ready block."""
    lines = []
    for char in characters:
        desc = char.description or "No description available."
        lines.append(
            f"- {char.name} (character_id: \"{char.character_id}\"): {desc}"
        )
    # Always append narrator as a fallback option
    lines.append(NARRATOR_CHARACTER)
    return "\n".join(lines)


class TemplateWriterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction for template_writer.

    Reads TITLE from state and fetches available characters from the API,
    then injects both into the writer prompt.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        title = ctx.state.get(TITLE, "")
        if not title:
            logger.warning("No title found in state for template writing")

        # Fetch characters from API
        characters: List[Character] = []
        try:
            characters = await list_characters()
            logger.info(f"Fetched {len(characters)} characters from API")
        except Exception as e:
            logger.warning(f"Could not fetch characters from API: {e}. Falling back to narrator only.")

        characters_block = _build_characters_block(characters)

        return WRITER_PROMPT.format(
            characters_block=characters_block,
            title=title,
        )


class TemplateFormatterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction for template_formatter.

    Reads RAW_TEMPLATE_TEXT from state (set by template_writer) and injects
    it into the formatting prompt.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_template = ctx.state.get(RAW_TEMPLATE_TEXT, "")
        if not raw_template:
            logger.warning("No raw template text found in state for formatting")
        return FORMATTER_PROMPT.format(raw_template=raw_template)