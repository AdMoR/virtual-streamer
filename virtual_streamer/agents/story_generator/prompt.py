"""
Story Generation Prompt Builder.

Supports two modes:
1. Template-based: Loads StoryTemplate from DB and builds prompt via meta-prompt
2. Fallback: Uses hardcoded default prompt for backward compatibility

The two-level prompt system:
- Template prompt: Raw text stored in DB (tone, rules, examples - story-specific)
- Meta prompt: Hardcoded in code, wraps template and injects {title}, {target_lines}, {characters}
"""

import logging
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.agents.common.state_keys import TITLE, STORY_TEMPLATE_ID
from virtual_streamer.lib.providers.instruction import InstructionProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# META PROMPT - Wraps template prompt and injects variables
# ═══════════════════════════════════════════════════════════════════════════════

META_PROMPT = """{template_prompt}

Characters available (use these exact character_id values):
{characters}

Generate a story with exactly {target_lines} dialogue lines.

Scenario: {title}

IMPORTANT: Your response must be structured with three parts:

1. **title**: Create a refined, more complete title for the story (based on the user's input: "{title}")
2. **story_plan**: Describe your overall plan and reasoning for creating this dialog (like a thinking process - what makes this scenario funny, what progression you're following, key elements you're including)
3. **dialog**: The actual dialog lines. Each line must include:
   - **character_id**: Use the exact ID from the characters list above (e.g., "fred", "jamy")
   - **dialog**: The spoken text (what the character says out loud)
   - **scene_description**: A visual description of the scene that will be used to search for matching video clips. Describe what should be visible: location, actions, objects, mood. Do NOT include the dialog text here.

Example dialog entry:
- character_id: "fred"
- dialog: "Eh dis donc Jamy, ça te dit de faire du surf?"
- scene_description: "A person talking enthusiastically to the camera in a beach setting with surfboards visible in the background"

Focus on:
- Making the refined title catchy and descriptive
- In story_plan, explain your creative choices and the comedic arc. Please plan for a short format respecting the size mentioned.
- In dialog, make scene_description specific enough to find visually matching videos
- Each turn of a character should be small, maximum one sentence of 15 words, for a long sentence, spead the sentence into several DialogLine.
- This is very import to create short sentences in each DialogLine.
- Last line of the serie must have a twist."""


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TEMPLATE PROMPT - Used when no story_template_id is provided
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TEMPLATE_PROMPT = """You are a designer of a humorous parody of C'est pas Sorcier the French Science discovery show.
This humorous version is set in the present days, 20 years after the last airing of C'est pas Sorcier and Fred and Jamy the 2 main presenters of the show are still working together.
Fred is always the main actor of the story.

Write a humorous parody in the style of the French educational TV show "C'est pas Sorcier," featuring the characters Fred and Jamy. The story should follow this structure:

Core Elements:

• Fred's Character: An overconfident, grandiose entrepreneur from the 1990s who constantly pitches half-baked business ideas with unwarranted certainty. He frequently references past glories from the show's heyday and uses casual French slang ("flouze," "pécho," "kiffer," "bizness").

• Jamy's Role: A passive, skeptical listener who serves as the straight man, receiving Fred's wild ideas without much pushback or resistance. As this is mainly a Fred's monologue, a Jamy dialogue line could only be at the start to launch Fred on a topic or at the end to acknowledge or be surprised. Jamy does not speak in the dialogue more than ONCE!

The dialogues are only dialogue lines and cannot contain descriptive details that are not verbally spoken.

Story Arc:

1. Fred discovers something new and pitches an idea to Jamy

2. The idea escalates into increasingly ridiculous territory with specific, oddly concrete details

3. Fred justifies the absurdity with pseudo-scientific reasoning or faux-logic borrowed from the show's educational format

4. Include nostalgic callbacks to 1990s like shows (ex: la roue de la fortune, Question pour un Champion), specific locations (ex: Bourg-en-Gonesse, Rennes), or dated technology (ex: Nokia 3310) ==> please do not re-use these specific examples but extrapolate from them.

4 bis. Reference can sometimes mention current events (current president in France or US, well known internet celebrities, etc) but mostly for a roasting joke. Use them more seldomly (like one per scenario)

5. The humor maintains affectionate absurdity rather than meanness

6. The character don't explain their joke.

Tone elements:

• Fred is overly excited by his brand new idea and overlooks the absurdity of his idea. The text is told as Fred is speaking to Jamy, but the video is seen from Jamy's point of view, Fred speaks to the camera while showing things.

• Fred may explain how combining their strengths would yield an incredible advantage in the newly found endeavor

• Fred has a very casual language and can swear or be mean to illustrate better his ideas

• Fred feels superior to the majority of the other people, even the most talented, he will often put himself and Jamy over the rest of the population

• Use authentic French cultural references and slang

• Make it accessible to both nostalgic fans and newcomers

• Avoid out of context reference

Other rules:

This script will be used to generate a humoristic video. Each entry like "Fred: ......" will be used for a single sequence.
A sequence should convey a single idea, where Fred is doing a single Action.

The story should be fast paced, one idea per sentence."""

DEFAULT_TARGET_LINES = 6
DEFAULT_CHARACTERS = "- Fred: Bombastic host of C'est pas Sorcier, overconfident entrepreneur\n- Jamy: Scientist and skeptical listener"


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_characters_block(characters: list[dict]) -> str:
    """
    Build the {characters} block from Character entities.
    
    Args:
        characters: List of character dicts with 'character_id', 'name' and 'description' keys
    
    Returns:
        Formatted character block string with character_id clearly shown
    """
    lines = []
    for char in characters:
        char_id = char.get("character_id", char.get("name", "unknown").lower())
        name = char.get("name", "Unknown")
        desc = char.get("description", "")
        if desc:
            lines.append(f"- {name} (character_id: \"{char_id}\"): {desc}")
        else:
            lines.append(f"- {name} (character_id: \"{char_id}\")")
    return "\n".join(lines)


def build_prompt_from_template(
    template_prompt: str,
    target_lines: int,
    characters: list[dict],
    title: str,
) -> str:
    """
    Build the final prompt using the meta-prompt system.
    
    Args:
        template_prompt: Raw prompt text from StoryTemplate
        target_lines: Target number of dialogue lines
        characters: List of character dicts with 'name' and 'description'
        title: Story title/scenario
    
    Returns:
        Complete formatted prompt
    """
    characters_block = build_characters_block(characters)
    
    return META_PROMPT.format(
        template_prompt=template_prompt,
        characters=characters_block,
        target_lines=target_lines,
        title=title,
    )


def format_story_prompt(title: str) -> str:
    """
    Format the default story generation prompt with the given title.
    
    This is the fallback when no story_template_id is provided.
    
    Args:
        title: The scenario/title for story generation
    
    Returns:
        Formatted prompt string
    """
    return build_prompt_from_template(
        template_prompt=DEFAULT_TEMPLATE_PROMPT,
        target_lines=DEFAULT_TARGET_LINES,
        characters=[
            {"character_id": "fred", "name": "Fred", "description": "Bombastic host of C'est pas Sorcier, overconfident entrepreneur"},
            {"character_id": "jamy", "name": "Jamy", "description": "Scientist and skeptical listener"},
        ],
        title=title,
    )


async def load_template_and_build_prompt(template_id: str, title: str) -> Optional[str]:
    """
    Load a story template from DB and build the prompt.
    
    Args:
        template_id: ID of the story template to load
        title: Story title/scenario
    
    Returns:
        Formatted prompt string, or None if template not found
    """
    from virtual_streamer.utils.entity_repository import get_entity_repository
    
    repo = get_entity_repository()
    
    # Load template
    template = await repo.get_story_template(template_id)
    if template is None:
        logger.warning(f"Story template '{template_id}' not found")
        return None
    
    # Load associated characters
    characters = []
    for char_id in template.get("character_ids", []):
        char = await repo.get_character(char_id)
        if char:
            characters.append({
                "character_id": char_id,
                "name": char.get("name", char_id),
                "description": char.get("description", ""),
            })
        else:
            logger.warning(f"Character '{char_id}' referenced by template not found")
    
    # Build prompt
    return build_prompt_from_template(
        template_prompt=template["prompt"],
        target_lines=template.get("target_lines", DEFAULT_TARGET_LINES),
        characters=characters,
        title=title,
    )


class StoryInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that reads title and optional story_template_id
    from state and builds the appropriate prompt.
    
    If story_template_id is provided:
        - Loads StoryTemplate from DB
        - Loads associated Characters
        - Builds prompt via meta-prompt system
    
    Otherwise:
        - Uses default hardcoded C'est pas Sorcier prompt
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        """
        Generate the instruction by reading title and template from state.

        Args:
            ctx: Readonly context with access to state

        Returns:
            Formatted prompt string
        """
        title = ctx.state.get(TITLE, "")
        if not title:
            logger.warning("No title found in state, using empty title")
        
        # Check if a template ID is specified
        template_id = ctx.state.get(STORY_TEMPLATE_ID)
        
        if template_id:
            logger.info(f"Using story template: {template_id}")
            prompt = await load_template_and_build_prompt(template_id, title)
            if prompt:
                return prompt
            else:
                logger.warning(f"Template '{template_id}' not found, falling back to default")
        
        # Fallback to default prompt
        return format_story_prompt(title)
