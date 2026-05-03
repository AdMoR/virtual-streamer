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

from virtual_streamer.agents.common.state_keys import TITLE, STORY_TEMPLATE_ID, NEWS_CONTEXT
from virtual_streamer.lib.providers.instruction import InstructionProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# META PROMPT - Wraps template prompt and injects variables
# ═══════════════════════════════════════════════════════════════════════════════

META_PROMPT = """{template_prompt}

Characters available (use these exact character_id values):
{characters}

Locations available for this story (assign one location_id to each dialog line):
{locations}

IMPORTANT: Each dialogue line must include a location_id from the list above.
If no locations are listed, invent descriptive location names (they can be registered later).
Keep location consistent: consecutive scenes in the same place must use the same location_id.

Generate a story with exactly {target_lines} dialogue lines.

Scenario: {title}

IMPORTANT: Your response must be structured with three parts:

1. **title**: Create a refined, more complete title for the story (based on the user's input: "{title}")
2. **story_plan**: Describe your overall plan and reasoning for creating this dialog (like a thinking process - what makes this scenario funny, what progression you're following, key elements you're including)
3. **dialog**: The actual dialog lines. Each line must include:
   - **character_id**: Use the exact ID from the characters list above (e.g., "fred", "jamy")
   - **dialog**: The spoken text (what the character says out loud)
   - **location_id**: The location ID for this scene from the locations list above (e.g., "ski-resort")
   - **scene_description**: A structured JSON object describing the visual scene for video generation. Do NOT include the dialog text here. Use the following schema:

```json
{{
  "scene": "Overall environment and context of the scene",
  "subjects": [
    {{
      "description": "Detailed visual description of the subject (appearance, clothing, expression)",
      "pose": "Pose or stance",
      "position": "Where in the frame",
      "color_palette": ["dominant colors for this subject"]
    }}
  ],
  "style": "Artistic/cinematic style (e.g. 'handheld documentary', 'cinematic 35mm')",
  "color_palette": ["dominant colors for the whole scene"],
  "lighting": "Lighting setup and quality",
  "mood": "Emotional tone or atmosphere",
  "background": "Background environment details",
  "composition": "Framing rule or layout (e.g. 'rule of thirds', 'centered')",
  "camera": {{
    "angle": "Camera angle (e.g. 'eye level', 'low angle')",
    "distance": "Shot distance (e.g. 'medium shot', 'close-up')",
    "focus": "What is in focus and how",
    "lens-mm": 35,
    "f-number": "f/2.8",
    "ISO": 400
  }}
}}
```

The scene_description must mainly support the dialog line, not the opposite.
When in a franchise, subjects must use the character names for better context.
Scene consistency: consecutive scenes with the same characters in the same place must share identical scene_description objects.

Example dialog entry:
- character_id: "fred"
- dialog: "Eh dis donc Jamy, ça te dit de faire du surf?"
- scene_description:
```json
{{
  "scene": "Sunset beach with surfboards lined up along the shore",
  "subjects": [
    {{
      "description": "Fred, middle-aged French man with grey hair, wearing a loud Hawaiian shirt, speaking energetically to camera",
      "pose": "Gesturing towards the ocean with both hands",
      "position": "Center frame foreground",
      "color_palette": ["Hawaiian shirt colors", "tanned skin"]
    }}
  ],
  "style": "Handheld documentary feel",
  "color_palette": ["golden sunset orange", "ocean blue", "sandy beige"],
  "lighting": "Warm golden hour backlight with soft fill from the sky",
  "mood": "Enthusiastic and adventurous",
  "background": "Ocean waves and surfers in the distance",
  "composition": "rule of thirds",
  "camera": {{
    "angle": "eye level",
    "distance": "medium shot",
    "focus": "Sharp on Fred, slightly blurred ocean background",
    "lens-mm": 35,
    "f-number": "f/4",
    "ISO": 400
  }}
}}
```


Focus on:
- Making the refined title catchy and descriptive
- In story_plan, explain your creative choices and the comedic arc. Please plan for a short format respecting the size mentioned.
- In dialog, make scene_description specific enough to find visually matching videos
- Each turn of a character should be small, maximum one sentence of 15 words, for a long sentence, spead the sentence into several DialogLine.
- This is very import to create short sentences in each DialogLine.
- Last line of the serie must have a twist.

VISUAL AND AUDIO CONTINUITY NOTES (for the video pipeline that will process this story):
- Locations: name recurring places consistently throughout the story (e.g. always call it "the ski resort" or "the lab"). Do not invent JSON — just refer to them by name naturally.
- Character presence: characters cannot teleport. If a character moves from one place to another, explicitly narrate the transition in the story. A character can only appear in a scene if they physically traveled there. Not every character needs to appear in every scene.
- Each scene has one speaker who says one short line. Make it clear who is speaking each line.
- Each scene should be visually self-contained — describe what is visible in the environment as if it will be turned into a short video clip."""


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

def build_locations_block(locations: list[dict]) -> str:
    """
    Build the {locations} block from Location entities.

    Args:
        locations: List of location dicts with 'location_id' and 'name' keys

    Returns:
        Formatted locations block string, or a placeholder when empty
    """
    if not locations:
        return "No locations defined yet — invent descriptive location names as needed."
    return "\n".join(
        f'- {loc["name"]} (location_id: "{loc["location_id"]}")'
        for loc in locations
    )


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
    locations: Optional[list[dict]] = None,
) -> str:
    """
    Build the final prompt using the meta-prompt system.

    Args:
        template_prompt: Raw prompt text from StoryTemplate
        target_lines: Target number of dialogue lines
        characters: List of character dicts with 'name' and 'description'
        title: Story title/scenario
        locations: Optional list of location dicts with 'location_id' and 'name'

    Returns:
        Complete formatted prompt
    """
    characters_block = build_characters_block(characters)
    locations_block = build_locations_block(locations or [])

    return META_PROMPT.format(
        template_prompt=template_prompt,
        characters=characters_block,
        locations=locations_block,
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
        locations=[],
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

    # Load associated locations
    locations = []
    try:
        loc_rows = await repo.list_locations_by_template(template_id)
        locations = [
            {"location_id": loc["location_id"], "name": loc["name"]}
            for loc in loc_rows
        ]
        if locations:
            logger.info(f"Injecting {len(locations)} location(s) into story prompt")
    except Exception as e:
        logger.warning(f"Could not load locations for template '{template_id}': {e}")

    # Build prompt
    return build_prompt_from_template(
        template_prompt=template["prompt"],
        target_lines=template.get("target_lines", DEFAULT_TARGET_LINES),
        characters=characters,
        title=title,
        locations=locations,
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
    
    If news_context is provided in state:
        - Appends news article context to inspire the story
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
            if prompt is None:
                logger.warning(f"Template '{template_id}' not found, falling back to default")
                prompt = format_story_prompt(title)
        else:
            # Fallback to default prompt
            prompt = format_story_prompt(title)
        
        # Optionally append news context
        news_context = ctx.state.get(NEWS_CONTEXT)
        if news_context:
            logger.info("Adding news context to story generation prompt")
            prompt = self._append_news_context(prompt, news_context)
        
        return prompt

    def _append_news_context(self, base_prompt: str, news_context: str) -> str:
        """
        Append news context to the base prompt.
        
        Args:
            base_prompt: The original prompt
            news_context: Formatted news context string
        
        Returns:
            Prompt with news context appended
        """
        news_section = f"""

---
Actualité de référence pour inspiration:
{news_context}
---

Utilise cette actualité comme point de départ pour créer une parodie humoristique.
Le titre de l'actualité peut servir de base pour le scénario, mais adapte-le 
au style et au ton des personnages."""

        return base_prompt + news_section
