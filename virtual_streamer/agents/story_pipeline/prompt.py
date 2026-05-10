"""
Story Pipeline Prompts.

StoryFormatterInstructionProvider       — kept for the legacy Wav2Lip pipeline
RecurrentLocationBuilderInstructionProvider — step 2 of the new 3-step LTX pipeline
DetailedSceneBuilderInstructionProvider     — step 3 of the new 3-step LTX pipeline
"""

import json
import logging

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import RAW_STORY_TEXT, RECURRENT_LOCATIONS

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy formatter prompt (kept for Wav2Lip pipeline / StoryFormatterAgent)
# ═══════════════════════════════════════════════════════════════════════════════

FORMATTER_PROMPT = """You are a structured data extractor.

You will receive the raw text of a story .
Your sole task is to extract and reformat this story into the required structured output — do NOT rewrite or alter the content.

## Extract the following fields:
- **title**: The refined story title.
- **story_plan**: The overall creative plan and reasoning described in the story (the thinking section).
- **dialog**: The list of dialogue lines. For each line extract:
  - **character_id**: The optional character identifier (e.g. "fred", "jamy"). Use lowercase.
  - **text**: The spoken dialogue text (what the character says out loud).
  - **scene_description**: Describe the action, characters and scene appearance
  - **location_id**: The location identifier for this scene (e.g. "ski-resort", "medieval-castle").
    You MAY re-use the location_id values present in the story text, but only if it fits the story. Please consider the story_template instruction for how to select or create a new location. 
- **locations**



## What Works Well for the scene description 
Strength	            Description
Cinematic compositions	Wide, medium, and close-up shots with thoughtful lighting, shallow depth of field, and natural motion
Emotive human moments	Strong single-subject emotional expressions, subtle gestures, and facial nuance
Atmosphere & setting	Fog, mist, golden-hour light, rain, reflections, ambient textures
Clear camera language	Explicit instructions like “slow dolly in” or “handheld tracking”
Stylized aesthetics	    Painterly, noir, analog film, fashion editorial, pixelated animation
Lighting & mood control	Backlighting, color palettes, rim light, flickering lamps


Example of scene_description : 
EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST The shot opens on a news reporter standing in front of a row of cordoned-off cars, 
yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air.
 The reporter, composed but visibly excited, looks directly into the camera, microphone in hand. Reporter (live): 
 “Thank you, Sylvia. And yes — this is a sentence I never thought I’d say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!” He gestures slightly toward the field behind him. Reporter (grinning): “If my cameraman can pan over, you’ll see what all the excitement’s about.” The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos. Reporter (off-screen, shouting over the noise): “There it is, folks — the moment New Castle will never forget!” The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil.


Raw story:
{raw_story}"""


class StoryFormatterInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider for story_formatter (legacy Wav2Lip pipeline).

    Reads RAW_STORY_TEXT from state (set by story_writer) and injects it
    into the formatting prompt.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_story = ctx.state.get(RAW_STORY_TEXT, "")
        if not raw_story:
            logger.warning("No raw story text found in state for formatting")
        return FORMATTER_PROMPT.format(raw_story=raw_story)


# ═══════════════════════════════════════════════════════════════════════════════
# RecurrentLocationBuilder prompt
# ═══════════════════════════════════════════════════════════════════════════════

FLUX_PROMPT_SCHEMA = """{
  "scene": "Overall environment and context (no characters)",
  "subjects": [],
  "style": "Cinematic/photographic style",
  "color_palette": ["dominant scene colors"],
  "lighting": "Lighting setup and quality",
  "mood": "Emotional tone or atmosphere",
  "background": "Background environment details",
  "composition": "Framing rule (e.g. rule of thirds)",
  "camera": {
    "angle": "Camera angle (e.g. eye level, low angle)",
    "distance": "Shot distance (e.g. wide shot, medium shot)",
    "focus": "What is in focus",
    "lens-mm": 35,
    "f-number": "f/5.6",
    "ISO": 400
  }
}"""

RECURRENT_LOCATION_BUILDER_PROMPT = """You are a visual location designer for a video generation pipeline.

You will receive the raw text of a story. Your task is to identify all distinct, recurring locations (sets/environments) that appear in the story and produce a structured description for each so that a base image can be generated for them.

A "recurring location" is any place that:
- Appears more than once across scenes, OR
- Serves as a clearly distinct named environment (e.g. "the lab", "the ski resort", "the café")

For each location produce:
- **location_id**: A unique lowercase slug with hyphens instead of spaces (e.g. "ski-resort", "fred-lab", "paris-cafe")
- **name**: Human-readable name (e.g. "Ski Resort", "Fred's Lab", "Paris Café")
- **flux_prompt**: A structured JSON object describing the environment for image generation.
  IMPORTANT: The flux_prompt must describe ONLY the environment — NO characters, NO people, NO persons.
  Use the following schema:

{flux_schema}

Output JSON conforming to this structure:
{{
  "locations": [
    {{
      "location_id": "...",
      "name": "...",
      "flux_prompt": {{ ...FluxPrompt object... }}
    }}
  ]
}}

Raw story:
{raw_story}"""


class RecurrentLocationBuilderInstructionProvider(InstructionProvider):
    """
    Reads RAW_STORY_TEXT from state and asks the LLM to identify all recurring
    locations and produce a RecurrentLocationsOutput.
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_story = ctx.state.get(RAW_STORY_TEXT, "")
        if not raw_story:
            logger.warning("No raw story text found in state for location building")
        return RECURRENT_LOCATION_BUILDER_PROMPT.format(
            flux_schema=FLUX_PROMPT_SCHEMA,
            raw_story=raw_story,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DetailedSceneBuilder prompt
# ═══════════════════════════════════════════════════════════════════════════════

DETAILED_SCENE_BUILDER_PROMPT = """You are a scene planning AI for a video generation pipeline.

You will receive:
1. The raw story text
2. A list of recurring locations already identified for this story

Your task is to produce one DetailedScene object per scene in the story.

Available locations:
{locations_block}

For each scene produce:
- **ltx_prompt**: A direct, self-contained cinematic video generation prompt for the LTX model.
  Describe motion, action, and visual dynamics — not static composition.
  Example: "Fred gestures excitedly toward a snowy mountain slope, his breath visible in cold air, camera following his movement at medium distance."

- **location**: The location_id from the list above if the scene is set in a known location, else null.

- **character_on_screen**: A JSON array of character_id strings for characters visually present in this scene.
  Apply strict teleportation logic:
  - A character can only appear in a scene if they were in the PREVIOUS scene at the SAME location,
    OR if the story explicitly described them moving/traveling to this location.
  - Not all characters need to be on screen in every scene.
  - If a character is speaking off-camera or not shown, do NOT include them.
  Example: ["fred"] or ["fred", "jamy"] or null if no character is visible.

- **scene_visual_description**: A FluxPrompt JSON object for generating the Flux conditioning image.
  Include environment + visible character visual details. Do NOT include dialogue text.
  Schema:
{flux_schema}

- **speaker_id**: The character_id of whoever speaks in this scene (e.g. "fred"), or null if no one speaks.

- **spoken_line**: The exact dialogue text the speaker says, or null if no dialogue.

- **title**: The refined story title (extract from the raw story — include this only in the first scene object and set to null for all others).

Output JSON conforming to:
{{
  "title": "...",
  "scenes": [
    {{
      "ltx_prompt": "...",
      "location": "..." or null,
      "character_on_screen": ["character_id", ...] or null,
      "scene_visual_description": {{ ...FluxPrompt... }},
      "speaker_id": "..." or null,
      "spoken_line": "..." or null
    }}
  ]
}}

Raw story:
{raw_story}"""


class DetailedSceneBuilderInstructionProvider(InstructionProvider):
    """
    Reads RAW_STORY_TEXT and RECURRENT_LOCATIONS from state, then asks the LLM
    to produce a DetailedScenesOutput (one DetailedScene per story scene).
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        raw_story = ctx.state.get(RAW_STORY_TEXT, "")
        if not raw_story:
            logger.warning("No raw story text found in state for scene building")

        raw_locations = ctx.state.get(RECURRENT_LOCATIONS)
        if raw_locations:
            try:
                if isinstance(raw_locations, str):
                    locations_data = json.loads(raw_locations)
                else:
                    locations_data = raw_locations
                locations_block = json.dumps(locations_data, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"Could not parse recurrent locations from state: {exc}")
                locations_block = "No locations defined."
        else:
            logger.warning("No recurrent locations found in state for scene building")
            locations_block = "No locations defined."

        return DETAILED_SCENE_BUILDER_PROMPT.format(
            locations_block=locations_block,
            flux_schema=FLUX_PROMPT_SCHEMA,
            raw_story=raw_story,
        )
