"""
Pydantic schemas for the new 3-step story pipeline.

RecurrentLocationsOutput  — output of RecurrentLocationBuilderAgent
DetailedScenesOutput      — output of DetailedSceneBuilderAgent
"""

from typing import Optional

from pydantic import BaseModel, Field

from virtual_streamer.image_generation.models import FluxPrompt


class RecurrentLocation(BaseModel):
    location_id: str = Field(description="Unique slug identifier, e.g. 'ski-resort'")
    name: str = Field(description="Human-readable location name")
    flux_prompt: FluxPrompt = Field(
        description="Structured Flux prompt for generating the base environment image. Must NOT include any character or person."
    )


class RecurrentLocationsOutput(BaseModel):
    locations: list[RecurrentLocation]


class DetailedScene(BaseModel):
    ltx_prompt: str = Field(
        description="Direct cinematic prompt string given to the LTX video generator. Describe motion and visual action, not static composition."
    )
    location: Optional[str] = Field(
        default=None,
        description="location_id from RecurrentLocationsOutput if the scene is set in a known location, else null."
    )
    character_on_screen: Optional[list[str]] = Field(
        default=None,
        description="List of character_id strings for characters visually present in this scene. Apply teleportation logic: only include characters who were in the previous scene at the same location, or who were explicitly shown traveling here."
    )
    scene_visual_description: FluxPrompt = Field(
        description="Structured FluxPrompt used to generate the Flux conditioning image. Includes environment and character visual details. No dialogue text."
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="character_id of the character who speaks in this scene. Used to select the TTS voice."
    )
    spoken_line: Optional[str] = Field(
        default=None,
        description="Exact text the speaker says. Used for TTS audio generation which conditions LTX generation."
    )


class DetailedScenesOutput(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Refined story title extracted from the raw story."
    )
    scenes: list[DetailedScene]


# ---------------------------------------------------------------------------
# State accessors — typed round-trip helpers to avoid raw-dict issues
# ---------------------------------------------------------------------------

def get_recurrent_locations_from_state(state: dict) -> RecurrentLocationsOutput:
    """Deserialize RECURRENT_LOCATIONS from session state."""
    raw = state.get("recurrent_locations")
    if raw is None:
        raise KeyError("recurrent_locations not found in state")
    if isinstance(raw, str):
        return RecurrentLocationsOutput.model_validate_json(raw)
    return RecurrentLocationsOutput.model_validate(raw)


def get_detailed_scenes_from_state(state: dict) -> DetailedScenesOutput:
    """Deserialize DETAILED_SCENES from session state."""
    raw = state.get("detailed_scenes")
    if raw is None:
        raise KeyError("detailed_scenes not found in state")
    if isinstance(raw, str):
        return DetailedScenesOutput.model_validate_json(raw)
    return DetailedScenesOutput.model_validate(raw)
