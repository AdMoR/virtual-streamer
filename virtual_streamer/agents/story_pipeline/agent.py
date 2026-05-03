"""
Story Pipeline Agent.

A three-step sequential pipeline for story generation:
1. story_writer              — generates a free-text story
2. recurrent_location_builder — extracts recurring locations with FluxPrompts
3. detailed_scene_builder     — produces one DetailedScene per scene
                               (ltx_prompt, location, characters, speaker/audio)

The legacy StoryFormatterAgent (step 2 of the old 2-step pipeline) is kept for
the Wav2Lip pipeline but is no longer part of StoryPipelineAgent.

State flow:
    Input:  title (str) in session state
    Step 1: story_writer               → RAW_STORY_TEXT (str)
    Step 2: recurrent_location_builder → RECURRENT_LOCATIONS (JSON str)
    Step 3: detailed_scene_builder     → DETAILED_SCENES (JSON str)
"""

import logging

from google.adk.agents import SequentialAgent

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.story_generator.prompt import StoryInstructionProvider
from virtual_streamer.agents.story_generator.callback import (
    SafetyFlagCheckerCallback,
    SplitStoryCallback,
)
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.agents.story_pipeline.schema import (
    RecurrentLocationsOutput,
    DetailedScenesOutput,
)
from virtual_streamer.agents.story_pipeline.callback import (
    StoreRawStoryCallback,
    StoreRecurrentLocationsCallback,
    StoreDetailedScenesCallback,
)
from virtual_streamer.agents.story_pipeline.prompt import (
    StoryFormatterInstructionProvider,
    RecurrentLocationBuilderInstructionProvider,
    DetailedSceneBuilderInstructionProvider,
)
from virtual_streamer.agents.guardrails_agent.agent import get_story_generation_guardrail_agent

logger = logging.getLogger(__name__)


class StoryWriterAgent(BaseLlmAgent):
    """
    Step 1: generates a free-text story from the title in state.

    Uses StoryInstructionProvider (same as StoryGeneratorAgent) but outputs
    plain text (no schema). The result is stored in state under RAW_STORY_TEXT
    by StoreRawStoryCallback.
    """

    def __init__(self):
        super().__init__(
            name="story_writer",
            instruction=StoryInstructionProvider(),
            output_schema=None,
            before_agent_callback=[SafetyFlagCheckerCallback()],
            after_model_callback=[StoreRawStoryCallback()],
        )


class RecurrentLocationBuilderAgent(BaseLlmAgent):
    """
    Step 2: reads RAW_STORY_TEXT and extracts recurring locations.

    For each distinct location in the story, produces a RecurrentLocation with:
    - location_id (slug), name, and a FluxPrompt for base image generation.

    Result stored as JSON under RECURRENT_LOCATIONS by StoreRecurrentLocationsCallback.
    """

    def __init__(self):
        super().__init__(
            name="recurrent_location_builder",
            instruction=RecurrentLocationBuilderInstructionProvider(),
            output_schema=RecurrentLocationsOutput,
            after_model_callback=[StoreRecurrentLocationsCallback()],
        )


class DetailedSceneBuilderAgent(BaseLlmAgent):
    """
    Step 3: reads RAW_STORY_TEXT + RECURRENT_LOCATIONS and produces one DetailedScene per scene.

    Each DetailedScene carries:
    - ltx_prompt (direct video generation prompt)
    - location (location_id or null)
    - character_on_screen (list of character_ids visible, with teleportation logic)
    - scene_visual_description (FluxPrompt for Flux conditioning image)
    - speaker_id / spoken_line (for TTS audio conditioning)

    Result stored as JSON under DETAILED_SCENES by StoreDetailedScenesCallback.
    """

    def __init__(self):
        super().__init__(
            name="detailed_scene_builder",
            instruction=DetailedSceneBuilderInstructionProvider(),
            output_schema=DetailedScenesOutput,
            after_model_callback=[StoreDetailedScenesCallback()],
        )


class StoryFormatterAgent(BaseLlmAgent):
    """
    Legacy formatter kept for the Wav2Lip pipeline.

    Reads RAW_STORY_TEXT from state and extracts the structured StoryOutput.
    SplitStoryCallback stores STORY_OUTPUT and SENTENCES in state.

    NOT part of StoryPipelineAgent — used only by legacy code paths.
    """

    def __init__(self):
        super().__init__(
            name="story_formatter",
            instruction=StoryFormatterInstructionProvider(),
            output_schema=StoryOutput,
            after_model_callback=[SplitStoryCallback()],
        )


class StoryPipelineAgent(SequentialAgent):
    """
    Sequential agent: story_writer → recurrent_location_builder → detailed_scene_builder.

    State flow:
        Input:  title (str) in session state
        Step 1: story_writer               → RAW_STORY_TEXT
        Step 2: recurrent_location_builder → RECURRENT_LOCATIONS (JSON str)
        Step 3: detailed_scene_builder     → DETAILED_SCENES (JSON str)
    """

    def __init__(self):
        super().__init__(
            name="story_pipeline",
            sub_agents=[
                get_story_generation_guardrail_agent(),
                StoryWriterAgent(),
                RecurrentLocationBuilderAgent(),
                DetailedSceneBuilderAgent(),
            ],
        )


def get_story_pipeline() -> StoryPipelineAgent:
    """Factory function returning a configured StoryPipelineAgent."""
    return StoryPipelineAgent()


root_agent = get_story_pipeline()
