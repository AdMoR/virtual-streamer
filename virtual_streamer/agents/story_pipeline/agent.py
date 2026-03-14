"""
Story Pipeline Agent.

A two-step sequential pipeline for story generation:
1. story_writer  — generates a free-text story using the same prompt as StoryGeneratorAgent
2. story_formatter — takes the raw text and formats it into a StoryOutput schema

This separates creative generation from structured extraction, which improves
reliability of the final structured output.
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
from virtual_streamer.agents.story_pipeline.callback import StoreRawStoryCallback
from virtual_streamer.agents.story_pipeline.prompt import StoryFormatterInstructionProvider
from virtual_streamer.agents.guardrails_agent.agent import get_story_generation_guardrail_agent
logger = logging.getLogger(__name__)


class StoryWriterAgent(BaseLlmAgent):
    """
    Step 1: generates a free-text story from the title in state.

    Uses the same StoryInstructionProvider as StoryGeneratorAgent but
    outputs plain text (no schema). The result is stored in state under
    RAW_STORY_TEXT by StoreRawStoryCallback.
    """

    def __init__(self):
        super().__init__(
            name="story_writer",
            instruction=StoryInstructionProvider(),
            output_schema=None,
            before_agent_callback=[SafetyFlagCheckerCallback()],
            after_model_callback=[StoreRawStoryCallback()],
        )


class StoryFormatterAgent(BaseLlmAgent):
    """
    Step 2: formats the raw story text into a structured StoryOutput.

    Reads RAW_STORY_TEXT from state (set by StoryWriterAgent) and asks the
    LLM to extract the structured fields. SplitStoryCallback then stores
    STORY_OUTPUT and SENTENCES in state for downstream agents.
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
    Sequential agent that chains StoryWriterAgent → StoryFormatterAgent.

    State flow:
        Input:  title (str) in session state
        Step 1: story_writer  → RAW_STORY_TEXT (str)
        Step 2: story_formatter → STORY_OUTPUT (dict), SENTENCES (list)
    """

    def __init__(self):
        super().__init__(
            name="story_pipeline",
            sub_agents=[
                get_story_generation_guardrail_agent(),
                StoryWriterAgent(),
                StoryFormatterAgent(),
            ],
        )


def get_story_pipeline() -> StoryPipelineAgent:
    """Factory function returning a configured StoryPipelineAgent."""
    return StoryPipelineAgent()


root_agent = get_story_pipeline()