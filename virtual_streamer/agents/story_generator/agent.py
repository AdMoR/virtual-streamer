"""
Story Generator Agent.

Generates structured story from a title using LLM.
This is a standard BaseLlmAgent with:
- InstructionProvider that reads TITLE from state
- AfterModelCallback that splits dialog into sentences
"""

import logging
from functools import lru_cache
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import BaseLlmAgent, AfterModelCallback
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import TITLE, STORY_OUTPUT, SENTENCES
from virtual_streamer.agents.common.utils import separation_fn
from virtual_streamer.agents.story_generator.schema import StoryOutput
from virtual_streamer.agents.story_generator.prompt import format_story_prompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction Provider
# ═══════════════════════════════════════════════════════════════════════════════


class StoryInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that reads the title from state
    and formats the story generation prompt.
    """
    
    async def __call__(self, ctx: ReadonlyContext) -> str:
        """
        Generate the instruction by reading title from state.
        
        Args:
            ctx: Readonly context with access to state
        
        Returns:
            Formatted prompt string
        """
        title = ctx.state.get(TITLE, "")
        if not title:
            logger.warning("No title found in state, using empty title")
        
        return format_story_prompt(title)


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════


class SplitStoryCallback(AfterModelCallback):
    """
    Callback that parses the LLM response and splits the dialog into sentences.
    
    This callback:
    1. Extracts the structured StoryOutput from the LLM response
    2. Splits the dialog into individual sentences using separation_fn
    3. Stores both in the state
    """
    
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """
        Parse story and split into sentences.
        
        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the LLM
        """
        from virtual_streamer.lib.agents.callbacks import extract_llm_response_json
        
        # Parse the structured output
        parsed = extract_llm_response_json(llm_response)
        
        if not parsed:
            logger.error("Failed to parse story output from LLM response")
            callback_context.state[STORY_OUTPUT] = {}
            callback_context.state[SENTENCES] = []
            return
        
        # Store the full story output
        story_output = {
            "title": parsed.get("title", ""),
            "story_plan": parsed.get("story_plan", ""),
            "dialog": parsed.get("dialog", ""),
        }
        callback_context.state[STORY_OUTPUT] = story_output
        
        # Split dialog into sentences
        dialog = parsed.get("dialog", "")
        sentences = separation_fn(dialog)
        callback_context.state[SENTENCES] = sentences
        
        logger.info(
            f"Generated story '{story_output['title']}' with {len(sentences)} sentences"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════


class StoryGeneratorAgent(BaseLlmAgent):
    """
    Agent that generates a story from a title.
    
    Uses:
    - StoryInstructionProvider to read title from state and format prompt
    - StoryOutput schema for structured output
    - SplitStoryCallback to parse response and split into sentences
    """
    
    def __init__(self):
        super().__init__(
            name="story_generator",
            instruction=StoryInstructionProvider(),
            output_schema=StoryOutput,
            after_model_callback=[SplitStoryCallback()],
        )


@lru_cache
def get_story_generator() -> StoryGeneratorAgent:
    """
    Factory function to get the StoryGeneratorAgent singleton.
    
    Returns:
        Configured StoryGeneratorAgent instance
    """
    return StoryGeneratorAgent()

