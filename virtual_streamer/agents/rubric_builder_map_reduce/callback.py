"""
Callbacks for Stateful Rubric Builder Agent.

These callbacks handle:
- InjectStoriesCallback: Reads story batch from state, formats as user message
- StoreRubricsCallback: Parses LLM response and stores rubrics in state
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from virtual_streamer.lib.agents import (
    StateInputCallback,
    StateOutputCallback,
    extract_llm_response_json,
)
from virtual_streamer.agents.rubric_builder_agent.schema import (
    StoryBatchInput,
    MapPhaseOutput,
)

logger = logging.getLogger(__name__)


STORY_INTRO = """These stories are fake articles coming from Legorafi.fr, a satirical news source.

"""

STORY_SEPARATOR = "\n======\n"


def format_story_batch(stories: StoryBatchInput) -> str:
    """
    Format a batch of stories for the LLM prompt.
    
    Args:
        stories: StoryBatchInput containing list of StoryItem
        
    Returns:
        Formatted string with introduction and stories separated by ======
    """
    parts = [STORY_INTRO]
    
    for story in stories.stories:
        story_text = f"""Title: {story.title}
Subtitle: {story.subtitle}
Body: {chr(10).join(story.body)}"""
        parts.append(story_text)
    
    return STORY_SEPARATOR.join(parts)


class InjectStoriesCallback(StateInputCallback):
    """
    Callback that injects the story batch into the LLM request as user message.

    Reads the story batch from the namespaced state key, formats it with
    introduction and separators, and injects it into the LLM request.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run (e.g., "w0")
        """
        super().__init__(
            input_key="story_batch",
            input_schema=StoryBatchInput,
            run_id=run_id,
        )

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        """
        Inject the formatted stories into the LLM request.

        Reads story batch from state, formats with introduction and separators,
        and appends to the LLM request as user message.

        Args:
            callback_context: Context with access to mutable state
            llm_request: The LLM request to modify

        Returns:
            None to continue with LLM call

        Raises:
            Exception: If story batch is missing from state
        """
        input_key = self.get_input_key()
        
        # Read from state
        state_data = callback_context.state.get(input_key)
        
        if not state_data:
            raise Exception(f"Could not find story batch at key '{input_key}' in state")
        
        # Parse the story batch
        story_batch = self.input_schema.model_validate_json(state_data)
        
        if not story_batch.stories:
            raise Exception(f"Story batch at key '{input_key}' is empty")
        
        # Format the stories
        formatted_stories = format_story_batch(story_batch)
        
        # Append to request contents as user message
        llm_request.contents[0].parts.append(
            types.Part.from_text(text=formatted_stories)
        )
        
        logger.info(
            f"Injected {len(story_batch.stories)} stories from key '{input_key}'"
        )
        
        return None


class StoreRubricsCallback(StateOutputCallback):
    """
    Callback that parses the LLM response and stores the rubrics in state.
    
    The output schema is MapPhaseOutput which contains a list of Rubric objects.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run (e.g., "w0")
        """
        super().__init__(
            output_key="rubrics",
            output_schema=MapPhaseOutput,
            run_id=run_id,
        )

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """
        Parse rubrics from LLM response and store in state.

        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the LLM

        Returns:
            None to keep original response
        """
        # Parse the LLM output into MapPhaseOutput
        logger.info(f"LLM response: {llm_response.content}")
        
        llm_output: MapPhaseOutput = extract_llm_response_json(
            llm_response, MapPhaseOutput
        )
        
        if llm_output is None:
            logger.warning("Failed to parse rubrics from LLM response")
            return None
        
        # Store the rubrics in state
        output_key = self.get_output_key()
        callback_context.state[output_key] = llm_output.model_dump_json()
        
        logger.info(
            f"Stored {len(llm_output.rubrics)} rubrics at '{output_key}'"
        )
        
        return None
