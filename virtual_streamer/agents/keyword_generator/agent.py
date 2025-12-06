"""
Keyword Generator Agent.

Generates alternative search keywords for video retrieval when
initial matching fails to find a good match.

This is a standard BaseLlmAgent created via factory function with:
- InstructionProvider that reads sentence and previous keywords from state
- AfterModelCallback that stores the new keyword in namespaced state
"""

import logging
from typing import List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents import (
    BaseLlmAgent,
    AfterModelCallback,
)
from virtual_streamer.lib.agents.callbacks import extract_llm_response_text
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import task_key, keyword_key
from virtual_streamer.agents.keyword_generator.prompt import format_keyword_prompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction Provider
# ═══════════════════════════════════════════════════════════════════════════════


class KeywordInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that reads sentence and previous keywords
    from namespaced state and formats the keyword generation prompt.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the instruction provider.
        
        Args:
            run_id: Unique ID for this keyword generation run
        """
        self.run_id = run_id
    
    async def __call__(self, ctx: ReadonlyContext) -> str:
        """
        Generate the instruction by reading from namespaced state.
        
        Args:
            ctx: Readonly context with access to state
        
        Returns:
            Formatted prompt string
        """
        sentence = ctx.state.get(task_key(self.run_id, "sentence"), "")
        prev_keywords = ctx.state.get(task_key(self.run_id, "prev_keywords"), [])
        
        if not sentence:
            logger.warning(f"No sentence found for run_id={self.run_id}")
        
        return format_keyword_prompt(sentence, prev_keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════


class StoreKeywordCallback(AfterModelCallback):
    """
    Callback that extracts the generated keyword and stores it
    in namespaced state.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the callback.
        
        Args:
            run_id: Unique ID for this keyword generation run
        """
        self.run_id = run_id
    
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """
        Extract keyword and store in namespaced state.
        
        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the LLM
        """
        # Extract the keyword from the response
        keyword = extract_llm_response_text(llm_response).strip()
        
        # Clean up the keyword (remove quotes, extra whitespace, etc.)
        keyword = keyword.strip('"\'').strip()
        
        if not keyword:
            logger.warning(f"Empty keyword generated for run_id={self.run_id}")
            keyword = "video"  # Fallback
        
        # Store in namespaced key
        callback_context.state[keyword_key(self.run_id)] = keyword
        
        logger.debug(f"Generated keyword for {self.run_id}: {keyword}")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════════


def get_keyword_generator(run_id: str) -> BaseLlmAgent:
    """
    Factory function to create a KeywordGeneratorAgent for a specific run.
    
    Args:
        run_id: Unique ID for this keyword generation run
                (e.g., "s0_kw0" for sentence 0, keyword attempt 0)
    
    Returns:
        Configured BaseLlmAgent for keyword generation
    
    Example:
        # Create keyword generator for sentence 0, attempt 1
        kw_gen = get_keyword_generator("s0_kw1")
    """
    return BaseLlmAgent(
        name="keyword_generator",
        instruction=KeywordInstructionProvider(run_id),
        output_schema=None,  # Simple text output, no structured schema
        after_model_callback=[StoreKeywordCallback(run_id)],
    )

