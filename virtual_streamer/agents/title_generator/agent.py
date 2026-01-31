"""Title Generator Agent."""
import logging
from virtual_streamer.agents.title_generator.prompt import TitleInstructionProvider
from virtual_streamer.agents.title_generator.schema import TitlesOutput
from virtual_streamer.lib.agents import BaseLlmAgent

logger = logging.getLogger(__name__)


class TitleGeneratorAgent(BaseLlmAgent):
    """
    Agent that generates creative titles for a story template.

    Uses:
    - TitleInstructionProvider to load template and build prompt
    - TitlesOutput schema for structured JSON output
    - No callback needed - output is parsed after Runner completes
    """

    def __init__(self, count: int = 50):
        super().__init__(
            name="title_generator",
            instruction=TitleInstructionProvider(count=count),
            output_schema=TitlesOutput,
            after_model_callback=[],  # No callback - parse JSON after Runner
        )


def get_title_generator(count: int = 50) -> TitleGeneratorAgent:
    """Factory function to create a TitleGeneratorAgent."""
    return TitleGeneratorAgent(count=count)
