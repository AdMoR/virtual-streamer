import logging

from virtual_streamer.agents.guardrails_agent.prompt import PROMPT
from virtual_streamer.agents.story_generator.prompt import META_PROMPT
from virtual_streamer.agents.guardrails_agent.schema import GuardrailsOutput
from virtual_streamer.lib.agents import BaseLlmAgent

logger = logging.getLogger(__name__)


class GuardrailAgent(BaseLlmAgent):

    def __init__(
            self,
            agent_name: str,
            agent_context: str
    ):
        """
        Initialize the Greeting Jesus Agent.
        """
        super().__init__(
            name="guardrail_agent",
            instruction=PROMPT.format(agent_type=agent_name, agent_context=agent_context),
            output_schema=GuardrailsOutput,
        )


def get_story_generation_guardrail_agent() -> BaseLlmAgent:
    return GuardrailAgent("StoryGenerator", META_PROMPT.replace("{", "[").replace("}", "]"))


root_agent = get_story_generation_guardrail_agent()
