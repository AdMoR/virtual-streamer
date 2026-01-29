import logging
from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.greeting_jesus_agent.prompt import PROMPT


logger = logging.getLogger(__name__)


class GreetingJesusAgent(BaseLlmAgent):

    def __init__(
            self,
    ):
        """
        Initialize the Greeting Jesus Agent.
        """
        super().__init__(
            name="greeting_jesus_agent",
            instruction=PROMPT,
            output_schema=None,
        )


def get_greeting_jesus_agent() -> BaseLlmAgent:
    return GreetingJesusAgent()


root_agent = get_greeting_jesus_agent()
