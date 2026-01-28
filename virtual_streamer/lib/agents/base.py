"""
Base agent class for Google ADK.

This module provides BaseLlmAgent, a wrapper around Google ADK's LlmAgent
that integrates with the configuration system and provides a standardized
interface for creating agents.

The agent uses LiteLlm for multi-provider LLM support, allowing you to
switch between providers (Google, Anthropic, OpenAI, Azure, etc.) by
changing only the YAML configuration.

Example usage:

    from lib.agents import BaseLlmAgent
    from lib.config import get_config_for_agent
    from pydantic import BaseModel, Field

    class MyOutput(BaseModel):
        answer: str = Field(description="The response")

    class MyAgent(BaseLlmAgent):
        def __init__(self):
            super().__init__(
                name="my_agent",
                instruction="You are a helpful assistant.",
                output_schema=MyOutput,
            )

    # Factory function pattern
    def get_my_agent():
        return MyAgent()

    root_agent = get_my_agent()

Configuration (configs/agents/my_agent.yaml):

    name: my_agent
    model:
      provider: anthropic
      model: anthropic/claude-sonnet-4-5-20250929
      parameters:
        temperature: 0.0
        max_output_tokens: 4096
        seed: 42
    metadata:
      version: "1.0.0"
      description: "My helpful assistant agent"
"""

import logging
from typing import Any, Callable, List, Optional, Type, Union

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel

from virtual_streamer.lib.agents.callbacks import (
    AfterModelCallback,
    AgentCallback,
    BeforeModelCallback,
)
from virtual_streamer.lib.config.settings import AgentConfig, get_config_for_agent
from virtual_streamer.lib.providers.instruction import InstructionProvider

logger = logging.getLogger(__name__)


# Type aliases for callbacks
BeforeAgentCallbackType = Union[AgentCallback, Callable, List[Union[AgentCallback, Callable]]]
AfterAgentCallbackType = Union[AgentCallback, Callable, List[Union[AgentCallback, Callable]]]
BeforeModelCallbackType = Union[
    BeforeModelCallback, Callable, List[Union[BeforeModelCallback, Callable]]
]
AfterModelCallbackType = Union[
    AfterModelCallback, Callable, List[Union[AfterModelCallback, Callable]]
]


class BaseLlmAgent(LlmAgent):
    """
    Base class for ADK agents with configuration injection.

    Attributes:
        agent_config: The loaded AgentConfig for this agent.

    Example:
        class QuestionAnswerAgent(BaseLlmAgent):
            def __init__(self):
                super().__init__(
                    name="qa_agent",
                    instruction=QA_PROMPT,
                    output_schema=QAOutput,
                    after_model_callback=[StoreAnswerCallback()],
                )

        # Always use factory pattern
        @lru_cache
        def get_qa_agent():
            return QuestionAnswerAgent()

        root_agent = get_qa_agent()
    """

    def __init__(
        self,
        name: str,
        instruction: Optional[Union[str, InstructionProvider]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        before_agent_callback: Optional[BeforeAgentCallbackType] = None,
        after_agent_callback: Optional[AfterAgentCallbackType] = None,
        before_model_callback: Optional[BeforeModelCallbackType] = None,
        after_model_callback: Optional[AfterModelCallbackType] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ):
        """Initialize the base agent.

        Args:
            name: Unique agent name. Must match the config file name if using
                  file-based configuration (e.g., "my_agent" -> configs/agents/my_agent.yaml).
            instruction: The agent's system prompt. Can be a static string or
                         an InstructionProvider for dynamic prompts.
            output_schema: Pydantic model for structured output. The LLM will
                          generate responses conforming to this schema.
            config: Optional AgentConfig. If not provided, will be loaded
                    automatically based on the agent name.
            before_agent_callback: Callback(s) to run when agent starts.
            after_agent_callback: Callback(s) to run when agent completes.
            before_model_callback: Callback(s) to run before LLM call.
            after_model_callback: Callback(s) to run after LLM returns.
            tools: List of tools available to this agent.
            **kwargs: Additional arguments passed to LlmAgent.
        """
        # Load configuration
        agent_config = get_config_for_agent(name)

        # Build LiteLlm model with parameters from config
        model = self._build_model(agent_config)
        model_string = agent_config.model.get_model_string()

        logger.info(f"Agent {name} config: {model_string}")

        # Initialize parent LlmAgent
        super().__init__(
            name=name,
            model=model,
            instruction=instruction,
            output_schema=output_schema,
            before_agent_callback=before_agent_callback if before_agent_callback else None,
            after_agent_callback=after_agent_callback if after_agent_callback else None,
            before_model_callback=before_model_callback if before_model_callback else None,
            after_model_callback=after_model_callback if after_model_callback else None,
            tools=tools or [],
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True,
            **kwargs,
        )

        # Log model and parameters for debugging
        params = agent_config.model.parameters.get_non_null_params()
        params_str = f", params={params}" if params else ""
        logger.info(
            f"Initialized agent '{name}' with model '{model_string}'{params_str} "
            f"(version: {agent_config.metadata.version})"
        )

    def _build_model(self, agent_config: AgentConfig) -> LiteLlm:
        """Build the LiteLlm model wrapper with configuration parameters.

        Creates a LiteLlm instance that provides a unified interface for
        multiple LLM providers (Google, Anthropic, OpenAI, Azure, etc.).
        Returns:
            Configured LiteLlm instance for the agent.

        Example:
            For a config with:
                model:
                  provider: anthropic
                  model: anthropic/claude-sonnet-4-5-20250929
                  parameters:
                    temperature: 0.0
                    max_output_tokens: 4096
                    seed: 42

            This creates:
                LiteLlm(
                    model="anthropic/claude-sonnet-4-5-20250929",
                    temperature=0.0,
                    max_output_tokens=4096,
                    seed=42,
                )
        """
        model_config = agent_config.model

        # Get the model string (handles provider prefix logic)
        model_string: str = model_config.get_model_string()

        # Get only explicitly set parameters (exclude None values)
        llm_kwargs = model_config.parameters.get_non_null_params()

        # Create LiteLlm with model and parameters
        return LiteLlm(model=model_string, **llm_kwargs)
