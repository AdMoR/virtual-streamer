"""
Base agent class for Google ADK.

This module provides BaseLlmAgent, a wrapper around Google ADK's LlmAgent
that integrates with the configuration system and provides a standardized
interface for creating agents.

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
"""

import logging
from typing import Any, Callable, List, Optional, Type, Union

from google.adk.agents import LlmAgent
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

    BaseLlmAgent extends Google ADK's LlmAgent with:
    - Automatic configuration loading from YAML files
    - Standardized callback integration
    - Support for dynamic instruction providers
    - Consistent logging and error handling

    Agents should inherit from this class and implement their
    specific behavior through prompts, output schemas, and callbacks.

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
        config: Optional[AgentConfig] = None,
        # Callbacks
        before_agent_callback: Optional[BeforeAgentCallbackType] = None,
        after_agent_callback: Optional[AfterAgentCallbackType] = None,
        before_model_callback: Optional[BeforeModelCallbackType] = None,
        after_model_callback: Optional[AfterModelCallbackType] = None,
        # Sub-agents for multi-agent systems
        sub_agents: Optional[List["BaseLlmAgent"]] = None,
        # Tools
        tools: Optional[List[Any]] = None,
        # Additional LlmAgent kwargs
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
            sub_agents: List of sub-agents for multi-agent systems.
            tools: List of tools available to this agent.
            **kwargs: Additional arguments passed to LlmAgent.
        """
        # Load configuration
        self.agent_config = config or get_config_for_agent(name)

        # Build model string from config
        model = self._build_model_string()

        # Prepare instruction
        resolved_instruction = self._resolve_instruction(instruction)

        # Normalize callbacks to lists
        before_agent = self._normalize_callbacks(before_agent_callback)
        after_agent = self._normalize_callbacks(after_agent_callback)
        before_model = self._normalize_callbacks(before_model_callback)
        after_model = self._normalize_callbacks(after_model_callback)

        # Initialize parent LlmAgent
        super().__init__(
            name=name,
            model=model,
            instruction=resolved_instruction,
            output_schema=output_schema,
            before_agent_callback=before_agent if before_agent else None,
            after_agent_callback=after_agent if after_agent else None,
            before_model_callback=before_model if before_model else None,
            after_model_callback=after_model if after_model else None,
            sub_agents=sub_agents or [],
            tools=tools or [],
            **kwargs,
        )

        logger.info(
            f"Initialized agent '{name}' with model '{model}' "
            f"(version: {self.agent_config.metadata.version})"
        )

    def _build_model_string(self) -> str:
        """Build the model string from configuration.

        Returns:
            Model identifier string for the LLM.
        """
        model_config = self.agent_config.model

        # For Google models, just use the model name directly
        if model_config.provider == "google":
            return model_config.model

        # For other providers, include the provider prefix
        return f"{model_config.provider}/{model_config.model}"

    def _resolve_instruction(
        self,
        instruction: Optional[Union[str, InstructionProvider]],
    ) -> Optional[Union[str, Callable]]:
        """Resolve the instruction to a string or callable.

        Args:
            instruction: Static string or InstructionProvider.

        Returns:
            Resolved instruction for the LlmAgent.
        """
        if instruction is None:
            return None

        if isinstance(instruction, str):
            return instruction

        if isinstance(instruction, InstructionProvider):
            # InstructionProvider is already a callable
            return instruction

        # Assume it's already a callable
        return instruction

    def _normalize_callbacks(
        self,
        callbacks: Optional[
            Union[
                AgentCallback,
                BeforeModelCallback,
                AfterModelCallback,
                Callable,
                List,
            ]
        ],
    ) -> Optional[List[Callable]]:
        """Normalize callbacks to a list.

        Args:
            callbacks: Single callback, list of callbacks, or None.

        Returns:
            List of callbacks or None.
        """
        if callbacks is None:
            return None

        if isinstance(callbacks, list):
            return callbacks

        return [callbacks]

    @property
    def version(self) -> str:
        """Get the agent version from config."""
        return self.agent_config.metadata.version

    @property
    def description(self) -> Optional[str]:
        """Get the agent description from config."""
        return self.agent_config.metadata.description

    @property
    def owners(self) -> List[str]:
        """Get the agent owners from config."""
        return self.agent_config.metadata.owners


def create_agent(
    name: str,
    instruction: Union[str, InstructionProvider],
    output_schema: Optional[Type[BaseModel]] = None,
    **kwargs: Any,
) -> BaseLlmAgent:
    """Factory function to create an agent with minimal boilerplate.

    This is a convenience function for simple agents that don't need
    a custom class. For more complex agents, inherit from BaseLlmAgent.

    Args:
        name: Agent name (used for config loading).
        instruction: System prompt or InstructionProvider.
        output_schema: Optional Pydantic model for structured output.
        **kwargs: Additional arguments passed to BaseLlmAgent.

    Returns:
        Configured BaseLlmAgent instance.

    Example:
        agent = create_agent(
            name="simple_qa",
            instruction="You answer questions concisely.",
            output_schema=QAOutput,
        )
    """
    return BaseLlmAgent(
        name=name,
        instruction=instruction,
        output_schema=output_schema,
        **kwargs,
    )

