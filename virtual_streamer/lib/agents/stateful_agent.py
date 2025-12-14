"""
Stateful LLM Agent for dynamic parallel workflows.

This module provides StatefulLlmAgent, an extension of BaseLlmAgent that
exposes input/output keys from its callbacks. This enables the
DynamicParallelAgent to discover where to write inputs and read outputs
without manual key management.

Usage:
    from virtual_streamer.lib.agents import StatefulLlmAgent
    
    def get_my_agent(run_id: str | None = None) -> StatefulLlmAgent:
        return StatefulLlmAgent(
            name="my_agent",
            instruction="Do something useful",
            input_callback=MyInputCallback(run_id),
            output_callback=MyOutputCallback(run_id),
        )
    
    # In DynamicParallelAgent:
    worker = get_my_agent(run_id="s0_w1")
    input_key = worker.get_input_key()  # "task:s0_w1:my_input"
    output_key = worker.get_output_key()  # "result:s0_w1:my_output"
"""

import logging
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from virtual_streamer.lib.agents.base import BaseLlmAgent
from virtual_streamer.lib.agents.stateful_callbacks import (
    StateInputCallback,
    StateOutputCallback,
)
from virtual_streamer.lib.providers.instruction import InstructionProvider

logger = logging.getLogger(__name__)


class StatefulLlmAgent(BaseLlmAgent):
    """
    LLM Agent that exposes input/output keys from its callbacks.
    
    This agent wraps StateInputCallback and StateOutputCallback instances,
    delegating key discovery to them. The DynamicParallelAgent can then
    query these keys to know where to write task inputs and read results.
    
    Attributes:
        _input_callback: The StateInputCallback for reading input from state
        _output_callback: The StateOutputCallback for writing output to state
    
    Example:
        # Define a factory function
        def get_video_matcher(run_id: str | None = None) -> StatefulLlmAgent:
            return StatefulLlmAgent(
                name="video_matcher",
                instruction=JUDGE_PROMPT,
                input_callback=InjectVisionFrameCallback(run_id),
                output_callback=StoreJudgementCallback(run_id),
            )
        
        # Use in DynamicParallelAgent
        worker = get_video_matcher("s0_w1")
        print(worker.get_input_key())   # "task:s0_w1:video_sentence"
        print(worker.get_output_key())  # "result:s0_w1:judgement"
    """
    
    # Store callbacks as regular attributes (not Pydantic fields)
    _input_callback: StateInputCallback
    _output_callback: StateOutputCallback
    
    def __init__(
        self,
        name: str,
        input_callback: StateInputCallback,
        output_callback: StateOutputCallback,
        instruction: Optional[Union[str, InstructionProvider]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the stateful LLM agent.
        
        Args:
            name: Unique agent name (must match config file if using file-based config)
            input_callback: StateInputCallback for reading input from state
            output_callback: StateOutputCallback for writing output to state
            instruction: System prompt (static string or InstructionProvider)
            output_schema: Optional Pydantic model for structured output
            tools: Optional list of tools available to this agent
            **kwargs: Additional arguments passed to BaseLlmAgent
        """
        # Store callbacks before super().__init__ since LlmAgent uses Pydantic
        # and may process attributes during initialization
        super().__init__(
            name=name,
            instruction=instruction,
            output_schema=output_schema,
            before_model_callback=input_callback,
            after_model_callback=output_callback,
            tools=tools,
            **kwargs,
        )
        
        # Store callbacks after super().__init__ for our delegation methods
        object.__setattr__(self, '_input_callback', input_callback)
        object.__setattr__(self, '_output_callback', output_callback)
        
        logger.debug(
            f"StatefulLlmAgent '{name}' initialized with "
            f"input_key='{self.get_input_key()}', "
            f"output_key='{self.get_output_key()}'"
        )
    
    def get_input_key(self) -> str:
        """
        Get the state key where input should be written.
        
        Delegates to the input callback's get_input_key method.
        
        Returns:
            The full namespaced input key (e.g., "task:s0_w1:video_sentence")
        """
        return self._input_callback.get_input_key()
    
    def get_input_schema(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for input validation.
        
        Delegates to the input callback's get_input_schema method.
        This enables the DynamicParallelAgent to validate items before
        writing them to state.
        
        Returns:
            The Pydantic model class for input validation
        """
        return self._input_callback.get_input_schema()
    
    def get_output_key(self) -> str:
        """
        Get the state key where output will be written.
        
        Delegates to the output callback's get_output_key method.
        
        Returns:
            The full namespaced output key (e.g., "result:s0_w1:judgement")
        """
        return self._output_callback.get_output_key()
    
    def get_output_schema(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for output validation.
        
        Delegates to the output callback's get_output_schema method.
        This enables aggregators to know what schema to expect when
        collecting results from workers.
        
        Returns:
            The Pydantic model class for output validation
        """
        return self._output_callback.get_output_schema()

