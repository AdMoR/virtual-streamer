"""
Stateful callback abstract classes for ADK agents.

This module provides abstract base classes for callbacks that interact with
namespaced state keys. These callbacks enable type-safe state management
for dynamic parallel agent workflows.

Key features:
- Input/output keys are automatically prefixed with run_id when set
- Schema validation via Pydantic models
- Clean abstraction for DynamicParallelAgent to discover keys

Usage:
    class MyInputCallback(StateInputCallback):
        def __init__(self, run_id: str | None = None):
            super().__init__(
                input_key="my_input",
                input_schema=MyInputSchema,
                run_id=run_id
            )
        
        async def __call__(self, ctx, request):
            key = self.get_input_key()
            data = self.input_schema.model_validate_json(ctx.state[key])
            # ... process data
"""

from abc import ABC, abstractmethod
from typing import Optional, Type

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types
from pydantic import BaseModel

from virtual_streamer.lib.agents.callbacks import BeforeModelCallback, AfterModelCallback


class StateInputCallback(BeforeModelCallback, ABC):
    """
    Abstract callback that reads input from a namespaced state key.
    
    This callback provides a standardized way to read input data from
    the session state, with automatic key namespacing based on run_id.
    
    Subclasses must:
    - Call super().__init__() with input_key, input_schema, and optional run_id
    - Implement __call__ to process the input and optionally modify the request
    
    Example:
        class InjectDataCallback(StateInputCallback):
            def __init__(self, run_id: str | None = None):
                super().__init__(
                    input_key="user_data",
                    input_schema=UserDataInput,
                    run_id=run_id
                )
            
            async def __call__(self, ctx, request):
                key = self.get_input_key()
                data = self.input_schema.model_validate_json(ctx.state[key])
                # Add data to request...
                return None  # Continue to LLM
    """
    
    def __init__(
        self,
        input_key: str,
        input_schema: Type[BaseModel],
        run_id: Optional[str] = None
    ):
        """
        Initialize the state input callback.
        
        Args:
            input_key: Base key for reading input from state (e.g., "video_sentence")
            input_schema: Pydantic model for validating the input data
            run_id: Optional run ID for namespacing the key (e.g., "s0_w1")
        """
        self.input_key = input_key
        self.input_schema = input_schema
        self.run_id = run_id
    
    def get_input_key(self) -> str:
        """
        Return the full state key, prefixed with run_id if set.
        
        Returns:
            - If run_id is set: "task:{run_id}:{input_key}"
            - If run_id is None: "{input_key}"
        
        Example:
            >>> cb = MyCallback(input_key="data", run_id="s0_w1")
            >>> cb.get_input_key()
            "task:s0_w1:data"
        """
        if self.run_id:
            return f"task:{self.run_id}:{self.input_key}"
        return self.input_key
    
    def get_input_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model used for input validation."""
        return self.input_schema
    
    @abstractmethod
    async def __call__(
        self,
        callback_context: CallbackContext,
        request: LlmRequest,
    ) -> Optional[types.Content]:
        """
        Execute the callback before the model is called.
        
        Subclasses should:
        1. Read from state using self.get_input_key()
        2. Validate using self.input_schema.model_validate_json()
        3. Process the data and optionally modify the request
        
        Args:
            callback_context: Context providing access to shared state
            request: The LLM request that can be modified
        
        Returns:
            None to continue to LLM call, or Content to skip and use as response.
        """
        ...


class StateOutputCallback(AfterModelCallback, ABC):
    """
    Abstract callback that writes output to a namespaced state key.
    
    This callback provides a standardized way to write output data to
    the session state, with automatic key namespacing based on run_id.
    
    Subclasses must:
    - Call super().__init__() with output_key, output_schema, and optional run_id
    - Implement __call__ to parse LLM response and store in state
    
    Example:
        class StoreResultCallback(StateOutputCallback):
            def __init__(self, run_id: str | None = None):
                super().__init__(
                    output_key="result",
                    output_schema=ResultOutput,
                    run_id=run_id
                )
            
            async def __call__(self, ctx, llm_response):
                parsed = extract_llm_response_json(llm_response, self.output_schema)
                ctx.state[self.get_output_key()] = parsed.model_dump_json()
                return None
    """
    
    def __init__(
        self,
        output_key: str,
        output_schema: Type[BaseModel],
        run_id: Optional[str] = None
    ):
        """
        Initialize the state output callback.
        
        Args:
            output_key: Base key for writing output to state (e.g., "judgement")
            output_schema: Pydantic model for the output data structure
            run_id: Optional run ID for namespacing the key (e.g., "s0_w1")
        """
        self.output_key = output_key
        self.output_schema = output_schema
        self.run_id = run_id
    
    def get_output_key(self) -> str:
        """
        Return the full state key, prefixed with run_id if set.
        
        Returns:
            - If run_id is set: "result:{run_id}:{output_key}"
            - If run_id is None: "{output_key}"
        
        Example:
            >>> cb = MyCallback(output_key="result", run_id="s0_w1")
            >>> cb.get_output_key()
            "result:s0_w1:result"
        """
        if self.run_id:
            return f"result:{self.run_id}:{self.output_key}"
        return self.output_key
    
    def get_output_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model used for output structure."""
        return self.output_schema
    
    @abstractmethod
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """
        Execute the callback after the model returns.
        
        Subclasses should:
        1. Parse the LLM response (e.g., using extract_llm_response_json)
        2. Validate against self.output_schema
        3. Store in state using self.get_output_key()
        
        Args:
            callback_context: Context providing access to shared state
            llm_response: The response from the LLM
        
        Returns:
            None to keep original response, or modified LlmResponse.
        """
        ...

