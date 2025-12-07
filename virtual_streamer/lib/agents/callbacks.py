"""
Callback base classes for ADK agents.

This module provides abstract base classes for implementing lifecycle
callbacks in ADK agents. Callbacks allow you to:
- Intercept and modify agent behavior at various stages
- Store results in shared state
- Skip agent execution conditionally
- Log and monitor agent activity

Callback Lifecycle Order:
1. before_agent_callback  -> Agent starts
2. before_model_callback  -> Before LLM call (can skip)
3. **LLM Execution**
4. after_model_callback   -> After LLM returns
5. after_agent_callback   -> Agent completes
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union, overload

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Helper Functions
# =============================================================================


@overload
def extract_llm_response_json(
    llm_response: LlmResponse,
    response_model: None = None,
) -> Optional[Dict[str, Any]]: ...


@overload
def extract_llm_response_json(
    llm_response: LlmResponse,
    response_model: Type[T],
) -> Optional[T]: ...


def extract_llm_content_json(
    llm_content: types.Content,
    response_model: Optional[Type[T]] = None,
) -> Optional[Union[Dict[str, Any], T]]:
    return extract_llm_response_json(LlmResponse(content=llm_content), response_model)


def extract_llm_response_json(
    llm_response: LlmResponse,
    response_model: Optional[Type[T]] = None,
) -> Optional[Union[Dict[str, Any], T]]:
    """Extract JSON data from an LLM response, optionally parsing into a Pydantic model.

    This helper function attempts to parse the LLM response content as JSON.
    It handles various response formats and extracts the first valid JSON object.
    
    If a Pydantic model is provided, the JSON is validated and returned as an
    instance of that model.

    Args:
        llm_response: The LlmResponse object from the model callback.
        response_model: Optional Pydantic model class to parse the response into.
                       If provided, returns an instance of this model.
                       If None, returns a raw dictionary.

    Returns:
        If response_model is None: Parsed JSON as a dictionary, or None if parsing fails.
        If response_model is provided: Instance of the model, or None if parsing/validation fails.

    Example:
        # Without model (returns dict)
        >>> parsed = extract_llm_response_json(llm_response)
        >>> if parsed:
        ...     answer = parsed.get("answer")
        
        # With Pydantic model (returns typed instance)
        >>> from pydantic import BaseModel
        >>> class StoryOutput(BaseModel):
        ...     title: str
        ...     dialog: str
        >>> story = extract_llm_response_json(llm_response, StoryOutput)
        >>> if story:
        ...     print(story.title)  # Type-safe access
    """
    try:
        # Get the content from the response
        if not llm_response.content:
            logger.warning("LLM response has no content")
            return None

        # Extract text from content parts
        text_parts = []
        for part in llm_response.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if not text_parts:
            logger.warning("LLM response has no text parts")
            return None

        # Join all text parts
        full_text = "".join(text_parts)

        # Try to parse as JSON directly
        parsed_json: Optional[Dict[str, Any]] = None
        try:
            parsed_json = json.loads(full_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text (between { and })
        if parsed_json is None:
            start_idx = full_text.find("{")
            end_idx = full_text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = full_text[start_idx : end_idx + 1]
                try:
                    parsed_json = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        if parsed_json is None:
            logger.warning("Could not find valid JSON in LLM response")
            return None

        # If no model specified, return the raw dict
        if response_model is None:
            return parsed_json

        # Parse into Pydantic model
        try:
            return response_model.model_validate(parsed_json)
        except ValidationError as e:
            logger.warning(f"Failed to validate response with {response_model.__name__}: {e}")
            return None

    except Exception as e:
        logger.error(f"Failed to extract JSON from LLM response: {e}")
        return None


def extract_llm_response_text(llm_response: LlmResponse) -> str:
    """Extract plain text from an LLM response.

    Args:
        llm_response: The LlmResponse object from the model callback.

    Returns:
        The text content of the response, or empty string if no text.
    """
    try:
        if not llm_response.content:
            return ""

        text_parts = []
        for part in llm_response.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return "".join(text_parts)

    except Exception as e:
        logger.error(f"Failed to extract text from LLM response: {e}")
        return ""


# =============================================================================
# Callback Base Classes
# =============================================================================


class BeforeModelCallback(ABC):
    """Abstract base class for callbacks that run before the LLM call.

    BeforeModelCallback runs after the agent receives input but before
    the LLM is invoked. It can:
    - Modify or validate the input
    - Skip the LLM call entirely by returning Content
    - Set up state for downstream processing

    To skip the LLM call, return a types.Content object. Return None
    to continue normal execution.

    Example:
        class SkipIfCached(BeforeModelCallback):
            async def __call__(self, callback_context: CallbackContext):
                cached = callback_context.state.get("cached_response")
                if cached:
                    return types.Content(
                        parts=[types.Part(text=cached)],
                        role="model",
                    )
                return None  # Continue to LLM
    """

    @abstractmethod
    async def __call__(
        self,
        callback_context: CallbackContext,
        request: LlmRequest,
    ) -> Optional[types.Content]:
        """Execute the callback before the model is called.

        Args:
            callback_context: Context providing access to shared state.

        Returns:
            None to continue to LLM call, or Content to skip and use as response.
        """
        ...


class AfterModelCallback(ABC):
    """Abstract base class for callbacks that run after the LLM returns.

    AfterModelCallback runs immediately after the LLM generates a response.
    Use it to:
    - Parse and validate the LLM output
    - Store results in shared state
    - Log responses for monitoring
    - Transform the response before it's returned

    Note: AfterModelCallback cannot skip execution (LLM has already run).

    Example:
        class StoreResultCallback(AfterModelCallback):
            async def __call__(
                self,
                callback_context: CallbackContext,
                llm_response: LlmResponse,
            ) -> None:
                parsed = extract_llm_response_json(llm_response)
                if parsed:
                    callback_context.state["result"] = parsed
    """

    @abstractmethod
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """Execute the callback after the model returns.

        Args:
            callback_context: Context providing access to shared state.
            llm_response: The response from the LLM.
        """
        ...


class AgentCallback(ABC):
    """Abstract base class for agent-level lifecycle callbacks.

    AgentCallback can be used for both before_agent_callback and
    after_agent_callback. It runs at the start or end of the agent's
    execution lifecycle.

    When used as before_agent_callback, returning Content will skip
    the entire agent execution. Return None to continue.

    Example (before_agent):
        class ValidateInputCallback(AgentCallback):
            async def __call__(self, callback_context: CallbackContext):
                if not callback_context.state.get("user_authenticated"):
                    return types.Content(
                        parts=[types.Part(text="Authentication required")],
                        role="model",
                    )
                return None

    Example (after_agent):
        class ComputeMetricsCallback(AgentCallback):
            async def __call__(self, callback_context: CallbackContext):
                result = callback_context.state.get("result")
                if result:
                    callback_context.state["metrics"] = compute_metrics(result)
                return None
    """

    @abstractmethod
    async def __call__(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        """Execute the agent-level callback.

        Args:
            callback_context: Context providing access to shared state.

        Returns:
            None to continue, or Content to skip agent (before_agent only).
        """
        ...


# =============================================================================
# Common Callback Implementations
# =============================================================================


class LoggingBeforeCallback(BeforeModelCallback):
    """Log information before model execution."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def __call__(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        logger.info(f"[{self.agent_name}] Starting model execution")
        return None


class LoggingAfterCallback(AfterModelCallback):
    """Log information after model execution."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        text = extract_llm_response_text(llm_response)
        logger.info(
            f"[{self.agent_name}] Model returned response "
            f"({len(text)} chars)"
        )


class StoreResponseCallback(AfterModelCallback):
    """Store the parsed JSON response in state."""

    def __init__(self, state_key: str = "response"):
        self.state_key = state_key

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        parsed = extract_llm_response_json(llm_response)
        if parsed:
            callback_context.state[self.state_key] = parsed
        else:
            # Store raw text if JSON parsing fails
            callback_context.state[self.state_key] = extract_llm_response_text(
                llm_response
            )


class SkipIfStateCallback(BeforeModelCallback):
    """Skip model execution if a state condition is met."""

    def __init__(
        self,
        state_key: str,
        skip_message: str = "Skipped due to state condition",
        skip_if_exists: bool = True,
    ):
        """Initialize the callback.

        Args:
            state_key: The state key to check.
            skip_message: Message to return when skipping.
            skip_if_exists: If True, skip when key exists. If False, skip when missing.
        """
        self.state_key = state_key
        self.skip_message = skip_message
        self.skip_if_exists = skip_if_exists

    async def __call__(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        key_exists = self.state_key in callback_context.state

        should_skip = key_exists if self.skip_if_exists else not key_exists

        if should_skip:
            return types.Content(
                parts=[types.Part(text=self.skip_message)],
                role="model",
            )
        return None

