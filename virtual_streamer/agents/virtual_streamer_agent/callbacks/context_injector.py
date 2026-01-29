"""
Context Injector Callback for the Virtual Streamer Agent.

This callback injects context provider data as a tool response message
in the conversation, making dynamic context visible in the chat history
rather than hidden in the system prompt.
"""

import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from virtual_streamer.lib.agents.callbacks import BeforeModelCallback
from virtual_streamer.agents.virtual_streamer_agent.context.protocol import (
    ContextProviderProtocol,
)

logger = logging.getLogger(__name__)


class InjectContextCallback(BeforeModelCallback):
    """
    Injects context provider data as a tool response message.
    
    This callback runs before each LLM call and:
    1. Renders all context providers (queue status, system status, chat messages)
    2. Formats the data as a function_response message
    3. Appends it to the conversation so the LLM sees it as tool output
    
    This approach makes context visible in the conversation history,
    separate from the static system prompt instructions.
    
    Usage:
        providers = [queue_provider, system_provider, chat_provider]
        callback = InjectContextCallback(providers)
        
        agent = BaseLlmAgent(
            ...,
            before_model_callback=callback,
        )
    """
    
    TOOL_NAME = "get_system_context"
    
    def __init__(
        self,
        context_providers: List[ContextProviderProtocol],
        tool_name: str = TOOL_NAME,
    ):
        """
        Initialize the callback with context providers.
        
        Args:
            context_providers: List of providers to render and inject
            tool_name: Name for the simulated tool (default: "get_system_context")
        """
        self.context_providers = context_providers
        self.tool_name = tool_name
    
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        """
        Inject context as a tool response before the LLM call.
        
        Args:
            callback_context: Context providing access to shared state
            request: The LLM request to modify
            
        Returns:
            None to continue to LLM call (we never skip)
        """
        if not self.context_providers:
            logger.warning("No context providers configured, skipping injection")
            return None
        
        try:
            # Render all providers
            context_data = await self._render_providers()
            
            # Format as tool response and inject
            tool_response_part = self._format_as_tool_response(context_data)
            llm_request.contents[-1].parts.append(tool_response_part)
            
            logger.warning(
                f"Injected context from {len(self.context_providers)} providers "
                f"as '{self.tool_name}' tool response"
            )
            
        except Exception as e:
            logger.error(f"Failed to inject context: {e}", exc_info=True)
            # Don't block the LLM call on context injection failure
        
        return None
    
    async def _render_providers(self) -> Dict[str, Any]:
        """
        Render all context providers and collect their output.
        
        Returns:
            Dictionary with provider name -> rendered content
        """
        context_data = {}
        
        for provider in self.context_providers:
            try:
                rendered = await provider.render()
                context_data[provider.name] = rendered
                logger.debug(f"Rendered provider '{provider.name}'")
            except Exception as e:
                logger.error(f"Failed to render provider '{provider.name}': {e}")
                context_data[provider.name] = f"Error: {str(e)}"
        
        return context_data
    
    def _format_as_tool_response(self, context_data: Dict[str, Any]) -> types.Content:
        """
        Format the context data as a tool response message.
        
        According to ADK documentation, tool responses use:
        - role='user' for the Content (for LLM history)
        - function_response Part with name and response dict
        
        Args:
            context_data: Dictionary of provider name -> rendered content
            
        Returns:
            types.Content formatted as a tool response
        """
        # Create the function response part
        response_part = types.Part.from_function_response(
            name=self.tool_name,
            response=context_data,
        )
        
        # Wrap in Content with role='user' (required for tool responses in LLM history)
        return response_part
