"""
Context Provider Protocol for the Virtual Streamer Agent.

Defines the interface that all context providers must implement.
Both mock and real providers follow this protocol.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContextProviderProtocol(Protocol):
    """
    Protocol for context providers that render prompt sections.
    
    Each provider is responsible for:
    1. Fetching data (from mock state or real API)
    2. Rendering that data into a prompt section
    
    The render() method internally calls fetch() and formats the result.
    
    Usage:
        provider = MockProcessingQueueContextProvider()
        section = await provider.render()  # Returns formatted string
    """
    
    @property
    def name(self) -> str:
        """
        Provider name for logging and debugging.
        
        Returns:
            Unique identifier string (e.g., "processing_queue", "system_status")
        """
        ...
    
    async def render(self) -> str:
        """
        Fetch data and render this provider's section of the prompt.
        
        This method should:
        1. Call internal _fetch() to get data
        2. Call internal _format() to render the data
        
        Returns:
            Formatted string ready to be included in the prompt
        """
        ...
