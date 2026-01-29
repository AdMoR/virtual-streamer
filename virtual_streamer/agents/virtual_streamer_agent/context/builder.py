"""
Context builder for the Virtual Streamer Agent.

This module provides the ContextBuilder class that assembles all
context information into a format suitable for the agent's state.
"""

import logging
from typing import Any, Dict, List, Optional

from virtual_streamer.agents.virtual_streamer_agent.schema import (
    ChatMessage,
    QueueInfo,
    SystemStatus,
)
from virtual_streamer.agents.virtual_streamer_agent.context.conversation import (
    ConversationManager,
    ConversationManagerStrategy,
    KeepLastN,
)
from virtual_streamer.agents.virtual_streamer_agent.context.providers import (
    ContextProviders,
    QueueInfoProvider,
    WorkloadProvider,
)
from virtual_streamer.agents.common.state_keys import (
    STATE_QUEUE_INFO,
    STATE_SYSTEM_STATUS,
    STATE_CHAT_MESSAGES,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds context for the Virtual Streamer Agent.
    
    This class aggregates data from various sources:
    - Chat messages from ConversationManager
    - Queue info from QueueInfoProvider
    - System status from WorkloadProvider
    
    The resulting context dictionary can be passed to the agent's state.
    
    Usage:
        builder = ContextBuilder(
            programmation_id="my-prog",
            conversation_manager=ConversationManager(),
        )
        
        # Add messages as they come in
        builder.add_chat_message(msg)
        
        # Build context for agent
        context = await builder.build()
        # Pass to agent state
    """
    
    def __init__(
        self,
        programmation_id: str,
        conversation_manager: Optional[ConversationManager] = None,
        conversation_strategy: Optional[ConversationManagerStrategy] = None,
        api_url: Optional[str] = None,
        max_chat_messages: int = 100,
    ):
        """
        Initialize the context builder.
        
        Args:
            programmation_id: ID of the programmation for queue queries
            conversation_manager: Custom conversation manager (optional)
            conversation_strategy: Strategy for message selection (optional)
            api_url: Base URL for API calls
            max_chat_messages: Maximum chat messages in context
        """
        self.programmation_id = programmation_id
        self.max_chat_messages = max_chat_messages
        
        # Set up conversation manager
        if conversation_manager is not None:
            self.conversation_manager = conversation_manager
        else:
            strategy = conversation_strategy or KeepLastN(max_chat_messages)
            self.conversation_manager = ConversationManager(strategy=strategy)
        
        # Set up providers
        self.providers = ContextProviders(api_url=api_url)
        
        logger.info(
            f"ContextBuilder initialized for programmation '{programmation_id}' "
            f"with strategy: {self.conversation_manager.strategy.name}"
        )
    
    def add_chat_message(self, message: ChatMessage) -> None:
        """
        Add a chat message to the conversation history.
        
        Args:
            message: Message to add
        """
        self.conversation_manager.add_message(message)
    
    def add_chat_messages(self, messages: List[ChatMessage]) -> None:
        """
        Add multiple chat messages.
        
        Args:
            messages: Messages to add
        """
        self.conversation_manager.add_messages(messages)
    
    async def build(self) -> Dict[str, Any]:
        """
        Build the complete context for the agent.
        
        Fetches queue info and system status from APIs, combines
        with chat messages, and returns a dictionary ready for
        agent state.
        
        Returns:
            Dictionary with keys:
            - queue_info: QueueInfo object
            - system_status: SystemStatus object
            - chat_messages: List of ChatMessage objects
        """
        # Fetch queue info
        queue_info = await self.providers.get_queue_info(self.programmation_id)
        
        # Fetch system status
        system_status = await self.providers.get_system_status()
        
        # Update system status with queue pending count
        system_status.queue_pending = queue_info.pending_count
        
        # Update queue info with active jobs from system status
        queue_info.active_jobs = system_status.active_jobs
        
        # Get chat messages using the conversation manager's strategy
        chat_messages = self.conversation_manager.get_messages_for_context()
        
        logger.debug(
            f"Built context: queue_pending={queue_info.pending_count}, "
            f"workload={system_status.workload.value}, "
            f"chat_messages={len(chat_messages)}"
        )
        
        return {
            STATE_QUEUE_INFO: queue_info,
            STATE_SYSTEM_STATUS: system_status,
            STATE_CHAT_MESSAGES: chat_messages,
        }
    
    def build_state_dict(
        self,
        queue_info: QueueInfo,
        system_status: SystemStatus,
        chat_messages: List[ChatMessage],
    ) -> Dict[str, Any]:
        """
        Build state dictionary from provided data (sync version).
        
        Useful when you already have the data and just need to format it.
        
        Args:
            queue_info: Queue information
            system_status: System status
            chat_messages: Chat messages
            
        Returns:
            Dictionary ready for agent state
        """
        return {
            STATE_QUEUE_INFO: queue_info,
            STATE_SYSTEM_STATUS: system_status,
            STATE_CHAT_MESSAGES: chat_messages,
        }
    
    def clear_chat_history(self) -> None:
        """Clear all chat message history."""
        self.conversation_manager.clear()
    
    @property
    def chat_message_count(self) -> int:
        """Number of chat messages currently stored."""
        return self.conversation_manager.message_count
