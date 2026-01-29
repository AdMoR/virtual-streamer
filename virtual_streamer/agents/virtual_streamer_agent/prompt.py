"""
System prompts for the Virtual Streamer Agent.

Contains the main instruction prompt and helper functions for building
dynamic context sections.
"""

from typing import List, Callable
from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.agents.virtual_streamer_agent import tools
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.virtual_streamer_agent.schema import (
    QueueInfo,
    SystemStatus,
    ChatMessage,
    WorkloadStatus,
)
from virtual_streamer.agents.common.state_keys import STATE_QUEUE_INFO, STATE_SYSTEM_STATUS, STATE_CHAT_MESSAGES


# =============================================================================
# Core System Prompt
# =============================================================================

VIRTUAL_STREAMER_SYSTEM_PROMPT = """

    You are a stream host and manager of a Twitch channel.
    Your role is to select the right for the situation and maintain the best quality on the channel 
    
    The user will provide the message of a user on the twitch chat.
    1. Understand the user need.
    2. Use either: 
         create_video
         answer_chat_message
    3. Return the tool call used.  

"""


# =============================================================================
# Context Formatting Functions
# =============================================================================

def format_queue_context(queue_info: QueueInfo) -> str:
    """Format queue information for the agent context."""
    lines = [
        "## État de la Queue",
        "",
        f"- Vidéos fraîches en attente: {queue_info.pending_count}",
        f"- Vidéos disponibles pour replay: {queue_info.played_count}",
        f"- Jobs de génération en cours: {queue_info.active_jobs}",
        f"- En mode replay: {'Oui' if queue_info.is_replaying else 'Non'}",
    ]
    
    if queue_info.next_videos:
        lines.append("")
        lines.append("**Prochaines vidéos:**")
        for i, title in enumerate(queue_info.next_videos[:5], 1):
            lines.append(f"{i}. {title}")
    
    return "\n".join(lines)


def format_system_context(system_status: SystemStatus) -> str:
    """Format system status for the agent context."""
    return f"""## État du Système

- Charge: {system_status.workload.value}
- Jobs actifs: {system_status.active_jobs}
- Queue pending: {system_status.queue_pending}"""


def format_chat_context(messages: List[ChatMessage], max_messages: int = 50) -> str:
    """Format chat messages for the agent context."""
    if not messages:
        return "## Chat Récent\n\n*Aucun message récent*"
    
    lines = ["## Chat Récent", ""]
    
    # Take most recent messages
    recent = messages[-max_messages:]
    
    for msg in recent:
        mention_marker = " 📢" if msg.is_mention else ""
        lines.append(f"[{msg.timestamp}] @{msg.username}{mention_marker}: {msg.message}")
    
    return "\n".join(lines)


def build_full_context(
        tools,
    queue_info: QueueInfo,
    system_status: SystemStatus,
    messages: List[ChatMessage],
    max_chat_messages: int = 50,
) -> str:
    """Build the complete context string for the agent."""
    sections = [
        format_queue_context(queue_info),
        "",
        format_system_context(system_status),
        "",
        format_chat_context(messages, max_chat_messages),
    ]
    
    return "\n".join(sections)


class VirtualStreamerInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that builds context-aware prompts.

    Reads queue info, system status, and chat messages from state
    and formats them into the agent's instruction.
    """

    def __init__(self, tools: List[Callable],  max_chat_messages: int = 100):
        """
        Initialize the instruction provider.

        Args:
            max_chat_messages: Maximum number of chat messages to include in context
        """
        self.tools = tools
        self.max_chat_messages = max_chat_messages

    async def __call__(self, context: ReadonlyContext) -> str:
        """Generate the instruction with current context."""
        # Extract context from state
        queue_info = self._get_queue_info(context)
        system_status = self._get_system_status(context)
        messages = self._get_chat_messages(context)
        tools = self._get_tools()

        # Build dynamic context section
        dynamic_context = build_full_context(
            tools=tools,
            queue_info=queue_info,
            system_status=system_status,
            messages=messages,
            max_chat_messages=self.max_chat_messages,
        )

        # Combine system prompt with dynamic context
        full_prompt = f"{VIRTUAL_STREAMER_SYSTEM_PROMPT}\n\n---\n\n{dynamic_context}"

        return full_prompt

    def _get_tools(self):
        return '\n'.join(f.__name__ for f in self.tools)

    def _get_queue_info(self, context: ReadonlyContext) -> QueueInfo:
        """Extract queue info from state or return defaults."""
        raw = context.state.get(STATE_QUEUE_INFO)
        if raw is None:
            return QueueInfo(
                pending_count=0,
                played_count=0,
                next_videos=[],
                is_replaying=False,
                active_jobs=0,
            )
        if isinstance(raw, QueueInfo):
            return raw
        if isinstance(raw, dict):
            return QueueInfo(**raw)
        return QueueInfo(
            pending_count=0,
            played_count=0,
            next_videos=[],
            is_replaying=False,
            active_jobs=0,
        )

    def _get_system_status(self, context: ReadonlyContext) -> SystemStatus:
        """Extract system status from state or return defaults."""
        raw = context.state.get(STATE_SYSTEM_STATUS)
        if raw is None:
            return SystemStatus(
                workload=WorkloadStatus.UNKNOWN,
                active_jobs=0,
                queue_pending=0,
            )
        if isinstance(raw, SystemStatus):
            return raw
        if isinstance(raw, dict):
            return SystemStatus(**raw)
        return SystemStatus(
            workload=WorkloadStatus.UNKNOWN,
            active_jobs=0,
            queue_pending=0,
        )

    def _get_chat_messages(self, context: ReadonlyContext) -> List[ChatMessage]:
        """Extract chat messages from state or return empty list."""
        raw = context.state.get(STATE_CHAT_MESSAGES)
        if raw is None:
            return []
        if isinstance(raw, list):
            messages = []
            for item in raw:
                if isinstance(item, ChatMessage):
                    messages.append(item)
                elif isinstance(item, dict):
                    messages.append(ChatMessage(**item))
            return messages
        return []
