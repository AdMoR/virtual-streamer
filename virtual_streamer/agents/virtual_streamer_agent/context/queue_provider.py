"""
Processing Queue Context Provider for the Virtual Streamer Agent.

Provides video queue status information for the agent's prompt.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class MockProcessingQueueContextProvider:
    """
    Mock provider for video queue status.
    
    Returns configurable fixed data for testing. A real implementation
    would call: GET /api/v1/programmations/{id}/playlist
    
    Usage:
        provider = MockProcessingQueueContextProvider()
        provider.set_pending_count(2)  # Configure for testing
        section = await provider.render()  # Get formatted prompt section
    """
    
    # Default mock data
    DEFAULT_PENDING = 5
    DEFAULT_PLAYED = 10
    DEFAULT_NEXT_VIDEOS = [
        "Fred se lance dans l'IA",
        "Pourquoi les chats retombent sur leurs pattes",
        "Les mystères du fromage français",
    ]
    
    def __init__(self):
        """Initialize with default values."""
        # Configurable state
        self._pending_count = self.DEFAULT_PENDING
        self._played_count = self.DEFAULT_PLAYED
        self._next_videos = self.DEFAULT_NEXT_VIDEOS.copy()
        self._is_replaying = False
        self._active_jobs = 0
    
    @property
    def name(self) -> str:
        """Provider name for logging/debugging."""
        return "processing_queue"
    
    async def render(self) -> str:
        """
        Fetch data and render the queue status section.
        
        Returns:
            Formatted queue status string for the prompt
        """
        data = await self._fetch()
        return self._format(data)
    
    async def _fetch(self) -> dict:
        """
        Fetch queue data.
        
        Mock: returns internal configurable state
        Real: would call GET /api/v1/programmations/{id}/playlist
        
        Returns:
            Dictionary with queue status data
        """
        return {
            "pending_count": self._pending_count,
            "played_count": self._played_count,
            "next_videos": self._next_videos,
            "is_replaying": self._is_replaying,
            "active_jobs": self._active_jobs,
        }
    
    def _format(self, data: dict) -> str:
        """
        Format the data into a prompt section.
        
        Args:
            data: Queue status data dictionary
            
        Returns:
            Formatted markdown string
        """
        lines = [
            "## Queue Status",
            "",
            f"- Fresh videos pending: {data['pending_count']}",
            f"- Videos available for replay: {data['played_count']}",
            f"- Active generation jobs: {data['active_jobs']}",
            f"- Currently in replay mode: {'Yes' if data['is_replaying'] else 'No'}",
        ]
        
        if data['next_videos']:
            lines.append("")
            lines.append("**Next videos in queue:**")
            for i, title in enumerate(data['next_videos'][:5], 1):
                lines.append(f"{i}. {title}")
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Configuration methods for testing
    # -------------------------------------------------------------------------
    
    def set_pending_count(self, count: int) -> None:
        """Set the number of pending videos."""
        self._pending_count = count
        logger.debug(f"Queue pending count set to {count}")
    
    def set_played_count(self, count: int) -> None:
        """Set the number of played videos."""
        self._played_count = count
    
    def set_next_videos(self, videos: List[str]) -> None:
        """Set the list of next video titles."""
        self._next_videos = videos
    
    def set_replay_mode(self, is_replaying: bool) -> None:
        """Set whether the stream is in replay mode."""
        self._is_replaying = is_replaying
    
    def set_active_jobs(self, count: int) -> None:
        """Set the number of active generation jobs."""
        self._active_jobs = count
    
    def reset_to_defaults(self) -> None:
        """Reset all values to defaults."""
        self._pending_count = self.DEFAULT_PENDING
        self._played_count = self.DEFAULT_PLAYED
        self._next_videos = self.DEFAULT_NEXT_VIDEOS.copy()
        self._is_replaying = False
        self._active_jobs = 0
        logger.debug("Queue provider reset to defaults")
    
    # -------------------------------------------------------------------------
    # Getters for inspection
    # -------------------------------------------------------------------------
    
    def get_pending_count(self) -> int:
        """Get current pending count."""
        return self._pending_count
    
    def get_played_count(self) -> int:
        """Get current played count."""
        return self._played_count
    
    def is_replaying(self) -> bool:
        """Check if in replay mode."""
        return self._is_replaying
