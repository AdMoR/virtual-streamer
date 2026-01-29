"""
System Status Context Provider for the Virtual Streamer Agent.

Provides system workload information for the agent's prompt.
"""

import logging

from virtual_streamer.agents.virtual_streamer_agent.schema import WorkloadStatus

logger = logging.getLogger(__name__)


class MockSystemStatusContextProvider:
    """
    Mock provider for system workload status.
    
    Returns configurable fixed data. A real implementation would
    query: GET /api/v1/video-generation/jobs or system metrics.
    
    Usage:
        provider = MockSystemStatusContextProvider()
        provider.set_workload(WorkloadStatus.HIGH)  # Configure for testing
        section = await provider.render()  # Get formatted prompt section
    """
    
    def __init__(self):
        """Initialize with default values (low workload, no active jobs)."""
        self._workload = WorkloadStatus.LOW
        self._active_jobs = 0
        self._queue_pending = 0
    
    @property
    def name(self) -> str:
        """Provider name for logging/debugging."""
        return "system_status"
    
    async def render(self) -> str:
        """
        Fetch data and render the system status section.
        
        Returns:
            Formatted system status string for the prompt
        """
        data = await self._fetch()
        return self._format(data)
    
    async def _fetch(self) -> dict:
        """
        Fetch system status.
        
        Mock: returns internal state
        Real: would query job API or system metrics
        
        Returns:
            Dictionary with system status data
        """
        return {
            "workload": self._workload,
            "active_jobs": self._active_jobs,
            "queue_pending": self._queue_pending,
        }
    
    def _format(self, data: dict) -> str:
        """
        Format the data into a prompt section.
        
        Args:
            data: System status data dictionary
            
        Returns:
            Formatted markdown string
        """
        return f"""## System Status

- Workload: {data['workload'].value}
- Active jobs: {data['active_jobs']}
- Queue pending: {data['queue_pending']}"""
    
    # -------------------------------------------------------------------------
    # Configuration methods for testing
    # -------------------------------------------------------------------------
    
    def set_workload(self, workload: WorkloadStatus) -> None:
        """Set the workload status level."""
        self._workload = workload
        logger.debug(f"Workload set to {workload.value}")
    
    def set_active_jobs(self, count: int) -> None:
        """Set the number of active jobs."""
        self._active_jobs = count
    
    def set_queue_pending(self, count: int) -> None:
        """Set the queue pending count."""
        self._queue_pending = count
    
    def reset_to_defaults(self) -> None:
        """Reset all values to defaults."""
        self._workload = WorkloadStatus.LOW
        self._active_jobs = 0
        self._queue_pending = 0
        logger.debug("System provider reset to defaults")
    
    # -------------------------------------------------------------------------
    # Getters for inspection
    # -------------------------------------------------------------------------
    
    def get_workload(self) -> WorkloadStatus:
        """Get current workload status."""
        return self._workload
    
    def get_active_jobs(self) -> int:
        """Get current active job count."""
        return self._active_jobs
