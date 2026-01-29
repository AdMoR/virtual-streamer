"""
Context providers for the Virtual Streamer Agent.

This module provides classes that fetch context information from
various sources (API, system, etc.).
"""

import logging
import os
from typing import List, Optional

import httpx

from virtual_streamer.agents.virtual_streamer_agent.schema import (
    QueueInfo,
    SystemStatus,
    WorkloadStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10.0


# =============================================================================
# Queue Info Provider
# =============================================================================

class QueueInfoProvider:
    """
    Fetches current playlist/queue status from the API.
    
    This provider calls the playlist API to get information about
    pending and played videos.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the provider.
        
        Args:
            api_url: Base URL of the Virtual Streamer API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url or os.environ.get("API_URL", API_URL)
        self.timeout = timeout
    
    async def get_queue_info(self, programmation_id: str) -> QueueInfo:
        """
        Fetch queue information for a programmation.
        
        Args:
            programmation_id: ID of the programmation to query
            
        Returns:
            QueueInfo with current queue status
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Get pending videos
                pending_response = await client.get(
                    f"{self.api_url}/api/v1/programmations/{programmation_id}/playlist",
                    params={"status": "pending"}
                )
                pending_data = pending_response.json() if pending_response.is_success else []
                
                # Get played videos
                played_response = await client.get(
                    f"{self.api_url}/api/v1/programmations/{programmation_id}/playlist",
                    params={"status": "played"}
                )
                played_data = played_response.json() if played_response.is_success else []
                
                # Extract video titles
                next_videos = []
                for entry in pending_data[:5]:
                    metadata = entry.get("metadata", {})
                    title = metadata.get("title") or metadata.get("requested_title") or "Sans titre"
                    next_videos.append(title)
                
                pending_count = len(pending_data) if isinstance(pending_data, list) else 0
                played_count = len(played_data) if isinstance(played_data, list) else 0
                
                return QueueInfo(
                    pending_count=pending_count,
                    played_count=played_count,
                    next_videos=next_videos,
                    is_replaying=pending_count == 0 and played_count > 0,
                    active_jobs=0,  # Will be set by WorkloadProvider
                )
                
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch queue info: {e}")
            return QueueInfo(
                pending_count=0,
                played_count=0,
                next_videos=[],
                is_replaying=False,
                active_jobs=0,
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching queue info: {e}")
            return QueueInfo(
                pending_count=0,
                played_count=0,
                next_videos=[],
                is_replaying=False,
                active_jobs=0,
            )


# =============================================================================
# Workload Provider
# =============================================================================

class WorkloadProvider:
    """
    Provides information about system workload.
    
    This is a placeholder implementation that returns unknown workload.
    In the future, this could query actual system metrics.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the provider.
        
        Args:
            api_url: Base URL of the Virtual Streamer API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url or os.environ.get("API_URL", API_URL)
        self.timeout = timeout
    
    async def get_system_status(self) -> SystemStatus:
        """
        Get current system status.
        
        Returns:
            SystemStatus with current workload information
        """
        # For now, return unknown workload
        # Future: query GPU metrics, job queue depth, etc.
        try:
            active_jobs = await self._get_active_job_count()
            
            # Simple workload estimation based on active jobs
            if active_jobs == 0:
                workload = WorkloadStatus.LOW
            elif active_jobs <= 2:
                workload = WorkloadStatus.NORMAL
            elif active_jobs <= 5:
                workload = WorkloadStatus.HIGH
            else:
                workload = WorkloadStatus.CRITICAL
            
            return SystemStatus(
                workload=workload,
                active_jobs=active_jobs,
                queue_pending=0,  # Will be updated by ContextBuilder
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return SystemStatus(
                workload=WorkloadStatus.UNKNOWN,
                active_jobs=0,
                queue_pending=0,
            )
    
    async def _get_active_job_count(self) -> int:
        """
        Get count of active video generation jobs.
        
        Returns:
            Number of jobs currently running
        """
        # This is a placeholder - in a real implementation,
        # we would query the job store or API
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try to get job count from API if endpoint exists
                response = await client.get(f"{self.api_url}/api/v1/video-generation/jobs")
                if response.is_success:
                    data = response.json()
                    # Count running jobs
                    if isinstance(data, list):
                        return sum(1 for j in data if j.get("status") == "running")
                    elif isinstance(data, dict):
                        return data.get("running_count", 0)
        except Exception:
            # Endpoint might not exist, that's ok
            pass
        
        return 0


# =============================================================================
# Combined Provider
# =============================================================================

class ContextProviders:
    """
    Container for all context providers.
    
    Provides a unified interface to fetch all context data.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize all providers.
        
        Args:
            api_url: Base URL for API calls
            timeout: Request timeout in seconds
        """
        self.queue_provider = QueueInfoProvider(api_url, timeout)
        self.workload_provider = WorkloadProvider(api_url, timeout)
    
    async def get_queue_info(self, programmation_id: str) -> QueueInfo:
        """Fetch queue info."""
        return await self.queue_provider.get_queue_info(programmation_id)
    
    async def get_system_status(self) -> SystemStatus:
        """Fetch system status."""
        return await self.workload_provider.get_system_status()
