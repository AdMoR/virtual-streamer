"""
Shared callbacks for ADK agents.

This module provides callbacks used across multiple agents in the
video generation pipeline.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

from virtual_streamer.lib.agents.callbacks import (
    AfterModelCallback,
    AgentCallback,
    extract_llm_response_json,
)
from virtual_streamer.agents.common.state_keys import (
    VIDEO_SEGMENTS,
    FINAL_VIDEO_PATH,
)
from virtual_streamer.agents.common.utils import concatenate_videos

logger = logging.getLogger(__name__)


class FinalizeVideoCallback(AgentCallback):
    """
    Callback that concatenates all video segments into the final video.
    
    This callback runs as after_agent_callback on the VideoGenerationOrchestrator.
    It reads the VIDEO_SEGMENTS from state and creates the final concatenated video.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory where the final video will be saved
        """
        self.output_dir = output_dir
    
    async def __call__(self, callback_context: CallbackContext) -> None:
        """
        Concatenate video segments into final video.
        
        Args:
            callback_context: Context providing access to shared state
        """
        segments = callback_context.state.get(VIDEO_SEGMENTS, [])
        
        if not segments:
            logger.warning("No video segments found to concatenate")
            return None
        
        logger.info(f"Concatenating {len(segments)} video segments")
        
        try:
            final_path = concatenate_videos(segments, self.output_dir)
            callback_context.state[FINAL_VIDEO_PATH] = final_path
            logger.info(f"Final video saved to: {final_path}")
        except Exception as e:
            logger.error(f"Failed to concatenate videos: {e}")
            raise
        
        return None


class LoggingCallback(AgentCallback):
    """
    Generic logging callback for debugging agent execution.
    """
    
    def __init__(self, agent_name: str, phase: str = ""):
        """
        Initialize the callback.
        
        Args:
            agent_name: Name of the agent for logging
            phase: Phase description (e.g., "before", "after")
        """
        self.agent_name = agent_name
        self.phase = phase
    
    async def __call__(self, callback_context: CallbackContext) -> None:
        """Log agent execution phase."""
        logger.info(f"[{self.agent_name}] {self.phase}")
        return None

