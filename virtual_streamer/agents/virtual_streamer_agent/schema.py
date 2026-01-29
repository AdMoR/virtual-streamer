"""
Output schemas for the Virtual Streamer Agent.

These Pydantic models define the structured output format for the agent's
responses when not using tool calls.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ActionType(str, Enum):
    """Type of action the agent decided to take."""
    NO_ACTION = "no_action"
    RESPOND = "respond"
    CREATE_VIDEO = "create_video"
    RESPOND_AND_CREATE = "respond_and_create"


class AgentDecision(BaseModel):
    """
    Structured output for agent decision when not using tool calls directly.
    
    This schema is used when the agent needs to output a structured decision
    without immediately executing tools.
    """
    
    action: ActionType = Field(
        description="The type of action to take based on the current context"
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this action was chosen"
    )
    
    response_message: Optional[str] = Field(
        default=None,
        description="Message to send to Twitch chat (if action involves responding)"
    )
    
    video_title: Optional[str] = Field(
        default=None,
        description="Title for the video to create (if action involves video creation)"
    )
    
    video_template: Optional[str] = Field(
        default=None,
        description="Story template ID to use for video creation"
    )
    
    target_user: Optional[str] = Field(
        default=None,
        description="Username being responded to or who requested the video"
    )


class ChatMessage(BaseModel):
    """A single chat message from Twitch."""
    
    timestamp: str = Field(description="ISO timestamp of the message")
    username: str = Field(description="Twitch username of the sender")
    message: str = Field(description="Content of the message")
    is_mention: bool = Field(
        default=False,
        description="Whether the message mentions the bot"
    )


class QueueInfo(BaseModel):
    """Information about the current video queue status."""
    
    pending_count: int = Field(
        description="Number of fresh videos waiting to be played"
    )
    played_count: int = Field(
        description="Number of videos that have been played (available for replay)"
    )
    next_videos: List[str] = Field(
        default_factory=list,
        description="Titles of the next videos in queue (up to 5)"
    )
    is_replaying: bool = Field(
        default=False,
        description="Whether the stream is currently playing replays"
    )
    active_jobs: int = Field(
        default=0,
        description="Number of video generation jobs currently running"
    )


class WorkloadStatus(str, Enum):
    """Current system workload level."""
    UNKNOWN = "unknown"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SystemStatus(BaseModel):
    """Current system status information."""
    
    workload: WorkloadStatus = Field(
        default=WorkloadStatus.UNKNOWN,
        description="Current system workload level"
    )
    active_jobs: int = Field(
        default=0,
        description="Number of video generation jobs in progress"
    )
    queue_pending: int = Field(
        default=0,
        description="Number of videos pending in queue"
    )
