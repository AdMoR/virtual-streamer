"""
High-level API: shared data models for video generation.
"""
import os
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, ConfigDict, Field

from virtual_streamer.agents.story_pipeline.schema import RecurrentLocationsOutput, DetailedScenesOutput
from virtual_streamer.video_generation.ltx_client import VideoGenerationParams, DEFAULT_NEGATIVE_PROMPT


class StoryPipelineResult(BaseModel):
    """Result of the 3-step story pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    recurrent_locations: RecurrentLocationsOutput
    detailed_scenes: DetailedScenesOutput
    title: Optional[str] = None
    raw_story_text: Optional[str] = None


class VideoGenerationRequest(BaseModel):
    """Request model for LTX-2 video generation from a title or story text."""

    # Input (mutually exclusive)
    title: Optional[str] = None
    story_text: Optional[str] = None

    story_template_id: str

    # Video generation parameters
    video_width: int = 1280
    video_height: int = 720
    video_duration_seconds: float = 5.0
    video_fps: int = 24
    video_steps: int = 20
    video_cfg_scale: float = 4.0
    video_seed: int = -1
    enable_audio: bool = True

    # TTS — Fish-Speech service (Docker Compose service name "tts")
    tts_host: str = os.environ.get("FISH_TTS_HOST", "tts")
    tts_port: int = int(os.environ.get("FISH_TTS_PORT", "8003"))
    adapt_duration_to_audio: bool = True

    output_dir: Optional[str] = None
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting."

    # LLM configuration (for ADK agents)
    llm_provider: Optional[str] = "anthropic"
    llm_model: Optional[str] = "claude-sonnet-4-5-20250929"

    # Subtitle options
    enable_subtitles: bool = True
    subtitle_fontsize: int = 14

    def to_video_params(self, prompt: str = "", **overrides) -> VideoGenerationParams:
        return VideoGenerationParams(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=self.video_width,
            height=self.video_height,
            duration_seconds=self.video_duration_seconds,
            fps=self.video_fps,
            steps=self.video_steps,
            cfg_scale=self.video_cfg_scale,
            seed=self.video_seed,
            enable_audio=self.enable_audio,
            **overrides,
        )


class VideoGenerationResponse(BaseModel):
    """Response for video generation submission."""

    job_id: str
    status: str
    message: str


class VideoFromScriptRequest(BaseModel):
    """
    Generate LTX video from a pre-built (possibly user-edited) script.

    Scenes and locations come from a prior call to /story-pipeline/run and
    may have been modified by the user in the UI.
    """

    story_title: str
    story_template_id: str

    scenes: List[Dict[str, Any]]
    locations: List[Dict[str, Any]]

    # Video params
    video_width: int = 1280
    video_height: int = 720
    video_duration_seconds: float = 5.0
    video_fps: int = 24
    video_steps: int = 20
    video_cfg_scale: float = 4.0
    video_seed: int = -1
    enable_audio: bool = True
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting."

    # TTS
    tts_host: str = os.environ.get("FISH_TTS_HOST", "tts")
    tts_port: int = int(os.environ.get("FISH_TTS_PORT", "8003"))
    adapt_duration_to_audio: bool = True

    output_dir: Optional[str] = None

    # Subtitle options
    enable_subtitles: bool = False
    subtitle_fontsize: int = 14

    def to_video_params(self, prompt: str = "", **overrides) -> VideoGenerationParams:
        return VideoGenerationParams(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=self.video_width,
            height=self.video_height,
            duration_seconds=self.video_duration_seconds,
            fps=self.video_fps,
            steps=self.video_steps,
            cfg_scale=self.video_cfg_scale,
            seed=self.video_seed,
            enable_audio=self.enable_audio,
            **overrides,
        )


class GenerateFromBroadcastRequest(BaseModel):
    """Request model for video generation from active broadcast."""

    stream_id: str
    title: str
    user: Optional[str] = None

    skip_queue_limit: bool = Field(
        default=False,
        description="[ADMIN] Bypass MAX_PENDING_JOBS queue limit. For batch operations only.",
    )


class GenerateFromBroadcastResponse(BaseModel):
    """Response model for video generation from broadcast."""

    job_id: str
    status: str
    message: str
    story_template_id: str


class FeedbackRequest(BaseModel):
    """Request model for video feedback."""

    entry_id: str
    user: str
    feedback: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
