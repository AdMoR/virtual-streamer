"""
LTX Video API Client

Async client for interacting with the LTX Video Generation API to generate videos
using LTX-2 text-to-video models.

Usage:
    from virtual_streamer.video_generation.ltx_client import (
        LTXVideoClient, LTXVideoConfig, VideoGenerationParams
    )
    
    async with LTXVideoClient(LTXVideoConfig()) as client:
        result = await client.generate_video(
            VideoGenerationParams(prompt="A serene forest at sunrise")
        )
        print(f"Video saved to: {result.video_path}")
"""

import asyncio
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

class LTXVideoConfig(BaseModel):
    """Configuration for LTX Video API server connection."""
    
    server_url: str = Field(
        default="http://localhost:8081",
        description="Base URL of the LTX Video API server"
    )
    timeout: float = Field(
        default=600.0,
        description="Request timeout in seconds (video generation can be slow)"
    )
    poll_interval: float = Field(
        default=2.0,
        description="Interval in seconds between status polls"
    )


# Backward compatibility alias
ComfyUIConfig = LTXVideoConfig


# =============================================================================
# Request/Response Models
# =============================================================================

class JobStatus(str, Enum):
    """Status of a video generation job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LTXJobRequest(BaseModel):
    """Request model for submitting a video generation job to the LTX API."""
    
    model_type: str = Field(
        default="ltx2_distilled",
        description="Model identifier (e.g., 'ltx2_distilled')"
    )
    prompt: str = Field(
        description="Text prompt describing the video to generate"
    )
    num_inference_steps: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Number of denoising steps"
    )
    video_length: int = Field(
        default=241,
        description="Number of frames (should be 8*k+1 for LTX-2)"
    )
    guidance_scale: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="CFG scale for classifier-free guidance"
    )
    resolution: str = Field(
        default="1024x768",
        description="Output resolution as WxH string"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    # Image/video inputs for image-to-video or video continuation
    image_prompt_type: Optional[str] = Field(
        default=None,
        description="Input mode: '', 'S', 'SE', 'V'"
    )
    image_start: Optional[str] = Field(
        default=None,
        description="Path to start image"
    )
    image_end: Optional[str] = Field(
        default=None,
        description="Path to end image"
    )
    video_source: Optional[str] = Field(
        default=None,
        description="Path to source video for continuation"
    )
    
    # Additional parameters
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional generation parameters"
    )


class LTXJobResponse(BaseModel):
    """Response model when a job is submitted."""
    job_id: str
    status: str
    message: str


class LTXJobStatusResponse(BaseModel):
    """Response model for job status queries."""
    job_id: str
    status: str
    progress: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    prompt_preview: Optional[str] = None


# =============================================================================
# Video Generation Parameters (User-facing)
# =============================================================================

class VideoGenerationParams(BaseModel):
    """Parameters for LTX-2 video generation (user-facing interface)."""
    
    prompt: str = Field(
        description="Text prompt describing the video to generate"
    )
    negative_prompt: str = Field(
        default="",
        description="What to avoid in the generated video (stored for reference, not sent to API)"
    )
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Video width"
    )
    height: int = Field(
        default=768,
        ge=256,
        le=2048,
        description="Video height"
    )
    duration_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Video duration in seconds"
    )
    fps: int = Field(
        default=24,
        ge=8,
        le=60,
        description="Frames per second"
    )
    steps: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Number of inference steps"
    )
    cfg_scale: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale"
    )
    seed: int = Field(
        default=-1,
        description="Random seed (-1 for random)"
    )
    model_type: str = Field(
        default="ltx2_distilled",
        description="Model type to use"
    )
    enable_audio: bool = Field(
        default=True,
        description="Generate synchronized audio (if supported)"
    )
    
    @property
    def frame_count(self) -> int:
        """
        Calculate frame count satisfying LTX-2's 8n + 1 constraint.
        """
        raw_frames = int(self.duration_seconds * self.fps)
        # Round to nearest valid value: 8n + 1
        n = round((raw_frames - 1) / 8)
        return max(8 * n + 1, 9)  # Minimum 9 frames
    
    @property
    def actual_duration(self) -> float:
        """Actual duration based on frame_count constraint."""
        return self.frame_count / self.fps
    
    @property
    def resolution(self) -> str:
        """Resolution as WxH string for the API."""
        return f"{self.width}x{self.height}"
    
    def to_job_request(self) -> LTXJobRequest:
        """Convert to LTXJobRequest for API submission."""
        seed = self.seed if self.seed >= 0 else None
        
        return LTXJobRequest(
            model_type=self.model_type,
            prompt=self.prompt,
            num_inference_steps=self.steps,
            video_length=self.frame_count,
            guidance_scale=self.cfg_scale,
            resolution=self.resolution,
            seed=seed,
        )


# =============================================================================
# Result Model
# =============================================================================

class VideoGenerationResult(BaseModel):
    """Result of video generation."""
    
    video_path: str = Field(description="Path to the generated video file")
    audio_path: Optional[str] = Field(
        default=None,
        description="Path to separate audio file (if generated)"
    )
    duration_seconds: float = Field(description="Actual video duration")
    width: int = Field(description="Video width")
    height: int = Field(description="Video height")
    fps: int = Field(description="Frames per second")
    prompt_id: str = Field(description="Job ID for tracking")


# =============================================================================
# LTX Video Client
# =============================================================================

class LTXVideoClient:
    """
    Async client for LTX Video Generation API.
    
    Provides methods to submit jobs, track progress, and download outputs.
    
    Example:
        async with LTXVideoClient(config) as client:
            result = await client.generate_video(params)
    """
    
    def __init__(self, config: Optional[LTXVideoConfig] = None):
        """
        Initialize the LTX Video client.
        
        Args:
            config: Server configuration (defaults to localhost:8081)
        """
        self.config = config or LTXVideoConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "LTXVideoClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def submit_job(self, request: LTXJobRequest) -> LTXJobResponse:
        """
        Submit a video generation job to the API.
        
        Args:
            request: Job request parameters
            
        Returns:
            LTXJobResponse with job_id and initial status
            
        Raises:
            httpx.HTTPStatusError: On HTTP error
        """
        client = self._get_client()
        
        # Build payload, excluding None values
        payload = request.model_dump(exclude_none=True)
        
        response = await client.post(
            f"{self.config.server_url}/jobs",
            json=payload
        )
        response.raise_for_status()
        
        return LTXJobResponse(**response.json())
    
    async def get_status(self, job_id: str) -> LTXJobStatusResponse:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID to query
            
        Returns:
            LTXJobStatusResponse with current status
        """
        client = self._get_client()
        
        response = await client.get(
            f"{self.config.server_url}/jobs/{job_id}/status"
        )
        response.raise_for_status()
        
        return LTXJobStatusResponse(**response.json())
    
    async def download_result(
        self,
        job_id: str,
        output_dir: str = "./output"
    ) -> str:
        """
        Download the generated video from a completed job.
        
        Args:
            job_id: The job ID to download
            output_dir: Local directory to save to
            
        Returns:
            Path to the downloaded video file
            
        Raises:
            httpx.HTTPStatusError: On HTTP error (including 202 if not ready)
        """
        client = self._get_client()
        
        response = await client.get(
            f"{self.config.server_url}/jobs/{job_id}/result"
        )
        response.raise_for_status()
        
        # Extract filename from Content-Disposition header
        content_disposition = response.headers.get("content-disposition", "")
        filename = f"{job_id}.mp4"  # Default filename
        
        if "filename=" in content_disposition:
            # Parse filename from header
            parts = content_disposition.split("filename=")
            if len(parts) > 1:
                filename = parts[1].strip('"\'')
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        file_path = output_path / filename
        file_path.write_bytes(response.content)
        
        return str(file_path)
    
    async def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LTXJobStatusResponse:
        """
        Wait for a job to complete.
        
        Args:
            job_id: The job ID to wait for
            progress_callback: Optional callback(progress, message)
            
        Returns:
            Final status response
            
        Raises:
            RuntimeError: If job fails
        """
        while True:
            status = await self.get_status(job_id)
            
            if status.status == JobStatus.PENDING.value:
                if progress_callback:
                    progress_callback(0.1, "Waiting in queue...")
            
            elif status.status == JobStatus.PROCESSING.value:
                progress_msg = status.progress or "Generating video..."
                if progress_callback:
                    progress_callback(0.5, progress_msg)
            
            elif status.status == JobStatus.COMPLETED.value:
                if progress_callback:
                    progress_callback(0.9, "Complete!")
                return status
            
            elif status.status == JobStatus.FAILED.value:
                raise RuntimeError(
                    f"Job {job_id} failed: {status.error or 'Unknown error'}"
                )
            
            await asyncio.sleep(self.config.poll_interval)
    
    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> VideoGenerationResult:
        """
        Generate a video using LTX-2.
        
        This is the high-level method that:
        1. Submits the job
        2. Waits for completion
        3. Downloads the output
        4. Returns the result
        
        Args:
            params: Video generation parameters
            output_dir: Directory to save output files
            progress_callback: Optional callback(progress, message)
            
        Returns:
            VideoGenerationResult with paths to output files
        """
        if progress_callback:
            progress_callback(0.0, "Submitting job...")
        
        # Convert params to job request
        job_request = params.to_job_request()
        
        # Submit job
        job_response = await self.submit_job(job_request)
        job_id = job_response.job_id
        
        if progress_callback:
            progress_callback(0.05, f"Job submitted: {job_id}")
        
        # Wait for completion
        await self.wait_for_completion(
            job_id,
            progress_callback=progress_callback
        )
        
        if progress_callback:
            progress_callback(0.9, "Downloading output...")
        
        # Download the video
        video_path = await self.download_result(job_id, output_dir)
        
        if progress_callback:
            progress_callback(1.0, "Done!")
        
        return VideoGenerationResult(
            video_path=video_path,
            audio_path=None,
            duration_seconds=params.actual_duration,
            width=params.width,
            height=params.height,
            fps=params.fps,
            prompt_id=job_id
        )


# Backward compatibility alias
ComfyUIClient = LTXVideoClient


# =============================================================================
# Convenience Function
# =============================================================================

async def generate_video(
    prompt: str,
    output_dir: str = "./output",
    server_url: str = "http://localhost:8081",
    **kwargs
) -> VideoGenerationResult:
    """
    Generate a video with a single function call.
    
    Args:
        prompt: Text prompt for video generation
        output_dir: Directory to save output
        server_url: LTX Video API server URL
        **kwargs: Additional VideoGenerationParams fields
        
    Returns:
        VideoGenerationResult
    """
    config = LTXVideoConfig(server_url=server_url)
    params = VideoGenerationParams(prompt=prompt, **kwargs)
    
    async with LTXVideoClient(config) as client:
        return await client.generate_video(params, output_dir=output_dir)
