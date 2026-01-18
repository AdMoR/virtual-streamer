"""
ComfyUI Client for LTX-2 Video Generation

Async client for interacting with ComfyUI server API to generate videos
using the LTX-2 text-to-video model.

Usage:
    from virtual_streamer.video_generation.comfyui_client import (
        ComfyUIClient, ComfyUIConfig, VideoGenerationParams
    )
    
    async with ComfyUIClient(ComfyUIConfig()) as client:
        result = await client.generate_video(
            VideoGenerationParams(prompt="A serene forest at sunrise")
        )
        print(f"Video saved to: {result.video_path}")
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
from pydantic import BaseModel, Field, computed_field


class ComfyUIConfig(BaseModel):
    """Configuration for ComfyUI server connection."""
    
    server_url: str = Field(
        default="http://localhost:8188",
        description="Base URL of the ComfyUI server"
    )
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds (video generation can be slow)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for ComfyUI cloud or partner nodes"
    )
    
    @computed_field
    @property
    def ws_url(self) -> str:
        """WebSocket URL derived from server URL."""
        return self.server_url.replace("http://", "ws://").replace("https://", "wss://")


class VideoGenerationParams(BaseModel):
    """Parameters for LTX-2 video generation."""
    
    prompt: str = Field(
        description="Text prompt describing the video to generate"
    )
    negative_prompt: str = Field(
        default="",
        description="What to avoid in the generated video"
    )
    width: int = Field(
        default=768,
        ge=256,
        le=2048,
        description="Video width (must be multiple of 32)"
    )
    height: int = Field(
        default=512,
        ge=256,
        le=2048,
        description="Video height (must be multiple of 32)"
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
        default=20,
        ge=1,
        le=100,
        description="Number of sampling steps"
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
    enable_audio: bool = Field(
        default=True,
        description="Generate synchronized audio"
    )
    
    @computed_field
    @property
    def frame_count(self) -> int:
        """
        Calculate frame count satisfying LTX-2's 8n + 1 constraint.
        """
        raw_frames = int(self.duration_seconds * self.fps)
        # Round to nearest valid value: 8n + 1
        n = round((raw_frames - 1) / 8)
        return max(8 * n + 1, 9)  # Minimum 9 frames
    
    @computed_field
    @property
    def actual_duration(self) -> float:
        """Actual duration based on frame_count constraint."""
        return self.frame_count / self.fps


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
    prompt_id: str = Field(description="ComfyUI prompt ID for tracking")


# Default LTX-2 workflow template
# This is a simplified template - in production, export from ComfyUI in API format
LTX2_WORKFLOW_TEMPLATE: Dict[str, Any] = {
    "3": {
        "class_type": "LTXVLoader",
        "inputs": {
            "ckpt_name": "ltxv-13b-0.9.7-dev.safetensors",
            "dtype": "bfloat16"
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "{prompt}",
            "clip": ["3", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "{negative_prompt}",
            "clip": ["3", 1]
        }
    },
    "10": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": "{width}",
            "height": "{height}",
            "length": "{frame_count}",
            "batch_size": 1
        }
    },
    "13": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "{seed}",
            "steps": "{steps}",
            "cfg": "{cfg_scale}",
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["3", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["10", 0]
        }
    },
    "17": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["13", 0],
            "vae": ["3", 2]
        }
    },
    "19": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": "{fps}",
            "loop_count": 0,
            "filename_prefix": "ltx2_output",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True,
            "images": ["17", 0]
        }
    }
}


class ComfyUIClient:
    """
    Async client for ComfyUI server API.
    
    Provides methods to submit workflows, track progress, and download outputs.
    
    Example:
        async with ComfyUIClient(config) as client:
            result = await client.generate_video(params)
    """
    
    def __init__(
        self,
        config: Optional[ComfyUIConfig] = None,
        workflow_template: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ComfyUI client.
        
        Args:
            config: Server configuration (defaults to localhost:8188)
            workflow_template: Custom workflow template (defaults to LTX-2 template)
        """
        self.config = config or ComfyUIConfig()
        self.workflow_template = workflow_template or LTX2_WORKFLOW_TEMPLATE
        self._client: Optional[httpx.AsyncClient] = None
        self._client_id = str(uuid.uuid4())
    
    async def __aenter__(self) -> "ComfyUIClient":
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
    
    def _build_workflow(self, params: VideoGenerationParams) -> Dict[str, Any]:
        """
        Build workflow JSON from template and parameters.
        
        Args:
            params: Video generation parameters
            
        Returns:
            Workflow dict ready for submission
        """
        # Deep copy the template
        workflow = json.loads(json.dumps(self.workflow_template))
        
        # Parameter substitutions
        substitutions = {
            "{prompt}": params.prompt,
            "{negative_prompt}": params.negative_prompt,
            "{width}": params.width,
            "{height}": params.height,
            "{frame_count}": params.frame_count,
            "{fps}": params.fps,
            "{steps}": params.steps,
            "{cfg_scale}": params.cfg_scale,
            "{seed}": params.seed if params.seed >= 0 else int(uuid.uuid4().int % (2**32)),
        }
        
        # Recursively substitute values
        def substitute(obj: Any) -> Any:
            if isinstance(obj, str):
                for key, value in substitutions.items():
                    if obj == key:
                        return value
                    elif key in obj:
                        obj = obj.replace(key, str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            return obj
        
        return substitute(workflow)
    
    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Submit a workflow to the ComfyUI queue.
        
        Args:
            workflow: Workflow dict in ComfyUI API format
            
        Returns:
            Prompt ID for tracking the execution
            
        Raises:
            httpx.HTTPStatusError: On HTTP error
        """
        client = self._get_client()
        
        payload = {
            "prompt": workflow,
            "client_id": self._client_id
        }
        
        if self.config.api_key:
            payload["extra_data"] = {
                "api_key_comfy_org": self.config.api_key
            }
        
        response = await client.post(
            f"{self.config.server_url}/prompt",
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data["prompt_id"]
    
    async def get_queue(self) -> Dict[str, Any]:
        """
        Get the current queue status.
        
        Returns:
            Queue information dict
        """
        client = self._get_client()
        response = await client.get(f"{self.config.server_url}/queue")
        response.raise_for_status()
        return response.json()
    
    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get execution history for a prompt.
        
        Args:
            prompt_id: The prompt ID to query
            
        Returns:
            History dict containing outputs
        """
        client = self._get_client()
        response = await client.get(
            f"{self.config.server_url}/history/{prompt_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_dir: str = "./output",
        output_type: str = "output"
    ) -> str:
        """
        Download a generated output file from the server.
        
        Args:
            filename: Name of the file to download
            subfolder: Subfolder on the server
            output_dir: Local directory to save to
            output_type: Type of output (output, input, temp)
            
        Returns:
            Path to the downloaded file
        """
        client = self._get_client()
        
        params = {
            "filename": filename,
            "type": output_type
        }
        if subfolder:
            params["subfolder"] = subfolder
        
        response = await client.get(
            f"{self.config.server_url}/view",
            params=params
        )
        response.raise_for_status()
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        file_path = output_path / filename
        file_path.write_bytes(response.content)
        
        return str(file_path)
    
    async def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for a prompt to complete execution.
        
        Args:
            prompt_id: The prompt ID to wait for
            poll_interval: Seconds between status checks
            progress_callback: Optional callback(progress, message)
            
        Returns:
            History dict with outputs
        """
        while True:
            # Check queue status
            queue = await self.get_queue()
            
            # Check if still running
            running = queue.get("queue_running", [])
            pending = queue.get("queue_pending", [])
            
            is_running = any(item[1] == prompt_id for item in running)
            is_pending = any(item[1] == prompt_id for item in pending)
            
            if is_pending and progress_callback:
                progress_callback(0.1, "Waiting in queue...")
            elif is_running and progress_callback:
                progress_callback(0.5, "Generating video...")
            
            if not is_running and not is_pending:
                # Check if completed
                history = await self.get_history(prompt_id)
                if prompt_id in history:
                    if progress_callback:
                        progress_callback(1.0, "Complete!")
                    return history[prompt_id]
            
            await asyncio.sleep(poll_interval)
    
    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> VideoGenerationResult:
        """
        Generate a video using LTX-2.
        
        This is the high-level method that:
        1. Builds the workflow from parameters
        2. Submits to the queue
        3. Waits for completion
        4. Downloads the output
        5. Returns the result
        
        Args:
            params: Video generation parameters
            output_dir: Directory to save output files
            progress_callback: Optional callback(progress, message)
            
        Returns:
            VideoGenerationResult with paths to output files
        """
        if progress_callback:
            progress_callback(0.0, "Building workflow...")
        
        # Build workflow
        workflow = self._build_workflow(params)
        
        if progress_callback:
            progress_callback(0.05, "Submitting to queue...")
        
        # Submit to queue
        prompt_id = await self.queue_prompt(workflow)
        
        # Wait for completion
        history = await self.wait_for_completion(
            prompt_id,
            progress_callback=progress_callback
        )
        
        if progress_callback:
            progress_callback(0.9, "Downloading output...")
        
        # Find output files in history
        video_path = None
        audio_path = None
        
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            # Look for video/gif output
            if "gifs" in node_output:
                for gif_info in node_output["gifs"]:
                    filename = gif_info.get("filename")
                    subfolder = gif_info.get("subfolder", "")
                    if filename:
                        video_path = await self.download_output(
                            filename=filename,
                            subfolder=subfolder,
                            output_dir=output_dir
                        )
                        break
            
            # Look for images/video output
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    if filename and (filename.endswith(".mp4") or filename.endswith(".webm")):
                        video_path = await self.download_output(
                            filename=filename,
                            subfolder=subfolder,
                            output_dir=output_dir
                        )
                        break
        
        if not video_path:
            raise RuntimeError(
                f"No video output found in history for prompt {prompt_id}. "
                f"Outputs: {list(outputs.keys())}"
            )
        
        if progress_callback:
            progress_callback(1.0, "Done!")
        
        return VideoGenerationResult(
            video_path=video_path,
            audio_path=audio_path,
            duration_seconds=params.actual_duration,
            width=params.width,
            height=params.height,
            fps=params.fps,
            prompt_id=prompt_id
        )


# Convenience function for one-off generation
async def generate_video(
    prompt: str,
    output_dir: str = "./output",
    server_url: str = "http://localhost:8188",
    **kwargs
) -> VideoGenerationResult:
    """
    Generate a video with a single function call.
    
    Args:
        prompt: Text prompt for video generation
        output_dir: Directory to save output
        server_url: ComfyUI server URL
        **kwargs: Additional VideoGenerationParams fields
        
    Returns:
        VideoGenerationResult
    """
    config = ComfyUIConfig(server_url=server_url)
    params = VideoGenerationParams(prompt=prompt, **kwargs)
    
    async with ComfyUIClient(config) as client:
        return await client.generate_video(params, output_dir=output_dir)
