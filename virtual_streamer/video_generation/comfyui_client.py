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


# Default LTX-2 workflow template (exported from ComfyUI in API format)
# This workflow includes audio generation and 2x spatial upscaling
LTX2_WORKFLOW_TEMPLATE: Dict[str, Any] = {
    "75": {
        "inputs": {
            "filename_prefix": "video/LTX-2",
            "format": "mp4",
            "codec": "auto",
            "video": ["92:97", 0]
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"}
    },
    "92:9": {
        "inputs": {
            "steps": 20,
            "max_shift": 2.05,
            "base_shift": 0.95,
            "stretch": True,
            "terminal": 0.1,
            "latent": ["92:56", 0]
        },
        "class_type": "LTXVScheduler",
        "_meta": {"title": "LTXVScheduler"}
    },
    "92:60": {
        "inputs": {
            "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
            "ckpt_name": "ltx-2-19b-dev-fp8.safetensors",
            "device": "default"
        },
        "class_type": "LTXAVTextEncoderLoader",
        "_meta": {"title": "LTXV Audio Text Encoder Loader"}
    },
    "92:73": {
        "inputs": {"sigmas": "0.909375, 0.725, 0.421875, 0.0"},
        "class_type": "ManualSigmas",
        "_meta": {"title": "ManualSigmas"}
    },
    "92:76": {
        "inputs": {"model_name": "ltx-2-spatial-upscaler-x2-1.0.safetensors"},
        "class_type": "LatentUpscaleModelLoader",
        "_meta": {"title": "Load Latent Upscale Model"}
    },
    "92:81": {
        "inputs": {
            "positive": ["92:22", 0],
            "negative": ["92:22", 1],
            "latent": ["92:80", 0]
        },
        "class_type": "LTXVCropGuides",
        "_meta": {"title": "LTXVCropGuides"}
    },
    "92:82": {
        "inputs": {
            "cfg": 1,
            "model": ["92:68", 0],
            "positive": ["92:81", 0],
            "negative": ["92:81", 1]
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "CFGGuider"}
    },
    "92:90": {
        "inputs": {
            "upscale_method": "lanczos",
            "scale_by": 0.5,
            "image": ["92:89", 0]
        },
        "class_type": "ImageScaleBy",
        "_meta": {"title": "Upscale Image By"}
    },
    "92:91": {
        "inputs": {"image": ["92:90", 0]},
        "class_type": "GetImageSize",
        "_meta": {"title": "Get Image Size"}
    },
    "92:51": {
        "inputs": {
            "frames_number": ["92:62", 0],
            "frame_rate": ["92:99", 0],
            "batch_size": 1,
            "audio_vae": ["92:48", 0]
        },
        "class_type": "LTXVEmptyLatentAudio",
        "_meta": {"title": "LTXV Empty Latent Audio"}
    },
    "92:22": {
        "inputs": {
            "frame_rate": ["92:102", 0],
            "positive": ["92:3", 0],
            "negative": ["92:4", 0]
        },
        "class_type": "LTXVConditioning",
        "_meta": {"title": "LTXVConditioning"}
    },
    "92:43": {
        "inputs": {
            "width": ["92:91", 0],
            "height": ["92:91", 1],
            "length": ["92:62", 0],
            "batch_size": 1
        },
        "class_type": "EmptyLTXVLatentVideo",
        "_meta": {"title": "EmptyLTXVLatentVideo"}
    },
    "92:56": {
        "inputs": {
            "video_latent": ["92:43", 0],
            "audio_latent": ["92:51", 0]
        },
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "LTXVConcatAVLatent"}
    },
    "92:4": {
        "inputs": {
            "text": "blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles",
            "clip": ["92:60", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "92:89": {
        "inputs": {"width": 1280, "height": 720, "batch_size": 1, "color": 0},
        "class_type": "EmptyImage",
        "_meta": {"title": "EmptyImage"}
    },
    "92:62": {
        "inputs": {"value": 121},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Length"}
    },
    "92:41": {
        "inputs": {
            "noise": ["92:11", 0],
            "guider": ["92:47", 0],
            "sampler": ["92:8", 0],
            "sigmas": ["92:9", 0],
            "latent_image": ["92:56", 0]
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"}
    },
    "92:67": {
        "inputs": {"noise_seed": 0},
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"}
    },
    "92:11": {
        "inputs": {"noise_seed": 10},
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"}
    },
    "92:80": {
        "inputs": {"av_latent": ["92:41", 0]},
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "LTXVSeparateAVLatent"}
    },
    "92:83": {
        "inputs": {
            "video_latent": ["92:84", 0],
            "audio_latent": ["92:80", 1]
        },
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "LTXVConcatAVLatent"}
    },
    "92:84": {
        "inputs": {
            "samples": ["92:81", 2],
            "upscale_model": ["92:76", 0],
            "vae": ["92:1", 2]
        },
        "class_type": "LTXVLatentUpsampler",
        "_meta": {"title": "spatial"}
    },
    "92:70": {
        "inputs": {
            "noise": ["92:67", 0],
            "guider": ["92:82", 0],
            "sampler": ["92:66", 0],
            "sigmas": ["92:73", 0],
            "latent_image": ["92:83", 0]
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"}
    },
    "92:3": {
        "inputs": {
            "text": "A cheerful puppet girl with yarn hair",
            "clip": ["92:60", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"}
    },
    "92:97": {
        "inputs": {
            "fps": ["92:102", 0],
            "images": ["92:98", 0],
            "audio": ["92:96", 0]
        },
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"}
    },
    "92:48": {
        "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        "class_type": "LTXVAudioVAELoader",
        "_meta": {"title": "LTXV Audio VAE Loader"}
    },
    "92:94": {
        "inputs": {"av_latent": ["92:70", 1]},
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "LTXVSeparateAVLatent"}
    },
    "92:98": {
        "inputs": {
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 4096,
            "temporal_overlap": 8,
            "samples": ["92:94", 0],
            "vae": ["92:1", 2]
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {"title": "VAE Decode (Tiled)"}
    },
    "92:96": {
        "inputs": {
            "samples": ["92:94", 1],
            "audio_vae": ["92:48", 0]
        },
        "class_type": "LTXVAudioVAEDecode",
        "_meta": {"title": "LTXV Audio VAE Decode"}
    },
    "92:47": {
        "inputs": {
            "cfg": 4,
            "model": ["92:1", 0],
            "positive": ["92:22", 0],
            "negative": ["92:22", 1]
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "CFGGuider"}
    },
    "92:102": {
        "inputs": {"value": 24.0},
        "class_type": "PrimitiveFloat",
        "_meta": {"title": "Frame Rate(float)"}
    },
    "92:99": {
        "inputs": {"value": 24},
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Frame Rate(int)"}
    },
    "92:68": {
        "inputs": {
            "lora_name": "ltx-2-19b-distilled-lora-384.safetensors",
            "strength_model": 1,
            "model": ["92:1", 0]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoraLoaderModelOnly"}
    },
    "92:8": {
        "inputs": {"sampler_name": "euler_ancestral"},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"}
    },
    "92:66": {
        "inputs": {"sampler_name": "euler_ancestral"},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"}
    },
    "92:1": {
        "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"}
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
    
    @staticmethod
    def load_workflow_from_file(workflow_path: str) -> Dict[str, Any]:
        """
        Load a workflow from a JSON file (API format).
        
        Args:
            workflow_path: Path to the workflow JSON file
            
        Returns:
            Workflow dict ready for use as template
        """
        path = Path(workflow_path)
        with open(path) as f:
            data = json.load(f)
        
        # If it's a UI format workflow with embedded API prompt, extract it
        if "extra" in data and "prompt" in data.get("extra", {}):
            return data["extra"]["prompt"]
        
        # If it has "prompt" at top level (wrapped API format), extract it
        if "prompt" in data and isinstance(data["prompt"], dict):
            # Check if it looks like API format (keys are node IDs)
            prompt = data["prompt"]
            if prompt and all(isinstance(v, dict) for v in prompt.values()):
                return prompt
        
        # Otherwise assume it's already in API format (dict of node IDs)
        # Node IDs are typically strings like "1", "92:3", etc.
        if data and all(isinstance(v, dict) and "class_type" in v for v in data.values()):
            return data
        
        raise ValueError(
            f"Could not parse workflow from {workflow_path}. "
            "Please export in API format from ComfyUI (enable Dev Mode first)."
        )
    
    def _build_workflow(self, params: VideoGenerationParams) -> Dict[str, Any]:
        """
        Build workflow JSON from template and parameters.
        
        Args:
            params: Video generation parameters
            
        Returns:
            Workflow dict ready for submission
        """
        import copy
        workflow = copy.deepcopy(self.workflow_template)
        
        # Generate seed if random
        seed = params.seed if params.seed >= 0 else int(uuid.uuid4().int % (2**32))
        
        # Inject prompt text (node 92:3)
        if "92:3" in workflow:
            workflow["92:3"]["inputs"]["text"] = params.prompt
        
        # Inject negative prompt (node 92:4)
        if "92:4" in workflow:
            workflow["92:4"]["inputs"]["text"] = params.negative_prompt or \
                "blurry, low quality, still frame, frames, watermark, overlay, titles"
        
        # Inject resolution - width/height (node 92:89 EmptyImage)
        if "92:89" in workflow:
            workflow["92:89"]["inputs"]["width"] = params.width
            workflow["92:89"]["inputs"]["height"] = params.height
        
        # Inject frame count (node 92:62 PrimitiveInt)
        if "92:62" in workflow:
            workflow["92:62"]["inputs"]["value"] = params.frame_count
        
        # Inject seed for both noise nodes
        if "92:11" in workflow:
            workflow["92:11"]["inputs"]["noise_seed"] = seed
        if "92:67" in workflow:
            workflow["92:67"]["inputs"]["noise_seed"] = seed
        
        # Inject frame rate (both float and int nodes)
        if "92:102" in workflow:
            workflow["92:102"]["inputs"]["value"] = float(params.fps)
        if "92:99" in workflow:
            workflow["92:99"]["inputs"]["value"] = params.fps
        
        # Inject CFG scale (node 92:47)
        if "92:47" in workflow:
            workflow["92:47"]["inputs"]["cfg"] = params.cfg_scale
        
        # Inject steps (node 92:9 LTXVScheduler)
        if "92:9" in workflow:
            workflow["92:9"]["inputs"]["steps"] = params.steps
        
        return workflow
    
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
            # Look for video output from SaveVideo node
            if "video" in node_output:
                video_info = node_output["video"]
                if isinstance(video_info, dict):
                    filename = video_info.get("filename")
                    subfolder = video_info.get("subfolder", "")
                    if filename:
                        video_path = await self.download_output(
                            filename=filename,
                            subfolder=subfolder,
                            output_dir=output_dir
                        )
                        break
            
            # Look for video/gif output (VHS_VideoCombine style)
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
            
            # Look for videos list output
            if "videos" in node_output:
                for vid_info in node_output["videos"]:
                    filename = vid_info.get("filename")
                    subfolder = vid_info.get("subfolder", "")
                    if filename:
                        video_path = await self.download_output(
                            filename=filename,
                            subfolder=subfolder,
                            output_dir=output_dir
                        )
                        break
        
        if not video_path:
            raise RuntimeError(
                f"No video output found in history for prompt {prompt_id}. "
                f"Available outputs: {json.dumps(outputs, indent=2)}"
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
