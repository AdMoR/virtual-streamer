"""
LTX Video Client (WanGP REST API)

Client for generating videos via a remote WanGP server (wangp_server.py).
Supports text-to-video, image-to-video, audio-conditioned i2v, and video-to-video.

Usage:
    from virtual_streamer.video_generation.ltx_client import (
        WanGPLTXClient, LTXVideoConfig, VideoGenerationParams
    )

    params = VideoGenerationParams(
        prompt="a person talking to camera",
        image_path="path/to/start.jpg",
        audio_path="path/to/voice.wav",   # optional — enables audio conditioning
    )
    async with WanGPLTXClient(LTXVideoConfig()) as client:
        result = await client.generate_video(params, output_dir="./output")
        print(result.video_path)

    # Video-to-video: edit an existing clip
    params = VideoGenerationParams(
        prompt="same scene but at night, neon lights",
        video_path="path/to/source.mp4",
        denoising_strength=0.7,
    )
    async with WanGPLTXClient(LTXVideoConfig()) as client:
        result = await client.generate_video(params, output_dir="./output")
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

_DEFAULTS = {
    "model_type":      "ltx2_22B_distilled",
    "resolution":      "1280x720",
    "frames":          97,
    "steps":           8,
    "guidance_scale":  3.0,
    "flow_shift":      3.0,
    "seed":            -1,
    "fps":             "24",
    "audio_scale":     1.0,
    "audio_guidance":  4.5,
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
}

_DEFAULTS_QUALITY = {
    "model_type":      "ltx2_22B",
    "resolution":      "1280x720",
    "frames":          97,
    "steps":           30,
    "guidance_scale":  3.0,
    "flow_shift":      3.0,
    "seed":            -1,
    "fps":             "24",
    "audio_scale":     1.0,
    "audio_guidance":  4.5,
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
}


_DEFAULTS_HIGH_QUALITY = {
    "model_type":      "ltx2_22B",
    "resolution":      "1280x720",
    "frames":          97,
    "steps":           30,
    "guidance_scale":  3.0,
    "flow_shift":      3.0,
    "seed":            -1,
    "fps":             "50",
    "audio_scale":     1.0,
    "audio_guidance":  4.5,
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
}

# Union-control IC-LoRA filename by model — required for DVG/PVG/OVG/EVG v2v modes.
_V2V_UNION_CONTROL_LORA: dict[str, str] = {
    "ltx2_22B_distilled":     "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "ltx2_22B_distilled_1_1": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "ltx2_distilled":         "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
}
_DEFAULT_V2V_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

# Named presets exposed to callers (API, UI, etc.)
VIDEO_PRESETS: dict[str, dict] = {
    "fast":         _DEFAULTS,
    "quality":      _DEFAULTS_QUALITY,
    "high_quality": _DEFAULTS_HIGH_QUALITY,
}


# =============================================================================
# Configuration
# =============================================================================

class LTXVideoConfig(BaseModel):
    """Connection settings for the remote WanGP REST server."""

    server_url: str = Field(
        default="http://localhost:8082",
        description="URL of the running WanGP REST server (wangp_server.py)",
    )
    timeout: float = Field(
        default=600.0,
        description="HTTP timeout in seconds for uploads and downloads",
    )
    stream_timeout: float = Field(
        default=1800.0,
        description="Timeout in seconds for the SSE event stream",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for X-API-Key header (matches WANGP_API_KEY env var on server)",
    )


# =============================================================================
# User-facing Parameters
# =============================================================================

class VideoGenerationParams(BaseModel):
    """
    Parameters for LTX video generation via WanGP.

    Primary fields:
        prompt, image_path, audio_path, model_type, resolution, frames,
        steps, guidance_scale, flow_shift, seed, force_fps,
        audio_scale, audio_guidance, negative_prompt

    Legacy / convenience fields (auto-converted to primary fields):
        width, height     → resolution  (e.g. 1280 × 720 → "1280x720")
        duration_seconds  → frames      (computed as 8n+1 nearest)
        fps               → force_fps   (str cast)
        cfg_scale         → guidance_scale
        enable_audio      (informational; audio conditioning uses audio_path)
    """

    # --- Core ---
    prompt: str = Field(description="Text prompt describing the video content")
    negative_prompt: str = Field(default=_DEFAULTS["negative_prompt"])

    # --- I2V inputs ---
    image_path: Optional[str] = Field(
        default=None,
        description="Local path to the start image (JPEG/PNG). Required for i2v.",
    )
    audio_path: Optional[str] = Field(
        default=None,
        description=(
            "Local path to a conditioning audio file (WAV/MP3/FLAC). "
            "When set the audio drives video motion. "
            "Only distilled LTX models support this."
        ),
    )

    # --- V2V inputs ---
    video_path: Optional[str] = Field(
        default=None,
        description=(
            "Local path to a source video (MP4/WebM). "
            "When set the generation runs in video-to-video mode. "
            "Can be combined with image_path to pin the first frame via image_start."
        ),
    )
    video_prompt_type: str = Field(
        default="DVG",
        description=(
            "V2V preprocessing mode. 'DVG'=depth map, 'PVG'=pose, 'OVG'=pose+align, "
            "'EVG'=Canny edges (all use union-control LoRA). 'VG'=raw (task-specific LoRA)."
        ),
    )
    denoising_strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "V2V Control Video Strength: higher = output closer to source (0.9–1.0 subtle, "
            "0.6–0.8 moderate, 0.3–0.5 light)."
        ),
    )

    # --- Generation settings ---
    model_type: str = Field(default=_DEFAULTS["model_type"])
    resolution: str = Field(
        default="",
        description="WxH string (e.g. '1280x720'). Derived from width/height when empty.",
    )
    frames: int = Field(
        default=0,
        description="Frame count (must satisfy 8n+1). Derived from duration_seconds/fps when 0.",
    )
    steps: int = Field(default=_DEFAULTS["steps"], ge=1, le=200)
    guidance_scale: float = Field(default=_DEFAULTS["guidance_scale"], ge=0.0, le=30.0)
    flow_shift: float = Field(default=_DEFAULTS["flow_shift"])
    seed: int = Field(default=_DEFAULTS["seed"], description="-1 for random")
    force_fps: str = Field(
        default="",
        description="Output frame-rate string. Derived from fps when empty.",
    )
    audio_scale: float = Field(
        default=_DEFAULTS["audio_scale"],
        description="Prompt Audio Strength (0–1): how strongly audio drives video.",
    )
    audio_guidance: float = Field(
        default=_DEFAULTS["audio_guidance"],
        description="Audio CFG guidance scale (1–20): higher = more audio-faithful.",
    )

    # --- Legacy / convenience fields ---
    width: int = Field(default=1280, ge=64)
    height: int = Field(default=720, ge=64)
    duration_seconds: float = Field(default=4.0, ge=0.1)
    fps: int = Field(default=24, ge=1)
    cfg_scale: float = Field(
        default=_DEFAULTS["guidance_scale"],
        description="Legacy alias for guidance_scale.",
    )
    enable_audio: bool = Field(
        default=False,
        description="Informational flag. Actual audio conditioning uses audio_path.",
    )

    @model_validator(mode="after")
    def _resolve_fields(self) -> "VideoGenerationParams":
        if self.cfg_scale != _DEFAULTS["guidance_scale"]:
            self.guidance_scale = self.cfg_scale
        return self

    # --- Computed properties ---

    @property
    def effective_resolution(self) -> str:
        return self.resolution if self.resolution else f"{self.width}x{self.height}"

    @property
    def effective_frames(self) -> int:
        if self.frames > 0:
            return self.frames
        raw = int(self.duration_seconds * self.fps)
        n = round((raw - 1) / 8)
        return max(8 * n + 1, 9)

    @property
    def effective_fps(self) -> str:
        return self.force_fps if self.force_fps else str(self.fps)

    @property
    def actual_duration(self) -> float:
        fps_val = int(self.effective_fps) if self.effective_fps.isdigit() else self.fps
        return self.effective_frames / fps_val

    @property
    def frame_count(self) -> int:
        return self.effective_frames

    @classmethod
    def from_preset(
        cls,
        preset_name: str = "fast",
        prompt: str = "",
        **overrides,
    ) -> "VideoGenerationParams":
        """Create VideoGenerationParams from a named quality preset ('fast', 'quality', 'high_quality')."""
        preset = VIDEO_PRESETS[preset_name]
        return cls(prompt=prompt, **preset, **overrides)


# =============================================================================
# Result Model
# =============================================================================

class VideoGenerationResult(BaseModel):
    """Result returned after successful video generation."""

    video_path: str = Field(description="Local path to the downloaded video file")
    audio_path: Optional[str] = Field(default=None, description="Separate audio file (if any)")
    duration_seconds: float
    width: int
    height: int
    fps: int
    prompt_id: str = Field(description="Identifier for this generation (basename of video file)")


# =============================================================================
# Interface
# =============================================================================

class LTXClientInterface(ABC):
    """Abstract interface for LTX video generation clients."""

    @abstractmethod
    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoGenerationResult:
        """Generate a video from *params* and return the result."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources."""
        ...

    async def __aenter__(self) -> "LTXClientInterface":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


# =============================================================================
# WanGP REST Implementation
# =============================================================================

class WanGPLTXClient(LTXClientInterface):
    """
    LTX video generation backed by the WanGP REST server (wangp_server.py).

    Replaces the old Gradio-based 5-step pipeline with a simple REST workflow:
      1. Upload any local files (image, audio) → file_id
      2. POST /jobs with a settings dict → job_id
      3. Stream SSE events from GET /jobs/{job_id}/events until completed
      4. GET /files/{filename} to download the output video
    """

    def __init__(self, config: Optional[LTXVideoConfig] = None) -> None:
        self.config = config or LTXVideoConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.config.api_key:
            h["X-API-Key"] = self.config.api_key
        return h

    def _check_health(self) -> None:
        url = f"{self.config.server_url.rstrip('/')}/health"
        r = requests.get(url, headers=self._headers(), timeout=10)
        body = r.json()
        if r.status_code != 200 or not body.get("runtime_loaded"):
            raise RuntimeError(f"WanGP server not ready: {body}")
        print(f"[health] status={body.get('status')}  queue_depth={body.get('queue_depth')}")

    def _upload_file(self, path: str) -> str:
        """Upload *path* to the server and return the file_id."""
        url = f"{self.config.server_url.rstrip('/')}/files/upload"
        p = Path(path)
        with p.open("rb") as fh:
            r = requests.post(
                url,
                files={"file": (p.name, fh)},
                headers=self._headers(),
                timeout=self.config.timeout,
            )
        r.raise_for_status()
        file_id: str = r.json()["file_id"]
        print(f"  uploaded {p.name} → {file_id}")
        return file_id

    def _build_settings(
        self,
        params: VideoGenerationParams,
        image_file_id: Optional[str],
        audio_file_id: Optional[str],
        video_file_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "model_type":           params.model_type,
            "prompt":               params.prompt,
            "negative_prompt":      params.negative_prompt,
            "num_inference_steps":  params.steps,
            "video_length":         params.effective_frames,
            "resolution":           params.effective_resolution,
            "guidance_scale":       params.guidance_scale,
            "flow_shift":           params.flow_shift,
            "seed":                 params.seed,
            "force_fps":            params.effective_fps,
        }

        if video_file_id is not None:
            settings["video_guide"] = f"file:{video_file_id}"
            settings["video_prompt_type"] = params.video_prompt_type
            settings["denoising_strength"] = params.denoising_strength
            # DVG/PVG/OVG/EVG all require the union-control IC-LoRA; VG needs a task-specific one.
            if params.video_prompt_type != "VG":
                lora = _V2V_UNION_CONTROL_LORA.get(params.model_type, _DEFAULT_V2V_LORA)
                settings["activated_loras"] = [lora]
                settings["loras_multipliers"] = "1"

        if image_file_id is not None:
            settings["image_start"] = f"file:{image_file_id}"
            settings["image_prompt_type"] = "S"

        if audio_file_id is not None:
            settings["audio_guide"] = f"file:{audio_file_id}"
            settings["audio_prompt_type"] = "A"
            settings["audio_guidance_scale"] = params.audio_guidance
            settings["audio_scale"] = params.audio_scale

        return settings

    def _submit_job(self, settings: Dict[str, Any]) -> tuple[str, int]:
        """Submit a job and return (job_id, queue_position)."""
        url = f"{self.config.server_url.rstrip('/')}/jobs"
        r = requests.post(
            url,
            json={"settings": settings},
            headers=self._headers(),
            timeout=30,
        )
        if r.status_code != 202:
            raise RuntimeError(f"Job submission failed ({r.status_code}): {r.json()}")
        body = r.json()
        job_id: str = body["job_id"]
        queue_pos: int = body.get("queue_position", 0)
        print(f"[submit] job_id={job_id}  queue_position={queue_pos}")
        return job_id, queue_pos

    def _poll_job(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[float, str], None]],
        poll_interval: float = 2.0,
    ) -> List[str]:
        """Poll GET /jobs/{job_id} until the job is done. Returns list of output URLs."""
        url = f"{self.config.server_url.rstrip('/')}/jobs/{job_id}"
        print(f"[poll] polling {url} …")
        deadline = time.time() + self.config.stream_timeout

        while time.time() < deadline:
            r = requests.get(url, headers=self._headers(), timeout=30)
            if r.status_code == 404:
                raise RuntimeError(f"Job {job_id!r} not found (evicted or invalid)")
            r.raise_for_status()
            body = r.json()
            status = body.get("status")

            if status in ("queued", "running"):
                pct   = float(body.get("progress") or 0.0)
                phase = body.get("phase") or status
                pos   = body.get("queue_position")
                bar   = "#" * int(pct * 10) + "-" * (10 - int(pct * 10))
                if status == "queued":
                    print(f"\r  [queue] position={pos}    ", end="", flush=True)
                else:
                    print(
                        f"\r  [{bar}] {pct:5.1%}  {phase}    ",
                        end="",
                        flush=True,
                    )
                if progress_callback:
                    progress_callback(pct, phase)
                time.sleep(poll_interval)
                continue

            # Terminal states
            print()
            files   = body.get("generated_files", [])
            errors  = body.get("errors", [])
            success = body.get("success", False)

            if status == "completed" and success:
                print(f"  [completed] files={files}")
                return files

            raise RuntimeError(
                f"WanGP job {status}: success={success}  errors={errors}"
            )

        raise TimeoutError(
            f"Timed out waiting for job {job_id!r} after {self.config.stream_timeout}s"
        )

    def _download_output(self, file_url: str, output_dir: str) -> str:
        """Download *file_url* (a complete URL) and save to *output_dir*. Returns local path."""
        from virtual_streamer.utils.file_manager import get_file_manager
        fm = get_file_manager()
        filename = file_url.rstrip("/").rsplit("/", 1)[-1]
        dst = Path(output_dir) / fm.naming.final_output_filename(filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {filename} …")
        with requests.get(
            file_url,
            headers=self._headers(),
            stream=True,
            timeout=self.config.timeout,
        ) as r:
            r.raise_for_status()
            with dst.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f"  saved → {dst}")
        return str(dst)

    def _run_generation_sync(
        self,
        params: VideoGenerationParams,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> List[str]:
        """Execute the full REST generation pipeline synchronously. Returns local paths."""
        base = self.config.server_url.rstrip("/")

        if params.video_path and params.image_path:
            mode_label = "v2v+image_start"
        elif params.video_path:
            mode_label = "v2v"
        elif params.image_path:
            mode_label = "audio-conditioned i2v" if params.audio_path else "i2v"
        else:
            mode_label = "t2v"

        print(f"[1] Health check ({base})")
        self._check_health()

        image_file_id: Optional[str] = None
        audio_file_id: Optional[str] = None
        video_file_id: Optional[str] = None

        if params.video_path:
            print(f"[2a] Uploading source video ({mode_label})")
            video_file_id = self._upload_file(params.video_path)

        if params.image_path:
            print(f"[2b] Uploading start image")
            image_file_id = self._upload_file(params.image_path)

        if params.audio_path:
            print("[2c] Uploading audio guide")
            audio_file_id = self._upload_file(params.audio_path)

        print(f"[3] Submitting job  model={params.model_type}  mode={mode_label}")
        settings = self._build_settings(params, image_file_id, audio_file_id, video_file_id)
        job_id, _ = self._submit_job(settings)

        print("[4] Polling job status …")
        t0 = time.time()
        filenames = self._poll_job(job_id, progress_callback)
        print(f"    finished in {time.time() - t0:.1f}s")

        if not filenames:
            raise RuntimeError("WanGP generation produced no output files")

        print("[5] Downloading output file(s)")
        local_paths: List[str] = []
        for name in filenames:
            local_paths.append(self._download_output(name, output_dir))

        return local_paths

    # ------------------------------------------------------------------
    # LTXClientInterface implementation
    # ------------------------------------------------------------------

    async def close(self) -> None:
        pass

    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoGenerationResult:
        """
        Generate a video via the WanGP REST server.

        The blocking HTTP calls run in a thread so this coroutine stays non-blocking.

        Args:
            params: Generation parameters.
                - Text-to-video: only ``prompt`` required.
                - Image-to-video: set ``image_path``.
                - Audio-conditioned i2v: set both ``image_path`` and ``audio_path``.
                - Video-to-video: set ``video_path``.
                - V2V with pinned first frame: set both ``video_path`` and ``image_path``.
            output_dir: Local directory to save the downloaded video.
            progress_callback: Optional ``callback(fraction, message)``.

        Returns:
            :class:`VideoGenerationResult` with local video path and metadata.

        Raises:
            RuntimeError: If the server is not ready, inference fails, or no files produced.
        """
        if progress_callback:
            progress_callback(0.0, "Starting WanGP generation…")

        output_files: List[str] = await asyncio.to_thread(
            self._run_generation_sync, params, output_dir, progress_callback
        )

        if not output_files:
            raise RuntimeError("WanGP generation produced no output files")

        if progress_callback:
            progress_callback(1.0, "Done!")

        w_str, h_str = params.effective_resolution.split("x")
        fps_val = int(params.effective_fps) if params.effective_fps.isdigit() else params.fps

        return VideoGenerationResult(
            video_path=output_files[0],
            audio_path=None,
            duration_seconds=params.actual_duration,
            width=int(w_str),
            height=int(h_str),
            fps=fps_val,
            prompt_id=Path(output_files[0]).stem,
        )


# =============================================================================
# Convenience async function
# =============================================================================

async def generate_video(
    prompt: str,
    image_path: Optional[str] = None,
    output_dir: str = "./output",
    server_url: str = "http://localhost:8082",
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    **kwargs,
) -> VideoGenerationResult:
    """
    Generate a video with a single async call.

    Modes (determined by which path arguments are set):
        - Text-to-video: only ``prompt``.
        - Image-to-video: ``image_path`` (+ optional ``audio_path``).
        - Video-to-video: ``video_path``.
        - V2V with pinned first frame: ``video_path`` + ``image_path``.

    Args:
        prompt: Text prompt describing the video content.
        image_path: Local path to the start image. Used alone for i2v, or with video_path to pin the first frame.
        output_dir: Directory to save the output video.
        server_url: URL of the remote WanGP REST server.
        audio_path: Optional conditioning audio file. Enables audio-driven i2v.
        video_path: Local path to the source video (v2v). Can be combined with image_path.
        **kwargs: Additional :class:`VideoGenerationParams` fields
                  (e.g. ``denoising_strength``, ``video_prompt_type``).

    Returns:
        :class:`VideoGenerationResult`
    """
    config = LTXVideoConfig(server_url=server_url)
    params = VideoGenerationParams(
        prompt=prompt,
        image_path=image_path,
        audio_path=audio_path,
        video_path=video_path,
        **kwargs,
    )
    async with WanGPLTXClient(config) as client:
        return await client.generate_video(params, output_dir=output_dir)