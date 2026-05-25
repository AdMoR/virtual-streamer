"""
LTX Video Client (WanGP REST API — /jobs/raw endpoint)

Client for generating videos via a remote WanGP server using POST /jobs/raw.
All generation flags must be set explicitly by the caller — there are no
server-side auto-defaults applied by this endpoint.

Supported modes (caller sets flags in image_prompt_type / video_prompt_type / audio_prompt_type):
  - Text-to-video (t2v):          no file paths needed
  - Image-to-video (i2v):         image_start + image_prompt_type="S"
  - Start + end frame (SE):       image_start + image_end + image_prompt_type="SE"
  - Audio-conditioned i2v:        audio_guide + audio_prompt_type="A"
  - Video-to-video (depth):       video_guide + video_prompt_type="DVG"
  - Video-to-video (pose):        video_guide + video_prompt_type="PVG"
  - Identity reference (I mode):  image_refs=["ref.jpg"] + video_prompt_type="I"
  - Keyframe interpolation:       model_type="ltx2_22B_keyframe" + keyframes=[...]
  - Talking head (ID-LoRA):       audio_guide + image_start + audio_prompt_type="A1O"
                                  + image_prompt_type="S"

Upload media with POST /files/upload, then reference it via "file:<file_id>".  The
client handles uploads and reference substitution automatically.

Usage::

    params = VideoGenerationParams(
        prompt="A woman walking in a park, cinematic, soft light",
        image_start="path/to/start.jpg",
        image_prompt_type="S",
        audio_guide="path/to/voice.wav",
        audio_prompt_type="A",
        audio_scale=1.0,
        audio_guidance_scale=4.5,
        model_type="ltx2_22B_distilled_1_1",
        resolution="832x480",
        video_length=97,
        num_inference_steps=8,
    )
    async with WanGPLTXClient(LTXVideoConfig()) as client:
        result = await client.generate_video(params, output_dir="./output")
        print(result.video_path)

    # Keyframe interpolation::

    params = VideoGenerationParams(
        prompt="Smooth cinematic transition through an autumn forest",
        model_type="ltx2_22B_keyframe",
        keyframes=[
            ["frame0.png",   0, 1.0],
            ["frame60.png", 60, 1.0],
            ["frame120.png", 120, 1.0],
        ],
        video_length=121,
        resolution="1280x720",
        num_inference_steps=40,
    )
"""

from __future__ import annotations

import asyncio
import struct
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

_DEFAULTS_FAST: dict = {
    "model_type":          "ltx2_22B_distilled_1_1",
    "resolution":          "1280x720",
    "video_length":        97,
    "num_inference_steps": 8,
    "guidance_scale":      1.0,
    "flow_shift":          3.0,
}

_DEFAULTS: dict = {
    "model_type":          "ltx2_22B",
    "resolution":          "1280x720",
    "video_length":        97,
    "num_inference_steps": 30,
    "guidance_scale":      3.0,
    "flow_shift":          3.0,
}

_DEFAULTS_HIGH_QUALITY: dict = {
    "model_type":          "ltx2_22B_pure_dev",
    "resolution":          "1280x720",
    "video_length":        97,
    "num_inference_steps": 50,
    "guidance_scale":      3.0,
    "flow_shift":          3.0,
}

# Named presets exposed to callers (API, UI, etc.)
VIDEO_PRESETS: dict[str, dict] = {
    "fast":         _DEFAULTS_FAST,
    "quality":      _DEFAULTS,
    "high_quality": _DEFAULTS_HIGH_QUALITY,
}

# Client-side file path fields — uploaded before submission, excluded from the JSON payload
_PATH_FIELDS: frozenset[str] = frozenset({
    "image_start", "image_end", "audio_guide", "video_guide",
    "video_mask", "image_refs", "keyframes",
})

# Convenience-only fields — used to compute API params but not sent themselves
_CONVENIENCE_FIELDS: frozenset[str] = frozenset({"duration_seconds", "fps"})

# String fields that should be omitted from the payload when empty
_STRIP_EMPTY_STRINGS = ("video_prompt_type", "image_prompt_type", "audio_prompt_type", "loras_multipliers")


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
        default=12 * 3600.0,
        description="Timeout in seconds for polling",
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
    Parameters for LTX video generation via WanGP POST /jobs/raw.

    All generation flags (video_prompt_type, image_prompt_type, audio_prompt_type)
    must be set explicitly — there are no auto-defaults.

    File path fields (image_start, image_end, audio_guide, video_guide, video_mask,
    image_refs, keyframes) accept local filesystem paths.  The client uploads each
    file and substitutes "file:<id>" references before submitting the job.

    Convenience:
        Set duration_seconds (+ optionally fps, default 24) to compute video_length
        as the nearest 8n+1 value.  video_length takes precedence when both are set.
    """

    # ── File paths (local FS, not sent to API) ────────────────────────────────

    image_start: Optional[str] = Field(
        default=None,
        description=(
            "Local path to start/first-frame image. "
            "Enable with 'S' in image_prompt_type."
        ),
    )
    image_end: Optional[str] = Field(
        default=None,
        description=(
            "Local path to end/last-frame image. "
            "Enable with 'E' in image_prompt_type."
        ),
    )
    audio_guide: Optional[str] = Field(
        default=None,
        description=(
            "Local path to audio conditioning file (WAV/MP3/FLAC). "
            "Enable with 'A' in audio_prompt_type."
        ),
    )
    video_guide: Optional[str] = Field(
        default=None,
        description=(
            "Local path to source video for V2V. "
            "Set video_prompt_type accordingly (e.g. 'DVG', 'PVG', 'VG')."
        ),
    )
    video_mask: Optional[str] = Field(
        default=None,
        description=(
            "Local path to binary mask video (white=regenerate, black=keep). "
            "Add 'A' to video_prompt_type to activate masking."
        ),
    )
    image_refs: List[str] = Field(
        default_factory=list,
        description=(
            "Local paths to reference images. "
            "Used for identity ('I') mode — set video_prompt_type='I'."
        ),
    )
    keyframes: List[List] = Field(
        default_factory=list,
        description=(
            "Keyframe entries: [[local_image_path, frame_idx_0based, strength], ...]. "
            "Mirrors the API structure — the client uploads each image and replaces the "
            "path with 'file:<id>'. Requires model_type='ltx2_22B_keyframe'."
        ),
    )

    # ── Core settings (1:1 API fields) ───────────────────────────────────────

    model_type: str = Field(default="ltx2_22B_distilled_1_1")
    prompt: str = Field(default="", description="Text prompt describing the video content")
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE_PROMPT)
    resolution: str = Field(
        default="1280x720",
        description=(
            "WxH string (e.g. '1280x720', '832x480'). "
            "Must match any provided image dimensions — LTX is sensitive to mismatches."
        ),
    )
    video_length: int = Field(
        default=97,
        description=(
            "Frame count. Must satisfy 8n+1 (9, 17, 25 … 241). "
            "Auto-computed from duration_seconds/fps when duration_seconds is set."
        ),
    )
    num_inference_steps: int = Field(default=8, ge=1, le=200)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=30.0)
    flow_shift: float = Field(default=5.0)
    seed: int = Field(default=-1, description="-1 for random")

    # ── Conditioning flags (explicit, no auto-defaults) ───────────────────────

    video_prompt_type: str = Field(
        default="",
        description=(
            "V2V / identity flags. Examples: 'DVG' (depth+video+guide), 'PVG' (pose), "
            "'OVG' (aligned-pose), 'EVG' (Canny), 'VG' (raw passthrough), "
            "'I' (identity reference). "
            "D/P/O/E/I auto-load union-control LoRA on the server. "
            "Empty string = not sent."
        ),
    )
    image_prompt_type: str = Field(
        default="",
        description=(
            "Image conditioning flags: 'S' (pin start frame), 'E' (guide end frame), "
            "'SE' (both). Empty string = not sent."
        ),
    )
    audio_prompt_type: str = Field(
        default="",
        description=(
            "Audio flags: 'A' (audio conditioning), 'A1O' (audio + ID-LoRA + force output). "
            "Empty string = not sent."
        ),
    )

    # ── Conditioning strengths ────────────────────────────────────────────────

    denoising_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="V2V Control Video Strength (0–1). Higher = output closer to guide.",
    )
    audio_scale: float = Field(
        default=1.0,
        description="Prompt Audio Strength (0–1): how strongly the audio drives video motion.",
    )
    audio_guidance_scale: float = Field(
        default=7.0,
        description="Audio CFG guidance scale (1–20): higher = more audio-faithful output.",
    )
    remove_background_images_ref: Optional[int] = Field(
        default=None,
        description=(
            "Strip background from image_refs before encoding (0=keep, 1=strip). "
            "Server default is 1 (strip). Only relevant when image_refs is set."
        ),
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────

    activated_loras: List[str] = Field(
        default_factory=list,
        description="LoRA filenames from the loras/ltx2/ directory on the server.",
    )
    loras_multipliers: str = Field(
        default="",
        description="Space-separated multipliers, one per entry in activated_loras.",
    )

    # ── Two-stage pipeline ───────────────────────────────────────────────────

    guidance_phases: Optional[int] = Field(
        default=None,
        description="1 = dev only; 2 = dev + distilled-LoRA phase.",
    )
    sample_solver: Optional[str] = Field(
        default=None,
        description="Solver type, e.g. 'euler', 'res2s', 'distilled_8_steps'.",
    )
    alt_guidance_scale: Optional[float] = Field(default=None)
    alt_scale: Optional[float] = Field(default=None)

    # ── SLG / NAG ────────────────────────────────────────────────────────────

    perturbation_switch: Optional[int] = Field(
        default=None,
        description="Skip-Layer Guidance: 0=off, 1=SLG, 2=skip self-attention.",
    )
    perturbation_layers: Optional[List[int]] = Field(
        default=None,
        description="Transformer layer indices for SLG.",
    )
    perturbation_start_perc: Optional[float] = Field(
        default=None,
        description="% of total steps at which SLG activates.",
    )
    perturbation_end_perc: Optional[float] = Field(
        default=None,
        description="% of total steps at which SLG deactivates.",
    )
    NAG_scale: Optional[float] = Field(
        default=None,
        description="Negative Attention Guidance strength (1.0 = off).",
    )
    NAG_tau: Optional[float] = Field(default=None)
    NAG_alpha: Optional[float] = Field(default=None)

    # ── Frame control ────────────────────────────────────────────────────────

    keep_frames_video_guide: Optional[str] = Field(
        default=None,
        description=(
            "Frame range to blank from guide, e.g. '17:-1' blanks first 17 frames "
            "(model generates freely from image_start before guide conditioning takes over). "
            "Use with video_guide + image_start to smooth hard cuts."
        ),
    )
    masking_strength: Optional[float] = Field(
        default=None,
        description="Mask reinjection strength per step.",
    )
    mask_expand: Optional[int] = Field(
        default=None,
        description="Pixels to expand mask boundary.",
    )

    # ── Sliding window (long video) ──────────────────────────────────────────

    sliding_window_size: Optional[int] = Field(default=None)
    sliding_window_overlap: Optional[int] = Field(default=None)
    sliding_window_color_correction_strength: Optional[float] = Field(default=None)
    sliding_window_overlap_noise: Optional[float] = Field(default=None)
    sliding_window_discard_last_frames: Optional[int] = Field(default=None)

    # ── Convenience (not sent to API) ────────────────────────────────────────

    duration_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Convenience: compute video_length = nearest 8n+1 to duration_seconds × fps. "
            "Ignored when video_length has been set explicitly (i.e. differs from default 97). "
            "When in doubt, set video_length directly."
        ),
    )
    fps: int = Field(
        default=24,
        description="Used with duration_seconds to compute video_length and actual_duration.",
    )

    # ── Validator ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _compute_video_length(self) -> "VideoGenerationParams":
        """Override video_length from duration_seconds when duration is provided."""
        if self.duration_seconds is not None:
            raw = int(self.duration_seconds * self.fps)
            n = round((raw - 1) / 8)
            self.video_length = max(8 * n + 1, 9)
        return self

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def effective_resolution(self) -> str:
        """The resolution string that will be sent to the API."""
        return self.resolution

    @property
    def actual_duration(self) -> float:
        """Approximate duration of the generated video in seconds."""
        return self.video_length / self.fps

    @property
    def frame_count(self) -> int:
        """Alias for video_length."""
        return self.video_length

    # Backward-compat read-only shims used by story_to_video and legacy callers

    @property
    def width(self) -> int:
        return int(self.resolution.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.resolution.split("x")[1])

    @property
    def steps(self) -> int:
        return self.num_inference_steps

    @property
    def effective_frames(self) -> int:
        return self.video_length

    @property
    def effective_fps(self) -> str:
        return str(self.fps)

    # ── Class method ─────────────────────────────────────────────────────────

    @classmethod
    def from_preset(
        cls,
        preset_name: str = "fast",
        prompt: str = "",
        **overrides,
    ) -> "VideoGenerationParams":
        """Create VideoGenerationParams from a named quality preset.

        Args:
            preset_name: ``'fast'``, ``'quality'``, or ``'high_quality'``.
            prompt: Text prompt.
            **overrides: Any VideoGenerationParams field to override.
        """
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
# Image dimension utility
# =============================================================================

def _get_image_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Return *(width, height)* of an image, or *None* if undetectable.

    Tries PIL first; falls back to a stdlib PNG-header parse.
    """
    # Prefer Pillow when available
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except ImportError:
        pass
    except Exception:
        return None

    # Stdlib PNG fallback (reads only the 24-byte IHDR)
    try:
        with open(path, "rb") as f:
            header = f.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", header[16:24])
            return w, h
    except Exception:
        pass

    return None


# =============================================================================
# WanGP REST Implementation
# =============================================================================

class WanGPLTXClient(LTXClientInterface):
    """
    LTX video generation backed by the WanGP REST server (POST /jobs/raw).

    Workflow:
      1. Health check (GET /health)
      2. Upload local files (POST /files/upload) → file_ids
      3. Resolve resolution against provided image dimensions
      4. Submit job (POST /jobs/raw) with explicit settings
      5. Poll (GET /jobs/{job_id}) until completion
      6. Download output (GET /files/{filename})
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
        """Upload *path* to the server and return the ``file_id``."""
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

    @staticmethod
    def _get_mode_label(params: VideoGenerationParams) -> str:
        if params.video_guide:
            vpt = params.video_prompt_type or "VG"
            if params.image_start:
                return f"v2v+image_start ({vpt})"
            return f"v2v ({vpt})"
        if params.keyframes:
            return "keyframe-interpolation"
        if "I" in params.video_prompt_type:
            return "identity-ref (I)"
        if params.image_start and params.image_end:
            return "i2v SE (start+end)"
        if params.image_start:
            return "audio-conditioned i2v" if params.audio_guide else "i2v"
        if params.image_end:
            return "i2v E (end-frame only)"
        if params.audio_guide:
            return "t2v+audio"
        return "t2v"

    def _build_settings(
        self,
        params: VideoGenerationParams,
        image_start_id: Optional[str] = None,
        image_end_id: Optional[str] = None,
        audio_guide_id: Optional[str] = None,
        video_guide_id: Optional[str] = None,
        video_mask_id: Optional[str] = None,
        image_ref_ids: Optional[List[str]] = None,
        keyframe_ids: Optional[List[str]] = None,
        resolution_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the ``/jobs/raw`` settings payload.

        Serialises *params* (excluding file path and convenience fields), injects
        uploaded file references, and applies an optional resolution override.
        """
        # Serialize all API settings, drop None values
        settings: Dict[str, Any] = params.model_dump(
            exclude=_PATH_FIELDS | _CONVENIENCE_FIELDS,
            exclude_none=True,
        )

        # Drop empty string flags — server rejects empty strings for flag fields
        for key in _STRIP_EMPTY_STRINGS:
            if not settings.get(key):
                settings.pop(key, None)

        # Drop empty LoRA list
        if not settings.get("activated_loras"):
            settings.pop("activated_loras", None)

        # Apply optional resolution override (from image dimension auto-detection)
        if resolution_override:
            settings["resolution"] = resolution_override

        # Inject uploaded file references
        if image_start_id:
            settings["image_start"] = f"file:{image_start_id}"
        if image_end_id:
            settings["image_end"] = f"file:{image_end_id}"
        if audio_guide_id:
            settings["audio_guide"] = f"file:{audio_guide_id}"
        if video_guide_id:
            settings["video_guide"] = f"file:{video_guide_id}"
        if video_mask_id:
            settings["video_mask"] = f"file:{video_mask_id}"
        if image_ref_ids:
            settings["image_refs"] = [f"file:{fid}" for fid in image_ref_ids]
        if keyframe_ids and params.keyframes:
            settings["keyframes"] = [
                [f"file:{fid}", entry[1], entry[2]]
                for fid, entry in zip(keyframe_ids, params.keyframes)
            ]

        return settings

    def _submit_job(self, settings: Dict[str, Any]) -> tuple[str, int]:
        """Submit a raw job and return *(job_id, queue_position)*."""
        url = f"{self.config.server_url.rstrip('/')}/jobs/raw"
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
        poll_interval: float = 5.0,
    ) -> List[str]:
        """Poll ``GET /jobs/{job_id}`` until done. Returns list of output file URLs."""
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
                    print(f"\r  [{bar}] {pct:5.1%}  {phase}    ", end="", flush=True)
                if progress_callback:
                    progress_callback(pct, phase)
                time.sleep(poll_interval)
                continue

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
        """Download *file_url* and save to *output_dir*. Returns local path."""
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
    ) -> tuple[List[str], str]:
        """
        Execute the full REST generation pipeline synchronously.

        Returns:
            *(local_paths, effective_resolution)* — list of downloaded file paths
            and the resolution string that was actually submitted.
        """
        base = self.config.server_url.rstrip("/")
        mode_label = self._get_mode_label(params)

        print(f"[1] Health check ({base})")
        self._check_health()

        # ── Upload files ───────────────────────────────────────────────────────
        image_start_id: Optional[str]  = None
        image_end_id: Optional[str]    = None
        audio_guide_id: Optional[str]  = None
        video_guide_id: Optional[str]  = None
        video_mask_id: Optional[str]   = None
        image_ref_ids: List[str]       = []
        keyframe_ids: List[str]        = []

        if params.video_guide:
            print(f"[2a] Uploading video guide ({mode_label})")
            video_guide_id = self._upload_file(params.video_guide)
        if params.image_start:
            print("[2b] Uploading start image")
            image_start_id = self._upload_file(params.image_start)
        if params.audio_guide:
            print("[2c] Uploading audio guide")
            audio_guide_id = self._upload_file(params.audio_guide)
        if params.image_end:
            print("[2d] Uploading end image")
            image_end_id = self._upload_file(params.image_end)
        if params.video_mask:
            print("[2e] Uploading video mask")
            video_mask_id = self._upload_file(params.video_mask)
        if params.image_refs:
            print(f"[2f] Uploading {len(params.image_refs)} reference image(s)")
            for ref_path in params.image_refs:
                image_ref_ids.append(self._upload_file(ref_path))
        if params.keyframes:
            print(f"[2g] Uploading {len(params.keyframes)} keyframe image(s)")
            for entry in params.keyframes:
                keyframe_ids.append(self._upload_file(entry[0]))

        # ── Dimension guard ────────────────────────────────────────────────────
        # Detect actual image dimensions and warn or auto-correct resolution.
        primary_image = params.image_start or (params.image_refs[0] if params.image_refs else None)
        resolution_override: Optional[str] = None
        if primary_image:
            detected = _get_image_dimensions(primary_image)
            if detected is not None:
                detected_res = f"{detected[0]}x{detected[1]}"
                if detected_res != params.resolution:
                    if params.resolution == "1280x720":
                        # Resolution is still at the default → auto-correct silently
                        resolution_override = detected_res
                        print(
                            f"  [resolution] auto-set to {detected_res} "
                            f"(matched image dimensions; was default '1280x720')"
                        )
                    else:
                        warnings.warn(
                            f"Resolution mismatch: params.resolution={params.resolution!r} "
                            f"but the primary image is {detected_res}. "
                            "LTX is sensitive to dimension mismatches — "
                            "align resolution to image dimensions to avoid artifacts.",
                            stacklevel=2,
                        )

        effective_resolution = resolution_override or params.resolution

        # ── Submit job ─────────────────────────────────────────────────────────
        print(f"[3] Submitting job  model={params.model_type}  mode={mode_label}")
        settings = self._build_settings(
            params,
            image_start_id=image_start_id,
            image_end_id=image_end_id,
            audio_guide_id=audio_guide_id,
            video_guide_id=video_guide_id,
            video_mask_id=video_mask_id,
            image_ref_ids=image_ref_ids or None,
            keyframe_ids=keyframe_ids or None,
            resolution_override=resolution_override,
        )
        job_id, _ = self._submit_job(settings)

        # ── Poll ───────────────────────────────────────────────────────────────
        print("[4] Polling job status …")
        t0 = time.time()
        filenames = self._poll_job(job_id, progress_callback)
        print(f"    finished in {time.time() - t0:.1f}s")

        if not filenames:
            raise RuntimeError("WanGP generation produced no output files")

        # ── Download ───────────────────────────────────────────────────────────
        print("[5] Downloading output file(s)")
        local_paths: List[str] = []
        for name in filenames:
            local_paths.append(self._download_output(name, output_dir))

        return local_paths, effective_resolution

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
        Generate a video via the WanGP REST server (POST /jobs/raw).

        All HTTP calls run in a thread so this coroutine stays non-blocking.

        Args:
            params: Generation parameters. Set file path fields (image_start,
                audio_guide, video_guide, …) and the corresponding flag fields
                (image_prompt_type, audio_prompt_type, video_prompt_type).
            output_dir: Local directory to save the downloaded video.
            progress_callback: Optional ``callback(fraction, message)``.

        Returns:
            :class:`VideoGenerationResult` with local video path and metadata.

        Raises:
            RuntimeError: If the server is not ready, inference fails, or no
                files were produced.
        """
        if progress_callback:
            progress_callback(0.0, "Starting WanGP generation…")

        output_files, effective_resolution = await asyncio.to_thread(
            self._run_generation_sync, params, output_dir, progress_callback
        )

        if not output_files:
            raise RuntimeError("WanGP generation produced no output files")

        if progress_callback:
            progress_callback(1.0, "Done!")

        w_str, h_str = effective_resolution.split("x")

        return VideoGenerationResult(
            video_path=output_files[0],
            audio_path=None,
            duration_seconds=params.actual_duration,
            width=int(w_str),
            height=int(h_str),
            fps=params.fps,
            prompt_id=Path(output_files[0]).stem,
        )


# =============================================================================
# Convenience async function
# =============================================================================

async def generate_video(
    prompt: str,
    image_start: Optional[str] = None,
    output_dir: str = "./output",
    server_url: str = "http://localhost:8082",
    audio_guide: Optional[str] = None,
    video_guide: Optional[str] = None,
    image_end: Optional[str] = None,
    image_refs: Optional[List[str]] = None,
    image_prompt_type: str = "",
    video_prompt_type: str = "",
    audio_prompt_type: str = "",
    **kwargs,
) -> VideoGenerationResult:
    """
    Generate a video with a single async call.

    Args:
        prompt: Text prompt.
        image_start: Local path to start-frame image (add ``'S'`` to image_prompt_type).
        output_dir: Directory to save the output video.
        server_url: URL of the remote WanGP REST server.
        audio_guide: Local path to conditioning audio (add ``'A'`` to audio_prompt_type).
        video_guide: Local path to source video for V2V (set video_prompt_type accordingly).
        image_end: Local path to end-frame image (add ``'E'`` to image_prompt_type).
        image_refs: Local paths to identity-reference images (set video_prompt_type='I').
        image_prompt_type: Image conditioning flags (e.g. ``'S'``, ``'SE'``).
        video_prompt_type: V2V / identity flags (e.g. ``'DVG'``, ``'I'``).
        audio_prompt_type: Audio flags (e.g. ``'A'``, ``'A1O'``).
        **kwargs: Additional :class:`VideoGenerationParams` fields.

    Returns:
        :class:`VideoGenerationResult`
    """
    config = LTXVideoConfig(server_url=server_url)
    params = VideoGenerationParams(
        prompt=prompt,
        image_start=image_start,
        audio_guide=audio_guide,
        video_guide=video_guide,
        image_end=image_end,
        image_refs=image_refs or [],
        image_prompt_type=image_prompt_type,
        video_prompt_type=video_prompt_type,
        audio_prompt_type=audio_prompt_type,
        **kwargs,
    )
    async with WanGPLTXClient(config) as client:
        return await client.generate_video(params, output_dir=output_dir)
