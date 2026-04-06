"""
LTX Video Client (WanGP Gradio API)

Client for generating videos via a remote WanGP server using the LTX model.
Supports image-to-video with optional audio conditioning.

The WanGP pipeline requires five sequential API calls:
  1. /change_model               — select the LTX model
  2. /save_inputs                — push all generation settings into server state
  3. /process_prompt_and_add_tasks — enqueue the task
  4. /process_tasks               — run inference (streaming, drain until done)
  5. /finalize_generation + /refresh_gallery — collect output file(s)

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
"""

import asyncio
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import requests
from gradio_client import Client, handle_file
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Defaults
# =============================================================================

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
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
}


# =============================================================================
# Configuration
# =============================================================================

class LTXVideoConfig(BaseModel):
    """Connection settings for the remote WanGP instance."""

    server_url: str = Field(
        default="http://localhost:7860",
        description="URL of the running WanGP server",
    )
    timeout: float = Field(
        default=600.0,
        description="HTTP timeout in seconds (video generation can be slow)",
    )
    save_inputs_api: str = Field(
        default="/save_inputs",
        description=(
            "Gradio API name for save_inputs. Run --list-api on the server "
            "to confirm the exact name (may be /save_inputs_1 etc. on some versions)."
        ),
    )



# =============================================================================
# User-facing Parameters
# =============================================================================

class VideoGenerationParams(BaseModel):
    """
    Parameters for LTX image-to-video generation via WanGP.

    Primary fields (WanGP / i2v):
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
        # cfg_scale → guidance_scale when guidance_scale is still at default
        if self.cfg_scale != _DEFAULTS["guidance_scale"]:
            self.guidance_scale = self.cfg_scale
        return self

    # --- Computed properties ---

    @property
    def effective_resolution(self) -> str:
        """Resolution string to send to WanGP."""
        return self.resolution if self.resolution else f"{self.width}x{self.height}"

    @property
    def effective_frames(self) -> int:
        """Frame count satisfying 8n+1 constraint."""
        if self.frames > 0:
            return self.frames
        raw = int(self.duration_seconds * self.fps)
        n = round((raw - 1) / 8)
        return max(8 * n + 1, 9)

    @property
    def effective_fps(self) -> str:
        """FPS string to send to WanGP."""
        return self.force_fps if self.force_fps else str(self.fps)

    @property
    def actual_duration(self) -> float:
        fps_val = int(self.effective_fps) if self.effective_fps.isdigit() else self.fps
        return self.effective_frames / fps_val

    # Backward-compat alias used in story_to_video.py
    @property
    def frame_count(self) -> int:
        return self.effective_frames


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
    """
    Abstract interface for LTX video generation clients.

    Implementors must provide :meth:`generate_video` and :meth:`close`.
    The class also acts as an async context manager via ``async with``.
    """

    @abstractmethod
    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoGenerationResult:
        """
        Generate a video from *params* and return the result.

        Args:
            params: Generation parameters (prompt, image, audio, model settings…)
            output_dir: Local directory where the output video will be saved.
            progress_callback: Optional ``callback(fraction: float, message: str)``.

        Returns:
            :class:`VideoGenerationResult` with the local video path and metadata.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (connections, threads, …)."""
        ...

    async def __aenter__(self) -> "LTXClientInterface":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


# =============================================================================
# WanGP Implementation
# =============================================================================

class WanGPLTXClient(LTXClientInterface):
    """
    LTX video generation backed by a remote WanGP Gradio server.

    Supports:
    - Image-to-video (i2v): start image + text prompt
    - Audio-conditioned i2v: start image + audio + text prompt
      (requires a distilled LTX model)

    The Gradio client is created lazily on first use and reused across calls.
    """

    def __init__(self, config: Optional[LTXVideoConfig] = None) -> None:
        self.config = config or LTXVideoConfig()
        self._gradio_client: Optional[Client] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_gradio_client(self) -> Client:
        if self._gradio_client is None:
            self._gradio_client = Client(self.config.server_url)
        return self._gradio_client

    def _build_save_inputs_args(
        self,
        params: VideoGenerationParams,
        image_ref,
        audio_ref,
    ) -> list:
        """
        Build the ordered argument list for the WanGP ``save_inputs`` Gradio call.

        The Gradio server expects every parameter positionally in the order below.
        When *audio_ref* is not None, audio-conditioning fields are activated:
          - audio_prompt_type = "A"
          - audio_guide      = audio_ref
          - audio_guidance_scale / audio_scale use params values

        All other audio fields remain at neutral defaults.
        """
        audio_prompt_type = "A" if audio_ref is not None else ""
        audio_guidance    = params.audio_guidance if audio_ref is not None else 1.0
        audio_scale       = params.audio_scale    if audio_ref is not None else 1.0

        return [
            # target_state  (gr.Text visible=False, value="state") — MUST be first
            "state",
            # image_mask_guide
            None,
            # lset_name
            None,
            # image_mode  (0 = standard)
            0,
            # prompt
            params.prompt,
            # alt_prompt
            "",
            # negative_prompt
            params.negative_prompt,
            # resolution
            params.effective_resolution,
            # video_length  (frame count)
            params.effective_frames,
            # duration_seconds  (0 = use video_length)
            0,
            # pause_seconds
            0,
            # batch_size
            1,
            # seed
            params.seed,
            # force_fps  (Dropdown; string choices: '24', 'auto', …)
            params.effective_fps,
            # num_inference_steps
            params.steps,
            # guidance_scale
            params.guidance_scale,
            # guidance2_scale
            1.0,
            # guidance3_scale
            1.0,
            # switch_threshold
            0.5,
            # switch_threshold2
            0.5,
            # guidance_phases  (1/2/3)
            1,
            # model_switch_phase
            1,
            # alt_guidance_scale
            1.0,
            # alt_scale
            1.0,
            # audio_guidance_scale  ← Audio CFG guidance (active when audio_ref set)
            audio_guidance,
            # audio_scale  ← Prompt Audio Strength (active when audio_ref set)
            audio_scale,
            # flow_shift
            params.flow_shift,
            # sample_solver  (LTX only exposes [""])
            "",
            # embedded_guidance_scale
            1.0,
            # repeat_generation
            1,
            # multi_prompts_gen_type
            0,
            # multi_images_gen_type
            0,
            # skip_steps_cache_type
            "",
            # skip_steps_multiplier
            1.5,
            # skip_steps_start_step_perc
            0.0,
            # loras_choices
            [],
            # loras_multipliers
            "",
            # image_prompt_type  — "S" = start-image (i2v) | "" = text-to-video
            "S" if image_ref is not None else "",
            # image_start  (gr.Gallery — each item: {"image": FileData}) | None for t2v
            [{"image": image_ref}] if image_ref is not None else None,
            # image_end
            None,
            # model_mode
            None,
            # video_source
            None,
            # keep_frames_video_source
            False,
            # input_video_strength
            0.85,
            # video_guide_outpainting
            None,
            # video_prompt_type
            "",
            # image_refs
            None,
            # frames_positions
            "",
            # video_guide
            None,
            # image_guide
            None,
            # keep_frames_video_guide
            False,
            # denoising_strength
            0.85,
            # masking_strength
            1.0,
            # video_mask
            None,
            # image_mask
            None,
            # control_net_weight
            1.0,
            # control_net_weight2
            1.0,
            # control_net_weight_alt
            1.0,
            # motion_amplitude
            1.0,
            # mask_expand
            0,
            # audio_guide  ← conditioning audio (None when no audio)
            audio_ref,
            # audio_guide2  (second speaker — unused)
            None,
            # custom_guide
            None,
            # audio_source  (distinct from audio conditioning — unused)
            None,
            # audio_prompt_type  ← "A" = condition on audio_guide
            audio_prompt_type,
            # speakers_locations  (Multitalk multi-speaker bbox — unused)
            "",
            # sliding_window_size
            81,
            # sliding_window_overlap
            8,
            # sliding_window_color_correction_strength
            0.5,
            # sliding_window_overlap_noise
            0,
            # sliding_window_discard_last_frames
            0,
            # image_refs_relative_size
            False,
            # remove_background_images_ref
            False,
            # temporal_upsampling
            "",
            # spatial_upsampling
            "",
            # film_grain_intensity
            0,
            # film_grain_saturation
            0.5,
            # MMAudio_setting
            0,
            # MMAudio_prompt
            "",
            # MMAudio_neg_prompt
            "",
            # RIFLEx_setting
            0,
            # NAG_scale
            0.0,
            # NAG_tau
            2.0,
            # NAG_alpha
            0.5,
            # perturbation_switch
            0,
            # perturbation_layers  (multiselect Dropdown)
            [],
            # perturbation_start_perc
            0.0,
            # perturbation_end_perc
            1.0,
            # apg_switch
            0,
            # cfg_star_switch
            0,
            # cfg_zero_step
            0,
            # prompt_enhancer
            "",
            # min_frames_if_references
            1,
            # override_profile
            -1,
            # override_attention
            "",
            # temperature
            1.0,
            # custom_setting_1 … custom_setting_5
            "", "", "", "", "",
            # top_p
            1.0,
            # top_k
            0,
            # self_refiner_setting
            0,
            # self_refiner_plan
            "",
            # self_refiner_f_uncertainty
            0.5,
            # self_refiner_certain_percentage
            0.5,
            # output_filename  (empty = auto)
            "",
            # mode  (gr.Text visible=False; empty = normal generation)
            "",
        ]

    def _run_generation_sync(
        self,
        params: VideoGenerationParams,
        output_dir: str,
    ) -> List[str]:
        """
        Execute the full 5-step WanGP pipeline synchronously.

        Returns a list of local paths to downloaded output files.
        """

        client = self._get_gradio_client()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_ref = handle_file(params.image_path) if params.image_path else None
        audio_ref = handle_file(params.audio_path) if params.audio_path else None

        if image_ref is None:
            mode_label = "t2v"
        elif audio_ref is not None:
            mode_label = "audio-conditioned i2v"
        else:
            mode_label = "i2v"
        print(f"[1/5] Selecting model: {params.model_type}  ({mode_label})")
        client.predict(params.model_type, api_name="/change_model")

        print("[2/5] Saving generation settings into server state...")
        save_args = self._build_save_inputs_args(params, image_ref, audio_ref)
        client.predict(*save_args, api_name=self.config.save_inputs_api)

        print("[3/5] Queuing task...")
        client.predict(
            0,                  # current_gallery_tab (0 = video)
            params.model_type,  # model_choice — must match state["model_type"]
            api_name="/process_prompt_and_add_tasks",
        )

        print("[4/5] Running inference (this may take a while)...")
        t0 = time.time()
        job = client.submit(api_name="/process_tasks")
        try:
            for _ in job:
                elapsed = time.time() - t0
                print(f"\r  elapsed: {elapsed:.0f}s", end="", flush=True)
        except Exception as exc:
            raise RuntimeError(f"process_tasks error: {exc}") from exc
        print(f"\n  inference finished in {time.time() - t0:.1f}s")

        print("[5/5] Collecting output files...")
        client.predict(api_name="/finalize_generation")
        result = client.predict(api_name="/refresh_gallery")

        server_root = client.src.rstrip("/")
        gallery_dict = _find_gallery_dict(result)
        output_files: List[str] = []

        if gallery_dict and gallery_dict.get("value"):
            gallery_items = gallery_dict["value"]
            idx = gallery_dict.get("selected_index") or 0
            if not (0 <= idx < len(gallery_items)):
                idx = len(gallery_items) - 1

            target_item = gallery_items[idx]
            temp_path = _extract_path(target_item)
            print(f"  gallery selected_index={idx}, temp path: {temp_path}")

            if temp_path:
                temp_basename = Path(temp_path).name
                actual_filename = _temp_to_output_filename(temp_basename)
                output_rel = f"outputs/{actual_filename}"
                dst = out_path / actual_filename
                _download_gradio_file(server_root, output_rel, dst)
                output_files.append(str(dst))
                print(f"  saved: {dst}")

        if not output_files:
            print("  No output files found.")
            print("  Raw refresh_gallery result:", result)

        return output_files

    # ------------------------------------------------------------------
    # LTXClientInterface implementation
    # ------------------------------------------------------------------

    async def close(self) -> None:
        self._gradio_client = None

    async def generate_video(
        self,
        params: VideoGenerationParams,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> VideoGenerationResult:
        """
        Generate a video via WanGP.

        The blocking Gradio calls are run in a thread so this coroutine
        remains non-blocking in an async context.

        Args:
            params: Generation parameters.
                - Text-to-video: only ``prompt`` required; leave ``image_path`` unset.
                - Image-to-video: set ``image_path`` to the start image.
                - Audio-conditioned i2v: set both ``image_path`` and ``audio_path``
                  (distilled models only).
            output_dir: Local directory to save the downloaded video.
            progress_callback: Optional ``callback(fraction, message)``.

        Returns:
            :class:`VideoGenerationResult` with local video path and metadata.

        Raises:
            RuntimeError: If inference fails or no output files are produced.
        """
        if progress_callback:
            progress_callback(0.0, "Starting WanGP generation…")

        output_files: List[str] = await asyncio.to_thread(
            self._run_generation_sync, params, output_dir
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
# File helpers (ported from reference scripts)
# =============================================================================

def _download_gradio_file(server_root: str, output_rel: str, dst: Path) -> None:
    """
    Download a file from the WanGP Gradio server via its outputs/ path.

    Files under /tmp/gradio/ return 403; only outputs/ paths are accessible.
    """
    url = f"{server_root}/gradio_api/file={output_rel}"
    print(f"  downloading {url} …")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)


def _temp_to_output_filename(temp_basename: str) -> str:
    """
    Reverse Gradio's parenthesis-stripping in temp filenames.

    WanGP names outputs 0.mp4, 0(2).mp4, 0(3).mp4 …
    Gradio strips parens when copying to /tmp: 0(18).mp4 → 018.mp4
    This reverses that: 018.mp4 → 0(18).mp4
    """
    m = re.match(r"^0(\d+)(\.[^.]+)$", temp_basename)
    if m:
        digits, ext = m.groups()
        return f"0({int(digits)}){ext}"
    return temp_basename  # e.g. "0.mp4" — first generation


def _find_gallery_dict(result) -> Optional[dict]:
    """Return the first Gradio Gallery update dict in a result tuple."""
    if not isinstance(result, (list, tuple)):
        return None
    for item in result:
        if isinstance(item, dict) and "selected_index" in item and "value" in item:
            return item
    return None


def _extract_path(item) -> str:
    """Extract a remote file path from one gallery item regardless of format."""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)):
        return str(item[0]) if item[0] else ""
    if isinstance(item, dict):
        for key in ("image", "video"):
            if key in item:
                inner = item[key]
                if isinstance(inner, dict):
                    return inner.get("path") or inner.get("url") or ""
                if isinstance(inner, str):
                    return inner
        return item.get("path") or item.get("url") or ""
    return ""


# =============================================================================
# Convenience async function
# =============================================================================

async def generate_video(
    prompt: str,
    image_path: str,
    output_dir: str = "./output",
    server_url: str = "http://localhost:7860",
    audio_path: Optional[str] = None,
    **kwargs,
) -> VideoGenerationResult:
    """
    Generate a video with a single async call.

    Args:
        prompt: Text prompt describing the video content.
        image_path: Local path to the start image.
        output_dir: Directory to save the output video.
        server_url: URL of the remote WanGP instance.
        audio_path: Optional conditioning audio file. Enables audio-driven i2v.
        **kwargs: Additional :class:`VideoGenerationParams` fields.

    Returns:
        :class:`VideoGenerationResult`
    """
    config = LTXVideoConfig(server_url=server_url)
    params = VideoGenerationParams(
        prompt=prompt,
        image_path=image_path,
        audio_path=audio_path,
        **kwargs,
    )
    async with WanGPLTXClient(config) as client:
        return await client.generate_video(params, output_dir=output_dir)
