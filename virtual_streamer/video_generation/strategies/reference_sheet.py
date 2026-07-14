"""Reference-sheet (ingredient IC-LoRA) conditioning strategy.

Identity/prop/location consistency is carried into the video via a single
"reference sheet" image: a grid of labelled cells (setting, character,
props, logo, ...) each described in prose. LTX consumes the sheet as a
video-to-video guide — the still frozen into a static video looped to the
output's length and frame rate — with the reference-sheet description
prepended to the target prompt, exactly as the model card documents:
no extra reference downscaling or color/space transform is applied.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

from virtual_streamer.video_generation.ltx_client import VideoGenerationParams
from virtual_streamer.video_generation.ltx_prompt_builder import build_negative_prompt
from virtual_streamer.video_generation.strategies.base import (
    ConditioningContext,
    VideoConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.frame_utils import frames_from_duration

logger = logging.getLogger(__name__)

#: IC-LoRA filename on the WanGP server for reference-sheet (ingredient) conditioning.
REFERENCE_SHEET_LORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
REFERENCE_SHEET_LORA_MULTIPLIER = "1.0"

#: The IC-LoRA is trained on the full LTX-2.3 base model, not the distilled variant.
REFERENCE_SHEET_MODEL_TYPE = "ltx2_22B"


def _freeze_image_as_video(
    image_path: str, output_dir: str, num_frames: int, fps: int, resolution: str
) -> str:
    """Loop a still image into a silent video of *num_frames* frames at *fps*.

    LTX consumes the reference sheet as a video_guide, not a plain image ref —
    the model was trained on the sheet frozen into a static clip. The sheet is
    scaled to *resolution* ("WxH"): the IC-LoRA uses a reference downscale
    factor of 1, so the guide must match the output resolution exactly.
    """
    width, height = (int(v) for v in resolution.lower().split("x"))
    video_path = os.path.join(output_dir, "reference_sheet.mp4")
    duration = num_frames / fps
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", f"{duration:.3f}",
        "-r", str(fps),
        "-vf", f"scale={width}:{height},setsar=1",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(video_path):
        raise RuntimeError(f"Failed to freeze reference sheet into a video: {proc.stderr}")
    return video_path


class ReferenceSheetStrategy(VideoConditioningStrategy):
    """Ingredient-LoRA generation guided by a labelled reference-sheet image."""

    name = "reference-sheet ingredient-lora"

    def applies_to(self, ctx: ConditioningContext) -> bool:
        return bool(ctx.scene_input.reference_sheet_path)

    async def build_params(self, ctx: ConditioningContext) -> VideoGenerationParams:
        scene_input = ctx.scene_input
        video_params = ctx.video_params
        frames = frames_from_duration(video_params.duration_seconds or 5.0, video_params.fps)

        # ffmpeg runs in a thread so the event loop can keep driving the other
        # scenes' generations concurrently.
        reference_video = await asyncio.to_thread(
            _freeze_image_as_video,
            scene_input.reference_sheet_path,
            ctx.output_dir,
            frames,
            video_params.fps,
            video_params.resolution,
        )

        prompt = scene_input.ltx_prompt
        if scene_input.reference_sheet_description:
            prompt = f"{scene_input.reference_sheet_description}\n\n### Target Description\n{prompt}"

        logger.info(f"[scene {scene_input.scene_index}] reference-sheet ingredient-lora  frames={frames}")

        return VideoGenerationParams(
            prompt=prompt,
            negative_prompt=build_negative_prompt(),
            resolution=video_params.resolution,
            video_length=frames,
            fps=video_params.fps,
            num_inference_steps=video_params.num_inference_steps,
            guidance_scale=video_params.guidance_scale,
            seed=video_params.seed,
            model_type=REFERENCE_SHEET_MODEL_TYPE,
            video_guide=reference_video,
            video_prompt_type="VG",
            activated_loras=[REFERENCE_SHEET_LORA],
            loras_multipliers=REFERENCE_SHEET_LORA_MULTIPLIER,
        )
