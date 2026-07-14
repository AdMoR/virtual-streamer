"""Talking-head (A1O) conditioning strategy.

Audio-conditioned generation: a start-frame image plus an audio guide drive
an ID-LoRA that lip-syncs the character to the spoken line.
"""

from __future__ import annotations

import logging
import os

from virtual_streamer.video_generation.ltx_client import VideoGenerationParams
from virtual_streamer.video_generation.ltx_prompt_builder import (
    build_negative_prompt,
    build_talking_head_prompt,
)
from virtual_streamer.video_generation.strategies.base import (
    ConditioningContext,
    VideoConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.frame_utils import video_length_from_spoken_line

logger = logging.getLogger(__name__)

#: ID-LoRA filename on the WanGP server used for audio-conditioned talking-head generation.
TALKING_HEAD_LORA = "id-lora-celebvhq-ltx2.3.safetensors"
TALKING_HEAD_LORA_MULTIPLIER = "1.0"

#: Distilled-model settings shared by all talking-head segments (fast, default).
TALKING_HEAD_PARAMS: dict = {
    "model_type":          "ltx2_22B_distilled_1_1",
    "num_inference_steps": 8,
    "guidance_scale":      1.0,
    "flow_shift":          5.0,
    "guidance_phases":     2,
    "sample_solver":       "distilled_8_steps",
    "audio_scale":         1.0,
    "audio_guidance_scale": 5.0,
}

#: Non-distilled ("quality") talking-head settings. Uses the full ltx2_22B base
#: model for the generation pass instead of the distilled variant: higher
#: fidelity and motion, but ~4x slower and a much larger model load (watch GPU /
#: unified memory). Experimental — the A1O ID-LoRA path is primarily validated on
#: the distilled model, so validate a single scene before a full run.
TALKING_HEAD_PARAMS_QUALITY: dict = {
    "model_type":          "ltx2_22B",
    "num_inference_steps": 30,
    "guidance_scale":      3.0,
    "flow_shift":          5.0,
    "guidance_phases":     1,
    # sample_solver left unset -> server default for the non-distilled model
    "audio_scale":         1.0,
    "audio_guidance_scale": 5.0,
}


def _talking_head_params() -> dict:
    """Talking-head generation params, distilled by default.

    Set ``TALKING_HEAD_MODEL=quality`` (or ``non_distilled``) in the API
    environment to run the first (generation) pass on the full, non-distilled
    ltx2_22B model instead of the distilled one.
    """
    mode = os.environ.get("TALKING_HEAD_MODEL", "distilled").strip().lower()
    if mode in ("quality", "non_distilled", "nondistilled", "full", "hq"):
        logger.info("[talking-head] using NON-DISTILLED (quality) model params")
        return TALKING_HEAD_PARAMS_QUALITY
    return TALKING_HEAD_PARAMS


class TalkingHeadStrategy(VideoConditioningStrategy):
    """Audio-conditioned (A1O) talking-head generation."""

    name = "talking-head A1O"

    def applies_to(self, ctx: ConditioningContext) -> bool:
        return bool(ctx.audio_path and os.path.exists(ctx.audio_path))

    async def build_params(self, ctx: ConditioningContext) -> VideoGenerationParams:
        scene_input = ctx.scene_input
        video_params = ctx.video_params

        if scene_input.scene_visual_description:
            try:
                from virtual_streamer.image_generation.models import FluxPrompt
                flux_prompt = FluxPrompt.model_validate(scene_input.scene_visual_description)
                visual = flux_prompt.to_prompt()
            except Exception:
                visual = scene_input.ltx_prompt
        else:
            visual = scene_input.ltx_prompt

        prompt = build_talking_head_prompt(
            visual_description=visual,
            spoken_line=scene_input.spoken_line or "",
        )
        video_length = video_length_from_spoken_line(scene_input.spoken_line, video_params.fps)

        logger.info(
            f"[scene {scene_input.scene_index}] talking-head  video_length={video_length}  "
            f"words={len((scene_input.spoken_line or '').split())}"
        )

        return VideoGenerationParams(
            prompt=prompt,
            negative_prompt=build_negative_prompt(),
            # Resolution left at default — client auto-corrects to match the start image
            resolution=video_params.resolution,
            video_length=video_length,
            fps=video_params.fps,
            seed=video_params.seed,
            # Talking-head conditioning
            image_start=ctx.image_path,
            image_prompt_type="S" if ctx.image_path else "",
            audio_guide=ctx.audio_path,
            audio_prompt_type="A1O",
            # Model + LoRA settings for the A1O pipeline (distilled by default;
            # TALKING_HEAD_MODEL=quality switches to the non-distilled base)
            **_talking_head_params(),
            activated_loras=[TALKING_HEAD_LORA],
            loras_multipliers=TALKING_HEAD_LORA_MULTIPLIER,
        )
