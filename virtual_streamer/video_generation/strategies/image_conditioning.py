"""Plain image-to-video / text-to-video conditioning strategy.

Default fallback: a single optional start-frame image (``image_prompt_type="S"``),
or pure text-to-video when no image is available.
"""

from __future__ import annotations

import logging

from virtual_streamer.video_generation.ltx_client import VideoGenerationParams
from virtual_streamer.video_generation.ltx_prompt_builder import build_negative_prompt
from virtual_streamer.video_generation.strategies.base import (
    ConditioningContext,
    VideoConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.frame_utils import frames_from_duration

logger = logging.getLogger(__name__)


class ImageConditioningStrategy(VideoConditioningStrategy):
    """Single start-frame i2v, or t2v when no conditioning image is available.

    Applies to everything the more specific strategies don't claim — always
    keep this one last in the factory's candidate list.
    """

    name = "i2v/t2v"

    def applies_to(self, ctx: ConditioningContext) -> bool:
        return True

    async def build_params(self, ctx: ConditioningContext) -> VideoGenerationParams:
        scene_input = ctx.scene_input
        video_params = ctx.video_params
        frames = frames_from_duration(video_params.duration_seconds or 5.0, video_params.fps)

        logger.info(
            f"[scene {scene_input.scene_index}] "
            f"{'i2v' if ctx.image_path else 't2v'}  frames={frames}"
        )

        return VideoGenerationParams(
            prompt=scene_input.ltx_prompt,
            negative_prompt=build_negative_prompt(),
            resolution=video_params.resolution,
            video_length=frames,
            fps=video_params.fps,
            num_inference_steps=video_params.num_inference_steps,
            guidance_scale=video_params.guidance_scale,
            seed=video_params.seed,
            image_start=ctx.image_path,
            image_prompt_type="S" if ctx.image_path else "",
        )
