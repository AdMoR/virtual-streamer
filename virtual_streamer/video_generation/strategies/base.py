"""
Video conditioning strategy interface.

A "conditioning strategy" decides *how* a scene's inputs (a conditioning
image, an audio guide, a reference sheet, ...) are turned into the
``VideoGenerationParams`` LTX actually consumes — i.e. which
image_prompt_type / video_prompt_type / audio_prompt_type / LoRA combination
to use. Adding a new LTX conditioning mode (e.g. keyframe interpolation, an
ingredient-lora reference sheet, ...) means adding one new strategy class
here, not branching inside the pipeline loop.

The high-level pipeline (template -> scenes -> video chunks) never encodes
this choice itself; it asks the factory (`strategies/factory.py`) which
strategy applies to a given scene and delegates to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from virtual_streamer.video_generation.ltx_client import VideoGenerationParams
from virtual_streamer.video_generation.scene_input import SceneInput


class ConditioningContext(BaseModel):
    """Everything a strategy needs to build one segment's generation params."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_input: SceneInput
    video_params: VideoGenerationParams
    output_dir: str
    image_path: str | None = None
    audio_path: str | None = None


class VideoConditioningStrategy(ABC):
    """One LTX conditioning mode (talking-head, plain i2v/t2v, reference sheet, ...)."""

    #: Short identifier used in logs.
    name: str = "unnamed"

    @abstractmethod
    def applies_to(self, ctx: ConditioningContext) -> bool:
        """Whether this strategy should handle *ctx*."""

    @abstractmethod
    async def build_params(self, ctx: ConditioningContext) -> VideoGenerationParams:
        """Build the ``VideoGenerationParams`` for one segment."""
