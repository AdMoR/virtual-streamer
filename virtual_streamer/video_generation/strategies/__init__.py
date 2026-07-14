"""LTX video conditioning strategies.

Each strategy encapsulates one way of turning a scene's optional
conditioning inputs (image, audio, reference sheet, ...) into the LTX
`VideoGenerationParams` payload. `select_strategy()` picks the right one
per scene so the pipeline (template -> scenes -> video chunks) stays
agnostic to conditioning-mode details.
"""

from virtual_streamer.video_generation.strategies.base import (
    ConditioningContext,
    VideoConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.factory import select_strategy
from virtual_streamer.video_generation.strategies.image_conditioning import (
    ImageConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.reference_sheet import ReferenceSheetStrategy
from virtual_streamer.video_generation.strategies.talking_head import TalkingHeadStrategy

__all__ = [
    "ConditioningContext",
    "VideoConditioningStrategy",
    "select_strategy",
    "TalkingHeadStrategy",
    "ImageConditioningStrategy",
    "ReferenceSheetStrategy",
]
