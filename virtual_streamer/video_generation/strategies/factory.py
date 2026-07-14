"""Strategy selection for one scene's conditioning inputs.

Order matters: strategies are tried in sequence and the first one whose
`applies_to` returns True wins. `ImageConditioningStrategy` applies to
everything, so it must stay last as the fallback.
"""

from __future__ import annotations

import logging
from typing import List

from virtual_streamer.video_generation.strategies.base import (
    ConditioningContext,
    VideoConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.image_conditioning import (
    ImageConditioningStrategy,
)
from virtual_streamer.video_generation.strategies.reference_sheet import ReferenceSheetStrategy
from virtual_streamer.video_generation.strategies.talking_head import TalkingHeadStrategy

logger = logging.getLogger(__name__)

_STRATEGIES: List[VideoConditioningStrategy] = [
    TalkingHeadStrategy(),
    ReferenceSheetStrategy(),
    ImageConditioningStrategy(),
]


def select_strategy(ctx: ConditioningContext) -> VideoConditioningStrategy:
    for strategy in _STRATEGIES:
        if strategy.applies_to(ctx):
            _warn_on_shadowed_strategies(ctx, strategy)
            return strategy
    raise RuntimeError("No conditioning strategy applies — ImageConditioningStrategy should always match")


def _warn_on_shadowed_strategies(
    ctx: ConditioningContext, selected: VideoConditioningStrategy
) -> None:
    """Log when a lower-priority strategy also matched but was shadowed.

    A scene carrying e.g. both TTS audio (talking-head) and a reference sheet
    silently uses only the higher-priority mode — surface that in the logs so
    a dropped conditioning input is visible.
    """
    for strategy in _STRATEGIES:
        if strategy is selected or isinstance(strategy, ImageConditioningStrategy):
            continue
        if strategy.applies_to(ctx):
            logger.warning(
                f"[scene {ctx.scene_input.scene_index}] strategy {strategy.name!r} also "
                f"matched but is shadowed by {selected.name!r} — its conditioning inputs are ignored"
            )
