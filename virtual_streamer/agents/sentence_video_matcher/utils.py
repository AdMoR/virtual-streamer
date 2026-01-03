from virtual_streamer.agents.story_generator.schema import DialogLine, DialogLines
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherOutput,
    DialogLineMatch,
)
import logging
from typing import Dict, List

from virtual_streamer.agents.video_matcher import (
    VideoMatchResult,
    ContextualRating,
)

logger = logging.getLogger(__name__)


def _select_best(
        results: List[VideoMatchResult]
) -> VideoMatchResult | None:
    """
    Select the best match from a list of results.

    Priority:
    1. CONTEXTUAL with highest grade
    2. NEUTRAL with highest grade
    3. NOT_CONTEXTUAL with highest grade
    """
    if not results:
        return None

    rating_priority = [
        ContextualRating.CONTEXTUAL,
        ContextualRating.NEUTRAL,
        ContextualRating.NOT_CONTEXTUAL,
    ]

    for target_rating in rating_priority:
        matches = [r for r in results if r.rating == target_rating]
        if matches:
            return max(matches, key=lambda x: x.grade)

    # Fallback to highest grade overall
    return max(results, key=lambda x: x.grade)