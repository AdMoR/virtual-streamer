"""
Aggregators for VideoMatcher results.

BestMatchAggregator selects the best video match from parallel workers
based on rating priority (CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL) and grade.
"""

import logging
from typing import List, Optional

from virtual_streamer.lib.agents import AggregatorAgent
from virtual_streamer.agents.video_matcher.schema import (
    ContextualRating,
    VideoMatchResult,
)

logger = logging.getLogger(__name__)


class BestMatchAggregator(AggregatorAgent[VideoMatchResult]):
    """
    Aggregator that selects the best video match from parallel workers.
    
    Selection priority:
    1. CONTEXTUAL rating (best) - pick highest grade among these
    2. NEUTRAL rating - pick highest grade among these
    3. NOT_CONTEXTUAL rating (fallback) - pick highest grade
    
    Example:
        # After MapperAgent runs:
        output_keys = mapper.get_output_keys()
        
        aggregator = BestMatchAggregator(
            input_keys=output_keys,
            output_key="best_video_match",
        )
        async for event in aggregator.run_async(ctx):
            yield event
        
        # Result stored at "best_video_match"
        best = ctx.session.state.get("best_video_match")
    """
    
    def __init__(
        self,
        input_keys: List[str],
        output_key: str = "best_video_match",
        name: str = "best_match_aggregator",
    ):
        """
        Initialize the aggregator.
        
        Args:
            input_keys: List of state keys to read VideoMatchResult from.
                       These are typically mapper.get_output_keys().
            output_key: Key to store the best match result in state.
            name: Name for this agent.
        """
        super().__init__(
            name=name,
            input_keys=input_keys,
            input_schema=VideoMatchResult,
            output_key=output_key,
        )
    
    async def aggregation_fn(
        self, results: List[VideoMatchResult]
    ) -> Optional[VideoMatchResult]:
        """
        Select the best video match from results.
        
        Priority:
        1. CONTEXTUAL with highest grade
        2. NEUTRAL with highest grade
        3. NOT_CONTEXTUAL with highest grade
        
        Args:
            results: List of VideoMatchResult from parallel workers.
                    Guaranteed to be non-empty.
        
        Returns:
            The best matching VideoMatchResult.
        """
        if not results:
            logger.warning("No results to aggregate")
            return None
        
        # Try each rating level in priority order
        rating_priority = [
            ContextualRating.CONTEXTUAL,
            ContextualRating.NEUTRAL,
            ContextualRating.NOT_CONTEXTUAL,
        ]
        
        for target_rating in rating_priority:
            matches = [r for r in results if r.rating == target_rating]
            if matches:
                best = max(matches, key=lambda x: x.grade)
                logger.info(
                    f"Selected best match: rating={best.rating}, "
                    f"grade={best.grade}, video={best.video_path}"
                )
                return best
        
        # Fallback: just return the one with highest grade
        best = max(results, key=lambda x: x.grade)
        logger.warning(f"No standard rating found, using highest grade: {best}")
        return best
