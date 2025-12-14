"""
Aggregators for VideoMatcher results.

BestMatchAggregator selects the best video match from parallel workers
based on rating priority (CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL) and grade.
"""

import logging
from typing import List, Optional

from virtual_streamer.lib.agents.dynamic_parallel_processor import AbstractAggregator
from virtual_streamer.agents.video_matcher.schema import (
    ContextualRating,
    VideoMatchResult,
)

logger = logging.getLogger(__name__)


class BestMatchAggregator(AbstractAggregator[VideoMatchResult]):
    """
    Aggregator that selects the best video match from parallel workers.
    
    Selection priority:
    1. CONTEXTUAL rating (best) - pick highest grade among these
    2. NEUTRAL rating - pick highest grade among these
    3. NOT_CONTEXTUAL rating (fallback) - pick highest grade
    
    Example:
        # After MapperAgent runs parallel VideoMatchers:
        output_keys = [worker.get_output_key() for worker in workers]
        
        aggregator = BestMatchAggregator(
            state_input_keys=output_keys,
            result_state_key="best_video_match",
        )
        async for event in aggregator.run_async(ctx):
            yield event
        
        # Result stored at "best_video_match"
        best = ctx.session.state.get("best_video_match")
    """
    
    def __init__(
        self,
        state_input_keys: List[str],
        result_state_key: Optional[str] = None,
        name: str = "best_match_aggregator",
    ):
        """
        Initialize the aggregator.
        
        Args:
            state_input_keys: List of state keys to read VideoMatchResult from.
                             These are typically the output keys from VideoMatcher workers.
            result_state_key: Optional key to store the best match result in state.
            name: Name for this agent.
        """
        super().__init__(
            name=name,
            state_input_keys=state_input_keys,
            input_schema=VideoMatchResult,
            result_state_key=result_state_key,
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

