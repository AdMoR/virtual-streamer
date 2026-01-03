"""
Sentence Video Matcher Agent.

This agent takes DialogLines from the story generator and finds the best 
matching video for each dialog line using:
1. VideoSearchClient to find candidate videos via remote embedding server
2. Parallel video matching using MapperAgent + VideoMatcher workers
3. AggregatorAgent to select the best match per dialog line

The agent uses the MapReduceAgent pattern for clean orchestration.
"""
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.invocation_context import InvocationContext

from virtual_streamer.lib.agents import (
    MapperAgent,
    AggregatorAgent,
    MapReduceAgent,
)
from virtual_streamer.agents.video_matcher import (
    get_video_matcher,
    VideoMatchResult,
    ContextualRating,
)
from virtual_streamer.agents.story_generator.schema import DialogLine, DialogLines
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherOutput,
    DialogLineMatch,
)
from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_video_retriever,
)
from virtual_streamer.video_generation.interfaces import VideoRetrieverInterface

logger = logging.getLogger(__name__)

# State keys
SENTENCES_KEY = "sentences"
MATCHES_KEY = "video_matches"


class SentenceVideoMapper(MapperAgent):
    """
    Mapper that expands DialogLines into (dialog_line, video_path) pairs.
    
    For each dialog line in state:
    1. Searches video retriever for candidates using dialog text
    2. Creates one item per (dialog_line, candidate) pair
    
    Each item is processed by a VideoMatcher worker.
    """
    
    def __init__(
        self,
        video_retriever: VideoRetrieverInterface,
        max_candidates: int = 5,
        name: str = "sentence_video_mapper",
    ):
        """
        Initialize the mapper.
        
        Args:
            video_retriever: Interface for searching video clips by text
            max_candidates: Maximum number of candidate videos per dialog line
            name: Agent name
        """
        super().__init__(
            worker_factory=get_video_matcher,
            name=name,
        )
        self._video_retriever = video_retriever
        self._max_candidates = max_candidates
    
    def build_items_from_state(self, ctx: InvocationContext) -> List[Dict[str, Any]]:
        """
        Build (character, sentence, video_path) items from DialogLines in state.
        
        Reads DialogLines from state, parses into DialogLine objects,
        searches for candidate videos, and creates items matching 
        VideoSentenceInput schema.
        
        Returns:
            List of {"character": str, "sentence": str, "video_path": str} dicts
        """
        sentences_data = ctx.session.state.get(SENTENCES_KEY, {})
        
        # Parse DialogLines from state
        dialog_lines = DialogLines.model_validate(sentences_data)
        
        if not dialog_lines.lines:
            logger.warning("No dialog lines found in state")
            raise Exception("No dialog lines found in state")
        
        logger.info(f"Building items for {len(dialog_lines.lines)} dialog lines")
        
        items = []
        for dialog_line in dialog_lines.lines:
            # Search using the dialog text - returns VideoSearchResult objects
            search_results = self._video_retriever.search(
                dialog_line.dialog, 
                top_k=self._max_candidates
            )
            
            if not search_results:
                logger.warning(f"No candidates found for: {dialog_line.dialog[:50]}...")
                continue
            
            for result in search_results:
                items.append({
                    "character": dialog_line.character,
                    "sentence": dialog_line.dialog,
                    "video_path": result.path,
                })
        
        logger.info(f"Built {len(items)} items for parallel processing")
        return items


class SentenceVideoAggregator(AggregatorAgent[VideoMatchResult]):
    """
    Aggregator that selects the best video match for each dialog line.
    
    Groups results by sentence, then selects the best match per group
    based on rating priority (CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL)
    and grade within each rating level.
    
    Returns DialogLineMatch objects with full character info.
    """
    
    def __init__(
        self,
        input_keys: List[str],
        output_key: str = MATCHES_KEY,
        name: str = "sentence_video_aggregator",
    ):
        """
        Initialize the aggregator.
        
        Args:
            input_keys: State keys to read VideoMatchResult from
            output_key: State key to write final output
            name: Agent name
        """
        super().__init__(
            name=name,
            input_keys=input_keys,
            input_schema=VideoMatchResult,
            output_key=output_key,
        )
    
    async def aggregation_fn(
        self, results: List[VideoMatchResult]
    ) -> SentenceVideoMatcherOutput:
        """
        Group results by sentence and select best match per dialog line.
        
        Args:
            results: All VideoMatchResult from parallel workers
        
        Returns:
            SentenceVideoMatcherOutput with DialogLineMatch per dialog line
        """
        # Group by (character, sentence) tuple to handle same sentence from different characters
        by_dialog: Dict[tuple, List[VideoMatchResult]] = {}
        for result in results:
            key = (result.character, result.sentence)
            by_dialog.setdefault(key, []).append(result)
        
        logger.info(f"Aggregating results for {len(by_dialog)} dialog lines")
        
        # Select best from each group and convert to DialogLineMatch
        matches = []
        for (character, sentence), group in by_dialog.items():
            best = self._select_best(group)
            if best:
                # Convert VideoMatchResult to DialogLineMatch
                dialog_line_match = DialogLineMatch(
                    dialog_line=DialogLine(character=character, dialog=sentence),
                    video_path=best.video_path,
                    rating=best.rating,
                    grade=best.grade,
                    reasoning=best.reasoning,
                )
                matches.append(dialog_line_match)
                logger.debug(
                    f"Best for '{character}: {sentence[:30]}...': "
                    f"{best.video_path} (rating={best.rating}, grade={best.grade})"
                )
        
        return SentenceVideoMatcherOutput(matches=matches)
    
    def _select_best(
        self, results: List[VideoMatchResult]
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


def create_sentence_video_matcher(
    video_retriever: VideoRetrieverInterface,
    max_candidates: int = 5,
) -> MapReduceAgent:
    """
    Factory function to create a SentenceVideoMatcher agent.
    
    Args:
        video_retriever: Interface for searching video clips
        max_candidates: Maximum candidate videos per dialog line
    
    Returns:
        Configured MapReduceAgent for sentence-video matching
    """
    mapper = SentenceVideoMapper(
        video_retriever=video_retriever,
        max_candidates=max_candidates,
        name="sentence_video_mapper",
    )
    
    return MapReduceAgent(
        mapper=mapper,
        aggregator_factory=lambda keys: SentenceVideoAggregator(
            input_keys=keys,
            output_key=MATCHES_KEY,
            name="sentence_video_aggregator",
        ),
        name="sentence_video_matcher",
    )
