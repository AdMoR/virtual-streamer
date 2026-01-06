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
from virtual_streamer.agents.sentence_video_matcher.utils import _select_best
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
        max_candidates: int = 1,
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
        Build (character_id, sentence, scene_description, video_path) items from DialogLines in state.
        
        Reads DialogLines from state, parses into DialogLine objects,
        searches for candidate videos using scene_description, and creates 
        items matching VideoSentenceInput schema.
        
        Returns:
            List of {"character_id": str, "sentence": str, "scene_description": str, "video_path": str} dicts
        """
        sentences_data = ctx.session.state.get(SENTENCES_KEY, {})
        
        # Parse DialogLines from state
        dialog_lines = DialogLines.model_validate(sentences_data)
        
        if not dialog_lines.lines:
            logger.warning("No dialog lines found in state")
            raise Exception("No dialog lines found in state")
        
        logger.info(f"Building items for {len(dialog_lines.lines)} dialog lines")
        
        items = []
        for line_id, dialog_line in enumerate(dialog_lines.lines):
            # Search using scene_description for video embedding search
            search_results = self._video_retriever.search(
                dialog_line.scene_description, 
                top_k=self._max_candidates
            )
            
            if not search_results:
                logger.warning(f"No candidates found for line {line_id}: {dialog_line.scene_description[:50]}...")
                continue
            
            for result in search_results:
                items.append({
                    "line_id": line_id,
                    "character_id": dialog_line.character_id,
                    "sentence": dialog_line.text,
                    "scene_description": dialog_line.scene_description,
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
            self,
            results: List[VideoMatchResult]
    ) -> SentenceVideoMatcherOutput:
        """
        Group results by line_id and select best match per dialog line.

        Args:
            results: All VideoMatchResult from parallel workers

        Returns:
            SentenceVideoMatcherOutput with DialogLineMatch per dialog line
        """
        # Group by line_id (guaranteed unique per dialog line)
        by_line: Dict[int, List[VideoMatchResult]] = {}
        for result in results:
            by_line.setdefault(result.line_id, []).append(result)

        logger.info(f"Aggregating results for {len(by_line)} dialog lines")

        # Select best from each group and convert to DialogLineMatch
        # Sort by line_id to preserve original order
        matches = []
        for line_id in sorted(by_line.keys()):
            group = by_line[line_id]
            best = _select_best(group)
            if best:
                # Convert VideoMatchResult to DialogLineMatch
                dialog_line_match = DialogLineMatch(
                    dialog_line=DialogLine(
                        character_id=best.character_id,
                        text=best.sentence,
                        scene_description=best.scene_description,
                    ),
                    video_path=best.video_path,
                    rating=best.rating,
                    grade=best.grade,
                    reasoning=best.reasoning,
                )
                matches.append(dialog_line_match)
                logger.debug(
                    f"Best for line {line_id} '{best.character_id}: {best.sentence[:30]}...': "
                    f"{best.video_path} (rating={best.rating}, grade={best.grade})"
                )

        return SentenceVideoMatcherOutput(matches=matches)


def create_sentence_video_matcher(
    video_retriever: VideoRetrieverInterface,
    max_candidates: int = 1,
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
