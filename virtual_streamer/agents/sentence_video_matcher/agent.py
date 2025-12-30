"""
Sentence Video Matcher Agent.

This agent takes a list of sentences and finds the best matching video
for each sentence using:
1. Vector store search to find candidate videos
2. Parallel video matching using MapperAgent + VideoMatcher workers
3. AggregatorAgent to select the best match per sentence

The agent uses the MapReduceAgent pattern for clean orchestration.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.invocation_context import InvocationContext
from google.genai.types import Content, Part

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
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherOutput,
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
    Mapper that expands sentences into (sentence, video_path) pairs.
    
    For each sentence in state:
    1. Searches video retriever for candidates
    2. Creates one item per (sentence, candidate) pair
    
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
            max_candidates: Maximum number of candidate videos per sentence
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
        Build (sentence, video_path) items from sentences in state.
        
        Reads sentences from state, searches for candidate videos,
        and creates items matching VideoSentenceInput schema.
        
        Returns:
            List of {"sentence": str, "video_path": str} dicts
        """
        sentences = ctx.session.state.get(SENTENCES_KEY, [])

        if len(sentences) == 0:
            part: Part = ctx.user_content.parts[0]
            sentences = json.loads(part.text)

        
        if not sentences:
            logger.warning("No sentences found in state")
            raise Exception("No sentences found in state")
            return []
        
        logger.info(f"Building items for {len(sentences)} sentences")
        
        items = []
        for sentence in sentences:
            candidates = self._video_retriever.search(
                sentence, 
                top_k=self._max_candidates
            )
            
            if not candidates:
                logger.warning(f"No candidates found for: {sentence[:50]}...")
                continue
            
            for video_path in candidates:
                items.append({
                    "sentence": sentence,
                    "video_path": video_path,
                })
        
        logger.info(f"Built {len(items)} items for parallel processing")
        return items


class SentenceVideoAggregator(AggregatorAgent[VideoMatchResult]):
    """
    Aggregator that selects the best video match for each sentence.
    
    Groups results by sentence, then selects the best match per group
    based on rating priority (CONTEXTUAL > NEUTRAL > NOT_CONTEXTUAL)
    and grade within each rating level.
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
        Group results by sentence and select best match per sentence.
        
        Args:
            results: All VideoMatchResult from parallel workers
        
        Returns:
            SentenceVideoMatcherOutput with one best match per sentence
        """
        # Group by sentence
        by_sentence: Dict[str, List[VideoMatchResult]] = {}
        for result in results:
            by_sentence.setdefault(result.sentence, []).append(result)
        
        logger.info(f"Aggregating results for {len(by_sentence)} sentences")
        
        # Select best from each group
        matches = []
        for sentence, group in by_sentence.items():
            best = self._select_best(group)
            if best:
                matches.append(best)
                logger.debug(
                    f"Best for '{sentence[:30]}...': "
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
    config: Optional[VideoGenerationConfig] = None
) -> MapReduceAgent:
    """
    Factory function to create a SentenceVideoMatcher agent.
    """
    if config is None:
        config = VideoGenerationConfig()
    video_retriever = create_video_retriever(config.video_retrieval)
    return SentenceVideoMatcherAgent(
        video_retriever=video_retriever,
    )


# Backward compatibility: keep the class-based interface
class SentenceVideoMatcherAgent(MapReduceAgent):
    """
    Agent that finds the best matching video for each sentence.
    
    This is a convenience wrapper around create_sentence_video_matcher
    that provides the same interface as before refactoring.
    
    Example:
        retriever = MyVideoRetriever()
        agent = SentenceVideoMatcherAgent(video_retriever=retriever)
        
        # Set sentences in state
        ctx.session.state["sentences"] = ["Hello world", "Goodbye"]
        
        # Run agent
        async for event in agent.run_async(ctx):
            yield event
        
        # Get results
        matches = ctx.session.state["video_matches"]
    """
    
    def __init__(
        self,
        video_retriever: VideoRetrieverInterface,
        name: str = "sentence_video_matcher",
    ):
        """
        Initialize the agent.
        
        Args:
            video_retriever: Interface for searching video clips by text
            max_candidates: Maximum number of candidate videos per sentence
            name: Agent name
        """
        mapper = SentenceVideoMapper(
            video_retriever=video_retriever,
            name=f"{name}_mapper",
        )
        
        super().__init__(
            mapper=mapper,
            aggregator_factory=lambda keys: SentenceVideoAggregator(
                input_keys=keys,
                output_key=MATCHES_KEY,
                name=f"{name}_aggregator",
            ),
            name=name,
        )
        
        # Store for backward compatibility with tests
        self._video_retriever = video_retriever
        self._max_candidates = 5


root_agent = create_sentence_video_matcher()
