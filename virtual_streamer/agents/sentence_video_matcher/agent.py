"""
Sentence Video Matcher Agent.

This agent takes a list of sentences and finds the best matching video
for each sentence using:
1. Vector store search to find candidate videos
2. Parallel video matching using MapperAgent + VideoMatcher
3. BestMatchAggregator to select the best match per sentence

The output is a list of VideoMatchResult, one per input sentence.
"""

import logging
from typing import AsyncGenerator, List, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from virtual_streamer.lib.agents.dynamic_parallel_processor import MapperAgent
from virtual_streamer.agents.video_matcher import (
    get_video_matcher,
    BestMatchAggregator,
    VideoMatchResult,
)
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherInput,
    SentenceVideoMatcherOutput,
)
from virtual_streamer.video_generation.interfaces import VideoRetrieverInterface

logger = logging.getLogger(__name__)

# State keys
SENTENCES_KEY = "sentences"
MATCHES_KEY = "video_matches"


class SentenceVideoMatcherAgent(BaseAgent):
    """
    Agent that finds the best matching video for each sentence.
    
    For each sentence:
    1. Searches vector store for 5-10 candidate videos
    2. Runs parallel VideoMatchers to judge each (sentence, video) pair
    3. Aggregates results to select the best match
    
    The result is stored in state at MATCHES_KEY as a list of VideoMatchResult.
    
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
        max_candidates: int = 5,
        name: str = "sentence_video_matcher",
    ):
        """
        Initialize the agent.
        
        Args:
            video_retriever: Interface for searching video clips by text
            max_candidates: Maximum number of candidate videos per sentence
            name: Agent name
        """
        super().__init__(name=name)
        # Use underscore prefix to bypass Pydantic's field validation
        # (Pydantic ignores attributes starting with underscore)
        self._video_retriever = video_retriever
        self._max_candidates = max_candidates
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Process all sentences and find best matching videos.
        
        Reads sentences from state, processes each one, and stores
        the list of VideoMatchResult in state.
        """
        # Read sentences from state
        sentences = ctx.session.state.get(SENTENCES_KEY, [])
        
        if not sentences:
            logger.warning("No sentences found in state")
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="No sentences to process")],
                ),
            )
            return
        
        logger.info(f"Processing {len(sentences)} sentences")
        
        all_matches: List[VideoMatchResult] = []
        
        for idx, sentence in enumerate(sentences):
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(
                        text=f"Processing sentence {idx + 1}/{len(sentences)}: {sentence[:50]}..."
                    )],
                ),
            )
            
            # Find best match for this sentence
            best_match = await self._process_sentence(ctx, idx, sentence)
            
            if best_match:
                all_matches.append(best_match)
                logger.info(
                    f"Sentence {idx}: matched with {best_match.video_path} "
                    f"(rating={best_match.rating}, grade={best_match.grade})"
                )
            else:
                logger.warning(f"Sentence {idx}: no match found")
        
        # Store final results in state
        output = SentenceVideoMatcherOutput(matches=all_matches)
        
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(
                    text=f"Completed: matched {len(all_matches)}/{len(sentences)} sentences"
                )],
            ),
            actions=EventActions(
                state_delta={
                    MATCHES_KEY: [m.model_dump() for m in all_matches]
                }
            ),
        )
        
        logger.info(f"Sentence video matching complete: {len(all_matches)} matches")
    
    async def _process_sentence(
        self,
        ctx: InvocationContext,
        sentence_idx: int,
        sentence: str,
    ) -> Optional[VideoMatchResult]:
        """
        Process a single sentence: search, match in parallel, aggregate.
        
        Args:
            ctx: Invocation context
            sentence_idx: Index of this sentence (for namespacing)
            sentence: The sentence to match
        
        Returns:
            The best VideoMatchResult, or None if no match found
        """
        # Step 1: Search for candidate videos
        candidates = self._video_retriever.search(sentence, top_k=self._max_candidates * 2)
        
        if not candidates:
            logger.warning(f"No candidate videos found for sentence: {sentence[:50]}")
            return None
        
        # Limit to max_candidates
        candidates = candidates[:self._max_candidates]
        
        logger.debug(f"Found {len(candidates)} candidate videos for sentence {sentence_idx}")
        
        # Step 2: Build items for parallel matching
        items = [
            {"sentence": sentence, "video_path": video_path}
            for video_path in candidates
        ]
        
        # Step 3: Create and run MapperAgent
        run_id = f"s{sentence_idx}"
        
        mapper = MapperAgent(
            items=items,
            worker_factory=get_video_matcher,
            name=f"mapper_{run_id}",
        )
        
        # Run the mapper (this will update state with results)
        async for event in mapper.run_async(ctx):
            pass  # Events are processed internally
        
        # Step 4: Get output keys directly from mapper (no state scanning!)
        output_keys = mapper.get_output_keys()
        
        if not output_keys:
            logger.warning(f"No results found for sentence {sentence_idx}")
            return None
        
        logger.debug(f"Found {len(output_keys)} results to aggregate")
        
        # Step 5: Aggregate to find best match
        aggregator = BestMatchAggregator(
            state_input_keys=output_keys,
            result_state_key=f"best_match_{sentence_idx}",
            name=f"aggregator_{run_id}",
        )
        
        async for event in aggregator.run_async(ctx):
            pass  # Events are processed internally
        
        # Step 6: Retrieve the best match
        best_match_data = ctx.session.state.get(f"best_match_{sentence_idx}")
        
        if not best_match_data:
            logger.warning(f"No best match found for sentence {sentence_idx}")
            return None
        
        # Parse the result
        if isinstance(best_match_data, str):
            return VideoMatchResult.model_validate_json(best_match_data)
        elif isinstance(best_match_data, dict):
            return VideoMatchResult.model_validate(best_match_data)
        else:
            return best_match_data

