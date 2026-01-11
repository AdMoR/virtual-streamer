"""
Rubric Builder Map-Reduce Agent.

This agent extracts rubrics from stories using a map-reduce pattern:
1. MapperAgent chunks stories into batches of 5
2. StatefulRubricBuilder workers process each batch in parallel
3. AggregatorAgent collects rubrics and writes to JSONL file

Usage:
    from virtual_streamer.agents.rubric_builder_map_reduce import (
        create_rubric_builder_map_reduce,
    )
    
    agent = create_rubric_builder_map_reduce(
        output_path=Path("rubrics.jsonl"),
        batch_size=5,
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents.invocation_context import InvocationContext

from virtual_streamer.lib.agents import (
    MapperAgent,
    AggregatorAgent,
    MapReduceAgent,
    StatefulLlmAgent,
)
from virtual_streamer.agents.rubric_builder_agent.prompt import PROMPT
from virtual_streamer.agents.rubric_builder_agent.schema import (
    MapPhaseOutput,
    StoryItem,
    StoryBatchInput,
    Rubric,
)
from virtual_streamer.agents.rubric_builder_map_reduce.callback import (
    InjectStoriesCallback,
    StoreRubricsCallback,
)

logger = logging.getLogger(__name__)

# State key for input stories
STORIES_KEY = "stories"

# Default batch size
DEFAULT_BATCH_SIZE = 5


# =============================================================================
# Stateful Worker
# =============================================================================


def get_stateful_rubric_builder(run_id: Optional[str] = None) -> StatefulLlmAgent:
    """
    Factory function to create a Stateful RubricBuilderAgent for a specific run.
    
    This creates a new agent instance for each batch of stories being processed,
    with callbacks configured for the specific run_id.
    
    Args:
        run_id: Unique ID for this processing run (e.g., "w0").
                If None, keys will not be namespaced.
    
    Returns:
        Configured StatefulLlmAgent for rubric building with:
        - get_input_key(): returns the state key for input
        - get_output_key(): returns the state key for output
    
    Example:
        # Without run_id (for standalone use)
        worker = get_stateful_rubric_builder()
        print(worker.get_input_key())  # "story_batch"
        
        # With run_id (for parallel processing)
        worker = get_stateful_rubric_builder(run_id="w0")
        print(worker.get_input_key())   # "task:w0:story_batch"
        print(worker.get_output_key())  # "result:w0:rubrics"
    """
    input_callback = InjectStoriesCallback(run_id)
    output_callback = StoreRubricsCallback(run_id)
    
    return StatefulLlmAgent(
        name="stateful_rubric_builder",
        instruction=PROMPT,
        output_schema=MapPhaseOutput,
        input_callback=input_callback,
        output_callback=output_callback,
    )


# =============================================================================
# Mapper
# =============================================================================


class RubricMapper(MapperAgent):
    """
    Mapper that chunks stories into batches and creates rubric builder workers.
    
    Reads stories from state at STORIES_KEY, chunks into batches of `batch_size`,
    and creates a StatefulRubricBuilder worker for each batch.
    
    Example:
        mapper = RubricMapper(batch_size=5)
        
        # After running, get output keys for aggregator:
        output_keys = mapper.get_output_keys()
    """
    
    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        name: str = "rubric_mapper",
    ):
        """
        Initialize the mapper.
        
        Args:
            batch_size: Number of stories per worker batch (default: 5)
            name: Name for this agent
        """
        super().__init__(
            worker_factory=get_stateful_rubric_builder,
            name=name,
        )
        self._batch_size = batch_size
    
    def build_items_from_state(self, ctx: InvocationContext) -> List[Dict[str, Any]]:
        """
        Build story batch items from stories in state.
        
        Reads stories from state, chunks into batches of batch_size,
        and creates items matching StoryBatchInput schema.
        
        Returns:
            List of {"stories": [StoryItem, ...]} dicts, one per batch
        """
        stories_data = ctx.session.state.get(STORIES_KEY, [])
        
        if not stories_data:
            logger.warning(f"No stories found in state at key '{STORIES_KEY}'")
            return []
        
        # Parse stories into StoryItem objects
        stories = []
        for story_dict in stories_data:
            try:
                story = StoryItem.model_validate(story_dict)
                stories.append(story)
            except Exception as e:
                logger.warning(f"Failed to parse story: {e}")
                continue
        
        if not stories:
            logger.warning("No valid stories after parsing")
            return []
        
        logger.info(f"Processing {len(stories)} stories with batch size {self._batch_size}")
        
        # Chunk stories into batches
        items = []
        for i in range(0, len(stories), self._batch_size):
            batch = stories[i:i + self._batch_size]
            items.append({
                "stories": [s.model_dump() for s in batch]
            })
        
        logger.info(f"Created {len(items)} batches for parallel processing")
        return items


# =============================================================================
# Aggregator
# =============================================================================


class RubricAggregator(AggregatorAgent[MapPhaseOutput]):
    """
    Aggregator that collects rubrics from all workers and writes to JSONL.
    
    Each worker produces a MapPhaseOutput with a list of rubrics.
    This aggregator:
    1. Collects all MapPhaseOutput from workers
    2. Flattens all rubrics into a single list
    3. Writes each rubric as a line in the output JSONL file
    
    Example:
        aggregator = RubricAggregator(
            input_keys=mapper.get_output_keys(),
            output_path=Path("rubrics.jsonl"),
        )
    """
    
    def __init__(
        self,
        input_keys: List[str],
        output_path: Path,
        output_key: str = "all_rubrics",
        name: str = "rubric_aggregator",
    ):
        """
        Initialize the aggregator.
        
        Args:
            input_keys: List of state keys to read MapPhaseOutput from.
                       These are typically mapper.get_output_keys().
            output_path: Path to write the JSONL output file.
            output_key: Key to store aggregated rubrics in state.
            name: Name for this agent.
        """
        super().__init__(
            name=name,
            input_keys=input_keys,
            input_schema=MapPhaseOutput,
            output_key=output_key,
        )
        self._output_path = output_path
    
    async def aggregation_fn(
        self, results: List[MapPhaseOutput]
    ) -> Optional[List[Rubric]]:
        """
        Aggregate rubrics from all workers and write to JSONL.
        
        Args:
            results: List of MapPhaseOutput from parallel workers.
        
        Returns:
            Flattened list of all Rubric objects.
        """
        if not results:
            logger.warning("No results to aggregate")
            return None
        
        # Flatten all rubrics from all workers
        all_rubrics: List[Rubric] = []
        for result in results:
            all_rubrics.extend(result.rubrics)
        
        logger.info(f"Aggregated {len(all_rubrics)} rubrics from {len(results)} workers")
        
        # Write to JSONL file
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._output_path, "w", encoding="utf-8") as f:
            for rubric in all_rubrics:
                f.write(rubric.model_dump_json() + "\n")
        
        logger.info(f"Wrote {len(all_rubrics)} rubrics to {self._output_path}")
        
        return all_rubrics


# =============================================================================
# Factory Function
# =============================================================================


def create_rubric_builder_map_reduce(
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> MapReduceAgent:
    """
    Factory function to create a RubricBuilder MapReduce agent.
    
    Args:
        output_path: Path to write the JSONL output file
        batch_size: Number of stories per batch (default: 5)
    
    Returns:
        Configured MapReduceAgent for rubric extraction
    
    Example:
        agent = create_rubric_builder_map_reduce(
            output_path=Path("rubrics.jsonl"),
            batch_size=5,
        )
    """
    mapper = RubricMapper(
        batch_size=batch_size,
        name="rubric_mapper",
    )
    
    return MapReduceAgent(
        mapper=mapper,
        aggregator_factory=lambda keys: RubricAggregator(
            input_keys=keys,
            output_path=output_path,
            name="rubric_aggregator",
        ),
        name="rubric_builder_map_reduce",
    )
