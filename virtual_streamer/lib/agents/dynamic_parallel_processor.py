"""
Dynamic Parallel Processing Agents.

This module provides agents for dynamically distributing tasks across
multiple workers and aggregating their results.

Key components:
- MapperAgent: Abstract base for distributing tasks to parallel workers
- AggregatorAgent: Abstract base for collecting and aggregating worker results
- MapReduceAgent: Orchestrates MapperAgent → AggregatorAgent pipeline

Usage:
    # Define concrete mapper
    class MyMapper(MapperAgent):
        def build_items_from_state(self, ctx):
            data = ctx.session.state.get("input_data", [])
            return [{"field": x} for x in data]
    
    # Define concrete aggregator
    class MyAggregator(AggregatorAgent):
        async def aggregation_fn(self, results):
            return max(results, key=lambda r: r.score)
    
    # Wire together
    mapper = MyMapper(worker_factory=get_my_worker)
    agent = MapReduceAgent(
        mapper=mapper,
        aggregator_factory=lambda keys: MyAggregator(input_keys=keys, ...),
    )
"""

import asyncio
import json
import logging
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Type, TypeVar

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default rate limit retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 60


class StatefulWorker(Protocol):
    """Protocol for workers that expose input/output keys and schemas."""

    def get_input_key(self) -> str:
        """Return the state key where input should be written."""
        ...

    def get_input_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for input validation."""
        ...

    def get_output_key(self) -> str:
        """Return the state key where output will be written."""
        ...

    def get_output_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for output validation."""
        ...


# Type alias for worker factory function
WorkerFactory = Callable[[str], StatefulWorker]


class MapperAgent(SequentialAgent, ABC):
    """
    Abstract base for parallel task distribution.
    
    Subclasses implement build_items_from_state() to:
    - Read input from session state
    - Generate items matching worker's input schema
    
    After running, use get_output_keys() to get the state keys where
    workers wrote their results.
    
    Example:
        class SentenceVideoMapper(MapperAgent):
            def __init__(self, video_retriever, max_candidates=5):
                super().__init__(
                    worker_factory=get_video_matcher,
                    name="sentence_mapper",
                )
                self._video_retriever = video_retriever
                self._max_candidates = max_candidates
            
            def build_items_from_state(self, ctx):
                sentences = ctx.session.state.get("sentences", [])
                collection = ctx.session.state.get("video_collection")
                items = []
                for sentence in sentences:
                    candidates = self._video_retriever.search(sentence, collection)
                    for video_path in candidates:
                        items.append({"sentence": sentence, "video_path": video_path})
                return items
    """

    _workers: List[StatefulWorker]

    def __init__(
            self,
            worker_factory: WorkerFactory,
            name: str = "mapper",
    ):
        """
        Initialize the mapper agent.
        
        Args:
            worker_factory: Function that creates a StatefulWorker given a run_id.
            name: Name for this agent (default: "mapper")
        """
        super().__init__(name=name, sub_agents=[])
        self._worker_factory = worker_factory
        self._workers = []

    @abstractmethod
    def build_items_from_state(self, ctx: InvocationContext) -> List[Dict[str, Any]]:
        """
        Build worker items by reading from session state.
        
        Subclasses implement this to:
        1. Read input data from ctx.session.state
        2. Generate items matching the worker's input schema
        
        Args:
            ctx: Invocation context with session state access
        
        Returns:
            List of dicts, each matching worker's input schema.
            Empty list if no items to process.
        """
        ...

    async def _run_async_impl(self, ctx: InvocationContext):
        """
        Build items from state, distribute to workers, run in parallel.
        
        Yields events for:
        1. State delta with all input data written to worker keys
        2. Events from parallel worker execution
        
        Raises:
            ValidationError: If any item fails schema validation
        """
        # Build items from state (subclass implements)
        items = self.build_items_from_state(ctx)

        if not items:
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="No items to process")]
                ),
            )
            return

        # Generate unique run ID for this batch
        run_id = secrets.token_hex(2)

        # Create workers and prepare state delta
        self._workers = []
        state_delta: Dict[str, str] = {"current_run": run_id}

        for i, item in enumerate(items):
            worker_run_id = f"{run_id}:w{i}"
            worker = self._worker_factory(worker_run_id)
            logger.info(worker, worker_run_id)

            # Get the input key and schema from the worker (type-safe!)
            input_key = worker.get_input_key()
            input_schema = worker.get_input_schema()

            # Validate item against the worker's input schema
            validated_item = input_schema.model_validate(item)

            # Serialize validated item to JSON and store in state delta
            state_delta[input_key] = validated_item.model_dump_json()
            self._workers.append(worker)

            logger.info(
                f"Validated item {i} for worker {worker_run_id}: "
                f"{input_schema.__name__}"
            )

        # Emit state delta with all inputs
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"Run {run_id}: distributing {len(items)} tasks")]
            ),
            actions=EventActions(state_delta=state_delta)
        )

        # Create parallel agent with workers and run
        parallel = SequentialAgent(
            name=f"parallel_{run_id}",
            sub_agents=self._workers  # type: ignore (workers are agents),

        )

        async for event in parallel.run_async(ctx):
            yield event

    def get_output_keys(self) -> List[str]:
        """
        Get output keys from all workers after run.
        
        Call this after running the mapper to get the state keys
        where workers wrote their results.
        
        Returns:
            List of output state keys from workers.
            Empty list if run() hasn't been called yet.
        """
        return [worker.get_output_key() for worker in self._workers]

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        """
        Get the output schema from workers.
        
        Returns:
            The output schema from the first worker, or None if no workers.
        """
        if self._workers:
            return self._workers[0].get_output_schema()
        return None


T = TypeVar("T", bound=BaseModel)


class AggregatorAgent(BaseAgent, Generic[T], ABC):
    """
    Abstract base for aggregating results from parallel workers.
    
    Subclasses implement aggregation_fn() to combine worker results.
    
    The base class handles:
    - Collecting results from specified state keys
    - Parsing each result with the input schema
    - Storing aggregated result in state
    - Event emission
    
    Example:
        class BestMatchAggregator(AggregatorAgent[VideoMatchResult]):
            def __init__(self, input_keys: List[str]):
                super().__init__(
                    name="best_match",
                    input_keys=input_keys,
                    input_schema=VideoMatchResult,
                    output_key="best_video_match",
                )
            
            async def aggregation_fn(self, results: List[VideoMatchResult]):
                return max(results, key=lambda r: r.grade)
    """

    def __init__(
            self,
            name: str,
            input_keys: List[str],
            input_schema: Type[T],
            output_key: str,
    ):
        """
        Initialize the aggregator.
        
        Args:
            name: Name for this agent
            input_keys: List of state keys to read results from.
                       These are typically mapper.get_output_keys().
            input_schema: Pydantic model to parse each result into.
            output_key: Key to store aggregated result in state.
        """
        super().__init__(name=name)
        self._input_keys = input_keys
        self._input_schema = input_schema
        self._output_key = output_key

    def _collect_results(self, ctx: InvocationContext) -> List[T]:
        """
        Collect and parse results from state.
        
        Reads from input_keys, parses each with input_schema.
        
        Args:
            ctx: Invocation context with session state
        
        Returns:
            List of validated Pydantic model instances
        """
        results: List[T] = []

        for key in self._input_keys:
            value = ctx.session.state.get(key)

            if value is None:
                logger.warning(f"No value found for state key: {key}")
                continue

            try:
                if isinstance(value, str):
                    parsed = self._input_schema.model_validate_json(value)
                elif isinstance(value, dict):
                    parsed = self._input_schema.model_validate(value)
                elif isinstance(value, self._input_schema):
                    parsed = value
                else:
                    logger.warning(
                        f"Unexpected value type for key {key}: {type(value)}"
                    )
                    continue

                results.append(parsed)
                logger.debug(f"Parsed result from {key}: {parsed}")

            except Exception as e:
                logger.error(f"Failed to parse result from {key}: {e}")

        return results

    async def _run_async_impl(self, ctx: InvocationContext):
        """
        Collect results from workers and aggregate them.
        
        1. Collects and parses results from input_keys
        2. Calls aggregation_fn with parsed results
        3. Stores result in state at output_key
        4. Emits event with aggregation summary
        """
        # Collect and parse results
        results = self._collect_results(ctx)

        if not results:
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="No valid results to aggregate")]
                ),
                actions=EventActions(escalate=True)
            )
            return

        logger.info(f"Aggregating {len(results)} results from {len(self._input_keys)} keys")

        # Aggregate results (subclass implements)
        aggregated = await self.aggregation_fn(results)

        # Prepare state delta
        state_delta: Dict[str, str] = {}
        if aggregated is not None:
            if isinstance(aggregated, BaseModel):
                state_delta[self._output_key] = aggregated.model_dump_json()
            else:
                state_delta[self._output_key] = json.dumps(aggregated)

            logger.debug(f"Stored aggregated result at {self._output_key}")

        # Emit completion event
        summary = f"Aggregated {len(results)} results -> {self._output_key}"

        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=summary)]
            ),
            actions=EventActions(
                state_delta=state_delta if state_delta else None,
            )
        )

    @abstractmethod
    async def aggregation_fn(self, results: List[T]) -> Optional[T]:
        """
        Aggregate parsed results from workers.
        
        Args:
            results: List of validated Pydantic model instances.
                    Guaranteed to be non-empty when called.
        
        Returns:
            Aggregated result. Can be:
            - A single model instance (e.g., best match)
            - A new model instance (e.g., combined result)
            - None if aggregation produces no result
        """
        ...


# Type alias for aggregator factory
AggregatorFactory = Callable[[List[str]], AggregatorAgent]


class MapReduceAgent(BaseAgent):
    """
    Orchestrates a MapperAgent → AggregatorAgent pipeline.
    
    Handles the boilerplate of:
    1. Running the mapper to generate and process items
    2. Wiring mapper's output keys to aggregator's input keys
    3. Running the aggregator to combine results
    
    Example:
        mapper = SentenceVideoMapper(video_retriever, max_candidates=5)
        
        agent = MapReduceAgent(
            mapper=mapper,
            aggregator_factory=lambda keys: SentenceVideoAggregator(
                input_keys=keys,
                output_key="video_matches",
            ),
            name="sentence_video_matcher",
        )
        
        # Run the pipeline
        async for event in agent.run_async(ctx):
            yield event
        
        # Results are in ctx.session.state["video_matches"]
    """

    def __init__(
            self,
            mapper: MapperAgent,
            aggregator_factory: AggregatorFactory,
            name: str = "map_reduce",
            max_retries: int = DEFAULT_MAX_RETRIES,
            retry_delay_seconds: int = DEFAULT_RETRY_DELAY_SECONDS,
    ):
        """
        Initialize the map-reduce agent.
        
        Args:
            mapper: A concrete MapperAgent that implements build_items_from_state
            aggregator_factory: Function that takes input_keys and returns
                               a concrete AggregatorAgent
            name: Name for this agent
            max_retries: Maximum retries on rate limit errors (default: 3)
            retry_delay_seconds: Seconds to wait between retries (default: 60)
        """
        super().__init__(name=name, )
        self._mapper = mapper
        self._aggregator_factory = aggregator_factory
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def _dump_state_on_failure(
        self,
        ctx: InvocationContext,
        error: Exception,
        phase_name: str,
    ) -> Path:
        """
        Dump current session state to JSON file for recovery/debugging.
        
        Args:
            ctx: Invocation context with session state
            error: The exception that caused the failure
            phase_name: Name of the phase that failed ("map" or "reduce")
        
        Returns:
            Path to the dumped state file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = Path(f"state_dump_{self.name}_{phase_name}_{timestamp}.json")
        
        # Collect state data
        state_data = {}
        try:
            for key in ctx.session.state.keys():
                value = ctx.session.state.get(key)
                # Try to serialize - some values might not be JSON-serializable
                if isinstance(value, str):
                    state_data[key] = value
                elif hasattr(value, 'model_dump'):
                    state_data[key] = value.model_dump()
                elif hasattr(value, '__dict__'):
                    state_data[key] = str(value)
                else:
                    state_data[key] = str(value)
        except Exception as e:
            logger.warning(f"Error collecting state for dump: {e}")
        
        dump_content = {
            "agent_name": self.name,
            "phase": phase_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": timestamp,
            "state": state_data,
        }
        
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"State dumped to {dump_path}")
        return dump_path

    async def _run_async_impl(self, ctx: InvocationContext):
        """
        Run the map-reduce pipeline with rate limit retry handling.
        
        1. Run mapper (builds items from state, processes in parallel)
        2. Create aggregator with mapper's output keys
        3. Run aggregator (collects results, applies aggregation_fn)
        
        On rate limit errors (litellm.RateLimitError), retries up to max_retries
        times with retry_delay_seconds between attempts. On final failure,
        dumps state to a JSON file for debugging/recovery.
        """
        # Phase 1: Run mapper with retry
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text="Starting map phase...")]
            ),
        )

        map_retries = 0
        while map_retries <= self._max_retries:
            try:
                async for event in self._mapper.run_async(ctx):
                    yield event
                break  # Success - exit retry loop
            except Exception as e:
                if self._is_rate_limit_error(e) and map_retries < self._max_retries:
                    map_retries += 1
                    await asyncio.sleep(self._retry_delay_seconds)
                else:
                    dump_path = self._dump_state_on_failure(ctx, e, "map")
                    raise

        # Phase 2: Create aggregator with mapper's output keys
        output_keys = self._mapper.get_output_keys()

        if not output_keys:
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="Map phase produced no results")]
                ),
            )
            return

        aggregator = self._aggregator_factory(output_keys)

        # Phase 3: Run aggregator with retry
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"Starting reduce phase with {len(output_keys)} results...")]
            ),
        )

        reduce_retries = 0
        while reduce_retries <= self._max_retries:
            try:
                async for event in aggregator.run_async(ctx):
                    yield event
                break  # Success - exit retry loop
            except Exception as e:
                if self._is_rate_limit_error(e) and reduce_retries < self._max_retries:
                    reduce_retries += 1
                    await asyncio.sleep(self._retry_delay_seconds)
                else:
                    dump_path = self._dump_state_on_failure(ctx, e, "reduce")
                    raise

        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text="Map-reduce complete")]
            ),
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if an exception is a rate limit error.
        
        Handles litellm.RateLimitError and similar rate limit exceptions.
        
        Args:
            error: The exception to check
        
        Returns:
            True if it's a rate limit error, False otherwise
        """
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Check by class name
        if error_type == "RateLimitError":
            return True
        
        # Check inheritance chain for RateLimitError
        for cls in type(error).__mro__:
            if cls.__name__ == "RateLimitError":
                return True
        
        # Check error message for rate limit indicators
        if "rate" in error_str and "limit" in error_str:
            return True
        if "429" in error_str:  # HTTP 429 Too Many Requests
            return True
        if "quota" in error_str and "exceeded" in error_str:
            return True
        
        return False


# Keep backward compatibility alias
AbstractAggregator = AggregatorAgent
