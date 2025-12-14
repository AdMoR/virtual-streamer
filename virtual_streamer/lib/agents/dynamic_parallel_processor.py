"""
Dynamic Parallel Processing Agents.

This module provides agents for dynamically distributing tasks across
multiple workers and aggregating their results. It leverages StatefulLlmAgent
for type-safe state key management.

Key components:
- MapperAgent: Distributes tasks to workers, writing inputs to their state keys
- AbstractAggregator: Base class for collecting and aggregating results

Usage:
    from virtual_streamer.lib.agents import StatefulLlmAgent
    from virtual_streamer.lib.agents.dynamic_parallel_processor import MapperAgent
    
    # Define a worker factory
    def create_worker(run_id: str) -> StatefulLlmAgent:
        return get_video_matcher(run_id)
    
    # Create mapper with items to process
    items = [
        {"sentence": "Hello", "video_path": "/path/video1.mp4"},
        {"sentence": "World", "video_path": "/path/video2.mp4"},
    ]
    mapper = MapperAgent(items=items, worker_factory=create_worker)
"""

import json
import logging
import secrets
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Type, TypeVar

from google.adk.events import Event, EventActions
from google.adk.agents import BaseAgent, ParallelAgent
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


class MapperAgent(ParallelAgent):
    """
    Distributes tasks and dynamically creates workers for parallel processing.
    
    This agent takes a list of items and a worker factory, then:
    1. Generates a unique run_id for this batch
    2. Creates a worker for each item using the factory
    3. Writes each item to the worker's input key in state
    4. Runs all workers in parallel
    
    The worker factory receives a unique run_id (e.g., "a1b2:w0", "a1b2:w1")
    that the worker uses to namespace its state keys.
    
    After running, use get_output_keys() to get the state keys where
    workers wrote their results.
    
    Example:
        def create_matcher(run_id: str) -> StatefulLlmAgent:
            return get_video_matcher(run_id)
        
        items = [{"sentence": "test", "video_path": "/test.mp4"}]
        mapper = MapperAgent(items=items, worker_factory=create_matcher)
        
        # When run, mapper will:
        # 1. Create worker with run_id="a1b2:w0"
        # 2. Write {"sentence": ..., "video_path": ...} to "task:a1b2:w0:video_sentence"
        # 3. Run worker which reads from that key and writes to "result:a1b2:w0:judgement"
        
        # After running:
        output_keys = mapper.get_output_keys()  # ["result:a1b2:w0:judgement"]
    """
    
    # Track workers created during run
    _workers: List[StatefulWorker]

    def __init__(
        self,
        items: List[Dict[str, Any]],
        worker_factory: WorkerFactory,
        name: str = "mapper",
    ):
        """
        Initialize the mapper agent.
        
        Args:
            items: List of input items to distribute to workers.
                   Each item should be a dict matching the worker's input schema.
            worker_factory: Function that creates a StatefulWorker given a run_id.
            name: Name for this agent (default: "mapper")
        """
        super().__init__(name=name, sub_agents=[])
        # Use underscore prefix to bypass Pydantic's field validation
        self._items = items
        self._worker_factory = worker_factory
        self._workers = []

    async def _run_async_impl(self, ctx):
        """
        Distribute items to workers and run them in parallel.
        
        Yields events for:
        1. State delta with all input data written to worker keys
        2. Events from parallel worker execution
        
        Raises:
            ValidationError: If any item fails schema validation
        """
        # Generate unique run ID for this batch
        run_id = secrets.token_hex(2)

        # Create workers and prepare state delta
        # Store workers on instance so get_output_keys() can access them
        self._workers = []
        state_delta: Dict[str, str] = {"current_run": run_id}
        
        for i, item in enumerate(self._items):
            worker_run_id = f"{run_id}:w{i}"
            worker = self._worker_factory(worker_run_id)
            
            # Get the input key and schema from the worker (type-safe!)
            input_key = worker.get_input_key()
            input_schema = worker.get_input_schema()
            
            # Validate item against the worker's input schema
            validated_item = input_schema.model_validate(item)
            
            # Serialize validated item to JSON and store in state delta
            state_delta[input_key] = validated_item.model_dump_json()
            self._workers.append(worker)
            
            logger.debug(
                f"Validated item {i} for worker {worker_run_id}: "
                f"{input_schema.__name__}"
            )
        
        # Emit state delta with all inputs
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"Run {run_id}: distributing {len(self._items)} tasks")]
            ),
            actions=EventActions(state_delta=state_delta)
        )
        
        # Create parallel agent with workers and run
        parallel = ParallelAgent(
            name=f"parallel_{run_id}",
            sub_agents=self._workers  # type: ignore (workers are agents)
        )
        
        async for event in parallel.run_async(ctx):
            yield event
    
    def get_output_keys(self) -> List[str]:
        """
        Get output keys from all workers after run.
        
        Call this after running the mapper to get the state keys
        where workers wrote their results. These can be passed to
        an aggregator's state_input_keys.
        
        Returns:
            List of output state keys from workers.
            Empty list if run() hasn't been called yet.
        
        Example:
            mapper = MapperAgent(items=items, worker_factory=factory)
            async for event in mapper.run_async(ctx):
                pass
            
            output_keys = mapper.get_output_keys()
            aggregator = BestMatchAggregator(state_input_keys=output_keys, ...)
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


class AbstractAggregator(BaseAgent, Generic[T]):
    """
    Abstract base class for aggregating results from parallel workers.
    
    From the aggregator's perspective:
    - state_input_keys: where to read results from (worker outputs = aggregator inputs)
    - input_schema: Pydantic model to parse each result
    - aggregation_fn: subclass implements to combine parsed results
    
    The base class handles:
    - Collecting results from specified state keys
    - Parsing each result with the input schema
    - Storing aggregated result in state (optional)
    - Event emission
    
    Example:
        class BestMatchAggregator(AbstractAggregator[VideoJudgementOutput]):
            def __init__(self, input_keys: List[str]):
                super().__init__(
                    name="best_match",
                    state_input_keys=input_keys,
                    input_schema=VideoJudgementOutput,
                    result_state_key="best_video_match",
                )
            
            async def aggregation_fn(self, results: List[VideoJudgementOutput]):
                return max(results, key=lambda r: r.grade)
    """
    
    def __init__(
        self,
        name: str,
        state_input_keys: List[str],
        input_schema: Type[T],
        result_state_key: Optional[str] = None,
    ):
        """
        Initialize the aggregator.
        
        Args:
            name: Name for this agent
            state_input_keys: List of state keys to read results from.
                             These are typically the output keys from workers.
            input_schema: Pydantic model to parse each result into.
            result_state_key: Optional key to store aggregated result in state.
                             If None, result is only emitted in event.
        """
        super().__init__(name=name)
        # Use underscore prefix to bypass Pydantic's field validation
        self._state_input_keys = state_input_keys
        self._input_schema = input_schema
        self._result_state_key = result_state_key
    
    def _collect_results(self, ctx) -> List[T]:
        """
        Collect and parse results from state.
        
        Reads from state_input_keys, parses each with input_schema.
        
        Args:
            ctx: Invocation context with session state
        
        Returns:
            List of validated Pydantic model instances
        """
        results: List[T] = []
        
        for key in self._state_input_keys:
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
    
    async def _run_async_impl(self, ctx):
        """
        Collect results from workers and aggregate them.
        
        1. Collects and parses results from state_input_keys
        2. Calls aggregation_fn with parsed results
        3. Stores result in state if result_state_key is set
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
        
        logger.info(f"Aggregating {len(results)} results from {len(self._state_input_keys)} keys")
        
        # Aggregate results (subclass implements)
        aggregated = await self.aggregation_fn(results)
        
        # Prepare state delta if result key is specified
        state_delta: Dict[str, str] = {}
        if self._result_state_key and aggregated is not None:
            if isinstance(aggregated, BaseModel):
                state_delta[self._result_state_key] = aggregated.model_dump_json()
            else:
                state_delta[self._result_state_key] = json.dumps(aggregated)
            
            logger.debug(f"Stored aggregated result at {self._result_state_key}")
        
        # Emit completion event
        summary = f"Aggregated {len(results)} results"
        if self._result_state_key:
            summary += f" -> {self._result_state_key}"
        
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=summary)]
            ),
            actions=EventActions(
                state_delta=state_delta if state_delta else None,
                escalate=True
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
        
        Example:
            async def aggregation_fn(self, results: List[VideoJudgementOutput]):
                # Return the result with highest grade
                return max(results, key=lambda r: r.grade)
        """
        raise NotImplementedError()
