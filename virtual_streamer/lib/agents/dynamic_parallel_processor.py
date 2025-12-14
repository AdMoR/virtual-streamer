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
import secrets
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Protocol, Optional

from google.adk.events import Event, EventActions
from google.adk.agents import BaseAgent, ParallelAgent
from google.genai import types


class StatefulWorker(Protocol):
    """Protocol for workers that expose input/output keys."""
    
    def get_input_key(self) -> str:
        """Return the state key where input should be written."""
        ...
    
    def get_output_key(self) -> str:
        """Return the state key where output will be written."""
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
    
    Example:
        def create_matcher(run_id: str) -> StatefulLlmAgent:
            return get_video_matcher(run_id)
        
        items = [{"sentence": "test", "video_path": "/test.mp4"}]
        mapper = MapperAgent(items=items, worker_factory=create_matcher)
        
        # When run, mapper will:
        # 1. Create worker with run_id="a1b2:w0"
        # 2. Write {"sentence": ..., "video_path": ...} to "task:a1b2:w0:video_sentence"
        # 3. Run worker which reads from that key and writes to "result:a1b2:w0:judgement"
    """

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
        self.items = items
        self.worker_factory = worker_factory

    async def _run_async_impl(self, ctx):
        """
        Distribute items to workers and run them in parallel.
        
        Yields events for:
        1. State delta with all input data written to worker keys
        2. Events from parallel worker execution
        """
        # Generate unique run ID for this batch
        run_id = secrets.token_hex(2)

        # Create workers and prepare state delta
        workers: List[StatefulWorker] = []
        state_delta: Dict[str, str] = {"current_run": run_id}
        
        for i, item in enumerate(self.items):
            worker_run_id = f"{run_id}:w{i}"
            worker = self.worker_factory(worker_run_id)
            
            # Get the input key from the worker (type-safe!)
            input_key = worker.get_input_key()
            
            # Serialize item to JSON and store in state delta
            state_delta[input_key] = json.dumps(item)
            workers.append(worker)
        
        # Emit state delta with all inputs
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"Run {run_id}: distributing {len(self.items)} tasks")]
            ),
            actions=EventActions(state_delta=state_delta)
        )
        
        # Create parallel agent with workers and run
        parallel = ParallelAgent(
            name=f"parallel_{run_id}",
            sub_agents=workers  # type: ignore (workers are agents)
        )
        
        async for event in parallel.run_async(ctx):
            yield event


class AbstractAggregator(BaseAgent):
    """
    Abstract base class for aggregating results from parallel workers.
    
    Subclasses must implement aggregation_fn to define how results
    are combined. The aggregator reads results from all worker output
    keys matching the current run.
    
    Example:
        class BestResultAggregator(AbstractAggregator):
            async def aggregation_fn(self, results: List[Any]) -> str:
                best = max(results, key=lambda r: r.get("grade", 0))
                return json.dumps(best)
    """
    
    def __init__(
        self,
        name: str = "aggregator",
        output_key_pattern: str = "result",
    ):
        """
        Initialize the aggregator.
        
        Args:
            name: Name for this agent
            output_key_pattern: Prefix pattern to match result keys
                               (e.g., "result" matches "result:a1b2:w0:...")
        """
        super().__init__(name=name)
        self.output_key_pattern = output_key_pattern
    
    async def _run_async_impl(self, ctx):
        """
        Collect results from workers and aggregate them.
        
        Reads all state keys matching "result:{current_run}:*"
        and passes values to aggregation_fn.
        """
        run_id = ctx.session.state.get("current_run")
        
        if not run_id:
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="No current_run found in state")]
                ),
                actions=EventActions(escalate=True)
            )
            return
        
        # Collect all results matching the pattern
        pattern = f"{self.output_key_pattern}:{run_id}:"
        results = []
        
        for key, value in ctx.session.state.items():
            if key.startswith(pattern):
                # Parse JSON if string, otherwise use as-is
                if isinstance(value, str):
                    try:
                        results.append(json.loads(value))
                    except json.JSONDecodeError:
                        results.append(value)
                else:
                    results.append(value)
        
        # Aggregate results
        aggregated = await self.aggregation_fn(results)
        
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=aggregated)]
            ),
            actions=EventActions(escalate=True)
        )

    @abstractmethod
    async def aggregation_fn(self, results: List[Any]) -> str:
        """
        Aggregate results from all workers.
        
        Args:
            results: List of parsed results from worker output keys
        
        Returns:
            Aggregated result as a string (can be JSON)
        
        Example:
            async def aggregation_fn(self, results):
                # Return the result with highest grade
                best = max(results, key=lambda r: r.get("grade", 0))
                return json.dumps(best)
        """
        raise NotImplementedError()
