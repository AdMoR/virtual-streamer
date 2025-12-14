import random, secrets
from abc import abstractmethod
from typing import ClassVar, List
from google.adk.events import Event, EventActions
from google.adk.agents import BaseAgent, ParallelAgent, SequentialAgent
from google.genai import types


class MapperAgent(ParallelAgent):
    """Distributes tasks and dynamically creates a ParallelAgent."""

    def __init__(self, items: list, mapper_cls: type):
        super().__init__(sub_agents=[])
        self.items = items
        self.mapper_cls = mapper_cls

    async def _run_async_impl(self, ctx):
        run_id = secrets.token_hex(2)

        task_delta = {f"task:{run_id}:w{i}": item
                      for i, item in enumerate(self.items)}
        yield Event(
            author=self.name,
            content=types.Content(role=self.name,
                   parts=[types.Part(text=f"Run {run_id} tasks {task_delta}")]),
            actions=EventActions(state_delta={"current_run": run_id, **task_delta})
        )
        parallel = ParallelAgent(
            name=f"mapper_{run_id}",
            sub_agents=[self.mapper_cls(run_id=run_id) for _ in self.items]
        )
        async for ev in parallel.run_async(ctx):
            yield ev

class AbstractAggregator(BaseAgent):
    """Aggregates results from workers."""
    async def _run_async_impl(self, ctx):
        run_id = ctx.session.state.get("current_run")
        vals = [v for k, v in ctx.session.state.items()
                if run_id and k.startswith(f"result:{run_id}:")]
        result = await self.aggregation_fn(vals)
        yield Event(
            author=self.name,
            content=types.Content(role=self.name,
                   parts=[types.Part(text=result)]),
            actions=EventActions(escalate=True)
        )

    @abstractmethod
    async def aggregation_fn(self, ctx):
        raise NotImplementedError()

