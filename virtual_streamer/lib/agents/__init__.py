"""
Agent base classes for Google ADK.
"""

from virtual_streamer.lib.agents.base import BaseLlmAgent
from virtual_streamer.lib.agents.callbacks import (
    AfterModelCallback,
    BeforeModelCallback,
    AgentCallback,
    extract_llm_response_json,
    extract_llm_response_text,
)
from virtual_streamer.lib.agents.stateful_callbacks import (
    StateInputCallback,
    StateOutputCallback,
)
from virtual_streamer.lib.agents.stateful_agent import StatefulLlmAgent
from virtual_streamer.lib.agents.dynamic_parallel_processor import (
    MapperAgent,
    AggregatorAgent,
    MapReduceAgent,
    StatefulWorker,
    WorkerFactory,
    AggregatorFactory,
    # Backward compatibility
    AbstractAggregator,
)

__all__ = [
    # Base agents
    "BaseLlmAgent",
    "StatefulLlmAgent",
    # Map-reduce agents
    "MapperAgent",
    "AggregatorAgent",
    "MapReduceAgent",
    "AbstractAggregator",  # Backward compatibility alias
    # Protocols and types
    "StatefulWorker",
    "WorkerFactory",
    "AggregatorFactory",
    # Callbacks
    "AfterModelCallback",
    "BeforeModelCallback",
    "AgentCallback",
    "StateInputCallback",
    "StateOutputCallback",
    # Utilities
    "extract_llm_response_json",
    "extract_llm_response_text",
]

