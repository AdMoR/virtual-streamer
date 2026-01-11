"""
Rubric Builder Agent.

A BaseLlmAgent that extracts rubrics from stories.
This is the base worker agent - for map-reduce processing, 
use rubric_builder_map_reduce instead.
"""

from virtual_streamer.agents.rubric_builder_agent.agent import (
    RubricBuilderAgent,
    get_rubric_builder,
    root_agent,
)
from virtual_streamer.agents.rubric_builder_agent.schema import (
    RubricExample,
    Rubric,
    MapPhaseOutput,
    ReducePhaseOutput,
    StoryItem,
    StoryBatchInput,
)

__all__ = [
    "RubricBuilderAgent",
    "get_rubric_builder",
    "root_agent",
    "RubricExample",
    "Rubric",
    "MapPhaseOutput",
    "ReducePhaseOutput",
    "StoryItem",
    "StoryBatchInput",
]
