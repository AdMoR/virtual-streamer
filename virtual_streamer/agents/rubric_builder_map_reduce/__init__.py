"""
Rubric Builder Map-Reduce Agent.

Extracts rubrics from stories using a map-reduce pattern:
- Map: Process batches of 5 stories in parallel
- Reduce: Aggregate rubrics and write to JSONL file
"""

from virtual_streamer.agents.rubric_builder_map_reduce.agent import (
    create_rubric_builder_map_reduce,
    RubricMapper,
    RubricAggregator,
    get_stateful_rubric_builder,
    STORIES_KEY,
)

__all__ = [
    "create_rubric_builder_map_reduce",
    "RubricMapper",
    "RubricAggregator",
    "get_stateful_rubric_builder",
    "STORIES_KEY",
]
