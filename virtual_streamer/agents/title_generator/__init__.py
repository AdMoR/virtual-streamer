"""
Title Generator Agent.

Generates creative titles for a story template using LLM.
"""

from virtual_streamer.agents.title_generator.agent import (
    TitleGeneratorAgent,
    get_title_generator,
)
from virtual_streamer.agents.title_generator.schema import TitlesOutput
from virtual_streamer.agents.title_generator.runner import run_title_generator

__all__ = [
    "TitleGeneratorAgent",
    "get_title_generator",
    "TitlesOutput",
    "run_title_generator",
]
