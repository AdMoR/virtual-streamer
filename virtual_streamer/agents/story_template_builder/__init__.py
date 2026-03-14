"""
Story Template Builder Agent.

Three-step sequential pipeline:
1. guardrail         — validates the user's story concept
2. TemplateWriterAgent    — generates a rich free-text template (characters fetched from API)
3. TemplateFormatterAgent — extracts structured StoryTemplateOutput (name, prompt, target_lines)
"""

from virtual_streamer.agents.story_template_builder.agent import (
    StoryTemplateBuilderAgent,
    TemplateWriterAgent,
    TemplateFormatterAgent,
    get_story_template_builder,
    get_template_builder_guardrail_agent,
)
from virtual_streamer.agents.story_template_builder.schema import StoryTemplateOutput

__all__ = [
    "StoryTemplateBuilderAgent",
    "TemplateWriterAgent",
    "TemplateFormatterAgent",
    "get_story_template_builder",
    "get_template_builder_guardrail_agent",
    "StoryTemplateOutput",
]