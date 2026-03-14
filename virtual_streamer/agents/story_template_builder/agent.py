"""
Story Template Builder Agent.

A three-step sequential pipeline:
1. guardrail         — validates the user's story concept (reuses GuardrailAgent)
2. template_writer   — generates a rich free-text story template, with characters
                       fetched from the API injected into the prompt
3. template_formatter — extracts the structured StoryTemplateOutput fields

State flow:
    Input:  title (str) — the user's story concept
    Step 1: guardrail           → SECURITY_FLAG
    Step 2: template_writer     → RAW_TEMPLATE_TEXT (str)
    Step 3: template_formatter  → TEMPLATE_OUTPUT (dict with name, prompt, target_lines)

This agent is responsible only for the creative parts of the template.
Fields like collection and character_ids are handled by the API layer.
"""

import logging

from google.adk.agents import SequentialAgent

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.guardrails_agent.agent import GuardrailAgent
from virtual_streamer.agents.story_generator.callback import SafetyFlagCheckerCallback
from virtual_streamer.agents.story_template_builder.schema import StoryTemplateOutput
from virtual_streamer.agents.story_template_builder.prompt import (
    TemplateWriterInstructionProvider,
    TemplateFormatterInstructionProvider,
    WRITER_PROMPT,
)
from virtual_streamer.agents.story_template_builder.callback import (
    StoreRawTemplateCallback,
    StoreTemplateOutputCallback,
)

logger = logging.getLogger(__name__)

# Context string for the guardrail: describe what this agent is used for
_GUARDRAIL_CONTEXT = WRITER_PROMPT.replace("{", "[").replace("}", "]")


def get_template_builder_guardrail_agent() -> BaseLlmAgent:
    """Guardrail agent scoped to the story template builder."""
    return GuardrailAgent("StoryTemplateBuilder", _GUARDRAIL_CONTEXT)


class TemplateWriterAgent(BaseLlmAgent):
    """
    Step 2: generates a rich free-text story template from the user's concept.

    Fetches available characters from the API and injects them alongside the
    user's story idea into the writer prompt. Outputs plain text stored under
    RAW_TEMPLATE_TEXT by StoreRawTemplateCallback.
    """

    def __init__(self):
        super().__init__(
            name="template_writer",
            instruction=TemplateWriterInstructionProvider(),
            output_schema=None,
            before_agent_callback=[SafetyFlagCheckerCallback()],
            after_model_callback=[StoreRawTemplateCallback()],
        )


class TemplateFormatterAgent(BaseLlmAgent):
    """
    Step 3: extracts structured fields from the raw template text.

    Reads RAW_TEMPLATE_TEXT from state and formats it into StoryTemplateOutput
    (name, prompt, target_lines). StoreTemplateOutputCallback then persists the
    result under TEMPLATE_OUTPUT.
    """

    def __init__(self):
        super().__init__(
            name="template_formatter",
            instruction=TemplateFormatterInstructionProvider(),
            output_schema=StoryTemplateOutput,
            after_model_callback=[StoreTemplateOutputCallback()],
        )


class StoryTemplateBuilderAgent(SequentialAgent):
    """
    Sequential agent: guardrail → template_writer → template_formatter.

    State flow:
        Input:  title (str) in session state
        Step 1: guardrail        → SECURITY_FLAG
        Step 2: template_writer  → RAW_TEMPLATE_TEXT (str)
        Step 3: template_formatter → TEMPLATE_OUTPUT (dict)
    """

    def __init__(self):
        super().__init__(
            name="story_template_builder",
            sub_agents=[
                get_template_builder_guardrail_agent(),
                TemplateWriterAgent(),
                TemplateFormatterAgent(),
            ],
        )


def get_story_template_builder() -> StoryTemplateBuilderAgent:
    """Factory function returning a configured StoryTemplateBuilderAgent."""
    return StoryTemplateBuilderAgent()


root_agent = get_story_template_builder()