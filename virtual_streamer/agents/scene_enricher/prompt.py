"""Prompts for SceneEnricherPipeline."""

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import (
    ENRICHMENT_VIDEO_DESCRIPTION,
    ENRICHMENT_SCENE_TEXT,
)

DESCRIBE_PROMPT = "Describe precisely the action of this video."

_ENRICHMENT_TEMPLATE = """\
Here are 4 images from a video. The video shows: {description}

Your goal is to enrich a text describing a small story to the action flow of this video. \
The result should be much more detailed than the initial text thanks to the input.
The movement description should match the original video whose 4 images are taken from, \
but be adapted to the SCENE described below.
Description of the scene and action has to be in English. Dialogues must be kept in the \
language provided in input.
Keep the description and dialogue relative ordering, your goal is to complete the text \
where it makes sense.
Do not add comment or notes, just the updated text

[SCENE]
{scene_text}"""


class SceneEnrichmentInstructionProvider(InstructionProvider):
    """
    Builds the enrichment prompt for SceneEnrichmentAgent by reading
    the video description (from Agent 1) and scene text from state.
    """

    async def __call__(self, context: ReadonlyContext) -> str:
        description = context.state.get(ENRICHMENT_VIDEO_DESCRIPTION, "")
        scene_text = context.state.get(ENRICHMENT_SCENE_TEXT, "")
        return _ENRICHMENT_TEMPLATE.format(
            description=description,
            scene_text=scene_text,
        )
