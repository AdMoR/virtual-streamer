import logging
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext

from virtual_streamer.agents.common.state_keys import TITLE, STORY_TEMPLATE_ID
from virtual_streamer.lib.providers.instruction import InstructionProvider

logger = logging.getLogger(__name__)


PROMPT = """You are an expert story analyzer.

You are responsible for identifying the Criteria (also called rubrics) that uniquely define the set of story that are provided to you. 

What you will receive : 
- A set of stories coming from the same source 
- This set of stories will come from the user message

Your task : 
- Identify patterns
- Transcribe these pattern into rubrics : explain what the pattern is, give examples matching this pattern and examples not matching it
- The pattern should apply to the provided stories provided in the user message
- Be sure to also identify the tricks **not used** by the stories compared to other similar stories from your personal knowledge

Example of rubric description : 
"L'article ménage des surprises, rebondissements et révélations qui relancent l'intérêt et décalent la perspective initiale. Ces chutes, souvent en fin d'article, amplifient l'effet comique et marquent durablement le lecteur.",

Example of rubric grading illustration : 
Rubric :  Détournement par l'absurde de l'actualité économique",
example: "Attendre le Black Friday pour acheter des fruits de base comme les clémentines",
Grading: True
Explanation : This example is based on the common belief that the economical trends are getting harsher and people struggle to afford basic necessities. 
But also use the common news on Black Friday to convey this idea in an absurd manner where the only way to afford fruit would be through a once a year occasion of the black friday.

Rubric :  Détournement par l'absurde de l'actualité économique",
example: "Le premier ministre Lionel Jospin propose d'augmenter le SMIC de 10%",
Grading: False
Explanation : While this statement is unlikely, it is not absurd nor exaggerated. It will not surprise the reader and lacks a submersive twist.


Please write all rubrics in english despite the example in French. You are allowed to use quotes in other languages.
"""


class StoryInstructionProvider(InstructionProvider):
    """
    """

    async def __call__(self, ctx: ReadonlyContext) -> str:
        """
        """

        return PROMPT
