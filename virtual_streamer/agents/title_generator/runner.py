"""Utility to run title generator and parse output."""
import uuid
import logging
from typing import List

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.agents.title_generator.agent import get_title_generator
from virtual_streamer.agents.title_generator.schema import TitlesOutput
from virtual_streamer.agents.common.state_keys import STORY_TEMPLATE_ID

logger = logging.getLogger(__name__)


async def run_title_generator(story_template_id: str, count: int = 50) -> List[str]:
    """
    Run TitleGeneratorAgent and return list of titles.

    Args:
        story_template_id: Template to generate titles for
        count: Number of titles to generate

    Returns:
        List of generated title strings
    """
    agent = get_title_generator(count=count)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="title_generator",
        session_service=session_service,
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"titles_{uuid.uuid4().hex[:8]}"

    session = await session_service.create_session(
        app_name="title_generator",
        user_id=user_id,
        session_id=session_id,
        state={STORY_TEMPLATE_ID: story_template_id},
    )

    content = types.Content(
        role="user", parts=[types.Part(text=f"Generate {count} titles")]
    )

    # Run agent
    logger.info(f"Running TitleGeneratorAgent for template: {story_template_id}")
    final_response = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response() and event.content:
            final_response = event.content

    # Parse JSON output from final response
    if final_response and final_response.parts:
        json_text = final_response.parts[0].text
        titles_output = TitlesOutput.model_validate_json(json_text)
        logger.info(f"Generated {len(titles_output.titles)} titles")
        return titles_output.titles

    raise RuntimeError("No response from TitleGeneratorAgent")
