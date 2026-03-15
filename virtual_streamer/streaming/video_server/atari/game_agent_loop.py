"""
game_agent_loop — single background task that runs ALE emulation + LLM agent.

Reads session.status to know when to stop.
Writes current_frame back to session store after each step.
"""

import asyncio
import logging
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.streaming.video_server.atari.action_mapper import action_name_to_int
from virtual_streamer.streaming.video_server.atari.engine import (
    GameStatus,
    ale_reset,
    ale_step,
    load_action_meanings,
)
from virtual_streamer.streaming.video_server.atari.session_store import (
    delete_runtime,
    get_runtime,
    get_session,
    set_status,
    update_frame,
)

logger = logging.getLogger(__name__)

_APP_NAME = "atari_game"


async def _run_agent(agent, user_message: str) -> str:
    """Run an ADK agent instance and return its text response."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    session = await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={},
    )

    content = types.Content(role="user", parts=[types.Part(text=user_message)])

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text

    return response_text


async def game_agent_loop(session_id: str) -> None:
    """
    Single background task: runs ALE emulation + LLM agent together.
    Reads session.status to know when to stop.
    Writes current_frame back to session store after each step.
    """
    from virtual_streamer.agents.atari_action_agent.agent import (
        AtariActionOutput,
        get_atari_action_agent,
    )

    runtime = get_runtime(session_id)
    if runtime is None:
        logger.error(f"[{session_id}] No runtime found, exiting loop.")
        return

    agent = get_atari_action_agent()

    # Load action meanings once from the live env
    action_meanings = load_action_meanings(runtime.ale)

    # First frame
    frame = await ale_reset(runtime)
    update_frame(session_id, frame)
    runtime.history.append((time.monotonic(), frame))

    while (
        get_session(session_id) is not None
        and get_session(session_id).status == GameStatus.RUNNING
    ):
        session = get_session(session_id)

        # Build text prompt
        prompt = (
            f"You are playing {session.rom_name}.\n"
            f"Legal actions: {', '.join(action_meanings)}\n"
            "Look at the frame and choose the best action."
        )

        # Inject current frame and call vision LLM
        agent.set_frame(frame)
        try:
            raw = await _run_agent(agent, prompt)
            output = AtariActionOutput.model_validate_json(raw)
            action_name = output.action
        except Exception as e:
            logger.warning(f"[{session_id}] Agent error: {e}. Using NOOP.")
            action_name = "NOOP"

        # Execute action in ALE (in executor thread)
        action_int = action_name_to_int(action_name, action_meanings)
        frame = await ale_step(runtime, action_int)

        # Write frame back to session store
        update_frame(session_id, frame)
        runtime.history.append((time.monotonic(), frame))

        # Game over from ALE
        if runtime.game_over:
            set_status(session_id, GameStatus.STOPPED)

    # Cleanup runtime
    runtime.executor.shutdown(wait=False)
    delete_runtime(session_id)
    logger.info(f"[{session_id}] Game agent loop exited.")
