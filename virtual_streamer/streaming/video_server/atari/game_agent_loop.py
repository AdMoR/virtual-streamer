"""
game_agent_loop — single background task that runs ALE emulation + LLM agent.

Reads session.status to know when to stop.
Writes current_frame back to session store after each step.

Uses AtariGameAgent (sequential: planning → action).  The planning sub-agent
runs only at the start of each session window; the action sub-agent runs every
move using the stored plan.

The ADK session is reset every PLAN_INTERVAL moves so that:
  - conversation history stays bounded
  - the planning agent is triggered at the start of each new window
    (PLANNING_NEEDED=True is set in the initial session state)
"""

import logging
import time
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from virtual_streamer.agents.atari_game_agent import (
    PLAN_INTERVAL,
    SESSION_KEY_PLANNING_NEEDED,
    get_atari_game_agent,
)
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

_APP_NAME = "atari_game_agent"


async def _new_adk_session(session_service: InMemorySessionService, user_id: str):
    """Create a fresh ADK session with PLANNING_NEEDED=True so the planning
    sub-agent fires on the very first move of this window."""
    return await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=f"session_{uuid.uuid4().hex[:8]}",
        state={SESSION_KEY_PLANNING_NEEDED: True},
    )


async def _run_agent_step(
    runner: Runner,
    user_id: str,
    adk_session_id: str,
    user_message: str,
) -> str:
    """Run one agent step on an existing ADK session and return the text response."""
    content = types.Content(role="user", parts=[types.Part(text=user_message)])
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=adk_session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
    return response_text


async def game_agent_loop(session_id: str, plan_interval: int = PLAN_INTERVAL) -> None:
    """
    Single background task: runs ALE emulation + LLM agent together.
    Reads session.status to know when to stop.
    Writes current_frame back to session store after each step.

    Args:
        plan_interval: Number of moves per planning window. The planning
            sub-agent runs at the start of each window; the ADK session is
            reset at the same cadence to bound context size.
    """
    from virtual_streamer.agents.atari_action_agent.agent import AtariActionOutput

    runtime = get_runtime(session_id)
    if runtime is None:
        logger.error(f"[{session_id}] No runtime found, exiting loop.")
        return

    agent = get_atari_game_agent(plan_interval=plan_interval)

    # Set up a single Runner + session service for the whole game loop
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    # Initial session: PLANNING_NEEDED=True so planning fires on first move
    adk_session = await _new_adk_session(session_service, user_id)

    # Load action meanings once from the live env
    action_meanings = load_action_meanings(runtime.ale)

    # First frame
    frame = await ale_reset(runtime)
    update_frame(session_id, frame)
    runtime.history.append((time.monotonic(), frame))

    move_count = 0

    while (
        get_session(session_id) is not None
        and get_session(session_id).status == GameStatus.RUNNING
    ):
        # Reset ADK session every plan_interval moves.
        # New session gets PLANNING_NEEDED=True, triggering re-planning.
        if move_count > 0 and move_count % plan_interval == 0:
            adk_session = await _new_adk_session(session_service, user_id)
            logger.info(f"[{session_id}] ADK session reset at move {move_count} (new plan window)")

        session = get_session(session_id)

        # Build text prompt (action meanings for the action sub-agent)
        prompt = (
            f"You are playing {session.rom_name}.\n"
            f"Legal actions: {', '.join(action_meanings)}\n"
            "Look at the frame and choose the best action."
        )

        # Inject current frame into both sub-agents and run the sequential agent
        agent.set_frame(frame)
        try:
            raw = await _run_agent_step(runner, user_id, adk_session.id, prompt)
            output = AtariActionOutput.model_validate({"action": raw})
            action_name = output.action
            logger.warning(f"[{session_id}] move={move_count} action={action_name}")
        except Exception as e:
            logger.warning(f"[{session_id}] move={move_count} Agent error: {e}. Using NOOP.")
            action_name = "NOOP"

        # Execute action in ALE (in executor thread)
        action_int = action_name_to_int(action_name, action_meanings)
        frame = await ale_step(runtime, action_int)
        move_count += 1

        # Write frame back to session store
        update_frame(session_id, frame)
        runtime.history.append((time.monotonic(), frame))

        # Game over from ALE
        if runtime.game_over:
            set_status(session_id, GameStatus.STOPPED)

    # Cleanup runtime
    runtime.executor.shutdown(wait=False)
    delete_runtime(session_id)
    logger.info(f"[{session_id}] Game agent loop exited after {move_count} moves.")