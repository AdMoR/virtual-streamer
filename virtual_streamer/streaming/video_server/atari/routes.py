"""
FastAPI router for Atari game session management.

Endpoints:
  POST   /api/sessions               — create session + start game loop
  DELETE /api/sessions/{id}          — stop session
  GET    /api/sessions               — list all sessions
  GET    /api/sessions/{id}/state    — session state
  GET    /api/sessions/{id}/frame    — current JPEG frame
  GET    /api/sessions/{id}/history  — last 30 s of frames as base64
"""

import asyncio
import base64
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from virtual_streamer.streaming.video_server.atari.engine import (
    ALERuntime,
    GameStatus,
    load_action_meanings,
)
from virtual_streamer.streaming.video_server.atari.session_store import (
    create_session,
    delete_session,
    get_runtime,
    get_session,
    list_sessions,
    set_runtime,
    set_status,
)
from virtual_streamer.streaming.video_server.atari.game_agent_loop import game_agent_loop

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["Atari Sessions"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    rom_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_env(rom_name: str):
    """Create a gymnasium Atari env (blocking — run in executor)."""
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    return gym.make(rom_name, render_mode="rgb_array")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
async def create_game_session(req: CreateSessionRequest):
    """
    Create a new game session and immediately start the agent loop.

    Returns session_id, rom_name, status, and the list of legal action names.
    """
    session_id = uuid4().hex[:8]
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ale_{session_id}")

    # Init the gymnasium env in the executor thread (blocks I/O)
    loop = asyncio.get_event_loop()
    logger.info("Creating new game session {} : {}".format(session_id, req.rom_name))
    try:
        env = await loop.run_in_executor(executor, _init_env, req.rom_name)
    except Exception as e:
        executor.shutdown(wait=False)
        raise HTTPException(status_code=400, detail=f"Failed to load ROM '{req.rom_name}': {e}")

    action_meanings = load_action_meanings(env)

    runtime = ALERuntime(
        ale=env,
        executor=executor,
        history=deque(maxlen=120),
    )

    await create_session(session_id, req.rom_name)
    set_runtime(session_id, runtime)

    asyncio.create_task(game_agent_loop(session_id))

    return {
        "session_id": session_id,
        "rom_name": req.rom_name,
        "status": GameStatus.RUNNING,
        "action_meanings": action_meanings,
    }


@router.delete("/{session_id}")
async def stop_game_session(session_id: str):
    """
    Stop a running session (sets status=stopped, loop exits on next iteration).
    If the session is already stopped, delete it from the store entirely.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status == GameStatus.RUNNING:
        set_status(session_id, GameStatus.STOPPED)
        return {"status": "stopped"}
    else:
        delete_session(session_id)
        return {"status": "deleted"}


@router.get("")
async def list_game_sessions():
    """List all sessions."""
    sessions = list_sessions()
    return [
        {
            "session_id": s.session_id,
            "rom_name": s.rom_name,
            "status": s.status,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/{session_id}/state")
async def get_session_state(session_id: str):
    """Get session metadata (no frame data)."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "rom_name": session.rom_name,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
    }


@router.get("/{session_id}/frame")
async def get_current_frame(session_id: str):
    """Return the latest game frame as JPEG."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.current_frame:
        raise HTTPException(status_code=204, detail="Frame not yet available")
    return Response(content=session.current_frame, media_type="image/jpeg")


@router.get("/{session_id}/history")
async def get_frame_history(session_id: str):
    """
    Return up to the last 30 seconds of frames as base64-encoded JPEGs.
    """
    runtime = get_runtime(session_id)
    if runtime is None:
        # Session may have stopped and runtime cleaned up
        return []

    cutoff = time.monotonic() - 30.0
    filtered = [(t, f) for t, f in runtime.history if t >= cutoff]

    return [
        {"timestamp": t, "frame": base64.b64encode(f).decode()}
        for t, f in filtered
    ]
