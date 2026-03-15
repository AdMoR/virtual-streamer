"""
In-memory session store for Atari game sessions.

Two dicts:
  _sessions: dict[str, GameSession]  — guarded by asyncio.Lock
  _runtimes: dict[str, ALERuntime]   — written only by the game loop task, no lock needed
"""

import asyncio
from datetime import datetime
from typing import Optional

from virtual_streamer.streaming.video_server.atari.engine import (
    ALERuntime,
    GameSession,
    GameStatus,
)

_sessions: dict = {}
_runtimes: dict = {}
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

async def create_session(session_id: str, rom_name: str) -> GameSession:
    session = GameSession(
        session_id=session_id,
        rom_name=rom_name,
        status=GameStatus.RUNNING,
        current_frame=b"",
        created_at=datetime.utcnow(),
    )
    async with _lock:
        _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[GameSession]:
    return _sessions.get(session_id)


def update_frame(session_id: str, frame: bytes) -> None:
    session = _sessions.get(session_id)
    if session is not None:
        session.current_frame = frame


def set_status(session_id: str, status: str) -> None:
    session = _sessions.get(session_id)
    if session is not None:
        session.status = status


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def list_sessions() -> list:
    return list(_sessions.values())


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def set_runtime(session_id: str, runtime: ALERuntime) -> None:
    _runtimes[session_id] = runtime


def get_runtime(session_id: str) -> Optional[ALERuntime]:
    return _runtimes.get(session_id)


def delete_runtime(session_id: str) -> None:
    _runtimes.pop(session_id, None)
