"""
ALE engine: async wrappers around gymnasium Atari env for use in the game agent loop.
"""

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GameStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class GameSession:
    session_id: str
    rom_name: str
    status: str          # use GameStatus values
    current_frame: bytes  # JPEG, updated by loop on each step; None until first frame
    created_at: datetime


@dataclass
class ALERuntime:
    ale: Any                              # gymnasium env
    executor: ThreadPoolExecutor          # single-thread per session
    history: deque                        # deque[tuple[float, bytes]], maxlen=120
    game_over: bool = False               # updated by ale_step / ale_reset


def _reset_env(env: Any):
    obs, info = env.reset()
    return obs


def _step_env(env: Any, action: int):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, bool(terminated or truncated)


def _encode_frame(rgb: np.ndarray) -> bytes:
    """Upscale to 640×420 with nearest-neighbour (crisp pixels), JPEG quality=90. Returns bytes."""
    img = cv2.resize(rgb, (640, 420), interpolation=cv2.INTER_NEAREST)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


async def ale_reset(runtime: ALERuntime) -> bytes:
    """Reset the env in its executor thread; return first frame as JPEG."""
    loop = asyncio.get_event_loop()
    rgb = await loop.run_in_executor(runtime.executor, _reset_env, runtime.ale)
    runtime.game_over = False
    return _encode_frame(rgb)


async def ale_step(runtime: ALERuntime, action: int) -> bytes:
    """Execute action in executor thread; return new frame as JPEG."""
    loop = asyncio.get_event_loop()
    rgb, done = await loop.run_in_executor(
        runtime.executor, _step_env, runtime.ale, action
    )
    runtime.game_over = done
    return _encode_frame(rgb)


def load_action_meanings(ale: Any) -> list:
    """Return list of action name strings for the loaded ROM (gymnasium env)."""
    return list(ale.unwrapped.get_action_meanings())
