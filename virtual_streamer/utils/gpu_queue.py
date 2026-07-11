"""
Single-worker priority queue for GPU-bound generation jobs.

The WanGP / Stable Diffusion servers are single-queue GPU services: running
several multi-scene jobs concurrently interleaves their segments and inflates
every job's latency. This module serializes all GPU-bound background jobs
through one FIFO worker with priorities, so:

  - full-video jobs never interleave,
  - small interactive jobs (scene regeneration, single clips) jump ahead of
    queued full videos — they are what a human is actively waiting on.

Usage (replaces FastAPI BackgroundTasks for GPU jobs):

    from virtual_streamer.utils.gpu_queue import enqueue_gpu_job, PRIORITY_INTERACTIVE

    await enqueue_gpu_job(job_id, lambda: _run_video_generation(job_id, request))
    await enqueue_gpu_job(job_id, lambda: _run_regenerate(...), priority=PRIORITY_INTERACTIVE)

Jobs stay in the job store as "pending" until the worker picks them up; the
job coroutine itself is responsible for flipping status to running/completed/
failed (all existing job runners already do).
"""

import asyncio
import itertools
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Lower value = served first
PRIORITY_INTERACTIVE = 0   # scene regeneration, single clips
PRIORITY_BATCH = 10        # full story-to-video jobs

_queue: Optional[asyncio.PriorityQueue] = None
_worker_task: Optional[asyncio.Task] = None
_counter = itertools.count()  # tie-breaker keeps FIFO order within a priority


async def _worker() -> None:
    logger.info("[gpu-queue] worker started")
    while True:
        priority, _, job_id, coro_factory = await _queue.get()
        logger.info(f"[gpu-queue] starting job {job_id} (priority={priority}, "
                    f"{_queue.qsize()} still queued)")
        try:
            await coro_factory()
        except Exception as exc:
            # Job runners handle their own failures; this guards the worker itself.
            logger.error(f"[gpu-queue] job {job_id} raised: {exc}", exc_info=True)
        finally:
            _queue.task_done()
            logger.info(f"[gpu-queue] finished job {job_id}")


def _ensure_worker() -> None:
    global _queue, _worker_task
    if _queue is None:
        _queue = asyncio.PriorityQueue()
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.get_running_loop().create_task(_worker())


async def enqueue_gpu_job(
    job_id: str,
    coro_factory: Callable[[], Awaitable[None]],
    priority: int = PRIORITY_BATCH,
) -> int:
    """
    Enqueue a GPU-bound job. Returns the queue position (0 = will run next).

    coro_factory is called (and awaited) only when the worker reaches the job,
    so no coroutine is left un-awaited while queued.
    """
    _ensure_worker()
    position = _queue.qsize()
    await _queue.put((priority, next(_counter), job_id, coro_factory))
    logger.info(f"[gpu-queue] enqueued job {job_id} at position {position} (priority={priority})")
    return position


def queue_depth() -> int:
    """Number of jobs currently waiting (excluding the one running)."""
    return _queue.qsize() if _queue else 0
