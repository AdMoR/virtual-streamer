"""
GPU Semaphore module for controlling concurrent GPU operations.

This module provides a global semaphore to prevent multiple GPU-intensive
operations (like Wav2Lip inference) from running simultaneously, which
could cause GPU memory overflow or degraded performance.

Usage:
    from virtual_streamer.api.utils.gpu_semaphore import run_on_gpu
    
    result = await run_on_gpu(blocking_gpu_function, arg1, arg2, kwarg=value)
"""

import asyncio
from typing import Callable, TypeVar
import os

T = TypeVar("T")

# Global GPU semaphore - limits concurrent GPU operations
_gpu_semaphore: asyncio.Semaphore | None = None


def get_gpu_semaphore() -> asyncio.Semaphore:
    """
    Get or create the global GPU semaphore.
    
    The semaphore limit can be configured via the MAX_CONCURRENT_GPU_OPS
    environment variable (defaults to 1 for single GPU).
    
    Returns:
        The global asyncio.Semaphore instance
    """
    global _gpu_semaphore
    if _gpu_semaphore is None:
        max_concurrent = int(os.environ.get("MAX_CONCURRENT_GPU_OPS", "1"))
        _gpu_semaphore = asyncio.Semaphore(max_concurrent)
        print(f"Initialized GPU semaphore with max_concurrent={max_concurrent}")
    return _gpu_semaphore


async def run_on_gpu(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a blocking GPU operation with concurrency control.
    
    This function:
    1. Acquires the GPU semaphore to prevent parallel GPU access
    2. Runs the blocking function in a thread pool to avoid blocking the event loop
    
    Args:
        func: The blocking function to run (e.g., wav2lip_exec)
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
        
    Example:
        # Instead of:
        result = blocking_gpu_function(arg1, arg2)
        
        # Use:
        result = await run_on_gpu(blocking_gpu_function, arg1, arg2)
    """
    semaphore = get_gpu_semaphore()
    async with semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)


def get_gpu_queue_status() -> dict:
    """
    Get the current status of the GPU semaphore.
    
    Returns:
        Dictionary with semaphore status information
    """
    semaphore = get_gpu_semaphore()
    return {
        "max_concurrent": int(os.environ.get("MAX_CONCURRENT_GPU_OPS", "1")),
        "available_slots": semaphore._value,
        "waiting_tasks": len(semaphore._waiters) if hasattr(semaphore, '_waiters') else 0,
    }

