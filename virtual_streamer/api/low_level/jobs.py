"""
Low-level API: Job management endpoints.
"""
import asyncio
import time
from typing import List

from fastapi import APIRouter, HTTPException, Query

from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.api.high_level.models import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])

_TERMINAL_STATUSES = {"completed", "failed"}


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a job by ID."""
    job_store = await get_global_job_store()
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/{job_id}/wait", response_model=JobStatusResponse)
async def wait_for_job(
    job_id: str,
    timeout: float = Query(default=600.0, le=3600.0, description="Max seconds to wait"),
    poll_interval: float = Query(default=2.0, ge=0.5, le=30.0),
):
    """
    Long-poll a job until it reaches a terminal state (completed/failed) or
    *timeout* elapses. One blocking request replaces client-side polling loops.
    Returns the job in whatever state it is in at timeout (check `status`).
    """
    job_store = await get_global_job_store()
    deadline = time.monotonic() + timeout
    while True:
        job = await job_store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in _TERMINAL_STATUSES or time.monotonic() >= deadline:
            return JobStatusResponse(**job)
        await asyncio.sleep(poll_interval)


@router.get("", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 20):
    """List recent jobs."""
    job_store = await get_global_job_store()
    jobs = await job_store.list_jobs(limit)
    return [JobStatusResponse(**job) for job in jobs]


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the tracking system (metadata only, not generated files)."""
    job_store = await get_global_job_store()
    deleted = await job_store.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}
