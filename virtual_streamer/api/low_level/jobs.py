"""
Low-level API: Job management endpoints.
"""
from typing import List

from fastapi import APIRouter, HTTPException

from virtual_streamer.utils.job_store import get_global_job_store
from virtual_streamer.api.high_level.models import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a job by ID."""
    job_store = await get_global_job_store()
    job = await job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


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
