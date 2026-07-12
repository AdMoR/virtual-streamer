"""
Low-level API: Generic evaluation bench.

Benches are datasets of cases per model kind (image/video/tts/llm_agent);
a run executes one bench against one model; humans review the samples in
apps/eval_bench.html and their labels are exported for model/judge tuning.

  GET  /eval/benches                      — all benches
  POST /eval/benches                      — create/update a bench
  POST /eval/benches/sync                 — load configs/eval_benches/*.yaml
  GET  /eval/benches/{bench_id}/runs      — runs with aggregate scores
  POST /eval/benches/{bench_id}/runs      — start a run (background job)
  GET  /eval/runs/{run_id}/samples        — samples with presigned artifact URLs
  POST /eval/samples/{sample_id}/feedback — human label on a sample
  GET  /eval/feedback/export              — labelled data for model/judge tuning
  POST /eval/backfill-candidates          — import seed-hunt takes (background job)
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from virtual_streamer.utils.eval_repository import get_eval_repository
from virtual_streamer.utils.job_store import get_global_job_store

router = APIRouter(tags=["Eval Bench"])

logger = logging.getLogger(__name__)


def _with_artifact_url(sample: dict) -> dict:
    """Attach a ready-to-use presigned URL so consumers never handle raw MinIO keys."""
    sample["artifact_url"] = None
    if sample.get("artifact_key"):
        try:
            from virtual_streamer.utils.minio_client import get_storage_client
            sample["artifact_url"] = get_storage_client().get_url(sample["artifact_key"])
        except Exception as exc:
            logger.warning(f"presign failed for {sample.get('artifact_key')}: {exc}")
    return sample


class BenchRequest(BaseModel):
    name: str
    model_kind: str = Field(description="'image', 'video', 'tts' or 'llm_agent'")
    cases: List[dict] = Field(description="[{case_id, label, params}]")
    description: Optional[str] = None
    judge_agent: Optional[str] = Field(
        default=None, description="Judge agent for auto-scoring (e.g. 'video_judge')"
    )


class RunRequest(BaseModel):
    model_id: str = Field(
        description="image: client impl · video: preset · tts: character_id · "
                    "llm_agent: config name under configs/agents/"
    )
    model_config_overrides: dict = Field(
        default_factory=dict, alias="model_config", description="Runner-specific config"
    )
    label: Optional[str] = None
    case_ids: Optional[List[str]] = Field(
        default=None, description="Subset of cases to run; all when null"
    )

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class FeedbackRequest(BaseModel):
    user: str = Field(default="anonymous")
    passed: Optional[bool] = Field(default=None, description="Is this generation usable?")
    score: Optional[float] = Field(default=None, ge=0, le=10)
    tags: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    preferred_over_sample_id: Optional[str] = Field(
        default=None, description="A/B: this sample beats that one"
    )


class BackfillRequest(BaseModel):
    story_id: Optional[str] = Field(default=None, description="One story; all recent when null")
    limit: int = Field(default=50, ge=1, le=500)


class EvalJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    run_id: Optional[str] = None


# ── Benches ───────────────────────────────────────────────────────────────────


@router.get("/eval/benches", response_model=List[dict])
async def list_benches():
    """All benches, grouped client-side by model_kind."""
    return await get_eval_repository().list_benches()


@router.post("/eval/benches", response_model=dict)
async def upsert_bench(request: BenchRequest):
    """Create a bench, or replace the cases of an existing bench with the same name."""
    repo = get_eval_repository()
    existing = await repo.get_bench_by_name(request.name)
    return await repo.upsert_bench(
        bench_id=existing["bench_id"] if existing else str(uuid.uuid4()),
        name=request.name,
        model_kind=request.model_kind,
        cases=request.cases,
        description=request.description,
        judge_agent=request.judge_agent,
    )


@router.post("/eval/benches/sync", response_model=List[dict])
async def sync_benches():
    """Load/refresh the YAML bench definitions from configs/eval_benches/."""
    from virtual_streamer.eval_bench.harness import sync_benches_from_configs
    return await sync_benches_from_configs()


# ── Runs ──────────────────────────────────────────────────────────────────────


@router.get("/eval/benches/{bench_id}/runs", response_model=List[dict])
async def list_runs(bench_id: str):
    """Runs of a bench with aggregate auto/human scores — the model-comparison view."""
    repo = get_eval_repository()
    if await repo.get_bench(bench_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bench not found")
    return await repo.list_runs_for_bench(bench_id)


async def _run_bench_job(job_id: str, run_id: str, bench_id: str, request: RunRequest):
    from virtual_streamer.eval_bench.harness import run_bench

    job_store = await get_global_job_store()
    repo = get_eval_repository()
    try:
        await job_store.update_job(job_id, status="running")
        summary = await run_bench(
            run_id=run_id,
            bench_id=bench_id,
            model_id=request.model_id,
            model_config=request.model_config_overrides,
            case_ids=request.case_ids,
        )
        await job_store.update_job(job_id, status="completed", result=summary)
    except Exception as exc:
        logger.error(f"[eval-run {job_id}] failed: {exc}", exc_info=True)
        await repo.finish_run(run_id, "failed", error=str(exc))
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post("/eval/benches/{bench_id}/runs", response_model=EvalJobResponse)
async def start_run(bench_id: str, request: RunRequest, background_tasks: BackgroundTasks):
    """Execute the bench against one model as a background job. Poll GET /jobs/{job_id}."""
    repo = get_eval_repository()
    bench = await repo.get_bench(bench_id)
    if bench is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bench not found")

    run_id = str(uuid.uuid4())
    await repo.create_run(
        run_id=run_id,
        bench_id=bench_id,
        model_id=request.model_id,
        model_config=request.model_config_overrides,
        label=request.label,
    )
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(
        job_id, {"bench_id": bench_id, "run_id": run_id, "pipeline": "eval-bench-run"}
    )
    background_tasks.add_task(_run_bench_job, job_id, run_id, bench_id, request)
    return EvalJobResponse(
        job_id=job_id, run_id=run_id, status="pending",
        message=f"Eval run started on bench '{bench['name']}'",
    )


# ── Samples & feedback ────────────────────────────────────────────────────────


@router.get("/eval/runs/{run_id}/samples", response_model=List[dict])
async def list_samples(run_id: str):
    """Samples of a run, each with a presigned artifact_url and its feedback."""
    repo = get_eval_repository()
    if await repo.get_run(run_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    samples = [_with_artifact_url(s) for s in await repo.list_samples_for_run(run_id)]
    feedback = await repo.list_feedback_for_run(run_id)
    by_sample = {}
    for fb in feedback:
        by_sample.setdefault(fb["sample_id"], []).append(fb)
    for s in samples:
        s["feedback"] = by_sample.get(s["sample_id"], [])
    return samples


@router.post("/eval/samples/{sample_id}/feedback", response_model=dict)
async def submit_feedback(sample_id: str, request: FeedbackRequest):
    """Record a human label on a sample."""
    repo = get_eval_repository()
    if await repo.get_sample(sample_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found")
    return await repo.create_feedback(
        feedback_id=str(uuid.uuid4()),
        sample_id=sample_id,
        user=request.user,
        passed=request.passed,
        score=request.score,
        tags=request.tags,
        comment=request.comment,
        preferred_over_sample_id=request.preferred_over_sample_id,
    )


@router.get("/eval/feedback/export", response_model=List[dict])
async def export_feedback(bench_id: Optional[str] = None, limit: int = 1000):
    """Export human labels with full sample/run/bench context — tuning data."""
    return await get_eval_repository().export_feedback(bench_id=bench_id, limit=limit)


# ── Backfill ──────────────────────────────────────────────────────────────────


async def _run_backfill_job(job_id: str, request: BackfillRequest):
    from virtual_streamer.eval_bench.backfill import backfill_candidates

    job_store = await get_global_job_store()
    try:
        await job_store.update_job(job_id, status="running")
        summary = await backfill_candidates(story_id=request.story_id, limit=request.limit)
        await job_store.update_job(job_id, status="completed", result=summary)
    except Exception as exc:
        logger.error(f"[eval-backfill {job_id}] failed: {exc}", exc_info=True)
        await job_store.update_job(job_id, status="failed", error=str(exc))


@router.post("/eval/backfill-candidates", response_model=EvalJobResponse)
async def backfill(request: BackfillRequest, background_tasks: BackgroundTasks):
    """Import existing seed-hunt candidates + judge feedback as eval benches (idempotent)."""
    job_store = await get_global_job_store()
    job_id = str(uuid.uuid4())
    await job_store.create_job(job_id, {"pipeline": "eval-backfill"})
    background_tasks.add_task(_run_backfill_job, job_id, request)
    return EvalJobResponse(job_id=job_id, status="pending", message="Backfill job submitted")
