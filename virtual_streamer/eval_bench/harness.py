"""
Evaluation bench harness.

run_bench() executes every case of a bench against one model sequentially
(single-GPU safety), uploads artifacts to MinIO under eval/{run_id}/ and
persists one eval_sample per case. Per-case failures are recorded on the
sample; the run continues. Meant to be executed as a background job.

sync_benches_from_configs() upserts the YAML bench definitions in
configs/eval_benches/ into the database (keyed by bench name).
"""

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

import yaml

from virtual_streamer.eval_bench.runners import RUNNERS, judge_artifact
from virtual_streamer.utils.eval_repository import MODEL_KINDS, get_eval_repository
from virtual_streamer.utils.minio_client import get_storage_client

logger = logging.getLogger(__name__)

BENCH_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval_benches"


async def run_bench(
    run_id: str,
    bench_id: str,
    model_id: str,
    model_config: Optional[dict] = None,
    case_ids: Optional[List[str]] = None,
) -> dict:
    """Execute a bench run (the eval_runs row must already exist). Returns a summary."""
    repo = get_eval_repository()
    storage = get_storage_client()
    bench = await repo.get_bench(bench_id)
    if bench is None:
        raise RuntimeError(f"Bench {bench_id} not found")
    run_case = RUNNERS[bench["model_kind"]]
    model_config = model_config or {}

    cases = bench["cases"]
    if case_ids:
        cases = [c for c in cases if c["case_id"] in case_ids]

    ok, failed = 0, 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for case in cases:
            sample_id = str(uuid.uuid4())
            case_params = case.get("params", {})
            started = time.monotonic()
            try:
                output = await run_case(case_params, model_id, model_config, tmpdir)
                latency = time.monotonic() - started

                artifact_key = None
                if output.artifact_path:
                    ext = Path(output.artifact_path).suffix.lstrip(".") or "bin"
                    artifact_key = f"eval/{run_id}/{case['case_id']}_{sample_id}.{ext}"
                    await storage.upload_file(output.artifact_path, artifact_key)

                auto_score, auto_verdict = None, None
                if bench.get("judge_agent") and output.artifact_path:
                    auto_verdict = await judge_artifact(
                        bench["judge_agent"], output.artifact_path, case_params
                    )
                    if auto_verdict:
                        auto_score = auto_verdict.get("score")

                output_json = output.output_json
                if output.metadata:
                    output_json = {**(output_json or {}), "_meta": output.metadata}

                await repo.create_sample(
                    sample_id=sample_id,
                    run_id=run_id,
                    case_id=case["case_id"],
                    input_params=case_params,
                    artifact_key=artifact_key,
                    output_json=output_json,
                    auto_score=auto_score,
                    auto_verdict=auto_verdict,
                    status="ok",
                    latency_seconds=latency,
                )
                ok += 1
            except Exception as exc:
                logger.error(
                    f"[run_bench {run_id}] case {case['case_id']} failed: {exc}",
                    exc_info=True,
                )
                await repo.create_sample(
                    sample_id=sample_id,
                    run_id=run_id,
                    case_id=case["case_id"],
                    input_params=case_params,
                    status="error",
                    error=str(exc),
                    latency_seconds=time.monotonic() - started,
                )
                failed += 1

    status = "completed" if ok else "failed"
    await repo.finish_run(run_id, status, error=None if ok else "all cases failed")
    return {"run_id": run_id, "status": status, "ok": ok, "failed": failed}


async def sync_benches_from_configs(config_dir: Optional[str] = None) -> List[dict]:
    """Upsert every configs/eval_benches/*.yaml into the eval_benches table."""
    repo = get_eval_repository()
    directory = Path(config_dir) if config_dir else BENCH_CONFIG_DIR
    synced = []
    for path in sorted(directory.glob("*.yaml")):
        with open(path) as f:
            spec = yaml.safe_load(f)
        if spec.get("model_kind") not in MODEL_KINDS:
            raise ValueError(f"{path.name}: model_kind must be one of {MODEL_KINDS}")
        existing = await repo.get_bench_by_name(spec["name"])
        bench = await repo.upsert_bench(
            bench_id=existing["bench_id"] if existing else str(uuid.uuid4()),
            name=spec["name"],
            model_kind=spec["model_kind"],
            cases=spec["cases"],
            description=spec.get("description"),
            judge_agent=spec.get("judge_agent"),
        )
        synced.append(bench)
    return synced
