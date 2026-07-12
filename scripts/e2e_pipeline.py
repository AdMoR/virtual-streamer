#!/usr/bin/env python3
"""
End-to-end pipeline compatibility check — ONE real micro-generation.

Verifies the internal pipeline contract against the REAL backends (WanGP LTX,
Stable Diffusion, local judge LLM, MySQL, MinIO) with the cheapest possible
generation: a single synthetic scene, one seed, 9 frames at 512x288.

Stages exercised (calling internal functions directly — no API server needed):
    SceneInput → conditioning image (SD) → generate_segment_from_input (WanGP)
    → seed hunt + video judge (local vision LLM, verdict must parse cleanly)
    → candidate persistence (MinIO + segment_candidates)
    → human-override + feedback + export roundtrip
    → recompose (_run_recompose: download → concat → upload → story update)

Exit code 0 = every contract holds. Any assertion failure or exception = 1.

Runtime: ~2-4 min (dominated by the LTX clip). Run from the repo root:
    uv run python scripts/e2e_pipeline.py

Used as a pre-commit hook gated on pipeline-contract files — SKIP with:
    SKIP=e2e-pipeline git commit ...
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

# Load .env so the script works standalone like the API does
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(".env.public")
except ImportError:
    pass

TEMPLATE_ID = "e2e-test"
STORY_TITLE = "E2E pipeline check"
LTX_URL = os.environ.get("LTX_SERVER_URL", "http://gx10-cbc5:8082")
SD_URL = os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def _ffprobe_ok(path: str) -> bool:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", path],
            capture_output=True, timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _preflight() -> list[str]:
    """Fast reachability check of all real backends; returns actionable errors."""
    import socket
    from urllib.parse import urlparse

    problems = []

    def tcp_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=4):
                return True
        except OSError:
            return False

    targets = {
        f"MySQL ({os.environ.get('MYSQL_HOST', 'localhost')}:{os.environ.get('MYSQL_PORT', '3306')})":
            (os.environ.get("MYSQL_HOST", "localhost"), int(os.environ.get("MYSQL_PORT", "3306")),
             "start it with: docker compose up -d mysql"),
        "MinIO": (urlparse(os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")).hostname or "localhost",
                  urlparse(os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")).port or 9000,
                  "start it with: docker compose up -d minio"),
        f"WanGP LTX ({LTX_URL})": (urlparse(LTX_URL).hostname, urlparse(LTX_URL).port or 80,
                                   "is the GPU box up / on the tailnet?"),
        f"Stable Diffusion ({SD_URL})": (urlparse(SD_URL).hostname, urlparse(SD_URL).port or 80,
                                         "is the SD server running on the GPU box?"),
    }
    for label, (host, port, hint) in targets.items():
        if not tcp_open(host, port):
            problems.append(f"{label} unreachable — {hint}")
    return problems


async def main() -> int:
    problems = _preflight()
    if problems:
        print("E2E preflight failed — required backends unreachable:")
        for p in problems:
            print(f"  ✗ {p}")
        print("\nFix the above (or bypass knowingly with: SKIP=e2e-pipeline git commit ...)")
        return 1

    from virtual_streamer.agents.video_judge.schema import JudgeVerdict
    from virtual_streamer.api.medium_level.review import RecomposeRequest, _run_recompose
    from virtual_streamer.utils.entity_repository import get_entity_repository
    from virtual_streamer.utils.job_store import get_global_job_store
    from virtual_streamer.utils.minio_client import get_storage_client
    from virtual_streamer.utils.story_repository import get_story_repository
    from virtual_streamer.video_generation.ltx_client import (
        LTXVideoConfig,
        VideoGenerationParams,
    )
    from virtual_streamer.video_generation.scene_input import SceneInput, StoryInput
    from virtual_streamer.video_generation.seed_hunting import SeedHuntConfig
    from virtual_streamer.video_generation.story_to_video import story_input_to_video

    t0 = time.time()
    entity_repo = get_entity_repository()
    story_repo = get_story_repository()
    storage = get_storage_client()

    # ── Fixture: dedicated template + story row (satisfies FK constraints) ──
    print("[1/6] Fixture setup (MySQL)")
    if not await entity_repo.get_story_template(TEMPLATE_ID):
        await entity_repo.create_story_template(
            template_id=TEMPLATE_ID,
            name="E2E pipeline test template",
            prompt="internal e2e compatibility check — do not use for content",
            collection="e2e",
            target_lines=1,
        )
    story_id = str(uuid.uuid4())
    await story_repo.create_story(
        story_id=story_id,
        story_template_id=TEMPLATE_ID,
        title=STORY_TITLE,
        story_plan="",
        raw_agent_output={"e2e": True},
        status="GENERATING",
    )
    check("template + story rows created", True, story_id)

    output_dir = tempfile.mkdtemp(prefix="e2e_pipeline_")
    try:
        # ── Synthetic single scene (fixed seed → reproducible) ──────────────
        scene = SceneInput(
            scene_index=0,
            ltx_prompt=(
                "A single red ceramic mug on a wooden table, gentle steam rising, "
                "static camera, soft daylight, photorealistic"
            ),
            spoken_line=None,  # no speaker → i2v mode (no voice sample dependency)
            scene_visual_description={
                "scene": "A quiet wooden table by a window",
                "subjects": [{
                    "description": "A single red ceramic mug with steam rising",
                    "position": "center foreground on the wooden table",
                }],
                "lighting": "Soft natural daylight from the left window",
            },
            raw_scene_data={"e2e": True},
        )
        story_input = StoryInput(
            title=STORY_TITLE,
            story_plan="",
            story_template_id=TEMPLATE_ID,
            raw_agent_output={"e2e": True},
            scenes=[scene],
        )

        # ── Real generation: SD image + LTX clip + judge + persistence ──────
        print(f"[2/6] Micro-generation (SD @ {SD_URL}, LTX @ {LTX_URL}) — this takes minutes")
        result = await story_input_to_video(
            story_input=story_input,
            ltx_config=LTXVideoConfig(server_url=LTX_URL, timeout=900.0),
            video_params=VideoGenerationParams.from_preset(
                "fast", resolution="512x288", video_length=9,
            ),
            output_dir=output_dir,
            sd_server_url=SD_URL,
            story_repo=story_repo,
            db_story_id=story_id,
            seed_hunt_config=SeedHuntConfig(max_candidates=1, accept_score=0.0, seeds=[1234]),
        )

        check("final video produced", os.path.getsize(result.final_video_path) > 0,
              result.final_video_path)
        check("final video ffprobe-parseable", _ffprobe_ok(result.final_video_path))
        check("one segment generated", len(result.segments) == 1)
        seg = result.segments[0]
        check("segment linked to DB scene", bool(seg.db_scene_id), str(seg.db_scene_id))

        # ── Candidate persistence + judge verdict contract ───────────────────
        print("[3/6] Candidate + judge verdict contract")
        candidates = await story_repo.list_candidates_for_scene(seg.db_scene_id)
        check("exactly one candidate persisted", len(candidates) == 1)
        cand = candidates[0]
        check("candidate seed recorded", cand["seed"] == 1234)
        check("candidate selected", cand["selected"] is True)
        verdict = JudgeVerdict.model_validate(cand["judge_verdict"])
        check("judge verdict parses as JudgeVerdict",
              True, f"passed={verdict.passed} score={verdict.score}")
        check("judge answered for real (no permissive fallback)",
              verdict.judge_error is None, verdict.judge_error or "")

        # ── Candidate video retrievable from MinIO ───────────────────────────
        local_cand = os.path.join(output_dir, "candidate_check.mp4")
        await storage.download_file(cand["video_key"], local_cand)
        check("candidate video downloadable from MinIO",
              os.path.getsize(local_cand) > 0, cand["video_key"])

        # ── Review loop: override + feedback + export ────────────────────────
        print("[4/6] Review-loop contract (override, feedback, export)")
        sel = await story_repo.set_selected_candidate(cand["candidate_id"], "human")
        check("human override applied", sel["selection_source"] == "human")
        fb_id = str(uuid.uuid4())
        await story_repo.create_judge_feedback(
            feedback_id=fb_id,
            candidate_id=cand["candidate_id"],
            user="e2e",
            human_passed=True,
            human_score=8.0,
            artifact_tags=["other"],
            comment="e2e roundtrip",
        )
        export = await story_repo.export_judge_feedback(limit=50)
        check("feedback visible in export",
              any(r["feedback_id"] == fb_id for r in export))

        # ── Recompose from the selected candidate ────────────────────────────
        print("[5/6] Recompose contract")
        job_store = await get_global_job_store()
        job_id = str(uuid.uuid4())
        await job_store.create_job(job_id, {"e2e": True})
        await _run_recompose(job_id, story_id, RecomposeRequest(enable_subtitles=False))
        job = await job_store.get_job(job_id)
        check("recompose job completed", job["status"] == "completed",
              job.get("error") or "")
        if job["status"] == "completed":
            final_key = job["result"]["final_video_key"]
            local_final = os.path.join(output_dir, "recomposed_check.mp4")
            await storage.download_file(final_key, local_final)
            check("recomposed video downloadable + parseable",
                  os.path.getsize(local_final) > 0 and _ffprobe_ok(local_final), final_key)
            story = await story_repo.get_story(story_id)
            check("story final_video_key updated", story["final_video_key"] == final_key)

    finally:
        # ── Cleanup: DB cascade + MinIO objects + local temp ─────────────────
        print("[6/6] Cleanup")
        try:
            candidates_prefix = f"candidates/{story_id}/"
            for key in await storage.list_objects(candidates_prefix):
                await storage.delete_object(key)
            story = await story_repo.get_story(story_id)
            if story and story.get("final_video_key"):
                await storage.delete_object(story["final_video_key"])
            await story_repo.delete_story(story_id)
            print("  cleaned DB story cascade + MinIO test objects")
        except Exception as exc:
            print(f"  cleanup warning (non-fatal): {exc}")
        shutil.rmtree(output_dir, ignore_errors=True)

    elapsed = time.time() - t0
    if _failures:
        print(f"\nE2E FAILED in {elapsed:.0f}s — broken contracts: {', '.join(_failures)}")
        return 1
    print(f"\nE2E OK in {elapsed:.0f}s — pipeline contract intact")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
