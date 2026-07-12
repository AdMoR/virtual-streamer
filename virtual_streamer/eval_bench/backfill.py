"""
Backfill importer: seed-hunt candidates → eval bench.

Imports every story's segment_candidates (and their judge_feedback human
labels) into the generic eval tables, so existing video takes become the
first reviewable benchmark. Idempotent: candidates already imported (matched
via eval_samples.candidate_id) are skipped.

Mapping per story:
  bench  "video/seed-hunt: {title}"  (model_kind=video, cases = scenes)
  run    one per story ("imported takes")
  sample one per candidate (video_key → artifact_key, judge verdict → auto_*)
  feedback: judge_feedback rows copied as eval_feedback
"""

import logging
import uuid
from typing import Optional

from virtual_streamer.utils.eval_repository import get_eval_repository
from virtual_streamer.utils.story_repository import get_story_repository

logger = logging.getLogger(__name__)


async def backfill_candidates(story_id: Optional[str] = None, limit: int = 50) -> dict:
    """Import candidates of one story (or the *limit* most recent stories)."""
    story_repo = get_story_repository()
    eval_repo = get_eval_repository()

    if story_id:
        story = await story_repo.get_story(story_id)
        stories = [story] if story else []
    else:
        stories = await story_repo.list_stories(limit=limit)

    imported_benches, imported_samples, imported_feedback, skipped = 0, 0, 0, 0
    for story in stories:
        scenes = await story_repo.list_scenes_for_story(story["story_id"])
        scene_candidates = {
            s["scene_id"]: await story_repo.list_candidates_for_scene(s["scene_id"])
            for s in scenes
        }
        to_import = {
            scene_id: [
                c for c in cands
                if not await eval_repo.sample_exists_for_candidate(c["candidate_id"])
            ]
            for scene_id, cands in scene_candidates.items()
        }
        skipped += sum(len(c) for c in scene_candidates.values()) - sum(
            len(c) for c in to_import.values()
        )
        if not any(to_import.values()):
            continue

        bench_name = f"video/seed-hunt: {story['title'][:200]}"
        existing = await eval_repo.get_bench_by_name(bench_name)
        bench = await eval_repo.upsert_bench(
            bench_id=existing["bench_id"] if existing else str(uuid.uuid4()),
            name=bench_name,
            model_kind="video",
            description=f"Imported seed-hunt takes of story {story['story_id']}",
            judge_agent="video_judge",
            cases=[
                {
                    "case_id": s["scene_id"],
                    "label": f"Scene {s['scene_index']}: {(s.get('spoken_line') or s['prompt'])[:80]}",
                    "params": {"prompt": s["prompt"], "spoken_line": s.get("spoken_line")},
                }
                for s in scenes
            ],
        )
        imported_benches += 1

        run_id = str(uuid.uuid4())
        await eval_repo.create_run(
            run_id=run_id,
            bench_id=bench["bench_id"],
            model_id="ltx",
            model_config={"imported_from_story": story["story_id"]},
            label="imported takes",
        )

        # Human labels are keyed by candidate in judge_feedback; re-fetch via export
        # would be global, so copy labels per candidate from the export join instead.
        all_feedback = await story_repo.export_judge_feedback(limit=10000)
        feedback_by_candidate = {}
        for fb in all_feedback:
            feedback_by_candidate.setdefault(fb["candidate_id"], []).append(fb)

        for scene in scenes:
            for cand in to_import.get(scene["scene_id"], []):
                sample_id = str(uuid.uuid4())
                await eval_repo.create_sample(
                    sample_id=sample_id,
                    run_id=run_id,
                    case_id=scene["scene_id"],
                    input_params={
                        "prompt": scene["prompt"],
                        "seed": cand["seed"],
                        **(cand.get("generation_params") or {}),
                    },
                    artifact_key=cand.get("video_key"),
                    auto_score=cand.get("judge_score"),
                    auto_verdict=cand.get("judge_verdict"),
                    candidate_id=cand["candidate_id"],
                    status="ok" if cand.get("video_key") else "error",
                    error=None if cand.get("video_key") else "candidate has no video",
                    latency_seconds=None,
                )
                imported_samples += 1

                for fb in feedback_by_candidate.get(cand["candidate_id"], []):
                    await eval_repo.create_feedback(
                        feedback_id=str(uuid.uuid4()),
                        sample_id=sample_id,
                        user=fb["user"],
                        passed=fb["human_passed"],
                        score=fb["human_score"],
                        tags=fb["artifact_tags"],
                        comment=fb["comment"],
                    )
                    imported_feedback += 1

        await eval_repo.finish_run(run_id, "completed")

    return {
        "benches": imported_benches,
        "samples": imported_samples,
        "feedback": imported_feedback,
        "skipped_existing": skipped,
    }
