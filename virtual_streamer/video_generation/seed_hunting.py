"""
Seed hunting for LTX video segments.

Generates up to N candidate takes of one scene with distinct seeds, judges each
with the local vision-LLM VideoJudgeAgent, and selects the best take. Every
candidate is kept (MinIO + segment_candidates table) so a human can override
the judge's choice afterwards and the final video can be recomposed.

Flow per scene:
    for seed in seeds:
        segment = generate_segment_from_input(seed=seed)
        verdict = run_video_judge(segment.video_path, scene_description)
        persist candidate (best-effort)
        if verdict.score >= accept_score: stop early
    select best candidate (highest score among passed; falls back to highest
    score overall with selection_source="fallback")
"""

import logging
import os
import random
import uuid
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from virtual_streamer.agents.video_judge.schema import JudgeVerdict
from virtual_streamer.video_generation.scene_input import SceneInput

logger = logging.getLogger(__name__)


class SeedHuntConfig(BaseModel):
    """Configuration for per-scene seed hunting."""

    enabled: bool = True
    max_candidates: int = Field(default=3, ge=1, le=10)
    accept_score: float = Field(
        default=7.5,
        description="Stop hunting as soon as a candidate reaches this judge score",
    )
    seeds: Optional[List[int]] = Field(
        default=None,
        description="Explicit seeds to try (e.g. for replay). Random seeds when None.",
    )
    judge_frames: int = Field(default=8, description="Frames sampled per judgement")

    def resolve_seeds(self) -> List[int]:
        if self.seeds:
            return self.seeds[: self.max_candidates]
        return [random.randint(0, 2**31 - 1) for _ in range(self.max_candidates)]


class CandidateResult(BaseModel):
    """One generated + judged candidate take of a scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate_id: str
    seed: int
    segment: Any  # SegmentResult (avoids circular import with story_to_video)
    verdict: Optional[JudgeVerdict] = None
    selected: bool = False
    selection_source: str = "judge"

    @property
    def score(self) -> float:
        return self.verdict.score if self.verdict else 5.0

    @property
    def passed(self) -> bool:
        return self.verdict.passed if self.verdict else True


def select_best(candidates: List[CandidateResult]) -> CandidateResult:
    """Pick the best candidate: highest score among passing takes, else highest overall."""
    passing = [c for c in candidates if c.passed]
    if passing:
        best = max(passing, key=lambda c: c.score)
        best.selection_source = "judge"
    else:
        best = max(candidates, key=lambda c: c.score)
        best.selection_source = "fallback"
        logger.warning(
            f"[seed-hunt] No candidate passed the judge — falling back to best score "
            f"({best.score:.1f}, seed={best.seed})"
        )
    best.selected = True
    return best


async def hunt_segment(
    generate_fn: Callable[..., Any],
    scene_input: SceneInput,
    hunt_config: SeedHuntConfig,
    judge_fn: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[CandidateResult]:
    """
    Generate and judge up to max_candidates takes of one scene.

    Args:
        generate_fn: async callable(seed: int) -> SegmentResult. The caller binds
            client/params/audio/image so this module stays decoupled.
        scene_input: The scene being generated (for the judge's context).
        hunt_config: Hunting parameters.
        judge_fn: async callable(video_path, scene_description) -> JudgeVerdict.
            Defaults to run_video_judge. Injectable for tests.
        progress: Optional callback(message) for status reporting.

    Returns:
        All candidates (at least one, unless every generation attempt raised).
        Exactly one has selected=True. Raises only if ALL generations failed.
    """
    if judge_fn is None:
        from virtual_streamer.agents.video_judge.agent import run_video_judge
        judge_fn = run_video_judge

    scene_description = scene_input.ltx_prompt
    if scene_input.spoken_line:
        scene_description += f'\nSpoken line: "{scene_input.spoken_line}"'

    candidates: List[CandidateResult] = []
    errors: List[str] = []

    for attempt, seed in enumerate(hunt_config.resolve_seeds()):
        label = f"scene {scene_input.scene_index} take {attempt + 1}/{hunt_config.max_candidates} (seed={seed})"
        if progress:
            progress(f"Generating {label}")
        try:
            segment = await generate_fn(seed=seed)
        except Exception as exc:
            logger.warning(f"[seed-hunt] Generation failed for {label}: {exc}")
            errors.append(f"seed {seed}: {exc}")
            continue

        if progress:
            progress(f"Judging {label}")
        verdict = await judge_fn(segment.video_path, scene_description)
        candidate = CandidateResult(
            candidate_id=str(uuid.uuid4()),
            seed=seed,
            segment=segment,
            verdict=verdict,
        )
        candidates.append(candidate)
        logger.info(
            f"[seed-hunt] {label}: passed={verdict.passed} score={verdict.score:.1f} "
            f"artifacts={[a.category for a in verdict.artifacts]}"
        )

        if verdict.passed and verdict.score >= hunt_config.accept_score:
            logger.info(f"[seed-hunt] Early accept at score {verdict.score:.1f} ({label})")
            break

    if not candidates:
        raise RuntimeError(
            f"Seed hunt: all {hunt_config.max_candidates} generation(s) failed for "
            f"scene {scene_input.scene_index}: {'; '.join(errors)}"
        )

    select_best(candidates)
    return candidates


async def persist_candidates(
    story_repo: Any,
    storage: Any,
    db_scene_id: str,
    candidates: List[CandidateResult],
    minio_prefix: str,
) -> None:
    """
    Upload every candidate video to MinIO and insert segment_candidates rows.
    Best-effort: logs but never raises (observability must not break generation).
    """
    for cand in candidates:
        try:
            video_key = None
            if storage and cand.segment.video_path and os.path.exists(cand.segment.video_path):
                video_key = f"{minio_prefix}/seed_{cand.seed}.mp4"
                await storage.upload_file(cand.segment.video_path, video_key)
            cand.segment.minio_video_key = video_key if cand.selected else cand.segment.minio_video_key

            if story_repo:
                await story_repo.create_candidate(
                    candidate_id=cand.candidate_id,
                    scene_id=db_scene_id,
                    seed=cand.seed,
                    generation_params={
                        "prompt_id": cand.segment.prompt_id,
                        "duration_seconds": cand.segment.duration_seconds,
                    },
                    video_key=video_key,
                    image_key=cand.segment.minio_image_key,
                    judge_verdict=cand.verdict.model_dump() if cand.verdict else None,
                    duration_seconds=cand.segment.duration_seconds,
                    selected=cand.selected,
                    selection_source=cand.selection_source,
                )
        except Exception as exc:
            logger.warning(
                f"[seed-hunt] Failed to persist candidate seed={cand.seed}: {exc}"
            )
