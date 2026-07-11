"""
Tests for video_generation/seed_hunting.py — hunting loop, selection logic and
persistence (all generation/judging mocked; no network or DB).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from virtual_streamer.agents.video_judge.schema import JudgeArtifact, JudgeVerdict
from virtual_streamer.video_generation.scene_input import SceneInput
from virtual_streamer.video_generation.seed_hunting import (
    CandidateResult,
    SeedHuntConfig,
    hunt_segment,
    persist_candidates,
    select_best,
)


def make_scene(index: int = 0) -> SceneInput:
    return SceneInput(
        scene_index=index,
        ltx_prompt="A man walks toward the camera in a workshop",
        spoken_line="Bonjour Jamy!",
        raw_scene_data={},
    )


def make_segment(seed: int):
    seg = MagicMock()
    seg.video_path = f"/tmp/video_seed_{seed}.mp4"
    seg.prompt_id = f"prompt_{seed}"
    seg.duration_seconds = 5.0
    seg.minio_image_key = None
    seg.minio_video_key = None
    return seg


def verdict(passed: bool, score: float) -> JudgeVerdict:
    return JudgeVerdict(passed=passed, score=score, reasoning="test")


# ---------------------------------------------------------------------------
# SeedHuntConfig
# ---------------------------------------------------------------------------


def test_resolve_seeds_explicit_truncated_to_max():
    cfg = SeedHuntConfig(max_candidates=2, seeds=[1, 2, 3])
    assert cfg.resolve_seeds() == [1, 2]


def test_resolve_seeds_random_count():
    cfg = SeedHuntConfig(max_candidates=4)
    seeds = cfg.resolve_seeds()
    assert len(seeds) == 4
    assert all(0 <= s < 2**31 for s in seeds)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------


def _cand(seed, passed, score):
    return CandidateResult(
        candidate_id=f"c{seed}", seed=seed, segment=make_segment(seed),
        verdict=verdict(passed, score),
    )


def test_select_best_prefers_highest_passing():
    cands = [_cand(1, True, 6.5), _cand(2, False, 9.0), _cand(3, True, 8.0)]
    best = select_best(cands)
    assert best.seed == 3
    assert best.selected and best.selection_source == "judge"


def test_select_best_fallback_when_none_pass():
    cands = [_cand(1, False, 3.0), _cand(2, False, 4.5)]
    best = select_best(cands)
    assert best.seed == 2
    assert best.selection_source == "fallback"


# ---------------------------------------------------------------------------
# hunt_segment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hunt_early_stops_on_accepting_score():
    generate = AsyncMock(side_effect=lambda seed: make_segment(seed))
    judge = AsyncMock(return_value=verdict(True, 9.0))
    cfg = SeedHuntConfig(max_candidates=3, accept_score=7.5, seeds=[11, 22, 33])

    candidates = await hunt_segment(generate, make_scene(), cfg, judge_fn=judge)

    assert len(candidates) == 1          # early stop after first take
    assert candidates[0].selected
    generate.assert_awaited_once_with(seed=11)


@pytest.mark.asyncio
async def test_hunt_tries_all_seeds_and_selects_best():
    generate = AsyncMock(side_effect=lambda seed: make_segment(seed))
    judge = AsyncMock(side_effect=[verdict(True, 6.0), verdict(False, 2.0), verdict(True, 7.0)])
    cfg = SeedHuntConfig(max_candidates=3, accept_score=9.5, seeds=[1, 2, 3])

    candidates = await hunt_segment(generate, make_scene(), cfg, judge_fn=judge)

    assert len(candidates) == 3
    selected = [c for c in candidates if c.selected]
    assert len(selected) == 1 and selected[0].seed == 3


@pytest.mark.asyncio
async def test_hunt_skips_failed_generations():
    async def generate(seed):
        if seed == 1:
            raise RuntimeError("wangp down")
        return make_segment(seed)

    judge = AsyncMock(return_value=verdict(True, 8.0))
    cfg = SeedHuntConfig(max_candidates=2, accept_score=7.5, seeds=[1, 2])

    candidates = await hunt_segment(generate, make_scene(), cfg, judge_fn=judge)
    assert len(candidates) == 1 and candidates[0].seed == 2


@pytest.mark.asyncio
async def test_hunt_raises_when_all_generations_fail():
    generate = AsyncMock(side_effect=RuntimeError("boom"))
    cfg = SeedHuntConfig(max_candidates=2, seeds=[1, 2])
    with pytest.raises(RuntimeError, match="all 2 generation"):
        await hunt_segment(generate, make_scene(), cfg, judge_fn=AsyncMock())


# ---------------------------------------------------------------------------
# JudgeVerdict fail-open default
# ---------------------------------------------------------------------------


def test_permissive_default_never_blocks():
    v = JudgeVerdict.permissive_default("judge exploded")
    assert v.passed and v.score == 5.0
    assert v.judge_error == "judge exploded"


def test_verdict_schema_roundtrip():
    v = JudgeVerdict(
        passed=False,
        score=2.5,
        artifacts=[JudgeArtifact(
            category="unrealistic_movement",
            description="person runs backward",
            severity="blocking",
        )],
        reasoning="motion reversed",
    )
    assert JudgeVerdict.model_validate(v.model_dump()) == v


# ---------------------------------------------------------------------------
# persist_candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_candidates_uploads_and_inserts(tmp_path):
    seg = make_segment(42)
    video = tmp_path / "video_seed_42.mp4"
    video.write_bytes(b"fake")
    seg.video_path = str(video)

    cand = CandidateResult(
        candidate_id="cid", seed=42, segment=seg, verdict=verdict(True, 8.0),
        selected=True,
    )
    repo = MagicMock(create_candidate=AsyncMock())
    storage = MagicMock(upload_file=AsyncMock())

    await persist_candidates(repo, storage, "scene-uuid", [cand], "candidates/s/x")

    storage.upload_file.assert_awaited_once_with(str(video), "candidates/s/x/seed_42.mp4")
    kwargs = repo.create_candidate.await_args.kwargs
    assert kwargs["scene_id"] == "scene-uuid"
    assert kwargs["seed"] == 42
    assert kwargs["selected"] is True
    assert kwargs["judge_verdict"]["score"] == 8.0


@pytest.mark.asyncio
async def test_persist_candidates_swallows_errors():
    cand = CandidateResult(
        candidate_id="cid", seed=1, segment=make_segment(1), verdict=verdict(True, 8.0),
    )
    repo = MagicMock(create_candidate=AsyncMock(side_effect=RuntimeError("db down")))
    # Must not raise — persistence is best-effort
    await persist_candidates(repo, None, "scene-uuid", [cand], "p")
