# Seed Hunting, LLM Judge & Review — Technical Design

Goal: one bad LTX generation must never ruin a video. Every scene is generated
multiple times (seed hunting), each take is judged by a local vision LLM, every
decision is persisted and overridable, and the final video can be recomposed at
any time. All of it is drivable by humans (HTML app) and agents (MCP + vsctl).

## Components

```
scene loop (story_to_video.story_input_to_video)
   └─ seed_hunting.hunt_segment(generate_fn, scene_input, SeedHuntConfig)
        ├─ generate_fn(seed)          → WanGP LTX segment (existing client)
        ├─ run_video_judge(video, …)  → JudgeVerdict (local Qwen3-VL, ADK agent)
        ├─ early stop at accept_score
        └─ select_best (passed → max score; else fallback)
   └─ seed_hunting.persist_candidates → MinIO + segment_candidates rows
```

### Judge (`virtual_streamer/agents/video_judge/`)
- `BaseLlmAgent` + `InjectCandidateFramesCallback` (8 evenly-spaced JPEG frames
  via `extract_evenly_spaced_frames`) + `StoreVerdictCallback` (JSON → `JudgeVerdict`).
- Model: `configs/agents/video_judge.yaml` → local llama.cpp OpenAI endpoint
  (same host as `scene_enrichment_agent`).
- Targets: unrealistic_movement, impossible_body, object_intersection,
  impossible_setting, identity_drift, visual_glitch.
- Fail-open: any judge error yields `JudgeVerdict.permissive_default` — the
  pipeline is never blocked by the judge.

### Persistence (`utils/story_repository.py`)
- `segment_candidates(candidate_id, scene_id, seed, video_key, generation_params,
  judge_verdict JSON, judge_score, judge_passed, selected, selection_source
  ENUM(judge|human|fallback), duration_seconds)`
- `judge_feedback(feedback_id, candidate_id, user, human_passed, human_score,
  artifact_tags JSON, comment)` — human preference labels for judge improvement.
- Candidate videos: `candidates/{story_id}/{scene_id}/seed_{seed}.mp4` in MinIO.

### API
- `low_level/candidates.py`: list/get candidates, `POST /candidates/{id}/select`
  (human override, records implicit positive label), `POST /candidates/{id}/feedback`,
  `GET /judge-feedback/export` (judge-vs-human training data).
- `medium_level/review.py`:
  - `POST /stories/{id}/recompose` — concat currently selected takes → new final
    video, updates `stories.final_video_key`. Background job.
  - `POST /stories/{id}/scenes/{scene_id}/regenerate` — fresh seed hunt for one
    scene, reusing the persisted conditioning image and speaker voice sample.
- Request knobs on `VideoGenerationRequest` / `VideoFromScriptRequest`:
  `enable_seed_hunt`, `seed_hunt_max_candidates`, `seed_hunt_accept_score`,
  `seed_hunt_seeds` (explicit seeds ⇒ exact replay).

### Control surfaces
- `apps/seed_review.html` — review UX (candidates, override, feedback,
  regenerate, recompose).
- MCP tools in `mcp_server/server.py` (list_stories … export_judge_feedback).
- `scripts/vsctl.py` — OpenAPI-discovering CLI (`ops`/`describe`/`call`) so an
  agent can enumerate and invoke every endpoint without prior knowledge.
- Skill: `.claude/skills/video-review/SKILL.md`.

## Replayability
- Exact seed replay: pass `seed_hunt_seeds=[…]` (seeds are stored per candidate).
- Scene replay: `regenerate` rebuilds `SceneInput` from `scenes.raw_scene_data`.
- Composition replay: recompose result includes the full composition manifest
  (scene → candidate_id/seed/selection_source).

## Judge improvement loop
1. Humans label candidates in the review app (verdict, score, artifact tags).
2. `GET /api/v1/judge-feedback/export` joins labels with judge verdicts.
3. Disagreement rows drive prompt/model iteration on the judge; explicit-seed
   replay lets you re-judge the same videos with a new prompt.
