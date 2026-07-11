---
name: video-review
description: Operate the resilient video generation pipeline — seed hunting with LLM-judged takes, reviewing candidates, overriding the judge, regenerating scenes, and recomposing the final video. Use when asked to generate a video, fix a bad scene/take, review generation quality, or collect judge feedback.
---

# Video Review & Seed Hunting Workflow

For the full end-to-end generation recipe via the CLI (prerequisite checks,
template/script creation, regeneration cases) see the `vsctl-e2e-video` skill.

## Concepts

- A **story** is a list of **scenes**; each scene becomes one LTX-2 video segment.
- With **seed hunting** enabled (default), each scene is generated up to N times
  with distinct seeds. Every take is a **candidate**, judged by a local vision
  LLM (`video_judge` agent, Qwen3-VL via `configs/agents/video_judge.yaml`) for:
  unrealistic movements, impossible bodies, object intersections, impossible
  settings, identity drift, visual glitches. Verdict: `passed`, `score` 0–10,
  `artifacts[]`.
- The best passing take is auto-selected (`selection_source=judge`). If none
  pass, the highest-scoring take is used (`fallback`). A human/agent can
  override (`human`). All takes are kept in MinIO + the `segment_candidates`
  table, so decisions are observable, replayable and re-composable.
- Human labels land in `judge_feedback` and are exportable to improve the judge.

## Three equivalent control surfaces

1. **HTML app**: `apps/seed_review.html` (linked from `apps/index.html`) —
   full review UX: per-scene candidate grid, verdicts, override, feedback,
   regenerate, recompose.
2. **MCP tools** (`python -m virtual_streamer.mcp_server.server`):
   `list_stories`, `get_story_scenes`, `list_scene_candidates`,
   `select_candidate`, `submit_judge_feedback`, `regenerate_scene`,
   `recompose_story`, `export_judge_feedback`, `get_job_status`.
3. **Discoverable CLI**: `python scripts/vsctl.py ops [filter]` lists a
   *curated* set of ~25 generic generation operations under short stable names
   (`generate-video`, `generate-script`, `generate-voice`, `generate-scene-image`,
   `generate-location`, `generate-story-template`, entity CRUD, review loop ops,
   `job-status`, `presign`). `describe <op>` pulls the live param/body schema
   from OpenAPI; `call <op> -p k=v --json '{}'` invokes. `ops --all` shows the
   raw API surface if something is missing. Set `VS_API_URL` or `--api`
   (default `http://localhost:8000`). Character-specific agents (jesus…),
   streaming/playlist/twitch and admin endpoints are deliberately excluded.

## Standard workflows

### Generate a video with seed hunting
POST `/api/v1/video-generation/generate` (or MCP `create_video_ltx`) with:
`enable_seed_hunt` (default true), `seed_hunt_max_candidates` (default 3),
`seed_hunt_accept_score` (default 7.5), `seed_hunt_seeds` (explicit seeds for
replay). Poll `/api/v1/jobs/{job_id}`.

### Fix a bad scene the judge got wrong
1. `list_stories` → pick story → `get_story_scenes`.
2. `list_scene_candidates(story_id, scene_id)` — inspect verdicts/scores.
3. If a better take already exists: `select_candidate(candidate_id)` —
   this also records a positive preference label.
4. If no take is good: `regenerate_scene(story_id, scene_id)` (new seed hunt,
   background job — poll `get_job_status`).
5. `recompose_story(story_id)` — rebuilds the final video from the currently
   selected take of every scene and updates `stories.final_video_key`.

### Collect / use judge feedback
- Label any candidate with `submit_judge_feedback(candidate_id, human_passed,
  human_score, artifact_tags, comment)`. Artifact tag vocabulary matches the
  judge: `unrealistic_movement`, `impossible_body`, `object_intersection`,
  `impossible_setting`, `identity_drift`, `visual_glitch`, `other`.
- `export_judge_feedback` / GET `/api/v1/judge-feedback/export` returns human
  labels joined with judge verdicts — disagreements (`human_passed !=
  judge_passed`) are the signal for tuning the judge prompt
  (`virtual_streamer/agents/video_judge/prompt.py`) or model
  (`configs/agents/video_judge.yaml`).

## Gotchas

- The judge never blocks generation: on any judge failure it returns a
  permissive default (`passed=true, score=5.0, judge_error` set). Filter on
  `judge_error` when analyzing verdicts.
- Recompose uses MinIO candidate videos; scenes generated before seed hunting
  fall back to `scenes.video_segment_key` (debug uploads must have been enabled).
- Candidate MinIO keys: `candidates/{story_id}/{scene_id}/seed_{seed}.mp4`.
  Presign with GET `/api/v1/storage/presign?key=...`.
- LTX server: `LTX_SERVER_URL` (default `http://gx10-cbc5:8082`); SD server:
  `SD_SERVER_URL` (default `http://gx10-cbc5:1234`); local judge LLM:
  llama.cpp OpenAI endpoint at `http://100.114.182.89:8081`.
