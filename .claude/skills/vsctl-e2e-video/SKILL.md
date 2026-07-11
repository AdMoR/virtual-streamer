---
name: vsctl-e2e-video
description: Generate a video end-to-end from a script using the vsctl CLI — covers all three subcases (no existing story template, existing template, regeneration of a previous run with edits) plus the service prerequisites to check before starting. Use when asked to produce, script, or regenerate a video via vsctl / the REST API from the terminal or by an agent.
---

# End-to-end video generation with vsctl

`scripts/vsctl.py` exposes a curated set of generic operations (run
`uv run python scripts/vsctl.py ops` to list them; `describe <op>` pulls the
live schema). Everything below uses `vsctl <cmd>` as shorthand for
`uv run python scripts/vsctl.py <cmd>`. Set `VS_API_URL` or `--api`
(default `http://localhost:8000`).

## 0. Prerequisites — check BEFORE generating

All of these must be up, or generation fails mid-pipeline:

| Service | What it does | Check |
|---|---|---|
| REST API (`virtual_streamer_api`, port 8000) | Orchestrates everything | `curl -s $VS_API_URL/health` |
| WanGP LTX-2 server (`LTX_SERVER_URL`, default `http://gx10-cbc5:8082`) | Video model | `curl -s http://gx10-cbc5:8082/health` → `runtime_loaded: true` |
| Stable Diffusion server (`SD_SERVER_URL`, default `http://gx10-cbc5:1234`) | First-frame / location images | `curl -s http://gx10-cbc5:1234/health` |
| Local judge/scene LLM (llama.cpp, `http://100.114.182.89:8081`) | Seed-hunt judge + story pipeline agents (Qwen3-VL) | `curl -s http://100.114.182.89:8081/v1/models` |
| MySQL + MinIO (compose services `mysql`, `minio`) | Persistence of stories/scenes/candidates + binaries | `docker compose ps mysql minio` |
| Anthropic API key (`ANTHROPIC_API_KEY` in `.env`) | `story_writer` agent uses Claude | present in env |

Quick overall check: `vsctl call job-status -p job_id=x` returning a clean 404
(not a connection error) proves the API+DB path works. If the judge LLM is
down, generation still succeeds — the judge fails open (`judge_error` set,
every take accepted at score 5.0) — but seed hunting loses its value.

Characters referenced by the script must exist with **identity images** (for
first-frame conditioning) and **voice samples** (for talking-head A1O audio):
check with `vsctl call get-character -p character_id=ID`.

## Case A — No existing story template

1. **Create the template** (defines tone, character roster, style):
   ```bash
   vsctl call generate-story-template \
     --json '{"story_concept": "A parody of Cest pas Sorcier where Fred explains crypto"}'
   ```
   Note the returned template `id`. Verify: `vsctl call list-story-templates`.
2. **Create/verify characters** the template needs
   (`vsctl call create-character --json @character.json`, then upload identity
   images and voice samples via the API or the Character Registration app —
   these are multipart uploads, easier through `apps/creation_interface.html`).
3. Optionally pre-create locations: `vsctl call generate-location
   --json '{"location_name": "Workshop", "story_template_id": "TPL"}'`.
   Otherwise the pipeline creates locations automatically from the script.
4. Continue with Case B.

## Case B — Existing template: script → review → video

1. **Generate the script** (nothing persisted — safe to iterate):
   ```bash
   vsctl call generate-script \
     --json '{"title": "Fred se lance dans l IA", "story_template_id": "TPL"}'
   ```
   Returns `raw_story_text`, `recurrent_locations`, `detailed_scenes`.
2. **Edit the script if needed** (change `spoken_line`, `ltx_prompt`,
   speakers, cut scenes) — it's plain JSON.
3. **Generate the video from the (edited) script**:
   ```bash
   vsctl call generate-video-from-script --json @request.json
   ```
   where `request.json` = `{"story_title": ..., "story_template_id": "TPL",
   "scenes": [...detailed_scenes.scenes...], "locations":
   [...recurrent_locations.locations...]}` plus optional knobs:
   `enable_seed_hunt` (default true), `seed_hunt_max_candidates` (3),
   `seed_hunt_accept_score` (7.5), `seed_hunt_seeds` (explicit seeds ⇒ exact
   replay), `enable_subtitles`.
   - Shortcut when no script editing is wanted: `vsctl call generate-video
     --json '{"title": "...", "story_template_id": "TPL"}'` runs script +
     video in one job.
4. **Poll**: `vsctl call job-status -p job_id=JOB` until
   `completed`/`failed`. The result contains `story_id` implicitly via
   `list-stories`, segment details, and `metadata.video_url`.
5. **View**: `vsctl call presign -p key=MINIO_KEY` on
   `metadata.minio_video_key` (or any candidate `video_key`).

## Case C — Regeneration of a pre-existing generation, with updates

Everything from a previous run is persisted (story, scenes with
`raw_scene_data`, every seed-hunt candidate). Three levels of rework, cheapest
first:

1. **Only the selected take is wrong** (judge picked badly):
   ```bash
   vsctl call list-stories -p limit=10
   vsctl call get-story-scenes -p story_id=STORY
   vsctl call list-candidates -p story_id=STORY -p scene_id=SCENE
   vsctl call select-candidate -p candidate_id=CAND --json '{"user": "agent", "comment": "why"}'
   vsctl call recompose-story -p story_id=STORY --json '{}'   # → job_id
   ```
   `select-candidate` also records a preference label that feeds judge
   improvement. Consider `submit-feedback` on the take the judge wrongly
   passed/failed.
2. **No good take exists for a scene** — new seed hunt for just that scene
   (reuses the stored conditioning image and voice sample; scene text comes
   from `scenes.raw_scene_data`):
   ```bash
   vsctl call regenerate-scene -p story_id=STORY -p scene_id=SCENE \
     --json '{"max_candidates": 3, "accept_score": 7.5}'
   vsctl call job-status -p job_id=JOB          # wait for completed
   vsctl call recompose-story -p story_id=STORY --json '{}'
   ```
3. **The script itself needs updates** (new lines, different scenes):
   fetch the original script with `vsctl call get-story -p story_id=STORY`
   (`raw_agent_output.scenes` / `.locations` hold the exact inputs), edit the
   JSON, and rerun Case B step 3 (`generate-video-from-script`). This creates
   a *new* story row; the old one stays replayable. To reproduce specific
   takes, pass the original seeds (visible per candidate) via
   `seed_hunt_seeds`.

## Gotchas

- `generate-video*` and `regenerate/recompose` are async: always poll
  `job-status`; jobs report `failed` with an `error` string.
- Recompose does not re-burn subtitles.
- Scenes generated before seed hunting existed have no candidates —
  `list-candidates` returns `[]`; recompose then falls back to the scene's
  original `video_segment_key`.
- Candidate/final videos are MinIO keys, never local paths — always `presign`.
- The heavy GPU services (WanGP, SD) are single-queue: don't launch parallel
  full-video jobs; sequence them.
