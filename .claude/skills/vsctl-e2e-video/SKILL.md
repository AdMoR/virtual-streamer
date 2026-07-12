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

Run the precheck script instead of curling each endpoint by hand — it checks
all services below in one shot and exits non-zero if anything is down:

```bash
uv run python scripts/vsctl_precheck.py
```

All of these must be up, or generation fails mid-pipeline: REST API
(`virtual_streamer_api`, port 8000, orchestrates everything), WanGP LTX-2
server (`LTX_SERVER_URL`, default `http://gx10-cbc5:8082`, video model),
Stable Diffusion server (`SD_SERVER_URL`, default `http://gx10-cbc5:1234`,
first-frame/location images — it has **no** `/health`, only `/v1/models`),
local judge/scene LLM (llama.cpp, `http://100.114.182.89:8081`, seed-hunt
judge + story pipeline agents), MySQL + MinIO (compose services `mysql`,
`minio`, persistence of stories/scenes/candidates + binaries), and the
Anthropic API key (`ANTHROPIC_API_KEY` in `.env`, used by the `story_writer`
agent).

If a check fails, inspect that service directly (e.g. `curl -s
$VS_API_URL/health`, `docker compose ps mysql minio`) — the script only
reports pass/fail, not how to fix it.

If the judge LLM is down, generation still succeeds — the judge fails open
(`judge_error` set, every take accepted at score 5.0) — but seed hunting
loses its value.

Characters referenced by the script must exist with **identity images** (for
first-frame conditioning) and **voice samples** (for talking-head A1O audio):
check with `vsctl call get-character -p character_id=ID`.

**The GPU host is shared and its services auto-restart.** `gx10-cbc5` (a single
GB10 box with unified CPU/GPU memory) runs the LTX video model (8082), SD
(1234), and the judge LLM (8081) together. The WanGP LTX server is a Docker
container (`wangp-server`) with a restart policy, so it **comes back up on its
own after a crash** — a single passing `/health` check before a run does *not*
prove it stayed up *during* the run. Heavy generations can exhaust the shared
memory pool and get the process killed. So:
- Monitor `/health` **throughout** a run (watch `generation_in_progress` /
  `queue_depth`), not just at the start.
- A crash mid-run leaves a fresh, healthy-looking server afterward, so
  post-hoc `/health` tells you nothing — check the run's actual output instead.

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

## Async jobs, files & queueing

- `generate-video*`, `regenerate-scene`, `recompose-story`,
  `backfill-candidates` are async and return a `job_id`. Add `--wait
  [--timeout N]` to any `vsctl call` to block until the job completes/fails
  (uses the server long-poll `wait-job` = `GET /jobs/{id}/wait`; vsctl exits
  non-zero on failure). Failed jobs carry an `error` string.
- **`completed` does not mean every scene rendered — the pipeline fails soft.**
  If the video server dies mid-run, failed scenes are dropped and the job
  composes whatever succeeded (a 6-scene job can return a 1-scene video). Always
  check `result.segment_count` == scenes submitted (and `total_duration_seconds`
  is plausible) before trusting a `completed` status. Dropped scenes leave no
  candidates / no `video_segment_key`; re-run them (Case C) once healthy.
- Recompose re-burns subtitles when you pass
  `--json '{"enable_subtitles": true}'` — each segment's own audio track is
  transcribed (Whisper accepts video input), so it works for any take.
- For a story generated before seed hunting, run
  `vsctl call backfill-candidates -p story_id=UUID --wait` once: it wraps each
  existing segment into a judged, selected candidate so review/override/
  recompose work exactly like a seed-hunted story.
- Candidate responses include a ready `video_url` (presigned) — no manual
  presign needed. To download any MinIO key locally:
  `vsctl fetch <key> [dest]`.
- GPU jobs are serialized server-side through a single priority queue
  (interactive jobs — scene regeneration, single clips — jump ahead of queued
  full videos). Submit freely; the response message includes the queue
  position, and `/api/v1/video-generation/health` reports `gpu_queue_depth`.

## Changing pipeline behavior (code / config)

- **The API's `virtual_streamer` package is baked into the Docker image, not
  volume-mounted** (only `./configs` is mounted, read-only). Editing source has
  **no effect on the running pipeline** until you rebuild and recreate the
  container: `docker compose -f compose.yaml up -d --build virtual_streamer_api`.
  Confirm a change actually landed with
  `docker exec <api-container> grep ... /app/virtual_streamer/...`. An env-only
  change still needs a recreate (`up -d`), not just `restart`.
- **Talking-head model quality is env-selectable.** `TALKING_HEAD_MODEL` in the
  API container env picks the generation ("first pass") model: unset/`distilled`
  = fast distilled `ltx2_22B_distilled_1_1` (8 steps); `quality` = full
  non-distilled `ltx2_22B` (30 steps) — higher fidelity but ~4x slower and a
  much larger model load (raises the shared-memory crash risk noted in the
  prereqs). The A1O talking-head path is primarily validated on the distilled
  model, so **validate one scene** (`regenerate-scene` with `max_candidates=1`)
  before committing to a full non-distilled run.
- The video **judge** rubric is `virtual_streamer/agents/video_judge/prompt.py`
  (+ artifact categories in `schema.py`). The judge is given the scene's
  `ltx_prompt` and, when present, its `Spoken line:` — so speech-aware rules
  (e.g. require visible lip movement) can be added to the prompt with no extra
  plumbing. It samples only ~8 frames per take, which is coarse for fast motion
  like lip-sync.
