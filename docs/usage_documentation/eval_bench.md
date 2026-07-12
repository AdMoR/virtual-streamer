# Eval Bench — human evaluation of generative models

A generic benchmark system covering every generative model in the codebase:

| model_kind  | What is evaluated                          | `model_id` means                                    |
|-------------|--------------------------------------------|-----------------------------------------------------|
| `image`     | SD conditioning-image generation           | client impl: `stable_cpp` (default) or `openai`     |
| `video`     | LTX/WanGP video generation                 | quality preset: `fast`, `quality`, `high_quality`   |
| `tts`       | Fish-Speech audio                          | `character_id` whose voice to clone, or `default`   |
| `llm_agent` | Any ADK agent (`configs/agents/*.yaml`)    | agent config name, e.g. `title_generator`           |

Core abstraction: a **bench** is a fixed dataset of cases; a **run** executes
the bench against one model; each case produces a **sample** (MinIO artifact or
JSON output, optionally auto-scored by a judge agent); humans review samples in
the HTML app and leave **feedback** (pass/fail, 0–10 score, issue tags,
comment). Runs of the same bench are directly comparable — that is the model
leaderboard.

## Components

- DB: `virtual_streamer/utils/eval_repository.py` (`eval_benches`, `eval_runs`,
  `eval_samples`, `eval_feedback`; auto-created on first use)
- Harness: `virtual_streamer/eval_bench/` (runners per model kind, `run_bench`,
  backfill importer)
- API: `virtual_streamer/api/low_level/eval_bench.py` under `/api/v1/eval/*`
- UI: `apps/eval_bench.html` (nginx-served, linked from `apps/index.html`)
- Bench definitions: `configs/eval_benches/*.yaml`

## Quick start (first benchmark for a human to review)

1. **Start the API** (tables are created lazily on first eval call):
   ```bash
   python -m virtual_streamer.api.main
   ```
2. **Load the seed benches** shipped in `configs/eval_benches/`:
   ```bash
   curl -X POST localhost:8000/api/v1/eval/benches/sync
   ```
   This registers four ready-made benchmarks:
   - `llm/title-generator` — 4 cases, cheapest to run (no GPU)
   - `image/scene-basics` — 5 cases (portrait, two people, location, text, hands)
   - `video/motion-basics` — 3 short t2v clips, auto-scored by `video_judge`
   - `tts/speech-basics` — 5 sentences (prosody, numbers, questions, long form)
3. **Run one** (LLM bench shown; grab `bench_id` from `GET /api/v1/eval/benches`):
   ```bash
   curl -X POST localhost:8000/api/v1/eval/benches/{bench_id}/runs \
     -H 'Content-Type: application/json' \
     -d '{"model_id": "title_generator", "label": "baseline"}'
   ```
   Poll `GET /api/v1/jobs/{job_id}` until completed.
4. **Review as a human**: open `apps/eval_bench.html` (or the launcher
   `apps/index.html` → “Eval Bench”). Select the bench → click the run → inspect
   each sample (image/video/audio player or JSON output) → click **💬 Feedback**
   and submit 👍/👎, a score, issue tags and a comment. Aggregates (human score,
   pass rate) appear in the runs table for side-by-side model comparison.
5. **Export the labels** for tuning models or judges:
   ```bash
   curl "localhost:8000/api/v1/eval/feedback/export?bench_id={bench_id}" > labels.json
   ```
   (or the ⬇ Export button in the app).

## Instant benchmark from existing work: backfill

If you already generated stories with seed hunting, import every take as a
reviewable benchmark (idempotent — safe to re-run):

```bash
curl -X POST localhost:8000/api/v1/eval/backfill-candidates \
  -H 'Content-Type: application/json' -d '{}'
```

Each story becomes a `video/seed-hunt: <title>` bench with one run of imported
takes, carrying the judge verdicts and any previous human labels. The video
seed-hunt *workflow* (select take, regenerate, recompose) stays in
`apps/seed_review.html`; the eval bench is for labelling and comparison.

## Comparing two models

Run the same bench twice with different `model_id` (or `model_config`, or after
editing an agent's YAML) and different labels, review both runs, and read the
runs table: `auto score`, `human score`, `pass rate` per run.

```bash
# e.g. video preset comparison
-d '{"model_id": "fast", "label": "fast preset"}'
-d '{"model_id": "quality", "label": "quality preset"}'
```

## Adding a new bench

Drop a YAML in `configs/eval_benches/` and sync:

```yaml
name: image/my-new-bench           # unique; syncing again updates the cases
model_kind: image                  # image | video | tts | llm_agent
description: What a reviewer should look for.
judge_agent: video_judge           # optional, video benches only (auto-scoring)
cases:
  - case_id: my_case               # stable id, used in artifact keys
    label: Human-readable card title
    params: { prompt: "...", seed: 42 }   # runner input (kind-specific)
```

Case `params` are the runner's input: `Txt2ImageParams` fields (image),
`VideoGenerationParams` fields (video), `{text: ...}` (tts),
`{message: ...}` (llm_agent). Benches can also be created ad hoc with
`POST /api/v1/eval/benches`.

## Storage layout

- Artifacts: MinIO `eval/{run_id}/{case_id}_{sample_id}.{ext}` (played in the
  app via presigned URLs)
- Metadata + labels: MySQL eval tables; imported video takes keep a
  `candidate_id` link back to `segment_candidates`.
