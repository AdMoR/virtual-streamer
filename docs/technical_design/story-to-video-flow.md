# Generation Flow: Story Template → LTX Video

This document precisely traces the full code path that turns a **story title + story template** into a final concatenated MP4 video, naming every intermediate data model, every parameter, and where each value comes from.

---

## Table of Contents

1. [Entry Points](#0-entry-points)
2. [3-Step Story Pipeline Agent](#1-step-1--3-step-story-pipeline-agent)
3. [FluxPrompt — Shared Visual Description Model](#2-fluxprompt--shared-visual-description-model)
4. [Location Base Image Generation](#3-step-2--location-base-image-generation)
5. [scenes_to_video](#4-step-3--scenes_to_video)
6. [story_input_to_video — Core Pipeline](#5-core-pipeline--story_input_to_video)
7. [Per-Scene Sub-Pipeline](#6-per-scene-sub-pipeline-loop-over-sceneinput)
8. [VideoGenerationParams at Segment Level](#7-videogenerationparams-at-segment-level)
9. [VideoGenerationParams — Full Schema](#8-videogenerationparams--full-schema)
10. [WanGP REST Client Protocol](#9-wangp-rest-client-protocol)
11. [Result Models](#10-segmentresult-and-storyvideoresult)
12. [Optional Post-Generation Steps](#11-optional-steps-after-generation)
13. [DB Persistence](#12-db-persistence)
14. [Full Flow Diagram](#13-full-flow-diagram)
15. [Key Files Reference](#key-files-reference)

---

## 0. Entry Points

Two HTTP endpoints exist on the API router (`/api/v1/video-generation`):

| Endpoint | Handler | Description |
|---|---|---|
| `POST /generate` | `_run_video_generation` | Title → 3-step pipeline → video |
| `POST /generate-from-script` | `_run_from_script` | Pre-built scenes → video (skips story agent) |

Both spawn a background task and return a `job_id` immediately.

**`VideoGenerationRequest`** fields (source: HTTP request body):

```
title: Optional[str]           — story title fed to the LLM
story_text: Optional[str]      — alternative raw story (exactly one of title/story_text)
story_template_id: str         — REQUIRED — FK to story_templates table
video_width: int = 1280
video_height: int = 720
video_duration_seconds: float = 5.0
video_fps: int = 24
video_steps: int = 20
video_cfg_scale: float = 4.0
video_seed: int = -1
enable_subtitles: bool = True
subtitle_fontsize: int = 14
```

→ `request.to_video_params()` converts these into a `VideoGenerationParams` (see [§8](#8-videogenerationparams--full-schema)).

**Server-side env vars** read in `video_generation.py`:

```
LTX_SERVER_URL    = os.environ.get("LTX_SERVER_URL",      "http://gx10-cbc5:8082")
LTX_TIMEOUT       = os.environ.get("LTX_TIMEOUT",         "3600.0")
SD_SERVER_URL     = os.environ.get("SD_SERVER_URL",       "http://gx10-cbc5:1234")
ENABLE_DEBUG_DUMP = os.environ.get("ENABLE_DEBUG_DUMP",   "true")
```

---

## 1. Step 1 — 3-Step Story Pipeline Agent

**Function**: `run_story_pipeline(title, story_template_id)` → `StoryPipelineResult`  
**File**: `virtual_streamer/api/high_level/video_generation.py`

Runs the ADK `StoryPipelineAgent` with an `InMemorySessionService`. Initial state:

```python
{ TITLE: title, STORY_TEMPLATE_ID: story_template_id }   # optional: NEWS_CONTEXT
```

The pipeline has **3 sequential sub-agents**:

### 1a. StoryWriterAgent

- **Model**: configured in YAML (LLM, currently local or Anthropic)
- **Input state key**: `TITLE`
- **Output state key**: `RAW_STORY_TEXT` (free-form string)
- **Prompt source**: template prompt from `story_templates` DB row (loaded by `StoryInstructionProvider`), wrapped in `META_PROMPT`
- **Template prompt variables**: `{title}`, `{characters}`, `{locations}`, `{target_lines}`

### 1b. RecurrentLocationBuilderAgent

- **Model**: `configs/agents/recurrent_location_builder.yaml` (Qwen3-VL-8B-Instruct-GGUF)
- **Input state key**: `RAW_STORY_TEXT`
- **Output state key**: `RECURRENT_LOCATIONS` (JSON string)
- **Output schema** (`RecurrentLocationsOutput` — `virtual_streamer/agents/story_pipeline/schema.py`):

```
RecurrentLocationsOutput
  locations: List[RecurrentLocation]
    location_id: str        # unique slug, e.g. "ski-resort"
    name: str               # human-readable
    flux_prompt: FluxPrompt # environment-only, NO characters
```

### 1c. DetailedSceneBuilderAgent

- **Model**: `configs/agents/detailed_scene_builder.yaml` (Qwen3-VL-8B-Instruct-GGUF)
- **Input state keys**: `RAW_STORY_TEXT`, `RECURRENT_LOCATIONS`
- **Output state key**: `DETAILED_SCENES` (JSON string)
- **Output schema** (`DetailedScenesOutput` — `virtual_streamer/agents/story_pipeline/schema.py`):

```
DetailedScenesOutput
  title: Optional[str]
  scenes: List[DetailedScene]
    ltx_prompt: str                           # direct cinematic prompt for LTX
    location: Optional[str]                   # location_id reference or null
    character_on_screen: Optional[List[str]]  # character_ids visible (teleportation rule)
    scene_visual_description: FluxPrompt      # Flux conditioning image prompt
    speaker_id: Optional[str]                 # character_id of speaker
    spoken_line: Optional[str]                # exact text spoken (→ TTS / video_length)
```

### StoryPipelineResult (return value)

```
StoryPipelineResult
  recurrent_locations: RecurrentLocationsOutput
  detailed_scenes: DetailedScenesOutput
  title: Optional[str]          # from detailed_scenes.title
  raw_story_text: Optional[str] # from RAW_STORY_TEXT state key
```

---

## 2. FluxPrompt — Shared Visual Description Model

**File**: `virtual_streamer/image_generation/models.py`

Used in both `RecurrentLocation.flux_prompt` (environment only, no people) and `DetailedScene.scene_visual_description` (scene + character descriptions).

```
FluxPrompt
  scene: str                              # overall scene context / environment
  subjects: List[Subject]                 # subjects ordered by visual prominence
    description: str
    pose: Optional[str]
    position: str                         # placement within frame
    color_palette: Optional[List[str]]
  style: Optional[str]                    # artistic / photographic style
  color_palette: Optional[List[str]]      # global palette
  lighting: str
  mood: Optional[str]
  background: Optional[str]
  composition: Optional[str]              # e.g. "rule of thirds"
  camera: Camera
    angle: str                            # "high angle", "eye level", "low angle"
    distance: str                         # "wide shot", "medium shot", "close-up"
    focus: Optional[str]
    lens_mm: Optional[int]
    f_number: Optional[str]
    ISO: Optional[int]
```

**`flux_prompt.to_prompt()`** → serializes to a flat string for Stable Diffusion or for building the LTX talking-head prompt visual section.

---

## 3. Step 2 — Location Base Image Generation

**Where**: inside `_run_video_generation`, immediately after `run_story_pipeline`.

For each `RecurrentLocation` that does **not** already exist in the DB:

| Step | Detail |
|---|---|
| Generator | Stable Diffusion `txt2image` (no people) |
| Prompt | `flux_prompt.to_prompt() + ", no people, cinematic composition, photorealistic, high quality"` |
| Negative prompt | `"text, watermark, blurry, distorted, people, persons, characters"` |
| Width × Height | `VideoGenerationRequest.video_width × video_height` (default `1280×720`) |
| SD server | `SD_URL` env var (default `http://gx10-cbc5:1234`) |
| MinIO key | `locations/{story_template_id}/{location_id}.png` |
| DB write | `create_location(...)` — sets `image_path` field |

> **Note**: If the location already exists in the DB its image is reused and no regeneration happens.

---

## 4. Step 3 — scenes_to_video

**Function**: `scenes_to_video(scenes, story_title, ltx_config, video_params, ...)` → `StoryVideoResult`  
**File**: `virtual_streamer/video_generation/story_to_video.py`

Converts `List[DetailedScene]` into `StoryInput` via the `DetailedSceneInput` factory:

```python
# Factory: DetailedSceneInput.from_detailed_scene(scene, index)
#          virtual_streamer/video_generation/scene_input.py

DetailedSceneInput(SceneInput)
  scene_index: int
  ltx_prompt: str                      ← scene.ltx_prompt
  speaker_id: Optional[str]            ← scene.speaker_id
  spoken_line: Optional[str]           ← scene.spoken_line
  location_id: Optional[str]           ← scene.location  (field name differs from DetailedScene)
  character_ids_on_screen: List[str]   ← scene.character_on_screen or []
  scene_visual_description: dict       ← scene.scene_visual_description.model_dump()
  raw_scene_data: dict                 ← scene.model_dump()  (for DB storage / replay)
```

Wrapped in:

```python
StoryInput
  title: str
  story_plan: str = ""
  story_template_id: Optional[str]
  raw_agent_output: dict               # full serialised scenes + locations
  scenes: List[SceneInput]
```

Then delegates to `story_input_to_video(...)`.

> **Design note**: `SceneInput`/`StoryInput` are a stable abstraction layer — `story_to_video.py` never imports `DetailedScene` or `DialogLine` directly. Agent-format changes only require updating the `from_*` factory methods.

---

## 5. Core Pipeline — story_input_to_video

**File**: `virtual_streamer/video_generation/story_to_video.py`

**Function signature**:

```python
async def story_input_to_video(
    story_input: StoryInput,
    ltx_config: Optional[LTXVideoConfig] = None,         # default: LTXVideoConfig()
    video_params: Optional[VideoGenerationParams] = None, # default: preset "fast", 5s
    output_dir: str = "./output",
    progress_callback: Optional[Callable] = None,
    sd_server_url: Optional[str] = None,                  # fallback: SD_SERVER_URL env
    debug_minio_prefix: Optional[str] = None,
    reference_videos: Optional[Dict[int, str]] = None,
    story_repo: Optional[Any] = None,
    db_story_id: Optional[str] = None,
    keep_segments: bool = False,
)
```

**Defaults applied at the start**:

```python
config = ltx_config or LTXVideoConfig()
#   → server_url = "http://localhost:8082"  (override: LTX_SERVER_URL env)

params = video_params or VideoGenerationParams.from_preset("fast", duration_seconds=5.0)
#   → model_type="ltx2_22B_distilled_1_1", steps=8, guidance=1.0, flow_shift=3.0

_sd_url = sd_server_url or os.environ.get("SD_SERVER_URL", "http://gx10-cbc5:1234")
```

**Pre-loading from DB** (via `get_entity_repository()`):

```
location_map:  Dict[str, dict]  — all locations belonging to story_template_id
character_map: Dict[str, dict]  — all characters referenced in scenes
                                   (speakers + characters_on_screen)
```

---

## 6. Per-Scene Sub-Pipeline (loop over SceneInput)

For each `scene_input` in `story_input.scenes`:

### 6a. Optional: Prompt Enrichment

If `reference_videos[i]` is supplied → `run_scene_enricher(ref_path, ltx_prompt)` updates `scene_input.ltx_prompt` before generation.

### 6b. DB: Create Scene Row

Writes to `scenes` table before generation begins: `scene_index`, `prompt`, `speaker_id`, `spoken_line`, `location_id`, `raw_scene_data`.

### 6c. Conditioning Image (Stable Diffusion)

**Function**: `generate_scene_image_from_input(scene_input, location, character_dicts, output_dir, video_params, sd_server_url)`  
**File**: `virtual_streamer/video_generation/story_to_video.py`

| Step | Detail |
|---|---|
| Prompt | `FluxPrompt.model_validate(scene_input.scene_visual_description).to_prompt()` (fallback: `scene_input.ltx_prompt`) |
| Reference images | up to 3 total: `location["image_path"]` (1) + `character["identity_images"][0]` (1 per char on screen) |
| With refs → mode | `image_edit(denoising_strength=0.1, width=params.width, height=params.height)` |
| Without refs → mode | `txt2image(negative_prompt += ", people, persons, characters")` |
| Output size | `video_params.width × video_params.height` (default `1280×720`) |
| After generation | Upload to MinIO → `conditioning_images/{template_id}/{scene_id}.png` |
| DB write | `create_conditioning_image_artifact(...)` |

Failure is **graceful**: returns `None`, segment falls back to t2v mode.

### 6d. Speaker Voice Reference Download

```python
# From character_map (loaded from DB)
voice_sample_key = character_map[speaker_id]["voice_samples"][0]["sample_storage_path"]
# Downloaded from MinIO → local speaker_audio.wav
```

Failure is **graceful**: `speaker_audio_path = None` → segment uses i2v mode (no audio conditioning).

### 6e. Segment Generation

**Function**: `generate_segment_from_input(client, scene_input, output_dir, video_params, audio_path, image_path)`

**Mode selection** (automatic):

```
talking_head = audio_path is not None AND os.path.exists(audio_path)

mode = "talking-head A1O"  ← audio available
     | "i2v"               ← image available, no audio
     | "t2v"               ← neither
```

---

## 7. VideoGenerationParams at Segment Level

### Mode A: Talking-Head (A1O) — audio available

| Parameter | Value | Source |
|---|---|---|
| `model_type` | `"ltx2_22B_distilled_1_1"` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `num_inference_steps` | `8` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `guidance_scale` | `1.0` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `flow_shift` | `5.0` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `guidance_phases` | `2` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `sample_solver` | `"distilled_8_steps"` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `audio_scale` | `1.0` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `audio_guidance_scale` | `5.0` | **HARDCODED** in `TALKING_HEAD_PARAMS` |
| `activated_loras` | `["id-lora-celebvhq-ltx2.3.safetensors"]` | **HARDCODED** (`TALKING_HEAD_LORA`) |
| `loras_multipliers` | `"1.0"` | **HARDCODED** |
| `audio_prompt_type` | `"A1O"` | **HARDCODED** |
| `resolution` | `video_params.resolution` | From request (default `"1280x720"`) |
| `fps` | `video_params.fps` | From request (default `24`) |
| `seed` | `video_params.seed` | From request (default `-1`) |
| `video_length` | computed from `spoken_line` word count | See formula below |
| `prompt` | `build_talking_head_prompt(visual, spoken_line)` | `[VISUAL]/[SPEECH]/[SOUNDS]` sections |
| `image_start` | `image_path` (SD conditioning image) | §6c output |
| `image_prompt_type` | `"S"` if image_path else `""` | Automatic |
| `audio_guide` | `speaker_audio_path` | Character voice sample from MinIO |

**`video_length` formula** (function `_video_length_from_spoken_line`):

```python
_WORDS_PER_SECOND  = 2.2
_MIN_SPEECH_SECONDS = 5.0
_MAX_SPEECH_SECONDS = 15.0

words    = len(spoken_line.split())
duration = max(5.0, min(15.0, words / 2.2 + 1.5))   # clamped [5, 15] s
raw      = int(duration * fps)
n        = max(round((raw - 1) / 8), 1)
video_length = 8 * n + 1                              # always 8n+1, min 9
```

### Mode B: i2v / t2v — no audio

| Parameter | Value | Source |
|---|---|---|
| `model_type` | `video_params.model_type` | Preset / request |
| `num_inference_steps` | `video_params.num_inference_steps` | Request (default `20`) |
| `guidance_scale` | `video_params.guidance_scale` | Request (default `4.0`) |
| `seed` | `video_params.seed` | Request (default `-1`) |
| `resolution` | `video_params.resolution` | Request (default `"1280x720"`) |
| `fps` | `video_params.fps` | Request (default `24`) |
| `video_length` | from `video_params.duration_seconds` | `_frames_from_duration(duration, fps)` |
| `prompt` | `scene_input.ltx_prompt` | From `DetailedScene.ltx_prompt` |
| `image_start` | `image_path` | SD conditioning image (§6c) |
| `image_prompt_type` | `"S"` if image_path else `""` | Automatic |

**`_frames_from_duration` formula**:

```python
raw = int(duration_seconds * fps)
n   = max(round((raw - 1) / 8), 1)
video_length = 8 * n + 1   # always 8n+1, minimum 9
```

---

## 8. VideoGenerationParams — Full Schema

**File**: `virtual_streamer/video_generation/ltx_client.py`

### Core Settings

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_type` | str | `"ltx2_22B_distilled_1_1"` | LTX model variant |
| `prompt` | str | `""` | Text prompt |
| `negative_prompt` | str | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | `DEFAULT_NEGATIVE_PROMPT` |
| `resolution` | str | `"1280x720"` | WxH; auto-corrected to match image dimensions when still at default |
| `video_length` | int | `97` | Frame count; **must be 8n+1** |
| `num_inference_steps` | int | `8` | Denoising steps |
| `guidance_scale` | float | `1.0` | CFG scale |
| `flow_shift` | float | `5.0` | Flow shift |
| `seed` | int | `-1` | -1 = random |

### Conditioning Flags (must be set explicitly — no auto-defaults)

| Field | Type | Default | Values |
|---|---|---|---|
| `image_prompt_type` | str | `""` | `"S"` (start), `"E"` (end), `"SE"` (both) |
| `audio_prompt_type` | str | `""` | `"A"` (audio), `"A1O"` (audio + ID-LoRA) |
| `video_prompt_type` | str | `""` | `"DVG"` (depth), `"PVG"` (pose), `"I"` (identity), etc. |

### File Path Fields (uploaded before submission)

| Field | Type | Notes |
|---|---|---|
| `image_start` | Optional[str] | Local path, used with `image_prompt_type="S"` |
| `image_end` | Optional[str] | Local path, used with `image_prompt_type="E"` |
| `audio_guide` | Optional[str] | Local WAV/MP3/FLAC, used with `audio_prompt_type` |
| `video_guide` | Optional[str] | Local MP4, used with `video_prompt_type` |
| `image_refs` | List[str] | Local paths for identity mode |
| `keyframes` | List[List] | `[[path, frame_idx, strength], ...]` |

### Conditioning Strengths

| Field | Default | Notes |
|---|---|---|
| `audio_scale` | `1.0` | Audio conditioning strength |
| `audio_guidance_scale` | `7.0` | Audio CFG scale |
| `denoising_strength` | `1.0` | V2V control strength (0–1) |

### LoRA

| Field | Default | Notes |
|---|---|---|
| `activated_loras` | `[]` | LoRA filenames from server's `loras/ltx2/` dir |
| `loras_multipliers` | `""` | Space-separated weights, one per LoRA |

### Two-Stage Pipeline (Distilled Model)

| Field | Default | Notes |
|---|---|---|
| `guidance_phases` | None | `1` = dev only; `2` = dev + distilled-LoRA phase |
| `sample_solver` | None | e.g. `"distilled_8_steps"` |

### Convenience Fields (not sent to API)

| Field | Default | Notes |
|---|---|---|
| `duration_seconds` | None | Auto-computes `video_length = nearest 8n+1` when set |
| `fps` | `24` | Used with `duration_seconds` |

### Named Presets (`VideoGenerationParams.from_preset(name)`)

| Preset | model_type | steps | guidance | flow_shift |
|---|---|---|---|---|
| `"fast"` | `ltx2_22B_distilled_1_1` | 8 | 1.0 | 3.0 |
| `"quality"` | `ltx2_22B` | 30 | 3.0 | 3.0 |
| `"high_quality"` | `ltx2_22B_pure_dev` | 50 | 3.0 | 3.0 |

---

## 9. WanGP REST Client Protocol

**File**: `virtual_streamer/video_generation/ltx_client.py` — class `WanGPLTXClient`

### LTXVideoConfig

```
server_url: str   = "http://localhost:8082"   # override: LTX_SERVER_URL env
timeout: float    = 600.0                      # HTTP timeout (uploads/downloads)
stream_timeout: float = 43200.0               # polling timeout (12 h)
api_key: Optional[str] = None                 # X-API-Key header
```

### Per-Segment REST Workflow

1. `GET /health` — verify server ready (`runtime_loaded=true`)
2. Upload each local file path → `POST /files/upload` → `file_id`
   - `image_start`, `image_end`, `audio_guide`, `video_guide`, `video_mask`, `image_refs`, `keyframes`
3. Auto-detect start-image dimensions → override `resolution` silently if it was still `"1280x720"` (the default)
4. Build JSON payload (excluding file path fields and convenience fields)
5. `POST /jobs/raw` → `job_id`
6. Poll `GET /jobs/{job_id}` every 5 s until `status == "completed"`
7. `GET /files/{filename}` → download MP4

---

## 10. SegmentResult and StoryVideoResult

**`SegmentResult`** — one per scene (`virtual_streamer/video_generation/story_to_video.py`):

```
index: int
video_path: str               # local MP4 path
duration_seconds: float
prompt_id: str                # basename of downloaded video file
scene_input: Optional[SceneInput]
audio_path: Optional[str]     # WAV used for audio conditioning
image_path: Optional[str]     # PNG conditioning image
minio_video_key: Optional[str]
minio_audio_key: Optional[str]
minio_image_key: Optional[str]
db_scene_id: Optional[str]    # FK → scenes table
```

**`StoryVideoResult`** — final pipeline output:

```
final_video_path: str
segments: List[SegmentResult]
story_title: str
total_duration_seconds: float
debug_minio_prefix: Optional[str]
minio_final_video_key: Optional[str]
minio_manifest_key: Optional[str]
db_story_id: Optional[str]    # FK → stories table
```

---

## 11. Optional Steps After Generation

### 11a. Subtitle Burning (`enable_subtitles=True`)

When enabled, `keep_segments=True` is passed to `scenes_to_video` to preserve individual segment files on disk. Then:

1. For each `SegmentResult` with an `audio_path`:
   - `transcribe_to_srt(audio_path, srt_file)` — Whisper transcription
   - `add_subtitle_from_srt(video_path, srt_file, output, fontsize)` — ffmpeg burn-in
2. Re-concatenate all (subtitled + non-subtitled) segments → `final_with_subtitles.mp4`

### 11b. Video Concatenation

```python
concatenate_videos(video_paths, output_path, temp_dir)
```

- **Attempt 1** — stream copy: `ffmpeg -f concat -safe 0 -c copy` (fast, lossless)
- **Fallback** — re-encode: `ffmpeg -c:v libx264 -c:a aac -preset fast`

Single-segment shortcut: `shutil.copy2` (no ffmpeg call needed).

### 11c. Final MinIO Upload

Key: `generated_videos/ltx/{job_id}.mp4`

---

## 12. DB Persistence (Best-Effort — Never Aborts Generation)

| Event | Table | Operation |
|---|---|---|
| Before generation starts | `stories` | `create_story(status="GENERATING")` |
| Before each scene | `scenes` | `create_scene(...)` |
| After conditioning image upload | `conditioning_image_artifacts` | `create_conditioning_image_artifact(...)` |
| After each segment | `scenes` | `update_scene_artifacts(video_key, audio_key, duration)` |
| After final video upload | `stories` | `update_story_status("COMPLETED", final_video_key)` |
| On unrecoverable exception | `stories` | `update_story_status("FAILED")` |

All DB calls are wrapped in `try/except`; failures log a warning and do not abort generation.

---

## 13. Full Flow Diagram

```
POST /api/v1/video-generation/generate
  │  VideoGenerationRequest { title, story_template_id, video_width/height/fps/steps/seed/cfg }
  │
  ▼ run_story_pipeline(title, story_template_id)
  │    StoryPipelineAgent (ADK, 3 sequential sub-agents)
  │
  ├─► StoryWriterAgent
  │     IN:  TITLE, STORY_TEMPLATE_ID
  │     OUT: RAW_STORY_TEXT (free-form string)
  │
  ├─► RecurrentLocationBuilderAgent
  │     IN:  RAW_STORY_TEXT
  │     OUT: RECURRENT_LOCATIONS → RecurrentLocationsOutput
  │            └── locations: [RecurrentLocation { location_id, name, flux_prompt }]
  │
  └─► DetailedSceneBuilderAgent
        IN:  RAW_STORY_TEXT, RECURRENT_LOCATIONS
        OUT: DETAILED_SCENES → DetailedScenesOutput
               └── scenes: [DetailedScene { ltx_prompt, location, character_on_screen,
                                            scene_visual_description, speaker_id, spoken_line }]

  ▼ For each new RecurrentLocation:
  │   SD txt2image(flux_prompt.to_prompt(), no people)
  │   → MinIO: locations/{template_id}/{location_id}.png
  │   → DB: create_location(image_path=...)
  │
  ▼ scenes_to_video(detailed_scenes.scenes, ...)
  │   DetailedSceneInput.from_detailed_scene() → SceneInput  (×N scenes)
  │   StoryInput { title, story_template_id, scenes }
  │
  ▼ story_input_to_video(StoryInput, LTXVideoConfig, VideoGenerationParams)
  │   Load: location_map, character_map from DB
  │   DB: create_story(status="GENERATING")
  │
  │   ┌─ For each SceneInput i ─────────────────────────────────────────────┐
  │   │                                                                     │
  │   │  [optional] run_scene_enricher(ref_video, ltx_prompt)              │
  │   │               → enriched ltx_prompt                                │
  │   │                                                                     │
  │   │  DB: create_scene(scene_index, prompt, speaker_id, ...)            │
  │   │                                                                     │
  │   │  generate_scene_image_from_input(...)                              │
  │   │    prompt = flux_prompt.to_prompt()                                │
  │   │    refs   = [location.image_path] + [char.identity_images[0] ×N]  │
  │   │    IF refs → SD image_edit(denoising_strength=0.1)                │
  │   │    ELSE    → SD txt2image                                          │
  │   │    size: video_params.width × video_params.height                  │
  │   │    → conditioning_image.png                                        │
  │   │    → MinIO upload + DB: create_conditioning_image_artifact         │
  │   │                                                                     │
  │   │  Download character voice sample from MinIO                        │
  │   │    character_map[speaker_id]["voice_samples"][0]["sample_storage_path"]
  │   │    → speaker_audio.wav  (or None on failure)                       │
  │   │                                                                     │
  │   │  generate_segment_from_input(client, scene_input, params, audio, image)
  │   │    IF audio → Talking-Head A1O                                     │
  │   │      prompt        = build_talking_head_prompt([VISUAL]/[SPEECH]/[SOUNDS])
  │   │      video_length  = 8n+1  (from word count of spoken_line)       │
  │   │      HARDCODED: model=distilled_1_1, steps=8, guidance=1.0        │
  │   │                  LoRA=id-lora-celebvhq-ltx2.3.safetensors         │
  │   │    ELSE → i2v or t2v                                               │
  │   │      prompt        = scene_input.ltx_prompt                       │
  │   │      video_length  = 8n+1  (from video_params.duration_seconds)   │
  │   │      uses: video_params.num_inference_steps, guidance_scale        │
  │   │    → WanGPLTXClient.generate_video(params)                        │
  │   │        upload files → POST /jobs/raw → poll → download MP4        │
  │   │    → SegmentResult { video_path, duration, prompt_id }            │
  │   │                                                                     │
  │   │  DB: update_scene_artifacts(video_key, audio_key, duration)       │
  │   └─────────────────────────────────────────────────────────────────────┘
  │
  ▼ concatenate_videos([seg_0.mp4, seg_1.mp4, ...]) → final.mp4
  │   (stream-copy first; re-encode libx264/aac on fallback)
  │
  ▼ [optional] _apply_subtitles(segments)
  │   Whisper transcribe → SRT → ffmpeg burn-in per segment → re-concat
  │
  ▼ MinIO upload: generated_videos/ltx/{job_id}.mp4
  │   DB: update_story_status("COMPLETED", final_video_key)
  │
  ▼ job_store.update_job(status="completed", result={...})
```

---

## Key Files Reference

| Concern | File |
|---|---|
| HTTP API + orchestration | `virtual_streamer/api/high_level/video_generation.py` |
| Story pipeline runner | `virtual_streamer/api/high_level/video_generation.py` → `run_story_pipeline` |
| 3-step pipeline agent schemas | `virtual_streamer/agents/story_pipeline/schema.py` |
| SceneInput abstraction layer | `virtual_streamer/video_generation/scene_input.py` |
| Core video pipeline | `virtual_streamer/video_generation/story_to_video.py` |
| LTX client + VideoGenerationParams | `virtual_streamer/video_generation/ltx_client.py` |
| FluxPrompt model | `virtual_streamer/image_generation/models.py` |
| Entity repository (DB) | `virtual_streamer/utils/entity_repository.py` |
| Story repository (DB) | `virtual_streamer/utils/story_repository.py` |
| Talking-head constants | `virtual_streamer/video_generation/story_to_video.py` → `TALKING_HEAD_PARAMS`, `TALKING_HEAD_LORA` |
| LTX prompt builders | `virtual_streamer/video_generation/ltx_prompt_builder.py` |
