# `script_to_video` — Technical Reference

This document covers everything a newcomer needs to understand the `script_to_video`
function and the two generation pipelines that call it.

---

## Overview

`script_to_video` is the final assembly step of the **Traditional Pipeline**
(TTS + Wav2Lip). It takes matched dialog lines and turns them into a single
concatenated MP4 by calling three webservices (TTS, Wav2Lip, STT) per segment,
then running local ffmpeg passes for combination and subtitles.

It is distinct from the **LTX-2 Pipeline** (`scenes_to_video`), which generates
video and audio simultaneously via the WanGP REST server.

---

## The Two Generation Pipelines

```
Title / story text
        │
        ▼
 StoryPipelineAgent  (ADK — 3-step: writer → location builder → scene builder)
        │
        ├──► Traditional Pipeline ──────────────────────────────────────────────┐
        │         │                                                              │
        │    SentenceVideoMatcher (ADK — match each dialog line to a clip)      │
        │         │                                                              │
        │         ▼                                                              │
        │    script_to_video()  ← THIS FUNCTION                                 │
        │     Per segment:                                                       │
        │       1. TTS  (Fish-Speech)                                            │
        │       2. Wav2Lip                                                       │
        │       3. combine video + audio  (ffmpeg)                               │
        │       4. STT → SRT subtitles    (ffmpeg burnin)                        │
        │     → concat all segments                                              │
        │     → upload to MinIO                                                  │
        │                                                                        │
        └──► LTX-2 Pipeline ────────────────────────────────────────────────────┘
                  │
             scenes_to_video()
               Per scene:
                 1. TTS (Fish-Speech, pre-computed)
                 2. Stable Diffusion → conditioning image
                 3. WanGP LTX-2 (audio-conditioned i2v)
               → concat all segments
               → upload to MinIO
```

---

## Function Signature

**File:** `virtual_streamer/api/high_level/video_generation.py:411`

```python
async def script_to_video(
    matches: List[DialogLineMatch],
    client: WebserviceClient,
    config: VideoGenerationConfig,
    ltx_config: Optional[LTXConfig] = None,
    enable_ltx_fallback: bool = False,
    progress_callback: Optional[callable] = None,
    debug_upload_prefix: Optional[str] = None,
) -> str
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `matches` | `List[DialogLineMatch]` | yes | Output of `SentenceVideoMatcher` — one entry per dialog line |
| `client` | `WebserviceClient` | yes | HTTP client for TTS / Wav2Lip / STT calls |
| `config` | `VideoGenerationConfig` | yes | Output dirs, subtitle font size, codec settings |
| `ltx_config` | `LTXConfig` | no | Required only when `enable_ltx_fallback=True` |
| `enable_ltx_fallback` | `bool` | no | Generate video with LTX-2 when a match is not `CONTEXTUAL` |
| `progress_callback` | `callable(str)` | no | Called with a human-readable status string before each segment |
| `debug_upload_prefix` | `str` | no | MinIO prefix for intermediate artifact uploads (e.g. `"debug/video-generation/tpl/job_id"`) |

**Returns:** local path to the final concatenated video (str).

**Raises:** re-raises any exception from the first failing segment — there is no per-segment retry.

---

## Per-Segment Steps

```
for i, match in enumerate(matches):

  [optional] LTX-2 fallback
    └─ if match.needs_generation AND enable_ltx_fallback AND ltx_config
          generate_ltx_fallback_video(scene_description, ltx_config, temp_dir, i)
          → replaces match.video_path with freshly generated clip

  [1/4] TTS
    └─ WebserviceClient.generate_tts(text, character_id) → audio_path (.wav)

  [2/4] Wav2Lip
    └─ WebserviceClient.generate_wav2lip(audio_path, video_path, character_id) → lip_synced_video

  [3/4] Combine video + audio
    └─ combine_video_and_short_audio(lip_synced_video, audio_path, combined_path)
        ffmpeg: h264_nvenc + aac, 720x480, 30 fps

  [4/4] Subtitles
    └─ WebserviceClient.transcribe_to_srt(audio_path) → srt_path
    └─ add_subtitle_from_srt(combined_path, srt_path, segment_path, fontsize)
        ffmpeg: burnin, white text, black box, position lower-center

CONCAT all segment_paths → video_{timestamp}.mp4
```

---

## Input: `DialogLineMatch`

**File:** `virtual_streamer/agents/sentence_video_matcher/schema.py`

```python
class DialogLineMatch(BaseModel):
    dialog_line: DialogLine      # text, character_id, scene description
    video_path: str              # path to the pre-existing clip to lip-sync
    rating: ContextualRating     # CONTEXTUAL / NEUTRAL / NOT_CONTEXTUAL / FAILURE
    grade: int                   # 0–10 relevance score
    reasoning: str

    @property
    def needs_generation(self) -> bool:
        return self.rating != ContextualRating.CONTEXTUAL
```

`DialogLine` (from `virtual_streamer/video_generation/config.py:401`):

```python
class DialogLine(BaseModel):
    character_id: str           # e.g. "fred" — used for TTS voice selection
    text: str                   # spoken text → TTS input
    scene_description: FluxPrompt  # structured visual description
    location_id: Optional[str]  # references a DB Location entity
```

---

## Configuration: `VideoGenerationConfig`

**File:** `virtual_streamer/video_generation/config.py:193`

The relevant nested object for `script_to_video` is:

```python
class VideoProcessingConfig(BaseModel):
    resolution: str = "720x480"
    codec: str = "h264_nvenc"       # use libx264 if no NVIDIA GPU
    bitrate: str = "3000k"
    fontsize: int = 14              # subtitle text size
```

Top-level fields used by `script_to_video`:

| Field | Default | Used for |
|-------|---------|---------|
| `output_dir` | `"./output"` | final video destination |
| `temp_dir` | `"./temp"` | segment working files |
| `video_processing.fontsize` | `14` | subtitle burn-in |

---

## LTX Fallback: `LTXConfig`

**File:** `virtual_streamer/video_generation/config.py`

When a matched clip is not `CONTEXTUAL`, the function can generate a fresh clip
with LTX-2 instead of using the library clip:

```python
class LTXConfig(BaseModel):
    server_url: str = "http://gx10-cbc5:8082"   # WanGP REST server
    timeout: float = 600.0
    width: int = 1280
    height: int = 720
    duration_seconds: float = 5.0
    fps: int = 24
    steps: int = 8
    cfg_scale: float = 4.0
    style_suffix: str = "Cinematic quality, smooth motion, natural lighting."

    def to_video_params(self, prompt: str = "", **overrides) -> VideoGenerationParams: ...
```

The fallback generates a **text-to-video** clip using `scene_description.to_prompt() + style_suffix`.

---

## WebserviceClient

**File:** `virtual_streamer/api/clients/webservice_client.py`

```python
class APIConfig:
    base_url: str = "http://localhost:8000"   # set via API_BASE_URL env var
    timeout: float = 120.0

class WebserviceClient:
    async def generate_tts(
        text: str, character_id: Optional[str], entry_id: str
    ) -> str  # → local .wav path

    async def generate_wav2lip(
        audio_path: str, video_path: str,
        character_id: Optional[str], output_dir: Optional[str]
    ) -> str  # → local .mp4 path

    async def transcribe_to_srt(audio_path: str) -> str  # → local .srt path
```

The client is used as an async context manager (`async with WebserviceClient(...) as client`).

---

## Debug Artifact Uploads

When `debug_upload_prefix` is set, every intermediate file is uploaded to MinIO:

```
{debug_upload_prefix}/
├── ltx_fallback/segment_{i}.mp4    (only when LTX fallback triggered)
├── tts/segment_{i}.wav
├── wav2lip/segment_{i}.mp4
├── combined/segment_{i}.mp4        (video + audio, before subtitles)
└── subtitles/segment_{i}.srt
```

Upload failures are logged but do not stop the pipeline.

The caller (`_run_video_generation`) also uploads a **GenerationBlueprint** to
`debug/{api_endpoint}/{story_template_id}/{job_id}/blueprint.json` when
`enable_blueprint_dump=True`. The blueprint captures the full story output and
all video match decisions before `script_to_video` runs.

---

## API Entry Points

### Traditional pipeline — `POST /api/v1/video-generation/submit`

```json
{
  "title": "A ghost haunts the town hall",
  "story_template_id": "horror_short",

  "enable_ltx_fallback": false,
  "ltx_server_url": "http://gx10-cbc5:8082",
  "ltx_timeout": 600.0,

  "output_dir": "./output",
  "max_parallel_llm_calls": 5,
  "verbose": false,
  "enable_blueprint_dump": false,

  "llm_provider": "anthropic",
  "llm_model": "claude-sonnet-4-5-20250929"
}
```

`story_template_id` is **required** — it selects which Qdrant video collection
the `SentenceVideoMatcher` searches and which characters/voices are used.
Exactly one of `title` or `story_text` must be provided.

### LTX-2 pipeline — `POST /api/v1/video-generation/generate-ltx`

```json
{
  "title": "A robot discovers music",
  "story_template_id": "sci_fi",

  "ltx_server_url": "http://gx10-cbc5:8082",
  "ltx_timeout": 600.0,

  "video_width": 1280,
  "video_height": 720,
  "video_duration_seconds": 5.0,
  "video_fps": 24,
  "video_steps": 20,
  "video_cfg_scale": 4.0,
  "video_seed": -1,
  "enable_audio": true,

  "tts_host": "tts",
  "tts_port": 8003,
  "adapt_duration_to_audio": true,

  "style_suffix": "Cinematic quality, smooth motion, natural lighting.",
  "enable_debug_dump": true
}
```

> **Docker Compose note:** Inside the compose stack the Fish-Speech TTS service
> is always `tts:8003`, never `localhost:8003`. The `FISH_TTS_HOST` / `FISH_TTS_PORT`
> environment variables drive the default values.

### Single clip — `POST /api/v1/video-generation/single-clip`

Low-level endpoint that bypasses story and matching, sending one set of
generation params directly to the WanGP server.

| Mode | Required inputs |
|------|----------------|
| Text-to-video | `prompt` |
| Image-to-video | `prompt` + `image` file |
| Audio-conditioned i2v | `prompt` + `image` + `audio` |
| Video-to-video | `prompt` + `video` |
| V2V + pinned frame | `prompt` + `video` + `image` |

Quality presets: `quality_preset = fast | quality | high_quality`
(sets `model_type`, `steps`, `fps`; explicit fields override the preset).

---

## Job Lifecycle

All three endpoints are asynchronous. They return a `job_id` immediately:

```
POST /submit  →  { job_id, status: "pending" }

GET  /jobs/{job_id}  →  { status: "pending|running|completed|failed", result?, error? }

result (on success):
  {
    "video_path": "...",
    "metadata": {
      "minio_video_key": "generated_videos/{collection}/{job_id}.mp4",
      "video_url": "https://...",
      "total_duration": 42.3,
      "sentence_count": 6
    },
    "story_output": { ... },
    "video_matches": [ ... ]
  }
```

For the broadcast pipeline (`generate-from-broadcast`), a completed job also
triggers an automatic playlist insert — the `entry_id` is appended to `result`.

Queue limit: **5 pending jobs** per `story_template_id` (bypassed with
`skip_queue_limit: true` — admin only).

---

## Error Behaviour

| Pipeline | On segment failure |
|----------|--------------------|
| `script_to_video` (Traditional) | Raises immediately — entire job fails |
| `scenes_to_video` (LTX-2) | Logs warning, skips segment, continues |

If `scenes_to_video` skips every segment it raises `RuntimeError("All N segment(s) failed")`.

---

## Key External Dependencies

| Service | Default address | Purpose |
|---------|----------------|---------|
| Fish-Speech TTS | `tts:8003` | Text-to-speech audio |
| Wav2Lip | via `WebserviceClient` | Lip-sync a clip to audio |
| Whisper STT | via `WebserviceClient` | Transcribe audio → SRT |
| WanGP LTX-2 | `gx10-cbc5:8082` | Text/image/audio-to-video |
| Stable Diffusion cpp | `gx10-cbc5:1234` | Conditioning images for LTX-2 |
| MinIO | (env-driven) | Store generated videos + debug artifacts |
| Qdrant | (env-driven) | Vector search for video clip matching |
