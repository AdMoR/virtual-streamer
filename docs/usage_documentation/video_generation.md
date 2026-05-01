# Video Generation

## Generate a video for the current broadcast (recommended)

To generate a video and automatically add it to the active playlist, use `create_video_from_broadcast`. This is the standard path during a live stream — it resolves the active programmation, enforces a queue limit (max 5 pending), and handles playlist insertion.

```
create_video_from_broadcast(title="Why cats knock things off tables")
```

To attribute the generation to a specific user (e.g., a viewer who requested it):

```
create_video_from_broadcast(
    title="Why cats knock things off tables",
    user="CoolViewer42"
)
```

---

## Generate a video with an explicit story template

To generate a video using a specific story template (outside the broadcast context), use `create_video`.

```
create_video(
    title="The mystery of disappearing socks",
    story_template_id="cest_pas_sorcier"
)
```

You can list available templates with `list_story_templates`.

---

## Generate a video with LTX-2 (text-to-video)

To generate a video using the LTX-2 model with synchronized audio, use `create_video_ltx`. This is slower and targets a specific WanGP server.

```
create_video_ltx(
    story_template_id="cest_pas_sorcier",
    title="How black holes eat stars"
)
```

To provide the story text directly instead of generating it from a title:

```
create_video_ltx(
    story_template_id="cest_pas_sorcier",
    story_text="Once upon a time, a black hole was very hungry...",
    video_duration_seconds=8.0
)
```

`title` and `story_text` are mutually exclusive. The call blocks until the video is produced (up to 2 hours). Use `get_job_status` to track progress if you submit via `create_video` instead.

---

## Generate a video from an image (image-to-video)

To animate a still image into a video using LTX-2, use `create_video_ltx_i2v`. The image is used as the first frame; the model generates motion from the text prompt.

```
create_video_ltx_i2v(
    prompt="A person smiling and slowly turning their head to the left, soft daylight",
    image_path="/path/to/character.png"
)
```

Key optional parameters:

```
create_video_ltx_i2v(
    prompt="...",
    image_path="/path/to/character.png",
    resolution="1280x720",      # or "854x480", "1920x1080"
    duration_seconds=4.0,
    steps=20,                   # higher = better quality, slower
    seed=42
)
```

Returns a `job_id` immediately. Poll with `get_job_status` until `status == "completed"`. The result contains `video_b64` (base64-encoded MP4) plus `width`, `height`, `fps`, and `duration_seconds`.

---

## Generate a video from an image guided by audio

To generate a video where the audio track drives the character's motion and expression, use `create_video_ltx_audio_i2v`. Requires both a conditioning image and a WAV audio file.

```
create_video_ltx_audio_i2v(
    prompt="A person speaking expressively, natural head motion, warm lighting",
    image_path="/path/to/character.png",
    audio_path="/path/to/speech.wav"
)
```

Fine-tuning the audio influence:

```
create_video_ltx_audio_i2v(
    prompt="...",
    image_path="/path/to/character.png",
    audio_path="/path/to/speech.wav",
    audio_scale=1.0,      # 0.0–1.0, strength of audio conditioning
    audio_guidance=4.5,   # higher = motion follows audio more strictly
    duration_seconds=4.0
)
```

Returns a `job_id` immediately. Poll with `get_job_status` until `status == "completed"`. The result contains `video_b64` (base64-encoded MP4) plus metadata.

---

## Track a video generation job

To check the status of a video submitted with `create_video`, use `get_job_status`.

```
get_job_status(job_id="abc123")
```

To see all recent jobs:

```
list_jobs(limit=20)
```

---

## Submit viewer feedback on a video

To record a viewer's reaction to a played video, use `submit_feedback`.

```
submit_feedback(
    entry_id="playlist-entry-id",
    user="CoolViewer42",
    feedback="+"
)
```

Feedback is stored against the playlist entry and can be used to evaluate content quality.
