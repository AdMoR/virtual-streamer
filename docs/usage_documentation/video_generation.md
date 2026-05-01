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
