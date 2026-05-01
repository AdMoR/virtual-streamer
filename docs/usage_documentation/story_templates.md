# Story Templates

Story templates define the characters, prompts, and video collections used when generating content. You must know the `story_template_id` before generating a video.

---

## Browse available templates

To list all registered story templates:

```
list_story_templates()
```

To get the details of a specific template:

```
get_story_template(template_id="cest_pas_sorcier")
```

All templates are also available as a resource:

```
virtual-streamer://templates/list
```

---

## Create a new story template

To generate a new story template from a creative concept, use `create_story_template`. This runs a multi-step LLM pipeline (guardrail → writer → formatter) and registers the result in the database.

```
create_story_template(
    story_concept="A parody documentary where an AI tries to understand why humans love eating cereal at midnight."
)
```

The returned object contains the `template_id` (derived from the generated name, e.g. `midnight_cereal_ai`), which you can immediately pass to `create_video`.

**Note:** This call takes 30–60 seconds. Do not use it during time-sensitive interactions.

Fixed constraints applied by this tool:
- `collection`: always `"random"` (generic video pool)
- `character`: always `"narrator"` (off-screen voice, no character asset)

---

## List available characters

To see all characters with their voice samples and video clips, use `list_characters`.

```
list_characters()
```

Character IDs from this list can be passed to `greet_viewer` and `answer_viewer_question`.
