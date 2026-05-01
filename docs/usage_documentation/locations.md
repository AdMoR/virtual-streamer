# Locations

Locations define scene backgrounds used for conditioned image-to-video generation with LTX. Each location has a diffusion-model description that generates a conditioning image before the video segment is produced.

Locations must exist in the database **before** running video generation — the pipeline validates that all referenced location IDs are registered.

---

## List locations for a story template

To see which locations are already registered for a given template:

```
list_locations(story_template_id="cest_pas_sorcier")
```

---

## Create a new location

To generate and register a new location, use `create_location`. An LLM pipeline writes a detailed diffusion-model description for the environment and stores it scoped to the story template.

```
create_location(
    location_name="Medieval Castle",
    story_template_id="cest_pas_sorcier"
)
```

The `location_id` is derived automatically from the name: `"Medieval Castle"` → `"medieval-castle"`.

**Note:** This call takes 20–40 seconds. Do not use it during time-sensitive interactions.
