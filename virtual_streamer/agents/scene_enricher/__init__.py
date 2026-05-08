"""Scene Enricher Pipeline: enriches ltx_prompt using reference video frames."""

from virtual_streamer.agents.scene_enricher.agent import (
    SceneEnricherPipeline,
    get_scene_enricher_pipeline,
    run_scene_enricher,
)

__all__ = [
    "SceneEnricherPipeline",
    "get_scene_enricher_pipeline",
    "run_scene_enricher",
]
