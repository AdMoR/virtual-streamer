"""
Stable abstraction layer for scene/story input to the video generation pipeline.

story_to_video.py works exclusively with SceneInput / StoryInput — it never
imports DetailedScene or DialogLine directly. All agent-format-specific parsing
happens in the factory class methods defined here.

This insulation means an agent schema change only requires updating the
from_* factory method, not the core generation logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from virtual_streamer.agents.story_pipeline.schema import DetailedScene
    from virtual_streamer.video_generation.config import DialogLine


class SceneInput(BaseModel):
    """Stable interface for one video segment's input data.

    story_to_video.py works exclusively with this type — it is deliberately
    decoupled from the volatile DetailedScene / DialogLine agent output formats.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_index: int
    ltx_prompt: str
    speaker_id: Optional[str] = None
    spoken_line: Optional[str] = None
    location_id: Optional[str] = None
    character_ids_on_screen: List[str] = []
    scene_visual_description: Optional[dict] = None  # FluxPrompt.model_dump(), used for SD conditioning
    raw_scene_data: dict  # full model_dump() of source agent object — for DB storage and replay


class StoryInput(BaseModel):
    """Stable wrapper around story-level data consumed by the pipeline."""

    title: str
    story_plan: str  # RAW_STORY_TEXT from pipeline state, or story_plan from StoryOutput
    story_template_id: Optional[str] = None
    raw_agent_output: dict  # full model_dump() of the source agent object — for replay
    scenes: List[SceneInput]


class DetailedSceneInput(SceneInput):
    """Concrete SceneInput parsed from a DetailedScene (3-step pipeline)."""

    @classmethod
    def from_detailed_scene(cls, scene: "DetailedScene", index: int) -> "DetailedSceneInput":
        return cls(
            scene_index=index,
            ltx_prompt=scene.ltx_prompt,
            speaker_id=scene.speaker_id,
            spoken_line=scene.spoken_line,
            location_id=scene.location,  # field name differs in DetailedScene
            character_ids_on_screen=scene.character_on_screen or [],
            scene_visual_description=scene.scene_visual_description.model_dump(by_alias=True),
            raw_scene_data=scene.model_dump(by_alias=True),
        )


class DialogLineInput(SceneInput):
    """Concrete SceneInput parsed from a DialogLine (legacy pipeline)."""

    @classmethod
    def from_dialog_line(
        cls, dialog_line: "DialogLine", index: int, built_prompt: str
    ) -> "DialogLineInput":
        return cls(
            scene_index=index,
            ltx_prompt=built_prompt,
            speaker_id=dialog_line.character_id,
            spoken_line=dialog_line.text,
            location_id=dialog_line.location_id,
            character_ids_on_screen=[dialog_line.character_id],
            scene_visual_description=(
                dialog_line.scene_description.model_dump(by_alias=True)
                if dialog_line.scene_description
                else None
            ),
            raw_scene_data=dialog_line.model_dump(by_alias=True),
        )
