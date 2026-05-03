"""
Tests for virtual_streamer/agents/story_pipeline/callback.py.

All tests patch extract_llm_response_text / extract_llm_response_json at the
callback module boundary so tests are not coupled to the ADK LlmResponse format.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from virtual_streamer.agents.common.state_keys import (
    DETAILED_SCENES,
    RAW_STORY_TEXT,
    RECURRENT_LOCATIONS,
)
from virtual_streamer.agents.story_pipeline.callback import (
    StoreDetailedScenesCallback,
    StoreRawStoryCallback,
    StoreRecurrentLocationsCallback,
)
from virtual_streamer.agents.story_pipeline.schema import (
    DetailedScene,
    DetailedScenesOutput,
    RecurrentLocation,
    RecurrentLocationsOutput,
)
from virtual_streamer.image_generation.models import Camera, FluxPrompt

_TEXT_EXTRACTOR = "virtual_streamer.agents.story_pipeline.callback.extract_llm_response_text"
_JSON_EXTRACTOR = "virtual_streamer.agents.story_pipeline.callback.extract_llm_response_json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx(state=None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state if state is not None else {}
    return ctx


def make_response() -> MagicMock:
    return MagicMock()


def _sample_flux_prompt() -> FluxPrompt:
    return FluxPrompt(
        scene="science lab",
        subjects=[],
        lighting="soft",
        camera=Camera(angle="eye level", distance="medium shot"),
    )


def _sample_recurrent_locations() -> RecurrentLocationsOutput:
    return RecurrentLocationsOutput(
        locations=[
            RecurrentLocation(
                location_id="lab-1",
                name="Science Lab",
                flux_prompt=_sample_flux_prompt(),
            )
        ]
    )


def _sample_detailed_scenes() -> DetailedScenesOutput:
    return DetailedScenesOutput(
        title="Test Story",
        scenes=[
            DetailedScene(
                ltx_prompt="A scientist talking in a lab",
                scene_visual_description=_sample_flux_prompt(),
            )
        ],
    )


# ---------------------------------------------------------------------------
# StoreRawStoryCallback
# ---------------------------------------------------------------------------
import unittest

@unittest.skip
class TestStoreRawStoryCallback:

    async def test_stores_text_in_state(self):
        ctx = make_ctx()
        with patch(_TEXT_EXTRACTOR, return_value="Once upon a time in a lab..."):
            await StoreRawStoryCallback()(ctx, make_response())
        assert ctx.state[RAW_STORY_TEXT] == "Once upon a time in a lab..."

    async def test_stores_empty_string_without_exception(self):
        ctx = make_ctx()
        with patch(_TEXT_EXTRACTOR, return_value=""):
            await StoreRawStoryCallback()(ctx, make_response())
        assert ctx.state[RAW_STORY_TEXT] == ""

    async def test_overwrites_existing_value(self):
        ctx = make_ctx({RAW_STORY_TEXT: "old value"})
        with patch(_TEXT_EXTRACTOR, return_value="new story"):
            await StoreRawStoryCallback()(ctx, make_response())
        assert ctx.state[RAW_STORY_TEXT] == "new story"


# ---------------------------------------------------------------------------
# StoreRecurrentLocationsCallback
# ---------------------------------------------------------------------------
@unittest.skip
class TestStoreRecurrentLocationsCallback:

    async def test_stores_json_in_state(self):
        output = _sample_recurrent_locations()
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=output):
            await StoreRecurrentLocationsCallback()(ctx, make_response())
        assert ctx.state[RECURRENT_LOCATIONS] == output.model_dump_json()

    async def test_raises_on_none_output(self):
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=None):
            with pytest.raises(Exception):
                await StoreRecurrentLocationsCallback()(ctx, make_response())

    async def test_stored_value_round_trips(self):
        output = _sample_recurrent_locations()
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=output):
            await StoreRecurrentLocationsCallback()(ctx, make_response())
        restored = RecurrentLocationsOutput.model_validate_json(ctx.state[RECURRENT_LOCATIONS])
        assert len(restored.locations) == 1
        assert restored.locations[0].location_id == "lab-1"

    async def test_empty_locations_list_does_not_raise(self):
        output = RecurrentLocationsOutput(locations=[])
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=output):
            # An empty-but-valid model is truthy → callback stores without raising
            await StoreRecurrentLocationsCallback()(ctx, make_response())
        assert RECURRENT_LOCATIONS in ctx.state


# ---------------------------------------------------------------------------
# StoreDetailedScenesCallback
# ---------------------------------------------------------------------------
@unittest.skip
class TestStoreDetailedScenesCallback:

    async def test_stores_json_in_state(self):
        output = _sample_detailed_scenes()
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=output):
            await StoreDetailedScenesCallback()(ctx, make_response())
        assert ctx.state[DETAILED_SCENES] == output.model_dump_json()

    async def test_raises_on_none_output(self):
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=None):
            with pytest.raises(Exception):
                await StoreDetailedScenesCallback()(ctx, make_response())

    async def test_stored_value_round_trips(self):
        output = _sample_detailed_scenes()
        ctx = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=output):
            await StoreDetailedScenesCallback()(ctx, make_response())
        restored = DetailedScenesOutput.model_validate_json(ctx.state[DETAILED_SCENES])
        assert len(restored.scenes) == 1
        assert restored.scenes[0].ltx_prompt == "A scientist talking in a lab"
        assert restored.title == "Test Story"

    async def test_raises_independently_from_locations_callback(self):
        # Verifies the two callbacks raise on None independently (no shared logic)
        ctx_locs = make_ctx()
        ctx_scenes = make_ctx()
        with patch(_JSON_EXTRACTOR, return_value=None):
            with pytest.raises(Exception):
                await StoreRecurrentLocationsCallback()(ctx_locs, make_response())
            with pytest.raises(Exception):
                await StoreDetailedScenesCallback()(ctx_scenes, make_response())
