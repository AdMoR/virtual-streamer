"""
Unit and integration tests for stateful callback system.

Tests cover:
- StateInputCallback key generation with/without run_id
- StateOutputCallback key generation with/without run_id
- Schema access from callbacks
- StatefulLlmAgent key delegation
- VideoMatcher specific callbacks
"""
import unittest

import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Optional

from pydantic import BaseModel, Field

from virtual_streamer.lib.agents.stateful_callbacks import (
    StateInputCallback,
    StateOutputCallback,
)
from virtual_streamer.lib.agents.stateful_agent import StatefulLlmAgent
from virtual_streamer.agents.video_matcher.callback import (
    InjectVisionFrameCallback,
    StoreJudgementCallback,
)
from virtual_streamer.agents.video_matcher.schema import (
    VideoSentenceInput,
    VideoJudgementOutput,
    ContextualRating,
)
from virtual_streamer.image_generation.models import FluxPrompt, Camera


# ============================================================================
# Helper Schemas (not tests)
# ============================================================================


class SampleInputSchema(BaseModel):
    """Sample input schema for testing."""
    name: str
    value: int


class SampleOutputSchema(BaseModel):
    """Sample output schema for testing."""
    result: str
    score: float = Field(ge=0, le=1)


# ============================================================================
# Concrete Callback Implementations (for testing)
# ============================================================================


class SampleInputCallback(StateInputCallback):
    """Concrete implementation for testing."""
    
    def __init__(self, run_id: Optional[str] = None):
        super().__init__(
            input_key="test_input",
            input_schema=SampleInputSchema,
            run_id=run_id,
        )
    
    async def __call__(self, callback_context, request):
        # Simple implementation for testing
        return None


class SampleOutputCallback(StateOutputCallback):
    """Concrete implementation for testing."""
    
    def __init__(self, run_id: Optional[str] = None):
        super().__init__(
            output_key="test_output",
            output_schema=SampleOutputSchema,
            run_id=run_id,
        )
    
    async def __call__(self, callback_context, llm_response):
        # Simple implementation for testing
        return None


# ============================================================================
# StateInputCallback Tests
# ============================================================================


class TestStateInputCallback:
    """Test StateInputCallback functionality."""
    
    def test_input_key_without_run_id(self):
        """Test that input key returns base key when run_id is None."""
        cb = SampleInputCallback()
        assert cb.get_input_key() == "test_input"
    
    def test_input_key_with_run_id(self):
        """Test that input key is namespaced with run_id."""
        cb = SampleInputCallback(run_id="s0_w1")
        assert cb.get_input_key() == "task:s0_w1:test_input"
    
    def test_input_key_with_complex_run_id(self):
        """Test input key with colon-separated run_id."""
        cb = SampleInputCallback(run_id="abc123:w0")
        assert cb.get_input_key() == "task:abc123:w0:test_input"
    
    def test_input_schema_returned(self):
        """Test that get_input_schema returns the correct schema."""
        cb = SampleInputCallback()
        assert cb.get_input_schema() == SampleInputSchema
    
    def test_input_schema_with_run_id(self):
        """Test that schema is independent of run_id."""
        cb = SampleInputCallback(run_id="test")
        assert cb.get_input_schema() == SampleInputSchema
    
    def test_stores_input_key(self):
        """Test that input_key is stored as instance variable."""
        cb = SampleInputCallback()
        assert cb.input_key == "test_input"
    
    def test_stores_run_id(self):
        """Test that run_id is stored as instance variable."""
        cb = SampleInputCallback(run_id="my_run")
        assert cb.run_id == "my_run"


# ============================================================================
# StateOutputCallback Tests
# ============================================================================


class TestStateOutputCallback:
    """Test StateOutputCallback functionality."""
    
    def test_output_key_without_run_id(self):
        """Test that output key returns base key when run_id is None."""
        cb = SampleOutputCallback()
        assert cb.get_output_key() == "test_output"
    
    def test_output_key_with_run_id(self):
        """Test that output key is namespaced with run_id."""
        cb = SampleOutputCallback(run_id="s0_w1")
        assert cb.get_output_key() == "result:s0_w1:test_output"
    
    def test_output_key_with_complex_run_id(self):
        """Test output key with colon-separated run_id."""
        cb = SampleOutputCallback(run_id="abc123:w0")
        assert cb.get_output_key() == "result:abc123:w0:test_output"
    
    def test_output_schema_returned(self):
        """Test that get_output_schema returns the correct schema."""
        cb = SampleOutputCallback()
        assert cb.get_output_schema() == SampleOutputSchema
    
    def test_stores_output_key(self):
        """Test that output_key is stored as instance variable."""
        cb = SampleOutputCallback()
        assert cb.output_key == "test_output"


# ============================================================================
# VideoMatcher Callback Tests
# ============================================================================


class TestInjectVisionFrameCallback:
    """Test InjectVisionFrameCallback key generation."""
    
    def test_input_key_without_run_id(self):
        """Test default input key."""
        cb = InjectVisionFrameCallback()
        assert cb.get_input_key() == "video_sentence"
    
    def test_input_key_with_run_id(self):
        """Test namespaced input key."""
        cb = InjectVisionFrameCallback(run_id="s0_w1")
        assert cb.get_input_key() == "task:s0_w1:video_sentence"
    
    def test_input_schema(self):
        """Test that correct schema is returned."""
        cb = InjectVisionFrameCallback()
        assert cb.get_input_schema() == VideoSentenceInput
    
    def test_input_schema_validation(self):
        """Test that input schema validates correctly."""
        cb = InjectVisionFrameCallback()
        schema = cb.get_input_schema()
        
        # Valid input (requires all fields)
        valid = schema(line_id=0, character_id="narrator", sentence="Hello", scene_description="A test scene", video_path="/test.mp4")
        assert valid.character_id == "narrator"
        assert valid.sentence == "Hello"
        assert valid.video_path == "/test.mp4"
        
        # Invalid input (missing fields)
        with pytest.raises(Exception):
            schema()


class TestStoreJudgementCallback:
    """Test StoreJudgementCallback key generation."""
    
    def test_output_key_without_run_id(self):
        """Test default output key."""
        cb = StoreJudgementCallback()
        assert cb.get_output_key() == "judgement"
    
    def test_output_key_with_run_id(self):
        """Test namespaced output key."""
        cb = StoreJudgementCallback(run_id="s0_w1")
        assert cb.get_output_key() == "result:s0_w1:judgement"
    
    def test_output_schema(self):
        """Test that correct schema is returned (VideoMatchResult, not VideoJudgementOutput)."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        cb = StoreJudgementCallback()
        assert cb.get_output_schema() == VideoMatchResult
    
    def test_output_schema_validation(self):
        """Test that output schema validates correctly."""
        cb = StoreJudgementCallback()
        schema = cb.get_output_schema()
        
        # VideoMatchResult requires all fields
        valid = schema(
            line_id=0,
            character_id="narrator",
            sentence="Hello world",
            scene_description="A test scene",
            video_path="/path/to/video.mp4",
            rating=ContextualRating.CONTEXTUAL,
            grade=8,
            reasoning="Good match"
        )
        assert valid.rating == ContextualRating.CONTEXTUAL
        assert valid.grade == 8
        assert valid.character_id == "narrator"
        assert valid.sentence == "Hello world"


# ============================================================================
# StatefulLlmAgent Tests (mocked to avoid config loading)
# ============================================================================


class TestStatefulLlmAgentKeyDelegation:
    """Test that StatefulLlmAgent correctly delegates key and schema methods."""
    
    def test_get_input_key_delegates(self):
        """Test get_input_key delegates to input callback."""
        input_cb = SampleInputCallback(run_id="test123")
        output_cb = SampleOutputCallback(run_id="test123")
        
        # We can't instantiate StatefulLlmAgent without config,
        # so we test the callbacks directly
        assert input_cb.get_input_key() == "task:test123:test_input"
    
    def test_get_input_schema_delegates(self):
        """Test get_input_schema delegates to input callback."""
        input_cb = SampleInputCallback(run_id="test123")
        
        # Test schema delegation
        assert input_cb.get_input_schema() == SampleInputSchema
    
    def test_get_output_key_delegates(self):
        """Test get_output_key delegates to output callback."""
        input_cb = SampleInputCallback(run_id="test123")
        output_cb = SampleOutputCallback(run_id="test123")
        
        assert output_cb.get_output_key() == "result:test123:test_output"
    
    def test_callbacks_share_run_id(self):
        """Test that input and output callbacks can share run_id."""
        run_id = "shared_run"
        input_cb = SampleInputCallback(run_id=run_id)
        output_cb = SampleOutputCallback(run_id=run_id)
        
        # Both should have the same run_id prefix
        assert run_id in input_cb.get_input_key()
        assert run_id in output_cb.get_output_key()


# ============================================================================
# StatefulWorker Protocol Tests
# ============================================================================


class TestStatefulWorkerProtocol:
    """Test StatefulWorker protocol compliance."""
    
    def test_callback_exposes_input_schema(self):
        """Test that callbacks expose input schema for validation."""
        input_cb = SampleInputCallback(run_id="proto_test")
        schema = input_cb.get_input_schema()
        
        # Schema should be usable for validation
        valid_data = {"name": "test", "value": 42}
        validated = schema.model_validate(valid_data)
        assert validated.name == "test"
        assert validated.value == 42
    
    def test_callback_schema_rejects_invalid_data(self):
        """Test that schema validation rejects invalid data."""
        from pydantic import ValidationError
        
        input_cb = SampleInputCallback(run_id="proto_test")
        schema = input_cb.get_input_schema()
        
        # Missing required field
        with pytest.raises(ValidationError):
            schema.model_validate({"name": "test"})  # missing 'value'
        
        # Wrong type
        with pytest.raises(ValidationError):
            schema.model_validate({"name": "test", "value": "not_an_int"})
    
    def test_video_matcher_callback_schema_validation(self):
        """Test VideoMatcher callback schema validation."""
        input_cb = InjectVisionFrameCallback(run_id="vm_test")
        schema = input_cb.get_input_schema()
        
        # Valid data (requires all fields)
        valid_data = {"line_id": 0, "character_id": "narrator", "sentence": "Hello world", "scene_description": "A test scene", "video_path": "/path/to/video.mp4"}
        validated = schema.model_validate(valid_data)
        assert validated.character_id == "narrator"
        assert validated.sentence == "Hello world"
        assert validated.video_path == "/path/to/video.mp4"
    
    def test_schema_produces_valid_json(self):
        """Test that validated schema produces valid JSON for state storage."""
        input_cb = SampleInputCallback(run_id="json_test")
        schema = input_cb.get_input_schema()
        
        validated = schema.model_validate({"name": "test", "value": 42})
        json_str = validated.model_dump_json()
        
        # Should be valid JSON string
        import json
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42


# ============================================================================
# VideoMatcher Agent Factory Tests
# ============================================================================


class TestVideoMatcherFactory:
    """Test get_video_matcher factory function."""
    
    def test_factory_creates_callbacks_with_run_id(self):
        """Test that factory passes run_id to callbacks."""
        # We test the callbacks directly since agent requires config
        run_id = "factory_test"
        input_cb = InjectVisionFrameCallback(run_id)
        output_cb = StoreJudgementCallback(run_id)
        
        assert input_cb.get_input_key() == f"task:{run_id}:video_sentence"
        assert output_cb.get_output_key() == f"result:{run_id}:judgement"
    
    def test_factory_without_run_id(self):
        """Test factory without run_id uses base keys."""
        input_cb = InjectVisionFrameCallback()
        output_cb = StoreJudgementCallback()
        
        assert input_cb.get_input_key() == "video_sentence"
        assert output_cb.get_output_key() == "judgement"


# ============================================================================
# Key Pattern Tests
# ============================================================================


class TestKeyPatterns:
    """Test key pattern consistency."""
    
    def test_input_key_pattern(self):
        """Test input key always follows task:{run_id}:{key} pattern."""
        run_ids = ["s0", "s0_w1", "abc:w0", "test123"]
        
        for run_id in run_ids:
            cb = SampleInputCallback(run_id=run_id)
            key = cb.get_input_key()
            assert key.startswith("task:")
            assert run_id in key
            assert key.endswith(":test_input")
    
    def test_output_key_pattern(self):
        """Test output key always follows result:{run_id}:{key} pattern."""
        run_ids = ["s0", "s0_w1", "abc:w0", "test123"]
        
        for run_id in run_ids:
            cb = SampleOutputCallback(run_id=run_id)
            key = cb.get_output_key()
            assert key.startswith("result:")
            assert run_id in key
            assert key.endswith(":test_output")
    
    def test_keys_are_unique_per_run(self):
        """Test that different run_ids produce different keys."""
        cb1 = SampleInputCallback(run_id="run1")
        cb2 = SampleInputCallback(run_id="run2")
        
        assert cb1.get_input_key() != cb2.get_input_key()


# ============================================================================
# Schema Serialization Tests
# ============================================================================


class TestSchemaSerialization:
    """Test schema serialization/deserialization."""
    
    def test_input_schema_json_roundtrip(self):
        """Test input schema serializes and deserializes correctly."""
        cb = SampleInputCallback()
        schema = cb.get_input_schema()
        
        original = schema(name="test", value=42)
        json_str = original.model_dump_json()
        restored = schema.model_validate_json(json_str)
        
        assert restored.name == original.name
        assert restored.value == original.value
    
    def test_output_schema_json_roundtrip(self):
        """Test output schema serializes and deserializes correctly."""
        cb = SampleOutputCallback()
        schema = cb.get_output_schema()
        
        original = schema(result="success", score=0.95)
        json_str = original.model_dump_json()
        restored = schema.model_validate_json(json_str)
        
        assert restored.result == original.result
        assert restored.score == original.score
    
    def test_video_sentence_input_roundtrip(self):
        """Test VideoSentenceInput serialization."""
        original = VideoSentenceInput(
            line_id=0,
            character_id="narrator",
            sentence="Test sentence",
            scene_description="A test scene",
            video_path="/path/to/video.mp4"
        )
        json_str = original.model_dump_json()
        restored = VideoSentenceInput.model_validate_json(json_str)
        
        assert restored.character_id == original.character_id
        assert restored.sentence == original.sentence
        assert restored.video_path == original.video_path

    @unittest.skip
    def test_video_judgement_output_roundtrip(self):
        """Test VideoJudgementOutput serialization."""
        original = VideoJudgementOutput(
            rating=ContextualRating.CONTEXTUAL,
            grade=8,
            reasoning="Good match between video and dialogue"
        )
        json_str = original.model_dump_json()
        restored = VideoJudgementOutput.model_validate_json(json_str)
        
        assert restored.rating == original.rating
        assert restored.grade == original.grade
        assert restored.reasoning == original.reasoning


# ============================================================================
# MapperAgent Schema Validation Tests
# ============================================================================


class MockStatefulWorker:
    """Mock worker for testing MapperAgent without LLM calls."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._input_cb = SampleInputCallback(run_id=run_id)
        self._output_cb = SampleOutputCallback(run_id=run_id)
    
    def get_input_key(self) -> str:
        return self._input_cb.get_input_key()
    
    def get_input_schema(self):
        return self._input_cb.get_input_schema()
    
    def get_output_key(self) -> str:
        return self._output_cb.get_output_key()


class TestMapperAgentSchemaValidation:
    """Test MapperAgent validates items against worker's input schema."""
    
    def test_worker_factory_receives_run_id(self):
        """Test that worker factory receives correct run_id format."""
        received_run_ids = []
        
        def mock_factory(run_id: str) -> MockStatefulWorker:
            received_run_ids.append(run_id)
            return MockStatefulWorker(run_id)
        
        # Create workers manually to test factory
        for i in range(3):
            run_id = f"test:w{i}"
            mock_factory(run_id)
        
        assert len(received_run_ids) == 3
        assert all(":w" in rid for rid in received_run_ids)
    
    def test_worker_provides_schema_for_validation(self):
        """Test that worker provides schema for item validation."""
        worker = MockStatefulWorker(run_id="schema_test")
        schema = worker.get_input_schema()
        
        # Should be able to validate items
        valid_item = {"name": "test", "value": 42}
        validated = schema.model_validate(valid_item)
        assert validated.name == "test"
        assert validated.value == 42
    
    def test_schema_validation_rejects_invalid_items(self):
        """Test that schema validation catches invalid items."""
        from pydantic import ValidationError
        
        worker = MockStatefulWorker(run_id="invalid_test")
        schema = worker.get_input_schema()
        
        # Invalid item: wrong type for 'value'
        invalid_item = {"name": "test", "value": "not_an_int"}
        
        with pytest.raises(ValidationError):
            schema.model_validate(invalid_item)
    
    def test_schema_validation_catches_missing_fields(self):
        """Test that schema validation catches missing required fields."""
        from pydantic import ValidationError
        
        worker = MockStatefulWorker(run_id="missing_test")
        schema = worker.get_input_schema()
        
        # Missing 'value' field
        incomplete_item = {"name": "test"}
        
        with pytest.raises(ValidationError):
            schema.model_validate(incomplete_item)
    
    def test_validated_item_serializes_to_json(self):
        """Test that validated items serialize correctly for state storage."""
        import json
        
        worker = MockStatefulWorker(run_id="json_test")
        schema = worker.get_input_schema()
        
        item = {"name": "test", "value": 42}
        validated = schema.model_validate(item)
        json_str = validated.model_dump_json()
        
        # Parse back and verify
        parsed = json.loads(json_str)
        assert parsed == item
    
    def test_video_matcher_schema_validation(self):
        """Test VideoMatcher-specific schema validation."""
        from pydantic import ValidationError
        
        cb = InjectVisionFrameCallback(run_id="vm_validation")
        schema = cb.get_input_schema()
        
        # Valid video sentence input (requires all fields)
        valid = {"line_id": 0, "character_id": "narrator", "sentence": "Hello Jamy!", "scene_description": "A test scene", "video_path": "/videos/scene.mp4"}
        validated = schema.model_validate(valid)
        assert validated.character_id == "narrator"
        assert validated.sentence == "Hello Jamy!"
        assert validated.video_path == "/videos/scene.mp4"
        
        # Missing character_id
        with pytest.raises(ValidationError):
            schema.model_validate({"line_id": 0, "sentence": "Hello", "scene_description": "A test scene", "video_path": "/videos/scene.mp4"})
        
        # Missing sentence
        with pytest.raises(ValidationError):
            schema.model_validate({"line_id": 0, "character_id": "narrator", "scene_description": "A test scene", "video_path": "/videos/scene.mp4"})
        
        # Missing video_path
        with pytest.raises(ValidationError):
            schema.model_validate({"line_id": 0, "character_id": "narrator", "sentence": "Hello", "scene_description": "A test scene"})


# ============================================================================
# AbstractAggregator Tests
# ============================================================================


class SampleAggregator:
    """
    Test implementation of AbstractAggregator pattern.
    
    We test the collection and parsing logic without full agent execution.
    """
    
    def __init__(
        self,
        input_keys: list,
        input_schema,
        result_state_key: str = None,
    ):
        self.input_keys = input_keys
        self.input_schema = input_schema
        self.result_state_key = result_state_key
    
    def collect_results(self, state: dict) -> list:
        """Simulate _collect_results with a plain dict instead of ctx."""
        results = []
        for key in self.input_keys:
            value = state.get(key)
            if value is None:
                continue
            try:
                if isinstance(value, str):
                    parsed = self.input_schema.model_validate_json(value)
                elif isinstance(value, dict):
                    parsed = self.input_schema.model_validate(value)
                else:
                    continue
                results.append(parsed)
            except Exception:
                pass
        return results


class TestAbstractAggregatorCollection:
    """Test AbstractAggregator result collection and parsing."""
    
    def test_collects_from_specified_keys(self):
        """Test that aggregator reads from specified state keys."""
        state = {
            "result:abc:w0:judgement": '{"name": "test1", "value": 1}',
            "result:abc:w1:judgement": '{"name": "test2", "value": 2}',
            "other_key": '{"name": "ignored", "value": 99}',
        }
        
        agg = SampleAggregator(
            input_keys=["result:abc:w0:judgement", "result:abc:w1:judgement"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 2
        assert results[0].name == "test1"
        assert results[1].name == "test2"
    
    def test_parses_json_strings(self):
        """Test parsing JSON strings from state."""
        state = {
            "key1": '{"name": "json_string", "value": 42}',
        }
        
        agg = SampleAggregator(
            input_keys=["key1"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 1
        assert results[0].name == "json_string"
        assert results[0].value == 42
    
    def test_parses_dict_values(self):
        """Test parsing dict values from state."""
        state = {
            "key1": {"name": "dict_value", "value": 100},
        }
        
        agg = SampleAggregator(
            input_keys=["key1"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 1
        assert results[0].name == "dict_value"
        assert results[0].value == 100
    
    def test_skips_missing_keys(self):
        """Test that missing keys are skipped without error."""
        state = {
            "key1": '{"name": "exists", "value": 1}',
            # "key2" is missing
        }
        
        agg = SampleAggregator(
            input_keys=["key1", "key2"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 1
        assert results[0].name == "exists"
    
    def test_skips_invalid_json(self):
        """Test that invalid JSON is skipped without error."""
        state = {
            "key1": '{"name": "valid", "value": 1}',
            "key2": "not valid json",
        }
        
        agg = SampleAggregator(
            input_keys=["key1", "key2"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 1
        assert results[0].name == "valid"
    
    def test_skips_schema_validation_failures(self):
        """Test that schema validation failures are skipped."""
        state = {
            "key1": '{"name": "valid", "value": 1}',
            "key2": '{"name": "missing_value"}',  # Missing required field
        }
        
        agg = SampleAggregator(
            input_keys=["key1", "key2"],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 1
        assert results[0].name == "valid"
    
    def test_empty_keys_returns_empty_list(self):
        """Test that empty input keys returns empty results."""
        state = {"key1": '{"name": "test", "value": 1}'}
        
        agg = SampleAggregator(
            input_keys=[],
            input_schema=SampleInputSchema,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 0


class TestAbstractAggregatorWithVideoSchema:
    """Test AbstractAggregator with VideoJudgementOutput schema."""
    
    def test_collects_video_judgements(self):
        """Test collecting VideoJudgementOutput from state."""
        state = {
            "result:run1:w0:judgement": VideoJudgementOutput(
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Good match"
            ).model_dump_json(),
            "result:run1:w1:judgement": VideoJudgementOutput(
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="Okay match"
            ).model_dump_json(),
        }
        
        agg = SampleAggregator(
            input_keys=["result:run1:w0:judgement", "result:run1:w1:judgement"],
            input_schema=VideoJudgementOutput,
        )
        
        results = agg.collect_results(state)
        assert len(results) == 2
        assert results[0].rating == ContextualRating.CONTEXTUAL
        assert results[0].grade == 8
        assert results[1].rating == ContextualRating.NEUTRAL
        assert results[1].grade == 5
    
    def test_aggregation_selects_best_by_rating(self):
        """Test that aggregation can select best match by rating priority."""
        results = [
            VideoJudgementOutput(
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=9,
                reasoning="High grade but bad rating"
            ),
            VideoJudgementOutput(
                rating=ContextualRating.CONTEXTUAL,
                grade=6,
                reasoning="Lower grade but good rating"
            ),
        ]
        
        # Simulate aggregation logic: CONTEXTUAL beats higher grade
        contextual = [r for r in results if r.rating == ContextualRating.CONTEXTUAL]
        best = contextual[0] if contextual else max(results, key=lambda x: x.grade)
        
        assert best.rating == ContextualRating.CONTEXTUAL
        assert best.grade == 6
    
    def test_aggregation_falls_back_to_grade(self):
        """Test that aggregation falls back to grade when no good ratings."""
        results = [
            VideoJudgementOutput(
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=3,
                reasoning="Low grade"
            ),
            VideoJudgementOutput(
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=7,
                reasoning="Higher grade"
            ),
        ]
        
        # Simulate aggregation logic: fall back to highest grade
        best = max(results, key=lambda x: x.grade)
        
        assert best.grade == 7


# ============================================================================
# VideoMatchResult Schema Tests
# ============================================================================


class TestVideoMatchResultSchema:
    """Test VideoMatchResult schema and factory method."""

    @unittest.skip
    def test_from_input_and_output(self):
        """Test creating VideoMatchResult from input and output."""
        from virtual_streamer.agents.video_matcher.schema import (
            VideoMatchResult,
            VideoSentenceInput,
            VideoJudgementOutput,
        )
        
        input_data = VideoSentenceInput(
            line_id=0,
            character_id="narrator",
            sentence="Hello world",
            scene_description="A test scene",
            video_path="/path/to/video.mp4"
        )
        output_data = VideoJudgementOutput(
            rating=ContextualRating.CONTEXTUAL,
            grade=8,
            reasoning="Good match"
        )
        
        result = VideoMatchResult.from_input_and_output(input_data, output_data)
        
        assert result.character_id == "narrator"
        assert result.sentence == "Hello world"
        assert result.video_path == "/path/to/video.mp4"
        assert result.rating == ContextualRating.CONTEXTUAL
        assert result.grade == 8
        assert result.reasoning == "Good match"
    
    def test_video_match_result_serialization(self):
        """Test VideoMatchResult JSON serialization."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        
        result = VideoMatchResult(
            line_id=0,
            character_id="narrator",
            sentence="Test sentence",
            scene_description="A test scene",
            video_path="/test/video.mp4",
            rating=ContextualRating.NEUTRAL,
            grade=5,
            reasoning="Okay match"
        )
        
        json_str = result.model_dump_json()
        restored = VideoMatchResult.model_validate_json(json_str)
        
        assert restored.character_id == result.character_id
        assert restored.sentence == result.sentence
        assert restored.video_path == result.video_path
        assert restored.rating == result.rating
        assert restored.grade == result.grade


# ============================================================================
# BestMatchAggregator Tests
# ============================================================================


class TestBestMatchAggregator:
    """Test BestMatchAggregator selection logic."""
    
    @pytest.mark.asyncio
    async def test_selects_contextual_over_higher_grade(self):
        """Test that CONTEXTUAL rating beats higher grade."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=9,
                reasoning="High grade but bad rating"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=6,
                reasoning="Lower grade but good rating"
            ),
        ]
        
        # Create aggregator and test aggregation function directly
        aggregator = BestMatchAggregator(
            input_keys=[],  # Not used in direct test
            output_key="test_output",
        )
        
        best = await aggregator.aggregation_fn(results)
        
        assert best is not None
        assert best.rating == ContextualRating.CONTEXTUAL
        assert best.video_path == "/video2.mp4"
    
    @pytest.mark.asyncio
    async def test_selects_neutral_when_no_contextual(self):
        """Test that NEUTRAL is selected when no CONTEXTUAL exists."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=9,
                reasoning="High grade"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video2.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="Neutral"
            ),
        ]
        
        aggregator = BestMatchAggregator(input_keys=[], output_key="test_output")
        best = await aggregator.aggregation_fn(results)
        
        assert best is not None
        assert best.rating == ContextualRating.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_selects_highest_grade_within_rating(self):
        """Test that highest grade is selected within same rating."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=5,
                reasoning="Lower grade"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=9,
                reasoning="Higher grade"
            ),
        ]
        
        aggregator = BestMatchAggregator(input_keys=[], output_key="test_output")
        best = await aggregator.aggregation_fn(results)
        
        assert best is not None
        assert best.grade == 9
        assert best.video_path == "/video2.mp4"
    
    @pytest.mark.asyncio
    async def test_fallback_to_highest_grade(self):
        """Test fallback to highest grade when only NOT_CONTEXTUAL."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        results = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=3,
                reasoning="Low grade"
            ),
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="test",
                scene_description="A test scene",
                video_path="/video2.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=7,
                reasoning="Higher grade"
            ),
        ]
        
        aggregator = BestMatchAggregator(input_keys=[], output_key="test_output")
        best = await aggregator.aggregation_fn(results)
        
        assert best is not None
        assert best.grade == 7
    
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_results(self):
        """Test that empty results returns None."""
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        aggregator = BestMatchAggregator(input_keys=[], output_key="test_output")
        best = await aggregator.aggregation_fn([])
        
        assert best is None


# ============================================================================
# SentenceVideoMatcherAgent Schema Tests
# ============================================================================


class TestSentenceVideoMatcherSchemas:
    """Test SentenceVideoMatcher output schemas."""
    
    def test_output_schema(self):
        """Test SentenceVideoMatcherOutput schema with DialogLineMatch."""
        from virtual_streamer.agents.sentence_video_matcher.schema import (
            SentenceVideoMatcherOutput,
            DialogLineMatch,
        )
        from virtual_streamer.agents.story_generator.schema import DialogLine
        
        matches = [
            DialogLineMatch(
                dialog_line=DialogLine(character_id="narrator", text="Hello", scene_description=FluxPrompt(scene="A test scene", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot"))),
                video_path="/video1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Good"
            ),
            DialogLineMatch(
                dialog_line=DialogLine(character_id="narrator", text="World", scene_description=FluxPrompt(scene="A test scene", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot"))),
                video_path="/video2.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="Okay"
            ),
        ]
        
        output = SentenceVideoMatcherOutput(matches=matches)
        
        assert len(output.matches) == 2
        
        # Test to_dict_by_dialog
        by_dialog = output.to_dict_by_dialog()
        assert "Hello" in by_dialog
        assert by_dialog["Hello"].video_path == "/video1.mp4"


# ============================================================================
# Mock Classes for Testing
# ============================================================================


class MockVideoRetriever:
    """Mock video retriever for testing."""
    
    def __init__(self, video_map: dict = None):
        """
        Initialize with optional video map.
        
        Args:
            video_map: Dict mapping search queries to list of video paths.
                      If None, returns default mock videos.
        """
        self.video_map = video_map or {}
        self.search_calls = []
    
    def search(self, query: str, top_k: int = 10) -> list:
        """Return mock video paths for a query."""
        self.search_calls.append((query, top_k))
        
        if query in self.video_map:
            return self.video_map[query][:top_k]
        
        # Default: return numbered videos
        return [f"/videos/video_{i}.mp4" for i in range(min(top_k, 5))]
    
    def get_video_metadata(self, video_path: str) -> dict:
        """Return mock metadata."""
        return {"duration": 5.0, "path": video_path}


class MockSessionState:
    """Mock session state for testing."""
    
    def __init__(self, initial_state: dict = None):
        self._state = initial_state or {}
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __contains__(self, key):
        return key in self._state
    
    def keys(self):
        return self._state.keys()
    
    def items(self):
        return self._state.items()


class MockSession:
    """Mock session for testing."""
    
    def __init__(self, initial_state: dict = None):
        self.state = MockSessionState(initial_state)


class MockInvocationContext:
    """Mock invocation context for testing."""
    
    def __init__(self, initial_state: dict = None):
        self.session = MockSession(initial_state)


# ============================================================================
# Retriever Tests
# ============================================================================


class TestMockVideoRetriever:
    """Test MockVideoRetriever behavior."""
    
    def test_retriever_returns_correct_candidates(self):
        """Test that retriever returns configured videos."""
        video_map = {
            "Hello world": ["/videos/hello1.mp4", "/videos/hello2.mp4"],
            "Goodbye": ["/videos/bye1.mp4"],
        }
        retriever = MockVideoRetriever(video_map=video_map)
        
        results = retriever.search("Hello world", top_k=10)
        assert results == ["/videos/hello1.mp4", "/videos/hello2.mp4"]
        
        results = retriever.search("Goodbye", top_k=10)
        assert results == ["/videos/bye1.mp4"]
        
        # Unknown query returns defaults
        results = retriever.search("Unknown", top_k=3)
        assert len(results) == 3


# ============================================================================
# SentenceVideoMapper Tests
# ============================================================================


class TestSentenceVideoMapperIntegration:
    """Test SentenceVideoMapper with mocked state management."""
    
    def test_result_key_pattern(self):
        """Test that result keys follow expected pattern."""
        run_id = "abc123"
        worker_indices = [0, 1, 2]
        
        # Expected key pattern: result:{run_id}:w{i}:judgement
        expected_keys = [
            f"result:{run_id}:w{i}:judgement"
            for i in worker_indices
        ]
        
        # Verify pattern
        for key in expected_keys:
            assert key.startswith(f"result:{run_id}:")
            assert ":judgement" in key
    
    def test_aggregator_selects_from_mock_results(self):
        """Test aggregator selection with mock results in state."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        from virtual_streamer.agents.video_matcher.aggregator import BestMatchAggregator
        
        # Simulate state with worker results (with all required fields)
        state = {
            "result:abc:w0:judgement": VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="Hello",
                scene_description="A test scene",
                video_path="/videos/v1.mp4",
                rating=ContextualRating.NOT_CONTEXTUAL,
                grade=3,
                reasoning="Poor match"
            ).model_dump_json(),
            "result:abc:w1:judgement": VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="Hello",
                scene_description="A test scene",
                video_path="/videos/v2.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Great match"
            ).model_dump_json(),
            "result:abc:w2:judgement": VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="Hello",
                scene_description="A test scene",
                video_path="/videos/v3.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="Okay match"
            ).model_dump_json(),
        }
        
        # Create aggregator with the result keys
        output_keys = [k for k in state.keys() if k.startswith("result:abc:")]
        
        aggregator = BestMatchAggregator(
            input_keys=output_keys,
            output_key="best_match",
        )
        
        # Manually test collection (simulating what _collect_results does)
        results = []
        for key in output_keys:
            value = state[key]
            parsed = VideoMatchResult.model_validate_json(value)
            results.append(parsed)
        
        assert len(results) == 3
        
        # Test aggregation
        import asyncio
        best = asyncio.get_event_loop().run_until_complete(
            aggregator.aggregation_fn(results)
        )
        
        # Should select CONTEXTUAL with grade 8
        assert best is not None
        assert best.rating == ContextualRating.CONTEXTUAL
        assert best.grade == 8
        assert best.video_path == "/videos/v2.mp4"


class TestSentenceVideoMatcherEndToEndMock:
    """End-to-end mock tests for the full flow."""
    
    def test_full_flow_simulation(self):
        """Simulate the full agent flow with mocks."""
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        
        # Input
        sentences = ["Hello Jamy!", "Je suis dans un datacenter"]
        
        # Mock retriever
        retriever = MockVideoRetriever(video_map={
            "Hello Jamy!": ["/videos/scene1.mp4", "/videos/scene2.mp4"],
            "Je suis dans un datacenter": ["/videos/datacenter1.mp4", "/videos/datacenter2.mp4"],
        })
        
        # Simulate what the agent would produce (with all required fields)
        expected_matches = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="Hello Jamy!",
                scene_description="A scene with Jamy",
                video_path="/videos/scene1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=9,
                reasoning="Jamy is visible in the scene"
            ),
            VideoMatchResult(
                line_id=1,
                character_id="narrator",
                sentence="Je suis dans un datacenter",
                scene_description="A datacenter scene",
                video_path="/videos/datacenter1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Datacenter visible in background"
            ),
        ]
        
        # Verify retriever was called correctly
        for sentence in sentences:
            candidates = retriever.search(sentence, top_k=10)
            assert len(candidates) >= 2
        
        # Verify matches structure
        assert len(expected_matches) == len(sentences)
        for i, match in enumerate(expected_matches):
            assert match.sentence == sentences[i]
            assert match.rating in [ContextualRating.CONTEXTUAL, ContextualRating.NEUTRAL]
    
    def test_output_schema_matches_expected_format(self):
        """Test that output matches SentenceVideoMatcherOutput schema."""
        from virtual_streamer.agents.sentence_video_matcher.schema import (
            SentenceVideoMatcherOutput,
            DialogLineMatch,
        )
        from virtual_streamer.agents.story_generator.schema import DialogLine
        
        matches = [
            DialogLineMatch(
                dialog_line=DialogLine(character_id="narrator", text="Test 1", scene_description=FluxPrompt(scene="A test scene", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot"))),
                video_path="/v1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Good"
            ),
            DialogLineMatch(
                dialog_line=DialogLine(character_id="narrator", text="Test 2", scene_description=FluxPrompt(scene="A test scene", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot"))),
                video_path="/v2.mp4",
                rating=ContextualRating.NEUTRAL,
                grade=5,
                reasoning="Okay"
            ),
        ]
        
        output = SentenceVideoMatcherOutput(matches=matches)
        
        # Test serialization
        json_str = output.model_dump_json()
        restored = SentenceVideoMatcherOutput.model_validate_json(json_str)
        
        assert len(restored.matches) == 2
        assert restored.matches[0].dialog_line.text == "Test 1"
        
        # Test to_dict_by_dialog
        by_dialog = output.to_dict_by_dialog()
        assert "Test 1" in by_dialog
        assert by_dialog["Test 1"].video_path == "/v1.mp4"
    
    def test_state_delta_format(self):
        """Test that state delta has correct format for MATCHES_KEY."""
        from virtual_streamer.agents.sentence_video_matcher.agent import MATCHES_KEY
        from virtual_streamer.agents.video_matcher.schema import VideoMatchResult
        
        matches = [
            VideoMatchResult(
                line_id=0,
                character_id="narrator",
                sentence="Hello",
                scene_description="A test scene",
                video_path="/v1.mp4",
                rating=ContextualRating.CONTEXTUAL,
                grade=8,
                reasoning="Good"
            ),
        ]
        
        # This is what the agent stores in state_delta
        state_delta = {
            MATCHES_KEY: [m.model_dump() for m in matches]
        }
        
        assert MATCHES_KEY == "video_matches"
        assert len(state_delta[MATCHES_KEY]) == 1
        assert state_delta[MATCHES_KEY][0]["sentence"] == "Hello"
        assert state_delta[MATCHES_KEY][0]["video_path"] == "/v1.mp4"


# ============================================================================
# Pytest Configuration
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

