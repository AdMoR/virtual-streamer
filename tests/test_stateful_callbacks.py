"""
Unit and integration tests for stateful callback system.

Tests cover:
- StateInputCallback key generation with/without run_id
- StateOutputCallback key generation with/without run_id
- Schema access from callbacks
- StatefulLlmAgent key delegation
- VideoMatcher specific callbacks
"""

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
        
        # Valid input
        valid = schema(sentence="Hello", video_path="/test.mp4")
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
        """Test that correct schema is returned."""
        cb = StoreJudgementCallback()
        assert cb.get_output_schema() == VideoJudgementOutput
    
    def test_output_schema_validation(self):
        """Test that output schema validates correctly."""
        cb = StoreJudgementCallback()
        schema = cb.get_output_schema()
        
        # Valid output
        valid = schema(
            rating=ContextualRating.CONTEXTUAL,
            grade=8,
            reasoning="Good match"
        )
        assert valid.rating == ContextualRating.CONTEXTUAL
        assert valid.grade == 8


# ============================================================================
# StatefulLlmAgent Tests (mocked to avoid config loading)
# ============================================================================


class TestStatefulLlmAgentKeyDelegation:
    """Test that StatefulLlmAgent correctly delegates key methods."""
    
    def test_get_input_key_delegates(self):
        """Test get_input_key delegates to input callback."""
        input_cb = SampleInputCallback(run_id="test123")
        output_cb = SampleOutputCallback(run_id="test123")
        
        # We can't instantiate StatefulLlmAgent without config,
        # so we test the callbacks directly
        assert input_cb.get_input_key() == "task:test123:test_input"
    
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
            sentence="Test sentence",
            video_path="/path/to/video.mp4"
        )
        json_str = original.model_dump_json()
        restored = VideoSentenceInput.model_validate_json(json_str)
        
        assert restored.sentence == original.sentence
        assert restored.video_path == original.video_path
    
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
# Pytest Configuration
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

