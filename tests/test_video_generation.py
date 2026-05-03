"""
Unit and integration tests for video generation script.

Tests cover:
- Configuration loading and validation
- Interface implementations with mocks
- Core logic functions
- Async workflow
- Config dump completeness
- Recreation from config dump
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from typing import Optional, List, Any
from unittest.mock import patch, MagicMock, AsyncMock

# Mock implementations for testing
from virtual_streamer.video_generation.interfaces import (
    LLMInterface,
    TTSInterface,
    STTInterface,
    VideoRetrieverInterface,
    PromptProviderInterface,
)
from virtual_streamer.video_generation.config import (
    VideoGenerationConfig,
    LLMConfig,
    TTSConfig,
    STTConfig,
    VideoRetrievalConfig,
    PromptConfig,
    ConfigDump,
)
from virtual_streamer.video_generation.core import (
    separation_fn,
    generate_story,
)
from virtual_streamer.video_search.client import VideoSearchResult, TagInfo


# ============================================================================
# Mock Implementations
# ============================================================================


class MockLLM(LLMInterface):
    """Mock LLM for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Mock story", "Mock keyword", "CONTEXTUAL"]
        self.call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response

    async def complete_structured(self, prompt: str, response_model, **kwargs):
        """Mock structured completion."""
        from virtual_streamer.video_generation.config import StoryOutput, DialogLine
        from virtual_streamer.image_generation.models import FluxPrompt, Camera

        if response_model == StoryOutput:
            return StoryOutput(
                title="Fred se lance dans l'IA (Version complète)",
                story_plan="Fred va découvrir l'IA et créer FredGPT avec un ton humoristique et absurde.",
                dialog=[
                    DialogLine(character_id="fred", text="Eh dis donc Jamy!", scene_description=FluxPrompt(scene="Fred talks to camera in a lab setting", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot"))),
                    DialogLine(character_id="fred", text="J'ai découvert l'IA!", scene_description=FluxPrompt(scene="Fred looks excited in the same lab", subjects=[], lighting="Natural", camera=Camera(angle="eye level", distance="medium shot")))
                ],
            )
        # Default fallback
        return response_model()

    async def complete_with_vision(
        self, prompt: str, image_base64: str, **kwargs
    ) -> str:
        return "Rating: CONTEXTUAL\nGrade: 8\nReasoning: Mock judgement"


class MockTTS(TTSInterface):
    """Mock TTS for testing."""

    def __init__(self):
        self.generated_files = []

    def generate_speech(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        # Create empty file
        with open(output_path, "wb") as f:
            f.write(b"MOCK_AUDIO")

        self.generated_files.append(output_path)
        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        return 3.5


class MockSTT(STTInterface):
    """Mock STT for testing."""

    def __init__(self):
        self.transcriptions = []

    def transcribe(self, audio_path: str, **kwargs) -> Any:
        return {"text": "Mock transcription"}

    def transcribe_to_srt(self, audio_path: str, srt_output_path: str, **kwargs) -> str:
        # Create mock SRT file
        with open(srt_output_path, "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:03,500\nMock transcription\n")

        self.transcriptions.append(srt_output_path)
        return srt_output_path


class MockVideoRetriever(VideoRetrieverInterface):
    """Mock video retriever for testing."""

    def __init__(self):
        self.mock_videos = ["/mock/video1.mp4", "/mock/video2.mp4", "/mock/video3.mp4"]

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[VideoSearchResult]:
        """Return mock VideoSearchResult objects."""
        results = []
        for i, path in enumerate(self.mock_videos[:top_k]):
            results.append(
                VideoSearchResult(
                    segment_id=f"seg_{i}",
                    video_id=f"vid_{i}",
                    segment_index=i,
                    duration=5.0,
                    path=path,
                    tags=[],
                    similarity=0.9 - (i * 0.1),
                )
            )
        return results

    def get_video_metadata(self, video_path: str) -> dict:
        return {"duration": 5.0, "who": "fred"}


class MockPromptProvider(PromptProviderInterface):
    """Mock prompt provider for testing."""

    def __init__(self):
        self.prompts = {"story_generation": "Generate a story about: {title}"}

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        template = self.prompts.get(prompt_name, "")
        if kwargs:
            return template.format(**kwargs)
        return template

    def list_prompts(self) -> List[str]:
        return list(self.prompts.keys())

    def get_raw_prompt(self, prompt_name: str) -> str:
        return self.prompts.get(prompt_name, "")


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test that default config initializes correctly."""
        config = VideoGenerationConfig()

        assert config.llm.provider == "anthropic"
        assert config.tts.provider == "fish"
        assert config.stt.provider == "whisper"
        assert config.max_sentence_length == 35
        assert config.enable_config_dump is True

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4
  temperature: 0.5
tts:
  provider: solero
  port: 9000
max_sentence_length: 50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = VideoGenerationConfig.from_yaml(yaml_path)

            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-4"
            assert config.llm.temperature == 0.5
            assert config.tts.provider == "solero"
            assert config.tts.port == 9000
            assert config.max_sentence_length == 50
        finally:
            os.unlink(yaml_path)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = VideoGenerationConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "llm" in config_dict
        assert "tts" in config_dict
        assert "stt" in config_dict

    def test_config_paths(self):
        """Test output and temp path generation."""
        config = VideoGenerationConfig(
            output_dir="./test_output", temp_dir="./test_temp"
        )

        output_path = config.get_output_path("test.mp4")
        temp_path = config.get_temp_path("temp.wav")

        assert "test_output" in output_path
        assert "test.mp4" in output_path
        assert "test_temp" in temp_path
        assert "temp.wav" in temp_path


# ============================================================================
# Text Processing Tests
# ============================================================================


class TestTextProcessing:
    """Test text processing and sentence separation.
    
    Note: separation_fn only splits by punctuation when text exceeds max_length.
    If text is shorter than max_length, it stays as a single segment.
    """

    def test_separation_fn_splits_when_over_max_length(self):
        """Test that text is split when it exceeds max_length."""
        # This text is 44 characters, so with max_length=30 it should split
        text = "Sentence one. Sentence two. Sentence three."
        sentences = separation_fn(text, max_length=30)

        # Should split by first punctuation found (.)
        assert len(sentences) == 3
        assert "Sentence one" in sentences[0]
        assert "Sentence two" in sentences[1]

    def test_separation_fn_no_split_when_under_max_length(self):
        """Test that text stays whole when under max_length."""
        text = "Short text."
        sentences = separation_fn(text, max_length=50)

        assert len(sentences) == 1
        assert sentences[0] == "Short text."

    def test_separation_fn_newlines(self):
        """Test separation respects newlines."""
        text = "Line one\nLine two\nLine three"
        sentences = separation_fn(text, max_length=50)

        assert len(sentences) == 3

    def test_separation_fn_empty(self):
        """Test empty input."""
        text = ""
        sentences = separation_fn(text, max_length=35)

        assert len(sentences) == 0


# ============================================================================
# Story Generation Tests
# ============================================================================


class TestStoryGeneration:
    """Test story generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_story(self):
        """Test story generation with mock LLM returns structured output."""
        mock_llm = MockLLM()
        mock_prompt = MockPromptProvider()
        config = VideoGenerationConfig()

        story_output = await generate_story(
            title="Fred test", llm=mock_llm, prompt_provider=mock_prompt, config=config
        )

        # Check it returns StoryOutput
        from virtual_streamer.video_generation.config import StoryOutput

        assert isinstance(story_output, StoryOutput)
        assert len(story_output.title) > 0
        assert len(story_output.story_plan) > 0
        assert len(story_output.dialog) > 0


# ============================================================================
# Interface Tests
# ============================================================================


class TestInterfaces:
    """Test interface implementations."""

    def test_mock_llm(self):
        """Test mock LLM implementation."""
        llm = MockLLM()

        assert isinstance(llm, LLMInterface)

    def test_mock_tts(self):
        """Test mock TTS implementation."""
        tts = MockTTS()

        audio_path = tts.generate_speech("Test text")
        assert os.path.exists(audio_path)
        assert tts.get_audio_duration(audio_path) == 3.5

        # Cleanup
        os.unlink(audio_path)

    def test_mock_stt(self):
        """Test mock STT implementation."""
        stt = MockSTT()

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            srt_path = f.name

        try:
            result = stt.transcribe_to_srt("/mock/audio.wav", srt_path)
            assert os.path.exists(result)

            with open(result, "r") as f:
                content = f.read()
                assert "Mock transcription" in content
        finally:
            if os.path.exists(srt_path):
                os.unlink(srt_path)

    def test_mock_video_retriever(self):
        """Test mock video retriever implementation."""
        retriever = MockVideoRetriever()

        results = retriever.search("test query", collection="test_collection", top_k=5)
        assert len(results) <= 5
        assert all("/mock/" in r.path for r in results)
        assert all(isinstance(r, VideoSearchResult) for r in results)

        metadata = retriever.get_video_metadata(results[0].path)
        assert "duration" in metadata

    def test_mock_prompt_provider(self):
        """Test mock prompt provider implementation."""
        provider = MockPromptProvider()

        prompts = provider.list_prompts()
        assert len(prompts) > 0

        prompt = provider.get_prompt("story_generation", title="Test")
        assert "Test" in prompt


# ============================================================================
# Config Dump Tests
# ============================================================================


class TestConfigDump:
    """Test config dump creation and loading."""

    # Note: test_config_dump_creation removed because it requires ffprobe
    # to run on real video files, which is not suitable for unit tests

    def test_config_dump_save_load(self):
        """Test saving and loading config dump."""
        config_dump = ConfigDump(
            timestamp="2025-01-01T00:00:00",
            input={"story": "Test"},
            config={},
            execution={},
            output={},
            models={},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dump_path = f.name

        try:
            config_dump.save(dump_path)
            assert os.path.exists(dump_path)

            loaded = ConfigDump.load(dump_path)
            assert loaded.version == config_dump.version
            assert loaded.timestamp == config_dump.timestamp
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self):
        """Test complete workflow with all mocks (no actual video generation)."""
        # This is a simplified test that doesn't actually create videos
        # but validates the workflow logic

        mock_llm = MockLLM(
            responses=[
                "Test story sentence one. Test story sentence two.",
                "Rating: CONTEXTUAL\nGrade: 8",
                "Rating: NEUTRAL\nGrade: 5",
            ]
        )
        mock_tts = MockTTS()
        mock_stt = MockSTT()
        mock_retriever = MockVideoRetriever()

        config = VideoGenerationConfig(
            output_dir=tempfile.mkdtemp(),
            temp_dir=tempfile.mkdtemp(),
            enable_config_dump=False,
        )

        # Test story generation
        story_output = await generate_story(
            title="Test",
            llm=mock_llm,
            prompt_provider=MockPromptProvider(),
            config=config,
        )

        from virtual_streamer.video_generation.config import StoryOutput

        assert isinstance(story_output, StoryOutput)
        assert len(story_output.dialog) > 0

    def test_mock_workflow_synchronous(self):
        """Test synchronous parts of the workflow."""
        # Test sentence separation - use newlines to create multiple segments
        story = "First sentence\nSecond sentence\nThird sentence"
        sentences = separation_fn(story, max_length=50)
        assert len(sentences) == 3

        # Test TTS
        tts = MockTTS()
        audio_path = tts.generate_speech(sentences[0])
        assert os.path.exists(audio_path)
        os.unlink(audio_path)

        # Test STT
        stt = MockSTT()
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            srt_path = f.name
        stt.transcribe_to_srt("/mock/audio.wav", srt_path)
        assert os.path.exists(srt_path)
        os.unlink(srt_path)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
