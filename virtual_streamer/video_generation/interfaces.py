"""
Abstract interfaces for video generation components.

This module defines interfaces for all major components to enable:
- Easy swapping of implementations
- Future API integration
- Testing with mocks
- Extensibility
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import stable_whisper

from virtual_streamer.video_search.client import VideoSearchResult


class LLMInterface(ABC):
    """Abstract interface for Language Model providers."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def complete_structured(
        self, prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """
        Generate structured completion conforming to a Pydantic model.

        Args:
            prompt: The input prompt
            response_model: Pydantic model class for the response structure
            **kwargs: Additional provider-specific parameters

        Returns:
            Instance of response_model with generated data
        """
        pass

    @abstractmethod
    async def complete_with_vision(
        self, prompt: str, image_base64: str, **kwargs
    ) -> str:
        """
        Generate completion with vision input (for video frame judgement).

        Args:
            prompt: The text prompt
            image_base64: Base64-encoded image
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        pass


class TTSInterface(ABC):
    """Abstract interface for Text-to-Speech providers."""

    @abstractmethod
    def generate_speech(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            output_path: Path for output audio file (auto-generated if None)
            **kwargs: Additional provider-specific parameters

        Returns:
            Path to generated audio file
        """
        pass

    @abstractmethod
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        pass


class STTInterface(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Any:
        """
        Transcribe audio to text with timing information.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters

        Returns:
            Transcription object (provider-specific, but should support .to_srt_vtt())
        """
        pass

    @abstractmethod
    def transcribe_to_srt(self, audio_path: str, srt_output_path: str, **kwargs) -> str:
        """
        Transcribe audio and save as SRT file.

        Args:
            audio_path: Path to audio file
            srt_output_path: Path for output SRT file
            **kwargs: Additional parameters

        Returns:
            Path to SRT file
        """
        pass


class VideoRetrieverInterface(ABC):
    """Abstract interface for video retrieval systems."""

    @abstractmethod
    def search(
        self, query: str, top_k: int = 10, tags: Optional[List[str]] = None
    ) -> List[VideoSearchResult]:
        """
        Search for videos matching the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of VideoSearchResult objects with path, similarity, tags, etc.
        """
        pass

    @abstractmethod
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Get metadata for a video.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with metadata (duration, resolution, etc.)
        """
        pass


class PromptProviderInterface(ABC):
    """Abstract interface for prompt management systems."""

    @abstractmethod
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt template by name.

        Args:
            prompt_name: Name/identifier of the prompt
            **kwargs: Variables to interpolate into the prompt

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def list_prompts(self) -> List[str]:
        """
        List all available prompt names.

        Returns:
            List of prompt identifiers
        """
        pass

    @abstractmethod
    def get_raw_prompt(self, prompt_name: str) -> str:
        """
        Get the raw (unformatted) prompt template.

        Args:
            prompt_name: Name/identifier of the prompt

        Returns:
            Raw prompt template string
        """
        pass


class VideoJudgementResult:
    """Result of video-dialogue matching judgement."""

    def __init__(
        self,
        video_path: str,
        rating: str,  # CONTEXTUAL, NEUTRAL, NOT_CONTEXTUAL
        grade: int,
        reasoning: str,
    ):
        self.video_path = video_path
        self.rating = rating
        self.grade = grade
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "rating": self.rating,
            "grade": self.grade,
            "reasoning": self.reasoning,
        }


class VideoMatchResult:
    """Result of finding best video match for a sentence."""

    def __init__(
        self,
        sentence: str,
        selected_video: str,
        rating: str,
        grade: int,
        reasoning: str,
        alternatives_tried: List[Dict[str, Any]] = None,
    ):
        self.sentence = sentence
        self.selected_video = selected_video
        self.rating = rating
        self.grade = grade
        self.reasoning = reasoning
        self.alternatives_tried = alternatives_tried or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentence": self.sentence,
            "selected_video": self.selected_video,
            "rating": self.rating,
            "grade": self.grade,
            "reasoning": self.reasoning,
            "alternatives_tried": self.alternatives_tried,
        }


class ProgressCallback(ABC):
    """Abstract interface for progress reporting."""

    @abstractmethod
    def update(self, message: str, progress: Optional[float] = None) -> None:
        """
        Update progress.

        Args:
            message: Progress message
            progress: Optional progress value (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def set_total_steps(self, total: int) -> None:
        """
        Set total number of steps for progress tracking.

        Args:
            total: Total number of steps
        """
        pass

    @abstractmethod
    def increment_step(self, message: Optional[str] = None) -> None:
        """
        Increment progress by one step.

        Args:
            message: Optional message for this step
        """
        pass


class SimpleProgressCallback(ProgressCallback):
    """Simple console-based progress callback."""

    def __init__(self):
        self.total_steps = 0
        self.current_step = 0

    def update(self, message: str, progress: Optional[float] = None) -> None:
        if progress is not None:
            print(f"[{progress * 100:.1f}%] {message}")
        else:
            print(f"[Progress] {message}")

    def set_total_steps(self, total: int) -> None:
        self.total_steps = total
        self.current_step = 0

    def increment_step(self, message: Optional[str] = None) -> None:
        self.current_step += 1
        if message:
            progress = (
                self.current_step / self.total_steps if self.total_steps > 0 else None
            )
            self.update(message, progress)
