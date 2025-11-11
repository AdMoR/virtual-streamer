"""
Video Generation Module

This module provides a complete async video generation pipeline for creating
videos from stories using AI models (LLM, TTS, STT) with optimized parallel processing.

Main Components:
    - config: Pydantic configuration with CLI/env/YAML support
    - interfaces: Abstract base classes for all components
    - implementations: Concrete implementations (LLM, TTS, STT, etc.)
    - core: Async workflow logic with concurrency control

Features:
    - Async optimization with semaphore-based rate limiting
    - Structured LLM outputs using Pydantic models
    - Event loop safety with lazy initialization
    - Complete config dumps for reproducibility
    - Interface-based design for easy extension

Usage:
    from virtual_streamer.video_generation import (
        VideoGenerationConfig,
        create_llm, create_tts, create_stt,
        create_video_retriever, create_prompt_provider,
        generate_story, generate_video_from_story
    )
"""

from .config import (
    VideoGenerationConfig,
    LLMConfig,
    TTSConfig,
    STTConfig,
    VideoRetrievalConfig,
    PromptConfig,
    VideoProcessingConfig,
    StoryOutput,
    GenerationResult,
    ConfigDump,
)

from .interfaces import (
    LLMInterface,
    TTSInterface,
    STTInterface,
    VideoRetrieverInterface,
    PromptProviderInterface,
    ProgressCallback,
    SimpleProgressCallback,
    VideoJudgementResult,
    VideoMatchResult,
)

from .implementations import (
    create_llm,
    create_tts,
    create_stt,
    create_video_retriever,
    create_prompt_provider,
)

from .core import (
    generate_story,
    generate_video_from_story,
    recreate_from_config_dump,
    find_best_video_for_sentence,
    judge_video_match,
    generate_search_keyword,
)

from .visualizer import (
    create_html_report,
    create_html_report_from_dump,
)

__all__ = [
    # Configuration
    "VideoGenerationConfig",
    "LLMConfig",
    "TTSConfig",
    "STTConfig",
    "VideoRetrievalConfig",
    "PromptConfig",
    "VideoProcessingConfig",
    "StoryOutput",
    "GenerationResult",
    "ConfigDump",
    # Interfaces
    "LLMInterface",
    "TTSInterface",
    "STTInterface",
    "VideoRetrieverInterface",
    "PromptProviderInterface",
    "ProgressCallback",
    "SimpleProgressCallback",
    "VideoJudgementResult",
    "VideoMatchResult",
    # Factory functions
    "create_llm",
    "create_tts",
    "create_stt",
    "create_video_retriever",
    "create_prompt_provider",
    # Core functions
    "generate_story",
    "generate_video_from_story",
    "recreate_from_config_dump",
    "find_best_video_for_sentence",
    "judge_video_match",
    "generate_search_keyword",
    # Visualizer functions
    "create_html_report",
    "create_html_report_from_dump",
]

__version__ = "1.3.0"
__author__ = "Virtual Streamer Team"
__description__ = "Async video generation with AI models"

