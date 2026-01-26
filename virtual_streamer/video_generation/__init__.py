"""
Video Generation Module

This module provides components for video generation using AI models.

Main Components:
    - config: Pydantic configuration with CLI/env/YAML support
    - interfaces: Abstract base classes for all components
    - implementations: Concrete implementations (LLM, TTS, STT, etc.)
    - core: Utility functions for video processing

Note: The main video generation flow is now handled by ADK agents.
See virtual_streamer.api.high_level.video_generation for the supported API.

Usage:
    from virtual_streamer.video_generation import (
        VideoGenerationConfig,
        create_llm, create_tts, create_stt,
        create_video_retriever, create_prompt_provider,
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
    GenerationBlueprint,
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
    judge_video_match,
    generate_search_keyword,
)

from .visualizer import (
    create_html_report,
    create_html_report_from_dump,
)

from .ltx_client import (
    # LTX Video API client
    LTXVideoClient,
    LTXVideoConfig,
    LTXJobRequest,
    LTXJobResponse,
    LTXJobStatusResponse,
    JobStatus,
    VideoGenerationParams,
    VideoGenerationResult,
    generate_video,
    # Backward compatibility aliases
    ComfyUIClient,
    ComfyUIConfig,
)

from .ltx_prompt_builder import (
    build_ltx_prompt,
    build_ltx_prompt_detailed,
    build_negative_prompt,
    build_prompts_from_story,
)

from .story_to_video import (
    story_to_video,
    title_to_video,
    concatenate_videos,
    generate_segment,
    SegmentResult,
    StoryVideoResult,
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
    "GenerationBlueprint",
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
    "judge_video_match",
    "generate_search_keyword",
    # Visualizer functions
    "create_html_report",
    "create_html_report_from_dump",
    # LTX Video API client
    "LTXVideoClient",
    "LTXVideoConfig",
    "LTXJobRequest",
    "LTXJobResponse",
    "LTXJobStatusResponse",
    "JobStatus",
    "VideoGenerationParams",
    "VideoGenerationResult",
    "generate_video",
    # Backward compatibility aliases
    "ComfyUIClient",
    "ComfyUIConfig",
    # LTX prompt builder
    "build_ltx_prompt",
    "build_ltx_prompt_detailed",
    "build_negative_prompt",
    "build_prompts_from_story",
    # Story to video pipeline
    "story_to_video",
    "title_to_video",
    "concatenate_videos",
    "generate_segment",
    "SegmentResult",
    "StoryVideoResult",
]

__version__ = "1.3.0"
__author__ = "Virtual Streamer Team"
__description__ = "Async video generation with AI models"
