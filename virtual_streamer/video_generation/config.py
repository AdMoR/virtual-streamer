"""
Configuration module for video generation using Pydantic Settings.

This module provides comprehensive configuration management for the video generation
workflow, including LLM, TTS, STT, video retrieval, and prompt settings.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
import sys


class LLMConfig(BaseModel):
    """Configuration for Language Model providers."""

    provider: str = Field(
        default="anthropic", description="LLM provider: anthropic, openai, or litellm"
    )
    model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Model identifier"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum tokens to generate"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key (defaults to env var)"
    )
    vision_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model for vision tasks (video judgement)",
    )


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech providers."""

    provider: str = Field(
        default="fish", description="TTS provider: fish, solero, or coqui"
    )
    host: str = Field(default="127.0.0.1", description="TTS service host")
    port: int = Field(default=8003, gt=0, le=65535, description="TTS service port")
    reference_audio: Optional[str] = Field(
        default=None, description="Path to reference audio for voice cloning"
    )
    reference_text: Optional[str] = Field(
        default=None, description="Reference text matching the reference audio"
    )
    # Fish-specific parameters
    max_new_tokens: int = Field(default=1024, gt=0)
    chunk_length: int = Field(default=300, gt=0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.0)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class STTConfig(BaseModel):
    """Configuration for Speech-to-Text providers."""

    provider: str = Field(
        default="whisper", description="STT provider: whisper or faster-whisper"
    )
    model: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large",
    )


class VideoRetrievalConfig(BaseModel):
    """Configuration for video retrieval system."""

    method: str = Field(
        default="bm25", description="Retrieval method: bm25, vector, or hybrid"
    )
    index_path: str = Field(
        default="/media/amor/data1/Downloads/CPS/clip_infos",
        description="Path to video index/clips info",
    )
    vector_store_path: Optional[str] = Field(
        default=None, description="Path to vector store (for vector/hybrid methods)"
    )
    embedding_model: str = Field(
        default="lightonai/modernbert-embed-large",
        description="Embedding model for vector search",
    )
    top_k: int = Field(default=10, gt=0, description="Number of videos to retrieve")
    character_filter: str = Field(
        default="fred", description="Filter by character name"
    )


class PromptConfig(BaseModel):
    """Configuration for prompt management."""

    provider: str = Field(
        default="local", description="Prompt provider: local or mlflow"
    )
    local_file: Optional[str] = Field(
        default=None, description="Path to local prompt file"
    )
    # MLflow specific
    mlflow_tracking_uri: Optional[str] = Field(
        default=None, description="MLflow tracking URI"
    )
    mlflow_experiment: Optional[str] = Field(
        default=None, description="MLflow experiment name"
    )


class VideoProcessingConfig(BaseModel):
    """Configuration for video processing parameters."""

    resolution: str = Field(default="720x480", description="Output video resolution")
    codec: str = Field(
        default="h264_nvenc",
        description="Video codec (h264_nvenc for GPU, libx264 for CPU)",
    )
    bitrate: str = Field(default="3000k", description="Video bitrate")
    fontsize: int = Field(default=14, gt=0, description="Subtitle font size")


class VideoGenerationConfig(BaseSettings):
    """Main configuration for video generation workflow.

    Configuration is loaded from (in order of precedence):
    1. Command-line arguments
    2. Environment variables (VG_ prefix)
    3. .env file (secrets)
    4. .env.public file (non-secrets)
    5. Default values

    Example CLI:
        python generate_video.py --title "Fred" --llm-provider anthropic

    Example env vars:
        VG_LLM__PROVIDER=openai
        VG_OUTPUT_DIR=/custom/output
    """

    # Character configuration (for voice cloning)
    character_name: Optional[str] = Field(
        default=None,
        description="Character name to load voice samples from entity service for voice cloning",
    )

    model_config = SettingsConfigDict(
        env_prefix="VG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=(".env", ".env.public"),
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,  # Enable automatic CLI parsing
        cli_prog_name="generate_video.py",
    )

    # ========================================================================
    # Input options (mutually exclusive in practice)
    # ========================================================================
    title: Optional[str] = Field(
        default=None,
        description="Title/topic for story generation (e.g., 'Fred se lance dans l'IA')",
    )
    story_file: Optional[str] = Field(
        default=None, description="Path to existing story file to convert to video"
    )
    from_config_dump: Optional[str] = Field(
        default=None, description="Recreate video from config dump (skips LLM calls)"
    )

    # ========================================================================
    # Configuration files
    # ========================================================================
    config: Optional[str] = Field(
        default=None, description="Path to YAML config file (overrides defaults)"
    )
    prompt_file: Optional[str] = Field(
        default=None, description="Path to custom prompt file or directory"
    )

    # ========================================================================
    # Display options
    # ========================================================================
    verbose: bool = Field(default=False, description="Enable verbose output")
    quiet: bool = Field(default=False, description="Suppress progress messages")

    # Component configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    video_retrieval: VideoRetrievalConfig = Field(default_factory=VideoRetrievalConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    video_processing: VideoProcessingConfig = Field(
        default_factory=VideoProcessingConfig
    )

    # Video search and matching
    max_search_attempts: int = Field(
        default=3, gt=0, description="Max attempts to find alternative search keywords"
    )
    max_video_judgement_attempts: int = Field(
        default=5, gt=0, description="Max videos to judge for each sentence"
    )

    # Text processing
    max_sentence_length: int = Field(
        default=35, gt=0, description="Maximum length for sentence splitting"
    )

    # Output paths
    output_dir: str = Field(
        default="./output", description="Output directory for final videos"
    )
    temp_dir: str = Field(
        default="./temp", description="Temporary directory for intermediate files"
    )

    # Parallelization
    max_parallel_llm_calls: int = Field(
        default=5, gt=0, description="Maximum parallel LLM API calls"
    )

    # Config dump
    enable_config_dump: bool = Field(
        default=True, description="Enable comprehensive config dumping"
    )
    config_dump_filename: str = Field(
        default="generation_config.json", description="Filename for config dump"
    )

    def validate_inputs(self):
        """Validate that exactly one input method is specified."""
        inputs = [self.title, self.story_file, self.from_config_dump]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Exactly one of --title, --story-file, or --from-config-dump must be provided"
            )

    def get_output_path(self, filename: str) -> str:
        """Get full path for output file."""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)

    def get_temp_path(self, filename: str) -> str:
        """Get full path for temporary file."""
        os.makedirs(self.temp_dir, exist_ok=True)
        return os.path.join(self.temp_dir, filename)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VideoGenerationConfig":
        """Load configuration from YAML file.

        Note: YAML values override .env values but are overridden by environment variables.
        """
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_cli_args(cls, args: Optional[List[str]] = None) -> "VideoGenerationConfig":
        """Create config from minimal CLI arguments.

        Supports only essential arguments:
        - --config: Path to YAML config file
        - --env-file: Path to additional .env file

        All other configuration via environment variables or config files.
        """
        if args is None:
            args = sys.argv[1:]

        # Simple parsing for config file path
        config_path = None
        env_file = None

        i = 0
        while i < len(args):
            if args[i] in ("--config", "-c") and i + 1 < len(args):
                config_path = args[i + 1]
                i += 2
            elif args[i] in ("--env-file", "-e") and i + 1 < len(args):
                env_file = args[i + 1]
                i += 2
            else:
                i += 1

        # Load from YAML if provided
        if config_path:
            return cls.from_yaml(config_path)

        # Otherwise load from environment and .env files
        if env_file:
            # Temporarily update env files list
            config = cls.model_config.copy()
            config["env_file"] = [env_file, ".env", ".env.public"]
            return cls()

        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump(mode="json")

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class StoryOutput(BaseModel):
    """Structured output from story generation."""

    title: str = Field(description="Refined/expanded title for the story")
    story_plan: str = Field(
        description="Overall plan and reasoning used to create the dialog"
    )
    dialog: str = Field(
        description="The actual dialog lines produced by Fred and other characters"
    )

    def get_full_text(self) -> str:
        """Get the complete story text for backwards compatibility."""
        return self.dialog


class GenerationResult(BaseModel):
    """Model for video generation results."""

    video_path: str
    config_dump_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    story_output: Optional[StoryOutput] = None  # Include story details if generated


class ConfigDump(BaseModel):
    """Comprehensive config dump for reproducibility."""

    version: str = "1.0"
    timestamp: str
    input: Dict[str, Any]
    config: Dict[str, Any]
    execution: Dict[str, Any]
    output: Dict[str, Any]
    models: Dict[str, Any]
    timing: Dict[str, float] = Field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save config dump to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConfigDump":
        """Load config dump from JSON file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
