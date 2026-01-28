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
    """Configuration for video retrieval system using remote VideoSearchClient."""

    server_url: Optional[str] = Field(
        default=None,
        description="Video search server URL (defaults to VIDEO_SEARCH_SERVER_URL env var or localhost:8003)",
    )
    top_k: int = Field(default=1, gt=0, description="Number of videos to retrieve")
    character_filter: Optional[str] = Field(
        default=None, description="Filter by character name/tag"
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


class NewsConfig(BaseModel):
    """Configuration for news feed integration."""

    enabled: bool = Field(
        default=False, 
        description="Enable news feed integration for story generation"
    )
    sources: list[str] = Field(
        default_factory=lambda: [
            "https://www.lemonde.fr/rss/une.xml",
            "https://www.francetvinfo.fr/titres.rss",
            "https://news.google.com/rss?hl=fr&gl=FR&ceid=FR:fr",
        ],
        description="List of RSS feed URLs to fetch news from",
    )
    db_path: str = Field(
        default="./data/news_articles.db",
        description="Path to SQLite database for article metadata",
    )
    storage_prefix: str = Field(
        default="articles",
        description="Object storage prefix for article content",
    )
    fetch_interval_minutes: int = Field(
        default=30,
        gt=0,
        description="How often to fetch new articles (in minutes)",
    )
    max_article_age_hours: int = Field(
        default=24,
        gt=0,
        description="Maximum age of articles to consider for story generation",
    )


class LTXConfig(BaseModel):
    """Configuration for LTX-2 video generation (reusable for T2V pipeline and fallback)."""

    server_url: str = Field(
        default="http://gx10-cbc5:8081",
        description="LTX Video API server URL",
    )
    timeout: float = Field(
        default=600.0,
        description="Request timeout in seconds",
    )
    width: int = Field(default=1280, description="Video width")
    height: int = Field(default=720, description="Video height")
    duration_seconds: float = Field(default=5.0, description="Video duration in seconds")
    fps: int = Field(default=24, description="Frames per second")
    steps: int = Field(default=8, description="Number of inference steps")
    cfg_scale: float = Field(default=4.0, description="CFG scale for guidance")
    style_suffix: str = Field(
        default="Cinematic quality, smooth motion, natural lighting.",
        description="Style suffix appended to prompts",
    )


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
        cli_parse_args=False,  # Enable automatic CLI parsing
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

    # Story template configuration
    story_template_id: Optional[str] = Field(
        default=None,
        description="ID of story template to use for generation. "
        "If not set, uses default C'est pas Sorcier template.",
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
    news: NewsConfig = Field(default_factory=NewsConfig)
    ltx: LTXConfig = Field(default_factory=LTXConfig)

    # LTX fallback behavior
    enable_ltx_fallback: bool = Field(
        default=False,
        description="Enable LTX-2 video generation for non-CONTEXTUAL matches",
    )

    # Video search and matching
    max_search_attempts: int = Field(
        default=1, gt=0, description="Max attempts to find alternative search keywords"
    )
    max_video_candidates: int = Field(
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
        default=1, gt=0, description="Maximum parallel LLM API calls"
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
        inputs = [self.title, self.story_file]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Exactly one of --title or --story-file must be provided"
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


from pydantic import BaseModel, Field


class DialogLine(BaseModel):
    character_id: str = Field(..., description="ID of the character speaking", example="fred")
    text: str = Field(..., description="The dialogue text", example="Eh dis donc Jamy, ça te dit de faire du surf?")
    scene_description: str = Field(..., description="The description of the scene", 
    example="The scene is set in a ski resort in the Alps. A person is talking to the camera.")


class DialogLines(BaseModel):
    lines: list[DialogLine]


class StoryOutput(BaseModel):
    """
    Structured output from story generation.

    The LLM generates a story with three components:
    - title: A refined, catchy title
    - story_plan: The creative reasoning and comedic arc
    - dialog: Dialog lines with character_id, dialog text, and scene_description
    """

    title: str = Field(
        description="Refined/expanded title for the story. Should be catchy and descriptive."
    )
    story_plan: str = Field(
        description="Overall plan and reasoning used to create the dialog. "
                    "Explains the creative choices and comedic arc."
    )
    dialog: list[DialogLine] = Field(
        description="The dialog lines. Each line has character_id (from the template), "
                    "text (spoken text), and scene_description (visual description for video search)."
    )

    def get_character_names(self):
        return [x.character_id for x in self.dialog]


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


class GenerationBlueprint(BaseModel):
    """Blueprint capturing all generation parameters before video creation.
    
    Used for debugging purposes - captures the full state of story output,
    video matches, and planned TTS before actual video generation begins.
    Uploaded to MinIO at: debug/{api_endpoint}/{story_template_id}/{job_id}/blueprint.json
    """

    timestamp: str = Field(description="ISO timestamp when blueprint was created")
    job_id: str = Field(description="Unique job identifier")
    api_endpoint: str = Field(
        description="API endpoint used, e.g., 'video-generation' or 'generate-ltx'"
    )
    story_template_id: str = Field(description="Story template ID used for generation")
    story_output: StoryOutput = Field(description="Generated story with dialog lines")
    video_matches: List[Dict[str, Any]] = Field(
        description="Serialized DialogLineMatch list with video paths and ratings"
    )
    planned_tts: List[Dict[str, Any]] = Field(
        description="Planned TTS entries - text per character for audio generation"
    )
    collection: Optional[str] = Field(
        default=None, description="Video collection used for matching"
    )

    def get_storage_path(self) -> str:
        """Get the MinIO storage path for this blueprint."""
        return f"debug/{self.api_endpoint}/{self.story_template_id}/{self.job_id}/blueprint.json"
