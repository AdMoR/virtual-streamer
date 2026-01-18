"""
State key constants for ADK agents.

These constants define the keys used in CallbackContext.state / session.state
to pass data between agents in the video generation pipeline.

State Flow:
    Input → StoryGenerator → SentenceProcessor → FinalizeCallback → Output

Usage:
    from virtual_streamer.agents.common.state_keys import TITLE, SENTENCES
    
    # Reading from state
    title = ctx.session.state.get(TITLE)
    
    # Writing to state via EventActions
    yield Event(
        author=self.name,
        actions=EventActions(state_delta={SENTENCES: ["sentence1", ...]})
    )
"""

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT STATE (set before orchestrator runs)
# ═══════════════════════════════════════════════════════════════════════════════

TITLE = "title"
"""str: Input title for story generation. Example: "Fred se lance dans l'IA" """

STORY_TEMPLATE_ID = "story_template_id"
"""str: Optional story template ID to use for generation. If not set, uses default prompt."""

VIDEO_COLLECTION = "video_collection"
"""str: Qdrant collection name for video search. Loaded from StoryTemplate."""

CONFIG = "config"
"""dict: Serialized VideoGenerationConfig with all settings."""

NEWS_ARTICLE_ID = "news_article_id"
"""str: ID of the news article used for story generation (optional)."""

NEWS_CONTEXT = "news_context"
"""str: Formatted news context string for prompt enrichment (optional).

When set, this provides additional context from a news article to inform
story generation. Format:
    Titre: {headline}
    Résumé: {summary}
    Source: {source}
    Date: {published_date}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STORY GENERATION STATE (after StoryGeneratorAgent)
# ═══════════════════════════════════════════════════════════════════════════════

STORY_OUTPUT = "story_output"
"""dict: Generated story with keys: {title, story_plan, dialog}"""

SENTENCES = "sentences"
"""List[str]: Dialog split into individual sentences for processing."""

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO MATCHING STATE (during SentenceProcessorAgent iteration)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-run namespaced keys (format: "prefix:{run_id}:suffix")
# These are used internally by SentenceProcessorAgent

# Task setup keys (written before parallel matching)
# - task:{run_id}:sentence        -> str: Current sentence being processed
# - task:{run_id}:{worker}:video  -> str: Video path for this worker
# - task:{run_id}:{worker}:frame  -> str: Base64 frame for vision LLM

# Result keys (written by VideoMatcherAgent after_model callback)
# - result:{run_id}:{worker}      -> dict: {video_path, rating, grade, reasoning}

# Keyword generation keys
# - task:{run_id}:prev_keywords   -> List[str]: Previously tried keywords
# - keyword:{run_id}              -> str: New generated keyword

# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATED RESULTS STATE (after SentenceProcessorAgent completes)
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_MATCHES = "video_matches"
"""List[dict]: All video matches, one per sentence.
Each dict has: {sentence, video_path, rating, grade, reasoning}"""

AUDIO_FILES = "audio_files"
"""List[str]: Generated audio file paths, one per sentence."""

SUBTITLE_FILES = "subtitle_files"
"""List[str]: Generated SRT subtitle file paths, one per sentence."""

VIDEO_SEGMENTS = "video_segments"
"""List[str]: Combined video segment paths (video+audio+subtitles), one per sentence."""

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT STATE (after FinalizeVideoCallback)
# ═══════════════════════════════════════════════════════════════════════════════

FINAL_VIDEO_PATH = "final_video_path"
"""str: Path to the final concatenated video."""

CONFIG_DUMP_PATH = "config_dump_path"
"""str: Path to the saved config dump JSON for reproducibility."""

TIMING = "timing"
"""dict: Timing metrics for each phase of the pipeline."""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions for namespaced keys
# ═══════════════════════════════════════════════════════════════════════════════


def task_key(run_id: str, suffix: str) -> str:
    """Generate a task key for the given run_id and suffix.
    
    Example:
        task_key("s0", "sentence") -> "task:s0:sentence"
        task_key("s0", "m0:video") -> "task:s0:m0:video"
    """
    return f"task:{run_id}:{suffix}"


def result_key(run_id: str, worker_name: str) -> str:
    """Generate a result key for the given run_id and worker.
    
    Example:
        result_key("s0", "m0") -> "result:s0:m0"
    """
    return f"result:{run_id}:{worker_name}"


def keyword_key(run_id: str) -> str:
    """Generate a keyword key for the given run_id.
    
    Example:
        keyword_key("s0_kw1") -> "keyword:s0_kw1"
    """
    return f"keyword:{run_id}"

