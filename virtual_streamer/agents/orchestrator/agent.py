"""
Video Generation Orchestrator.

Main entry point for the ADK video generation pipeline.
Uses SequentialAgent to chain:
1. StoryGeneratorAgent - generates story from title
2. SentenceVideoMatcher - matches each dialog line to a video

Output: video_matches in state containing List[DialogLineMatch]
"""

import logging
import os
from functools import lru_cache

from google.adk.agents import SequentialAgent

from virtual_streamer.agents.sentence_video_matcher.agent import (
    create_sentence_video_matcher,
)
from virtual_streamer.agents.story_generator import get_story_generator
from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_video_retriever,
)
from virtual_streamer.video_generation.interfaces import VideoRetrieverInterface

logger = logging.getLogger(__name__)

# Default configuration values (from compose.yaml and VideoGenerationConfig)
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/media/amor/data1/Downloads/CPS/clip_infos")


def get_video_generation_orchestrator(
        video_retriever: VideoRetrieverInterface,
        max_video_candidates: int ,
) -> SequentialAgent:
    """
    Create the main video generation orchestrator.
    
    This creates a SequentialAgent that chains together the agents
    needed for video generation:
    
    1. StoryGeneratorAgent: Takes title from state, generates story
       with DialogLines
    2. SentenceVideoMatcher: For each dialog line, finds the best
       matching video using vision LLM
    
    Args:
        video_retriever: Interface for searching video clips
        max_video_candidates: Max videos to judge per dialog line
    
    Returns:
        Configured SequentialAgent orchestrator
    
    Output State:
        - video_matches: SentenceVideoMatcherOutput with List[DialogLineMatch]
          Each DialogLineMatch contains:
          - dialog_line: DialogLine(character, dialog)
          - video_path: str
          - rating: ContextualRating
          - grade: int
          - reasoning: str
    
    Usage:
        # Initialize video retriever
        video_retriever = create_video_retriever(config.video_retrieval)
        
        # Create orchestrator
        orchestrator = get_video_generation_orchestrator(
            video_retriever=video_retriever,
        )
        
        # Run with initial state
        from google.adk.runners import Runner
        from virtual_streamer.agents.common.state_keys import TITLE
        
        runner = Runner(agent=orchestrator, app_name="video_gen")
        session = runner.session_service.create_session(
            app_name="video_gen",
            user_id="user1",
        )
        session.state[TITLE] = "Fred se lance dans l'IA"
        
        async for event in runner.run_async(session_id=session.id, ...):
            print(event)
        
        # Access results
        matches = session.state["video_matches"]
    """
    # Create the story generator
    story_generator = get_story_generator()

    # Create the sentence video matcher
    sentence_video_matcher = create_sentence_video_matcher(
        video_retriever=video_retriever,
        max_candidates=max_video_candidates,
    )

    # Create the sequential orchestrator
    orchestrator = SequentialAgent(
        name="video_generator",
        sub_agents=[
            story_generator,
            sentence_video_matcher,
        ],
    )

    logger.info("Created video generation orchestrator")

    return orchestrator


@lru_cache
def create_root_agent() -> SequentialAgent:
    """
    Create the root agent for ADK server deployment.
    
    This is the parameterless entry point required by ADK. It instantiates
    all required components using default configuration from environment
    variables and VideoGenerationConfig defaults.
    
    Configuration (via environment variables):
        - DATA_DIR: Video index path (default: /media/amor/data1/Downloads/CPS/clip_infos)
    
    Returns:
        Configured SequentialAgent as root_agent
    
    Example:
        # In ADK deployment
        from virtual_streamer.agents.orchestrator import create_root_agent
        root_agent = create_root_agent()
        
        # The agent expects TITLE in state before running
    """
    # Load configuration from environment and defaults
    config = VideoGenerationConfig()

    # Override from environment if available
    config.video_retrieval.index_path = DEFAULT_DATA_DIR

    logger.info(f"Initializing ADK root agent with config:")
    logger.info(f"  Video Retrieval: {config.video_retrieval.method}")
    logger.info(f"  Index Path: {config.video_retrieval.index_path}")

    # Create video retriever
    video_retriever = create_video_retriever(config.video_retrieval)

    return get_video_generation_orchestrator(
        video_retriever=video_retriever,
        max_video_candidates=5,
    )


# Expose root_agent at module level for ADK discovery
# ADK auto-discovers agents that have `root_agent` defined at module level
root_agent = create_root_agent()
