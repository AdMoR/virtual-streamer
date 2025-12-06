"""
Video Generation Orchestrator.

Main entry point for the ADK video generation pipeline.
Uses SequentialAgent to chain:
1. StoryGeneratorAgent - generates story from title
2. SentenceProcessorAgent - processes sentences into video segments

The FinalizeVideoCallback concatenates all segments into the final video.
"""

import logging
from typing import Optional

from google.adk.agents import SequentialAgent

from virtual_streamer.agents.story_generator import get_story_generator
from virtual_streamer.agents.sentence_processor import SentenceProcessorAgent
from virtual_streamer.agents.common.callbacks import FinalizeVideoCallback
from virtual_streamer.video_generation.interfaces import (
    VideoRetrieverInterface,
    TTSInterface,
    STTInterface,
)

logger = logging.getLogger(__name__)


def get_video_generation_orchestrator(
    video_retriever: VideoRetrieverInterface,
    tts: TTSInterface,
    stt: STTInterface,
    output_dir: str,
    temp_dir: str,
    max_video_candidates: int = 5,
    max_search_attempts: int = 3,
    fontsize: int = 14,
) -> SequentialAgent:
    """
    Create the main video generation orchestrator.
    
    This creates a SequentialAgent that chains together all the
    agents needed for video generation:
    
    1. StoryGeneratorAgent: Takes title from state, generates story,
       splits into sentences
    2. SentenceProcessorAgent: For each sentence, finds matching video,
       generates audio/subtitles, combines segments
    
    The FinalizeVideoCallback runs after all agents complete and
    concatenates all segments into the final video.
    
    Args:
        video_retriever: Interface for searching video clips
        tts: Interface for text-to-speech
        stt: Interface for speech-to-text
        output_dir: Directory for final output
        temp_dir: Directory for temporary files
        max_video_candidates: Max videos to judge per sentence
        max_search_attempts: Max keyword generation retries
        fontsize: Subtitle font size
    
    Returns:
        Configured SequentialAgent orchestrator
    
    Usage:
        # Initialize interfaces
        video_retriever = create_video_retriever(config.video_retrieval)
        tts = create_tts(config.tts)
        stt = create_stt(config.stt)
        
        # Create orchestrator
        orchestrator = get_video_generation_orchestrator(
            video_retriever=video_retriever,
            tts=tts,
            stt=stt,
            output_dir="./output",
            temp_dir="./temp",
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
    """
    # Create the story generator (singleton)
    story_generator = get_story_generator()
    
    # Create the sentence processor with all dependencies
    sentence_processor = SentenceProcessorAgent(
        video_retriever=video_retriever,
        tts=tts,
        stt=stt,
        output_dir=output_dir,
        temp_dir=temp_dir,
        max_video_candidates=max_video_candidates,
        max_search_attempts=max_search_attempts,
        fontsize=fontsize,
    )
    
    # Create the finalize callback
    finalize_callback = FinalizeVideoCallback(output_dir=output_dir)
    
    # Create the sequential orchestrator
    orchestrator = SequentialAgent(
        name="video_generator",
        sub_agents=[
            story_generator,
            sentence_processor,
        ],
        after_agent_callback=[finalize_callback],
    )
    
    logger.info("Created video generation orchestrator")
    
    return orchestrator


# Expose as root_agent for ADK compatibility (if running standalone)
def create_root_agent(
    video_retriever: VideoRetrieverInterface,
    tts: TTSInterface,
    stt: STTInterface,
    output_dir: str = "./output",
    temp_dir: str = "./temp",
) -> SequentialAgent:
    """
    Create the root agent for ADK server deployment.
    
    This is a convenience wrapper around get_video_generation_orchestrator
    with default settings suitable for production use.
    
    Args:
        video_retriever: Interface for searching video clips
        tts: Interface for text-to-speech
        stt: Interface for speech-to-text
        output_dir: Directory for final output
        temp_dir: Directory for temporary files
    
    Returns:
        Configured SequentialAgent as root_agent
    """
    return get_video_generation_orchestrator(
        video_retriever=video_retriever,
        tts=tts,
        stt=stt,
        output_dir=output_dir,
        temp_dir=temp_dir,
    )

