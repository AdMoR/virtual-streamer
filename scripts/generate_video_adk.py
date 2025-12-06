#!/usr/bin/env python3
"""
Virtual Streamer Video Generation Script (ADK Version)

This script generates videos from stories using Google ADK agents.

WORKFLOW:
=========
1. StoryGeneratorAgent: Generates a story from a title using LLM
2. SentenceProcessorAgent: For each sentence:
   - Parallel video matching using VideoMatcherAgents
   - Optional keyword generation with KeywordGeneratorAgent
   - TTS audio generation
   - STT subtitle generation
   - Video segment combining
3. FinalizeVideoCallback: Concatenates all segments into final video

USAGE:
======
# Generate video from a title
python scripts/generate_video_adk.py --title "Fred se lance dans l'IA"

# Use custom output directory
python scripts/generate_video_adk.py --title "Fred" --output-dir ./my_videos

ARCHITECTURE:
=============
This uses Google ADK (Agent Development Kit) for:
- Structured agent orchestration with SequentialAgent
- Parallel video matching with ParallelAgent
- State-based data flow between agents
- Event-driven progress tracking
"""

import asyncio
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_tts,
    create_stt,
    create_video_retriever,
)
from virtual_streamer.agents import (
    get_video_generation_orchestrator,
    TITLE,
    CONFIG,
    FINAL_VIDEO_PATH,
    VIDEO_MATCHES,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_video_generation(
    title: str,
    config: VideoGenerationConfig,
) -> str:
    """
    Run the video generation pipeline using ADK agents.
    
    Args:
        title: The title/topic for story generation
        config: Video generation configuration
    
    Returns:
        Path to the generated video
    """
    # Initialize interfaces
    logger.info("Initializing video retriever, TTS, and STT...")
    video_retriever = create_video_retriever(config.video_retrieval)
    tts = create_tts(config.tts, character_name=config.character_name)
    stt = create_stt(config.stt)
    
    # Create the orchestrator
    logger.info("Creating ADK orchestrator...")
    orchestrator = get_video_generation_orchestrator(
        video_retriever=video_retriever,
        tts=tts,
        stt=stt,
        output_dir=config.output_dir,
        temp_dir=config.temp_dir,
        max_video_candidates=config.max_video_judgement_attempts,
        max_search_attempts=config.max_search_attempts,
        fontsize=config.video_processing.fontsize,
    )
    
    # Create a simple runner context
    # Note: For full ADK deployment, use google.adk.runners.Runner
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions import Session
    
    # Create session with initial state
    session = Session(
        id="video_gen_session",
        app_name="virtual_streamer",
        user_id="user",
        state={
            TITLE: title,
            CONFIG: config.to_dict(),
        },
    )
    
    # Create invocation context
    ctx = InvocationContext(
        invocation_id="video_gen_invocation",
        session=session,
        agent=orchestrator,
    )
    
    # Run the orchestrator and collect events
    logger.info(f"Starting video generation for title: {title}")
    
    async for event in orchestrator.run_async(ctx):
        # Log progress events
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    logger.info(f"[{event.author}] {part.text}")
    
    # Get the final video path from state
    final_video_path = session.state.get(FINAL_VIDEO_PATH, "")
    
    if not final_video_path:
        raise RuntimeError("Video generation completed but no output path found")
    
    return final_video_path


async def main():
    """Main entry point."""
    # Load configuration from CLI args, env vars, and .env files
    config = VideoGenerationConfig()
    
    # Validate inputs
    if not config.title:
        print("Error: --title is required", file=sys.stderr)
        print("\nUsage: python scripts/generate_video_adk.py --title 'Your Title'")
        return 1
    
    # Print configuration summary
    if not config.quiet:
        print("=" * 70)
        print("Virtual Streamer Video Generation (ADK)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Title: {config.title}")
        print(f"  TTS: {config.tts.provider} @ {config.tts.host}:{config.tts.port}")
        print(f"  STT: {config.stt.provider}/{config.stt.model}")
        print(f"  Video Retrieval: {config.video_retrieval.method}")
        print(f"  Output: {config.output_dir}")
        print("=" * 70)
        print()
    
    try:
        # Run the video generation
        final_video_path = await run_video_generation(
            title=config.title,
            config=config,
        )
        
        # Print results
        if not config.quiet:
            print("\n" + "=" * 70)
            print("Video Generation Complete!")
            print("=" * 70)
            print(f"\n✓ Video saved to: {final_video_path}")
            print("\n" + "=" * 70)
        else:
            print(final_video_path)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

