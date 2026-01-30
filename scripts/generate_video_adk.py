#!/usr/bin/env python3
"""
Virtual Streamer Video Generation Script (ADK Version)

This script generates videos from stories using Google ADK agents for dialog/video
matching, then calls the webservice API for TTS and Wav2Lip processing.

WORKFLOW:
=========
1. ADK Orchestrator:
   - StoryGeneratorAgent: Generates story with DialogLines from title
   - SentenceVideoMatcher: Matches each dialog line to a video
   
2. Webservice API calls (for each dialog line):
   - POST /api/v1/tts/generate: Generate audio from dialog text
   - POST /api/v1/wav2lip/generate: Generate lip-synced video
   - POST /api/v1/stt/transcribe-to-srt: Generate subtitles

3. Video composition:
   - Combine audio with video
   - Add subtitles
   - Concatenate all segments into final video

USAGE:
======
# Generate video from a title (requires API server running)
python scripts/generate_video_adk.py --title "Fred se lance dans l'IA"

# Use custom API URL
python scripts/generate_video_adk.py --title "Fred" --api-url http://localhost:8000

# Use custom character
python scripts/generate_video_adk.py --title "Fred" --character-id fred

REQUIREMENTS:
=============
- The unified API server must be running (python -m virtual_streamer.api.main)
- Video retrieval index must be configured
"""

import asyncio
import sys
import os
import logging
import argparse
from datetime import datetime
from typing import List, Optional

import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_streamer.video_generation import (
    VideoGenerationConfig,
    create_video_retriever,
)
from virtual_streamer.agents.orchestrator import get_video_generation_orchestrator
from virtual_streamer.agents.common.state_keys import TITLE, VIDEO_MATCHES
from virtual_streamer.agents.sentence_video_matcher.schema import (
    SentenceVideoMatcherOutput,
    DialogLineMatch,
)
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# API Client (imported from shared module)
# ============================================================================

from virtual_streamer.api.clients.webservice_client import WebserviceClient, APIConfig


# ============================================================================
# Video Composition
# ============================================================================


async def compose_video_from_matches(
    matches: List[DialogLineMatch],
    client: WebserviceClient,
    output_dir: str,
    temp_dir: str,
    fontsize: int = 14,
) -> str:
    """
    Compose final video from dialog line matches using webservice APIs.
    
    For each match:
    1. Generate audio via TTS API
    2. Generate lip-synced video via Wav2Lip API
    3. Combine audio with video
    4. Generate and add subtitles
    
    Then concatenate all segments into final video.
    
    Args:
        matches: List of DialogLineMatch from orchestrator
        client: Webservice API client
        output_dir: Directory for final output
        temp_dir: Directory for intermediate files
        fontsize: Subtitle font size
    
    Returns:
        Path to final concatenated video
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    video_segments = []
    
    for i, match in enumerate(matches):
        logger.info(f"Processing segment {i+1}/{len(matches)}: {match.dialog_line.text[:50]}...")
        
        dialog = match.dialog_line.text
        video_path = match.video_path
        
        try:
            # Step 1: Generate audio via TTS API
            logger.info(f"  [1/4] Generating TTS audio...")
            audio_path = await client.generate_tts(
                text=dialog,
                entry_id=f"segment_{i}",
            )
            logger.info(f"  Audio generated: {audio_path}")
            
            # Step 2: Generate lip-synced video via Wav2Lip API
            logger.info(f"  [2/4] Generating Wav2Lip video...")
            wav2lip_output_dir = os.path.join(temp_dir, f"wav2lip_{i}")
            lip_synced_video = await client.generate_wav2lip(
                audio_path=audio_path,
                video_path=video_path,
                output_dir=wav2lip_output_dir,
            )
            logger.info(f"  Lip-synced video: {lip_synced_video}")
            
            # Step 3: Combine video and audio
            logger.info(f"  [3/4] Combining video and audio...")
            combined_path = os.path.join(temp_dir, f"combined_{i}.mp4")
            combine_video_and_short_audio(lip_synced_video, audio_path, combined_path)
            
            # Step 4: Generate subtitles and add to video
            logger.info(f"  [4/4] Adding subtitles...")
            srt_path = await client.transcribe_to_srt(audio_path)
            segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
            add_subtitle_from_srt(combined_path, srt_path, segment_path, fontsize=fontsize)
            
            video_segments.append(segment_path)
            logger.info(f"  Segment {i+1} complete: {segment_path}")
            
        except Exception as e:
            logger.error(f"  Error processing segment {i+1}: {e}")
            raise
    
    # Concatenate all segments
    logger.info(f"Concatenating {len(video_segments)} segments...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    
    combine_part_in_concat_file(video_segments, concat_file, final_video_path)
    
    logger.info(f"Final video created: {final_video_path}")
    return final_video_path


# ============================================================================
# Main Pipeline
# ============================================================================


async def run_video_generation(
    title: str,
    config: VideoGenerationConfig,
    api_config: APIConfig,
) -> str:
    """
    Run the video generation pipeline.
    
    1. Use ADK orchestrator to generate story and match videos
    2. Use webservice API for TTS and Wav2Lip
    3. Compose final video from segments
    
    Args:
        title: The title/topic for story generation
        config: Video generation configuration
        api_config: API client configuration
    
    Returns:
        Path to the generated video
    """
    # Initialize video retriever
    logger.info("Initializing video retriever...")
    video_retriever = create_video_retriever(config.video_retrieval)
    
    # Create the orchestrator (only needs video_retriever and max_candidates)
    logger.info("Creating ADK orchestrator...")
    orchestrator = get_video_generation_orchestrator(
        video_retriever=video_retriever,
        max_video_candidates=config.max_video_candidates,
    )
    
    # Create session and run orchestrator
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    
    APP_NAME = "virtual_streamer"
    user_id = "user"
    session_id = "video_gen_session"
    
    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
    )
    
    # Create session with initial state
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={TITLE: title},
    )
    
    # Create message content
    content = types.Content(role="user", parts=[types.Part(text=title)])
    
    # Run the orchestrator via runner (state_delta is applied automatically)
    logger.info(f"Running orchestrator for title: {title}")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                logger.debug(f"Final response from {event.author}")
        
        # Log progress events
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    logger.info(f"[{event.author}] {part.text}")
    
    # Get updated session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    
    # Get video matches from state
    video_matches_data = session.state.get(VIDEO_MATCHES)
    if not video_matches_data:
        raise RuntimeError("Orchestrator completed but no video matches found in state")
    
    # Parse matches - state_delta stores values as JSON strings
    if isinstance(video_matches_data, SentenceVideoMatcherOutput):
        matches = video_matches_data.matches
    elif isinstance(video_matches_data, str):
        # JSON string from state_delta
        output = SentenceVideoMatcherOutput.model_validate_json(video_matches_data)
        matches = output.matches
    elif isinstance(video_matches_data, dict):
        output = SentenceVideoMatcherOutput.model_validate(video_matches_data)
        matches = output.matches
    else:
        raise RuntimeError(f"Unexpected video_matches type: {type(video_matches_data)}")
    
    logger.info(f"Orchestrator produced {len(matches)} video matches")
    
    # Compose final video using webservice APIs
    async with WebserviceClient(api_config) as client:
        final_video_path = await compose_video_from_matches(
            matches=matches,
            client=client,
            output_dir=config.output_dir,
            temp_dir=config.temp_dir,
            fontsize=config.video_processing.fontsize,
        )
    
    return final_video_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate video from title using ADK agents and webservice API"
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Title/topic for story generation",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="Base URL for webservice API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--character-id",
        type=str,
        default=os.environ.get("CHARACTER_ID", "fred"),
        help="Character ID for TTS and Wav2Lip (default: fred)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for final video",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="./temp",
        help="Temporary directory for intermediate files",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum video candidates per dialog line",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=14,
        help="Subtitle font size",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output the final video path",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build configuration
    config = VideoGenerationConfig()
    config.output_dir = args.output_dir
    config.temp_dir = args.temp_dir
    config.max_video_candidates = args.max_candidates
    config.video_processing.fontsize = args.fontsize
    
    api_config = APIConfig(
        base_url=args.api_url,
        character_id=args.character_id,
    )
    
    # Print configuration summary
    if not args.quiet:
        print("=" * 70)
        print("Virtual Streamer Video Generation (ADK + Webservice)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Title: {args.title}")
        print(f"  API URL: {api_config.base_url}")
        print(f"  Character: {api_config.character_id}")
        print(f"  Video Retrieval: {config.video_retrieval.method}")
        print(f"  Max Candidates: {args.max_candidates}")
        print(f"  Output: {config.output_dir}")
        print("=" * 70)
        print()
    
    try:
        # Run the video generation
        final_video_path = await run_video_generation(
            title=args.title,
            config=config,
            api_config=api_config,
        )
        
        # Print results
        if not args.quiet:
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
    
    except httpx.HTTPStatusError as e:
        print(f"\n\nAPI Error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
