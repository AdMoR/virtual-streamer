#!/usr/bin/env python3
"""
Virtual Streamer Video Generation Script

This script generates videos from stories using AI models with async optimization.

WORKFLOW:
=========
1. Story Generation: Generates a story from a title using LLM
2. Video Generation: Creates a video from the story by:
   - Splitting story into sentences
   - Finding matching video clips (parallel LLM calls for efficiency)
   - Generating audio (serial, using local TTS)
   - Creating subtitles (serial, using local STT)
   - Combining all into final video

OPTIMIZATION:
=============
- LLM calls (video judgement, keyword generation) are made in PARALLEL
  for maximum efficiency since they are I/O-bound API calls
- TTS and STT are run SERIALLY since they use local GPU/CPU resources
- Config dumping enables exact reproduction without recomputing

USAGE EXAMPLES:
===============

# Generate video from a title
python scripts/generate_video.py --title "Fred se lance dans l'IA"

# Generate from an existing story file
python scripts/generate_video.py --story-file story.txt

# Use custom configuration
python scripts/generate_video.py --title "Fred" --config configs/custom.yaml

# Override specific settings
python scripts/generate_video.py --title "Fred" --llm-provider anthropic --llm-model claude-sonnet-4-5-20250929

# Recreate video from previous config dump (skips expensive LLM calls)
python scripts/generate_video.py --from-config-dump output/config_20251111_103000.json

# Use custom prompt file
python scripts/generate_video.py --title "Fred" --prompt-file prompts/my_prompt.txt

CONFIGURATION:
==============
Configuration can be provided through:
1. Command-line arguments (highest priority)
2. YAML config file (--config)
3. Environment variables (prefix: VG_)
4. Default values

Example config.yaml:
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  temperature: 0.7
tts:
  provider: fish
  host: 127.0.0.1
  port: 8003
video_retrieval:
  method: bm25
  index_path: /path/to/clips
output_dir: ./output
```

REPRODUCIBILITY:
================
Every run generates a comprehensive config dump that includes:
- All input parameters
- All configuration settings
- All intermediate selections (video matches, audio files, etc.)
- All model versions and parameters
- Timing information

This dump can be used to recreate the exact same video without
rerunning expensive LLM API calls using:
  python scripts/generate_video.py --from-config-dump <dump_file>

INTERFACES:
===========
All components use abstract interfaces for extensibility:
- LLM: anthropic, openai, litellm
- TTS: fish (Fish-Speech), solero, coqui
- STT: whisper (stable-whisper), faster-whisper
- Video Retrieval: bm25, vector, hybrid
- Prompts: local files, mlflow

This design allows easy swapping of implementations and future API integration.
"""

import argparse
import asyncio
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.video_generation_config import VideoGenerationConfig, LLMConfig, TTSConfig
from scripts.video_generation_interfaces import SimpleProgressCallback
from scripts.video_generation_impl import (
    create_llm, create_tts, create_stt,
    create_video_retriever, create_prompt_provider
)
from scripts.video_generation_core import (
    generate_story, generate_video_from_story, recreate_from_config_dump
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Virtual Streamer Video Generation - Generate videos from stories using AI",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ========================================================================
    # Input options (mutually exclusive)
    # ========================================================================
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--title",
        help="Title/topic for story generation (e.g., 'Fred se lance dans l'IA')"
    )
    input_group.add_argument(
        "--story-file",
        help="Path to existing story file to convert to video"
    )
    input_group.add_argument(
        "--from-config-dump",
        help="Recreate video from config dump (skips LLM calls, saves API costs)"
    )
    
    # ========================================================================
    # Configuration files
    # ========================================================================
    parser.add_argument(
        "--config",
        help="Path to YAML config file (overrides defaults)"
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to custom prompt file or directory"
    )
    
    # ========================================================================
    # Output options
    # ========================================================================
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for generated video (default: ./output)"
    )
    parser.add_argument(
        "--temp-dir",
        default="./temp",
        help="Temporary directory for intermediate files (default: ./temp)"
    )
    parser.add_argument(
        "--no-config-dump",
        action="store_true",
        help="Disable config dump generation"
    )
    
    # ========================================================================
    # LLM configuration
    # ========================================================================
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "litellm"],
        help="LLM provider (default: anthropic)"
    )
    llm_group.add_argument(
        "--llm-model",
        help="LLM model identifier (default: claude-sonnet-4-5-20250929)"
    )
    llm_group.add_argument(
        "--llm-temperature",
        type=float,
        help="LLM temperature for sampling (default: 0.7)"
    )
    llm_group.add_argument(
        "--llm-api-key",
        help="LLM API key (defaults to env var)"
    )
    
    # ========================================================================
    # TTS configuration
    # ========================================================================
    tts_group = parser.add_argument_group("TTS Configuration")
    tts_group.add_argument(
        "--tts-provider",
        choices=["fish", "solero", "coqui"],
        help="TTS provider (default: fish for Fish-Speech)"
    )
    tts_group.add_argument(
        "--tts-host",
        help="TTS service host (default: 127.0.0.1)"
    )
    tts_group.add_argument(
        "--tts-port",
        type=int,
        help="TTS service port (default: 8003 for Fish-Speech)"
    )
    tts_group.add_argument(
        "--tts-reference-audio",
        help="Path to reference audio for voice cloning"
    )
    tts_group.add_argument(
        "--tts-reference-text",
        help="Reference text matching the reference audio"
    )
    
    # ========================================================================
    # Video retrieval configuration
    # ========================================================================
    video_group = parser.add_argument_group("Video Retrieval Configuration")
    video_group.add_argument(
        "--video-method",
        choices=["bm25", "vector", "hybrid"],
        help="Video retrieval method (default: bm25)"
    )
    video_group.add_argument(
        "--video-index-path",
        help="Path to video index/clips info"
    )
    video_group.add_argument(
        "--video-character",
        default="fred",
        help="Filter videos by character name (default: fred)"
    )
    
    # ========================================================================
    # Processing options
    # ========================================================================
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--max-parallel-llm",
        type=int,
        default=5,
        help="Maximum parallel LLM API calls (default: 5)"
    )
    proc_group.add_argument(
        "--max-sentence-length",
        type=int,
        default=35,
        help="Maximum sentence length for splitting (default: 35)"
    )
    proc_group.add_argument(
        "--max-search-attempts",
        type=int,
        default=3,
        help="Max attempts for alternative search keywords (default: 3)"
    )
    
    # ========================================================================
    # Display options
    # ========================================================================
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    return parser.parse_args()


def load_config(args) -> VideoGenerationConfig:
    """
    Load configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Command-line arguments
    2. YAML config file
    3. Environment variables
    4. Default values
    """
    # Start with defaults or YAML file
    if args.config:
        config = VideoGenerationConfig.from_yaml(args.config)
    else:
        config = VideoGenerationConfig()
    
    # Override with command-line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.temp_dir:
        config.temp_dir = args.temp_dir
    if args.no_config_dump:
        config.enable_config_dump = False
    
    # LLM overrides
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.llm_model:
        config.llm.model = args.llm_model
    if args.llm_temperature is not None:
        config.llm.temperature = args.llm_temperature
    if args.llm_api_key:
        config.llm.api_key = args.llm_api_key
    
    # TTS overrides
    if args.tts_provider:
        config.tts.provider = args.tts_provider
    if args.tts_host:
        config.tts.host = args.tts_host
    if args.tts_port:
        config.tts.port = args.tts_port
    if args.tts_reference_audio:
        config.tts.reference_audio = args.tts_reference_audio
    if args.tts_reference_text:
        config.tts.reference_text = args.tts_reference_text
    
    # Video retrieval overrides
    if args.video_method:
        config.video_retrieval.method = args.video_method
    if args.video_index_path:
        config.video_retrieval.index_path = args.video_index_path
    if args.video_character:
        config.video_retrieval.character_filter = args.video_character
    
    # Processing overrides
    if args.max_parallel_llm:
        config.max_parallel_llm_calls = args.max_parallel_llm
    if args.max_sentence_length:
        config.max_sentence_length = args.max_sentence_length
    if args.max_search_attempts:
        config.max_search_attempts = args.max_search_attempts
    
    # Prompt file override
    if args.prompt_file:
        config.prompt.local_file = args.prompt_file
    
    return config


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set up progress callback
    progress = None if args.quiet else SimpleProgressCallback()
    
    # Print configuration summary
    if not args.quiet:
        print("=" * 70)
        print("Virtual Streamer Video Generation")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  LLM: {config.llm.provider}/{config.llm.model}")
        print(f"  TTS: {config.tts.provider} @ {config.tts.host}:{config.tts.port}")
        print(f"  STT: {config.stt.provider}/{config.stt.model}")
        print(f"  Video Retrieval: {config.video_retrieval.method}")
        print(f"  Output: {config.output_dir}")
        print(f"  Parallel LLM calls: {config.max_parallel_llm_calls}")
        print("=" * 70)
        print()
    
    try:
        # Handle recreate from config dump
        if args.from_config_dump:
            if not args.quiet:
                print(f"Recreating video from config dump: {args.from_config_dump}\n")
            
            # Only need TTS and STT for recreation
            tts = create_tts(config.tts)
            stt = create_stt(config.stt)
            
            result = await recreate_from_config_dump(
                args.from_config_dump,
                tts,
                stt,
                config,
                progress
            )
        
        else:
            # Initialize all components
            if progress:
                progress.update("Initializing components...")
            
            llm = create_llm(config.llm)
            tts = create_tts(config.tts)
            stt = create_stt(config.stt)
            video_retriever = create_video_retriever(config.video_retrieval)
            prompt_provider = create_prompt_provider(config.prompt)
            
            # Generate or load story
            if args.title:
                if not args.quiet:
                    print(f"Generating story from title: {args.title}\n")
                
                story = await generate_story(
                    args.title,
                    llm,
                    prompt_provider,
                    config,
                    progress
                )
                
                if not args.quiet:
                    print(f"\n{'='*70}")
                    print("Generated Story:")
                    print('='*70)
                    print(story)
                    print('='*70)
                    print()
            else:
                if not args.quiet:
                    print(f"Loading story from file: {args.story_file}\n")
                
                with open(args.story_file, 'r') as f:
                    story = f.read()
            
            # Generate video
            if not args.quiet:
                print("Starting video generation...\n")
            
            result = await generate_video_from_story(
                story,
                llm,
                tts,
                stt,
                video_retriever,
                config,
                progress
            )
        
        # Print results
        if not args.quiet:
            print("\n" + "=" * 70)
            print("Video Generation Complete!")
            print("=" * 70)
            print(f"\n✓ Video saved to: {result.video_path}")
            print(f"  Duration: {result.metadata.get('total_duration', 'N/A'):.2f}s")
            print(f"  Sentences: {result.metadata.get('sentence_count', 'N/A')}")
            
            if 'timing' in result.metadata:
                timing = result.metadata['timing']
                print(f"\n  Timing:")
                print(f"    Video search: {timing.get('video_search', 0):.2f}s")
                print(f"    Audio generation: {timing.get('audio_generation', 0):.2f}s")
                print(f"    Subtitle generation: {timing.get('subtitle_generation', 0):.2f}s")
                print(f"    Total: {timing.get('total', 0):.2f}s")
            
            if result.config_dump_path:
                print(f"\n✓ Config dump saved to: {result.config_dump_path}")
                print(f"\n  To recreate this exact video:")
                print(f"    python scripts/generate_video.py --from-config-dump {result.config_dump_path}")
            
            print("\n" + "=" * 70)
        else:
            # Quiet mode: just print the video path
            print(result.video_path)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

