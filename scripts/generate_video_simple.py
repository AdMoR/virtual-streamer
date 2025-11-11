#!/usr/bin/env python3
"""
Virtual Streamer Video Generation Script - Simplified Version

This script uses pydantic-settings for all configuration management.
Configuration is loaded from:
1. Environment variables (VG_ prefix)
2. .env file (secrets like API keys)
3. .env.public file (non-secrets)
4. --config YAML file (if provided)
5. Default values

MINIMAL COMMAND-LINE ARGUMENTS:
  --title TEXT          Title/topic for story generation
  --story-file PATH     Load story from file instead of generating
  --from-config-dump PATH  Recreate from previous config dump
  --config PATH         Path to YAML config file (optional)
  --env-file PATH       Path to additional .env file (optional)
  --quiet              Suppress progress messages
  --help               Show this help

All other configuration via environment variables or config files!

USAGE EXAMPLES:

# Basic - uses environment variables or .env files
python scripts/generate_video_simple.py --title "Fred se lance dans l'IA"

# With custom config file
python scripts/generate_video_simple.py --title "Fred" --config my_config.yaml

# With custom .env file
python scripts/generate_video_simple.py --title "Fred" --env-file production.env

# From story file
python scripts/generate_video_simple.py --story-file story.txt

# Recreate from config dump
python scripts/generate_video_simple.py --from-config-dump output/config_20251111.json

# Quiet mode
python scripts/generate_video_simple.py --title "Fred" --quiet

CONFIGURATION:

Set environment variables with VG_ prefix:
  export VG_LLM__PROVIDER=openai
  export VG_LLM__MODEL=gpt-4o
  export VG_OUTPUT_DIR=/custom/output

Or create .env and .env.public files (see .env.example and .env.public.example)

For all configuration options, see configs/default_config.yaml
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.video_generation_config import VideoGenerationConfig
from scripts.video_generation_interfaces import SimpleProgressCallback
from scripts.video_generation_impl import (
    create_llm, create_tts, create_stt,
    create_video_retriever, create_prompt_provider
)
from scripts.video_generation_core import (
    generate_story, generate_video_from_story, recreate_from_config_dump
)


def parse_minimal_args():
    """Parse minimal command-line arguments."""
    args = {
        'title': None,
        'story_file': None,
        'from_config_dump': None,
        'config': None,
        'env_file': None,
        'quiet': False,
        'help': False
    }
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ('--help', '-h'):
            args['help'] = True
            break
        elif arg == '--title' and i + 1 < len(sys.argv):
            args['title'] = sys.argv[i + 1]
            i += 2
        elif arg == '--story-file' and i + 1 < len(sys.argv):
            args['story_file'] = sys.argv[i + 1]
            i += 2
        elif arg == '--from-config-dump' and i + 1 < len(sys.argv):
            args['from_config_dump'] = sys.argv[i + 1]
            i += 2
        elif arg in ('--config', '-c') and i + 1 < len(sys.argv):
            args['config'] = sys.argv[i + 1]
            i += 2
        elif arg in ('--env-file', '-e') and i + 1 < len(sys.argv):
            args['env_file'] = sys.argv[i + 1]
            i += 2
        elif arg == '--quiet':
            args['quiet'] = True
            i += 1
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            print("Use --help to see available options", file=sys.stderr)
            sys.exit(1)
    
    return args


def print_help():
    """Print help message."""
    print(__doc__)


async def main():
    """Main entry point."""
    # Parse minimal arguments
    args = parse_minimal_args()
    
    if args['help']:
        print_help()
        return 0
    
    # Validate input
    if not any([args['title'], args['story_file'], args['from_config_dump']]):
        print("Error: Must provide --title, --story-file, or --from-config-dump", file=sys.stderr)
        print("Use --help for usage information", file=sys.stderr)
        return 1
    
    # Load configuration from environment, .env files, and optionally config file
    try:
        if args['config']:
            config = VideoGenerationConfig.from_yaml(args['config'])
        else:
            # Load from environment and .env files
            if args['env_file']:
                # Prepend custom env file
                os.environ.setdefault('VG_ENV_FILE', args['env_file'])
            config = VideoGenerationConfig()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Set up progress callback
    progress = None if args['quiet'] else SimpleProgressCallback()
    
    # Print configuration summary
    if not args['quiet']:
        print("=" * 70)
        print("Virtual Streamer Video Generation (Simplified)")
        print("=" * 70)
        print(f"\nConfiguration (from env vars / .env files):")
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
        if args['from_config_dump']:
            if not args['quiet']:
                print(f"Recreating video from config dump: {args['from_config_dump']}\n")
            
            # Only need TTS and STT for recreation
            tts = create_tts(config.tts)
            stt = create_stt(config.stt)
            
            result = await recreate_from_config_dump(
                args['from_config_dump'],
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
            
            # Create semaphore for LLM concurrency control
            llm_semaphore = asyncio.Semaphore(config.max_parallel_llm_calls)
            
            # Generate or load story
            story_output = None
            if args['title']:
                if not args['quiet']:
                    print(f"Generating story from title: {args['title']}\n")
                
                story_output = await generate_story(
                    args['title'],
                    llm,
                    prompt_provider,
                    config,
                    progress,
                    llm_semaphore
                )
                
                if not args['quiet']:
                    print(f"\n{'='*70}")
                    print("Generated Story:")
                    print('='*70)
                    print(f"\n📝 Title: {story_output.title}\n")
                    print(f"🎯 Story Plan:")
                    print('-' * 70)
                    print(story_output.story_plan)
                    print('-' * 70)
                    print(f"\n💬 Dialog:")
                    print('-' * 70)
                    print(story_output.dialog)
                    print('='*70)
                    print()
                
                # Use the dialog for video generation
                story = story_output.dialog
            else:
                if not args['quiet']:
                    print(f"Loading story from file: {args['story_file']}\n")
                
                with open(args['story_file'], 'r') as f:
                    story = f.read()
            
            # Generate video
            if not args['quiet']:
                print("Starting video generation...\n")
            
            result = await generate_video_from_story(
                story,
                llm,
                tts,
                stt,
                video_retriever,
                config,
                progress,
                story_output=story_output
            )
        
        # Print results
        if not args['quiet']:
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
                print(f"    python scripts/generate_video_simple.py --from-config-dump {result.config_dump_path}")
            
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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

