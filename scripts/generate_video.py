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

# Use custom configuration file
python scripts/generate_video.py --title "Fred" --config configs/custom.yaml

# Override specific settings via CLI
python scripts/generate_video.py --title "Fred" --llm.provider anthropic --llm.model claude-sonnet-4-5-20250929

# Recreate video from previous config dump (skips expensive LLM calls)
python scripts/generate_video.py --from-config-dump output/config_20251111_103000.json

# Use custom prompt file
python scripts/generate_video.py --title "Fred" --prompt-file prompts/my_prompt.txt

CONFIGURATION:
==============
Configuration is loaded from (in order of precedence):
1. Command-line arguments (--title, --llm.provider, etc.)
2. Environment variables (VG_TITLE, VG_LLM__PROVIDER, etc.)
3. .env file (secrets like API keys)
4. .env.public file (non-secret settings)
5. Default values

Example .env:
```
ANTHROPIC_API_KEY=sk-ant-...
VG_LLM__PROVIDER=anthropic
VG_OUTPUT_DIR=./output
```

Example CLI with nested config:
```bash
python scripts/generate_video.py \
  --title "Fred découvre l'IA" \
  --llm.provider anthropic \
  --llm.temperature 0.8 \
  --tts.host localhost \
  --tts.port 8003 \
  --output-dir ./my_videos
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


async def main():
    """Main entry point."""
    # Load configuration from CLI args, env vars, and .env files
    # Pydantic will automatically parse CLI arguments!
    config = VideoGenerationConfig()
    
    # Validate inputs
    try:
        config.validate_inputs()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nUsage: python scripts/generate_video.py --title 'Your Title'")
        print("   or: python scripts/generate_video.py --story-file story.txt")
        print("   or: python scripts/generate_video.py --from-config-dump config.json")
        return 1
    
    # Override prompt file if specified
    if config.prompt_file:
        config.prompt.local_file = config.prompt_file
    
    # Set up progress callback
    progress = None if config.quiet else SimpleProgressCallback()
    
    # Print configuration summary
    if not config.quiet:
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
        if config.from_config_dump:
            if not config.quiet:
                print(f"Recreating video from config dump: {config.from_config_dump}\n")
            
            # Only need TTS and STT for recreation
            tts = create_tts(config.tts)
            stt = create_stt(config.stt)
            
            result = await recreate_from_config_dump(
                config.from_config_dump,
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
            story_output = None
            if config.title:
                if not config.quiet:
                    print(f"Generating story from title: {config.title}\n")
                
                story_output = await generate_story(
                    config.title,
                    llm,
                    prompt_provider,
                    config,
                    progress
                )
                
                if not config.quiet:
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
                if not config.quiet:
                    print(f"Loading story from file: {config.story_file}\n")
                
                with open(config.story_file, 'r') as f:
                    story = f.read()
            
            # Generate video
            if not config.quiet:
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
        if not config.quiet:
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
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
