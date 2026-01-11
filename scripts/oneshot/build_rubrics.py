#!/usr/bin/env python3
"""
Build Rubrics from Stories (Map-Reduce)

This script processes a JSON file of stories and extracts rubrics using
the map-reduce rubric builder agent. Stories are processed in batches of 5,
and rubrics are output to a JSONL file.

USAGE:
======
# Basic usage with JSON input file
python scripts/oneshot/build_rubrics.py --input stories.json --output rubrics.jsonl

# With custom batch size
python scripts/oneshot/build_rubrics.py --input stories.json --output rubrics.jsonl --batch-size 10

# Verbose mode
python scripts/oneshot/build_rubrics.py --input stories.json --output rubrics.jsonl -v

INPUT FORMAT:
=============
JSON file with a list of story objects. Each story should have:
- title: str
- subtitle: str
- body: list[str]

Example:
[
  {
    "title": "Article title",
    "subtitle": "Article subtitle",
    "body": ["Paragraph 1", "Paragraph 2", ...]
  },
  ...
]

OUTPUT FORMAT:
==============
JSONL file with one rubric per line:
{"description": "...", "examples": [...]}
{"description": "...", "examples": [...]}
"""

import asyncio
import sys
import os
import json
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from pydantic import ValidationError

from virtual_streamer.agents.rubric_builder_map_reduce import (
    create_rubric_builder_map_reduce,
    STORIES_KEY,
)
from virtual_streamer.agents.rubric_builder_agent.schema import StoryItem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_stories(input_path: Path) -> list:
    """
    Load and validate stories from JSON file.
    
    Each story is validated against the StoryItem schema to catch
    errors early before passing to the agent.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        List of validated story dicts
        
    Raises:
        ValueError: If input format is invalid or stories fail validation
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both list format and dict with "stories" key
    n_stories = None
    if isinstance(data, list):
        raw_stories = data
        n_stories = len(raw_stories)
    elif isinstance(data, dict) and "stories" in data:
        raw_stories = data["stories"]
        n_stories = len(raw_stories)
    else:
        raise ValueError(
            f"Invalid input format. Expected list or dict with 'stories' key, "
            f"got {type(data).__name__}"
        )
    
    if not raw_stories:
        raise ValueError("No stories found in input file")
    
    # Validate each story against StoryItem schema
    validated_stories = []
    errors = []
    
    for i, story_dict in enumerate(raw_stories):
        try:
            story = StoryItem.model_validate(story_dict)
            validated_stories.append(story.model_dump())
        except ValidationError as e:
            errors.append(f"Story {i}: {e}")
    
    if errors:
        error_summary = "\n".join(errors[:5])  # Show first 5 errors
        if len(errors) > int(0.01 * n_stories):
            error_summary += f"\n... and {len(errors) - 5} more errors"
        if len(errors) > int(0.5 * n_stories):
            raise ValueError(
                f"Schema validation failed for {len(errors)}/{len(raw_stories)} stories:\n{error_summary}"
            )
    
    logger.info(f"Validated {len(validated_stories)} stories against StoryItem schema")
    return validated_stories


def create_rubric_agent(output_path: Path, batch_size: int):
    """
    Create the map-reduce rubric builder agent.
    
    Args:
        output_path: Path to output JSONL file
        batch_size: Number of stories per batch
        
    Returns:
        Configured MapReduceAgent
    """
    return create_rubric_builder_map_reduce(
        output_path=output_path,
        batch_size=batch_size,
    )


async def run_rubric_extraction(
    stories: list,
    output_path: Path,
    batch_size: int,
) -> int:
    """
    Run the rubric extraction pipeline.
    
    Args:
        stories: List of story dicts
        output_path: Path to output JSONL file
        batch_size: Number of stories per batch
        
    Returns:
        Number of rubrics extracted
    """
    # Create the agent
    agent = create_rubric_agent(output_path, batch_size)
    
    # Setup session service and runner
    APP_NAME = "rubric_builder"
    user_id = "user"
    session_id = "rubric_session"
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    
    # Create session with stories in state
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={STORIES_KEY: stories},
    )
    
    # Create message content (trigger for the agent)
    content = types.Content(
        role="user",
        parts=[types.Part(text="Extract rubrics from the provided stories")]
    )
    
    # Run the agent
    logger.info(f"Processing {len(stories)} stories in batches of {batch_size}...")
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        # Log progress events
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    logger.info(f"[{event.author}] {part.text}")
    
    # Count output rubrics
    rubric_count = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            rubric_count = sum(1 for _ in f)
    
    return rubric_count


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract rubrics from stories using map-reduce agent"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSON file with stories",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSONL file for rubrics",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Number of stories per batch (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output the final summary",
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.quiet:
        print("=" * 70)
        print("Rubric Builder (Map-Reduce)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Batch Size: {args.batch_size}")
        print("=" * 70)
        print()
    
    try:
        # Load stories
        stories = load_stories(args.input)
        if not args.quiet:
            print(f"Loaded {len(stories)} stories from {args.input}")
        
        # Run extraction
        rubric_count = await run_rubric_extraction(
            stories=stories,
            output_path=args.output,
            batch_size=args.batch_size,
        )
        
        # Print results
        if not args.quiet:
            print("\n" + "=" * 70)
            print("Rubric Extraction Complete!")
            print("=" * 70)
            print(f"\n  Stories processed: {len(stories)}")
            print(f"  Rubrics extracted: {rubric_count}")
            print(f"  Output file: {args.output}")
            print("\n" + "=" * 70)
        else:
            print(f"{rubric_count} rubrics written to {args.output}")
        
        return 0
        
    except FileNotFoundError:
        print(f"\nError: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    except json.JSONDecodeError as e:
        print(f"\nError: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
