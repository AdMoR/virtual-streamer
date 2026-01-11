#!/usr/bin/env python3
"""
Seed initial streaming data for testing.

Creates a default stream configuration and 24/7 programmation for testing.

Usage:
    python scripts/seed_streaming_data.py
    
    # Clear existing data first
    python scripts/seed_streaming_data.py --clear
"""
import asyncio
import argparse
import os
import sys
from datetime import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def seed_data(clear: bool = False):
    """Seed initial streaming data."""
    # Import here to avoid import errors if deps not installed
    from virtual_streamer.streaming.store import get_streaming_store
    
    store = await get_streaming_store()
    
    if clear:
        print("Clearing existing data...")
        # Delete in correct order due to foreign keys
        streams = await store.list_streams()
        for stream in streams:
            await store.delete_stream(stream.stream_id)
        print("Existing data cleared.")
    
    # Create default stream
    print("Creating default stream...")
    try:
        stream = await store.create_stream({
            "stream_id": "default",
            "name": "Default Stream",
            "description": "Main streaming channel for video playback",
        })
        print(f"  Created stream: {stream.stream_id} - {stream.name}")
    except Exception as e:
        print(f"  Stream may already exist: {e}")
    
    # Create a 24/7 programmation (for testing)
    print("Creating 24/7 test programmation...")
    try:
        prog = await store.create_programmation({
            "programmation_id": "default-prog",
            "stream_id": "default",
            "story_template_id": "cest_pas_sorcier",  # From existing templates
            "name": "All Day Programming",
            "start_time": time(0, 0),
            "end_time": time(23, 59),
            "priority": 0,
        })
        print(f"  Created programmation: {prog.programmation_id} - {prog.name}")
        print(f"    Time slot: {prog.start_time} - {prog.end_time}")
        print(f"    Story template: {prog.story_template_id}")
    except Exception as e:
        print(f"  Programmation may already exist: {e}")
    
    print("\nSeed data created successfully!")
    print("\nTo add videos to the playlist, use the API:")
    print("  POST /api/v1/programmations/default-prog/playlist")
    print("  Body: {\"video_storage_key\": \"path/to/video.mp4\"}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed streaming test data")
    parser.add_argument(
        "--clear", 
        action="store_true", 
        help="Clear existing data before seeding"
    )
    args = parser.parse_args()
    
    asyncio.run(seed_data(clear=args.clear))
