#!/usr/bin/env python3
"""
Bootstrap playlist from MinIO videos.

Populates a MediaProgrammation's playlist with existing videos from MinIO
based on bucket path pattern. Skips videos already in the playlist.

Usage:
    # With explicit programmation
    python scripts/api_helpers/bootstrap_playlist.py \
        --programmation-id default-prog \
        --bucket virtual-streamer \
        --minio-prefix "generated_videos/cps"
    
    # Interactive mode (prompts for programmation selection)
    python scripts/api_helpers/bootstrap_playlist.py \
        --minio-prefix "generated_videos/cps"
    
    # Dry run (list videos without adding)
    python scripts/api_helpers/bootstrap_playlist.py \
        --minio-prefix "generated_videos/cps" \
        --dry-run
"""
import asyncio
import argparse
import fnmatch
import os
import sys
from typing import Optional

import dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

dotenv.load_dotenv()


async def select_programmation_interactive(store) -> str:
    """Display programmations and prompt user to select by index."""
    # Get all streams first
    streams = await store.list_streams()
    
    all_progs = []
    for stream in streams:
        progs = await store.list_programmations(stream.stream_id)
        for prog in progs:
            playlist = await store.get_playlist(prog.programmation_id)
            all_progs.append((prog, stream, len(playlist)))
    
    if not all_progs:
        raise ValueError("No programmations found in database")
    
    print("\nAvailable programmations:")
    for i, (prog, stream, count) in enumerate(all_progs):
        print(f"  [{i}] {prog.name} (stream: {stream.name}) - {count} entries")
    
    while True:
        try:
            choice = input("\nSelect programmation by index: ")
            idx = int(choice)
            if 0 <= idx < len(all_progs):
                return all_progs[idx][0].programmation_id
            print(f"Invalid index. Please enter a number between 0 and {len(all_progs) - 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


async def bootstrap_playlist(
    programmation_id: Optional[str],
    bucket: str,
    minio_prefix: str,
    pattern: str = "*.mp4",
    dry_run: bool = False,
):
    """
    Bootstrap a playlist from MinIO videos.
    
    Args:
        programmation_id: Target programmation (interactive if None)
        bucket: MinIO bucket name
        minio_prefix: Path prefix in MinIO (e.g., 'generated_videos/cps')
        pattern: Glob pattern for filtering files (default: '*.mp4')
        dry_run: If True, only list videos without adding
    """
    # Import here to avoid import errors if deps not installed
    from virtual_streamer.streaming.store import get_streaming_store
    from virtual_streamer.utils.minio_client import MinIOClient
    
    # 1. Get streaming store and MinIO client
    print(f"Connecting to MinIO bucket: {bucket}")
    store = await get_streaming_store()
    minio = MinIOClient(endpoint="http://localhost:9000", bucket=bucket)
    
    # 2. List videos from MinIO first
    print(f"Listing objects with prefix: {minio_prefix}")
    all_keys = await minio.list_objects(minio_prefix)
    video_keys = [k for k in all_keys if fnmatch.fnmatch(os.path.basename(k), pattern)]
    print(f"Found {len(video_keys)} videos matching '{minio_prefix}/{pattern}'")
    
    if not video_keys:
        print("No videos found. Exiting.")
        return
    
    # 3. Select programmation (interactive if not provided)
    if not programmation_id:
        programmation_id = await select_programmation_interactive(store)
    
    # 4. Validate programmation exists
    prog = await store.get_programmation(programmation_id)
    if not prog:
        raise ValueError(f"Programmation '{programmation_id}' not found")
    print(f"\nTarget programmation: {prog.name} ({prog.programmation_id})")
    
    # 5. Get existing entries to skip duplicates
    existing = await store.get_playlist(programmation_id)
    existing_keys = {e.video_storage_key for e in existing}
    print(f"Existing playlist entries: {len(existing)}")
    
    # 6. Filter and add new videos
    new_keys = sorted([k for k in video_keys if k not in existing_keys])
    skipped = len(video_keys) - len(new_keys)
    print(f"New videos to add: {len(new_keys)} (skipping {skipped} duplicates)")
    
    if not new_keys:
        print("No new videos to add. Exiting.")
        return
    
    if dry_run:
        print("\n[DRY RUN] Would add the following videos:")
        for k in new_keys:
            print(f"  - {k}")
        return
    
    # 7. Add videos to playlist
    print("\nAdding videos to playlist:")
    for key in new_keys:
        entry = await store.add_to_playlist(programmation_id, key)
        print(f"  Added: {key} (order: {entry.play_order})")
    
    print(f"\nSuccessfully added {len(new_keys)} videos to playlist '{prog.name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap playlist from MinIO videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Add all videos from a collection
    python bootstrap_playlist.py --minio-prefix generated_videos/cps
    
    # Specify bucket and programmation
    python bootstrap_playlist.py \\
        --programmation-id default-prog \\
        --bucket my-bucket \\
        --minio-prefix generated_videos/cps
    
    # Filter by pattern
    python bootstrap_playlist.py \\
        --minio-prefix generated_videos/cps \\
        --pattern "video_*.mp4"
    
    # Dry run to preview
    python bootstrap_playlist.py \\
        --minio-prefix generated_videos/cps \\
        --dry-run
        """
    )
    parser.add_argument(
        "--programmation-id",
        type=str,
        default=None,
        help="Target programmation ID (interactive selection if omitted)"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("MINIO_BUCKET", "virtual-streamer"),
        help="MinIO bucket name (default: MINIO_BUCKET env or 'virtual-streamer')"
    )
    parser.add_argument(
        "--minio-prefix",
        type=str,
        required=True,
        help="MinIO path prefix (e.g., 'generated_videos/cps')"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for filtering files (default: '*.mp4')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List videos without adding them"
    )
    
    args = parser.parse_args()
    
    asyncio.run(bootstrap_playlist(
        programmation_id=args.programmation_id,
        bucket=args.bucket,
        minio_prefix=args.minio_prefix,
        pattern=args.pattern,
        dry_run=args.dry_run,
    ))
