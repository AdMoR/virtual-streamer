#!/usr/bin/env python3
"""
Batch scene detection and video splitting using PySceneDetect.

This script processes video files in parallel, splitting them into scenes
and skipping videos that have already been processed.

Usage:
    python scripts/batch_scene_split.py /path/to/videos /path/to/output
    python scripts/batch_scene_split.py /path/to/videos /path/to/output --workers 8
    python scripts/batch_scene_split.py /path/to/videos /path/to/output --min-scene-len 3.0
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_processed_video_names(output_path: Path) -> set[str]:
    """
    Scan output directory once and return set of video names that have been processed.
    
    Parses filenames like 'video001-Scene-001.mp4' to extract 'video001'.
    """
    processed = set()
    scene_pattern = re.compile(r"^(.+)-Scene-\d+\.mp4$")
    
    for f in output_path.iterdir():
        if f.is_file():
            match = scene_pattern.match(f.name)
            if match:
                processed.add(match.group(1))
    
    return processed


def process_video(
    video_file: Path,
    output_path: Path,
    min_scene_len: float,
) -> tuple[Path, bool, str]:
    """Process a single video file with scenedetect."""

    try:
        result = subprocess.run(
            [
                "scenedetect",
                "-i", str(video_file),
                "--min-scene-len", str(min_scene_len),
                "--merge-last-scene",
                "split-video",
                "-o", str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return (video_file, True, "success")
        else:
            error_msg = result.stderr[:200] if result.stderr else "unknown error"
            return (video_file, False, f"error: {error_msg}")
    except FileNotFoundError:
        return (video_file, False, "error: scenedetect not found. Install with: pip install scenedetect[opencv]")
    except Exception as e:
        return (video_file, False, f"exception: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch scene detection and video splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s /media/videos /media/output
    %(prog)s /media/videos /media/output --workers 8
    %(prog)s /media/videos /media/output --min-scene-len 3.0 --pattern "*.mkv"
        """,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing video files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for split scene files",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--min-scene-len", "-m",
        type=float,
        default=2.5,
        help="Minimum scene length in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="**/*.mp4",
        help="Glob pattern for video files (default: **/*.mp4)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be processed without actually processing",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find video files
    files = list(args.input_dir.glob(args.pattern))
    print(f"Found {len(files)} video files matching '{args.pattern}'")

    if not files:
        print("No files to process.")
        sys.exit(0)

    # Scan output directory once to get already processed video names
    processed_names = get_processed_video_names(args.output_dir)
    print(f"Found {len(processed_names)} already processed videos in output directory")

    # Filter out already processed files using set lookup (O(1) per file)
    to_process = [f for f in files if f.stem not in processed_names]
    skipped = len(files) - len(to_process)
    print(f"Skipping {skipped} already processed videos, {len(to_process)} to process")

    if not to_process:
        print("All videos already processed.")
        sys.exit(0)

    if args.dry_run:
        print("\nDry run - would process:")
        for f in to_process:
            print(f"  {f}")
        sys.exit(0)

    # Process videos in parallel
    print(f"\nProcessing with {args.workers} workers...")
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, f, args.output_dir, args.min_scene_len): f
            for f in to_process
        }

        for i, future in enumerate(as_completed(futures), 1):
            video_file, success, message = future.result()
            status = "✓" if success else "✗"
            print(f"[{i}/{len(to_process)}] {status} {video_file.name}: {message}")

            if success and "skipped" not in message:
                success_count += 1
            elif not success:
                error_count += 1

    print(f"\nDone! Processed: {success_count}, Errors: {error_count}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
