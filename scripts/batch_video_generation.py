#!/usr/bin/env python3
"""
Batch Video Generation CLI

Reads titles from a text file and sequentially submits video generation jobs
using a specified story template, waiting for each to complete before starting the next.

Usage:
    python scripts/batch_video_generation.py \
        --titles-file titles.txt \
        --story-template-id "cest-pas-sorcier" \
        --api-url http://localhost:8000/api/v1 \
        --poll-interval 10
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch video generation from a list of titles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/batch_video_generation.py \\
        --titles-file titles.txt \\
        --story-template-id "cest-pas-sorcier"
        """,
    )
    parser.add_argument(
        "--titles-file",
        type=Path,
        required=True,
        help="Path to text file with one title per line",
    )
    parser.add_argument(
        "--story-template-id",
        type=str,
        required=True,
        help="Story template ID to use for all titles",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/api/v1",
        help="API base URL (default: http://localhost:8000/api/v1)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between status checks (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--stories-output",
        type=Path,
        default=None,
        help="Path to JSON file where all generated stories will be saved",
    )
    return parser.parse_args()


def read_titles(titles_file: Path) -> list[str]:
    """Read titles from file, one per line, skipping empty lines."""
    if not titles_file.exists():
        raise FileNotFoundError(f"Titles file not found: {titles_file}")

    titles = []
    with open(titles_file, "r", encoding="utf-8") as f:
        for line in f:
            title = line.strip()
            if title:  # Skip empty lines
                titles.append(title)

    return titles


def submit_job(
    api_url: str,
    title: str,
    story_template_id: str,
    output_dir: Optional[str] = None,
    verbose: bool = False,
) -> Optional[str]:
    """Submit a video generation job and return the job ID."""
    request_data = {
        "title": title,
        "story_template_id": story_template_id,
        "verbose": verbose,
    }
    if output_dir:
        request_data["output_dir"] = output_dir

    try:
        response = requests.post(
            f"{api_url}/video-generation/submit",
            json=request_data,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result["job_id"]
    except requests.RequestException as e:
        print(f"  ERROR: Failed to submit job: {e}", file=sys.stderr)
        return None


def get_job_status(api_url: str, job_id: str) -> Optional[dict]:
    """Get the status of a job."""
    try:
        response = requests.get(
            f"{api_url}/video-generation/jobs/{job_id}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  ERROR: Failed to get job status: {e}", file=sys.stderr)
        return None


def wait_for_job(
    api_url: str, job_id: str, poll_interval: int
) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    Poll job status until completed or failed.

    Returns a tuple of (success, result_data).
    """
    while True:
        status = get_job_status(api_url, job_id)
        if status is None:
            return False, None

        job_status = status.get("status", "unknown")
        progress = status.get("progress", "")

        if job_status == "completed":
            result = status.get("result", {})
            metadata = result.get("metadata", {})
            duration = metadata.get("total_duration")
            video_url = metadata.get("video_url", "")

            print(f"  ✓ Completed successfully!")
            if duration:
                print(f"    Duration: {duration:.1f}s")
            if video_url:
                print(f"    Video URL: {video_url[:80]}...")
            return True, result

        elif job_status == "failed":
            error = status.get("error", "Unknown error")
            print(f"  ✗ Failed: {error}", file=sys.stderr)
            return False, None

        else:
            # Still pending or running
            status_msg = f"  Status: {job_status}"
            if progress:
                status_msg += f" - {progress}"
            print(status_msg)
            time.sleep(poll_interval)


def main():
    """Main entry point."""
    args = parse_args()

    # Read titles
    print(f"Reading titles from: {args.titles_file}")
    try:
        titles = read_titles(args.titles_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not titles:
        print("ERROR: No titles found in file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(titles)} title(s) to process")
    print(f"Story template: {args.story_template_id}")
    print(f"API URL: {args.api_url}")
    if args.stories_output:
        print(f"Stories output: {args.stories_output}")
    print("-" * 60)

    # Process each title
    success_count = 0
    failure_count = 0
    collected_stories: list[dict[str, Any]] = []

    for i, title in enumerate(titles, 1):
        print(f"\n[{i}/{len(titles)}] Processing: {title}")

        # Submit job
        job_id = submit_job(
            api_url=args.api_url,
            title=title,
            story_template_id=args.story_template_id,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

        if job_id is None:
            failure_count += 1
            print(f"  ✗ Failed to submit job")
            continue

        print(f"  Job ID: {job_id}")

        # Wait for completion
        success, result = wait_for_job(args.api_url, job_id, args.poll_interval)
        if success:
            success_count += 1
            # Collect story output and video matches if stories-output is enabled
            if args.stories_output and result:
                story_output = result.get("story_output")
                video_matches = result.get("video_matches")
                if story_output:
                    collected_stories.append({
                        "original_title": title,
                        "job_id": job_id,
                        "story_output": story_output,
                        "video_matches": video_matches,
                    })
        else:
            failure_count += 1

    # Write collected stories to JSON file
    if args.stories_output and collected_stories:
        stories_data = {
            "generated_at": datetime.now().isoformat(),
            "story_template_id": args.story_template_id,
            "stories": collected_stories,
        }
        # Ensure parent directory exists
        args.stories_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stories_output, "w", encoding="utf-8") as f:
            json.dump(stories_data, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Stories saved to: {args.stories_output}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:     {len(titles)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed:    {failure_count}")
    if args.stories_output:
        print(f"Stories saved: {len(collected_stories)}")

    # Exit code
    if failure_count > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

