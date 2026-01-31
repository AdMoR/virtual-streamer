#!/usr/bin/env python3
"""
Batch Title-to-Video Generation

Generates titles for a programmation's story template, then submits
video generation jobs for each title.

Usage:
    python scripts/batch_title_to_video.py \
        --programmation-id default-prog \
        --count 50 \
        --api-url http://localhost:8000/api/v1

This script uses the admin flag (skip_queue_limit=True) to bypass
the normal queue limit. For batch/admin use only.
"""
import argparse
import asyncio
import sys
import os
import time

import requests
import dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv()


def get_programmation_info(api_url: str, programmation_id: str) -> tuple[str, str]:
    """Get story_template_id and stream_id from programmation."""
    resp = requests.get(f"{api_url}/programmations/{programmation_id}")
    resp.raise_for_status()
    data = resp.json()
    return data["story_template_id"], data["stream_id"]


async def generate_titles(story_template_id: str, count: int) -> list[str]:
    """Generate titles using TitleGeneratorAgent."""
    from virtual_streamer.agents.title_generator.runner import run_title_generator

    return await run_title_generator(story_template_id, count)


def submit_video_job(
    api_url: str,
    stream_id: str,
    title: str,
    user: str = "batch_script",
) -> str | None:
    """Submit a single video job via generate-from-broadcast endpoint."""
    try:
        resp = requests.post(
            f"{api_url}/video-generation/generate-from-broadcast",
            json={
                "stream_id": stream_id,
                "title": title,
                "user": user,
                "skip_queue_limit": True,  # ADMIN: bypass queue limit
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]
    except Exception as e:
        print(f"  ERROR submitting job: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Batch title-to-video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 50 titles and submit video jobs
    python scripts/batch_title_to_video.py \\
        --programmation-id default-prog \\
        --count 50

    # Dry run (generate titles only, no video jobs)
    python scripts/batch_title_to_video.py \\
        --programmation-id default-prog \\
        --count 50 \\
        --dry-run

    # With custom delay between submissions
    python scripts/batch_title_to_video.py \\
        --programmation-id default-prog \\
        --count 50 \\
        --delay 2.0
        """,
    )
    parser.add_argument(
        "--programmation-id", required=True, help="Programmation ID to generate videos for"
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Number of titles to generate (default: 50)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="API base URL (default: http://localhost:8000/api/v1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between job submissions (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate titles only, do not submit video jobs",
    )
    args = parser.parse_args()

    print(f"Fetching programmation: {args.programmation_id}")
    try:
        story_template_id, stream_id = get_programmation_info(
            args.api_url, args.programmation_id
        )
    except requests.HTTPError as e:
        print(f"ERROR: Failed to fetch programmation: {e}")
        sys.exit(1)

    print(f"  story_template_id: {story_template_id}")
    print(f"  stream_id: {stream_id}")

    print(f"\nGenerating {args.count} titles...")
    try:
        titles = await generate_titles(story_template_id, args.count)
    except Exception as e:
        print(f"ERROR: Failed to generate titles: {e}")
        sys.exit(1)

    print(f"Generated {len(titles)} titles:")
    for i, title in enumerate(titles[:5]):
        print(f"  {i+1}. {title}")
    if len(titles) > 5:
        print(f"  ... and {len(titles) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Skipping video job submission")
        print("\nAll generated titles:")
        for i, title in enumerate(titles):
            print(f"  {i+1}. {title}")
        return

    print(f"\nSubmitting {len(titles)} video jobs (skip_queue_limit=True)...")
    job_ids = []
    for i, title in enumerate(titles):
        print(f"  [{i+1}/{len(titles)}] {title[:50]}...")
        job_id = submit_video_job(args.api_url, stream_id, title)
        if job_id:
            job_ids.append(job_id)
            print(f"    -> job_id: {job_id}")
        time.sleep(args.delay)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Titles generated: {len(titles)}")
    print(f"Jobs submitted:   {len(job_ids)}")
    print(f"Failed:           {len(titles) - len(job_ids)}")

    if job_ids:
        print(f"\nFirst few job IDs:")
        for job_id in job_ids[:5]:
            print(f"  - {job_id}")
        if len(job_ids) > 5:
            print(f"  ... and {len(job_ids) - 5} more")


if __name__ == "__main__":
    asyncio.run(main())
