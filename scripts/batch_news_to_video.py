#!/usr/bin/env python3
"""
Batch News-to-Video Generation

Fetches unused news articles and submits video generation jobs for each.
Uses the article title as the video topic and enriches story generation
with news context.

Usage:
    python scripts/batch_news_to_video.py \
        --programmation-id default-prog \
        --count 20 \
        --api-url http://localhost:8000/api/v1

This script uses the admin flag (skip_queue_limit=True) to bypass
the normal queue limit. For batch/admin use only.
"""
import argparse
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


def fetch_unused_articles(api_url: str, limit: int) -> list[dict]:
    """Fetch unused news articles from the API.

    Returns:
        List of article metadata dicts with keys: id, title, source, published_at, etc.
    """
    resp = requests.get(f"{api_url}/articles/unused", params={"limit": limit})
    resp.raise_for_status()
    return resp.json()


def get_article_context(api_url: str, article_id: str) -> dict:
    """Get full news context for an article.

    Returns:
        NewsContext dict with keys: article_id, title, summary, source,
        published_at, prompt_context
    """
    resp = requests.get(f"{api_url}/articles/{article_id}/context")
    resp.raise_for_status()
    return resp.json()


def submit_video_job_with_news(
    api_url: str,
    stream_id: str,
    article: dict,
    context: dict,
    user: str = "news_batch",
) -> str | None:
    """Submit a video job with news context.

    Args:
        api_url: API base URL
        stream_id: Stream ID from programmation
        article: Article metadata dict
        context: NewsContext dict with prompt_context
        user: User identifier for the job

    Returns:
        job_id if successful, None on error
    """
    try:
        resp = requests.post(
            f"{api_url}/video-generation/generate-from-broadcast",
            json={
                "stream_id": stream_id,
                "title": article["title"],
                "user": user,
                "skip_queue_limit": True,  # ADMIN: bypass queue limit
                "news_article_id": article["id"],
                "news_context": context["prompt_context"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]
    except Exception as e:
        print(f"  ERROR submitting job: {e}")
        return None


def mark_article_used(api_url: str, article_id: str, job_id: str) -> bool:
    """Mark an article as used in a story.

    Returns:
        True if successful, False on error
    """
    try:
        resp = requests.post(
            f"{api_url}/articles/{article_id}/mark-used",
            params={"story_id": job_id},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  ERROR marking article used: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch news-to-video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate videos from 20 news articles
    python scripts/batch_news_to_video.py \\
        --programmation-id default-prog \\
        --count 20

    # Dry run (preview articles, no video jobs)
    python scripts/batch_news_to_video.py \\
        --programmation-id default-prog \\
        --count 5 \\
        --dry-run

    # With custom delay between submissions
    python scripts/batch_news_to_video.py \\
        --programmation-id default-prog \\
        --count 20 \\
        --delay 30.0
        """,
    )
    parser.add_argument(
        "--programmation-id",
        required=True,
        help="Programmation ID to generate videos for"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of articles to process (default: 10)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="API base URL (default: http://localhost:8000/api/v1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=60.0,
        help="Delay in seconds between job submissions (default: 60.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview articles only, do not submit video jobs",
    )
    parser.add_argument(
        "--user",
        default="news_batch",
        help="User identifier for jobs (default: news_batch)",
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

    print(f"\nFetching {args.count} unused articles...")
    try:
        articles = fetch_unused_articles(args.api_url, args.count)
    except Exception as e:
        print(f"ERROR: Failed to fetch articles: {e}")
        sys.exit(1)

    if not articles:
        print("WARNING: No unused articles available")
        print("  Run 'curl -X POST {}/articles/fetch' to fetch fresh news".format(
            args.api_url
        ))
        sys.exit(0)

    print(f"Found {len(articles)} unused articles:")
    for i, article in enumerate(articles[:5]):
        source = article.get("source", "unknown")
        print(f"  {i+1}. [{source}] {article['title'][:60]}...")
    if len(articles) > 5:
        print(f"  ... and {len(articles) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Skipping video job submission")
        print("\nAll articles:")
        for i, article in enumerate(articles):
            source = article.get("source", "unknown")
            print(f"  {i+1}. [{source}] {article['title']}")
        return

    print(f"\nSubmitting {len(articles)} video jobs (skip_queue_limit=True)...")
    job_ids = []
    marked_count = 0

    for i, article in enumerate(articles):
        article_id = article["id"]
        title = article["title"]
        print(f"  [{i+1}/{len(articles)}] {title[:60]}...")

        # Get full context
        try:
            context = get_article_context(args.api_url, article_id)
        except Exception as e:
            print(f"    ERROR fetching context: {e}")
            continue

        # Submit job
        job_id = submit_video_job_with_news(
            args.api_url, stream_id, article, context, args.user
        )
        if job_id:
            job_ids.append(job_id)
            print(f"    -> job_id: {job_id}")

            # Mark article as used
            if mark_article_used(args.api_url, article_id, job_id):
                marked_count += 1
                print(f"    -> marked as used")

        # Wait before next submission (except on last item)
        if i < len(articles) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Articles fetched:    {len(articles)}")
    print(f"Jobs submitted:      {len(job_ids)}")
    print(f"Articles marked used: {marked_count}")
    print(f"Failed:              {len(articles) - len(job_ids)}")

    if job_ids:
        print(f"\nFirst few job IDs:")
        for job_id in job_ids[:5]:
            print(f"  - {job_id}")
        if len(job_ids) > 5:
            print(f"  ... and {len(job_ids) - 5} more")


if __name__ == "__main__":
    main()
