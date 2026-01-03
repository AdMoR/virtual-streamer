"""
Video Search module for querying the remote Video Embedding Server.

This module provides a client for searching video segments by text similarity
using VideoPrism embeddings with optional tag filtering.

Example usage:
    from virtual_streamer.video_search import VideoSearchClient

    client = VideoSearchClient()
    results = client.search(
        query="person dancing",
        collection="my_videos",
        top_k=10,
    )
    for result in results:
        print(f"{result.video_id}: {result.similarity:.4f}")
"""

from virtual_streamer.video_search.client import (
    TagInfo,
    VideoSearchClient,
    VideoSearchResult,
)

__all__ = [
    "VideoSearchClient",
    "VideoSearchResult",
    "TagInfo",
]


